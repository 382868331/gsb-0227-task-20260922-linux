"""固定 NTT 素域上的多点求值与插值（乘积树 / 余式树，复用 NTT 乘法）。

设计要点：

* 只在本库已有的三个 NTT 素数（exact_conv.ntt.MODULI）上工作，
  多项式乘法直接复用 exact_conv.ntt.convolution_mod，
  不引入任何外部代数库，也不把整数 CRT 重建结果当作域元素——
  所有输入输出系数都规范到 [0, p-1]。
* 多点求值：先对求值点构建*乘积树*（自底向上成对相乘，奇数个节点
  用常数多项式 1 补齐以保持配对），再自顶向下做*余式树*下降，
  叶子即各点函数值。零多项式、单点、点 0、次数高于点数等情形
  都走同一条路径。
* 插值：由乘积树根得到 M(x) = 连乘 (x - x_i)，对 M 求形式导数 M'，
  复用余式树在各点求值得到 M'(x_i)，再按 Lagrange 公式
  f = 求和 y_i / M'(x_i) * M/(x - x_i) 累加。点必须互异，否则拒绝。
* 规模限制：点数 <= 128，多项式次数 <= 128（系数至多 129 项）。
* 统计*实际多项式乘法次数*（乘积树合并中对 convolution_mod 的
  调用次数），经结果对象的 multiplication_count 暴露。余式树的
  多项式取余与插值的综合除法用朴素长除法，不调用多项式乘法，
  因此不计入。
"""

from .errors import (
    DuplicatePointError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidModulusError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
)
from .ntt import MODULI, convolution_mod

#: 多点求值 / 插值的点数上限（接口边界）。
MAX_POINTS = 128

#: 多点求值 / 插值的多项式次数上限（接口边界，系数至多 129 项）。
MAX_DEGREE = 128

__all__ = [
    "MAX_POINTS",
    "MAX_DEGREE",
    "EvaluationResult",
    "InterpolationResult",
    "multipoint_evaluate",
    "interpolate",
]


class EvaluationResult:
    """多点求值结果（不可变）。

    Attributes:
        values: 各点函数值，顺序与输入点一致，均在 [0, p-1]。
        modulus: 使用的素数域 p。
        multiplication_count: 本次求值实际执行的多项式乘法次数。
    """

    __slots__ = ("values", "modulus", "multiplication_count")

    def __init__(self, values, modulus, multiplication_count):
        self.values = values
        self.modulus = modulus
        self.multiplication_count = multiplication_count

    def __repr__(self):
        return (f"EvaluationResult(points={len(self.values)}, "
                f"modulus={self.modulus}, "
                f"multiplications={self.multiplication_count})")


class InterpolationResult:
    """插值结果（不可变）。

    Attributes:
        coefficients: 次数小于点数的唯一多项式系数（升幂），
            规范到 [0, p-1]，已去掉高位零（零多项式为 [0]）。
        modulus: 使用的素数域 p。
        multiplication_count: 本次插值实际执行的多项式乘法次数。
    """

    __slots__ = ("coefficients", "modulus", "multiplication_count")

    def __init__(self, coefficients, modulus, multiplication_count):
        self.coefficients = coefficients
        self.modulus = modulus
        self.multiplication_count = multiplication_count

    def __repr__(self):
        return (f"InterpolationResult(degree={len(self.coefficients) - 1}, "
                f"modulus={self.modulus}, "
                f"multiplications={self.multiplication_count})")


class _MultiplicationCounter:
    """统计实际多项式乘法次数的可变计数器。"""

    __slots__ = ("count",)

    def __init__(self):
        self.count = 0


def _validate_modulus(p):
    """模数必须是固定 NTT 素数之一（拒绝 bool 与非 int）。"""
    if isinstance(p, bool) or not isinstance(p, int):
        raise InvalidModulusError(
            f"modulus must be an int, got {type(p).__name__} ({p!r})"
        )
    if p not in MODULI:
        raise InvalidModulusError(
            f"modulus {p} is not one of the fixed NTT primes "
            f"{list(MODULI)}; only these are supported so the existing "
            f"NTT can be reused"
        )
    return p


def _validate_field_elements(values, name, allow_empty):
    """校验系数/点/值序列：精确 int（拒绝 bool），返回 int 列表。"""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise InvalidPolynomialError(
            f"{name!r} must be an iterable of integers, "
            f"got {type(values).__name__}"
        )
    items = list(values)
    if not items and not allow_empty:
        raise LengthOutOfBoundError(f"{name!r} must be non-empty")
    for index, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidCoefficientError(
                f"{name}[{index}] must be an exact int, "
                f"got {type(value).__name__} ({value!r})"
            )
    return items


def _validate_points(points, p):
    """点序列：1..MAX_POINTS 个整数，规范到 [0, p-1]。"""
    items = _validate_field_elements(points, "points", allow_empty=False)
    if len(items) > MAX_POINTS:
        raise LengthOutOfBoundError(
            f"'points' length {len(items)} exceeds MAX_POINTS={MAX_POINTS}"
        )
    return [x % p for x in items]


def _trim(poly):
    """去掉高位零（零多项式的内部约定表示为 []）。"""
    end = len(poly)
    while end > 0 and poly[end - 1] == 0:
        end -= 1
    return poly[:end]


def _poly_mul(first, second, p, counter):
    """用已有 NTT 在 F_p 上相乘，并计入乘法次数。

    任一因子为零返回零；任一因子为单位多项式 1 时直接返回另一因子
    （未发生实际乘法，不计数）。
    """
    if not first or not second:
        return []
    if first == [1]:
        return second
    if second == [1]:
        return first
    counter.count += 1
    size = 1 << (len(first) + len(second) - 2).bit_length()
    return _trim(convolution_mod(first, second, size, p))


def _poly_mod(poly, modulus_poly, p):
    """朴素长除法求 poly mod modulus_poly（首一），不占用乘法计数。"""
    remainder = list(poly)
    top = len(modulus_poly) - 1
    if top < 0:
        raise ExactConvolutionError("modulus polynomial must be non-zero")
    while len(remainder) > top and remainder:
        factor = remainder[-1]  # 首一除式，无需乘逆元
        shift = len(remainder) - 1 - top
        for i in range(top + 1):
            remainder[shift + i] = (remainder[shift + i]
                                    - factor * modulus_poly[i]) % p
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return remainder


def _poly_derivative(poly, p):
    """形式导数。"""
    return _trim([(i * poly[i]) % p for i in range(1, len(poly))])


def _build_product_tree(points, p, counter):
    """自底向上构建乘积树，返回按层组织的列表；顶层唯一节点为 M(x)。

    叶子先用常数多项式 [1] 补齐到 2 的幂，保证每个节点恒有两个孩子
    （下一层的 2j 与 2j+1），余式树下降无需特判；与 [1] 的合并
    由 _poly_mul 短路，不产生实际乘法。
    """
    leaf_count = 1 << (len(points) - 1).bit_length()
    level = [[(-x) % p, 1] for x in points]  # 叶子：(x - x_i)
    level += [[1]] * (leaf_count - len(level))
    tree = []
    while len(level) > 1:
        tree.append(level)
        upper = []
        for i in range(0, len(level), 2):
            upper.append(_poly_mul(level[i], level[i + 1], p, counter))
        level = upper
    tree.append(level)
    return tree


def _remainder_tree_evaluate(coeffs, points, p, counter):
    """余式树多点求值：返回与 points 同序的函数值列表。"""
    tree = _build_product_tree(points, p, counter)
    remainders = [_poly_mod(coeffs, tree[-1][0], p)]
    for level in range(len(tree) - 1, 0, -1):
        children = tree[level - 1]
        lower = []
        for index, rem in enumerate(remainders):
            lower.append(_poly_mod(rem, children[2 * index], p))
            lower.append(_poly_mod(rem, children[2 * index + 1], p))
        remainders = lower
    # 补齐的叶子对应常数多项式 1，取余恒为 0，只取前 len(points) 个。
    return [rem[0] if rem else 0 for rem in remainders[: len(points)]]


def multipoint_evaluate(coeffs, points, p):
    """在素域 F_p 上对多项式做多点求值（乘积树 + 余式树）。

    参数:
        coeffs: 升幂系数，精确 int（拒绝 bool），任意大小均可，
            会被规范到 [0, p-1]；长度 0..129（空序列即零多项式），
            次数可高于点数。
        points: 1..128 个求值点（整数，取模规范化；允许重复——
            重复点只是重复求值，只有插值才要求互异）。
        p: 固定 NTT 素数之一（exact_conv.MODULI）。

    返回:
        EvaluationResult，values 与 points 同序，均在 [0, p-1]。
    """
    _validate_modulus(p)
    raw = _validate_field_elements(coeffs, "coeffs", allow_empty=True)
    if len(raw) > MAX_DEGREE + 1:
        raise LengthOutOfBoundError(
            f"'coeffs' degree {len(raw) - 1} exceeds MAX_DEGREE={MAX_DEGREE}"
        )
    pts = _validate_points(points, p)

    counter = _MultiplicationCounter()
    poly = _trim([c % p for c in raw])
    if not poly:
        # 零多项式：所有点函数值均为 0，无需建树。
        return EvaluationResult([0] * len(pts), p, counter.count)
    values = _remainder_tree_evaluate(poly, pts, p, counter)
    return EvaluationResult(values, p, counter.count)


def _exact_quotient_by_linear(monic_poly, root, p):
    """首一多项式精确除以 (x - root) 的综合除法（不占用乘法计数）。"""
    n = len(monic_poly) - 1
    quotient = [0] * n
    carry = 0  # 已知商的最高次系数为 1（被除式首一）
    for k in range(n - 1, -1, -1):
        carry = (monic_poly[k + 1] + root * carry) % p
        quotient[k] = carry
    return quotient


def interpolate(points, values, p):
    """在素域 F_p 上插值：返回次数 < 点数的唯一多项式。

    参数:
        points: 1..128 个*互异*插值点（取模后判重；重复即拒绝）。
        values: 与 points 等长的函数值。
        p: 固定 NTT 素数之一。

    返回:
        InterpolationResult，系数升幂、规范到 [0, p-1]、去掉高位零
        （零多项式为 [0]）。

    异常:
        DuplicatePointError: 插值点在模 p 后不互异。
    """
    _validate_modulus(p)
    pts = _validate_points(points, p)
    vals = _validate_field_elements(values, "values", allow_empty=False)
    if len(vals) != len(pts):
        raise ExactConvolutionError(
            f"'values' length {len(vals)} != 'points' length {len(pts)}"
        )
    if len(set(pts)) != len(pts):
        raise DuplicatePointError(
            "interpolation points must be distinct modulo "
            f"{p}; got {len(pts) - len(set(pts))} duplicate(s)"
        )
    vals = [v % p for v in vals]

    counter = _MultiplicationCounter()
    n = len(pts)
    tree = _build_product_tree(pts, p, counter)
    root = tree[-1][0]  # M(x) = 连乘 (x - x_i)，首一，次数 n

    # M'(x_i) = 连乘_{j!=i}(x_i - x_j)，复用余式树在全部点上求值。
    derivative = _poly_derivative(root, p)
    deriv_values = _remainder_tree_evaluate(derivative, pts, p, counter)

    # Lagrange：f = 求和 y_i * inv(M'(x_i)) * M/(x - x_i)。
    total = [0] * n
    for i in range(n):
        scale = vals[i] * pow(deriv_values[i], p - 2, p) % p
        if scale == 0:
            continue
        quotient = _exact_quotient_by_linear(root, pts[i], p)
        for k in range(len(quotient)):
            total[k] = (total[k] + scale * quotient[k]) % p
    coeffs = _trim(total)
    return InterpolationResult(coeffs if coeffs else [0], p, counter.count)
