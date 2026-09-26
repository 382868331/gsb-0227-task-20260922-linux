"""指定 NTT 素域上的多点求值与插值（乘积树 / 余式树）。

复用 :mod:`exact_conv.ntt` 的迭代蝶形 NTT 做域上多项式乘法，构建乘积树
与余式树复用子问题：

* 多点求值：先对插值点构建乘积树，再把被求值多项式沿树自顶向下取余
  （余式树），叶子即各点函数值。
* 插值：对乘积树根 M(x) 求导并在各点求值（复用同一余式树），得到
  拉格朗日权重 c_i = y_i / M'(x_i)，再沿乘积树自底向上合并
  P = P_left * M_right + P_right * M_left。

所有系数与点都规范到 ``[0, p-1]`` 的域元素；本模块只做域 F_p 上的运算，
绝不把整数 CRT 重建结果直接当作域元素使用。多项式乘法的实际调用次数
通过结果对象的 ``multiplications`` 字段暴露。

限制（本迭代契约）：仅支持 :data:`exact_conv.ntt.MODULI` 中的素数；
点数与次数均不超过 128。
"""

from .errors import (
    DuplicatePointError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    UnsupportedModulusError,
)
from .ntt import MODULI, convolution_mod

#: 多点求值 / 插值的最大点数（接口边界）。
MAX_POINTS = 128

#: 被求值 / 插值多项式的最大次数（接口边界）。
MAX_DEGREE = 128

__all__ = [
    "evaluate_multi",
    "interpolate",
    "EvaluationResult",
    "InterpolationResult",
    "MAX_POINTS",
    "MAX_DEGREE",
]


class EvaluationResult:
    """多点求值结果（不可变）。

    Attributes:
        values: 各点函数值，与输入点顺序一致，均在 ``[0, p-1]``。
        modulus: 使用的素数 ``p``。
        multiplications: 实际执行的多项式乘法（NTT 卷积）次数。
    """

    __slots__ = ("values", "modulus", "multiplications")

    def __init__(self, values, modulus, multiplications):
        self.values = values
        self.modulus = modulus
        self.multiplications = multiplications

    def __repr__(self):
        return (f"EvaluationResult(points={len(self.values)}, "
                f"modulus={self.modulus}, "
                f"multiplications={self.multiplications})")


class InterpolationResult:
    """插值结果（不可变）。

    Attributes:
        coefficients: 唯一次数小于点数的多项式系数（低次在前），
            规范到 ``[0, p-1]``，已去掉高位零。
        modulus: 使用的素数 ``p``。
        multiplications: 实际执行的多项式乘法（NTT 卷积）次数。
    """

    __slots__ = ("coefficients", "modulus", "multiplications")

    def __init__(self, coefficients, modulus, multiplications):
        self.coefficients = coefficients
        self.modulus = modulus
        self.multiplications = multiplications

    def __repr__(self):
        return (f"InterpolationResult(degree={len(self.coefficients) - 1}, "
                f"modulus={self.modulus}, "
                f"multiplications={self.multiplications})")


class _MultiplicationCounter:
    """统计域上多项式乘法（NTT 卷积）的实际调用次数。"""

    __slots__ = ("count",)

    def __init__(self):
        self.count = 0


def _check_modulus(p):
    if isinstance(p, bool) or not isinstance(p, int):
        raise UnsupportedModulusError(
            f"p must be an int prime from MODULI, got {type(p).__name__} ({p!r})"
        )
    if p not in MODULI:
        raise UnsupportedModulusError(
            f"unsupported modulus {p}: only the fixed NTT primes "
            f"{MODULI} are supported"
        )


def _validate_field_elements(values, name, limit, what):
    """校验整数序列并规范到 [0, p-1] 之前的结构/类型检查。"""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise InvalidPolynomialError(
            f"{name!r} must be an iterable of integers, "
            f"got {type(values).__name__}"
        )
    items = list(values)
    if not items:
        raise LengthOutOfBoundError(
            f"{name!r} must be non-empty ({what} count between 1 and {limit})"
        )
    if len(items) > limit:
        raise LengthOutOfBoundError(
            f"{name!r} has {len(items)} {what}, exceeds limit {limit}"
        )
    for index, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidCoefficientError(
                f"{name}[{index}] must be an int, got "
                f"{type(value).__name__} ({value!r})"
            )
    return items


def _trim(coeffs):
    """去掉高位零，零多项式规范为 [0]。"""
    last = len(coeffs) - 1
    while last > 0 and coeffs[last] == 0:
        last -= 1
    return coeffs[: last + 1]


def _normalize(coeffs, p):
    """逐系数取模到 [0, p-1] 并去高位零。"""
    return _trim([c % p for c in coeffs])


def _poly_mul(a, b, p, counter):
    """域 F_p 上的多项式乘法（NTT 卷积），并计入乘法次数。"""
    a = _trim(a)
    b = _trim(b)
    if a == [0] or b == [0]:
        return [0]
    size = 1 << (len(a) + len(b) - 2).bit_length()
    counter.count += 1
    return _trim(convolution_mod(a, b, size, p))


def _poly_add(a, b, p):
    size = max(len(a), len(b))
    out = [0] * size
    for i in range(size):
        out[i] = ((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p
    return _trim(out)


def _poly_mod(f, g, p):
    """首一除式 g 的域上多项式取余（综合长除法，精确模算术）。"""
    rem = list(f)
    deg_g = len(g) - 1
    # g 首一（乘积树节点均为首一），无需除以首项系数。
    for k in range(len(rem) - 1, deg_g - 1, -1):
        coeff = rem[k] % p
        if coeff:
            for j in range(deg_g):
                rem[k - deg_g + j] = (rem[k - deg_g + j] - coeff * g[j]) % p
        rem[k] = 0
    return _trim([c % p for c in rem[:deg_g]] or [0])


def _build_product_tree(points, p, counter):
    """自底向上构建乘积树，返回节点列表的层序结构（最后一层为根）。

    每层是多项式列表；叶子层为 (x - x_i)。树以层列表返回，便于余式树
    自顶向下复用同一结构。
    """
    level = [[(-x) % p, 1] for x in points]
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_poly_mul(level[i], level[i + 1], p, counter))
        if len(level) % 2:
            nxt.append(level[-1])
        levels.append(nxt)
        level = nxt
    return levels


def _remainder_tree_values(f, levels, p):
    """沿乘积树自顶向下取余，返回 f 在各叶子点的函数值列表。"""
    root = levels[-1][0]
    if len(f) < len(root):
        root_rem = _trim([c % p for c in f])
    else:
        root_rem = _poly_mod(f, root, p)
    remainders = [root_rem]
    for level in reversed(levels[:-1]):
        nxt = []
        for i, node in enumerate(level):
            parent = remainders[i // 2]
            # 奇数层最后一个节点被原样带上上一层（父节点即同一多项式），
            # 其余式直接继承，无需重复取余。
            if len(level) % 2 and i == len(level) - 1:
                nxt.append(parent)
            else:
                nxt.append(_poly_mod(parent, node, p))
        remainders = nxt
    # 叶子节点为 (x - x_i)，余式为常数，即函数值。
    return [r[0] % p for r in remainders]


def evaluate_multi(coeffs, points, p):
    """在域 F_p 上对多项式做批量多点求值。

    参数:
        coeffs: 整数系数（低次在前），长度 1..129（次数 <= 128）。
            支持零多项式与次数高于点数的情形；系数先规范到 [0, p-1]。
        points: 1..128 个整数求值点，按模 p 约化；允许重复（求值
            不去重，逐点返回）。点 0 合法。
        p: 必须是 :data:`exact_conv.ntt.MODULI` 中的固定 NTT 素数。

    返回:
        :class:`EvaluationResult`，``values[i] = f(points[i]) mod p``。
    """
    _check_modulus(p)
    raw_coeffs = _validate_field_elements(coeffs, "coeffs", MAX_DEGREE + 1,
                                          "coefficient")
    raw_points = _validate_field_elements(points, "points", MAX_POINTS, "point")
    f = _normalize(raw_coeffs, p)
    xs = [x % p for x in raw_points]

    counter = _MultiplicationCounter()
    levels = _build_product_tree(xs, p, counter)
    values = _remainder_tree_values(f, levels, p)
    return EvaluationResult(values, p, counter.count)


def interpolate(points, values, p):
    """在域 F_p 上插值：返回过所有 (x_i, y_i) 的唯一次数 < n 的多项式。

    参数:
        points: 1..128 个整数插值点，按模 p 约化后**必须互异**，
            否则抛 :class:`DuplicatePointError`。
        values: 与 ``points`` 等长的整数函数值。
        p: 必须是 :data:`exact_conv.ntt.MODULI` 中的固定 NTT 素数。

    返回:
        :class:`InterpolationResult`，系数规范到 [0, p-1]，次数 < 点数。
    """
    _check_modulus(p)
    raw_points = _validate_field_elements(points, "points", MAX_POINTS, "point")
    raw_values = _validate_field_elements(values, "values", MAX_POINTS, "value")
    if len(raw_points) != len(raw_values):
        raise ExactConvolutionError(
            f"points and values must have equal length, got "
            f"{len(raw_points)} and {len(raw_values)}"
        )
    xs = [x % p for x in raw_points]
    ys = [y % p for y in raw_values]
    seen = {}
    for index, x in enumerate(xs):
        if x in seen:
            raise DuplicatePointError(
                f"duplicate interpolation point: points[{seen[x]}] and "
                f"points[{index}] are both {x} (mod {p})"
            )
        seen[x] = index

    n = len(xs)
    if n == 1:
        return InterpolationResult([ys[0]], p, 0)

    counter = _MultiplicationCounter()
    levels = _build_product_tree(xs, p, counter)
    root = levels[-1][0]
    # M'(x)：乘积树根的导数。
    derivative = _trim([(i * root[i]) % p for i in range(1, len(root))] or [0])
    # 复用余式树在各点求 M'(x_i)。
    deriv_at = _remainder_tree_values(derivative, levels, p)
    weights = [ys[i] * pow(deriv_at[i], p - 2, p) % p for i in range(n)]

    # 自底向上合并：P = P_left * M_right + P_right * M_left。
    level_polys = [[w] for w in weights]
    for level in levels[:-1]:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            left_p, right_p = level_polys[i], level_polys[i + 1]
            left_m, right_m = level[i], level[i + 1]
            merged = _poly_add(_poly_mul(left_p, right_m, p, counter),
                               _poly_mul(right_p, left_m, p, counter), p)
            nxt.append(merged)
        if len(level) % 2:
            nxt.append(level_polys[-1])
        level_polys = nxt
    return InterpolationResult(_normalize(level_polys[0], p), p, counter.count)
