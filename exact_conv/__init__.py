"""精确整数多项式卷积与循环/负循环乘法：多模 NTT + CRT 重建。

公开接口：

- :func:`convolve`：精确*线性*整数卷积；
- :func:`cyclic_mul`：商环 ``Z[x]/(x^N-1)`` 中的乘法（循环卷积）；
- :func:`negacyclic_mul`：商环 ``Z[x]/(x^N+1)`` 中的乘法（负循环卷积）；
- :func:`plan_moduli`：纯函数的模数规划。

线性卷积算法流程：

1. 严格校验输入（结构、bool、NaN/Infinity、长度与系数界），绝不静默纠正。
2. 用实际系数界 B = min(n, m) * max|A| * max|B| 选择*最少*的固定模数，
   要求模数乘积严格大于 2*B，保证 CRT 可在对称区间唯一恢复。
3. 每个选定模数做一次补零到 2 的幂的迭代 NTT 卷积。
4. Garner CRT 把各系数的多模余数恢复为有符号精确整数。

环乘（``N`` 为 1..256 的 2 的幂，输入长度 1..N）先得到精确线性卷积
``C``（长 ``n+m-1``），再在 Python 精确有符号整数上按指数折返：

- 循环（``x^N = 1``）：``R[r] = sum_q C[qN+r]``；
- 负循环（``x^N = -1``）：``R[r] = sum_q (-1)^q C[qN+r]``。

折返后每个系数至多是 ``K = ceil((n+m-1)/N)`` 个线性系数之和，故折返
绝对界为 ``K*B``；选模要求模数乘积严格大于 ``2*(K*B)``，即 CRT 覆盖的
是*折返相加之后*的界。负循环变号作用在 CRT 重建出的有符号整数上，
绝不在模 p 余数上变号后忘记重建。

所有运算均为精确整数运算（Python 大整数 + 模算术），不存在浮点，
因此结果无任何舍入误差。
"""

from .crt import reconstruct
from .errors import (
    CoefficientBoundError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    UnprovableBoundError,
)
from .ntt import MODULI, PRIMITIVE_ROOT, convolution_mod

#: 单个多项式的最大长度（接口边界）。
MAX_POLY_LENGTH = 256

#: 单个系数绝对值的上限（接口边界）。
MAX_COEFF_ABS = 10**6

__all__ = [
    "convolve",
    "cyclic_mul",
    "negacyclic_mul",
    "plan_moduli",
    "ConvolutionResult",
    "RingProductResult",
    "MODULI",
    "PRIMITIVE_ROOT",
    "MAX_POLY_LENGTH",
    "MAX_COEFF_ABS",
    "ExactConvolutionError",
    "InvalidPolynomialError",
    "InvalidCoefficientError",
    "LengthOutOfBoundError",
    "CoefficientBoundError",
    "UnprovableBoundError",
]


class ConvolutionResult:
    """卷积结果及其可证性元数据（不可变）。

    Attributes:
        coefficients: 精确卷积系数 ``C[k] = sum_i A[i] B[k-i]``，
            长度 ``len(a) + len(b) - 1``，为 Python 精确整数。
        moduli: 实际使用的 NTT 模数（有序元组，按优先使用顺序）。
        coefficient_bound: 实际采用的系数界
            ``min(n, m) * max|A| * max|B|``（非负整数）。
        modulus_product: 所选模数的乘积；满足
            ``modulus_product > 2 * coefficient_bound``。
        transform_length: 补零后的 NTT 长度（2 的幂）。
    """

    __slots__ = ("coefficients", "moduli", "coefficient_bound",
                 "modulus_product", "transform_length")

    def __init__(self, coefficients, moduli, coefficient_bound,
                 modulus_product, transform_length):
        self.coefficients = coefficients
        self.moduli = tuple(moduli)
        self.coefficient_bound = coefficient_bound
        self.modulus_product = modulus_product
        self.transform_length = transform_length

    def __repr__(self):
        return (f"ConvolutionResult(length={len(self.coefficients)}, "
                f"moduli={self.moduli}, bound={self.coefficient_bound}, "
                f"product={self.modulus_product}, ntt_len={self.transform_length})")

    def __eq__(self, other):
        if not isinstance(other, ConvolutionResult):
            return NotImplemented
        return (self.coefficients == other.coefficients
                and self.moduli == other.moduli)

    def __hash__(self):
        return hash((tuple(self.coefficients), self.moduli))


class RingProductResult:
    """环乘结果及其可证性元数据（不可变）。

    Attributes:
        coefficients: 长度恰为 ``N`` 的精确整数系数（下标即次数，
            高位不足项补零；这是商环的规范表示，不是静默预约化输入）。
        relation: 关系字符串，``"x^N - 1"`` 或 ``"x^N + 1"``。
        ring_size: 环参数 ``N``（1..256 的 2 的幂）。
        moduli: 实际使用的 NTT 模数（有序元组）。
        modulus_product: 所选模数乘积，满足
            ``modulus_product > 2 * coefficient_bound``。
        linear_coefficient_bound: 底层线性卷积的逐项绝对界
            ``B = min(n,m) * max|A| * max|B|``。
        coefficient_bound: *折返后*系数的可核验绝对界
            ``ceil((n+m-1)/N) * B``；输出逐系数满足
            ``|coefficients[r]| <= coefficient_bound``。
        transform_length: 底层线性卷积补零后的 NTT 长度（2 的幂）。
    """

    __slots__ = ("coefficients", "relation", "ring_size", "moduli",
                 "modulus_product", "linear_coefficient_bound",
                 "coefficient_bound", "transform_length")

    def __init__(self, coefficients, relation, ring_size, moduli,
                 modulus_product, linear_coefficient_bound,
                 coefficient_bound, transform_length):
        self.coefficients = coefficients
        self.relation = relation
        self.ring_size = ring_size
        self.moduli = tuple(moduli)
        self.modulus_product = modulus_product
        self.linear_coefficient_bound = linear_coefficient_bound
        self.coefficient_bound = coefficient_bound
        self.transform_length = transform_length

    def __repr__(self):
        return (f"RingProductResult(relation={self.relation!r}, N={self.ring_size}, "
                f"moduli={self.moduli}, linear_bound={self.linear_coefficient_bound}, "
                f"folded_bound={self.coefficient_bound}, "
                f"product={self.modulus_product}, ntt_len={self.transform_length})")

    def __eq__(self, other):
        if not isinstance(other, RingProductResult):
            return NotImplemented
        return (self.coefficients == other.coefficients
                and self.relation == other.relation
                and self.ring_size == other.ring_size
                and self.moduli == other.moduli)

    def __hash__(self):
        return hash((tuple(self.coefficients), self.relation,
                     self.ring_size, self.moduli))


def _validate_polynomial(values, name):
    """校验单个多项式，返回系数元组；错误信息带参数名与下标，可定位。"""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise InvalidPolynomialError(
            f"{name!r} must be an iterable of integer coefficients, "
            f"got {type(values).__name__}"
        )
    coeffs = list(values)
    if not coeffs:
        raise LengthOutOfBoundError(
            f"{name!r} must be non-empty (length between 1 and {MAX_POLY_LENGTH})"
        )
    if len(coeffs) > MAX_POLY_LENGTH:
        raise LengthOutOfBoundError(
            f"{name!r} length {len(coeffs)} exceeds MAX_POLY_LENGTH="
            f"{MAX_POLY_LENGTH}"
        )
    for index, value in enumerate(coeffs):
        # bool 是 int 的子类，按契约必须拒绝。
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float):
                # float 一律拒绝：本接口只接受精确整数；对 NaN/Infinity
                # 给出明确可定位的说明。
                if value != value:
                    detail = "NaN"
                elif value in (float("inf"), float("-inf")):
                    detail = "Infinity"
                else:
                    detail = repr(value)
                raise InvalidCoefficientError(
                    f"{name}[{index}] must be an exact int, got float {detail}"
                )
            raise InvalidCoefficientError(
                f"{name}[{index}] must be an int, got {type(value).__name__}"
                f" ({value!r})"
            )
        if abs(value) > MAX_COEFF_ABS:
            raise CoefficientBoundError(
                f"{name}[{index}] = {value} has abs value {abs(value)} > "
                f"MAX_COEFF_ABS={MAX_COEFF_ABS}; refusing to produce a result "
                f"that cannot be proven exact"
            )
    return coeffs


def _next_power_of_two(value):
    """返回 >= value 的最小 2 的幂（value >= 1）。"""
    return 1 << (value - 1).bit_length()


def _require_nonneg_int(value, name):
    """公开函数的整数参数：拒绝 bool（bool 是 int 子类）与非 int、负数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExactConvolutionError(
            f"{name} must be an int, got {type(value).__name__} ({value!r})"
        )
    if value < 0:
        raise ExactConvolutionError(f"{name} must be non-negative, got {value}")


def _validate_ring_size(n):
    """校验环参数 N：必须是 1..256 的 2 的幂（拒绝 bool）。"""
    if isinstance(n, bool) or not isinstance(n, int):
        raise ExactConvolutionError(
            f"N must be an int, got {type(n).__name__} ({n!r})"
        )
    if n < 1 or n > MAX_POLY_LENGTH:
        raise LengthOutOfBoundError(
            f"N must be between 1 and {MAX_POLY_LENGTH}, got {n}"
        )
    if n & (n - 1):
        raise ExactConvolutionError(
            f"N must be a power of two, got {n}"
        )
    return n


def plan_moduli(n, m, max_abs_a, max_abs_b):
    """按实际系数界选择最少模数。

    四个参数均必须是非负 ``int``（拒绝 bool）。系数界取
    ``B = min(n, m) * max_abs_a * max_abs_b``。依次加入固定模数，
    返回第一个使模数乘积严格大于 ``2*B`` 的前缀；全部用完仍不够则抛
    :class:`UnprovableBoundError`（“证明无解”，与预算耗尽等情形区分）。

    返回 ``(moduli_tuple, coefficient_bound, modulus_product)``。
    """
    for value, pname in ((n, "n"), (m, "m"),
                         (max_abs_a, "max_abs_a"), (max_abs_b, "max_abs_b")):
        _require_nonneg_int(value, pname)
    bound = min(n, m) * max_abs_a * max_abs_b
    moduli, product = _plan_for_bound(bound)
    return moduli, bound, product


def _plan_for_bound(bound):
    """返回第一个使模数乘积严格大于 ``2*bound`` 的固定模数前缀。

    返回 ``(moduli_tuple, modulus_product)``；全部用完仍不够则抛
    :class:`UnprovableBoundError`。
    """
    chosen = []
    product = 1
    for mod in MODULI:
        chosen.append(mod)
        product *= mod
        if product > 2 * bound:
            return tuple(chosen), product
    raise UnprovableBoundError(
        f"coefficient bound B={bound}: product of all fixed moduli {product} "
        f"is not greater than 2*B={2 * bound}; exact reconstruction impossible"
    )


def _exact_linear(coeff_a, coeff_b, bound):
    """按给定可证绝对界 ``bound`` 计算精确线性卷积。

    返回 ``(exact_coeffs, moduli, modulus_product, ntt_size)``。选模基于
    调用方给出的界（环乘时为折返后的界，必然不小于线性逐项界），故 CRT
    对称区间恢复对线性系数同样无歧义。
    """
    output_len = len(coeff_a) + len(coeff_b) - 1
    size = _next_power_of_two(output_len)
    moduli, product = _plan_for_bound(bound)
    residue_rows = [convolution_mod(coeff_a, coeff_b, size, mod) for mod in moduli]
    exact = [
        reconstruct([residue_rows[r][k] for r in range(len(moduli))], moduli)
        for k in range(output_len)
    ]
    return exact, moduli, product, size


def convolve(a, b):
    """计算两个整数多项式的*精确*卷积（多模 NTT + CRT）。

    参数:
        a: 可迭代整数序列，长度 1..256，每个系数为 ``int``（拒绝 bool、
            float、NaN、Infinity），绝对值 <= 10**6。
        b: 同上。

    返回:
        :class:`ConvolutionResult`，其中 ``coefficients`` 为精确整数列表，
        长度 ``len(a) + len(b) - 1``。

    异常:
        InvalidPolynomialError: 输入不是整数系数的可迭代对象。
        LengthOutOfBoundError: 长度为空或超过 256。
        InvalidCoefficientError: 系数类型非法（bool/float/NaN/Infinity 等）。
        CoefficientBoundError: 系数绝对值超过 10**6。
        UnprovableBoundError: 固定模数乘积不足以保证精确恢复。
    """
    coeff_a = _validate_polynomial(a, "a")
    coeff_b = _validate_polynomial(b, "b")

    n, m = len(coeff_a), len(coeff_b)
    max_abs_a = max((abs(x) for x in coeff_a), default=0)
    max_abs_b = max((abs(x) for x in coeff_b), default=0)

    bound = min(n, m) * max_abs_a * max_abs_b
    exact, moduli, product, size = _exact_linear(coeff_a, coeff_b, bound)
    return ConvolutionResult(exact, moduli, bound, product, size)


def _ring_multiply(a, b, n, negative):
    """环乘公共实现；``negative=False`` 为 x^N-1，``True`` 为 x^N+1。"""
    ring_size = _validate_ring_size(n)
    coeff_a = _validate_polynomial(a, "a")
    coeff_b = _validate_polynomial(b, "b")
    if len(coeff_a) > ring_size:
        raise LengthOutOfBoundError(
            f"a length {len(coeff_a)} exceeds ring size N={ring_size}; "
            f"refusing to truncate silently"
        )
    if len(coeff_b) > ring_size:
        raise LengthOutOfBoundError(
            f"b length {len(coeff_b)} exceeds ring size N={ring_size}; "
            f"refusing to truncate silently"
        )

    linear_len = len(coeff_a) + len(coeff_b) - 1
    max_abs_a = max((abs(x) for x in coeff_a), default=0)
    max_abs_b = max((abs(x) for x in coeff_b), default=0)
    linear_bound = min(len(coeff_a), len(coeff_b)) * max_abs_a * max_abs_b
    # 折返后每个余数至多叠加 ceil(linear_len / N) 个线性系数。
    blocks = (linear_len + ring_size - 1) // ring_size
    folded_bound = blocks * linear_bound

    linear, moduli, product, size = _exact_linear(
        coeff_a, coeff_b, folded_bound)

    # 折返在 CRT 重建之后的 Python 精确有符号整数上进行：负循环的变号
    # 是有符号整数取负，而不是模 p 余数取负。
    folded = [0] * ring_size
    for k, value in enumerate(linear):
        q, r = divmod(k, ring_size)
        if negative and (q & 1):
            folded[r] -= value
        else:
            folded[r] += value

    relation = f"x^{ring_size} + 1" if negative else f"x^{ring_size} - 1"
    return RingProductResult(
        folded, relation, ring_size, moduli, product,
        linear_bound, folded_bound, size)


def cyclic_mul(a, b, n):
    """商环 ``Z[x]/(x^N-1)`` 中的多项式乘法（循环卷积）。

    ``N`` 必须是 1..256 的 2 的幂（拒绝 bool/浮点）；``a``、``b`` 为
    长度 1..N 的整数系数序列（系数规则同 :func:`convolve`），长度超过
    ``N`` 直接拒绝，不做静默截断/预约化。

    返回 :class:`RingProductResult`，``coefficients`` 长度恰为 ``N``，
    满足 ``R[r] = sum_{q>=0} C[qN+r]``，其中 ``C`` 是精确线性卷积。
    """
    return _ring_multiply(a, b, n, negative=False)


def negacyclic_mul(a, b, n):
    """商环 ``Z[x]/(x^N+1)`` 中的多项式乘法（负循环卷积）。

    规则同 :func:`cyclic_mul`，折返时高次块带符号：
    ``R[r] = sum_{q>=0} (-1)^q C[qN+r]``（``x^N = -1``）。变号在
    CRT 重建出的有符号整数上完成。
    """
    return _ring_multiply(a, b, n, negative=True)
