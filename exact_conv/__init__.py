"""精确整数多项式卷积：多模 NTT + CRT 重建。

公开接口见模块级函数 :func:`convolve` 与 :func:`plan_moduli`，以及
``ConvolutionResult`` 结果对象。算法流程：

1. 严格校验输入（结构、bool、NaN/Infinity、长度与系数界），绝不静默纠正。
2. 用实际系数界 B = min(n, m) * max|A| * max|B| 选择*最少*的固定模数，
   要求模数乘积严格大于 2*B，保证 CRT 可在对称区间唯一恢复。
3. 每个选定模数做一次补零到 2 的幂的迭代 NTT 卷积。
4. Garner CRT 把各系数的多模余数恢复为有符号精确整数。

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
    "plan_moduli",
    "ConvolutionResult",
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
    chosen = []
    product = 1
    for mod in MODULI:
        chosen.append(mod)
        product *= mod
        if product > 2 * bound:
            return tuple(chosen), bound, product
    raise UnprovableBoundError(
        f"coefficient bound B={bound}: product of all fixed moduli {product} "
        f"is not greater than 2*B={2 * bound}; exact reconstruction impossible"
    )


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

    moduli, bound, product = plan_moduli(n, m, max_abs_a, max_abs_b)

    output_len = n + m - 1
    size = _next_power_of_two(output_len)

    # 每个选定模数独立做 NTT 卷积，收集该模数下的系数余数。
    residue_rows = [convolution_mod(coeff_a, coeff_b, size, mod) for mod in moduli]

    exact = [
        reconstruct([residue_rows[r][k] for r in range(len(moduli))], moduli)
        for k in range(output_len)
    ]
    return ConvolutionResult(exact, moduli, bound, product, size)
