"""精确整数多项式卷积：多模 NTT + CRT 重建。

公开接口见模块级函数 :func:`convolve`、:func:`convolve_blocked` 与
:func:`plan_moduli`，以及 ``ConvolutionResult`` / ``BlockedConvolutionResult``
结果对象。算法流程：

1. 严格校验输入（结构、bool、NaN/Infinity、长度与系数界），绝不静默纠正。
2. 用实际系数界 B = min(n, m) * max|A| * max|B| 选择*最少*的固定模数，
   要求模数乘积严格大于 2*B，保证 CRT 可在对称区间唯一恢复。
3. 每个选定模数做一次补零到 2 的幂的迭代 NTT 卷积。
4. Garner CRT 把各系数的多模余数恢复为有符号精确整数。

:func:`convolve_blocked` 在此之上提供*固定块长*入口：把（可超过 256 的）
输入按给定块长切块，对每一对块调用 :func:`convolve` 做旧的精确卷积，再把
各块卷积按 ``(块a起点 + 块b起点)`` 偏移重叠累加。系数允许负数，结果与
学校乘法（朴素定义卷积）逐项一致。

所有运算均为精确整数运算（Python 大整数 + 模算术），不存在浮点，
因此结果无任何舍入误差。
"""

from .crt import reconstruct
from .errors import (
    CoefficientBoundError,
    ExactConvolutionError,
    InvalidBlockLengthError,
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
    "convolve_blocked",
    "plan_moduli",
    "ConvolutionResult",
    "BlockedConvolutionResult",
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
    "InvalidBlockLengthError",
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


class WorkArrayStats:
    """分块卷积的工作数组大小统计（不可变，单位：整数槽位）。

    一个“槽位”指一个存放 Python 整数的列表元素（不解释为字节数：
    Python 大整数的实际字节占用随数值增长）。统计的是数组*长度*，
    与运行期分配器无关，可按公式复核。

    Attributes:
        accumulator_slots: 重叠累加输出数组长度 ``n + m - 1``，全程存活。
        transform_length: 任意一对块卷积使用的最大 NTT 长度（2 的幂）。
        total_ntt_slots: 所有块对卷积 NTT 工作数组长度之和
            ``Σ_{i,j} next_pow2(len(a_i)+len(b_j)-1)``，即累计申请过的
            变换工作数组槽位总数（各次调用串行、用后即释放）。
        total_block_result_slots: 所有块对精确卷积结果长度之和
            ``Σ_{i,j} (len(a_i)+len(b_j)-1)``，即重叠累加写入的总槽位。
        max_residue_row_slots: 最忙的一对块卷积中，各模数余数行的槽位
            ``max_{i,j} len(moduli_ij)*(len(a_i)+len(b_j)-1)``。
        peak_working_slots: 同一时刻存活槽位的*保守上界*：
            输出累加数组 + 一次块卷积内同时存在的 NTT 数组（至多 2 个
            ``transform_length`` 数组）+ 全部模数余数行 + 块精确结果。
    """

    __slots__ = ("accumulator_slots", "transform_length", "total_ntt_slots",
                 "total_block_result_slots", "max_residue_row_slots",
                 "peak_working_slots")

    def __init__(self, accumulator_slots, transform_length, total_ntt_slots,
                 total_block_result_slots, max_residue_row_slots,
                 peak_working_slots):
        self.accumulator_slots = accumulator_slots
        self.transform_length = transform_length
        self.total_ntt_slots = total_ntt_slots
        self.total_block_result_slots = total_block_result_slots
        self.max_residue_row_slots = max_residue_row_slots
        self.peak_working_slots = peak_working_slots

    def __repr__(self):
        return (f"WorkArrayStats(accumulator={self.accumulator_slots}, "
                f"ntt_len={self.transform_length}, "
                f"total_ntt={self.total_ntt_slots}, "
                f"total_block_results={self.total_block_result_slots}, "
                f"max_residue_rows={self.max_residue_row_slots}, "
                f"peak_working={self.peak_working_slots})")

    def __eq__(self, other):
        if not isinstance(other, WorkArrayStats):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name)
                   for name in self.__slots__)

    def __hash__(self):
        return hash(tuple(getattr(self, name) for name in self.__slots__))


class BlockedConvolutionResult:
    """固定块长分块卷积的结果与统计（不可变）。

    Attributes:
        coefficients: 精确卷积系数，长度 ``len(a) + len(b) - 1``。
        block_size: 实际使用的固定块长。
        blocks_a / blocks_b: 两个输入各被切成的块数（最后一块可较短）。
        block_pair_calls: 调用旧精确卷积 :func:`convolve` 的块对总数
            （= ``blocks_a * blocks_b``，零块也照常调用，不跳过）。
        transform_length: 块卷积的最大 NTT 长度（2 的幂）。
        moduli: 所有块对实际用到的模数的并集（按固定优先顺序排列）。
        work: :class:`WorkArrayStats` 工作数组大小统计。
    """

    __slots__ = ("coefficients", "block_size", "blocks_a", "blocks_b",
                 "block_pair_calls", "transform_length", "moduli", "work")

    def __init__(self, coefficients, block_size, blocks_a, blocks_b,
                 block_pair_calls, transform_length, moduli, work):
        self.coefficients = coefficients
        self.block_size = block_size
        self.blocks_a = blocks_a
        self.blocks_b = blocks_b
        self.block_pair_calls = block_pair_calls
        self.transform_length = transform_length
        self.moduli = tuple(moduli)
        self.work = work

    def __repr__(self):
        return (f"BlockedConvolutionResult(length={len(self.coefficients)}, "
                f"block_size={self.block_size}, blocks=({self.blocks_a},"
                f"{self.blocks_b}), pair_calls={self.block_pair_calls}, "
                f"ntt_len={self.transform_length}, moduli={self.moduli}, "
                f"work={self.work!r})")

    def __eq__(self, other):
        if not isinstance(other, BlockedConvolutionResult):
            return NotImplemented
        return (self.coefficients == other.coefficients
                and self.block_size == other.block_size)

    def __hash__(self):
        return hash((tuple(self.coefficients), self.block_size))


def _validate_polynomial(values, name, max_length=MAX_POLY_LENGTH):
    """校验系数序列，返回系数列表；错误信息带参数名与下标，可定位。

    ``max_length`` 为长度上限（含）；``None`` 表示仅要求非空、不设上限
    （供固定块长入口使用：长序列由分块处理，单块仍不超过 256）。
    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise InvalidPolynomialError(
            f"{name!r} must be an iterable of integer coefficients, "
            f"got {type(values).__name__}"
        )
    coeffs = list(values)
    if not coeffs:
        if max_length is None:
            raise LengthOutOfBoundError(
                f"{name!r} must be non-empty (got an empty sequence)")
        raise LengthOutOfBoundError(
            f"{name!r} must be non-empty (length between 1 and {max_length})"
        )
    if max_length is not None and len(coeffs) > max_length:
        raise LengthOutOfBoundError(
            f"{name!r} length {len(coeffs)} exceeds MAX_POLY_LENGTH="
            f"{max_length}"
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


def _validate_block_size(block_size):
    """校验固定块长：精确正整数（拒绝 bool）且不超过单多项式长度上限。"""
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise InvalidBlockLengthError(
            f"block_size must be an exact int, got "
            f"{type(block_size).__name__} ({block_size!r})"
        )
    if not 1 <= block_size <= MAX_POLY_LENGTH:
        raise InvalidBlockLengthError(
            f"block_size must be in 1..{MAX_POLY_LENGTH}, got {block_size}"
        )
    return block_size


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


def convolve_blocked(a, b, block_size):
    """固定块长的精确卷积：切块后对每对块调用旧精确卷积并重叠累加。

    与 :func:`convolve` 的差别仅在于 *输入长度不限* 于 256（系数约束
    不变：精确 ``int``、绝对值 <= 10**6，拒绝 bool/float/NaN/Infinity）。
    两个输入按同一个固定 ``block_size`` 从低次项起顺序切块，最后一块可
    较短；每个块长度 <= 256，因而总能直接交给 :func:`convolve`。

    对每一对块 ``(a_i, b_j)`` 调用一次旧精确卷积得到
    ``c_{ij}[t] = Σ a_i[p] b_j[q], p+q=t``，再按块起点偏移
    ``(i*block_size, j*block_size)`` 累加进总输出：

        C[i*block_size + j*block_size + t] += c_{ij}[t]

    这正是学校乘法中“分段相乘、错位相加”的精确整数版本；负系数由旧卷积
    的 CRT 对称区间恢复自然支持。**零块也照常调用**旧卷积（不跳过、不做
    特判），因此块对调用数恒为 ``blocks_a * blocks_b``。

    参数:
        a: 非空整数系数可迭代对象，长度不限（单系数仍须 <= 10**6）。
        b: 同上。
        block_size: 固定块长，精确正 ``int``（拒绝 bool），
            ``1 <= block_size <= 256``。

    返回:
        :class:`BlockedConvolutionResult`，含精确系数与工作数组统计。

    异常:
        InvalidBlockLengthError: ``block_size`` 类型或范围非法。
        InvalidPolynomialError / LengthOutOfBoundError /
        InvalidCoefficientError / CoefficientBoundError: 输入不合法。

    所有校验在任何计算或输出数组分配之前完成；校验失败不会产生累加数组，
    也不会留下半更新状态，输入本身不被修改。
    """
    # 校验先行：块长与两个输入全部合法后才分配任何数组。
    block_size = _validate_block_size(block_size)
    coeff_a = _validate_polynomial(a, "a", max_length=None)
    coeff_b = _validate_polynomial(b, "b", max_length=None)

    n, m = len(coeff_a), len(coeff_b)
    blocks_a = (n + block_size - 1) // block_size
    blocks_b = (m + block_size - 1) // block_size
    output_len = n + m - 1

    # 重叠累加输出数组（唯一的长生命周期工作数组）。
    accumulator = [0] * output_len

    total_ntt_slots = 0
    total_block_result_slots = 0
    max_residue_row_slots = 0
    max_transform_length = 0
    moduli_union = []

    for bi in range(blocks_a):
        a_lo = bi * block_size
        block_a = coeff_a[a_lo:a_lo + block_size]
        for bj in range(blocks_b):
            b_lo = bj * block_size
            block_b = coeff_b[b_lo:b_lo + block_size]

            # 每一对块都调用旧的（多模 NTT + CRT）精确卷积，零块不跳过。
            pair = convolve(block_a, block_b)
            pair_coeffs = pair.coefficients
            offset = a_lo + b_lo
            for t, value in enumerate(pair_coeffs):
                accumulator[offset + t] += value

            pair_len = len(block_a) + len(block_b) - 1
            total_ntt_slots += pair.transform_length
            total_block_result_slots += pair_len
            max_residue_row_slots = max(
                max_residue_row_slots, len(pair.moduli) * pair_len)
            max_transform_length = max(
                max_transform_length, pair.transform_length)
            for mod in pair.moduli:
                if mod not in moduli_union:
                    moduli_union.append(mod)

    # 并集按固定优先顺序排列（实际出现顺序本就大致有序，这里显式排序）。
    moduli_union = [mod for mod in MODULI if mod in moduli_union]

    # 峰值同存槽位的保守上界：输出数组 + 一次块卷积内至多 3 个同存的 NTT
    # 长度数组（convolution_mod 中 fa、fb，以及逆变换内部新分配的数组）
    # + 该次已积累的各模数余数行 + 一份块精确结果。余数行峰值取所有块对
    # 的观测最大值；块结果长度以两个满块为最长（2*block_size-1）。
    peak_working_slots = (
        output_len
        + 3 * max_transform_length
        + max_residue_row_slots
        + (2 * block_size - 1)
    )

    work = WorkArrayStats(
        accumulator_slots=output_len,
        transform_length=max_transform_length,
        total_ntt_slots=total_ntt_slots,
        total_block_result_slots=total_block_result_slots,
        max_residue_row_slots=max_residue_row_slots,
        peak_working_slots=peak_working_slots,
    )
    return BlockedConvolutionResult(
        accumulator, block_size, blocks_a, blocks_b,
        blocks_a * blocks_b, max_transform_length, moduli_union, work)
