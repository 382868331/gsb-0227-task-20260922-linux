"""固定块长重叠累加卷积。

把两个（可能长于 ``MAX_POLY_LENGTH`` 的）整数系数序列按固定块长 ``S``
切块：

    A_i = a[i*S : (i+1)*S],  B_j = b[j*S : (j+1)*S]

每一对块 ``(i, j)`` 调用既有的精确卷积 :func:`exact_conv.convolve`，得到
长度 ``|A_i| + |B_j| - 1`` 的块卷积，再按偏移 ``(i + j) * S`` 重叠累加进
总长 ``n + m - 1`` 的输出：

    C[(i+j)*S + k] += conv(A_i, B_j)[k]

这正是整数环上的分配律（“学校乘法”的分块形式），与整块卷积在数学上恒等。
负系数无需特殊处理：旧精确卷积本身在 CRT 对称区间恢复有符号整数。

本模块只做切块与重叠累加；块内的模数选择、NTT、CRT 全部沿用旧实现与旧
错误语义，因此原公开入口的所有保证对每个块逐字成立。
"""

from . import MAX_POLY_LENGTH, convolve
from .errors import InvalidBlockSizeError


class BlockCallRecord:
    """单次块对卷积的记录（不可变）。

    Attributes:
        block_a: A 侧块序号（从 0 开始）。
        block_b: B 侧块序号（从 0 开始）。
        len_a: 该 A 块的实际长度（末块可能短于块长）。
        len_b: 该 B 块的实际长度。
        offset: 块卷积写入输出的起始下标 ``(block_a + block_b) * block_size``。
        moduli: 该块卷积实际选用的 NTT 模数。
        transform_length: 该块卷积补零后的 NTT 长度（2 的幂）。
    """

    __slots__ = ("block_a", "block_b", "len_a", "len_b", "offset",
                 "moduli", "transform_length")

    def __init__(self, block_a, block_b, len_a, len_b, offset,
                 moduli, transform_length):
        self.block_a = block_a
        self.block_b = block_b
        self.len_b = len_b
        self.len_a = len_a
        self.offset = offset
        self.moduli = tuple(moduli)
        self.transform_length = transform_length

    def __repr__(self):
        return (f"BlockCallRecord(a{self.block_a}[{self.len_a}] x "
                f"b{self.block_b}[{self.len_b}] @offset {self.offset}, "
                f"moduli={self.moduli}, ntt_len={self.transform_length})")

    def __eq__(self, other):
        if not isinstance(other, BlockCallRecord):
            return NotImplemented
        return (self.block_a, self.block_b, self.len_a, self.len_b,
                self.offset, self.moduli, self.transform_length) == (
            other.block_a, other.block_b, other.len_a, other.len_b,
            other.offset, other.moduli, other.transform_length)

    def __hash__(self):
        return hash((self.block_a, self.block_b, self.len_a, self.len_b,
                     self.offset, self.moduli, self.transform_length))


class WorkingSetStats:
    """固定块长卷积的工作数组大小统计（不可变）。

    所有口径都是“数组槽位数”（Python int 引用槽），不是字节数；只统计
    本库自身显式分配的工作数组，不含 Python 大整数对象本身的开销。

    Attributes:
        block_size: 使用的固定块长 S。
        blocks_a / blocks_b: 两侧的块数（含可能更短的末块）。
        block_calls: 实际调用旧精确卷积的块对总数（零块不跳过，同样调用）。
        output_slots: 输出累加数组槽位 ``n + m - 1``。
        per_call_ntt_slots: 单次块卷积在峰值时同时存活的 NTT 工作数组槽位。
            旧实现对每个模数依次分配 3 个长度 ``transform_length`` 的数组
            （fa、fb、逆变换结果），块切片只建视图不另建大数组，故
            ``3 * transform_length`` 是该次调用的工作集；CRT 逐系数重建，
            不产生与变换等长的额外数组。
        peak_ntt_slots: 所有块对中最大的单次工作集槽位。
        total_ntt_slots: 各块对工作集槽位之和（累计分配量，非同时存活）。
        transform_lengths: 各块对使用的 NTT 长度元组（按块对调用顺序）。
    """

    __slots__ = ("block_size", "blocks_a", "blocks_b", "block_calls",
                 "output_slots", "per_call_ntt_slots", "peak_ntt_slots",
                 "total_ntt_slots", "transform_lengths")

    def __init__(self, block_size, blocks_a, blocks_b, output_slots,
                 per_call_ntt_slots, transform_lengths):
        self.block_size = block_size
        self.blocks_a = blocks_a
        self.blocks_b = blocks_b
        self.block_calls = len(per_call_ntt_slots)
        self.output_slots = output_slots
        self.per_call_ntt_slots = tuple(per_call_ntt_slots)
        self.transform_lengths = tuple(transform_lengths)
        self.peak_ntt_slots = max(per_call_ntt_slots, default=0)
        self.total_ntt_slots = sum(per_call_ntt_slots)

    @property
    def peak_working_slots(self):
        """同时存活意义下的峰值槽位：输出数组 + 最大一次块卷积工作集。"""
        return self.output_slots + self.peak_ntt_slots

    def __repr__(self):
        return (f"WorkingSetStats(block_size={self.block_size}, "
                f"blocks={self.blocks_a}x{self.blocks_b}, "
                f"calls={self.block_calls}, output_slots={self.output_slots}, "
                f"peak_ntt_slots={self.peak_ntt_slots}, "
                f"total_ntt_slots={self.total_ntt_slots})")


class BlockedConvolutionResult:
    """固定块长卷积的结果与统计（不可变）。

    Attributes:
        coefficients: 精确卷积系数 ``C[k] = sum_i A[i] B[k-i]``，长度
            ``len(a) + len(b) - 1``，为 Python 精确整数。
        block_size: 实际使用的固定块长。
        block_calls: 每次块对卷积的 :class:`BlockCallRecord`（按调用顺序）。
        working_set: :class:`WorkingSetStats` 工作数组大小统计。
    """

    __slots__ = ("coefficients", "block_size", "block_calls", "working_set")

    def __init__(self, coefficients, block_size, block_calls, working_set):
        self.coefficients = coefficients
        self.block_size = block_size
        self.block_calls = tuple(block_calls)
        self.working_set = working_set

    def __repr__(self):
        stats = self.working_set
        return (f"BlockedConvolutionResult(length={len(self.coefficients)}, "
                f"block_size={self.block_size}, "
                f"blocks={stats.blocks_a}x{stats.blocks_b}, "
                f"calls={stats.block_calls})")


def _split_blocks(values, block_size):
    """按固定块长切分，返回列表的列表（末块允许更短，每块非空）。"""
    return [values[i:i + block_size]
            for i in range(0, len(values), block_size)]


def _validate_block_size(block_size):
    """块长必须是 1..MAX_POLY_LENGTH 的精确 int（拒绝 bool/float/越界）。"""
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise InvalidBlockSizeError(
            f"block_size must be an int, got {type(block_size).__name__} "
            f"({block_size!r})"
        )
    if block_size < 1:
        raise InvalidBlockSizeError(
            f"block_size must be >= 1, got {block_size}"
        )
    if block_size > MAX_POLY_LENGTH:
        raise InvalidBlockSizeError(
            f"block_size {block_size} exceeds MAX_POLY_LENGTH="
            f"{MAX_POLY_LENGTH}; each block must be accepted by the exact "
            f"single-block convolution"
        )


def blocked_convolve(a, b, block_size):
    """固定块长重叠累加的精确整数卷积。

    长序列按 ``block_size`` 切块，对每一对块（含零块，不跳过）调用旧的
    :func:`exact_conv.convolve`，按 ``(i+j)*block_size`` 偏移把块卷积
    重叠累加进输出。结果与整块“学校乘法”逐系数相同，并支持负系数。

    参数:
        a: 整数系数可迭代对象，长度 >= 1（本入口不设 256 上限；块内仍受
            旧契约的长度与系数界保护）。
        b: 同上。
        block_size: 固定块长，``1 .. MAX_POLY_LENGTH`` 的精确 ``int``
            （拒绝 bool、float、0、负数与超过 256 的值）。

    返回:
        :class:`BlockedConvolutionResult`。

    异常:
        InvalidBlockSizeError: 块长类型或取值非法。
        InvalidPolynomialError / LengthOutOfBoundError /
        InvalidCoefficientError / CoefficientBoundError: 输入不满足旧契约
            （系数在切块前按原序列下标统一校验，错误信息如 ``a[200]``）。

    所有输入校验在输出累加器建立之前完成：校验失败时本函数不产生任何
    部分结果，也不修改输入。
    """
    # 延迟导入以复用包级校验（避免本模块与包 __init__ 之间的导入环顾虑）。
    from . import _validate_polynomial

    _validate_block_size(block_size)
    # 长输入不做 256 上限校验：上限只针对单个块。
    coeff_a = _validate_polynomial(a, "a", max_length=None)
    coeff_b = _validate_polynomial(b, "b", max_length=None)

    blocks_a = _split_blocks(coeff_a, block_size)
    blocks_b = _split_blocks(coeff_b, block_size)

    output_len = len(coeff_a) + len(coeff_b) - 1
    output = [0] * output_len

    records = []
    per_call_slots = []
    transform_lengths = []
    for i, block_a in enumerate(blocks_a):
        for j, block_b in enumerate(blocks_b):
            result = convolve(block_a, block_b)
            offset = (i + j) * block_size
            piece = result.coefficients
            for k, value in enumerate(piece):
                output[offset + k] += value
            records.append(BlockCallRecord(
                i, j, len(block_a), len(block_b), offset,
                result.moduli, result.transform_length))
            per_call_slots.append(3 * result.transform_length)
            transform_lengths.append(result.transform_length)

    stats = WorkingSetStats(
        block_size, len(blocks_a), len(blocks_b), output_len,
        per_call_slots, transform_lengths)
    return BlockedConvolutionResult(output, block_size, records, stats)
