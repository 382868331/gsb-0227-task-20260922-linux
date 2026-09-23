"""固定三个 NTT 素数与自行实现的迭代蝶形变换。

模数（均为 NTT 友好素数，p-1 含大的 2 的幂因子）：

    998244353  = 119 * 2**23 + 1
    1004535809 = 479 * 2**21 + 1
    469762049  = 7   * 2**26 + 1

原根均为 3。接口上限内卷积长度为 n + m - 1 <= 511，补零到 1024 点，
三个模数均支持（最浅的 1004535809 支持到 2**21 点）。
"""

from .errors import ExactConvolutionError

#: 固定 NTT 素数，按“优先使用”顺序（大的在前，使单模覆盖范围最大）。
MODULI = (1004535809, 998244353, 469762049)

#: 每个模数的原根。
PRIMITIVE_ROOT = 3

#: 各模数支持的最大变换长度（p - 1 中 2 的幂因子）。
_MAX_TRANSFORM_LENGTH = {
    469762049: 1 << 26,
    998244353: 1 << 23,
    1004535809: 1 << 21,
}


def _bit_reverse_permutation(values, size, mod):
    """原地位反转置换（迭代 NTT 的标准前置步骤）。"""
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]


def ntt(coeffs, size, mod, inverse=False):
    """对 ``coeffs`` 做长度 ``size`` 的迭代 NTT（正向或逆向），返回新列表。

    补零由调用方完成；``size`` 必须是 2 的幂且不超过该模数支持的长度。
    逆变换在每一层使用逆旋转因子，最后统一乘以逆长度 (mod p)。
    """
    if size & (size - 1):
        raise ExactConvolutionError(f"NTT length must be a power of two, got {size}")
    max_size = _MAX_TRANSFORM_LENGTH[mod]
    if size > max_size:
        raise ExactConvolutionError(
            f"NTT length {size} exceeds {max_size} supported by modulus {mod}"
        )

    values = [0] * size
    for i, value in enumerate(coeffs):
        values[i] = value % mod

    _bit_reverse_permutation(values, size, mod)

    root = PRIMITIVE_ROOT
    if inverse:
        # 逆变换：使用原根的逆元，并在末尾乘逆长度。
        root = pow(root, mod - 2, mod)

    length = 2
    while length <= size:
        half = length >> 1
        # 本层旋转因子 w_length = root^((mod-1)/length)
        w_length = pow(root, (mod - 1) // length, mod)
        for start in range(0, size, length):
            w = 1
            end = start + half
            for i in range(start, end):
                u = values[i]
                v = values[i + half] * w % mod
                values[i] = (u + v) % mod
                values[i + half] = (u - v) % mod
                w = w * w_length % mod
        length <<= 1

    if inverse:
        inv_size = pow(size, mod - 2, mod)
        for i in range(size):
            values[i] = values[i] * inv_size % mod

    return values


def convolution_mod(a, b, size, mod):
    """在域 F_p 上计算卷积的前 ``len(a)+len(b)-1`` 个系数（模 mod）。"""
    fa = ntt(a, size, mod, inverse=False)
    fb = ntt(b, size, mod, inverse=False)
    for i in range(size):
        fa[i] = fa[i] * fb[i] % mod
    full = ntt(fa, size, mod, inverse=True)
    return full[: len(a) + len(b) - 1]
