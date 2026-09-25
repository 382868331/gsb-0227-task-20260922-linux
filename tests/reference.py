"""独立朴素整数卷积参考。

只用双重循环和 Python 大整数直接按定义累加，*不导入* exact_conv，
因此测试的期望值独立于被测核心（NTT/CRT 出错不会污染参考）。
"""


def naive_convolve(a, b):
    """按 C[k] = sum_{i+j=k} a[i]*b[j] 的定义朴素计算。"""
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            result[i + j] += x * y
    return result


def _naive_ring_mul(a, b, size, sign_when_cross):
    """独立双循环：逐项 (i,j) 按指数 i+j 折返到 N 个余数类。

    ``sign_when_cross`` 为 +1（x^N ≡ 1，循环）或 -1（x^N ≡ -1，负循环）；
    负循环每跨过一个 N 次块符号再翻转一次。空序列按零多项式处理。
    """
    result = [0] * size
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            exponent = i + j
            block, remainder = divmod(exponent, size)
            term = x * y
            if block & 1:
                term *= sign_when_cross
            result[remainder] += term
    return result


def naive_cyclic_mul(a, b, size):
    """朴素定义：Z[x]/(x^N - 1) 中的乘积，逐项折返，不导入被测核心。"""
    return _naive_ring_mul(a, b, size, 1)


def naive_negacyclic_mul(a, b, size):
    """朴素定义：Z[x]/(x^N + 1) 中的乘积，奇数次跨块变号，不导入被测核心。"""
    return _naive_ring_mul(a, b, size, -1)
