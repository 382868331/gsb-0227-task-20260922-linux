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


def naive_ring_mul(a, b, n, negative=False):
    """按定义的朴素环乘：双循环逐项 a[i]*b[j]，按指数模 n 折返。

    cyclic（``negative=False``）：指数 k 折返到 k mod n；
    negacyclic（``negative=True``）：k = q*n + r，贡献带符号 (-1)^q。
    与 :func:`naive_convolve` 一样只用 Python 大整数，不导入被测核心。
    """
    result = [0] * n
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            k = i + j
            q, r = divmod(k, n)
            term = x * y
            if negative and (q & 1):
                term = -term
            result[r] += term
    return result
