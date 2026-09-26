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


def horner_evaluate(coeffs, x, p):
    """独立 Horner 逐点求值参考（不导入被测核心）。"""
    acc = 0
    for coeff in reversed(coeffs):
        acc = (acc * (x % p) + coeff) % p
    return acc


def vandermonde_interpolate(points, values, p):
    """独立 Vandermonde 消元插值参考（不导入被测核心）。

    在 F_p 上构造范德蒙矩阵并做高斯消元，解出次数 < n 的多项式系数。
    """
    n = len(points)
    xs = [x % p for x in points]
    # 增广矩阵 [V | y]。
    mat = []
    for i in range(n):
        row = [pow(xs[i], k, p) for k in range(n)]
        row.append(values[i] % p)
        mat.append(row)
    for col in range(n):
        pivot = next(r for r in range(col, n) if mat[r][col] % p != 0)
        mat[col], mat[pivot] = mat[pivot], mat[col]
        inv = pow(mat[col][col], p - 2, p)
        mat[col] = [(v * inv) % p for v in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] % p:
                factor = mat[r][col]
                mat[r] = [(a - factor * b) % p for a, b in zip(mat[r], mat[col])]
    return [mat[i][n] % p for i in range(n)]
