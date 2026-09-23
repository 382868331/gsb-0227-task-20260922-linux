"""Garner 混合基数 CRT，把多模余数恢复到对称整数区间。

增量 Garner：维护已满足前 i 个同余式的代表值 x 与模数乘积 M。加入
模数 p_i 时令 x' = x + M * t，其中

    t = (r_i - x) * M^{-1}  (mod p_i)，

则 x' in [0, M*p_i) 且满足前 i+1 个同余式。模数两两互素（不同素数），
故 M 在 F_{p_i} 中的逆元总存在。

本库需要的是卷积系数的*有符号*真值。调用方保证 |true value| <= B 且
2*B < M_total，于是把无符号代表折叠到对称区间 (-M_total/2, M_total/2]
即可无歧义恢复，且恢复结果必为真值。
"""


def reconstruct(residues, moduli):
    """把 ``residues``（按 ``moduli`` 顺序）恢复为对称区间内的精确整数。

    前置条件（由调用方保证）：真值绝对值满足
    ``2 * abs(true_value) < product(moduli)``。
    返回 Python 任意精度整数。
    """
    if len(residues) != len(moduli):
        raise ValueError("residues and moduli must have equal length")
    if not residues:
        raise ValueError("at least one modulus is required")

    value = 0          # 当前无符号代表，x in [0, product)
    product = 1        # 已合并模数的乘积
    for residue, mod in zip(residues, moduli):
        t = ((residue - value) % mod) * pow(product % mod, mod - 2, mod) % mod
        value += product * t
        product *= mod

    # 折叠到对称整数区间 (-M/2, M/2]。
    half = product // 2
    if value > half:
        value -= product
    return value
