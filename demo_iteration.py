"""固定输入演示：素域上的多点求值与插值（乘积树 / 余式树）。

运行：python demo_iteration.py

内容全部为仓库内确定性数据（固定种子），真实调用本库计算，不 sleep、
不预录结果。打印：
  1. 一次多点求值（含次数高于点数的情形），并与独立 Horner 逐点对照；
  2. 一次插值，并与独立 Vandermonde 消元对照，往返求值作补充；
  3. 一个实际触发的边界：重复插值点被拒绝（DuplicatePointError）。
"""

import random
import time

from exact_conv import (
    DuplicatePointError,
    evaluate_multi,
    interpolate,
)
from exact_conv.ntt import MODULI
from tests.reference import horner_evaluate, vandermonde_interpolate

P = MODULI[1]  # 998244353


def main():
    print("素域多点求值与插值演示（乘积树 / 余式树，复用 NTT 乘法）")
    print(f"素数 p = {P}，点数与次数上限均为 128")
    print()

    rng = random.Random(20260926)

    # 情形 1：多点求值，次数 100 > 点数 48。
    print("=== 情形 1：多点求值（次数 100，48 个点，次数高于点数）===")
    coeffs = [rng.randrange(P) for _ in range(101)]
    points = [rng.randrange(P) for _ in range(48)]
    started = time.perf_counter()
    ev = evaluate_multi(coeffs, points, P)
    elapsed = time.perf_counter() - started
    expect = [horner_evaluate(coeffs, x, P) for x in points]
    assert ev.values == expect, "与独立 Horner 对照不一致"
    print(f"实际多项式乘法次数: {ev.multiplications}")
    print(f"前 5 个函数值: {ev.values[:5]}")
    print(f"已与独立 Horner 逐点对照一致（48 点）；耗时 {elapsed * 1000:.1f} ms")
    print()

    # 情形 2：插值 64 个互异点，对照 Vandermonde 消元，往返作补充。
    print("=== 情形 2：插值（64 个互异点，唯一次数 < 64 多项式）===")
    ipoints = rng.sample(range(P), 64)
    ivalues = [rng.randrange(P) for _ in range(64)]
    started = time.perf_counter()
    ip = interpolate(ipoints, ivalues, P)
    elapsed = time.perf_counter() - started
    expect_coeffs = vandermonde_interpolate(ipoints, ivalues, P)
    while len(expect_coeffs) > 1 and expect_coeffs[-1] == 0:
        expect_coeffs.pop()
    assert ip.coefficients == expect_coeffs, "与 Vandermonde 消元对照不一致"
    back = evaluate_multi(ip.coefficients, ipoints, P)
    assert back.values == ivalues, "往返求值未回到原值"
    print(f"实际多项式乘法次数: {ip.multiplications}")
    print(f"次数: {len(ip.coefficients) - 1}（< 64），系数均在 [0, p-1]")
    print(f"前 5 个系数: {ip.coefficients[:5]}")
    print(f"已与独立 Vandermonde 消元对照一致，往返求值亦吻合；"
          f"耗时 {elapsed * 1000:.1f} ms")
    print()

    # 情形 3：边界——重复插值点（模 p 意义下）必须拒绝。
    print("=== 情形 3：重复插值点触发拒绝（预期 DuplicatePointError）===")
    try:
        interpolate([7, 100, 7 + P], [1, 2, 3], P)
    except DuplicatePointError as exc:
        print(f"如预期拒绝，错误类型: {type(exc).__name__}")
        print(f"可定位信息: {exc}")
        print("（7 与 7+p 在 F_p 中是同一点；插值唯一性被破坏，直接拒绝）")
    else:
        raise AssertionError("重复插值点竟被接受，契约被破坏")


if __name__ == "__main__":
    main()
