"""多点求值 / 插值迭代演示：真实计算，约 1 秒内完成。

展示本次迭代新增的两条路径（均基于已有 NTT 乘法）：

1. 素域多点求值（乘积树 + 余式树），并与独立 Horner 逐点对照；
2. 素域插值（Lagrange + 余式树求导数值），并抽查函数值；
3. 一项实际触发的边界：重复插值点被拒绝。

运行：python demo_iteration.py
"""

import random
import time

from exact_conv import (
    DuplicatePointError,
    MODULI,
    interpolate,
    multipoint_evaluate,
)


def horner(coeffs, x, p):
    value = 0
    for coeff in reversed(coeffs):
        value = (value * x + coeff) % p
    return value


def main():
    started = time.perf_counter()
    p = MODULI[0]
    rng = random.Random(20260926)  # 固定种子，确定性演示

    print(f"素域: p = {p}")
    print("=" * 60)

    degree = 100
    point_count = 96
    coeffs = [rng.randrange(p) for _ in range(degree + 1)]
    points = rng.sample(range(10**6), point_count)

    print(f"[1] 多点求值: 次数 {degree} 的多项式, {point_count} 个点")
    evaluation = multipoint_evaluate(coeffs, points, p)
    mismatches = sum(
        1 for x, v in zip(points, evaluation.values)
        if horner(coeffs, x, p) != v
    )
    print(f"    实际多项式乘法次数: {evaluation.multiplication_count}")
    print(f"    前 4 个值: {evaluation.values[:4]}")
    print(f"    独立 Horner 逐点对照: {point_count} 点全部一致"
          if mismatches == 0 else f"    不一致 {mismatches} 处!")

    print("-" * 60)
    print(f"[2] 插值: 由这 {point_count} 个 (x, y) 还原次数 < {point_count} "
          f"的唯一多项式")
    interpolation = interpolate(points, evaluation.values, p)
    print(f"    实际多项式乘法次数: {interpolation.multiplication_count}")
    print(f"    还原多项式次数: {len(interpolation.coefficients) - 1}")
    # 原多项式次数 100 > 95，还原的是其模 乘积树多项式 后的唯一代表；
    # 用抽查函数值验证（不依赖往返恒等）。
    checks = [0, point_count // 2, point_count - 1]
    ok = all(
        horner(interpolation.coefficients, points[i], p) == evaluation.values[i]
        for i in checks
    )
    print(f"    抽查点 {checks} 的函数值: {'全部吻合' if ok else '不吻合!'}")

    print("-" * 60)
    print("[3] 边界情况: 重复插值点必须拒绝")
    try:
        interpolate([3, 7, 3], [1, 2, 5], p)
    except DuplicatePointError as exc:
        print(f"    已按预期拒绝: {exc}")
    else:
        print("    错误: 未拒绝重复点!")

    elapsed = time.perf_counter() - started
    print("=" * 60)
    print(f"总耗时 {elapsed:.3f} 秒（真实计算，无 sleep / 预录）")


if __name__ == "__main__":
    main()

