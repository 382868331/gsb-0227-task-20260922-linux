"""固定输入演示：精确整数卷积（多模 NTT + CRT）。

运行：python demo.py

内容全部为仓库内确定性数据（固定种子的伪随机数），真实调用本库计算，
不 sleep、不使用预录结果。打印：
  1. 一个单模情形（小系数界）与一个双模情形（大系数界）的模数选择和精确结果；
  2. 一个实际触发的拒绝边界：系数绝对值 10^6+1 被拒绝，而非取模冒充精确；
  3. 固定块长入口：含负系数与零块的手算例子、超过 256 项的长输入、
     工作数组大小统计，以及块长非法 / 系数超界两个实际触发的错误。
"""

import random
import time

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_POLY_LENGTH,
    CoefficientBoundError,
    InvalidBlockSizeError,
    blocked_convolve,
    convolve,
)
from tests.reference import naive_convolve


def _show_case(title, a, b):
    print(f"=== {title} ===")
    print(f"输入长度: len(a)={len(a)}, len(b)={len(b)}")
    started = time.perf_counter()
    result = convolve(a, b)
    elapsed = time.perf_counter() - started
    print(f"系数界 B = min(n,m)*max|A|*max|B| = {result.coefficient_bound}")
    print(f"选用模数 ({len(result.moduli)} 个): {result.moduli}")
    two_bound = 2 * result.coefficient_bound
    ok = "成立" if result.modulus_product > two_bound else "不成立"
    print(f"模数乘积 M = {result.modulus_product} > 2*B = {two_bound} -> {ok}")
    print(f"补零后 NTT 长度: {result.transform_length}")
    print(f"精确卷积（共 {len(result.coefficients)} 个系数，前 8 个）:")
    print(f"  {result.coefficients[:8]}")
    # 与独立朴素定义对照（演示中的自检，朴素实现不调用被测核心）。
    assert result.coefficients == naive_convolve(a, b), "结果与朴素定义不一致"
    print(f"已与朴素整数卷积定义逐项核对一致；耗时 {elapsed * 1000:.1f} ms")
    print()
    return result


def main():
    print("精确整数多项式卷积：多模 NTT + CRT 重建演示")
    print("固定模数: 1004535809, 998244353, 469762049（原根均为 3）")
    print()

    # 情形 1：小系数界 -> 单模即可（B = 8*1000*1000 = 8e6 < p1/2）。
    a1 = [1000, -900, 800, -700, 600, -500, 400, -300]
    b1 = [100, -200, 300, -400, 500, -600, 700, -800]
    _show_case("情形 1：小系数界（预期单模）", a1, b1)

    # 情形 2：大系数界 -> 必须双模（固定种子确定性生成，真实计算）。
    rng = random.Random(20260923)
    a2 = [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(64)]
    b2 = [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(64)]
    result2 = _show_case("情形 2：大系数界（预期双模）", a2, b2)
    peak = max(abs(x) for x in result2.coefficients)
    print(f"情形 2 结果系数绝对值峰值 = {peak}（远超任一单模，证明 CRT 真正生效）")
    print()

    # 边界：系数 10^6+1 超出可证界，必须拒绝而不是静默取模。
    print("=== 情形 3：超界输入触发拒绝（预期 CoefficientBoundError）===")
    try:
        convolve([MAX_COEFF_ABS + 1, 2, 3], [4, 5])
    except CoefficientBoundError as exc:
        print(f"如预期拒绝输入，错误类型: {type(exc).__name__}")
        print(f"可定位信息: {exc}")
        print("（没有返回任何取模结果：本库不以取模冒充精确卷积）")
    else:  # pragma: no cover - 演示护栏：若未拒绝则明确失败
        raise AssertionError("超界输入竟被接受，契约被破坏")

    print()
    _show_blocked_cases()


def _show_blocked_cases():
    print("=== 情形 4：固定块长重叠累加（手算例子，含负系数与零块）===")
    # (1 - 2x + 3x^2)(4 - x) = 4 - 9x + 14x^2 - 3x^3；块长 2 时
    # A 切成 [1,-2],[3]，B 切成 [4],[-1]，4 个块对在边界上重叠累加。
    a = [1, -2, 3]
    b = [4, -1]
    result = blocked_convolve(a, b, 2)
    blocks_a_demo = [a[0:2], a[2:4]]
    blocks_b_demo = [b[0:2]]
    print(f"A={a}, B={b}, block_size=2")
    print(f"分块: A -> {blocks_a_demo}, B -> {blocks_b_demo}")
    print(f"精确结果: {result.coefficients}（手算期望 [4, -9, 14, -3]）")
    assert result.coefficients == [4, -9, 14, -3]
    assert result.coefficients == naive_convolve(a, b)
    for rec in result.block_calls:
        print(f"  块对 ({rec.block_a},{rec.block_b}) 偏移 {rec.offset}: "
              f"块长 {rec.len_a}x{rec.len_b}, NTT {rec.transform_length} 点, "
              f"模数 {rec.moduli}")
    stats = result.working_set
    print(f"工作数组统计: {stats.block_calls} 次块卷积, "
          f"输出 {stats.output_slots} 槽, 峰值 NTT 工作集 "
          f"{stats.peak_ntt_slots} 槽, 累计 {stats.total_ntt_slots} 槽")
    print()

    print("=== 情形 5：长输入（600x600，超过旧入口 256 上限）===")
    # 确定性构造：a 全 +1e6、b 全 -1e6 -> 每个卷积系数严格为
    # -min(k+1, 600, 1199-k)*1e12，峰值恰为 -600e12（解析可断言），
    # 同时压满块内双模 CRT 与跨块重叠累加。
    a = [MAX_COEFF_ABS] * 600
    b = [-MAX_COEFF_ABS] * 600
    started = time.perf_counter()
    result = blocked_convolve(a, b, 128)
    elapsed = time.perf_counter() - started
    stats = result.working_set
    print(f"len(a)=len(b)=600（a 全 +1e6，b 全 -1e6）, block_size=128 "
          f"-> {stats.blocks_a}x{stats.blocks_b} = {stats.block_calls} 个块对")
    print(f"输出长度 {len(result.coefficients)}（= 600+600-1）")
    assert result.coefficients == naive_convolve(a, b)
    print("已与朴素整数卷积定义逐项核对一致（重叠区多个块对的部分积相加）")
    peak = max(abs(x) for x in result.coefficients)
    print(f"结果系数绝对值峰值 = {peak}（学校乘法峰值 = 600*1e12 = 6e14）")
    assert peak == 600 * 10 ** 12
    assert result.coefficients[599] == -600 * 10 ** 12
    print("首/中/尾三个系数: "
          f"{result.coefficients[0]}, {result.coefficients[599]}, "
          f"{result.coefficients[-1]}（期望 -1e12, -6e14, -1e12）")
    print(f"工作数组: 输出 {stats.output_slots} 槽 + 峰值 NTT "
          f"{stats.peak_ntt_slots} 槽 = 同时存活峰值 "
          f"{stats.peak_working_slots} 槽; "
          f"各块 NTT 长度分布 {sorted(set(stats.transform_lengths))}")
    print(f"耗时 {elapsed * 1000:.1f} ms")
    print()

    print("=== 情形 6：固定块长入口的实际错误 ===")
    # 错误 1：块长非法（0 与超过单块上限）。
    for bad_block_size in (0, MAX_POLY_LENGTH + 1):
        try:
            blocked_convolve([1, 2, 3], [4, 5], bad_block_size)
        except InvalidBlockSizeError as exc:
            print(f"block_size={bad_block_size} -> "
                  f"{type(exc).__name__}: {exc}")
        else:  # pragma: no cover
            raise AssertionError("非法块长竟被接受")
    # 错误 2：长输入中靠后位置的超界系数（全局下标 a[300]）。
    bad_a = [1] * 301
    bad_a[300] = MAX_COEFF_ABS + 1
    try:
        blocked_convolve(bad_a, [1, 2], 64)
    except CoefficientBoundError as exc:
        print(f"超界长输入 -> {type(exc).__name__}: {exc}")
        print("（校验先于累加器建立：没有半更新状态，也没有取模结果）")
    else:  # pragma: no cover
        raise AssertionError("超界长输入竟被接受")


if __name__ == "__main__":
    main()
