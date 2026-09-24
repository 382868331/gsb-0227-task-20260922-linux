"""固定输入演示：精确整数卷积（多模 NTT + CRT）与固定块长入口。

运行：python demo.py

内容全部为仓库内确定性数据（固定种子的伪随机数/手算小例），真实调用
本库计算，不 sleep、不使用预录结果。打印：
  1. 旧入口：一个单模情形与一个双模情形的模数选择和精确结果；
  2. 一个实际触发的拒绝边界：系数绝对值 10^6+1 被拒绝，而非取模冒充；
  3. 固定块长入口：超过 256 长度的手算/随机例的重叠累加结果与工作数组
     大小统计（含内部零块）；
  4. 块长入口的实际错误：非法块长（bool、0）与超长输入中的超界系数。
"""

import random
import time

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_POLY_LENGTH,
    CoefficientBoundError,
    InvalidBlockLengthError,
    LengthOutOfBoundError,
    convolve,
    convolve_blocked,
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

    # ---- 固定块长入口 ----
    print("=== 情形 4：固定块长入口，长度超过旧上限 256（正常结果）===")
    # [1]*257 与 [1, -1] 的学校乘法：除首尾外全部抵消，
    # 结果必为 [1, 0, 0, ..., 0, -1]（长 258），可手算核对。
    a4 = [1] * (MAX_POLY_LENGTH + 1)
    b4 = [1, -1]
    started = time.perf_counter()
    r4 = convolve_blocked(a4, b4, 32)
    elapsed = time.perf_counter() - started
    expect4 = [1] + [0] * 256 + [-1]
    assert r4.coefficients == expect4 == naive_convolve(a4, b4)
    print(f"len(a)={len(a4)} > 256（旧入口直接拒绝），块长=32，"
          f"分块 {r4.blocks_a}x{r4.blocks_b}，块对调用 {r4.block_pair_calls} 次")
    print(f"结果长 {len(r4.coefficients)}：首项={r4.coefficients[0]}, "
          f"末项={r4.coefficients[-1]}，中间 {sum(1 for x in r4.coefficients if x == 0)} "
          f"个 0（错位重叠累加后恰好抵消）")
    print(f"工作数组统计: {r4.work}")
    print(f"已与朴素学校乘法逐项核对一致；耗时 {elapsed * 1000:.1f} ms")
    print()

    print("=== 情形 5：大块 + 内部零块 + 大系数（块长 64，300 项 ±10^6）===")
    rng = random.Random(20260924)
    # 中间恰好夹一个 64 项的全零块（第 4 块），零块也照常调用旧卷积。
    a5 = [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(192)] + [0] * 64 \
        + [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(44)]
    b5 = [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(300)]
    started = time.perf_counter()
    r5 = convolve_blocked(a5, b5, 64)
    elapsed = time.perf_counter() - started
    assert r5.coefficients == naive_convolve(a5, b5), "结果与朴素定义不一致"
    print(f"len(a)=len(b)=300，分块 {r5.blocks_a}x{r5.blocks_b}，"
          f"块对调用 {r5.block_pair_calls} 次（a 含 1 个内部全零块，未跳过）")
    print(f"结果系数绝对值峰值 = {max(map(abs, r5.coefficients))}，"
          f"前 5 项 = {r5.coefficients[:5]}")
    print(f"块内最大 NTT 长度 = {r5.transform_length}，用到模数 {r5.moduli}")
    print(f"工作数组统计: {r5.work}")
    print(f"已与朴素整数卷积定义逐项核对一致；耗时 {elapsed * 1000:.1f} ms")
    print()

    # 块长入口的实际错误：非法块长与超长输入中的超界系数。
    print("=== 情形 6：块长入口的实际拒绝（预期 InvalidBlockLengthError / "
          "CoefficientBoundError）===")
    for bad_block_size in (True, 0, MAX_POLY_LENGTH + 1):
        try:
            convolve_blocked([1, 2, 3, 4], [1, 1], bad_block_size)
        except InvalidBlockLengthError as exc:
            print(f"block_size={bad_block_size!r} -> {type(exc).__name__}: {exc}")
        else:  # pragma: no cover - 演示护栏
            raise AssertionError(f"非法块长 {bad_block_size!r} 竟被接受")
    # 旧入口对 257 长度报长度错误；新入口接受长度，但第 300 个系数超界，
    # 必须给出可定位错误且不留半更新状态。
    bad_long = [1] * 299 + [MAX_COEFF_ABS + 1]
    try:
        convolve_blocked(bad_long, [2], 64)
    except CoefficientBoundError as exc:
        print(f"长输入第 300 项超界 -> {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - 演示护栏
        raise AssertionError("超长输入中的超界系数竟未被拒绝")
    # 对照：同样的长度用旧入口会先撞长度上限。
    try:
        convolve(bad_long, [2])
    except LengthOutOfBoundError as exc:
        print(f"同一输入走旧入口 -> {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
