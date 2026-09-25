"""固定输入演示：精确整数卷积与循环/负循环多项式乘法（多模 NTT + CRT）。

运行：python demo.py

内容全部为仓库内确定性数据（固定种子的伪随机数），真实调用本库计算，
不 sleep、不使用预录结果。打印：
  1. 一个单模情形（小系数界）与一个双模情形（大系数界）的模数选择和精确结果；
  2. Z[x]/(x^N-1) 与 Z[x]/(x^N+1) 中的精确乘积，展示高次项折返符号；
  3. 实际触发的拒绝边界：超界系数、非法 N 与超长输入都被明确拒绝。
"""

import random
import time

from exact_conv import (
    MAX_COEFF_ABS,
    CoefficientBoundError,
    InvalidRingSizeError,
    LengthOutOfBoundError,
    convolve,
    cyclic_mul,
    negacyclic_mul,
)
from tests.reference import naive_convolve
from tests.reference import naive_cyclic_mul, naive_negacyclic_mul


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

    # 情形 4：同一组输入在两个商环中的符号差异。
    print()
    print("=== 情形 4：循环 vs 负循环折返（N=8）===")
    a4 = [3, 0, -2, 0, 7, 0, 0, 1]
    b4 = [1, 0, 4, 0, -5, 2, 0, 6]
    cyc = cyclic_mul(a4, b4, 8)
    neg = negacyclic_mul(a4, b4, 8)
    print(f"循环 Z[x]/(x^8-1) 结果  : {cyc.coefficients}")
    print(f"负循环 Z[x]/(x^8+1) 结果: {neg.coefficients}")
    assert cyc.coefficients == naive_cyclic_mul(a4, b4, 8)
    assert neg.coefficients == naive_negacyclic_mul(a4, b4, 8)
    assert cyc.modulus_product > 2 * cyc.coefficient_bound
    assert neg.modulus_product > 2 * neg.coefficient_bound
    print(f"折返后可核验系数界 B'=min(N,n+m-1)*max|A|*max|B| = "
          f"{cyc.coefficient_bound}")
    print(f"选用模数 {cyc.moduli}，乘积 {cyc.modulus_product} > 2*B'，"
          f"两个结果均与独立双循环折返参考逐项一致")
    print()

    # 边界：非法 N（非 2 的幂）与输入长度超过 N，都必须拒绝。
    print("=== 情形 5：环规模/长度边界（预期拒绝）===")
    try:
        cyclic_mul([1, 2, 3], [1], N=6)
    except InvalidRingSizeError as exc:
        print(f"N=6 非 2 的幂被拒绝: {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("非法 N 竟被接受")
    try:
        negacyclic_mul([1, 2, 3, 4, 5], [1], N=4)
    except LengthOutOfBoundError as exc:
        print(f"长度 5 > N=4 被拒绝（不静默截断/预约化）: {exc}")
    else:  # pragma: no cover
        raise AssertionError("超长输入竟被接受")


if __name__ == "__main__":
    main()
