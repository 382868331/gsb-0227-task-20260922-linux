# 精确整数卷积 / 循环与负循环乘法：多模 NTT + CRT 重建

离线、零依赖、可导入的整数多项式乘法库。系数用 Python 大整数与模算术
处理，**全程不使用浮点**，因此结果是数学上的精确值，不存在舍入误差，
也不需要任何误差容差。

- **Python 版本**：Python 3.14.7（Linux 原生实测；任何带大整数的 Python 3
  均可运行，仅用标准库）。
- **依赖**：无第三方依赖，安装 Python 后无需 `pip install` 任何东西。
- **离线**：核心、测试、演示均不访问网络，不需要账号、密钥、数据库、
  中间件、云 API 或 Docker。演示数据由仓库内固定种子的确定性代码生成。

## 快速开始

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 全部测试（48 个）
./.venv/bin/python demo.py                            # 约 1 秒内完成的真实计算演示
```

## 接口

```python
from exact_conv import convolve, cyclic_mul, negacyclic_mul, plan_moduli

result = convolve(a, b)
result.coefficients     # list[int]：C[k] = sum_{i+j=k} a[i]*b[j]，精确整数
result.moduli           # tuple[int,...]：实际选用的 NTT 模数（最少个数）
result.coefficient_bound    # B = min(n,m) * max|A| * max|B|
result.modulus_product      # 所选模数乘积 M，保证 M > 2*B
result.transform_length     # 补零后的 NTT 长度（2 的幂）
```

- `a`、`b`：整数系数的可迭代对象（list/tuple/生成器均可），长度 1..256。
- 系数必须是 `int`，绝对值 `<= 10**6`。
- 返回长度 `len(a) + len(b) - 1` 的精确整数列表；输入不会被修改。

`plan_moduli(n, m, max_abs_a, max_abs_b)` 是纯函数的模数规划，返回
`(moduli, bound, modulus_product)`，可在不做卷积时单独查询“最少要用几个模”。

### 循环 / 负循环乘法

```python
result = cyclic_mul(a, b, N)       # Z[x]/(x^N - 1)，x^N = 1
result = negacyclic_mul(a, b, N)   # Z[x]/(x^N + 1)，x^N = -1

result.coefficients                # list[int]，长度恰为 N（规范表示）
result.relation                    # "x^N - 1" 或 "x^N + 1"
result.ring_size                   # N
result.moduli                      # 实际选用的 NTT 模数
result.modulus_product             # M，保证 M > 2 * coefficient_bound
result.linear_coefficient_bound    # B = min(n,m)*max|A|*max|B|
result.coefficient_bound           # 折返后绝对界 ceil((n+m-1)/N) * B
result.transform_length            # 底层线性卷积的 NTT 长度（2 的幂）
```

- `N` 必须是 **1..256 的 2 的幂**（`1,2,4,...,256`），否则拒绝
  （非 2 幂 / 越界 / `bool` / 浮点均报错，不静默纠正）。
- `a`、`b` 长度 **1..N**（系数规则同 `convolve`）；长度超过 `N`
  **直接拒绝，不做静默截断或预约化**。短输入合法，输出长度仍恰为 `N`
  （不足的高次项补零是商环的规范表示，不是修改输入）。
- 算法：先做精确线性卷积 `C`（长 `n+m-1`），再在 **CRT 重建之后的
  Python 有符号整数**上按指数折返：
  - 循环：`R[r] = sum_q C[qN+r]`；
  - 负循环：`R[r] = sum_q (-1)^q C[qN+r]`。
  负循环的变号是有符号整数取负，**不是**模 `p` 余数取负后忘记重建。
- **折返界与模数覆盖**：折返后每个系数至多叠加
  `K = ceil((n+m-1)/N)` 个线性系数，故绝对界为 `K*B`。选模要求
  模数乘积 **`M > 2*(K*B)`**——覆盖折返相加之后的界。接口内
  `n+m-1 <= 2N-1`，故 `K <= 2`；最坏 `2*B = 512*1e12 ≈ 5.12e14`，
  双模乘积（约 `1e18`）足够。
- `N=1`：`Z[x]/(x-1)` 与 `Z[x]/(x+1)` 对常数乘法结果一致；
  零多项式结果是长度 `N` 的全零序列。

### 固定常量

| 项 | 值 |
|---|---|
| NTT 素数（按优先使用顺序） | `1004535809`, `998244353`, `469762049` |
| 原根 | 均为 `3` |
| 最长变换支持 | 分别 2^21 / 2^23 / 2^26 点（本库最多用到 1024 点） |
| 单多项式长度上限 / 最大环参数 N | `MAX_POLY_LENGTH = 256`（N 还须为 2 的幂） |
| 系数绝对值上限 | `MAX_COEFF_ABS = 10**6` |

### 错误语义

所有本库错误继承 `exact_conv.ExactConvolutionError`（同时是 `ValueError`）：

- `InvalidPolynomialError`：不是系数可迭代对象（如整数、`None`、字符串）。
- `LengthOutOfBoundError`：空序列或长度超过 256。
- `InvalidCoefficientError`：系数不是精确 `int`——**拒绝 `bool`**
  （`True/False` 虽是 `int` 子类也拒绝）、拒绝 `float`，并对
  `NaN` / `Infinity` 给出明确可定位信息；拒绝字符串、`None`、复数等。
- `CoefficientBoundError`：系数绝对值超过 10^6。**直接拒绝而非取模冒充
  精确结果。**
- `UnprovableBoundError`：固定三模数乘积仍不足以保证 CRT 唯一恢复（在
  256×10^6 的接口上限内不会发生）。它表示“数学上无法证明精确”，与任何
  搜索预算耗尽之类的运行期失败明确区分。

错误信息都带参数名与下标，例如 `a[2] ...`、`b[1] ...`，不静默纠正输入。
环乘中长度超过 `N` 抛 `LengthOutOfBoundError`；`N` 不是 1..256 的 2
的幂抛 `ExactConvolutionError`。
`convolve` 是纯函数：校验失败或成功都不改动输入，不产生部分结果。

## 算法与设计取舍

1. **校验先行**：任何计算前完成结构、类型、长度、系数界检查。
2. **按实际系数界选最少模数**。每个卷积系数是至多 `min(n,m)` 个
   `|a_i b_j| <= max|A|*max|B|` 之和，故
   `B = min(n,m) * max|A| * max|B|` 是可证上界。按固定顺序逐个加入模数，
   取第一个满足 **乘积严格大于 2B** 的前缀：
   - 单模覆盖约 `5e8`；
   - 双模（`1004535809*998244353 ≈ 1.003e18`）覆盖接口上限最坏界
     `256*1e12 = 2.56e14`，故接口内最多用两个模；第三个模
     `469762049` 固定保留并在 `plan_moduli` 中完整支持。
3. **迭代蝶形 NTT**（`exact_conv/ntt.py`）：补零到不短于 `n+m-1` 的最小
   2 的幂（接口内最多 1024 点）；原地 Cooley–Tukey 位反转蝶形，自行实现，
   不调用任何第三方同类算法。逆变换使用原根逆元并在末尾乘逆长度。
4. **Garner 混合基数 CRT**（`exact_conv/crt.py`）：增量合并各模余数，
   再把 `[0, M)` 代表折叠到对称整数区间 `(-M/2, M/2]`。由 `M > 2B` 且
   `|真值| <= B`，折叠结果与真值无歧义一致。
5. **朴素整数参考独立实现**（`tests/reference.py`）：只用双重循环按定义
   计算，**不导入被测核心**；测试期望值全部来自它。`naive_ring_mul`
   用独立双循环逐项按指数折返（负循环带 `(-1)^q` 符号）核对环乘。
   NTT 正逆往返、多模余数一致性等只作为补充检验。

## 测试覆盖

`./.venv/bin/python -m unittest discover -s tests -v` 共 48 个用例，含：

- 手算小例、确定性随机多组规模（1×1 到 256×256）；
- **全负**（负×负、负×正）、**全零/零散点**、**长尾零**（256 长但尾部
  大片零，卷积长 511）；
- **单模 / 双模切换**、模数选择严格阈值（`2B` 恰在模两侧）、三模档与
  “无法证明精确”路径；
- 接口上限 256 项 × ±10^6 对拍朴素参考；
- `bool` / `NaN` / `Infinity` / 有限 `float` / 字符串 / `None` 拒绝；
  超长、空、超系数界拒绝；错误信息可定位；输入不被修改。
- 环乘：手算折返例、循环/负循环符号差逐项核验、**大系数折返抵消**
  （线性系数远超单模数，检验 CRT 有符号重建后再整数变号）、**N=1**、
  **最高次乘最高次**（`x^(2N-2)` 在两种环中符号相反）、零多项式、短输入、
  无折返时退化为线性卷积、`N` 非法值（非 2 幂/越界/bool/浮点）拒绝、
  输入长度超过 `N` 拒绝；全部对拍独立双循环折返参考。

## 误差口径

本库没有浮点结果：输入是整数，中间为模算术，输出为 Python 精确大整数。
**误差恒为 0，不使用任何容差**。所谓“对称区间恢复”是整数集合上的精确
双射（在 `M > 2B` 前提下），不是近似。

## 已知限制 / 非目标

- 只实现本题契约的数学子集：三个固定素数、长度 ≤ 256、系数绝对值 ≤ 10^6、
  环参数 N 为 1..256 的 2 的幂；不宣称通用 NTT/CRT 行业标准兼容，
  不支持任意模数、任意原根配置或非 2 幂循环长度。
- 环乘不做分块长卷积，也不提供任何密码学安全声明。
- 纯 Python 实现，追求正确与可证性，而非 SIMD/原生级吞吐；接口规模下
  256×256 双模卷积为毫秒量级。
- 不提供网站、CLI 产品、报表等；终端命令仅用于跑测试与演示。
- 不保证输入序列惰性生成器可被重复消费（内部只迭代一次）。
