# 精确整数卷积：多模 NTT + CRT 重建

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
python -m unittest discover -s tests -v   # 运行全部必要测试
python demo.py                            # 约 1 秒内完成的真实计算演示
```
### 准备虚拟环境（可选，无第三方依赖）

\`\`\`bash
python -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python demo_iteration.py
\`\`\`

本次迭代新增演示：

\`\`\`bash
python demo_iteration.py   # 多点求值 + 插值 + 重复点拒绝边界，约 1 秒内
\`\`\`

## 接口

```python
from exact_conv import convolve, plan_moduli

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

## 素域多点求值与插值（本次迭代新增）

在已有 NTT 多项式乘法之上，为固定 NTT 素域增加批量求值与还原：

```python
from exact_conv import multipoint_evaluate, interpolate, MODULI

p = MODULI[0]                       # 只允许三个固定 NTT 素数之一
ev = multipoint_evaluate(coeffs, points, p)
ev.values                         # list[int]：各点函数值，均在 [0, p-1]
ev.multiplication_count           # 本次实际执行的多项式乘法次数

ip = interpolate(points, values, p)
ip.coefficients                   # 次数 < 点数的唯一多项式，升幂、
                                  # 规范到 [0, p-1]、去高位零（零多项式为 [0]）
ip.multiplication_count
```

- `coeffs`：升幂系数，精确 `int`（拒绝 bool），任意大小，先规范到
  `[0, p-1]`；长度 0..129（空序列即零多项式），**次数可高于点数**。
- `points`：1..128 个求值点；求值允许重复点（只是重复求值），
  **插值要求互异**（模 p 后判重，重复即 `DuplicatePointError`）。
- `values`：与 `points` 等长的函数值。
- 输入输出一律是域元素（`[0, p-1]` 的 `int`），**不把旧整数 CRT
  重建结果直接当域元素**；旧整数卷积路径完全不变。

### 算法

1. **乘积树**：叶子为 `(x - x_i)`，补齐到 2 的幂后自底向上成对相乘，
   乘法复用 `exact_conv.ntt.convolution_mod`；与单位多项式 1 的合并
   不产生实际乘法、不计数。
2. **余式树求值**：自顶向下逐层取余（首一除式朴素长除法，不占用乘法
   计数），叶子即各点函数值。零多项式、单点、点 0、次数高于点数均走
   同一条路径。
3. **插值**：根节点即 `M(x) = ∏(x - x_i)`，对形式导数 `M'` 复用余式树
   在全部点求值，再按 Lagrange 公式
   `f = Σ y_i · M'(x_i)⁻¹ · M/(x - x_i)` 累加（`M/(x - x_i)` 用首一
   线性因子综合除法）。
4. **乘法计数**：结果对象的 `multiplication_count` 统计实际发生的多项式
   乘法（乘积树合并）次数；取余与综合除法不调用多项式乘法。

### 新增限制与错误语义

| 项 | 值 |
|---|---|
| 可用素域 | 仅 `MODULI` 中三个固定 NTT 素数 |
| 点数上限 | `MAX_POINTS = 128` |
| 次数上限 | `MAX_DEGREE = 128`（系数至多 129 项） |

新增错误（仍继承 `ExactConvolutionError` / `ValueError`）：

- `InvalidModulusError`：模数不是精确 `int`（拒绝 bool/float），或不在
  固定 NTT 素数集合内——不接受任意模数。
- `DuplicatePointError`：插值点在模 p 后不互异，直接拒绝而非返回某个
  “看似合理”的多项式。

点数/次数超界抛 `LengthOutOfBoundError`；系数/点/值类型非法抛
`InvalidCoefficientError`（拒绝 bool 与 float）；非可迭代输入抛
`InvalidPolynomialError`。错误信息带参数名与下标，可定位。

### 验证口径

- 多点求值：与 `tests/test_multipoint.py` 中**独立 Horner 逐点**参考
  对照（不导入被测核心）。
- 插值：与独立 **Vandermonde 高斯消元**参考对照；求值→插值往返只作
  补充检验。

### 固定常量

| 项 | 值 |
|---|---|
| NTT 素数（按优先使用顺序） | `1004535809`, `998244353`, `469762049` |
| 原根 | 均为 `3` |
| 最长变换支持 | 分别 2^21 / 2^23 / 2^26 点（本库最多用到 1024 点） |
| 单多项式长度上限 | `MAX_POLY_LENGTH = 256` |
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
   计算，**不导入被测核心**；测试期望值全部来自它。NTT 正逆往返、多模余数
   一致性等只作为补充检验。

## 测试覆盖

`python -m unittest discover -s tests -v` 共 30+ 个用例，含：

- 手算小例、确定性随机多组规模（1×1 到 256×256）；
- **全负**（负×负、负×正）、**全零/零散点**、**长尾零**（256 长但尾部
  大片零，卷积长 511）；
- **单模 / 双模切换**、模数选择严格阈值（`2B` 恰在模两侧）、三模档与
  “无法证明精确”路径；
- 接口上限 256 项 × ±10^6 对拍朴素参考；
- `bool` / `NaN` / `Infinity` / 有限 `float` / 字符串 / `None` 拒绝；
  超长、空、超系数界拒绝；错误信息可定位；输入不被修改。

## 误差口径

本库没有浮点结果：输入是整数，中间为模算术，输出为 Python 精确大整数。
**误差恒为 0，不使用任何容差**。所谓“对称区间恢复”是整数集合上的精确
双射（在 `M > 2B` 前提下），不是近似。

## 已知限制 / 非目标

- 只实现本题契约的数学子集：三个固定素数、长度 ≤ 256、系数绝对值 ≤ 10^6；
  不宣称通用 NTT/CRT 行业标准兼容，不支持任意模数或任意原根配置。
- 纯 Python 实现，追求正确与可证性，而非 SIMD/原生级吞吐；接口规模下
  256×256 双模卷积为毫秒量级。
- 不提供网站、CLI 产品、报表等；终端命令仅用于跑测试与演示。
- 不保证输入序列惰性生成器可被重复消费（内部只迭代一次）。
