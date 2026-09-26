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
python -m venv .venv && . .venv/bin/activate   # 可选；仅用标准库，无依赖可装
python -m unittest discover -s tests -v        # 运行全部必要测试
python demo.py                                 # 整数卷积演示（约 1 秒）
python demo_iteration.py                       # 多点求值 / 插值演示（远小于 90 秒）
```

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

## 素域多点求值与插值（本迭代新增）

复用既有 NTT 多项式乘法，在固定 NTT 素域 F_p 上提供批量求值与还原：

```python
from exact_conv import evaluate_multi, interpolate

ev = evaluate_multi(coeffs, points, p)
ev.values            # list[int]：f(points[i]) mod p，与点顺序一致，均在 [0, p-1]
ev.modulus           # 使用的素数 p
ev.multiplications   # 实际执行的多项式乘法（NTT 卷积）次数

ip = interpolate(points, values, p)
ip.coefficients      # list[int]：唯一次数 < 点数的多项式系数（低次在前），
                     # 规范到 [0, p-1]，已去高位零
ip.modulus           # 使用的素数 p
ip.multiplications   # 实际多项式乘法次数
```

- `p` 必须是固定 NTT 素数之一（`1004535809` / `998244353` / `469762049`），
  不支持其他模数。
- 点数 `1..128`（`MAX_POINTS = 128`），次数 `<= 128`（`MAX_DEGREE = 128`，
  即 `coeffs` 最多 129 项）。
- 系数、点、函数值均为 `int`（拒绝 `bool`/`float` 等），先规范到
  `[0, p-1]` 再计算；负数与超过 p 的整数按模 p 约化。
- 求值支持零多项式、单点、点 0、**次数高于点数**的情形；求值点允许重复
  （逐点返回，不去重）。
- 插值点按模 p 约化后**必须互异**；插值只返回次数小于点数的唯一多项式。
- 本模块只做域 F_p 上的运算，不把整数 CRT 重建结果直接当作域元素。

### 算法

1. **乘积树**：对点集自底向上构建 `M(x) = prod (x - x_i)` 的二叉乘积树，
   子问题在各处复用；多项式乘法走既有迭代蝶形 NTT（`exact_conv/ntt.py`）。
2. **余式树求值**：被求值多项式先对树根取余，再沿树自顶向下对各子节点
   取余（首一除式综合长除法，精确模算术），叶子常数即各点函数值。
3. **插值**：对树根求导得 `M'(x)`，复用同一余式树在各点求值得拉格朗日
   权重 `c_i = y_i / M'(x_i)`，再沿乘积树自底向上合并
   `P = P_left * M_right + P_right * M_left`。
4. 两个结果对象都暴露 `multiplications`，统计实际 NTT 多项式乘法次数；
   旧整数卷积路径（`convolve` / `plan_moduli`）完全不变。

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
- `UnsupportedModulusError`：多点求值 / 插值请求的素数不在固定 NTT 素数
  集合内（或 `p` 不是 `int`）。
- `DuplicatePointError`：插值点按模 p 约化后出现重复（含 `x` 与 `x+p`
  这类域中相同的情形），唯一插值多项式不存在，直接拒绝。

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

`python -m unittest discover -s tests -v` 共 50+ 个用例，含：

- 手算小例、确定性随机多组规模（1×1 到 256×256）；
- **全负**（负×负、负×正）、**全零/零散点**、**长尾零**（256 长但尾部
  大片零，卷积长 511）；
- **单模 / 双模切换**、模数选择严格阈值（`2B` 恰在模两侧）、三模档与
  “无法证明精确”路径；
- 接口上限 256 项 × ±10^6 对拍朴素参考；
- `bool` / `NaN` / `Infinity` / 有限 `float` / 字符串 / `None` 拒绝；
  超长、空、超系数界拒绝；错误信息可定位；输入不被修改。
- 多点求值：手算小例、零多项式、单点、点 0、次数高于点数、三素数随机
  对拍**独立 Horner 逐点参考**、128 点上限、乘法次数确定性；
- 插值：手算小例、单点、点 0、三素数随机对拍**独立 Vandermonde 高斯消元
  参考**（`tests/reference.py`，不导入被测核心）、系数规范到 [0, p-1]、
  重复点（含模 p 意义重复）拒绝、点数/值数不等拒绝、往返求值仅作补充。

## 误差口径

本库没有浮点结果：输入是整数，中间为模算术，输出为 Python 精确大整数。
**误差恒为 0，不使用任何容差**。所谓“对称区间恢复”是整数集合上的精确
双射（在 `M > 2B` 前提下），不是近似。

## 已知限制 / 非目标

- 只实现本题契约的数学子集：三个固定素数、长度 ≤ 256、系数绝对值 ≤ 10^6；
  不宣称通用 NTT/CRT 行业标准兼容，不支持任意模数或任意原根配置。
- 多点求值 / 插值进一步限于点数 ≤ 128、次数 ≤ 128，且只在三个固定 NTT
  素数上工作；不提供任意素域或更大规模。
- 纯 Python 实现，追求正确与可证性，而非 SIMD/原生级吞吐；接口规模下
  256×256 双模卷积为毫秒量级。
- 不提供网站、CLI 产品、报表等；终端命令仅用于跑测试与演示。
- 不保证输入序列惰性生成器可被重复消费（内部只迭代一次）。
