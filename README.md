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

## 接口

```python
from exact_conv import convolve, blocked_convolve, plan_moduli

result = convolve(a, b)
result.coefficients     # list[int]：C[k] = sum_{i+j=k} a[i]*b[j]，精确整数
result.moduli           # tuple[int,...]：实际选用的 NTT 模数（最少个数）
result.coefficient_bound    # B = min(n,m) * max|A| * max|B|
result.modulus_product      # 所选模数乘积 M，保证 M > 2*B
result.transform_length     # 补零后的 NTT 长度（2 的幂）

# 固定块长入口：长序列按 S 切块，每对块调用 convolve 后按 (i+j)*S 重叠累加
result = blocked_convolve(a, b, block_size=128)
result.coefficients         # 同样是长度 n+m-1 的精确整数列表
result.block_size           # 实际使用的固定块长
result.block_calls          # 每个块对的 BlockCallRecord（块号、偏移、模数、NTT 长度）
result.working_set          # WorkingSetStats：工作数组大小统计
```

- `a`、`b`：整数系数的可迭代对象（list/tuple/生成器均可）。
  - `convolve`：长度 1..256，系数绝对值 `<= 10**6`；
  - `blocked_convolve`：长度 **无 256 上限**（可长达数千项），系数绝对值
    仍必须 `<= 10**6`（块内沿用旧的可证精确界）。
- 返回长度 `len(a) + len(b) - 1` 的精确整数列表；输入不会被修改。

`plan_moduli(n, m, max_abs_a, max_abs_b)` 是纯函数的模数规划，返回
`(moduli, bound, modulus_product)`，可在不做卷积时单独查询“最少要用几个模”。

### 固定块长入口与工作数组统计

`blocked_convolve(a, b, block_size)` 把两侧序列切成定长块（末块允许更短），
对**每一对块（零块也调用、不跳过）**调用旧的精确卷积，再把块卷积按
`(i+j)*block_size` 偏移重叠累加进总长 `n+m-1` 的输出。这是整数环分配律的
直接应用，结果与整块“学校乘法”逐系数恒等（测试对每个块长都对拍朴素定义）；
负系数无需特殊处理。块长必须是 `1..256` 的精确 `int`，否则抛
`InvalidBlockSizeError`。

`WorkingSetStats` 字段（口径均为“Python int 引用槽位数”，不是字节；
不含大整数对象本身开销）：

| 字段 | 含义 |
|---|---|
| `block_size` / `blocks_a` / `blocks_b` | 块长与两侧块数（含更短末块） |
| `block_calls` | 块卷积调用总数 `ceil(n/S)*ceil(m/S)` |
| `output_slots` | 输出累加数组槽位 `n+m-1` |
| `per_call_ntt_slots` | 各块对的 NTT 工作集（每次调用峰值同时存活 3 个 `transform_length` 数组：fa、fb、逆变换结果） |
| `peak_ntt_slots` / `total_ntt_slots` | 最大单次工作集 / 各次之和（累计分配量） |
| `peak_working_slots` | 同时存活峰值 = `output_slots + peak_ntt_slots` |
| `transform_lengths` | 各块对 NTT 长度（按调用顺序） |

每个块对另有 `BlockCallRecord`：`block_a`、`block_b`、`len_a`、`len_b`、
`offset = (block_a+block_b)*block_size`、`moduli`、`transform_length`。

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
- `InvalidBlockSizeError`：固定块长入口的 `block_size` 不是 `1..256` 的
  精确 `int`（拒绝 bool、float、0、负数与超过 256 的值）。

错误信息都带参数名与下标，例如 `a[2] ...`、`b[1] ...`（分块入口的下标
为原序列全局下标，校验在切块前完成），不静默纠正输入。
`convolve` / `blocked_convolve` 都是纯函数：校验失败或成功都不改动输入；
分块入口的全部校验在输出累加器建立之前完成，失败时不产生部分结果。

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
6. **固定块长重叠累加**（`exact_conv/blocked.py`）：长序列按用户给定块长
   `S` 切块，块对结果写入偏移 `(i+j)*S`，重叠位置直接整数相加（Python 大
   整数，不模任何数），因此即使多个块对的部分积叠加到同一系数也保持精确；
   每次块卷积的 NTT 工作数组随调用生灭，峰值工作集只取决于单个块，与总
   长度无关（输出累加数组除外）。

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
- **固定块长入口**：手算含负系数例子；{-1,0,1} 上长度 1..3 全部序列 ×
  块长 {1,2,3,5} 的小规模枚举（6084 次真实卷积）对拍朴素定义；块长整数倍
  / ±1 边界；零块照常处理（块对不跳过）；块长无关性（同一输入 6 种块长
  结果相同）；300..1000 项的超 256 长输入；384×384 ±10^6 块内双模 +
  跨块重叠峰值精确；统计字段按定义独立重算；非法块长（0/负/257/bool/
  float/字符串）、长输入中 `a[300]` 超界等失败用例与无半更新状态核验。

## 误差口径

本库没有浮点结果：输入是整数，中间为模算术，输出为 Python 精确大整数。
**误差恒为 0，不使用任何容差**。所谓“对称区间恢复”是整数集合上的精确
双射（在 `M > 2B` 前提下），不是近似。

## 已知限制 / 非目标

- 只实现本题契约的数学子集：三个固定素数、单块长度 ≤ 256、系数绝对值
  ≤ 10^6；`blocked_convolve` 解除的是*总长度*上限（块内仍受上述限制），
  不改变系数界；不宣称通用 NTT/CRT 行业标准兼容，不支持任意模数或任意
  原根配置。
- 固定块长入口对每个块对都调用一次旧卷积（零块不跳过），复杂度为
  O(ceil(n/S)·ceil(m/S)·S·log S)，不做零块剪枝、Karatsuba 或跨块 FFT
  复用；本次迭代不做外存检查点、自选模数与严格总内存优化。
- 纯 Python 实现，追求正确与可证性，而非 SIMD/原生级吞吐；接口规模下
  256×256 双模卷积为毫秒量级。
- 不提供网站、CLI 产品、报表等；终端命令仅用于跑测试与演示。
- 不保证输入序列惰性生成器可被重复消费（内部只迭代一次）。
