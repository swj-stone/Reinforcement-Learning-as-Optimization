# 两个基线设计（Random 与 Greedy）

> 本文件是「方法」阶段四个子项目中的第二个（B）。前一个：A 评测服务
> （`2026-09-29-evaluation-service-design.md`）。后续：C RL、D 实验与报告。
> 依据 `EXPERIMENT_PLAN.md` §5（三种方法的确切协议）、§8（E1–E4b）、
> §9 第 4 天（验收条件）、§10（产物）。
>
> **引用约定：** 本文内部的节引用写作 `§N`；对 `EXPERIMENT_PLAN.md` 的引用
> 一律写作 `计划 §N`。

## 1. 目标与范围

交付计划 §5 定义的两个基线方法及其产物。

**在范围内：**

- 单实例构造：Greedy 与 Random（每次试验一个解）
- 每次试验的种子派生（计划 §5 的 `6001` / 实例 ID / 试验号）
- 批量入口：在 **3,300** 个实例上运行并产出 `baseline_results.csv`（仓库根目录，入库）
- 全语料验收：两法的可行性与相对质量

**不在范围内：**

- 派生指标（`exact_gap`、`normalized_quality`、`relative_improvement`）、跨实例聚合、
  表格、图、1000 次配对 bootstrap —— 属 D
- DQN、环境、回放、训练 —— 属 C
- E5 混合消融 —— 计划 §8 把它列为一次独立的训练研究

**为什么是这 3,300 个实例。** `test`(2,400) 是 E1 的主比较集，E2 是它的分解（不产生
新数据），E4b 是它的子集；`stress/e3_size`(600) 与 `stress/e4_tail`(300) 是 E3 与 E4。
也就是说这 3,300 个覆盖了 E1–E4b 的全部；`train` / `validation` 只服务于 C 的训练与
检查点选择，基线不在其上报告。

实测确认：3,300 个 `instance_id` **全局唯一**（0 重复），且**全部**都能在已入库的
`reference_results.csv` 里找到（0 缺失）——后者是 §10 的验收赖以成立的前提。

行数：3,300 × (10 次随机试验 + 1 次贪心) = **36,300 行**。

**为什么两个基线共用一个产物文件。** 它们共用同一套列、同一套计时口径、同一批实例，
而 D 的每一张表都要按 `(instance_id, method)` 分组。分成两个文件只会让 D 多做一次
join，没有换来任何东西。DQN 在 C 里产出，由 D 把三方合并成计划 §10 的 `metrics.csv`。

## 2. 两个方法的协议

计划 §5 的那两行是**规格**，不是描述。下面把每句话落成可执行的动作，并标出原文里
被两种读法都能解释的地方。

### 2.1 Greedy

原文：*Sort by descending `v_i/w_i`; break ties by lower `w_i`, then stable item ID.
Include every item that fits and continue after a rejected item.*

1. 用 `reference.ratio_order(weights, values)` 取得物品 ID 的次序；
2. 沿该次序走，`weights[i] <= remaining` 就装入并扣减容量；
3. 装不下就**跳过并继续**（`continue`，不是 `break`）。

第 3 点是原文专门写出来的一句，因为 `break` 是更"自然"的写法，而且它在**多数**实例
上给出同样的答案——只有在一个轻物品被重物品挡在后面时才暴露。它必须由合成实例钉住
（§11）。

`ratio_order` 正是计划 §5 的次序：**整数叉乘**排 `v_i/w_i` 降序，同分取更小的 `w`，
再取原索引升序。它不是"另一个排序"——见 §6。

Greedy **不使用** `item_order`：它自己排序。实例里那份存储排列只对 Random 有意义。

### 2.2 Random

原文：*Hold the stored item permutation fixed. If an item fits, include it with
probability 0.5; otherwise skip. Run 10 trials with independent inclusion coins; derive
each coin-flip seed deterministically from evaluation seed `6001`, instance ID, and trial
index. Average outcomes rather than selecting the best trial.*

**"stored item permutation" 就是实例记录里的 `item_order` 字段。**
`generate/corpus.py` 与 `assemble/{real,stress}.py` 都由一条**独立的 order 随机流**
生成它（Plan 1 的 `STREAMS = ("items", "order")`），实测确认三来源的记录都带这个字段。
它在 10 次试验之间**固定不变**——原文 "hold … fixed" 正是这个意思。所以一次试验是
沿 `item_order` 走，而不是沿索引 `0..n-1`。

> 若沿索引顺序走，产出的仍然是一个确定的、可复现的 CSV——只是它不是计划描述的那个
> 实验。这类偏差不报错、不崩溃，只留下一份与计划不符的产物，所以它必须由测试钉住：
> 打乱 `item_order` 必须改变结果（§11）。

**硬币只铸给装得下的物品**（本设计的读法，记为读法 (a)）：

```
remaining = capacity
for i in item_order:
    if weights[i] <= remaining:      # 装得下才铸币
        if rng.random() < 0.5:       # 以 0.5 的概率装入
            装入 i；remaining -= weights[i]
    # 装不下：直接跳过，不铸币
```

原文那句有两种语法上都成立的读法：

| | 铸币时机 | 效果 |
| --- | --- | --- |
| **(a)** | 只有装得下的物品铸币 | "otherwise skip" 是无条件的替代分支 |
| (b) | 每个物品都铸币，再取 `装得下 and 正面` | 会为一件**不可能被选中**的物品铸币 |

两者**结果不同**，因为硬币流的推进量不同；而两者都是确定的、都能逐字节复现——所以
"随机种子可复现"这条验收**区分不了它们**。选 (a) 的理由：(b) 要求为一件在协议里
不可能入选的物品下判断，那枚硬币没有任何解释；且原文的分号把"装不下"写成无条件的
`otherwise skip`。

**10 次试验取平均，不取最好。** 原文 *Average outcomes rather than selecting the best
trial* 是一句**禁令**：best-of-ten 是个诱人的实现（它会让 Random 的数字好看得多），
而且它同样确定、同样可复现。产物里 10 行各记自己的观测值，**平均由 D 做**——B 不产出
"平均行"（§8）。

**`random.Random` 的用法。** 一次试验一个种子 → `random.Random(seed)` → 沿 `item_order`
顺序取币。

- 种子传 **int**，不传 `str`/`bytes`：`random.seed` 对 `str`/`bytes` 的哈希经历过
  `version=` 语义变更，而 int 种子是直接使用的；
- 用 `random()` 而不是 `getrandbits()`：标准库文档保证 `random()` 在给定兼容种子时
  **跨版本产出同一序列**，这是本方法可复现性的依据；
- 用**实例化**的 `Random(seed)`，不用模块级的 `random.seed()` / `random.random()`：
  后者是全局状态，会把"这一次试验"绑定到调用顺序上，并让并行的测试互相污染。

## 3. 模块结构

```
experiment/baselines/__init__.py
experiment/baselines/seeds.py      # trial_seed：种子派生
experiment/baselines/methods.py    # greedy_solution / random_solution：纯构造
experiment/baselines/cli.py        # 读配置 → 跑语料 → 写 baseline_results.csv
```

沿用 `experiment/{data/*,eval}` 的既有约定：一个可执行入口 `cli.py`，其余是纯函数模块。

`methods.py` 与 `seeds.py` **不读任何文件**：配置里的两个参数只在 `cli.main()` 读一次
（§8），作为实参传下去。这样两个方法在测试里可以被直接调用，不必造配置文件、也不必
准备临时目录。

```python
class BaselineSolution(NamedTuple):
    selected_indices: tuple[int, ...]   # 按物品 ID 升序
    value: int
    weight_used: int

def greedy_solution(instance: Mapping) -> BaselineSolution: ...
def random_solution(instance: Mapping, seed: int) -> BaselineSolution: ...
```

**不复用 A 的 `ReferenceSolution`**：它的字段里有 `upper_bound` 与 `certified`，基线
产不出这两个量，硬塞进去只能填假值——而一个填了假值的字段迟早会被当成真值读。

`selected_indices` 按 ID 升序返回，与 A 的 `solve` 一致。这个约定**不**影响
`item_set_hash`（它内部先排序），它影响的是 B 自己的契约。

## 4. 种子派生

```python
def trial_seed(instance_id: str, trial: int, base: int) -> int: ...
```

沿用 `generate/seeds.py` 的惯例：BLAKE2b、`digest_size=8`、`int.from_bytes(..., "big")`，
键为 `"|".join((str(base), instance_id, str(trial)))`。

- **内容寻址而非顺序编号**：种子是键的函数，不依赖任何全局计数器，也不依赖语料的
  文件顺序。同一次试验在两次运行里、在两台机器上、在并行测试中都是同一个种子。
- 8 字节已经是 64 位，故不再掩码（`seeds.py` 里那句 `& _MASK64` 在 `digest_size=8`
  下是无操作）。
- 键用 `|` 拼接，要求各分量不含 `|`；语料的 `instance_id` 不含，由 §10 的验收顺带断言。
- **三个分量缺一不可，且各自可区分**：漏掉 `instance_id` 会让所有实例共用同一串硬币；
  漏掉 `trial` 会让 10 次试验变成同一个解的 10 份副本。两条都要有测试（§11）。

## 5. 计时口径

计划 §5 末段：*Timing includes sorting, feature construction, action selection, and
feasibility checks, but excludes loading the shared corpus from disk. Report one-trial
random latency, not the duration of all ten trials as a single solution.*

- 计时区间 = **一次构造调用**：从 `greedy_solution` / `random_solution` 进入，到返回
  `BaselineSolution` 为止。包含排序、动作选择与可行性检查；**不包含**读取实例
  （`read_instances` 在计时之外），也不包含行构造与写盘。
- Random 的 10 行**各记自己那一次**的耗时。不存在"十次之和"这个数——计划 §5 专门
  禁止了它。
- 与 A 的 `solver_time_s` 同类：**墙钟测量，逐字节不可复现**。所以
  `baseline_results.csv` 与 `reference_results.csv` 一样，**价值列稳定、时间列不稳定**。
  §8 由这条推出一条对测试的硬约束。

格式：`f"{seconds:.6f}"`，与 A 的 `solver_time_s` 一致。

## 6. 对子项目 A 的修订：`ratio_order` 提升为公开

A 的 `reference.py:43` 有一个私有函数 `_ratio_order`，其 docstring 与实现正是计划 §5 的
Greedy 次序：整数叉乘排 `v_i/w_i` 降序，同分取更小的 `w`，再取原索引升序。也就是说
"A 算上界要的次序"与"Greedy 要的次序"是**同一个函数**——这不是巧合，是同一件数学事实：
分数背包的贪心与经典 0/1 贪心按同一个比值序扫描。

**决定：`_ratio_order` 改名为 `ratio_order` 并公开，B 直接 import。** 不复制一份：
两份实现会在某次修订后分歧，而分歧的症状是**上界与贪心用了不同的次序**——一个既不
报错也不崩溃、只在数字上体现的偏差。

影响面已实测。**代码与文档要分开看：前者必改，后者只能记录。**

**代码，共 4 处，且是全部 `.py` 文件里的全部出现：**

| 位置 | 改动 |
| --- | --- |
| `experiment/eval/reference.py:43` | `_ratio_order` → `ratio_order`（改 def 与 docstring） |
| `experiment/eval/reference.py:75` | **唯一**的生产调用点，同步改名 |
| `tests/test_eval_upper_bound.py:8` | 导入改名 |
| `tests/test_eval_upper_bound.py:71-80` | 一个测试函数内的 5 处引用改名（含测试名） |

**文档，A 的实现计划里 6 处，一处都不改：** `2026-09-29-evaluation-service.md` 的第
67 行（正文提到该函数）、128 行（测试 fence 的导入）、191–200 行（测试 fence 的名字与
4 条断言）、229 行（一条 `Expected:` 里写着 `ImportError: cannot import name
'_ratio_order'`）、262 行（`def _ratio_order` 的定义 fence）、294 行（调用点 fence）。

**只改名、不改行为。** 返回次序的契约一字不变，A 的全部测试（含该测试函数的 4 条
断言）必须仍然全绿——这是这次改动的验收条件。

> **这是本设计里唯一一处动已合并、已终审、且含冻结文件的代码。** 仍然要做的理由：
> 另一条路是 B 抄一份，而两份排序的长期成本远高于一次改名；改动面小（4 处，可逐行
> 核对），且改的是名字不是行为。故按 A 修订 计划 §7 的同一惯例——**单列一节记录，
> 不偷偷改**。
>
> **两处已知的漂移，都记在这里，不回填进 A 的文档：**
>
> 1. **公开面。** A 的计划写着公开面「恰好是 `solve` / `format_upper_bound` /
>    `fractional_upper_bound` / `item_set_hash` 四个」，新增后那句话不再完整。
> 2. **fence 本身。** 上表那 6 处文档引用里包括 `def _ratio_order` 的定义 fence 与整条
>    测试 fence——`_ratio_order` 这个名字在那些 fence 里已经不存在于代码中。
>
> 选择不回填的理由：那是**已执行完**的历史文档，回填会让它与当时的提交对不上。
> 另一种做法是按 A 修订 计划 §7 的同一惯例，在 A 的计划顶部加一段**修订说明**
> （加而不删原文）；此处先不做，因为它会再动一份已归档的文档——本文把漂移记全，
> 使那段说明随时可以补上。

## 7. 复用 A 的原语

| 复用 | 来自 | 理由 |
| --- | --- | --- |
| `ratio_order` | `reference` | §6 |
| `item_set_hash` | `reference` | 产物的 `item_set_hash` 列必须与 A 的**同一个函数**，否则 D 无法比较"两个方法是否落到同一集合" |
| `feasibility_violations` | `metrics` | 产物的 `violations` 列。两法在构造上恒可行、故恒为 0——写进产物不是装饰：它是一个**独立于构造过程**的复核，能抓到"扣容量写错但值算对"这类自洽缺陷 |

## 8. 产物：`baseline_results.csv`

仓库根目录，**入库**（与 `reference_results.csv` / `data_manifest.json` /
`experiment_config.yaml` 一致）。

列（列序即产物的一部分）：

| 列 | 说明 |
| --- | --- |
| `instance_id` | 与语料一致；D 用它 join 参考解与实例属性 |
| `split` | `test` / `e3_size` / `e4_tail`，取自目录层——**实例记录里没有这个字段** |
| `method` | `greedy` / `random`，与配置里 `evaluation.methods` 的取值一致，便于 D 与 DQN 行合并 |
| `trial` | `0..9`；Greedy 行为空 |
| `seed` | 该次试验的派生种子；Greedy 行为空。写进产物是为了让复核者**不必重跑**就能重算任意一行 |
| `value` | 所选集合的价值（整数） |
| `weight_used` | 所选集合的重量和（整数）。计划 §10 的 "selected value/resource" |
| `violations` | `feasibility_violations` 的返回值，恒为 0 |
| `item_set_hash` | 所选集合的规范哈希（A 的口径） |
| `latency_s` | 一次构造的耗时，6 位小数（§5） |

**为什么不存完整 `selected_indices`。** 36,300 行里最长的一行对应 500 个索引，原始
列表会让这个文件涨到几十 MB；而 D 需要的只是"两个方法是否选了同一集合"这类**相等性**
判断，摘要恰好够用。

**`item_set_hash` 不是单射。** A 的实测给出 17,700 行里只有 17,095 个不同哈希（183 个
碰撞组、605 行超出）。所以"哈希不等 ⇒ 集合不同"成立，"哈希相等 ⇒ 集合相同"**不**成立。
这一列在 D 手里是**筛子而不是判据**；要断言同一性得回去重跑。写在文档里，是因为它是
个很容易被当成判据用的列。

**行序**：按 `(instance_id, method, trial)` 升序。`instance_id` 在语料里全局唯一
（实测 3,300/3,300），故这个键是全序：同一实例下 `method` 先分开 `greedy` 与 `random`，
`random` 内部再由 `0..9` 分开，而 `trial` 为空的 Greedy 行不会与任何行并列。

**参数只在入口读一次。** `cli.main()` 读 `experiment_config.yaml` 的
`evaluation.evaluation_seed`（=6001）与 `evaluation.random_trials_per_instance`（=10），
作为实参传给 `run()`。这正是 `build_config` 的 docstring 写明的意图——「让后续阶段读
YAML 而不是回头读 Markdown」：计划若再修订，只需重新生成 YAML，B 不会与它背道而驰。
（这两项目前在 Python 侧**没有**可导入的常量，只存在于转录进 YAML 的 `_EVALUATION` 里，
所以读 YAML 也是唯一不需要再抄一遍常量的选择。）

**覆写守卫。** `--out` 默认指向已入库的 `baseline_results.csv`，而 `latency_s` 是墙钟——
裸跑会把一个已入库文件改写成不同字节、弄脏工作区，而这件事在 A 上**真的发生过**。
因此 B 的入口在目标已存在且未给 `--force` 时报错退出、**不改动文件**。

> 这是 B 与 A 的**有意不对称**：A 的 `cli.py` 没有这道守卫，对应的一条已记在 A 的终审
> 台账里、尚未应用。本节只处理 B，不去动已合并的 A。两处不一致是**已知且记在案**的，
> 不是疏漏。

## 9. 对 计划 §10 的说明：`baseline_results.csv` 是新增产物

计划 §10 的产物表里**没有**基线产物，只有一张 `metrics.csv`，其内容列为
「Instance ID, method, seed/trial, selected value/resource, violations, normalized
quality, and latency」。本设计把它拆成两件：**B 产出 `baseline_results.csv`**（原始
观测），**D 产出 `metrics.csv`**（三方合并 + 派生指标）。

两处差额要记明：

1. **文件名不同。** 计划 §10 的开头写的是 "the following files **or equivalent clearly
   named artifacts**"，故 `baseline_results.csv` 是一个等价的、名字更贴切的产物。
2. **`normalized_quality` 不在 B 的列里。** 它要 join 参考解的 `U`，而
   `relative_improvement` 需要三个方法都在手——按定义属 D。B 只记原始观测，使每一列
   都能由"这一次构造"直接产生，**没有跨文件依赖**。

> 这条差额来自用户的明确选择（在"独立产物 `baseline_results.csv`"与"并算好派生指标"
> 之间选了前者）。此处记下它相对计划 §10 缺了什么，免得日后读计划的人以为漏了东西。

**一处留给 D 的取舍（本文只记录，不决定）：** 计划 §8 的 E2 要求把 E1 按四个 family、
四个 size、三个容量比分解，而这些字段在实例记录里、不在本产物里。要只凭已入库的产物
生成 E2 的表，就得把 `source` / `family` / `n` / `alpha` 也列进来；否则 D 生成表时仍需
读 `datasets/`（它被 gitignore，换台机器就不在位）。这与 A 的 `reference_results.csv`
的处境相同，故保持列集一致、把取舍留给 D。

## 10. 全语料验收

3,300 个实例、36,300 行上的验收，作为 `tests/` 里的一个测试，用 `skipif` 守门：

```python
@pytest.mark.skipif(
    not (DATASETS_PROCESSED / "test" / "instances.jsonl").is_file(),
    reason="冻结语料不在位（datasets/ 被 gitignore）；在本仓库里会真跑",
)
```

守门的取舍：本项目的完成标准是**真仓库里 0 skipped**，而 `datasets/` 在这个仓库里
是真在的，所以这条测试在本机**会真跑**；在没有语料的临时副本里它跳过而不是失败。

它读 `datasets/processed`（实例）与 `reference_results.csv`（`V*`），先调 `cli.run()`
写到 `tmp_path`，再把 CSV **读回来**断言——顺带把序列化路径也验了。

断言：

| 断言 | 说明 |
| --- | --- |
| 行数 = 3,300 × 11 = 36,300 | 每实例 10 次随机 + 1 次贪心，一个不少 |
| 每个 `instance_id` 都能在 `reference_results.csv` 里找到 | B 的实例集与 A 的参考解集一致 |
| 每一行的 `value <= V*` | 两个**独立计算**的产物之间的一致性（见下） |
| `violations` 全为 0 | 可行性在**真实分布**上成立，而不只是在合成夹具上 |
| 无 `instance_id` 含 `\|` | §4 的键拼接无歧义 |
| Greedy 的平均 `V/U` **严格高于** Random 的平均 `V/U` | 见下 |

> **第三行挡的是什么，要说清。** 任何**可行**选择的 value 都不可能超过 `V*`——这是
> `V*` 的定义。所以这一行不是"贪心写对了"的证明，它挡的是**跨子系统的不一致**：两边
> 实例集错位、参考解列读错、`value` 从错误的数组求和。排序方向写反、`break` 代替
> `continue` 这类缺陷**能通过这一行**（它们给出的解仍然可行），它们由 §11 的合成实例
> 钉住。
>
> **第六行补的正是这个缺口。** 排序反了或提前 `break` 的贪心仍然可行，只是**更差**——
> 在 2,400 个实例的平均上会掉向 Random 的 0.5 掷币。语料是冻结的、方法是确定的，所以
> 这两个均值是**确定的数**，这条断言可复现。文档里记下实测的两个均值作为证据，但
> **只断言严格不等式、不写死阈值**：阈值会在语料修订后变成一条假失败，而不等式不会。

## 11. 测试与验收

| 测试文件 | 内容 | 它挡住的缺陷 |
| --- | --- | --- |
| `test_baselines_greedy.py` | **经典反例** `w=[10,20,30]`、`v=[60,100,120]`、`C=50`：比值贪心取前两件得 **160**，而 `V*=220` | 把贪心当成精确解、次序走错 |
| | **跳过继续** `w=[10,20,3]`、`v=[60,100,3]`、`C=15`：带 `continue` 得 **63**，带 `break` 得 **60** | `break` 代替 `continue` |
| | **同分裁决（`w` 不同）** `w=[1,2]`、`v=[2,4]`、`C=2`：先取 `w` 小的那件得 **2**，反序则得 **4** | 只按比值排序、丢掉次级键 |
| | **同分裁决（`w` 也相同）**：断言**索引集合**而非值（见下） | 依赖 `sort` 的稳定性 |
| `test_baselines_random.py` | **确定性**：同 `(instance, seed)` 两次调用逐位相同 | 用了全局 `random` 状态 |
| | **传播**：同一实例的不同 `trial` 至少有一对结果不同 | `trial` 没进种子 |
| | **平均非取最好**：10 次试验的 10 个 value 不全都相等，且各自等于自己那组 `selected_indices` 的求和 | best-of-ten（10 行同值）、或把均值写进每一行（不是任何子集的和） |
| | **(a) 读法**：在测试内部独立实现 (b)，搜出使 (a)、(b) 结果不同的实例与种子，断言实现与 (a) 一致 | 每件物品都铸币——结果不同，但同样确定、同样可复现 |
| | **遵守 `item_order`**：同实例同种子下打乱 `item_order` 必须改变结果 | 忽略存储排列、改按索引顺序走 |
| `test_baselines_seeds.py` | 同输入同输出；`instance_id` / `trial` / `base` 任一不同则种子不同 | 键里漏掉某个分量 |
| `test_baselines_cli.py` | 列序、行序、LF、无 BOM、行数 = 合成夹具实例数 × 11；Greedy 行的 `trial`/`seed` 为空 | 序列化漂移、行数算错 |
| | 改临时配置里的 `random_trials_per_instance` 后行数随之变 | 把 10 硬编码 |
| | 目标已存在且无 `--force` ⇒ 报错，且**文件未被改动** | 裸跑覆写已入库产物（§8） |
| `test_baselines_corpus.py` | §10 的全部断言 | 跨子系统不一致；贪心质量塌到随机水平 |

**"同分且同 `w`"为什么只能断言集合。** 两件物品 `v/w` 相等且 `w` 相等 ⇒ `v` 也相等，
它们**完全可互换**，所以无论取哪一件，`value` 与 `weight_used` 都一模一样。这个次序
差异只在 `selected_indices` / `item_set_hash` 上可见——而 `item_set_hash` 正是入库产物
的一列。于是这条测试必须断言集合，否则它会是一条**声称在测次序、实际测不了**的断言：
本例中把次级键改成索引降序，值断言仍然全绿。

**"平均非取最好"为什么要先证夹具能判别。** 若夹具恰好让 10 次试验给出同一个值
（例如所有硬币都是正面），那么"10 个值不全都相等"这句本身就成了空转，后面的断言也
跟着失效。所以顺序是：**先断言这 10 个值不全都相等**（证明夹具能区分 best-of-ten），
再依赖其余断言。这就是 A 的 §4 记下的那条教训——*先证明检查有判别力，再相信它的沉默*。

**"(a) 读法"为什么要搜而不是手算。** 硬币来自 BLAKE2b 派生的种子，手算不出来。测试
自己实现一份 (b)、在小范围内搜出两者分歧的实例与种子，然后断言实现站在 (a) 一边。
**并且断言"确实搜到了分歧"**——否则一次搜空会让这条测试在什么都不验的情况下通过。

**关于排序本身。** `ratio_order` 的次序契约已由 A 的
`tests/test_eval_upper_bound.py::test_ratio_order_breaks_ties_deterministically` 用
4 条断言钉住，B **不重复钉它**，只钉**Greedy 的选择确实沿这个次序发生**。

> **但 A 的那条不能替代 B 的这两条。** 一个性质在一个纯函数里被钉住，不等于在它的
> 调用点上被钉住：A 的断言喂给 `ratio_order` 的是它自己构造的小数组，而 Greedy 的调用
> 点喂的是**语料里的实例**——两者是不同的参数总体。所以 Greedy 那两条测试用的实例
> 必须让"次序不同 ⇒ 结果不同"，否则它们和 A 的那条一样，挡不住调用点上的缺陷。

**计时口径**（§5）也要有测试，因为它挡的是一个很容易漏的写法：

| 测试 | 内容 | 它挡住的缺陷 |
| --- | --- | --- |
| 计时区间 | 在大合成实例上测"调 10 次"的总耗时 `t10`，断言单行的 `latency_s < t10 / 2` | 把 10 次试验放进一个计时区间，再把总和写进每一行 |

十次之和会落在 `t10` 量级上，而单次远小于它；取 5 倍余量以避开计时噪声。断言写成
"`latency_s > 0`"是空转的——那样连"记的是十次之和"都挡不住。

## 12. 刻意不做

- **不复制 `ratio_order`** —— 提升为公开后共用（§6）。
- **不产出"10 次试验的平均行"** —— 平均属 D；B 产出 10 行原始观测（§2.2）。
- **不算 `exact_gap` / `normalized_quality` / `relative_improvement`** —— 属 D（§9）。
- **不在产物里存完整 `selected_indices`** —— 用 `item_set_hash`（§8）。
- **不跑 `train` / `validation`** —— 它们服务 C 的训练与检查点选择。
- **不做 E5** —— 计划 §8 把它列为一次独立的训练研究。
- **不引入 OR-Tools** —— 与 A 一致；基线不需要求解器。
- **不给 `baseline_results.csv` 做"重跑是否逐字节一致"的比对** —— `latency_s` 是墙钟，
  必然不同（§5）。要复核就比对除 `latency_s` 以外的列。
- **不修改 `experiment/data/generate/*` 与 `experiment/data/collect/*`** —— 冻结目录，只读复用。
- **不修 A 的 `cli.py` 覆写守卫** —— B 只给自己的入口加（§8）；A 的那条在终审台账里。
- **不在 B 里做跨实例聚合** —— 包括"两法平均 `V/U` 的对比"：§10 里那条是**验收断言**，
  不是产物；D 的聚合成品另行产出。
