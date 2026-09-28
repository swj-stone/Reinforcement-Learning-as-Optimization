# 设计文档：实例装配与冻结混合语料

**依据**：`EXPERIMENT_PLAN.md` 第 4 节（切分、实例构造、50:50）、第 8 节（E3/E4）、第 10 节（产物）。

**前置**：`docs/superpowers/specs/2026-09-25-raw-data-collection-and-generation-design.md` 描述的采集与生成阶段已完成，`datasets/raw/real/lots_clean.csv` 与 `datasets/raw/generation/` 均已就位。

---

## 1. 范围

把两个上游产物装配成**冻结的实验语料**：

1. **真实实例构造**——把清洗后的批次表（522,504 个 lot）按第 4 节切成候选池，抽样成 0/1 背包实例。
2. **核心混合语料**——每个 split 一个文件，内含合成与真实两种来源，逐 cell 严格 50:50。
3. **应力集**——E3（n=500）与 E4（强不平衡）。
4. **协议产物**——`data_manifest.json` 与 `experiment_config.yaml`。

### 明确不在本次范围

- **E4b 的标记动作**。第 8 节说 E4b 是从主真实测试集里"标记含极值 lot 的实例"，是主测试集的一个**子集视图**，不是独立数据集。本阶段只**算出并冻结阈值**（第 8 节要求阈值只用训练记录定），实际标记属于评估阶段。
- **E5**（可选混合消融）。第 8 节明说它是独立的训练研究且为 optional。且它不是"处理数据集"而是"重训三次"。
- **参考解、三种方法、RL 环境**。第 7 节之前的 E0–E2 运行全部属于后续阶段。
- 修改任何上游模块。`collect/`、`generate/` 的既有代码**一行不改**。

## 2. 环境与依赖

在现有 `requirements.txt` 上**新增 `PyYAML`**。理由：第 10 节要求 `experiment_config.yaml`，而手写 YAML 序列化是复现性的隐患（引号、转义、缩进、浮点表示任一处不稳，产物字节就不稳）。用 `yaml.safe_dump(..., sort_keys=True, allow_unicode=True)`。

代价：`requirements.txt` 变化意味着"换一台机器能否复现"那条验收要重跑一次。

## 3. 模块划分

新建 `experiment/data/assemble/`，与 `collect/`、`generate/` 平级。三个阶段三个包，每个包一个职责。

| 模块 | 职责 | 依赖 |
| --- | --- | --- |
| `lots.py` | 读 `lots_clean.csv`；发票→最早日期；日期切分；按日建候选池；合格日期筛选 | 无（纯数据变换） |
| `real.py` | 真实实例抽样与构造 | `lots.py`、`generate` 的纯函数、`seeds` |
| `stress.py` | E3 与 E4 实例集 | `generate` 的纯函数、`seeds`、`lots.py`（E3 真实部分） |
| `merge.py` | 读合成语料、合并两来源、50:50 断言、全局去重、落盘 | `real.py`、`stress.py` |
| `manifest.py` | `data_manifest.json` 与 `experiment_config.yaml` | 无（只消费返回值） |
| `cli.py` | 命令行入口 | `merge.py`、`manifest.py` |

**复用而不修改**：`generate.hashing.canonical_hash`、`generate.families.{capacity_for, compute_stats}`、`generate.corpus.write_cell`、`generate.seeds.derive_seed`。这些都是纯函数，且已在生成阶段被 golden 测试钉死。

## 4. `spec.py` 新增的冻结常量

```python
# --- 实例装配 ---------------------------------------------------------------

#: 真实实例每个 (n, alpha) cell 的配额（EXPERIMENT_PLAN.md 第 4 节）
REAL_QUOTA: dict[str, int] = {"train": 500, "validation": 100, "test": 100}

#: 日期切分的两个边界占比（第 4 节），均下取整
SPLIT_DATE_FRACTIONS: tuple[float, float] = (0.70, 0.85)

#: 真实实例在种子键中占用的 family 槽位记号。不在 FAMILIES 中，
#: 因此真实实例的种子与任何合成族不可能相同。
REAL_SEED_TOKEN: str = "real"

#: 应力集的流记号（第 8 节"保留 4001/5001 两条流"的落地，见第 12 节）
STRESS_SIZE_STREAM: str = "stream_size"
STRESS_TAIL_STREAM: str = "stream_tail"
#: E4 变体生成器在种子键中占用的 family 槽位记号
E4_SEED_TOKEN: str = "skewed_e4"

#: 第 8 节声明的应力流主种子，仅记录用途（见第 12 节）
STRESS_MASTER_SEEDS: dict[str, int] = {"size": 4001, "tail": 5001}

#: E3 配额（第 8 节）
E3_SIZE: int = 500
E3_SYNTHETIC_PER_FAMILY: int = 25
E3_REAL_PER_ALPHA: int = 100

#: E4 配额与变体参数（第 8 节）
E4_COUNT_PER_ALPHA: int = 100
E4_SKEW_K: int = 2
E4_SKEW_HEAVY_WEIGHT_RANGE: tuple[int, int] = (500, 2000)
E4_SKEW_HEAVY_DENSITY_RANGE: tuple[int, int] = (100, 200)

#: E4b 阈值分位数（第 8 节）
TAIL_QUANTILE: float = 0.99

# --- 输出路径 ---------------------------------------------------------------

DATASETS_PROCESSED: Path = REPO_ROOT / "datasets" / "processed"

#: 协议产物放在仓库根目录并纳入版本库。datasets/ 整体被 gitignore，
#: 若把审计凭据放在那里，换一台机器就没有任何东西能证明数据未被调换。
DATA_MANIFEST_PATH: Path = REPO_ROOT / "data_manifest.json"
EXPERIMENT_CONFIG_PATH: Path = REPO_ROOT / "experiment_config.yaml"
```

`SPLITS` 沿用既有的 `("train", "validation", "test")`——与第 4 节和 `datasets/raw/generation/` 一致。上游遗留的 `datasets/processed/val/` 空目录正名为 `validation`。

## 5. 种子派生

**完全复用 `derive_seed`，不新增派生函数。** 流之间的隔离靠**记号**，不靠调用顺序——这正是生成阶段选用内容寻址的原因。

| 实例集 | `split` 槽位 | `family` 槽位 | `n` | 落盘数据集名 |
| --- | --- | --- | --- | --- |
| 真实核心 | `train`/`validation`/`test` | `"real"` | 20/50/100/200 | 同名 split |
| E3 合成 | `"stream_size"` | 四个族之一 | 500 | `e3_size` |
| E3 真实 | `"stream_size"` | `"real"` | 500 | `e3_size` |
| E4 合成 | `"stream_tail"` | `"skewed_e4"` | 200 | `e4_tail` |

四行各自与其它三行在键上至少有一处不同，且与生成阶段的全部键（`family ∈ FAMILIES`，`n ∈ SIZES`）不同。`item_order` 一如既往取自独立的 `order` 流。

**流记号与数据集名是两个东西，不要混用。** `stream_size` / `stream_tail` 只活在种子键内部，不落盘；目录名与 `instance_id` 里用的是数据集名 `e3_size` / `e4_tail`。合并成一个会让第 8 节的目录结构被迫叫 `stream_size/`，也会让 `instance_id` 变成 `syn-stream_size-...`——把一个内部实现细节泄漏进数据契约。因此 `instance_id` 的构造**不能直接复用** `generate.corpus.instance_id`（它写死了 `syn-{split}-{family}-...` 的语义），`assemble` 侧自己拼：真实核心为 `real-{split}-n{n}-a{aa}-i{index:04d}`，合成沿用 `syn-{name}-{family}-n{n}-a{aa}-i{index:04d}`。`{aa}` 是 `alpha` 的两位整数（25/50/75），与既有文件名的 `a25` 写法一致。

**为什么不让 E3/E4 复用 `split="test"`**：E4 是 `n=200`，与主集 `n=200` 的 cell 同形，若共用键则 E4 的前若干条会与主集实例**同种子、不同内容**。虽不构成哈希冲突（内容不同），但那是把"两条流"退化成"一条流的两种用法"，违背第 8 节"保留独立流"的要求。

## 6. 真实实例构造

### 6.1 发票→日期

一张发票可能跨午夜。第 4 节规定：**发票归到它最早一笔交易的日历日**，全部 lot 随发票走。因此 `lots.py` 先算 `invoice → min(invoice_date)`，再据此给每个 lot 定日期。

### 6.2 日期切分

305 个不同日期去重排序后：

- `train` = 最早 `floor(0.70 × 305) = 213` 天
- `validation` = 其后 `floor(0.85 × 305) − 213 = 46` 天
- `test` = 其余 46 天

两个边界都**下取整**（第 4 节）。实测切分：`2010-12-01..2011-08-23` / `2011-08-24..2011-10-17` / `2011-10-18..2011-12-09`，无重叠。

第 4 节要求"记录实际行数与发票数占比，它们不必等于日期占比"——写进 manifest。

### 6.3 候选池与合格日期

每个日期一个候选池 = 该日全部 lot（保持 `lots_clean.csv` 中的行序）。**合格日期** = 该日 lot 数 ≥ n。第 4 节："合格日期不足 n 个 lot 的不可用；不得在实例内复制行来凑数。"

实测：每天最少 220 个 lot（中位数 1511），因此 **n=200 在每一天都成立**；n=500 只淘汰 4 天，且都在 train。第 4 节的降级条款（缩小 q）**用不上**。

### 6.4 抽样

对每个 (split, n, alpha) cell 的第 `index` 条实例：

```
rng   = default_rng(derive_seed(split, "real", n, alpha, index, attempt, "items"))
日期   = eligible_dates[rng.integers(len(eligible_dates))]     # 均匀取一天
行号   = rng.choice(len(pool[日期]), size=n, replace=False)     # 无放回抽 n 个 lot
w      = quantity_w[行号]        v = value_pence[行号]
C      = floor(alpha × sum(w))
order  = default_rng(derive_seed(..., "order")).permutation(n)
```

**rng 调用次序是冻结语义**：先取日期、再抽行号。`weights`/`values` 用 rng 给出的顺序，`item_order` 再叠一层排列——与合成实例的约定完全一致。

第 4 节允许"同一 split 内不同实例可以复用源记录"，但**任何源行或发票不得跨 split**——这一点由 6.2 的整票切分保证。

## 7. 应力集

### 7.1 E3（n=500）

合成 4 族 × 3 比率 × 25 = 300；真实 3 比率 × 100 = 300。真实部分**只用 test 源日期**，且必须与主集 canonical hash 不相交（第 8 节）。

### 7.2 E4（强不平衡，n=200）

300 条合成，每比率 100。第 8 节要求把 skewed 生成器改成 `k=2`、稀有权重 `U(500,2000)`、稀有密度 `U(100,200)`。

**不修改 `families.py` 的 `_skewed`**——它的抽样顺序是冻结语义、被 golden 测试钉死。做法：在 `stress.py` 内实现一个**独立的纯函数**，按同一套配方（`w, d, idx_w, 替换权重, idx_d, 替换密度, v = w×d`；两组稀有索引独立抽取、**可以重叠**；极值不裁剪）写出该变体，参数取第 4 节的 `E4_*` 常量。基础分布沿用 `SKEW_BASE_*`。

`capacity_for` / `compute_stats` / `canonical_hash` 照常复用。

### 7.3 应力集的记录形状

应力集**不新造记录模式**：E3 的合成分支复用合成实例的字段集，E4 同理，E3 的真实分支复用第 8.2 节真实实例的字段集。唯一的差别是 `instance_id` 前缀里的数据集名（`e3_size` / `e4_tail`）与 `family` 取值。

- E3 合成：`syn-e3_size-{family}-n500-a{aa}-i{index:04d}`，`source="synthetic"`，`family ∈ FAMILIES`
- E3 真实：`real-e3_size-n500-a{aa}-i{index:04d}`，`source="real"`，`family=null`，带 `source_date` / `source_row_ids` / `invoice_nos`
- E4 合成：`syn-e4_tail-skewed_e4-n200-a{aa}-i{index:04d}`，`source="synthetic"`，`family="skewed_e4"`

因为 `family` 对 E4 取 `"skewed_e4"`——一个**不在 `FAMILIES` 里**的取值——下游任何"按 `FAMILIES` 枚举族"的代码都会自动跳过 E4，不会被误当成第五个族。这也顺带给了 `family is None ⇔ source == "real"` 之外的第二条不变量：`family == "skewed_e4"` ⇔ 该实例属于 E4。

## 8. 输出格式与协议产物

### 8.1 目录

```
datasets/processed/
  train/instances.jsonl
  validation/instances.jsonl
  test/instances.jsonl
  stress/
    e3_size/instances.jsonl
    e4_tail/instances.jsonl
```

上游遗留的空目录 `datasets/processed/val/` 删除（正名为 `validation`）。

### 8.2 实例记录

合成实例现有 **13 个键**（实测 `datasets/raw/generation/train/correlated_n100_a25.jsonl` 首行）：

```
alpha, canonical_hash, capacity, family, instance_id, item_order,
n, resample_attempt, seed, source, stats, values, weights
```

真实实例在此之上多三个键（`source_date`、`source_row_ids`、`invoice_nos`），共 16 个：

```json
{
  "alpha": 0.25, "canonical_hash": "...", "capacity": 12345,
  "family": null, "instance_id": "real-train-n20-a25-i0000",
  "item_order": [ ... ], "n": 20, "resample_attempt": 0, "seed": 1234567,
  "source": "real", "source_date": "2010-12-01",
  "source_row_ids": [ ... ], "invoice_nos": [ ... ],
  "stats": { ... }, "values": [ ... ], "weights": [ ... ]
}
```

- `family` 对真实实例为 **`null`**：真实数据没有"族"。不变量：`family is None` ⇔ `source == "real"`，由测试钉住。种子键里的 `"real"` 是**内部记号**，不落盘，以免与"族"这个概念混淆。
- `source_row_ids` 是该实例所用 lot 在 `source_row_id` 上的取值（抽样顺序），`invoice_nos` 对应发票号。这两项是回连工作簿的唯一凭据。
- `instance_id`：真实为 `real-{split}-n{n}-a{aa}-i{index:04d}`，与合成的 `syn-` 前缀对照。

### 8.3 合并文件的顺序

文件内顺序是冻结语义，且要让 50:50 **肉眼可核**：

```
for n in SIZES:
  for alpha in ALPHAS:            # 12 个 (n, alpha) cell
    for family in FAMILIES:       # 4 个合成族，各 quota 条
      ...
    真实实例，quota_real 条
```

即每个 cell 内"合成在前、真实在后"。

### 8.4 序列化约定

沿用 `write_cell` 的约定：`json.dumps(..., sort_keys=True, ensure_ascii=False)`，LF 换行，末尾一个换行符。合成部分直接从 `datasets/raw/generation/` **读取**再重新序列化，而不是重新生成——保证合并文件里的合成半边与已审计的原始语料逐字节同源。

### 8.5 `data_manifest.json`

第 10 节要求它承载"源链接与校验和、清洗规则、日期、种子、计数、单位、源/组 ID、切分与哈希审计"。逐项落地。

**下文两个代码块里的 `//` 与 `#` 是给人看的注解，不是产物的内容**；`...` 表示"按上文已定的规则填满"。真正的字段集以本节文字为准，子代理不要照抄注解。

```json
{
  "schema_version": 1,
  "generated_at": "<ISO-8601 本地时间>",
  "provenance": {
    "workbook":  {"url": ..., "sha256": ..., "bytes": ...},
    "lots_clean": {"path": ..., "sha256": ..., "bytes": ..., "rows": 522504},
    "generation_manifest": {"path": ..., "sha256": ..., "generator_fingerprint": ...},
    "cleaning_rules": [...],          // 依次转录 spec.CLEANING_RULES 的规则名
    "units": {"weight": "quantity_w", "value": "value_pence"}
  },
  "split_policy": {
    "basis": "invoice earliest transaction calendar date",
    "fractions": [0.70, 0.85],
    "boundaries": {"train_end_exclusive": 213, "validation_end_exclusive": 259},
    "eligible_dates": {                    // 该 split 中 lot 数 >= n 的日期个数
      "train":      {"20": 213, "50": 213, "100": 213, "200": 213, "500": 209},
      "validation": {"20": 46,  "50": 46,  "100": 46,  "200": 46,  "500": 46},
      "test":       {"20": 46,  "50": 46,  "100": 46,  "200": 46,  "500": 46}
    }
  },
  "splits": {
    "train": {"dates": 213, "date_from": ..., "date_to": ...,
              "lots": 299597, "invoices": 12068, "lots_share": ..., "invoices_share": ...},
    "validation": { ... }, "test": { ... }
  },
  "streams": {
    "core":  {"seed_key": {"split": "<split>", "family": "real"}},
    "size":  {"seed_key": {"split": "stream_size", "family": "<family|real>"},
              "declared_master_seed": 4001},
    "tail":  {"seed_key": {"split": "stream_tail", "family": "skewed_e4"},
              "declared_master_seed": 5001}
  },
  "cells":  [ {"split": ..., "n": ..., "alpha": ..., "synthetic": ..., "real": ...} ],
  "counts": { "train": {"synthetic": 6000, "real": 6000, "total": 12000}, ... },
  "stress": { "e3_size": {"synthetic": 300, "real": 300},
              "e4_tail": {"synthetic": 300, "real": 0} },
  "tail_thresholds": { "quantity_w": <p99>, "unit_price": <p99>,
                       "basis": "train-split lots only", "quantile": 0.99 },
  "dedup": { "total_instances": ..., "distinct_canonical_hashes": ...,
             "collisions_resampled": ..., "max_attempt_used": ... },
  "files": [ {"path": ..., "sha256": ..., "bytes": ..., "lines": ...} ]
}
```

两处要说明的读法：

- `split_policy.boundaries.validation = 259` 是 `floor(0.85 × 305)` 这个**上边界**，train 的 213 是下边界。写成两个独立数字而不是再存一份 `0.85`，是为了让校验者不必重做浮点下取整。
- `splits.*.lots_share` 是**实测占比**，第 4 节明说它不必等于日期占比，所以要存下来供人核对而不是断言相等。

### 8.6 `experiment_config.yaml`

第 10 节要求它记录"生成器公式、RL 设置、评估规则、预算、固定的包版本"。前三项是**转录**，不是重算：从 `EXPERIMENT_PLAN.md` 的第 2/6/7 节抄成结构化形式并存盘，好让后续阶段读 YAML 而不是读 Markdown。

```yaml
schema_version: 1
generators:            # 第 2 节四个族的分布公式与参数
  uniform: {weight: U(1,100), value: U(1,100)}
  normal:  {...}
  correlated: {...}
  skewed:  {k: ..., heavy_weight: U(500,2000), heavy_density: U(1,20), base: ...}
  skewed_e4: {k: 2, heavy_weight: U(500,2000), heavy_density: U(100,200)}
cap_ratios: [0.25, 0.50, 0.75]
sizes: [20, 50, 100, 200]
real: {quota: {train: 500, validation: 100, test: 100}, split_fractions: [0.70, 0.85]}
evaluation:            # 第 7 节的评估口径
  methods: [random, greedy, dqn]
  reference: exact
  ...
budgets: {...}         # 第 6 节的训练预算
packages:              # 从 requirements.txt 转录的固定版本
  python: "3.12.x"
  numpy: "..."
  pandas: "..."
  openpyxl: "..."
  PyYAML: "..."
  pytest: "..."
```

`packages` 从 `requirements.txt` 逐行解析，不硬编码——硬编码就会在依赖升级后悄悄说谎，那正是本阶段反复出现的"常量对常量"病灶（第 12 节）。

用 `yaml.safe_dump(doc, sort_keys=True, allow_unicode=True, default_flow_style=False)`。`sort_keys=True` 保证字典键序稳定，`allow_unicode=True` 避免中文被转义成 `\uXXXX` 而让文件不可读。

## 9. 50:50 的机器可核形式

第 4 节："每个 size/capacity cell 在按来源汇总后是 50:50。"

| split | 每 cell 合成 | 每 cell 真实 | 断言 |
| --- | --- | --- | --- |
| train | 4 族 × 125 = 500 | 500 | 相等 |
| validation | 4 × 25 = 100 | 100 | 相等 |
| test | 4 × 25 = 100 | 100 | 相等 |

装配时对每个 (split, n, alpha) cell 断言两侧计数相等，不等即报错而非静默产出。

## 10. 验证策略

| 层次 | 检查 |
| --- | --- |
| 纯函数单测 | 日期切分边界（`floor` 两处）；发票整票随最早日期；合格日期筛选；无放回抽样；`C = floor(alpha×sum(w))` |
| 溯源回连 | 按 `source_row_ids` 回连 `lots_clean.csv`，`weights`/`values` 逐位相符。**这条是让"抽样被篡改"无处藏身的检查** |
| 不变量 | `family is None ⇔ source == "real"`；`len(weights) == len(values) == len(item_order) == n`；`values` 全正；`weights` 全正 |
| 50:50 | 每个 (split, n, alpha) cell 两侧计数相等 |
| 去重 | 全部实例（核心三 split + E3 + E4）的 `canonical_hash` 全局唯一；E3 与主集不相交 |
| 流隔离 | E4 的实例与主集 `n=200` 的实例**种子不同**（直接断言种子集合不相交） |
| 端到端 | 跑完整装配，逐文件 sha256 与 manifest 记录一致；**再跑一次，语料文件字节不变** |
| 依赖 | 全新 venv + `pip install -r requirements.txt`（含 PyYAML）后测试全绿 |

**`generated_at` 是时钟**，因此 manifest 本身不是字节稳定的——只有五个 `instances.jsonl` 是。这与采集阶段的结论一致，验收表述必须限定到语料文件。

为让测试跑得快，装配入口提供 `quota_override`，与 `run_generation` 的既有做法一致。

## 11. 验收标准

1. `datasets/processed/{train,validation,test}/instances.jsonl` 存在，行数分别为 12,000 / 2,400 / 2,400。
2. `datasets/processed/stress/{e3_size,e4_tail}/instances.jsonl` 存在，行数分别为 600 / 300。
3. 每个 (split, n, alpha) cell 的合成数与真实数相等。
4. 全部实例的 `canonical_hash` 全局唯一。
5. 随机抽若干真实实例，按 `source_row_ids` 回连 `lots_clean.csv` 逐位相符。
6. 重跑一次，五个语料文件的 sha256 与首次完全一致。
7. `data_manifest.json` 与 `experiment_config.yaml` 位于仓库根目录，**被 git 跟踪**。
8. `pytest` 全绿且无 skipped。
9. 全新 venv 装 `requirements.txt` 后测试全绿。

## 12. 歧义裁决记录

### 12.1 `MASTER_SEEDS` 是装饰品（既有事实，本阶段不改）

`spec.py` 的 `MASTER_SEEDS = {"train": 1001, "validation": 2001, "test": 3001}` **从未参与种子派生**。`derive_seed` 的键里只有 `"train"`/`"validation"`/`"test"` 三个**字符串**，那三个数字只被写进 manifest 的 `spec.master_seeds` 字段。效果上三个 split 确实分离（字符串不同），所以不是正确性缺陷，但那三个数是记录、不是机制——与 `xlsx_bytes` 曾经的问题同类。

第 8 节要求"保留 4001/5001 两条流"，照现有机制同样会是记录而非机制。本阶段的处理：**用显式流记号**（`"stream_size"` / `"stream_tail"`）承担真正的隔离，把 4001/5001 记入 `STRESS_MASTER_SEEDS` 作为声明标识，并在此处写明二者的关系。

**为什么不修**：把 master seed 真正塞进种子键，会让全部既有合成语料换种子、golden 全红、`datasets/raw/generation` 全部失效。代价与收益完全不成比例。

### 12.2 "value density" 的读法

第 8 节 E4b 说阈值取自"训练源 quantity 或 value density 的第 99 百分位"。`value density` 在源数据里有两种等价读法：`unit_price`（每单位价值）与 `value_pence / quantity_w`。二者只差浮点取整。**取 `unit_price`**：它是源表的一列，不引入额外的除法与舍入，且与第 3.1 节把 `v_i` 定义为"数量×单价"的口径一致。

### 12.3 `family` 对真实实例用 `null` 而非 `"real"`

第 10 节要求每条实例带 `family` 字段。真实数据没有族。用 `null` 并在种子键里另用内部记号，比把 `"real"` 冒充成一个族更诚实：下游任何按 `family` 分组的代码（第 8 节 E2 的分解）都不会把真实数据误当成第五个族。

### 12.4 E4 的生成器是变体而非改版

第 8 节说"Change the skewed generator to k=2…"。若字面改 `_skewed`，主语料的 skewed 半边会全部改变、golden 全红。**按"另立变体"实现**：主语料继续用第 2 节原配方，E4 用新变体，两者各自冻结、互不影响。

### 12.5 上游遗留的空目录

`datasets/processed/{train,val,test}/` 是上游遗留的空目录，`val` 与项目其余部分（`SPLITS`、`datasets/raw/generation/validation`、第 10 节）的 `validation` 不一致。本阶段把 `val` 正名为 `validation`，并在 `datasets/processed/` 下新增 `stress/`。
