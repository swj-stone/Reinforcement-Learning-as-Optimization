# 设计文档：原始数据采集与合成数据生成

**日期：** 2026-09-25
**范围：** `experiment/data/{collect,generate}` → `datasets/raw/{real,generation}`
**上游规格：** [EXPERIMENT_PLAN.md](../../../../EXPERIMENT_PLAN.md) 第 2、3.1 节
**状态：** 待实现

---

## 1. 范围

本次交付**只做**两件事：

1. **采集** UCI Online Retail，落原始工作簿、校验和、清洗后的交易行池。
2. **生成** 四个合成数据族的实例，按 cell 分片落到 `datasets/raw/generation/`。

### 明确不在本次范围

- `experiment/data/procession/`：日期切分、真实实例构造、50:50 混合、写入 `datasets/processed/*`
- RL 环境、DQN、两个 baseline、参照求解器（计划第 5–8 节的 E0–E5）
- 其他数据源（Amazon Last Mile Routing、Dubai Customs 空运单）

依据：`datasets/raw/` 与 `datasets/processed/` 的目录划分，以及计划第 1 节把 Day 1（取数审计）与 Day 3（生成冻结语料）分列为两项任务。真实数据的**实例构造**需要计划第 4 节的日期切分逻辑，属于 `procession/`，本次不做。

---

## 2. 环境与依赖

| 项 | 值 |
| --- | --- |
| 解释器 | `venv/Scripts/python.exe`，CPython 3.12.10 |
| 直接依赖 | `numpy`、`pandas`、`openpyxl`、`pytest` |
| 版本策略 | 全部按 `==` 精确固定 |
| 下载实现 | 标准库 `urllib.request`，不引入 `requests` |

不引入 `requests` 的理由：全程只有一次 23 MB 的 GET 加一次校验和计算，用标准库即可，少一个运行时依赖。

---

## 3. 模块划分

```
experiment/
├─ __init__.py
└─ data/
   ├─ __init__.py
   ├─ spec.py                  # 唯一真源：族名/n/α/配额/master seeds/清洗常量
   ├─ collect/
   │  ├─ __init__.py
   │  ├─ download.py           # 拉 zip → 落 xlsx，记录下载日期与 SHA-256
   │  └─ clean.py              # 清洗规则 1–6 → lots_clean.csv + collect_report.json
   └─ generate/
      ├─ __init__.py
      ├─ seeds.py              # 种子派生（纯函数，内容寻址）
      ├─ families.py           # 4 个纯抽样函数 (rng, n) -> (w, v)
      ├─ hashing.py            # canonical instance hash
      ├─ corpus.py             # cell 编排 + 配额 + 跨 split 去重 + 落盘 + manifest
      └─ cli.py                # python -m experiment.data.generate.cli
tests/
├─ __init__.py
├─ test_seeds.py
├─ test_families.py
├─ test_hashing.py
└─ test_cleaning.py
```

两个入口点相互独立：`python -m experiment.data.collect.download` 需要网络，`python -m experiment.data.generate.cli` 不需要。二者都只写 `datasets/raw/`，不互相依赖。

**隔离边界：** `families.py` 只做抽样，不碰文件系统、不含 I/O、不持全局状态；`corpus.py` 只做编排与落盘，不知道任一族的抽样公式。两侧都能独立理解与测试。

---

## 4. `spec.py` 中的冻结常量

| 常量 | 值 |
| --- | --- |
| `FAMILIES` | `("uniform", "normal", "correlated", "skewed")` |
| `SIZES` | `(20, 50, 100, 200)` |
| `ALPHAS` | `(0.25, 0.50, 0.75)` |
| `MASTER_SEEDS` | `{"train": 1001, "validation": 2001, "test": 3001}` |
| `SYNTHETIC_QUOTA` | `{"train": 125, "validation": 25, "test": 25}`（每 cell） |
| `UCI_URL` | `https://archive.ics.uci.edu/static/public/352/online+retail.zip` |
| `UCI_ZIP_ENTRY` | `Online Retail.xlsx` |
| `UCI_XLSX_BYTES` | `23715344` |
| `STOCK_CODE_RE` | `^\d{5}[A-Za-z]*$` |

四个族 × 四个规模 × 三个容量比 = **48 个 cell**；每 split 的实例数 = 48 × 配额 → train 6000、validation 1200、test 1200，合计 **8400** 个合成实例。

---

## 5. 种子派生

内容寻址，纯函数：

```
derive_seed(split, family, n, alpha, index, attempt, stream) -> uint64
  = int.from_bytes(
      blake2b(f"{split}|{family}|{n}|{alpha}|{index}|{attempt}|{stream}".encode(),
              digest_size=8).digest(),
      "big")
```

`stream ∈ {"items", "order"}`：`"items"` 用于抽权重与价值/参数，`"order"` 用于抽物品排列。**两条流相互隔离**，所以将来在 `"items"` 流里增减一次抽样不会改变已冻结语料中任何实例的物品顺序。

随后 `rng = numpy.random.default_rng(seed)`。

### 为什么不这样写

- **不用 `numpy.random.SeedSequence.spawn()`**：其 key 是位置式的，`cells` 列表里插入一项会让其后所有 cell 的种子整体平移，已冻结语料立即失效。
- **不用内建 `hash()`**：结果随 `PYTHONHASHSEED` 变化，跨进程不可复现。
- **不用 `master_seed + 整数偏移`**：族/规模/容量比组合数固定时可行，但任何新增维度都会与其他维度冲突。

内容寻址的代价是哈希本身不保证唯一。第 7 节的去重与重抽机制正是为此存在。

---

## 6. 四个族的抽样公式

`U(a,b)` 表示**含两端点**的离散均匀抽样，实现为 `rng.integers(a, b + 1)`。

| 族 | 公式 |
| --- | --- |
| `uniform` | `w_i = U(1,100)`；`v_i = U(1,100)` |
| `normal` | `w_i = clip(round_half_up(N(50, 15²)), 1, 100)`；`v_i` 同式独立抽取 |
| `correlated` | `w_i = U(1,100)`；`e_i = U(-10,10)`；`v_i = max(1, w_i + e_i)` |
| `skewed` | `w_i = U(1,20)`；`d_i = U(1,5)`；`k = max(1, floor(0.05·n + 0.5))`；`idx_w = choice(n, k, replace=False)` → `w[idx_w] = U(200,500)`；**独立地** `idx_d = choice(n, k, replace=False)` → `d[idx_d] = U(30,60)`；最后 `v_i = w_i · d_i` |

**偏斜族的三处关键细节**（错一处则语义全变）：

1. `idx_w` 与 `idx_d` 从**同一 rng 独立抽取两次**，两者**可以重叠**，重叠项的权重与密度都被替换。
2. `v_i = w_i · d_i` 用的是**替换之后**的 `w` 与 `d`。因 `v = w·d` 且 `w ≥ 1`，`d_i = v_i / w_i` 可精确还原，测试据此断言。
3. 极端观测值**一律保留**，不做任何裁剪或缩尾。

`k` 随 `n` 取值：n=20 → 1，n=50 → 3，n=100 → 5，n=200 → 10。

**容量：** `C = floor(alpha × Σw_i)`。三个 α 取值（0.25 / 0.50 / 0.75）都是二进制精确值，不存在浮点误差；仍以 `math.floor` 显式下取整。

**取整：** `round_half_up(x) = floor(x + 0.5)`。选择 half-up 而非 numpy 默认的 `np.rint`（银行家舍入，把 x.5 舍向偶数），理由是与计划第 3.1 节对价格的取整要求保持一致，且可预测。

`normal` 族**先取整、后裁剪**，与计划第 2 节的措辞顺序一致。在本参数下两种顺序的结果恰好相同（因为取整是单调的、且裁剪边界就是整数 1 与 100），但实现照抄规格顺序，以免将来更换参数时两者分叉。

**物品排列：** 从 `"order"` 流独立抽一个 `range(n)` 的排列。计划第 4 节要求 random 与 RL 共用同一排列。

---

## 7. Canonical hash 与确定性重抽

### 7.1 哈希定义

```
canonical_hash(n, C, w, v)
  = sha256( f"{n}|{C}|" + ";".join(f"{w},{v}" for (w,v) in sorted(zip(w, v))) ).hexdigest()
```

对 `(w_i, v_i)` 对**排序后**再哈希，使「物品顺序被打乱的副本」得到相同哈希——计划第 2 节明确要求识别 `copies with reordered items`。

### 7.2 遍历顺序

去重需要全局视角，因此规定固定的生成顺序，逐条与已接受集合比对：

```
for split in ("train", "validation", "test"):
    for family in FAMILIES:
        for n in SIZES:
            for alpha in ALPHAS:
                for index in range(quota[split]):
                    ...
```

一个实例只可能与**先于它**的实例冲突，所以单遍生成即可判定，无需二次扫描。顺序是冻结语义的一部分。

### 7.3 重抽

`index` 处发生冲突时，`attempt` 从 0 递增并重新派生种子，直到哈希未被占用。每个实例记录 `resample_attempt`，manifest 记录总重抽次数与最大 attempt。

**配额不因重抽改变**：重抽只换实例内容，不减少条数。

---

## 8. 输出格式

### 8.1 生成产物

```
datasets/raw/generation/
├─ manifest.json
├─ train/        48 个 cell，各 125 行
├─ validation/   48 个 cell，各 25 行
└─ test/         48 个 cell，各 25 行
```

cell 文件名：`{family}_n{n}_a{alpha_pct}.jsonl`，如 `uniform_n20_a25.jsonl`、`skewed_n200_a75.jsonl`。

每行一个实例（JSON Lines，UTF-8，LF 换行，`sort_keys=True` 以保证字节级可复现）：

```json
{
  "alpha": 0.25,
  "canonical_hash": "3f2a…",
  "capacity": 137,
  "family": "uniform",
  "instance_id": "syn-train-uniform-n20-a25-i0000",
  "item_order": [7, 2, 19, …],
  "n": 20,
  "resample_attempt": 0,
  "seed": 9876543210123456789,
  "source": "synthetic",
  "stats": {
    "cv_value": 0.55,
    "cv_weight": 0.57,
    "max_over_median_value": 2.9,
    "max_over_median_weight": 3.1,
    "top5_value_share": 0.267,
    "top5_weight_share": 0.281
  },
  "values": [12, 88, …],
  "weights": [34, 91, …]
}
```

`stats` 对四个族都计算（计划第 2 节只针对偏斜族要求，但统一计算使第 8 节 E2 的分布对比可直接取用）。`instance_id` 格式 `syn-{split}-{family}-n{n}-a{alpha_pct}-i{index:04d}`。`item_order` 是 `range(n)` 的一个排列。

`stats` 六个字段的定义（`x` 代表 `weights` 或 `values` 数组，全程不取整）：

| 字段 | 定义 |
| --- | --- |
| `top5_weight_share` | 取 `weights` 降序前 `k5 = max(1, ceil(0.05·n))` 项，其和除以 `Σweights` |
| `top5_value_share` | 同上，对 `values` 取前 `k5` 项 |
| `max_over_median_weight` | `max(weights) / median(weights)` |
| `max_over_median_value` | `max(values) / median(values)` |
| `cv_weight` | `std(weights, ddof=0) / mean(weights)` |
| `cv_value` | `std(values, ddof=0) / mean(values)` |

`k5` 随 `n` 取值：n=20 → 1，n=50 → 3，n=100 → 5，n=200 → 10（与偏斜族的 `k` 恰好同值，但两者定义独立：`k5` 用 `ceil`，偏斜族 `k` 用 `floor(x+0.5)`）。标准差用**总体**标准差（`ddof=0`）。因所有权重与价值均为正整数，`mean` 恒不为 0，无需零除兜底。`median` 为 `numpy.median`，偶数个元素时取中间两项的算术平均。

字段以浮点写出，序列化前不做舍入；比较与断言一律用显式容差，不做字符串相等比较。

### 8.2 manifest.json

```json
{
  "dedup": {"collisions_resampled": 0, "max_resample_attempt": 0, "total_instances": 8400},
  "generated_at": "2026-09-25T12:00:00Z",
  "generator_fingerprint": "sha256 over sorted(module file hashes)",
  "schema_version": 1,
  "spec": {"alphas": [...], "families": [...], "master_seeds": {...},
           "quota": {...}, "sizes": [...]},
  "cells": [
    {"alpha": 0.25, "count": 125, "family": "uniform", "n": 20,
     "path": "train/uniform_n20_a25.jsonl", "resample_attempts": 0,
     "sha256": "…", "split": "train"}
  ]
}
```

`generator_fingerprint` 用于事后诊断「输出为何变了」。每个 cell 的 `sha256` 使 manifest 可校验落盘内容。

### 8.3 采集产物

```
datasets/raw/real/
├─ online_retail.zip      # 原始压缩包
├─ online_retail.xlsx     # 解出的工作簿（23,715,344 B）
├─ lots_clean.csv         # 清洗后交易行池
└─ collect_report.json
```

`lots_clean.csv` 列：`source_row_id, invoice_no, stock_code, invoice_date, quantity_w, unit_price, value_pence`。

按计划第 3.1 节规则 6，`CustomerID`、`Description`、`Country` **不进入清洗后行池**（原始工作簿已保留全部字段，信息不丢失）。

`collect_report.json` 记录：下载 URL、下载时间（UTC）、zip 与 xlsx 的 SHA-256 与字节数、原始行数、各清洗理由的丢弃计数、保留行数、去重规则说明、日期分布摘要（不重复日期数、每日行数的 min/median/max）。

---

## 9. 清洗规则

按固定顺序逐条应用，一行命中多条时**归因于首个命中的理由**，保证「原始行数 = 保留行数 + 各理由丢弃数之和」这一恒等式成立，可被测试断言。

| # | 规则 | 处置 |
| --- | --- | --- |
| 1 | 按原始表行序分配稳定 `source_row_id`（从 1 起） | 保留 |
| 2 | `InvoiceNo` 以 `C`/`c` 开头 | 丢弃（取消单） |
| 3 | `Quantity <= 0` 或 `UnitPrice <= 0` | 丢弃 |
| 4 | `InvoiceNo`/`StockCode`/`Quantity`/`InvoiceDate`/`UnitPrice` 任一缺失 | 丢弃 |
| 5 | `InvoiceDate` 无法解析 | 丢弃 |
| 6 | `Quantity` 非正整数（小数或非数字） | 丢弃，单独计数——**不做静默取整** |
| 7 | `StockCode` 不匹配 `^\d{5}[A-Za-z]*$` | 丢弃，单独计数（排除邮费、人工调整等非商品码） |
| 8 | 全字段精确重复行（含 NaN 视为相等） | 去重，保留 `source_row_id` 最小者；保留合法重复商品与不同订单行 |
| 9 | 转换后 `value_pence == 0` | 丢弃 |

**价值转换：** `value_pence = round_half_up(Decimal(str(UnitPrice)) × Decimal(Quantity) × 100)`，全程 `decimal.Decimal` 算术、`ROUND_HALF_UP`，先转字符串再进 `Decimal` 以避免二进制浮点残差。`w_i = Quantity`（正整数）。

**正异常值一律保留**，不按任何分位数裁剪；舍弃规则只在训练记录上审计，且在生成最终验证/测试语料前冻结。

---

## 10. 验证策略

测试驱动：先写测试锁定规格，再实现。

| 测试文件 | 断言 |
| --- | --- |
| `test_seeds.py` | 同输入同种子；split/family/n/α/index/attempt/stream 任一变化则种子变化；**与调用顺序无关**；结果落在 uint64 范围内 |
| `test_families.py` | `U(a,b)` 含端点（大样本下 a 与 b 均出现）；`uniform` 的 w/v 相关系数近 0；`normal` 全部落在 [1,100] 且均值 ≈ 50；`round_half_up` 直接单测 .5 用例；`correlated` 的 `v-w ∈ [-10,10]` 且 `v ≥ 1`；`skewed` 恰好 k 项 `w ≥ 200`、恰好 k 项 `v/w ≥ 30`、两集合可重叠、未替换项的 `w ∈ [1,20]` 且 `v/w ∈ [1,5]`；所有族所有 cell 的 `C == floor(α·Σw)` |
| `test_hashing.py` | 物品重排后哈希不变；容量改变则哈希改变；重复检测有效 |
| `test_cleaning.py` | C 前缀单被丢；非正数量/价格被丢；小数数量被**拒绝而非取整**且单独计数；商品码正则生效；`UnitPrice=0.005, Quantity=1` → `value_pence=1`（half-up）；`value_pence==0` 被拒；精确重复行去重；异常值保留；恒等式「原始 = 保留 + Σ丢弃」成立 |
| 金样本测试 | 固定 `(split, family, n, α, index)` 断言精确的 `w`/`v` 数组，把已冻结语义钉死 |

---

## 11. 验收标准

1. `pytest` 全绿。
2. 144 个 cell 全部存在，每 cell 条数与配额精确一致（train 125、validation/test 25）。
3. 8400 个实例的 `canonical_hash` **全局无重复**（重抽后）。
4. 每个 cell 文件的实际 SHA-256 与 manifest 记录一致。
5. **重跑生成器产出字节级相同的产物**（幂等性）。
6. `lots_clean.csv` 行数与各丢弃计数满足恒等式，且 `collect_report.json` 中的 xlsx SHA-256 与本地文件实测一致。
7. `requirements.txt` 中版本精确固定，且在干净 venv 中可 `pip install -r` 复现。

---

## 12. 歧义裁决记录

规格中有两处未定义，本设计做出的选择及理由：

| 歧义 | 裁决 | 理由 |
| --- | --- | --- |
| 计划第 2 节 `normal` 族只说 "round to the nearest integer"，未定义 .5 的取整方向 | **half-up**（`floor(x+0.5)`） | 与第 3.1 节对价格的 round-half-up 要求一致，且避免 numpy 默认银行家舍入带来的意外 |
| 计划第 2 节要求 "Reject duplicate canonical instance hashes across splits"，未说明拒绝后如何处理 | **确定性重抽**（`attempt` 递增重派生） | 配额因此仍可满足，且整个过程可复现；比「检出冲突即报错退出」更适配「先生成后审计」的流程 |

两项裁决均已反映在本文档正文，并由第 10 节的测试锁定。
