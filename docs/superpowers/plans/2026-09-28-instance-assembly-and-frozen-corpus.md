# 实例装配与冻结混合语料 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `datasets/raw/` 下的合成语料与清洗后的 UCI 批次表装配成冻结的 50:50 混合语料，写入 `datasets/processed/`，并把 `data_manifest.json` 与 `experiment_config.yaml` 写到仓库根目录。

**Architecture:** 新建 `experiment/data/assemble/` 包，与 `collect/`、`generate/` 平级。数据流单向：`lots.py`（CSV → 按日期分区的候选池）→ `real.py`（无放回抽样成实例）→ `stress.py`（E3/E4 两条保留流）→ `merge.py`（合并、50:50 断言、全局去重、落盘）→ `manifest.py`（协议产物），`cli.py` 是入口。序列化复用 `write_cell`，种子派生复用 `derive_seed`，计数公式复用 `capacity_for`/`compute_stats`。

**Tech Stack:** Python 3.12、numpy、PyYAML（新增）、pytest。

**依据:** `docs/superpowers/specs/2026-09-28-instance-assembly-and-frozen-corpus-design.md`（已提交，`2444e22`）。

---

## 0. 执行者必读

### 0.1 环境

- 工作目录：`C:\work\学校\A.SPBU\Saint-peter-26\AI-2\project`
- **必须用 venv 的解释器跑测试**：`./venv/Scripts/python.exe -m pytest ...`。裸 `python` 上没有 pytest。
- **`--basetemp` 是强制的，且必须指向仓库之外**：
  `./venv/Scripts/python.exe -m pytest tests/... -p no:cacheprovider --basetemp="$TEMP/ap_<task>" -q`
  仓库内的临时目录会被 git 看见；而默认的 `%TEMP%\pytest-of-<user>` 里有别的会话留下的、**不得触碰**的内容。
- 控制台是 GBK 代码页，中文可能显示成乱码。**判断内容正确性时比对哈希或字节，不要比对屏幕上的字符。**
- Bash 工具每次调用后工作目录会重置；命令里写绝对路径或用 `cd ... && ...`。
- **绝对不要删除、修复或接管 `C:\Users\swj17\AppData\Local\Temp\pytest-of-swj17`。** 它不属于本任务。
- 行尾必须是 LF、不带 BOM。`.gitattributes` 已对 `*.py`/`*.json`/`*.yaml`/`*.md` 强制 `eol=lf`。

### 0.2 分支与基线

当前在 `main`。**动手前先由控制者建好 `feat/instance-assembly` 并切过去**，不要在 `main` 上提交。

基线（实测于 2026-09-28）：`177 passed in 139.77s`，0 failed、0 skipped。

### 0.3 不可修改的文件

`experiment/data/spec.py` 只允许**新增**本计划 Task 1 列出的常量，已有序段一行不动。

以下文件**完全不许改**：`experiment/data/generate/*`（含 `families.py` 的 `_skewed`、`corpus.py`、`seeds.py`、`hashing.py`、`cli.py`）、`experiment/data/collect/*`，以及 `tests/` 下**本计划新建之外的一切既有测试文件**（`test_spec.py`、`test_seeds.py`、`test_families.py`、`test_hashing.py`、`test_golden.py`、`test_corpus.py`、`test_lots.py`、`test_pence.py`、`test_cleaning.py`、`test_collect_io.py`、`test_collect_integration.py`）。清单以 `git ls-files tests/` 为准，不逐条记忆。

理由：生成阶段的语料已被金样本钉死，改那些文件会作废已冻结的语料。

### 0.4 诊断纪律

- **永远不要为了做变异测试去改仓库里的真文件。** 验证一条测试是否真会失败，只能在仓库外写 scratch 脚本、用 `monkeypatch`，或复制一份再改。
- 探针脚本一律写在仓库外（`$TEMP` 下），不许往仓库里写任何临时文件。
- 每个 Task 结束时 `git status --short` 必须干净（除被 ignore 的 `.pyc`）。

### 0.5 已有的可复用纯函数（不要重写）

```python
from experiment.data.generate.hashing import canonical_hash          # (n, capacity, weights, values) -> str
from experiment.data.generate.families import capacity_for, compute_stats, generate_items
from experiment.data.generate.seeds import derive_seed               # (split, family, n, alpha, index, attempt=0, stream="items")
from experiment.data.generate.corpus import write_cell               # (path, instances) -> sha256
```

`write_cell` 的序列化约定（**必须沿用**）：每行 `json.dumps(inst, sort_keys=True, ensure_ascii=False)`，`"\n".join(lines) + "\n"`，UTF-8 落盘。

`derive_seed` 的键是 `"|".join((split, family, str(n), alpha_key(alpha), str(index), str(attempt), stream))`，`stream ∈ ("items", "order")`。**`MASTER_SEEDS` 与 `STRESS_MASTER_SEEDS` 都不参与这个键**——流之间的隔离靠字符串记号，不靠那个数字。这是设计文档 §12.1 记录的已知事实，不要试图"修好"它。

### 0.6 已核实的外部事实（写断言时会用到）

- `spec.CLEANING_RULES` 是 **8 个字符串**（规则名）的 tuple，不是 dict。
- 上游采集报告是 `datasets/raw/real/collect_report.json`（**不叫 manifest.json**），键为 `cleaning` / `generated_at` / `kept_rows` / `lots_columns` / `raw_rows` / `schema_version` / `source`；`source` 里有 `url` / `xlsx_bytes` / `xlsx_sha256` / `zip_sha256` / `lots_sha256`。`kept_rows` 实测 522504。
- 生成阶段清单是 `datasets/raw/generation/manifest.json`，含 `generator_fingerprint`（实测 `c59e47b1…`）。
- `datasets/processed/{train,test,val}/` 三个目录**都存在且都是空的**。
- `datasets/raw/real/lots_clean.csv`：522,504 行、19,773 张发票、305 个不同日期。`invoice_date` 形如 `2010-12-01 08:26:00`，`unit_price` 是十进制字符串，`quantity_w`/`value_pence` 是裸整数，`invoice_no` 是纯数字串。
- 实测切分：边界 `floor(0.70×305)=213`、`floor(0.85×305)=259`；train 213 天 / 299,597 批 / 12,068 票，validation 46 / 88,796 / 3,191，test 46 / 134,111 / 4,514。批次数 ≥500 的日期：train 209、validation 46、test 46。
- **生成语料的 cell 文件名只覆盖 `n ∈ {20,50,100,200}`**（每个 split 48 个文件）。任何"从合成语料读回"的测试**不能用 n=4**。

### 0.7 与设计文档的两处补充（不是偏离，是补全）

设计文档 §4 的常量块点名了 `e3_size` / `e4_tail` 两个数据集名与 E4 的 `n=200`，但没有给出承载它们的常量。本计划补上 `E3_NAME` / `E4_NAME` / `E4_SIZE`，并补一个 `capacity_cells()` 生成器（§8.3 的冻结遍历顺序）。三处都是 §5/§7.2/§7.3/§8.3 已经写死的事实的直接编码。

`experiment_config.yaml` 里，设计文档 §8.6 的草图把 `python` 放在 `packages` 内。本计划改成独立的 `runtime.python`，好让 `packages` 与 `requirements.txt` 保持逐项对应（否则"packages 是否忠于 requirements.txt"这条断言就没法写）。另把 §6 的训练预算段命名为 `budgets`（与设计文档 §8.6 一致），`methods` 取 `[random, greedy, dqn]`（设计文档 §8.6 的写法）。

### 0.8 执行后的修订（2026-09-28，Task 7 通过之后、Task 8 之前）

Task 1–7 全部实现、双阶段审查通过之后，控制者发现三处需要修订，逐项获准后落地。**本次只改了 Task 6 的围栏**（Task 1–5、7 的围栏一字未动），改完后 `manifest.py` 与围栏重新逐字节对齐，由 `check_fences.py` 复核：18 个围栏、6 个整文件命中。

1. **路径相对化。** `manifest.py` 新增 `_display_path()`：仓库内记 `relative_to(REPO_ROOT).as_posix()`，仓库外记 `.as_posix()`。`_file_fact` 的三处调用（`lots_clean` / `collect_report` / `generation_manifest`）与 `output_dir` 都改走它。**`output_dir` 原先在"仓库外"那一支用的是 `str()`，会给 JSON 产物写进反斜杠**；现在两条分支共用一条规则。绝对路径会把 `C:/work/学校/…` 这种本机信息写进**纳入版本库**的凭据里：换一台机器复核时，它既指向一个不存在的目录树，也无法与复核者 `git status` 里的字符串比对。

2. **§6 的正文补进 config。** 新增 `_RL_ENVIRONMENT` 常量与顶层 `rl_environment` 段，转录 §6 **表格之外**的正文：`gamma=1`、奖励整形 `v_i / Σv_j`、回合/状态/动作/终止语义、动作掩码、网络结构（2 层 64 单元 ReLU 编码器、128 单元头、两个 Q 值输出、池化规则、`log(1+n)`、"IDs and source labels are not prediction features"）、平衡块采样、检查点选择（按 `V/U`，从不用测试集）、以及 300k→1M 的上调条款。§10 要求 config 记录"RL 设置"，而原先只抄了表格——**表格里既没有 `gamma` 也没有网络结构**，这两项恰恰决定能不能复现训练。`budgets` 仍是那张表，两段不合并。

3. **新增冻结金样本。** 见 §0.9。

同批修掉两处文档勘误：设计文档 §8.6 的 `skewed` 参数写错了（原写 `heavy_weight U(500,2000)`、`heavy_density U(1,20)`；`spec.py` 实测是 `(200,500)` 与 `(30,60)`——`(500,2000)` 是 E4 的量程，`(1,20)` 是 `SKEW_BASE_WEIGHT_RANGE` 被串进了 heavy 槽位）；§10 末尾的 `quota_override` 表述已过时（入口暴露的是 `--synthetic-quota` / `--real-quota`）。

**第四项：Task 8 Step 8 的一处预测值是错的，已按代码实测改正。** 原文预测 `dedup.total_instances` 为 `17400`，但 `merge.py:234` 与 `:252` 表明这个累加器在三个主 split 之外**把两个应力集也算进去**——真实规模应为 `12000+2400+2400+600+300 = 17700`。`17400` 恰好等于「主集 + E3」，把 E4 的 300 漏掉了。金样本独立印证了这个语义：其夹具是 `288+600+300=1188`，若不计 E4 则只有 888。**这个错误若不改，实现者会拿一个正确的产物去对一个错的预期**，然后报告一次假失败，甚至去"修" `merge.py`。Task 6 测试夹具里那个同值的 `17400`（L2748）**不动**——它是 stub，`build_manifest` 原样回显，且 Task 6 已实现提交，改了就是 drift；它多半也正是这处预测错误的来源。

### 0.9 冻结金样本（Task 8 之前必须已入库）

`tests/golden/assemble_sections.json` + `tests/test_assemble_golden.py`（16 条）。

**补的是什么盲区。** Task 1–7 的测试在身份、计数、50:50、种子、流隔离上都强，但**派生内容**（切分边界、每 split 的合格日期与实测占比、每个语料文件的 sha256/行数/字节数）与**入库产物的字节**（缩进、键序、行尾、BOM）一条断言都没有：改一个取整方向、少算一个 split、把缩进从 2 改成 4，整套测试依然全绿。

**夹具是 10 天 × 600 批次，不是 `conftest.lots_csv` 的 10 × 24。** E3 的真实半边要从**单个**测试日期里抽出 500 个批次，而 24 批次/天在 n=50 就报错停下（`train 中没有任何日期能提供 n=50 个批次`）。夹具因此自建，并跑**生产形态**（`include_stress=True`）。

**钉住 manifest 的十段**：`cells` / `counts` / `dedup` / `files` / `schema_version` / `split_policy` / `splits` / `streams` / `stress` / `tail_thresholds`。**只有三段不钉**：`generated_at`（时钟）、`output_dir` 与 `provenance.*.path`（本次运行的临时目录）。有一条断言把这个差集写死，防止"没钉住"的范围悄悄变大。

**没有"缺失即生成"的回退。** `tests/test_golden.py` 有（首次运行 `pytest.skip`），那在本项目是陷阱：验收要求「全绿且无 skipped」（设计 §11.8），而缺失即跳过会让金样本在文件丢失时静默通过。这里文件缺失是硬错误。

**两处值得知情的事实。**

- `provenance.generation_manifest.sha256` 复现不了：它哈希的 `generation_dir/manifest.json` 内含**自己的** `generated_at`，每次重跑都不同。设计 §10 已声明 manifest 本身不是字节稳定的（只有五个语料文件是），故不是缺陷；但该字段不能充当跨机器复现的凭据。
- 金样本的 `files` 段钉住的是**夹具规模**（10 天 × 600 批次）下的行数与摘要，不是真实语料的。这是刻意的：它测的是装配逻辑与字节格式，真实规模由 Task 8 的验收步骤单独量。

**因此 Task 8 的测试总数由 291 变为 307**（Step 2 与 Step 9 的预期已同步更新）。

---

## 1. 文件结构

| 文件 | 职责 |
| --- | --- |
| `experiment/data/assemble/__init__.py` | 空包标记 |
| `experiment/data/assemble/lots.py` | 读 `lots_clean.csv`；发票→最早日期；日期切分；按日建池；合格日期；E4b 阈值 |
| `experiment/data/assemble/real.py` | 真实实例抽样与构造；`fill_cell` 重抽循环 |
| `experiment/data/assemble/stress.py` | E4 变体生成器；E3/E4 实例集 |
| `experiment/data/assemble/merge.py` | `assemble_split`、`run_assembly` |
| `experiment/data/assemble/manifest.py` | `upstream_provenance`、两个协议产物的构造与落盘 |
| `experiment/data/assemble/cli.py` | 命令行入口 |
| `tests/conftest.py` | 两个密闭夹具：`lots_csv`、`fixture_corpus` |
| `tests/test_assemble_spec.py` | 新增常量 |
| `tests/test_assemble_lots.py` | 切分、整票随日期、合格日期、阈值 |
| `tests/test_assemble_real.py` | 抽样、溯源回连、容量公式、重抽 |
| `tests/test_assemble_stress.py` | E4 变体、E3/E4 形状、流隔离 |
| `tests/test_assemble_merge.py` | 上游 provenance、顺序、50:50、全局去重、字节稳定、端到端 |
| `tests/test_assemble_manifest.py` | manifest/config 的字段、真实性与格式 |
| `tests/test_assemble_cli.py` | 入口的退出码与产物 |
| `tests/golden/assemble_sections.json` | 冻结金样本：manifest 十段的实测值（§0.9） |
| `tests/test_assemble_golden.py` | 金样本比对 + 两个协议产物的字节格式（§0.9，16 条） |

---

## Task 1: 装配阶段的冻结常量与 PyYAML 依赖

**Files:**
- Modify: `experiment/data/spec.py`（仅在文件末尾追加）
- Modify: `requirements.txt`
- Test: `tests/test_assemble_spec.py`（新建）

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_assemble_spec.py`：

```python
"""装配阶段新增常量的契约。

断言刻意是算术恒等式与集合关系，而不是「常量等于我抄的那个数」——
后者在常量被改错时同样通过。
"""

from experiment.data import spec


def test_real_quota_matches_the_plan():
    assert spec.REAL_QUOTA == {"train": 500, "validation": 100, "test": 100}


def test_capacity_cells_put_size_in_the_outer_loop():
    """12 个 cell 的顺序是冻结的产物顺序：n 在外层、alpha 在内层。"""
    assert spec.capacity_cells()[:3] == [(20, 0.25), (20, 0.50), (20, 0.75)]
    assert spec.capacity_cells()[-1] == (200, 0.75)
    assert len(spec.capacity_cells()) == len(spec.SIZES) * len(spec.ALPHAS)


def test_capacity_cells_nesting_differs_from_all_cells():
    """all_cells() 是 family 外层（服务生成的去重顺序），本函数是 n 外层。"""
    assert len(spec.all_cells()) == 48
    # all_cells() 的元素是 (family, n, alpha)：前三个同为 uniform_n20，只有 alpha 在变
    assert [c[2] for c in spec.all_cells()[:3]] == list(spec.ALPHAS)
    assert {c[0] for c in spec.capacity_cells()} == set(spec.SIZES)


def test_stress_quotas_sum_to_the_counts_the_plan_requires():
    """第 8 节：E3 合成 300 + 真实 300；E4 合成 300。"""
    assert spec.E3_SYNTHETIC_PER_FAMILY * len(spec.FAMILIES) * len(spec.ALPHAS) == 300
    assert spec.E3_REAL_PER_ALPHA * len(spec.ALPHAS) == 300
    assert spec.E4_COUNT_PER_ALPHA * len(spec.ALPHAS) == 300


def test_e4_variant_parameters_match_the_plan():
    assert spec.E4_SIZE == 200
    assert spec.E3_SIZE == 500
    assert spec.E4_SKEW_K == 2
    assert spec.E4_SKEW_HEAVY_WEIGHT_RANGE == (500, 2000)
    assert spec.E4_SKEW_HEAVY_DENSITY_RANGE == (100, 200)


def test_seed_tokens_are_outside_the_generation_domains():
    """流隔离的地基。

    种子键的 split 槽位装的是 split 名字、family 槽位装的是族名。一旦某个
    记号与生成阶段的取值重合，真实或应力实例就会与合成实例共用种子域，
    第 8 节「保留独立的 4001/5001 两条流」就名存实亡。
    """
    assert spec.REAL_SEED_TOKEN not in spec.FAMILIES
    assert spec.E4_SEED_TOKEN not in spec.FAMILIES
    assert spec.STRESS_SIZE_STREAM not in spec.SPLITS
    assert spec.STRESS_TAIL_STREAM not in spec.SPLITS
    assert spec.STRESS_SIZE_STREAM != spec.STRESS_TAIL_STREAM


def test_dataset_names_match_the_output_layout():
    assert spec.E3_NAME == "e3_size"
    assert spec.E4_NAME == "e4_tail"


def test_protocol_artifacts_live_at_the_repository_root():
    """datasets/ 被 gitignore，放在那里的审计凭据在新克隆上不存在。"""
    assert spec.DATA_MANIFEST_PATH == spec.REPO_ROOT / "data_manifest.json"
    assert spec.EXPERIMENT_CONFIG_PATH == spec.REPO_ROOT / "experiment_config.yaml"
    assert spec.DATASETS_PROCESSED == spec.REPO_ROOT / "datasets" / "processed"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_spec.py -p no:cacheprovider --basetemp="$TEMP/ap_t1" -q
```

预期：`AttributeError: module 'experiment.data.spec' has no attribute 'REAL_QUOTA'`，8 个全部失败。

- [ ] **Step 3: 在 `spec.py` 末尾追加常量**

在 `experiment/data/spec.py` 的末尾追加（**不要改动文件里任何已有内容**）：

```python


# --- 实例装配（第三阶段，依据 EXPERIMENT_PLAN.md 第 4 节与第 8 节）---------

#: 真实实例每个 (n, alpha) cell 的配额（第 4 节）
REAL_QUOTA: dict[str, int] = {"train": 500, "validation": 100, "test": 100}

#: 日期切分的两个边界占比（第 4 节），均下取整
SPLIT_DATE_FRACTIONS: tuple[float, float] = (0.70, 0.85)

#: 真实实例在种子键中占用的 family 槽位记号。刻意不在 FAMILIES 中：
#: 真实实例的种子因此不可能与任何合成族相同。
REAL_SEED_TOKEN: str = "real"

#: 应力集的流记号，只活在种子键内部，不落盘。落盘用的是 E3_NAME / E4_NAME。
#: 第 8 节要求「保留 4001/5001 两条流」；真正承担隔离的是这两个字符串记号，
#: 而不是 STRESS_MASTER_SEEDS 里的数字（见设计文档第 12.1 节）。
STRESS_SIZE_STREAM: str = "stream_size"
STRESS_TAIL_STREAM: str = "stream_tail"

#: E4 变体生成器在种子键中占用的 family 槽位记号
E4_SEED_TOKEN: str = "skewed_e4"

#: 第 8 节声明的应力流主种子。与 MASTER_SEEDS 同样是**声明标识**，不参与
#: derive_seed 的键；记录在此是为了让声明与机制的关系可被复核。
STRESS_MASTER_SEEDS: dict[str, int] = {"size": 4001, "tail": 5001}

#: 应力集的数据集名，同时是目录名与 instance_id 的组成部分
E3_NAME: str = "e3_size"
E4_NAME: str = "e4_tail"

#: E3（未见过的规模）配额，第 8 节
E3_SIZE: int = 500
E3_SYNTHETIC_PER_FAMILY: int = 25
E3_REAL_PER_ALPHA: int = 100

#: E4（更强的不平衡）配额与变体参数，第 8 节
E4_SIZE: int = 200
E4_COUNT_PER_ALPHA: int = 100
E4_SKEW_K: int = 2
E4_SKEW_HEAVY_WEIGHT_RANGE: tuple[int, int] = (500, 2000)
E4_SKEW_HEAVY_DENSITY_RANGE: tuple[int, int] = (100, 200)

#: E4b 阈值分位数（第 8 节）。阈值只用训练记录定；本阶段只算并冻结，
#: 实际标记属于评估阶段。
TAIL_QUANTILE: float = 0.99

# --- 装配阶段的输出路径 -----------------------------------------------------

DATASETS_PROCESSED: Path = REPO_ROOT / "datasets" / "processed"

#: 协议产物放在仓库根目录并纳入版本库。datasets/ 整体被 gitignore，把审计
#: 凭据放在那里等于换一台机器就无从证明数据未被调换。
DATA_MANIFEST_PATH: Path = REPO_ROOT / "data_manifest.json"
EXPERIMENT_CONFIG_PATH: Path = REPO_ROOT / "experiment_config.yaml"


def capacity_cells() -> list[tuple[int, float]]:
    """按冻结顺序枚举 12 个 ``(n, alpha)`` cell：n 在外层、alpha 在内层。

    这是合并文件里实例的排列顺序，也是 50:50 断言的遍历顺序。刻意与
    :func:`all_cells` 的 family-外层嵌套序不同：那个服务生成的去重顺序，
    本函数服务面向读者的产物顺序。
    """
    return [(n, a) for n in SIZES for a in ALPHAS]
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_spec.py -p no:cacheprovider --basetemp="$TEMP/ap_t1" -q
```

预期：`8 passed`。

- [ ] **Step 5: 跑全量测试确认没有破坏既有断言**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t1all" -q 2>&1 | tail -3
```

预期：`185 passed`（基线 177 + 新增 8），耗时约 140 秒，**不得出现 skipped 或 failed**。

- [ ] **Step 6: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/spec.py tests/test_assemble_spec.py && git commit -m "feat(data): 装配阶段的冻结常量与路径"
```

- [ ] **Step 7: 加入 PyYAML 依赖**

在 `requirements.txt` 的运行时依赖段末尾加一行：

```
PyYAML==6.0.2
```

- [ ] **Step 8: 安装并确认可用**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pip install -r requirements.txt 2>&1 | tail -3 && ./venv/Scripts/python.exe -c "import yaml; print('yaml', yaml.__version__)"
```

预期：打印 `yaml 6.0.2`。**若 pip 实际装到的版本与 Step 7 写的不一致，用实测版本回写 `requirements.txt`。**

- [ ] **Step 9: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add requirements.txt && git commit -m "chore(deps): 加入 PyYAML 以稳定生成 experiment_config.yaml"
```

---

## Task 2: `lots.py` —— 批次表到按日候选池

**Files:**
- Create: `experiment/data/assemble/__init__.py`
- Create: `experiment/data/assemble/lots.py`
- Create: `tests/conftest.py`
- Test: `tests/test_assemble_lots.py`

- [ ] **Step 1: 建两个夹具**

创建 `tests/conftest.py`：

```python
"""装配阶段的共享夹具。

夹具全部在 tmp_path 内自建，**不读 datasets/**：该目录被 gitignore，新克隆上
不存在，依赖它会让测试在别人机器上静默跳过或静默通过。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from experiment.data.generate.corpus import run_generation

LOTS_HEADER = (
    "source_row_id",
    "invoice_no",
    "stock_code",
    "invoice_date",
    "quantity_w",
    "unit_price",
    "value_pence",
)

#: 夹具批次表的天数与每天批次数。每天 24 条 ≥ 20，使 (n=20, alpha) 这个
#: **真实存在**的 cell 在密闭夹具上可装配（生成语料的 n 只取 20/50/100/200）。
FIXTURE_DAYS = 10
FIXTURE_LOTS_PER_DAY = 24

#: 夹具批次表的总行数，与随附的 collect_report.json 的 kept_rows 一致
FIXTURE_ROWS = FIXTURE_DAYS * FIXTURE_LOTS_PER_DAY


def write_lots_csv(path: Path, rows: list[tuple]) -> Path:
    """把 ``(row_id, invoice_no, day, qty, unit_price, value_pence)`` 写成批次表。

    列序与 ``spec.LOTS_COLUMNS`` 一致，日期统一补 `` 08:00:00``。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(LOTS_HEADER)]
    for row_id, invoice_no, day, qty, price, pence in rows:
        lines.append(f"{row_id},{invoice_no},85123A,{day} 08:00:00,{qty},{price},{pence}")
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


@pytest.fixture
def lots_csv(tmp_path: Path) -> Path:
    """10 天、每天 24 条的批次表，日期与行号的对应关系可预测。

    切分边界（第 4 节）：``floor(0.70*10)=7``、``floor(0.85*10)=8``，故
    train 7 天、validation 1 天、test 2 天。

    同目录下另写一份 `collect_report.json`，因为装配会拿它的 ``kept_rows``
    与批次表的实际行数对账——那是一条真实的上游产物一致性检查，夹具必须
    让它有得查。
    """
    rows = []
    row_id = 0
    for day_index in range(FIXTURE_DAYS):
        day = (date(2011, 1, 1) + timedelta(days=day_index)).isoformat()
        for slot in range(FIXTURE_LOTS_PER_DAY):
            rows.append(
                (
                    row_id,
                    100000 + day_index * 100 + slot,
                    day,
                    slot + 1,          # quantity_w 1..24
                    "2.50",
                    (slot + 1) * 250,  # value_pence = quantity_w * 250
                )
            )
            row_id += 1
    path = write_lots_csv(tmp_path / "lots_clean.csv", rows)
    (tmp_path / "collect_report.json").write_bytes(
        json.dumps(
            {
                "kept_rows": FIXTURE_ROWS,
                "schema_version": 1,
                "source": {
                    "url": "https://example.invalid/fixture.zip",
                    "xlsx_bytes": 123,
                    "xlsx_sha256": "0" * 64,
                },
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return path


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> Path:
    """每 cell 1 条的小型合成语料，供 merge 的密闭测试使用。

    走既有的 ``run_generation``，因此 cell 集合与真实语料完全一致
    （n ∈ SIZES，每 split 48 个文件）——测试里出现的 n 必须取自 SIZES。
    """
    out = tmp_path / "generation"
    run_generation(out_dir=out, quota_override={"train": 1, "validation": 1, "test": 1})
    return out
```

- [ ] **Step 2: 写失败的测试**

创建 `tests/test_assemble_lots.py`：

```python
"""日期切分、整票随最早日期、候选池与合格日期、E4b 阈值。"""

from decimal import Decimal

import pytest

from experiment.data.assemble import lots as L
from tests.conftest import FIXTURE_LOTS_PER_DAY, write_lots_csv


# --- 切分 -------------------------------------------------------------------


def test_split_dates_floors_both_boundaries():
    """第 4 节要求两个边界都下取整。

    取 13 个日期：floor(0.70*13)=9、floor(0.85*13)=11，故 train 9 天、
    validation 2 天、test 2 天。若实现用了 round 或 ceil 就会不同。
    """
    days = [f"2011-01-{d:02d}" for d in range(1, 14)]
    split = L.split_dates(days)
    assert len(split["train"]) == 9
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    assert split["train"][0] == "2011-01-01"
    assert split["test"][-1] == "2011-01-13"


def test_split_dates_are_disjoint_and_cover_everything():
    days = [f"2011-02-{d:02d}" for d in range(1, 29)]
    split = L.split_dates(days)
    flat = split["train"] + split["validation"] + split["test"]
    assert len(flat) == len(set(flat)) == 28
    assert set(flat) == set(days)
    assert max(split["train"]) < min(split["validation"]) < min(split["test"])


def test_split_dates_deduplicates_and_sorts_input():
    split = L.split_dates(["2011-01-03", "2011-01-01", "2011-01-03", "2011-01-02"])
    assert split["train"] + split["validation"] + split["test"] == [
        "2011-01-01",
        "2011-01-02",
        "2011-01-03",
    ]


def test_split_dates_can_leave_a_partition_empty():
    """3 天：floor(0.70*3)=2、floor(0.85*3)=2，validation 为空。"""
    split = L.split_dates(["2011-01-01", "2011-01-02", "2011-01-03"])
    assert split["train"] == ["2011-01-01", "2011-01-02"]
    assert split["validation"] == []
    assert split["test"] == ["2011-01-03"]


# --- 发票的最早日期 ---------------------------------------------------------


def _midnight_csv(tmp_path):
    rows = [
        (0, 900001, "2011-03-02", 3, "1.00", 300),
        (1, 900001, "2011-03-03", 4, "1.00", 400),
        (2, 900002, "2011-03-03", 5, "1.00", 500),
    ]
    return write_lots_csv(tmp_path / "mid.csv", rows)


def test_invoice_spanning_midnight_takes_its_earliest_date(tmp_path):
    """第 4 节：发票归到它最早一笔交易的日历日，即使跨午夜。"""
    index = L.load_index(_midnight_csv(tmp_path), fractions=(0.0, 0.0))
    assert index.first_dates["900001"] == "2011-03-02"
    assert index.first_dates["900002"] == "2011-03-03"
    assert index.date_splits["test"] == ["2011-03-02", "2011-03-03"]


def test_all_lots_of_an_invoice_land_in_the_earliest_days_pool(tmp_path):
    """跨午夜的发票，两笔都必须进「最早那天」的池子，而不是各自那天。"""
    index = L.load_index(_midnight_csv(tmp_path), fractions=(0.0, 0.0))
    assert [lot.source_row_id for lot in index.pools["test"]["2011-03-02"]] == [0, 1]
    assert [lot.source_row_id for lot in index.pools["test"]["2011-03-03"]] == [2]


# --- 候选池与合格日期 -------------------------------------------------------


def test_pool_preserves_file_row_order(lots_csv):
    index = L.load_index(lots_csv)
    for pool in index.pools["train"].values():
        ids = [lot.source_row_id for lot in pool]
        assert ids == sorted(ids)
        assert len(pool) == FIXTURE_LOTS_PER_DAY


def test_eligible_dates_excludes_pools_smaller_than_n(lots_csv):
    """第 4 节：不足 n 个批次的日期不合格；不得在实例内复制行凑数。"""
    index = L.load_index(lots_csv)
    assert len(index.eligible("train", FIXTURE_LOTS_PER_DAY)) == 7      # 每天正好 24 条
    assert index.eligible("train", FIXTURE_LOTS_PER_DAY + 1) == []      # 一天也凑不出 25 条
    assert index.eligible("train", 4) == index.date_splits["train"]


def test_eligible_dates_are_sorted(lots_csv):
    index = L.load_index(lots_csv)
    days = index.eligible("test", 4)
    assert days == sorted(days)
    assert len(days) == 2


def test_load_index_rejects_a_wrong_column_set(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="列"):
        L.load_index(path)


# --- E4b 阈值 ---------------------------------------------------------------


def test_tail_thresholds_ignore_rows_they_were_not_given(tmp_path):
    """阈值必须只由**传入的**批次定，不是由整张表定。

    构造两个日期：早的那天（会成为 train）quantity 全在 1..3，晚的那天
    （会成为 test）全是 1000。用 train 段算阈值必然落在 1..3；用全表算就会
    跳到 1000。两次调用一对比，「只看传入的行」就是一条会失败的性质——
    若实现偷偷去读了全表，第一次调用就会背叛自己。
    """
    rows = [(i, 700000 + i, "2011-01-01", 1 + i % 3, "1.00", 100 * (1 + i % 3)) for i in range(70)]
    rows += [(70 + i, 700070 + i, "2011-02-01", 1000, "9.99", 999000) for i in range(30)]
    index = L.load_index(write_lots_csv(tmp_path / "t.csv", rows))

    train_lots = [lot for pool in index.pools["train"].values() for lot in pool]
    assert len(train_lots) == 70 and len(index.lots) == 100

    on_train = L.tail_thresholds(train_lots, quantile=0.99)
    on_all = L.tail_thresholds(index.lots, quantile=0.99)
    assert int(on_train["quantity_w"]) <= 3      # 没被 1000 拉走
    assert int(on_all["quantity_w"]) == 1000     # 全表算就会是 1000
    assert on_train["basis"] == "train-split lots only"
    assert on_train["quantile"] == 0.99


def test_tail_thresholds_are_deterministic_and_exact_for_prices():
    lots = [
        L.Lot(i, f"7{i:05d}", "2011-01-01", 1, Decimal(p), 100)
        for i, p in enumerate(["1.00", "2.00", "3.00", "4.00", "5.00"])
    ]
    assert L.tail_thresholds(lots, quantile=0.99) == L.tail_thresholds(lots, quantile=0.99)
    # 最近秩法：ceil(0.99*5)-1 = 4，即最大值
    assert L.tail_thresholds(lots, quantile=0.99)["unit_price"] == "5.00"


def test_tail_thresholds_reject_an_empty_input():
    with pytest.raises(ValueError):
        L.tail_thresholds([])


def test_quantile_index_is_nearest_rank_and_clamped():
    assert L.quantile_index(5, 0.99) == 4
    assert L.quantile_index(100, 0.99) == 98
    assert L.quantile_index(1, 0.99) == 0
    assert L.quantile_index(3, 0.0) == 0
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_lots.py -p no:cacheprovider --basetemp="$TEMP/ap_t2" -q 2>&1 | tail -5
```

预期：`ModuleNotFoundError: No module named 'experiment.data.assemble'`。

- [ ] **Step 4: 写实现**

创建空文件 `experiment/data/assemble/__init__.py`。

创建 `experiment/data/assemble/lots.py`：

```python
"""把 ``lots_clean.csv`` 变成按日期分区的候选池。

只做数据变换：读表、定发票的最早日期、切分日期、建池、筛合格日期、算阈值。
不抽样、不落盘、不持全局状态。

依据 EXPERIMENT_PLAN.md 第 4 节。
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple, Sequence

from experiment.data.spec import (
    LOTS_COLUMNS,
    SPLITS,
    SPLIT_DATE_FRACTIONS,
    TAIL_QUANTILE,
)


class Lot(NamedTuple):
    """一条可用的批次。

    ``unit_price`` 保留 ``Decimal``：它是 E4b 阈值的输入，而阈值会被冻进
    manifest 并用于事后标记，浮点表示会让「同一份数据两次算出不同阈值」
    成为可能。``invoice_no`` 保留字符串：它是标识，不是数量。
    """

    source_row_id: int
    invoice_no: str
    invoice_date: str
    quantity_w: int
    unit_price: Decimal
    value_pence: int


class LotIndex(NamedTuple):
    """一次装表的全部派生结果，供 real.py 与 merge.py 消费。"""

    lots: list[Lot]
    first_dates: dict[str, str]
    date_splits: dict[str, list[str]]
    pools: dict[str, dict[str, list[Lot]]]

    def eligible(self, split: str, n: int) -> list[str]:
        """该 split 中批次数不少于 ``n`` 的日期，按日期升序。

        第 4 节：一个日期不足 ``n`` 个可用批次即不合格，且**不得在实例内
        复制行来凑数**——所以这里是硬筛选，不是「尽量」。
        """
        return sorted(day for day, pool in self.pools[split].items() if len(pool) >= n)


def read_lots(path: Path) -> list[Lot]:
    """按文件行序读回批次表，并校验列集合与规格一致。

    列名或列序变了就说明上游改了结构，此时继续往下算只会产出看似正常的
    错数据。与 ``collect.run.read_workbook`` 同一套做法。
    """
    lots: list[Lot] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != LOTS_COLUMNS:
            raise ValueError(f"批次表列与规格不符：期望 {LOTS_COLUMNS}，实际 {header}")
        for row in reader:
            lots.append(
                Lot(
                    source_row_id=int(row["source_row_id"]),
                    invoice_no=row["invoice_no"],
                    invoice_date=row["invoice_date"],
                    quantity_w=int(row["quantity_w"]),
                    unit_price=Decimal(row["unit_price"]),
                    value_pence=int(row["value_pence"]),
                )
            )
    return lots


def invoice_first_dates(lots: Sequence[Lot]) -> dict[str, str]:
    """发票号 -> 该发票最早一笔交易的日历日。

    一张发票可能跨午夜（第 4 节）。全部批次随发票走，所以先算出发票的归属日，
    再据此给批次定日期。
    """
    first: dict[str, str] = {}
    for lot in lots:
        day = lot.invoice_date[:10]
        current = first.get(lot.invoice_no)
        if current is None or day < current:
            first[lot.invoice_no] = day
    return first


def split_dates(
    days: Sequence[str],
    fractions: tuple[float, float] = SPLIT_DATE_FRACTIONS,
) -> dict[str, list[str]]:
    """把去重排序后的日期切成 train/validation/test（第 4 节）。

    两个边界都**下取整**：``floor(f0*D)`` 与 ``floor(f1*D)``。后者是
    validation 的**上界（开区间）**，不是它的天数——validation 的天数是两者之差。

    Args:
        days: 全部发票的最早日期，可含重复、可不排序
        fractions: 两个边界占比，默认取冻结常量；测试可传 ``(0.0, 0.0)``
            把全部日期归入 test
    """
    ordered = sorted(set(days))
    if not ordered:
        return {split: [] for split in SPLITS}
    low = math.floor(fractions[0] * len(ordered))
    high = math.floor(fractions[1] * len(ordered))
    return {
        "train": ordered[:low],
        "validation": ordered[low:high],
        "test": ordered[high:],
    }


def pools_by_split(
    lots: Sequence[Lot],
    first_dates: dict[str, str],
    date_splits: dict[str, list[str]],
) -> dict[str, dict[str, list[Lot]]]:
    """``split -> 发票归属日 -> 该日全部批次``，池内保持文件行序。

    **桶键是发票的最早日期，不是批次自己的 ``invoice_date``**（第 4 节 /
    设计文档 6.1：全部 lot 随发票走）。跨午夜的发票两笔都要落进「最早那天」
    的池子；若按批次自己的日期分桶，池键就会与 ``date_splits`` 的日期集合
    脱节，``eligible`` 会报出这个 split 并不拥有的日期。

    行序是冻结语义：抽样用 rng 在池内取下标，池的顺序一变，抽出的批次就变。
    """
    split_of_date = {day: split for split, days in date_splits.items() for day in days}
    pools: dict[str, dict[str, list[Lot]]] = {split: defaultdict(list) for split in SPLITS}
    for lot in lots:
        day = first_dates[lot.invoice_no]
        pools[split_of_date[day]][day].append(lot)
    return {split: dict(by_day) for split, by_day in pools.items()}


def load_index(
    path: Path,
    fractions: tuple[float, float] = SPLIT_DATE_FRACTIONS,
) -> LotIndex:
    """一次装表：读表、定发票日期、切分、建池。"""
    lots = read_lots(path)
    first_dates = invoice_first_dates(lots)
    date_splits = split_dates(list(first_dates.values()), fractions)
    pools = pools_by_split(lots, first_dates, date_splits)
    return LotIndex(lots, first_dates, date_splits, pools)


def quantile_index(count: int, quantile: float) -> int:
    """最近秩法的下标，钳到 ``[0, count-1]``。

    显式定义而非用 ``numpy.percentile``：后者的插值方式随版本与参数变化，
    而阈值要跨机器复现。``ceil(q*n)-1`` 是确定性的整数运算。
    """
    if count <= 0:
        raise ValueError("count must be positive")
    return max(0, min(count - 1, math.ceil(quantile * count) - 1))


def tail_thresholds(
    lots: Sequence[Lot],
    quantile: float = TAIL_QUANTILE,
) -> dict[str, object]:
    """E4b 的极值阈值：给定批次的 ``quantity_w`` 与 ``unit_price`` 的分位数。

    第 8 节硬要求阈值**只用训练记录**定。本函数只认传入的批次，调用方负责
    只传训练集那部分——这一点由
    ``test_tail_thresholds_ignore_rows_they_were_not_given`` 钉住。

    价格用 ``Decimal`` 排序而非 float：阈值会被冻进 manifest，浮点会让
    「同一份数据两次算出不同阈值」成为可能。返回值一律是字符串，JSON 无损。
    """
    if not lots:
        raise ValueError("tail_thresholds needs at least one lot")
    weights = sorted(lot.quantity_w for lot in lots)
    prices = sorted(lot.unit_price for lot in lots)
    return {
        "basis": "train-split lots only",
        "quantile": quantile,
        "quantity_w": str(weights[quantile_index(len(weights), quantile)]),
        "unit_price": str(prices[quantile_index(len(prices), quantile)]),
    }
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_lots.py -p no:cacheprovider --basetemp="$TEMP/ap_t2" -q 2>&1 | tail -3
```

预期：`14 passed`。

- [ ] **Step 6: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/assemble/__init__.py experiment/data/assemble/lots.py tests/conftest.py tests/test_assemble_lots.py && git commit -m "feat(data): 批次表切分为按日候选池"
```

---

## Task 3: `real.py` —— 真实实例构造

**Files:**
- Create: `experiment/data/assemble/real.py`
- Test: `tests/test_assemble_real.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_assemble_real.py`：

```python
"""真实实例的抽样、溯源回连、容量公式与重抽。"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from experiment.data.assemble import lots as L
from experiment.data.assemble import real as R
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import MAX_RESAMPLE_ATTEMPT, REAL_SEED_TOKEN

N = 6


@pytest.fixture
def index(lots_csv):
    return L.load_index(lots_csv)


def _table(path: Path) -> dict[int, dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {int(r["source_row_id"]): r for r in csv.DictReader(handle)}


def _pool_of(index, inst):
    """按 source_row_ids 从当日池里取回批次对象。"""
    pool = {lot.source_row_id: lot for lot in index.pools["train"][inst["source_date"]]}
    return [pool[rid] for rid in inst["source_row_ids"]]


# --- 标识与基本形状 ---------------------------------------------------------


def test_real_instance_id_format():
    assert R.real_instance_id("train", 20, 0.25, 7) == "real-train-n20-a25-i0007"
    assert R.real_instance_id("e3_size", 500, 0.75, 0) == "real-e3_size-n500-a75-i0000"


def test_real_instance_pads_index_to_four_digits(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 3)
    assert inst["instance_id"] == f"real-train-n{N}-a25-i0003"


def test_real_instance_has_the_frozen_field_set(index):
    """设计文档 §8.2：真实实例恰好在合成实例的 13 个键之上多三个。"""
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 0)
    assert set(inst) == {
        "alpha", "canonical_hash", "capacity", "family", "instance_id",
        "invoice_nos", "item_order", "n", "resample_attempt", "seed",
        "source", "source_date", "source_row_ids", "stats", "values", "weights",
    }


def test_real_instance_declares_itself_real_and_familyless(index):
    """真实数据没有「族」。不变量：family is None <=> source == "real"。"""
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 0)
    assert inst["source"] == "real"
    assert inst["family"] is None


def test_real_instance_lengths_are_consistent(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.50, 1)
    assert len(inst["weights"]) == len(inst["values"]) == N
    assert len(inst["item_order"]) == N
    assert sorted(inst["item_order"]) == list(range(N))
    assert len(inst["source_row_ids"]) == len(inst["invoice_nos"]) == N
    assert all(w > 0 for w in inst["weights"])
    assert all(v > 0 for v in inst["values"])


# --- 抽样语义 ---------------------------------------------------------------


def test_sample_comes_from_a_single_eligible_date(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 0)
    assert inst["source_date"] in index.eligible("train", N)
    pool_ids = {lot.source_row_id for lot in index.pools["train"][inst["source_date"]]}
    assert set(inst["source_row_ids"]) <= pool_ids


def test_sampled_rows_are_distinct(index):
    """第 4 节：无放回抽 n 个不同批次。"""
    for i in range(20):
        inst = R.build_real_instance(index, "train", "train", N, 0.25, i)
        ids = inst["source_row_ids"]
        assert len(set(ids)) == len(ids) == N


def test_sampled_rows_are_not_just_the_pool_prefix(index):
    """防退化：若实现偷懒取池子前 n 条，下标就会恒等于 (0,1,...,n-1)。

    夹具每天 24 条、抽 6 条，C(24,6)=134596 种组合里「正好是前 6 条」只占
    一种；连抽 30 次都命中前缀的概率可以忽略。这条断言让「取前缀」这种
    退化实现必然失败。
    """
    hits = 0
    for i in range(30):
        inst = R.build_real_instance(index, "train", "train", N, 0.25, i)
        pool = index.pools["train"][inst["source_date"]]
        positions = sorted(
            next(j for j, lot in enumerate(pool) if lot.source_row_id == rid)
            for rid in inst["source_row_ids"]
        )
        hits += positions == list(range(N))
    assert hits == 0, "抽样看起来只是取池子前 n 条"


def test_row_order_within_the_instance_follows_the_rng(index):
    """weights/values 必须与 source_row_ids 同序，顺序由 items 流决定。"""
    inst = R.build_real_instance(index, "train", "train", N, 0.25, 2)
    picked = _pool_of(index, inst)
    assert inst["weights"] == [lot.quantity_w for lot in picked]
    assert inst["values"] == [lot.value_pence for lot in picked]
    assert inst["invoice_nos"] == [lot.invoice_no for lot in picked]


# --- 溯源回连（本阶段最重要的一条检查）--------------------------------------


def test_sampled_values_join_back_to_the_lots_table(lots_csv, index):
    """按 source_row_ids 回连批次表，weights/values/invoice 必须逐位相符。

    这条让「抽样被篡改」无处藏身：任何一只手改动都会在回连时暴露。
    """
    table = _table(lots_csv)
    for i in range(5):
        inst = R.build_real_instance(index, "train", "train", N, 0.75, i)
        for rid, w, v, invoice in zip(
            inst["source_row_ids"], inst["weights"], inst["values"], inst["invoice_nos"]
        ):
            row = table[rid]
            assert w == int(row["quantity_w"])
            assert v == int(row["value_pence"])
            assert invoice == row["invoice_no"]
            assert inst["source_date"] == row["invoice_date"][:10]


def test_join_back_would_notice_a_permuted_weights_array(lots_csv, index):
    """回连检查必须是「按位置」比对的，不能是「按集合」。

    把某个实例的 weights 循环右移一位：按集合比对仍然相等，按位置比对必然
    不等——这条断言证明上面那条检查是位置敏感的，不是形同虚设。
    """
    table = _table(lots_csv)
    inst = R.build_real_instance(index, "train", "train", N, 0.75, 1)
    rotated = inst["weights"][1:] + inst["weights"][:1]
    by_position = all(
        w == int(table[rid]["quantity_w"]) for rid, w in zip(inst["source_row_ids"], rotated)
    )
    assert not by_position, "位置比对没有区分度，回连检查形同虚设"


# --- 容量与种子 -------------------------------------------------------------


def test_capacity_matches_the_frozen_formula(index):
    for alpha in (0.25, 0.50, 0.75):
        inst = R.build_real_instance(index, "train", "train", N, alpha, 0)
        assert inst["capacity"] == math.floor(alpha * sum(inst["weights"]))


def test_seed_matches_derivation_and_uses_the_real_token(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.50, 9)
    assert inst["seed"] == derive_seed("train", REAL_SEED_TOKEN, N, 0.50, 9, 0, "items")


def test_item_order_comes_from_the_order_stream(index):
    inst = R.build_real_instance(index, "train", "train", N, 0.50, 0)
    rng = np.random.default_rng(derive_seed("train", REAL_SEED_TOKEN, N, 0.50, 0, 0, "order"))
    assert inst["item_order"] == rng.permutation(N).tolist()


def test_seed_stream_can_be_overridden_for_the_stress_size_set(index):
    """E3 真实用 test 的日期池，但种子键的 split 槽位是保留流记号。"""
    inst = R.build_real_instance(
        index, "e3_size", "test", 4, 0.25, 0, seed_stream="stream_size"
    )
    assert inst["source_date"] in index.eligible("test", 4)
    assert inst["seed"] == derive_seed("stream_size", REAL_SEED_TOKEN, 4, 0.25, 0, 0, "items")
    assert inst["instance_id"].startswith("real-e3_size-n4-")


def test_real_instances_are_deterministic(index):
    a = R.build_real_instance(index, "train", "train", N, 0.25, 4)
    b = R.build_real_instance(index, "train", "train", N, 0.25, 4)
    assert a == b


# --- 空池与重抽 -------------------------------------------------------------


def test_sampling_raises_when_no_date_is_eligible(lots_csv):
    index = L.load_index(lots_csv)
    with pytest.raises(ValueError, match="n=25"):
        R.build_real_instance(index, "train", "train", 25, 0.25, 0)


def test_fill_cell_returns_exactly_quota_and_updates_seen(index):
    seen: set[str] = set()
    insts = R.fill_cell(
        lambda attempt: R.build_real_instance(index, "train", "train", N, 0.25, 0, attempt),
        quota=5,
        seen=seen,
    )
    assert len(insts) == 5
    assert {i["canonical_hash"] for i in insts} == seen
    assert len(seen) == 5


def test_fill_cell_resamples_when_the_hash_is_already_taken(index):
    known = R.build_real_instance(index, "train", "train", N, 0.25, 0)["canonical_hash"]
    insts = R.fill_cell(
        lambda attempt: R.build_real_instance(index, "train", "train", N, 0.25, 0, attempt),
        quota=1,
        seen={known},
    )
    assert insts[0]["canonical_hash"] != known
    assert insts[0]["resample_attempt"] >= 1


def test_fill_cell_raises_instead_of_looping_forever(index):
    """把上限设为 0，任何一次碰撞都立刻触顶——宁可报错也不死循环。"""
    known = R.build_real_instance(index, "train", "train", N, 0.25, 0)["canonical_hash"]
    with pytest.raises(RuntimeError, match="resample"):
        R.fill_cell(
            lambda attempt: R.build_real_instance(index, "train", "train", N, 0.25, 0, attempt),
            quota=1,
            seen={known},
            max_attempt=0,
        )


def test_max_resample_attempt_default_is_the_frozen_constant():
    import inspect

    default = inspect.signature(R.fill_cell).parameters["max_attempt"].default
    assert default == MAX_RESAMPLE_ATTEMPT


def test_build_real_cell_matches_the_quota_and_labels_the_ids(index):
    seen: set[str] = set()
    insts = R.build_real_cell(index, "train", "train", N, 0.25, 4, seen)
    assert len(insts) == 4
    assert [i["instance_id"] for i in insts] == [
        f"real-train-n{N}-a25-i{i:04d}" for i in range(4)
    ]
    assert len({i["canonical_hash"] for i in insts}) == 4
    assert len(seen) == 4
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_real.py -p no:cacheprovider --basetemp="$TEMP/ap_t3" -q 2>&1 | tail -5
```

预期：`ImportError: cannot import name 'real'`。

- [ ] **Step 3: 写实现**

创建 `experiment/data/assemble/real.py`：

```python
"""真实实例构造：从按日候选池里无放回抽样成 0/1 背包实例。

物品 = 一条批次（不可分割的履约批次），资源 = ``quantity_w``，
价值 = ``value_pence``，容量 = ``floor(alpha * sum(w))``。
依据 EXPERIMENT_PLAN.md 第 3.1 节与第 4 节。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from experiment.data.assemble.lots import Lot, LotIndex
from experiment.data.generate.families import capacity_for, compute_stats
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import MAX_RESAMPLE_ATTEMPT, REAL_SEED_TOKEN, alpha_token


def real_instance_id(name: str, n: int, alpha: float, index: int) -> str:
    """如 ``real-train-n20-a25-i0007``、``real-e3_size-n500-a75-i0000``。

    ``name`` 是落盘数据集名（核心集用 split 本身，E3 用 ``e3_size``），
    与种子键里的 split 槽位是两回事——应力集里二者不同。
    """
    return f"real-{name}-n{n}-{alpha_token(alpha)}-i{index:04d}"


def sample_lots(
    lot_index: LotIndex,
    date_split: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
    seed_stream: str | None = None,
) -> tuple[str, list[Lot]]:
    """选出一个日期与该日池内的 ``n`` 个批次。

    **rng 调用次序是冻结语义**：先均匀取一天，再在该日池内无放回抽 ``n`` 个。
    调换这两步会改变全部真实实例的内容。

    Args:
        date_split: 从哪个 split 的日期池里抽（决定候选池）
        seed_stream: 种子键的 split 槽位；``None`` 表示与 ``date_split`` 同名。
            E3 真实用它把种子挪到保留流上，同时仍从 test 的日期池里抽。
    """
    stream_key = date_split if seed_stream is None else seed_stream
    days = lot_index.eligible(date_split, n)
    if not days:
        raise ValueError(
            f"{date_split} 中没有任何日期能提供 n={n} 个批次；"
            "第 4 节禁止在实例内复制行凑数"
        )
    rng = np.random.default_rng(
        derive_seed(stream_key, REAL_SEED_TOKEN, n, alpha, index, attempt, "items")
    )
    day = days[int(rng.integers(len(days)))]
    pool = lot_index.pools[date_split][day]
    picked = rng.choice(len(pool), size=n, replace=False)
    return day, [pool[int(position)] for position in picked]


def build_real_instance(
    lot_index: LotIndex,
    name: str,
    date_split: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
    seed_stream: str | None = None,
) -> dict:
    """构造一个真实实例的完整记录。

    ``items`` 流决定抽哪些批次与它们的顺序，``order`` 流独立决定物品排列；
    两条流互不影响，与合成实例的约定完全一致。
    """
    stream_key = date_split if seed_stream is None else seed_stream
    day, lots = sample_lots(lot_index, date_split, n, alpha, index, attempt, seed_stream)

    weights = np.array([lot.quantity_w for lot in lots], dtype=np.int64)
    values = np.array([lot.value_pence for lot in lots], dtype=np.int64)
    capacity = capacity_for(alpha, weights)

    order_rng = np.random.default_rng(
        derive_seed(stream_key, REAL_SEED_TOKEN, n, alpha, index, attempt, "order")
    )

    return {
        "alpha": alpha,
        "canonical_hash": canonical_hash(n, capacity, weights, values),
        "capacity": capacity,
        "family": None,
        "instance_id": real_instance_id(name, n, alpha, index),
        "invoice_nos": [lot.invoice_no for lot in lots],
        "item_order": order_rng.permutation(n).tolist(),
        "n": n,
        "resample_attempt": attempt,
        "seed": derive_seed(stream_key, REAL_SEED_TOKEN, n, alpha, index, attempt, "items"),
        "source": "real",
        "source_date": day,
        "source_row_ids": [lot.source_row_id for lot in lots],
        "stats": compute_stats(weights, values),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }


def fill_cell(
    build: Callable[[int], dict],
    quota: int,
    seen: set[str],
    max_attempt: int = MAX_RESAMPLE_ATTEMPT,
) -> list[dict]:
    """反复调用 ``build(attempt)``，直到取满 ``quota`` 条哈希互不相同的实例。

    配额不因重抽改变：重抽只换实例内容，不减少条数。``seen`` 就地更新，
    因此它可以是跨 cell、跨 split 的全局集合。

    Raises:
        RuntimeError: ``attempt`` 超过 ``max_attempt`` 仍未取得唯一哈希。
    """
    instances: list[dict] = []
    for _ in range(quota):
        attempt = 0
        while True:
            instance = build(attempt)
            if instance["canonical_hash"] not in seen:
                break
            attempt += 1
            if attempt > max_attempt:
                raise RuntimeError(
                    f"resample exceeded {max_attempt} attempts for {instance['instance_id']}"
                )
        seen.add(instance["canonical_hash"])
        instances.append(instance)
    return instances


def build_real_cell(
    lot_index: LotIndex,
    name: str,
    date_split: str,
    n: int,
    alpha: float,
    quota: int,
    seen: set[str],
    max_attempt: int = MAX_RESAMPLE_ATTEMPT,
    seed_stream: str | None = None,
) -> list[dict]:
    """构造一个真实 cell 的全部实例。

    同一下标的重抽只递增 ``attempt``——与 ``generate.corpus.build_cell`` 的
    语义一致：重抽改内容，不改身份。
    """
    instances: list[dict] = []
    for index in range(quota):
        instances.extend(
            fill_cell(
                lambda attempt, index=index: build_real_instance(
                    lot_index, name, date_split, n, alpha, index, attempt, seed_stream
                ),
                quota=1,
                seen=seen,
                max_attempt=max_attempt,
            )
        )
    return instances
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_real.py -p no:cacheprovider --basetemp="$TEMP/ap_t3" -q 2>&1 | tail -3
```

预期：`22 passed`。

- [ ] **Step 5: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/assemble/real.py tests/test_assemble_real.py && git commit -m "feat(data): 真实实例构造与溯源回连"
```

---

## Task 4: `stress.py` —— E3 与 E4

**Files:**
- Create: `experiment/data/assemble/stress.py`
- Test: `tests/test_assemble_stress.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_assemble_stress.py`：

```python
"""E4 变体生成器、E3/E4 的形状与配额、流隔离。"""

import numpy as np
import pytest

from experiment.data.assemble import lots as L
from experiment.data.assemble import stress as S
from experiment.data.generate.families import generate_items
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    ALPHAS,
    E3_NAME,
    E3_SIZE,
    E4_NAME,
    E4_SEED_TOKEN,
    E4_SIZE,
    E4_SKEW_HEAVY_DENSITY_RANGE,
    E4_SKEW_HEAVY_WEIGHT_RANGE,
    E4_SKEW_K,
    FAMILIES,
    REAL_SEED_TOKEN,
    SKEW_HEAVY_DENSITY_RANGE,
    SKEW_HEAVY_WEIGHT_RANGE,
    SPLITS,
    STRESS_SIZE_STREAM,
    STRESS_TAIL_STREAM,
    skew_k,
)

N = 20


@pytest.fixture
def index(lots_csv):
    return L.load_index(lots_csv)


# --- E4 变体生成器 ----------------------------------------------------------


def test_e4_variant_has_exactly_k_heavy_weights():
    """第 8 节：k=2 个稀有权重落在 U(500,2000)，其余落在 U(1,20)。

    这是精确计数，不是统计——它同时钉住 k 与权重范围两件事。
    """
    w, _ = S.generate_skewed_e4(N, seed=12345)
    lo, hi = E4_SKEW_HEAVY_WEIGHT_RANGE
    heavy = [int(x) for x in w if lo <= x <= hi]
    light = [int(x) for x in w if x not in heavy]
    assert len(heavy) == E4_SKEW_K == 2
    assert all(1 <= x <= 20 for x in light)
    assert len(light) == N - E4_SKEW_K


def test_e4_variant_has_exactly_k_heavy_densities():
    """密度无法从 v=w*d 反推，所以生成器多返回一个数组供直接检验。"""
    _, _, d = S.generate_skewed_e4(N, seed=999, with_density=True)
    lo, hi = E4_SKEW_HEAVY_DENSITY_RANGE
    heavy = [int(x) for x in d if lo <= x <= hi]
    assert len(heavy) == E4_SKEW_K
    assert all(1 <= x <= 5 for x in d if x not in heavy)


def test_e4_variant_ranges_are_strictly_wider_than_the_frozen_family():
    """变体必须真的更强，否则 E4 就不是「更强的不平衡」。

    三条各管一件事：稀有权的下界抬高、上界抬高、稀有密度的下界高过主语料
    稀有密度的上界。主语料的 skewed 半边已被金样本钉死，这些数字只能是变体
    自己的，不能与主语料相等。
    """
    assert E4_SKEW_HEAVY_WEIGHT_RANGE[0] > SKEW_HEAVY_WEIGHT_RANGE[0]   # 500 > 200
    assert E4_SKEW_HEAVY_WEIGHT_RANGE[1] > SKEW_HEAVY_WEIGHT_RANGE[1]   # 2000 > 500
    assert E4_SKEW_HEAVY_DENSITY_RANGE[0] > SKEW_HEAVY_DENSITY_RANGE[1]  # 100 > 60


def test_e4_variant_is_not_the_frozen_skewed_generator():
    """同一个种子下，变体与既有族必须产出不同的数组。

    若二者相同，说明变体只是把 _skewed 抄了一遍却没换参数。
    """
    seed = 4242
    w_e4, v_e4 = S.generate_skewed_e4(N, seed)
    w_sk, v_sk = generate_items("skewed", N, seed)
    assert not np.array_equal(w_e4, w_sk)
    assert not np.array_equal(v_e4, v_sk)


def test_e4_variant_values_are_weight_times_density():
    w, v, d = S.generate_skewed_e4(N, seed=777, with_density=True)
    assert np.array_equal(v, (w * d).astype(np.int64))


def test_e4_variant_is_deterministic():
    a = S.generate_skewed_e4(N, 31337)
    b = S.generate_skewed_e4(N, 31337)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_frozen_skewed_generator_is_untouched():
    """本任务不得改动 generate/families.py 的 _skewed。

    单抽一个种子直接数稀有项——若有人顺手改了主语料，这里会亮。
    """
    w, _ = generate_items("skewed", N, seed=1)
    lo, hi = SKEW_HEAVY_WEIGHT_RANGE
    heavy = [int(x) for x in w if lo <= x <= hi]
    assert len(heavy) == skew_k(N)


# --- 流隔离 -----------------------------------------------------------------


def test_e4_seed_does_not_collide_with_any_generation_seed():
    """第 8 节「保留独立流」的可观测形式。

    若 E4 复用 split/family 记号，它就会与主集某些实例共用种子。
    """
    e4_seed = derive_seed(STRESS_TAIL_STREAM, E4_SEED_TOKEN, E4_SIZE, 0.25, 0, 0, "items")
    for split in SPLITS:
        for family in FAMILIES:
            assert e4_seed != derive_seed(split, family, E4_SIZE, 0.25, 0, 0, "items")
        assert e4_seed != derive_seed(split, REAL_SEED_TOKEN, E4_SIZE, 0.25, 0, 0, "items")


def test_e3_seed_does_not_collide_with_any_generation_seed():
    e3_seed = derive_seed(STRESS_SIZE_STREAM, REAL_SEED_TOKEN, E3_SIZE, 0.25, 0, 0, "items")
    for split in SPLITS:
        for family in FAMILIES:
            assert e3_seed != derive_seed(split, family, E3_SIZE, 0.25, 0, 0, "items")
        assert e3_seed != derive_seed(split, REAL_SEED_TOKEN, E3_SIZE, 0.25, 0, 0, "items")


# --- E3 合成 ----------------------------------------------------------------


def test_e3_synthetic_shapes_and_labels():
    inst = S.build_e3_synthetic("uniform", 0.25, 0, n=N)
    assert inst["n"] == N
    assert inst["source"] == "synthetic"
    assert inst["family"] == "uniform"
    assert inst["instance_id"] == f"syn-{E3_NAME}-uniform-n{N}-a25-i0000"
    assert len(inst["weights"]) == len(inst["values"]) == N
    assert sorted(inst["item_order"]) == list(range(N))


def test_e3_synthetic_uses_the_size_stream():
    inst = S.build_e3_synthetic("normal", 0.50, 3, n=N)
    assert inst["seed"] == derive_seed(STRESS_SIZE_STREAM, "normal", N, 0.50, 3, 0, "items")


def test_e3_synthetic_defaults_to_the_frozen_size():
    assert S.build_e3_synthetic("uniform", 0.25, 0)["n"] == E3_SIZE == 500


# --- E4 合成 ----------------------------------------------------------------


def test_e4_synthetic_shapes_and_labels():
    inst = S.build_e4_synthetic(0.25, 0, n=N)
    assert inst["n"] == N
    assert inst["source"] == "synthetic"
    assert inst["family"] == E4_SEED_TOKEN
    assert inst["instance_id"] == f"syn-{E4_NAME}-{E4_SEED_TOKEN}-n{N}-a25-i0000"


def test_e4_family_token_is_not_a_frozen_family():
    """下游按 FAMILIES 枚举族的代码必须自动跳过 E4。"""
    assert E4_SEED_TOKEN not in FAMILIES


def test_e4_synthetic_uses_the_tail_stream():
    inst = S.build_e4_synthetic(0.75, 5, n=N)
    assert inst["seed"] == derive_seed(STRESS_TAIL_STREAM, E4_SEED_TOKEN, N, 0.75, 5, 0, "items")


def test_e4_synthetic_carries_exactly_k_heavy_items():
    """从产出的记录本身也能数出 2 个极值——端到端的第二重确认。"""
    inst = S.build_e4_synthetic(0.25, 0, n=N)
    lo, hi = E4_SKEW_HEAVY_WEIGHT_RANGE
    assert sum(1 for w in inst["weights"] if lo <= w <= hi) == E4_SKEW_K


def test_e4_synthetic_defaults_to_the_frozen_size():
    assert S.build_e4_synthetic(0.25, 0)["n"] == E4_SIZE == 200


# --- E3 真实 ----------------------------------------------------------------


def test_e3_real_draws_from_the_test_date_pool(index):
    inst = S.build_e3_real(index, 0.25, 0, n=N)
    assert inst["source"] == "real"
    assert inst["family"] is None
    assert inst["source_date"] in index.eligible("test", N)
    assert inst["instance_id"] == f"real-{E3_NAME}-n{N}-a25-i0000"


def test_e3_real_uses_the_size_stream(index):
    inst = S.build_e3_real(index, 0.50, 2, n=N)
    assert inst["seed"] == derive_seed(STRESS_SIZE_STREAM, REAL_SEED_TOKEN, N, 0.50, 2, 0, "items")


# --- 两个应力集的整体 -------------------------------------------------------


def test_build_e3_matches_the_frozen_quotas(index):
    insts = S.build_e3(index, n=N, synthetic_per_family=2, real_per_alpha=3)
    assert sum(1 for i in insts if i["source"] == "synthetic") == len(FAMILIES) * len(ALPHAS) * 2
    assert sum(1 for i in insts if i["source"] == "real") == len(ALPHAS) * 3


def test_build_e3_without_an_index_emits_no_real_instances():
    """没有批次索引就没有真实半边，但合成半边必须**照样足额**产出。

    只断言 ``all(source == "synthetic")`` 会被空列表满足，因此先钉住条数。
    """
    insts = S.build_e3(None, n=N, synthetic_per_family=1, real_per_alpha=3)
    assert len(insts) == len(FAMILIES) * len(ALPHAS) * 1
    assert all(i["source"] == "synthetic" for i in insts)


def test_build_e4_matches_the_frozen_quota():
    insts = S.build_e4(n=N, per_alpha=2)
    assert len(insts) == len(ALPHAS) * 2
    assert all(i["source"] == "synthetic" for i in insts)
    assert all(i["family"] == E4_SEED_TOKEN for i in insts)


def test_build_e3_and_e4_hashes_are_globally_unique():
    """空结果会让两条唯一性断言同时空转，所以先钉住总数。

    ``build_e3(None, ...)`` 只有合成半边：4 族 × 3 比率 × 2 条；E4 是 3 比率 × 2 条。
    """
    seen: set[str] = set()
    insts = S.build_e3(None, seen=seen, n=N, synthetic_per_family=2, real_per_alpha=2)
    insts += S.build_e4(seen=seen, n=N, per_alpha=2)
    assert len(insts) == len(FAMILIES) * len(ALPHAS) * 2 + len(ALPHAS) * 2
    hashes = [i["canonical_hash"] for i in insts]
    assert len(set(hashes)) == len(hashes)
    assert set(hashes) == seen
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_stress.py -p no:cacheprovider --basetemp="$TEMP/ap_t4" -q 2>&1 | tail -5
```

预期：`ImportError: cannot import name 'stress'`。

- [ ] **Step 3: 写实现**

创建 `experiment/data/assemble/stress.py`：

```python
"""两个应力集：E3（未见过的规模）与 E4（更强的不平衡）。

E3 用 ``n=500`` 检验规模泛化，保留 50:50 来源平衡；E4 用 ``n=200`` 的
强不平衡合成实例检验极值表现。依据 EXPERIMENT_PLAN.md 第 8 节。

**两条保留流的隔离靠字符串记号，不靠 ``STRESS_MASTER_SEEDS`` 里的数字。**
``derive_seed`` 的键里没有那个数字——这是设计文档 §12.1 记录的已知事实。
"""

from __future__ import annotations

import numpy as np

from experiment.data.assemble.lots import LotIndex
from experiment.data.assemble.real import build_real_cell, build_real_instance, fill_cell
from experiment.data.generate.families import capacity_for, compute_stats, generate_items
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    ALPHAS,
    E3_NAME,
    E3_REAL_PER_ALPHA,
    E3_SIZE,
    E3_SYNTHETIC_PER_FAMILY,
    E4_COUNT_PER_ALPHA,
    E4_NAME,
    E4_SEED_TOKEN,
    E4_SIZE,
    E4_SKEW_HEAVY_DENSITY_RANGE,
    E4_SKEW_HEAVY_WEIGHT_RANGE,
    E4_SKEW_K,
    FAMILIES,
    REAL_SEED_TOKEN,
    SKEW_BASE_DENSITY_RANGE,
    SKEW_BASE_WEIGHT_RANGE,
    STRESS_SIZE_STREAM,
    STRESS_TAIL_STREAM,
    alpha_token,
)


def generate_skewed_e4(
    n: int,
    seed: int,
    with_density: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E4 的强不平衡变体生成器（第 8 节）。

    与 ``generate.families._skewed`` 的关系是**另立变体，不是改版**：主语料的
    skewed 半边已被金样本钉死，改它会作废全部既有语料。这里按同一套配方重写，
    只换参数：

    ==========  =============================  ======================
    参数         主语料 skewed                   E4 变体
    ==========  =============================  ======================
    稀有索引数    ``max(1, floor(0.05n+0.5))``   ``k = 2``
    稀有权重      ``U(200,500)``                 ``U(500,2000)``
    稀有密度      ``U(30,60)``                   ``U(100,200)``
    ==========  =============================  ======================

    rng 调用次序与 ``_skewed`` 完全一致（w, d, idx_w, 替换权重, idx_d,
    替换密度），这个次序本身是冻结语义。两处易错点同样适用：两组稀有索引
    独立抽取、**可以重叠**；``v = w * d`` 用替换**之后**的 w 与 d。

    Args:
        with_density: 为 ``True`` 时额外返回密度数组。密度无法从 ``v = w*d``
            反推，而测试需要直接检验稀有密度的个数与范围。
    """
    rng = np.random.default_rng(seed)
    lo_w, hi_w = SKEW_BASE_WEIGHT_RANGE
    lo_d, hi_d = SKEW_BASE_DENSITY_RANGE
    heavy_lo_w, heavy_hi_w = E4_SKEW_HEAVY_WEIGHT_RANGE
    heavy_lo_d, heavy_hi_d = E4_SKEW_HEAVY_DENSITY_RANGE

    w = rng.integers(lo_w, hi_w + 1, size=n, dtype=np.int64)
    d = rng.integers(lo_d, hi_d + 1, size=n, dtype=np.int64)

    k = E4_SKEW_K

    idx_w = rng.choice(n, size=k, replace=False)
    w[idx_w] = rng.integers(heavy_lo_w, heavy_hi_w + 1, size=k, dtype=np.int64)

    idx_d = rng.choice(n, size=k, replace=False)
    d[idx_d] = rng.integers(heavy_lo_d, heavy_hi_d + 1, size=k, dtype=np.int64)

    values = (w * d).astype(np.int64)
    if with_density:
        return w, values, d
    return w, values


def _finish(
    instance_id: str,
    family: str | None,
    source: str,
    n: int,
    alpha: float,
    order_seed: int,
    weights: np.ndarray,
    values: np.ndarray,
    extra: dict,
) -> dict:
    """把一份 (weights, values) 补成完整的合成实例记录。

    两条合成分支共用，避免把 13 个字段抄两遍。``order_seed`` 是 **order 流**
    的种子，不是 items 流的——调用方必须传对，否则排列会与主集的约定不一致。
    """
    capacity = capacity_for(alpha, weights)
    record = {
        "alpha": alpha,
        "canonical_hash": canonical_hash(n, capacity, weights, values),
        "capacity": capacity,
        "family": family,
        "instance_id": instance_id,
        "item_order": np.random.default_rng(order_seed).permutation(n).tolist(),
        "n": n,
        "source": source,
        "stats": compute_stats(weights, values),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }
    record.update(extra)
    return record


def build_e3_synthetic(
    family: str,
    alpha: float,
    index: int,
    attempt: int = 0,
    n: int = E3_SIZE,
) -> dict:
    """E3 的一条合成实例。

    ``n`` 默认是冻结的 500；测试可传小值以便在密闭夹具上跑。种子键的 split
    槽位是保留流记号 ``stream_size``，与主集的任何 cell 都不重合。
    """
    seed = derive_seed(STRESS_SIZE_STREAM, family, n, alpha, index, attempt, "items")
    weights, values = generate_items(family, n, seed)
    order_seed = derive_seed(STRESS_SIZE_STREAM, family, n, alpha, index, attempt, "order")
    return _finish(
        f"syn-{E3_NAME}-{family}-n{n}-{alpha_token(alpha)}-i{index:04d}",
        family,
        "synthetic",
        n,
        alpha,
        order_seed,
        weights,
        values,
        {"resample_attempt": attempt, "seed": seed},
    )


def build_e3_real(
    lot_index: LotIndex,
    alpha: float,
    index: int,
    attempt: int = 0,
    n: int = E3_SIZE,
) -> dict:
    """E3 的一条真实实例：从 **test** 的日期池抽，种子在保留流上。

    第 8 节要求 E3 真实只用合格的 test 源日期，且与主集的 canonical hash
    不相交——后者由 ``seen`` 的全局性保证，不是本函数的事。
    """
    return build_real_instance(
        lot_index, E3_NAME, "test", n, alpha, index, attempt, seed_stream=STRESS_SIZE_STREAM
    )


def build_e4_synthetic(
    alpha: float,
    index: int,
    attempt: int = 0,
    n: int = E4_SIZE,
) -> dict:
    """E4 的一条合成实例。族记号是 ``skewed_e4``，不在 FAMILIES 中。"""
    seed = derive_seed(STRESS_TAIL_STREAM, E4_SEED_TOKEN, n, alpha, index, attempt, "items")
    weights, values = generate_skewed_e4(n, seed)
    order_seed = derive_seed(
        STRESS_TAIL_STREAM, E4_SEED_TOKEN, n, alpha, index, attempt, "order"
    )
    return _finish(
        f"syn-{E4_NAME}-{E4_SEED_TOKEN}-n{n}-{alpha_token(alpha)}-i{index:04d}",
        E4_SEED_TOKEN,
        "synthetic",
        n,
        alpha,
        order_seed,
        weights,
        values,
        {"resample_attempt": attempt, "seed": seed},
    )


def build_e3(
    lot_index: LotIndex | None,
    seen: set[str] | None = None,
    n: int = E3_SIZE,
    synthetic_per_family: int = E3_SYNTHETIC_PER_FAMILY,
    real_per_alpha: int = E3_REAL_PER_ALPHA,
) -> list[dict]:
    """E3 全集：合成在前（族 × 比率），真实在后（比率）。

    ``lot_index`` 为 ``None`` 时只造合成部分，供不涉及真实数据的测试使用。
    """
    book = set() if seen is None else seen
    instances: list[dict] = []
    for family in FAMILIES:
        for alpha in ALPHAS:
            for index in range(synthetic_per_family):
                instances.extend(
                    fill_cell(
                        lambda attempt, f=family, a=alpha, i=index: build_e3_synthetic(
                            f, a, i, attempt, n
                        ),
                        quota=1,
                        seen=book,
                    )
                )
    if lot_index is not None:
        for alpha in ALPHAS:
            instances.extend(
                build_real_cell(
                    lot_index,
                    E3_NAME,
                    "test",
                    n,
                    alpha,
                    real_per_alpha,
                    book,
                    seed_stream=STRESS_SIZE_STREAM,
                )
            )
    return instances


def build_e4(
    seen: set[str] | None = None,
    n: int = E4_SIZE,
    per_alpha: int = E4_COUNT_PER_ALPHA,
) -> list[dict]:
    """E4 全集：每比率 ``per_alpha`` 条合成实例。"""
    book = set() if seen is None else seen
    instances: list[dict] = []
    for alpha in ALPHAS:
        for index in range(per_alpha):
            instances.extend(
                fill_cell(
                    lambda attempt, a=alpha, i=index: build_e4_synthetic(a, i, attempt, n),
                    quota=1,
                    seen=book,
                )
            )
    return instances
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_stress.py -p no:cacheprovider --basetemp="$TEMP/ap_t4" -q 2>&1 | tail -3
```

预期：`23 passed`。

- [ ] **Step 5: 全量测试**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t4all" -q 2>&1 | tail -3
```

预期：`244 passed`，0 failed、0 skipped。

- [ ] **Step 6: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/assemble/stress.py tests/test_assemble_stress.py && git commit -m "feat(data): E3/E4 应力集与独立流"
```

---

## Task 5: `merge.py` —— 合并、50:50、全局去重、落盘

**Files:**
- Create: `experiment/data/assemble/merge.py`
- Create: `experiment/data/assemble/manifest.py`（Task 5 交付 provenance 与写出函数，派生段留给 Task 6）
- Test: `tests/test_assemble_merge.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_assemble_merge.py`：

```python
"""合并顺序、逐 cell 50:50、全局去重、字节稳定与端到端。

所有测试走密闭夹具：夹具语料的 cell 集合与真实语料一致（n ∈ SIZES），
因此这里只出现 n=20 这类真实存在的 cell。
"""

import hashlib
import json
from pathlib import Path

import pytest

from experiment.data import spec
from experiment.data.assemble import lots as L
from experiment.data.assemble import merge as M
from experiment.data.spec import FAMILIES

CELL = (20, 0.25)

#: cell 级的配额是**标量**。``assemble_split`` 的 ``synthetic_quota`` 是
#: 「每个合成族每 cell 的条数」；夹具每个 cell 只提供 1 条，故取 1。
#: ``real_quota`` 必须等于族数 × 每族条数，否则 50:50 闸门会报错。
#:
#: 前缀 ``CELL_`` 是刻意加的：``spec.REAL_QUOTA``（= ``{"train": 500, …}``）说的是
#: 「每个 split 的真实条数」，与本处的「每个 cell 4 条」同名而不同物，量级还差两个
#: 数量级。不带前缀，读的人很容易把 4 当成 500。
CELL_SYN_QUOTA = 1
CELL_REAL_QUOTA = len(FAMILIES) * CELL_SYN_QUOTA

#: run 级的配额是**按 split 索引的映射**，只给 ``run_assembly`` 用。
#: 两个层级同名不同形，混用会让 ``read_cell(path)[:quota]`` 拿字典去切片。
QUOTAS = {
    "synthetic_quota": {split: CELL_SYN_QUOTA for split in spec.SPLITS},
    "real_quota": {split: CELL_REAL_QUOTA for split in spec.SPLITS},
}


@pytest.fixture
def index(lots_csv):
    return L.load_index(lots_csv)


def _read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


# --- 合成半边是「读回来」而不是「重新生成」----------------------------------


def test_read_cell_preserves_file_order(fixture_corpus):
    path = fixture_corpus / "train" / "uniform_n20_a25.jsonl"
    instances = M.read_cell(path)
    assert [i["instance_id"] for i in instances] == ["syn-train-uniform-n20-a25-i0000"]


def test_synthetic_cell_path_points_into_the_generation_tree(fixture_corpus):
    path = M.synthetic_cell_path(fixture_corpus, "train", "uniform", 20, 0.25)
    assert path.is_file()
    assert path.name == "uniform_n20_a25.jsonl"


def test_synthetic_half_is_identical_to_the_generation_corpus(fixture_corpus, index):
    """合并文件里的合成记录必须与生成阶段的语料逐字节同源。

    「重新生成一遍」与「读回来」在数学上等价，但前者会让合并产物依赖当前
    代码而非已审计的语料。这条断言钉住后者。
    """
    source = M.read_cell(M.synthetic_cell_path(fixture_corpus, "train", "uniform", 20, 0.25))
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    picked = [i for i in merged if i["instance_id"] == source[0]["instance_id"]]
    assert len(picked) == 1
    assert picked[0] == source[0]


# --- 顺序与 50:50 -----------------------------------------------------------


def test_cell_order_is_families_then_real(index, fixture_corpus):
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    assert [i["source"] for i in merged] == ["synthetic"] * 4 + ["real"] * 4
    assert [i["family"] for i in merged[:4]] == list(FAMILIES)
    assert all(i["family"] is None for i in merged[4:])


def test_every_cell_is_fifty_fifty(index, fixture_corpus):
    _, cells = M.assemble_split(
        "train", index, fixture_corpus, cells=[(20, 0.25), (20, 0.50)], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA
    )
    assert len(cells) == 2
    for cell in cells:
        assert cell["synthetic"] == cell["real"] == 4


def test_raises_when_the_two_sides_do_not_match(index, fixture_corpus):
    """50:50 检查必须是活的：配额不匹配时装配必须报错，而不是静默产出。"""
    with pytest.raises(ValueError, match="50:50"):
        M.assemble_split(
            "train", index, fixture_corpus, cells=[CELL],
            synthetic_quota=1, real_quota=5,
        )


def test_real_instances_use_the_family_count_as_the_quota(index, fixture_corpus):
    """真实半边没有族维度，其条数必须等于合成半边（族数 × 每族条数）。

    每族条数只能取 ``CELL_SYN_QUOTA``：夹具每个 cell 只有 1 条合成实例
    （``run_generation(quota_override=1)``），要求 3 条就会先被 50:50 闸门拦下。
    两边都断言，才能同时钉住「合成半边真的按族数产出」与「真实半边对齐它」。
    """
    merged, _ = M.assemble_split(
        "train", index, fixture_corpus, cells=[CELL],
        synthetic_quota=CELL_SYN_QUOTA, real_quota=len(FAMILIES) * CELL_SYN_QUOTA,
    )
    assert sum(1 for i in merged if i["source"] == "synthetic") == len(FAMILIES) * CELL_SYN_QUOTA
    assert sum(1 for i in merged if i["source"] == "real") == len(FAMILIES) * CELL_SYN_QUOTA


# --- 全局去重 ---------------------------------------------------------------


def test_all_hashes_in_the_merged_output_are_unique(index, fixture_corpus):
    merged, _ = M.assemble_split(
        "train", index, fixture_corpus, cells=[(20, 0.25), (20, 0.50)], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA
    )
    hashes = [i["canonical_hash"] for i in merged]
    assert len(set(hashes)) == len(hashes)


def test_seen_accumulates_across_cells_and_splits(index, fixture_corpus):
    seen: set[str] = set()
    M.assemble_split("train", index, fixture_corpus, cells=[CELL], seen=seen, synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    M.assemble_split("validation", index, fixture_corpus, cells=[CELL], seen=seen, synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    assert len(seen) == 16


def test_synthetic_hash_colliding_with_seen_raises(index, fixture_corpus):
    """合成半边一旦与已装配的实例重复，必须报错而不是被静默吞掉。

    传入上一次装配的全部哈希即可制造这种冲突——这条断言证明那道检查是活的。
    """
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    with pytest.raises(ValueError, match="合成"):
        M.assemble_split(
            "train",
            index,
            fixture_corpus,
            cells=[CELL],
            seen={i["canonical_hash"] for i in merged},
            synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA,
        )


# --- 上游 provenance --------------------------------------------------------


def _upstream_tree(tmp_path: Path) -> tuple[Path, Path]:
    """一份带 collect_report.json 的批次表与一个带 manifest.json 的生成目录。"""
    real = tmp_path / "real"
    real.mkdir(parents=True, exist_ok=True)
    lots = real / "lots_clean.csv"
    lots.write_bytes(b"source_row_id,invoice_no\n0,1\n")
    (real / "collect_report.json").write_bytes(
        json.dumps(
            {
                "kept_rows": 1,
                "source": {
                    "url": "https://example.invalid/x.zip",
                    "xlsx_bytes": 42,
                    "xlsx_sha256": "a" * 64,
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    gen = tmp_path / "generation"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "manifest.json").write_bytes(
        json.dumps({"generator_fingerprint": "b" * 64, "schema_version": 1}).encode("utf-8")
    )
    return lots, gen


def test_upstream_provenance_hashes_every_file_it_cites(tmp_path):
    lots, gen = _upstream_tree(tmp_path)
    prov = M.upstream_provenance(lots, gen)
    assert prov["lots_clean"]["sha256"] == hashlib.sha256(lots.read_bytes()).hexdigest()
    assert prov["lots_clean"]["bytes"] == len(lots.read_bytes())
    assert prov["lots_clean"]["columns"] == list(spec.LOTS_COLUMNS)
    assert prov["collect_report"]["sha256"] == hashlib.sha256(
        (lots.parent / "collect_report.json").read_bytes()
    ).hexdigest()
    assert prov["generation_manifest"]["sha256"] == hashlib.sha256(
        (gen / "manifest.json").read_bytes()
    ).hexdigest()


def test_upstream_provenance_reads_the_values_instead_of_copying_them(tmp_path):
    """`url`/`xlsx_sha256` 等必须来自 collect_report.json 的**内容**。

    写死一份副本就会在上游换文件后继续说谎——这正是本阶段的病灶。
    """
    lots, gen = _upstream_tree(tmp_path)
    prov = M.upstream_provenance(lots, gen)
    assert prov["workbook"]["url"] == "https://example.invalid/x.zip"
    assert prov["workbook"]["bytes"] == 42
    assert prov["workbook"]["sha256"] == "a" * 64
    assert prov["generation_manifest"]["generator_fingerprint"] == "b" * 64
    assert prov["lots_clean"]["rows"] == 1
    assert prov["units"] == {"value": "value_pence", "weight": "quantity_w"}
    assert prov["cleaning_rules"] == list(spec.CLEANING_RULES)


def test_upstream_provenance_follows_an_edited_report(tmp_path):
    """改掉报告里的 url，provenance 必须跟着变。"""
    lots, gen = _upstream_tree(tmp_path)
    report = lots.parent / "collect_report.json"
    doc = json.loads(report.read_text(encoding="utf-8"))
    doc["source"]["url"] = "https://example.invalid/CHANGED.zip"
    report.write_bytes(json.dumps(doc, sort_keys=True).encode("utf-8"))
    assert M.upstream_provenance(lots, gen)["workbook"]["url"].endswith("CHANGED.zip")


def test_upstream_provenance_reports_missing_files_as_none(tmp_path):
    """上游文件缺失时摘要为 None，不假装有。"""
    prov = M.upstream_provenance(tmp_path / "nothing.csv", tmp_path / "nowhere")
    assert prov["lots_clean"]["sha256"] is None
    assert prov["collect_report"]["sha256"] is None
    assert prov["generation_manifest"]["generator_fingerprint"] is None


# --- 落盘 -------------------------------------------------------------------


def test_write_split_writes_lf_jsonl_and_returns_the_digest(index, fixture_corpus, tmp_path):
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    path = tmp_path / "train" / "instances.jsonl"
    digest = M.write_split(path, merged)
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert digest == hashlib.sha256(raw).hexdigest()
    assert len(_read(path)) == 8


def test_write_split_is_byte_identical_on_rerun(index, fixture_corpus, tmp_path):
    merged, _ = M.assemble_split("train", index, fixture_corpus, cells=[CELL], synthetic_quota=CELL_SYN_QUOTA, real_quota=CELL_REAL_QUOTA)
    a, b = tmp_path / "a" / "instances.jsonl", tmp_path / "b" / "instances.jsonl"
    M.write_split(a, merged)
    M.write_split(b, merged)
    assert a.read_bytes() == b.read_bytes()


# --- 端到端 -----------------------------------------------------------------


def _run(tmp_path, lots_csv, fixture_corpus, **overrides):
    kwargs = dict(
        out_dir=tmp_path / "processed",
        lots_path=lots_csv,
        generation_dir=fixture_corpus,
        manifest_path=tmp_path / "data_manifest.json",
        config_path=tmp_path / "experiment_config.yaml",
        cells=[CELL],
        include_stress=False,
        **QUOTAS,
    )
    kwargs.update(overrides)
    return M.run_assembly(**kwargs)


def test_run_assembly_writes_every_artifact(lots_csv, fixture_corpus, tmp_path):
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    for split in ("train", "validation", "test"):
        assert (tmp_path / "processed" / split / "instances.jsonl").is_file()
    assert (tmp_path / "data_manifest.json").is_file()
    assert (tmp_path / "experiment_config.yaml").is_file()
    assert set(manifest["counts"]) == {"train", "validation", "test"}
    assert manifest["counts"]["train"] == {"synthetic": 4, "real": 4, "total": 8}


def test_run_assembly_is_deterministic_across_runs(lots_csv, fixture_corpus, tmp_path):
    _run(tmp_path, lots_csv, fixture_corpus, out_dir=tmp_path / "one")
    _run(tmp_path, lots_csv, fixture_corpus, out_dir=tmp_path / "two")
    for split in ("train", "validation", "test"):
        assert (tmp_path / "one" / split / "instances.jsonl").read_bytes() == (
            tmp_path / "two" / split / "instances.jsonl"
        ).read_bytes()


def test_run_assembly_records_real_file_hashes(lots_csv, fixture_corpus, tmp_path):
    """三个文件记录都必须能从磁盘上的真文件独立复算。

    摘要与字节数来自文件本身；行数来自**解析**出来的记录条数，而不是再数一遍
    换行符——后者与 ``_file_record`` 的实现是同一个表达式，永远相等，什么也
    证明不了。解析计数会抓到多行 JSON、多余空行这类真实的写入事故。
    """
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    assert len(manifest["files"]) == 3
    for record in manifest["files"]:
        path = tmp_path / "processed" / record["path"]
        raw = path.read_bytes()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["bytes"] == len(raw)
        assert record["lines"] == len(_read(path))


def test_dedup_block_is_measured_from_the_written_corpus(lots_csv, fixture_corpus, tmp_path):
    """去重统计必须能从落盘内容独立复算出来。

    四个数各有独立来源：总数来自写出的行数、唯一数来自哈希集合、碰撞数与
    最大重抽次数来自每条记录的 resample_attempt。**任何一个都不许是与另一个
    同源的表达式**——那正是本项目反复出现的「常量对常量」病灶。
    """
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    written = [
        inst
        for record in manifest["files"]
        for inst in _read(tmp_path / "processed" / record["path"])
    ]
    dedup = manifest["dedup"]
    assert dedup["total_instances"] == len(written)
    assert dedup["distinct_canonical_hashes"] == len({i["canonical_hash"] for i in written})
    assert dedup["collisions_resampled"] == sum(i["resample_attempt"] for i in written)
    assert dedup["max_attempt_used"] == max(i["resample_attempt"] for i in written)


def test_run_assembly_enforces_the_family_source_invariant(lots_csv, fixture_corpus, tmp_path):
    """设计文档 §10 的不变量：family is None <=> source == "real"。"""
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    written = [
        inst
        for record in manifest["files"]
        for inst in _read(tmp_path / "processed" / record["path"])
    ]
    assert written
    for inst in written:
        assert (inst["family"] is None) == (inst["source"] == "real")


def test_run_assembly_reports_training_only_tail_thresholds(lots_csv, fixture_corpus, tmp_path):
    manifest = _run(tmp_path, lots_csv, fixture_corpus)
    threshold = manifest["tail_thresholds"]
    assert threshold["basis"] == "train-split lots only"
    assert threshold["quantile"] == 0.99
    # 夹具 train 段 7 天的 quantity 为 1..24，阈值必然落在这个区间内
    assert 1 <= int(threshold["quantity_w"]) <= 24


def test_run_assembly_rejects_a_lots_table_that_contradicts_the_collect_report(
    lots_csv, fixture_corpus, tmp_path
):
    """上游报告与实际批次表对不上时必须报错。

    `kept_rows` 与 CSV 的实际行数是两个独立来源；不一致意味着批次表被替换过
    而报告没跟着更新，此时装配出的语料无法溯源。
    """
    report = lots_csv.parent / "collect_report.json"
    doc = json.loads(report.read_text(encoding="utf-8"))
    doc["kept_rows"] = 999999
    report.write_bytes(json.dumps(doc, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    with pytest.raises(ValueError, match="999999"):
        _run(tmp_path, lots_csv, fixture_corpus)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_merge.py -p no:cacheprovider --basetemp="$TEMP/ap_t5" -q 2>&1 | tail -5
```

预期：`ImportError: cannot import name 'merge'`。

- [ ] **Step 3: 先交付 `manifest.py` 的两块地基**

Task 6 才补派生段；但有两块是本任务**立刻**要用的，现在就定稿、Task 6 不再改：

1. `upstream_provenance` —— `run_assembly` 拿它的 `lots_clean.rows` 与批次表实际
   行数对账（第 5 节的验收项之一）；
2. `write_manifest` / `write_config` —— 两个文件的格式约定（缩进、键序、LF、
   无 BOM、末尾换行）在这里一次定死，免得 Task 6 再动它们时把已冻结的字节改掉。

创建 `experiment/data/assemble/manifest.py`：

```python
"""协议产物：``data_manifest.json`` 与 ``experiment_config.yaml``。

本模块不读语料、不碰数据装配，只做两件事：从上游产物里**摘出**审计字段
（:func:`upstream_provenance`），以及把调用方给的事实**排版**成两个文件。

Task 5 交付 :func:`upstream_provenance` 与两个写出函数；manifest 的派生段
（``splits`` / ``split_policy`` / ``streams`` / ``output_dir``）与完整的 config
由 Task 6 补齐——那时装配流程已经跑通，可以拿真实产物校对。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from experiment.data.spec import CLEANING_RULES, LOTS_COLUMNS


def _file_fact(path: Path) -> dict:
    """一个文件的路径与摘要；文件不存在时摘要为 ``None``（不假装有）。"""
    path = Path(path)
    if not path.is_file():
        return {"bytes": None, "path": path.as_posix(), "sha256": None}
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def upstream_provenance(lots_path: Path, generation_dir: Path) -> dict:
    """从上游两个阶段的产物里**摘出**审计字段。

    刻意读上游 JSON 的**内容**而不是把数值复制一份：复制来的常量会随上游
    变化而说谎，而本项目的病灶正是「常量对常量」。

    上游布局由前两个阶段固定：采集报告是批次表的同目录兄弟
    ``collect_report.json``，生成清单是 ``generation_dir/manifest.json``。
    """
    lots_path = Path(lots_path)
    report_path = lots_path.parent / "collect_report.json"
    generation_manifest = Path(generation_dir) / "manifest.json"

    report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    )
    generation = (
        json.loads(generation_manifest.read_text(encoding="utf-8"))
        if generation_manifest.is_file()
        else {}
    )
    source = report.get("source", {})

    return {
        "cleaning_rules": list(CLEANING_RULES),
        "collect_report": _file_fact(report_path),
        "generation_manifest": {
            **_file_fact(generation_manifest),
            "generator_fingerprint": generation.get("generator_fingerprint"),
        },
        "lots_clean": {
            **_file_fact(lots_path),
            "columns": list(LOTS_COLUMNS),
            "rows": report.get("kept_rows"),
        },
        "units": {"value": "value_pence", "weight": "quantity_w"},
        "workbook": {
            "bytes": source.get("xlsx_bytes"),
            "sha256": source.get("xlsx_sha256"),
            "url": source.get("url"),
        },
    }


def build_manifest(
    cells: Sequence[dict],
    counts: Mapping[str, dict],
    dedup: Mapping[str, int],
    files: Sequence[dict],
    generated_at: str,
    lot_index,
    out_dir: Path,
    provenance: Mapping,
    schema_version: int,
    stress: Mapping[str, dict],
    tail: Mapping[str, object],
) -> dict:
    """把调用方给的事实原样带出。派生段由 Task 6 补齐。"""
    return {
        "cells": list(cells),
        "counts": {split: dict(value) for split, value in counts.items()},
        "dedup": dict(dedup),
        "files": list(files),
        "generated_at": generated_at,
        "provenance": dict(provenance),
        "schema_version": schema_version,
        "stress": {name: dict(value) for name, value in stress.items()},
        "tail_thresholds": dict(tail),
    }


def write_manifest(path: Path, manifest: Mapping) -> None:
    """写出 manifest：缩进 2、键排序、LF、无 BOM、末尾一个换行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def build_config(out_dir: Path, generated_at: str) -> dict:
    """Task 6 补齐生成器、配额、预算与评估段。"""
    return {"generated_at": generated_at, "schema_version": 1}


def write_config(path: Path, config: Mapping) -> None:
    """写出 YAML：键排序、不转义中文、块状风格、LF、无 BOM。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        dict(config), allow_unicode=True, default_flow_style=False, sort_keys=True
    )
    path.write_bytes(payload.encode("utf-8"))
```

- [ ] **Step 4: 写 `merge.py`**

```python
"""把合成语料与真实实例合并成冻结的混合语料，并编排整条装配流程。

合并文件的排列顺序是冻结语义，且要让 50:50 肉眼可核：
``for n: for alpha: 四个族各 quota 条 -> 真实 quota 条``。

依据 EXPERIMENT_PLAN.md 第 4 节与设计文档 §8、§9。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from experiment.data.assemble.lots import LotIndex, load_index, tail_thresholds
from experiment.data.assemble.manifest import (
    build_config,
    build_manifest,
    upstream_provenance,
    write_config,
    write_manifest,
)
from experiment.data.assemble.real import build_real_cell
from experiment.data.assemble.stress import build_e3, build_e4
from experiment.data.generate.corpus import write_cell
from experiment.data.spec import (
    DATA_MANIFEST_PATH,
    DATASETS_PROCESSED,
    EXPERIMENT_CONFIG_PATH,
    E3_NAME,
    E4_NAME,
    FAMILIES,
    GENERATION_DIR,
    REAL_DIR,
    REAL_QUOTA,
    SCHEMA_VERSION,
    SPLITS,
    SYNTHETIC_QUOTA,
    TAIL_QUANTILE,
    capacity_cells,
    cell_filename,
)


def synthetic_cell_path(
    generation_dir: Path, split: str, family: str, n: int, alpha: float
) -> Path:
    """合成语料里一个 cell 的文件路径。"""
    return Path(generation_dir) / split / cell_filename(family, n, alpha)


def read_cell(path: Path) -> list[dict]:
    """读回一个合成 cell 的实例，保持文件行序。

    刻意是「读回来」而不是「重新生成」：前者保证合并文件里的合成半边与已
    审计的原始语料逐字节同源，后者会让它依赖当前代码。
    """
    instances: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(json.loads(line))
    return instances


def assemble_split(
    split: str,
    lot_index: LotIndex,
    generation_dir: Path,
    cells: Sequence[tuple[int, float]] | None = None,
    synthetic_quota: int | None = None,
    real_quota: int | None = None,
    seen: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """拼出一个 split 的全部实例，并返回逐 cell 的 50:50 审计。

    Args:
        cells: 要装配的 ``(n, alpha)`` 序列；``None`` 表示全部 12 个。测试可
            只传一个 cell 以便在密闭夹具上跑。
        synthetic_quota: 每个合成族每 cell 的条数；``None`` 取冻结配额
        real_quota: 每 cell 真实条数；``None`` 取冻结配额
        seen: 跨 split 的全局哈希集合，就地更新

    Returns:
        ``(instances, cells_audit)``

    Raises:
        ValueError: 某 cell 两侧计数不等（第 4 节的 50:50），或合成半边与
            已装配的实例哈希重复。
    """
    book = set() if seen is None else seen
    syn_quota = SYNTHETIC_QUOTA[split] if synthetic_quota is None else synthetic_quota
    real_q = REAL_QUOTA[split] if real_quota is None else real_quota
    plan = list(capacity_cells() if cells is None else cells)

    instances: list[dict] = []
    audit: list[dict] = []

    for n, alpha in plan:
        synthetic: list[dict] = []
        for family in FAMILIES:
            path = synthetic_cell_path(generation_dir, split, family, n, alpha)
            synthetic.extend(read_cell(path)[:syn_quota])

        hashes = {inst["canonical_hash"] for inst in synthetic}
        if len(hashes) != len(synthetic):
            raise ValueError(f"{split}/n={n}/alpha={alpha} 的合成 cell 内部有重复实例")
        if hashes & book:
            raise ValueError(
                f"{split}/n={n}/alpha={alpha} 的合成实例与已装配的实例重复"
            )
        book |= hashes

        real = build_real_cell(lot_index, split, split, n, alpha, real_q, book)

        if len(synthetic) != len(real):
            raise ValueError(
                f"{split}/n={n}/alpha={alpha} 不满足 50:50："
                f"合成 {len(synthetic)} 条，真实 {len(real)} 条"
            )

        instances.extend(synthetic)
        instances.extend(real)
        audit.append(
            {
                "alpha": alpha,
                "n": n,
                "real": len(real),
                "split": split,
                "synthetic": len(synthetic),
            }
        )

    return instances, audit


def write_split(path: Path, instances: list[dict]) -> str:
    """落盘一个 split 的实例，返回内容的 SHA-256。

    直接复用生成阶段的 ``write_cell``：序列化约定必须与合成语料完全一致，
    否则同一个实例在两个文件里的字节会不同。
    """
    return write_cell(path, instances)


def _file_record(path: Path, root: Path) -> dict:
    """一个语料文件的审计记录：相对路径、摘要、字节数、行数。"""
    raw = Path(path).read_bytes()
    return {
        "bytes": len(raw),
        "lines": raw.count(b"\n"),
        "path": Path(path).relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _resample_stats(instances: Sequence[dict]) -> tuple[int, int]:
    """``(碰撞次数, 最大重抽次数)``。

    每条记录的 ``resample_attempt`` 就是它经历过的碰撞数，因此求和即总碰撞数。
    这两个数是从产物本身量出来的，不是另算一份可能与之脱节的常量。
    """
    return (
        sum(inst["resample_attempt"] for inst in instances),
        max((inst["resample_attempt"] for inst in instances), default=0),
    )


def run_assembly(
    out_dir: Path = DATASETS_PROCESSED,
    lots_path: Path = REAL_DIR / "lots_clean.csv",
    generation_dir: Path = GENERATION_DIR,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    cells: Sequence[tuple[int, float]] | None = None,
    synthetic_quota: Mapping[str, int] | None = None,
    real_quota: Mapping[str, int] | None = None,
    include_stress: bool = True,
) -> dict:
    """装配全部语料并写出两个协议产物，返回 manifest。

    遍历顺序是冻结语义：先三个核心 split（train -> validation -> test），
    再 E3、最后 E4。``seen`` 跨全部五个数据集共享，因此「全局唯一」是一条
    真断言，而不是三次各自为政的检查。

    Args:
        include_stress: 为 ``False`` 时跳过 E3/E4，供快速端到端测试使用
            （E3 的 n=500 在密闭夹具上无法满足）
    """
    manifest_path = DATA_MANIFEST_PATH if manifest_path is None else Path(manifest_path)
    config_path = EXPERIMENT_CONFIG_PATH if config_path is None else Path(config_path)
    out_dir = Path(out_dir)

    lot_index = load_index(lots_path)
    provenance = upstream_provenance(lots_path, generation_dir)

    declared = provenance["lots_clean"]["rows"]
    actual = len(lot_index.lots)
    if declared is not None and declared != actual:
        raise ValueError(
            f"collect_report.json 记录 {declared} 行，但批次表实际读出 {actual} 行；"
            "上游产物与其报告不一致，装配结果无法溯源"
        )

    book: set[str] = set()
    counts: dict[str, dict] = {}
    cells_audit: list[dict] = []
    files: list[dict] = []
    total_instances = 0
    collisions = 0
    max_attempt = 0

    for split in SPLITS:
        instances, audit = assemble_split(
            split,
            lot_index,
            generation_dir,
            cells,
            None if synthetic_quota is None else synthetic_quota[split],
            None if real_quota is None else real_quota[split],
            book,
        )
        path = out_dir / split / "instances.jsonl"
        write_split(path, instances)
        files.append(_file_record(path, out_dir))

        synthetic = sum(cell["synthetic"] for cell in audit)
        real = sum(cell["real"] for cell in audit)
        counts[split] = {"real": real, "synthetic": synthetic, "total": synthetic + real}
        cells_audit.extend(audit)

        total_instances += len(instances)
        hits, worst = _resample_stats(instances)
        collisions += hits
        max_attempt = max(max_attempt, worst)

    stress_counts: dict[str, dict] = {}
    if include_stress:
        for name, instances in (
            (E3_NAME, build_e3(lot_index, book)),
            (E4_NAME, build_e4(book)),
        ):
            path = out_dir / "stress" / name / "instances.jsonl"
            write_split(path, instances)
            files.append(_file_record(path, out_dir))
            stress_counts[name] = {
                "real": sum(1 for i in instances if i["source"] == "real"),
                "synthetic": sum(1 for i in instances if i["source"] == "synthetic"),
            }
            total_instances += len(instances)
            hits, worst = _resample_stats(instances)
            collisions += hits
            max_attempt = max(max_attempt, worst)

    train_lots = [lot for pool in lot_index.pools["train"].values() for lot in pool]

    manifest = build_manifest(
        cells=cells_audit,
        counts=counts,
        dedup={
            "collisions_resampled": collisions,
            "distinct_canonical_hashes": len(book),
            "max_attempt_used": max_attempt,
            "total_instances": total_instances,
        },
        files=files,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        lot_index=lot_index,
        out_dir=out_dir,
        provenance=provenance,
        schema_version=SCHEMA_VERSION,
        stress=stress_counts,
        tail=tail_thresholds(train_lots, TAIL_QUANTILE),
    )
    write_manifest(manifest_path, manifest)
    write_config(
        config_path,
        build_config(out_dir=out_dir, generated_at=manifest["generated_at"]),
    )
    return manifest
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_merge.py -p no:cacheprovider --basetemp="$TEMP/ap_t5" -q 2>&1 | tail -3
```

预期：`23 passed`。

- [ ] **Step 6: 全量测试**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t5all" -q 2>&1 | tail -3
```

预期：`267 passed`，0 failed、0 skipped。

- [ ] **Step 7: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/assemble/merge.py experiment/data/assemble/manifest.py tests/test_assemble_merge.py && git commit -m "feat(data): 合并装配、逐 cell 50:50 与全局去重"
```

---

## Task 6: `manifest.py` —— 两个协议产物

**Files:**
- Modify: `experiment/data/assemble/manifest.py`（补 manifest 派生段与完整 config）
- Test: `tests/test_assemble_manifest.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_assemble_manifest.py`：

```python
"""两个协议产物的字段、真实性与格式。

`upstream_provenance` 与两个写出函数已在 Task 5 交付；本任务补 manifest 的
派生段与完整的 config。
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiment.data import spec
from experiment.data.assemble import manifest as M


def _tmp(tmp_path: Path) -> dict:
    """一份最小但完整的 manifest 输入。"""
    corpus = tmp_path / "processed" / "train"
    corpus.mkdir(parents=True, exist_ok=True)
    path = corpus / "instances.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    return dict(
        cells=[{"alpha": 0.25, "n": 20, "real": 500, "split": "train", "synthetic": 500}],
        counts={"train": {"real": 6000, "synthetic": 6000, "total": 12000}},
        dedup={
            "collisions_resampled": 0,
            "distinct_canonical_hashes": 17400,
            "max_attempt_used": 0,
            "total_instances": 17400,
        },
        files=[
            {
                "bytes": 16,
                "lines": 2,
                "path": "train/instances.jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
        generated_at="2026-09-28T00:00:00Z",
        lot_index=None,
        out_dir=tmp_path / "processed",
        provenance={"stub": True},
        schema_version=1,
        stress={"e3_size": {"real": 300, "synthetic": 300}, "e4_tail": {"real": 0, "synthetic": 300}},
        tail={
            "basis": "train-split lots only",
            "quantile": 0.99,
            "quantity_w": "1234",
            "unit_price": "5.00",
        },
    )


# --- manifest 的字段 --------------------------------------------------------


def test_manifest_has_the_sections_the_plan_requires(tmp_path):
    """§10 要求 manifest 载有源校验和、清洗规则、日期、种子、计数、单位、
    源/组 ID 与切分/哈希审计。"""
    man = M.build_manifest(**_tmp(tmp_path))
    assert set(man) >= {
        "schema_version", "generated_at", "provenance", "split_policy",
        "splits", "streams", "cells", "counts", "stress", "tail_thresholds",
        "dedup", "files", "output_dir",
    }


def test_manifest_carries_the_provenance_it_was_given(tmp_path):
    man = M.build_manifest(**_tmp(tmp_path))
    assert man["provenance"] == {"stub": True}


def test_streams_record_both_the_key_and_the_declared_master_seed(tmp_path):
    """§8 的 4001/5001 是声明标识，真正的隔离是字符串记号。两者都要记录，
    这个关系才看得出来。"""
    streams = M.build_manifest(**_tmp(tmp_path))["streams"]
    assert streams["size"]["seed_key"]["split"] == spec.STRESS_SIZE_STREAM
    assert streams["size"]["declared_master_seed"] == 4001
    assert streams["size"]["dataset"] == spec.E3_NAME
    assert streams["tail"]["seed_key"]["split"] == spec.STRESS_TAIL_STREAM
    assert streams["tail"]["seed_key"]["family"] == spec.E4_SEED_TOKEN
    assert streams["tail"]["declared_master_seed"] == 5001
    assert streams["core"]["seed_key"]["family"] == spec.REAL_SEED_TOKEN


def test_tail_thresholds_are_carried_through_unchanged(tmp_path):
    facts = _tmp(tmp_path)
    assert M.build_manifest(**facts)["tail_thresholds"] == facts["tail"]


def test_manifest_is_json_serialisable(tmp_path):
    json.dumps(M.build_manifest(**_tmp(tmp_path)), sort_keys=True, ensure_ascii=False)


def test_write_manifest_uses_lf_and_no_bom(tmp_path):
    path = tmp_path / "data_manifest.json"
    M.write_manifest(path, M.build_manifest(**_tmp(tmp_path)))
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8"))["schema_version"] == 1


def test_write_manifest_is_byte_stable_for_the_same_input(tmp_path):
    man = M.build_manifest(**_tmp(tmp_path))
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    M.write_manifest(a, man)
    M.write_manifest(b, man)
    assert a.read_bytes() == b.read_bytes()


# --- config 的字段 ----------------------------------------------------------


def _config(tmp_path: Path, **kwargs) -> dict:
    kwargs.setdefault("requirements", tmp_path / "requirements.txt")
    if not kwargs["requirements"].is_file():
        kwargs["requirements"].write_text("numpy==2.5.3\n", encoding="utf-8")
    return M.build_config(out_dir=tmp_path, generated_at="2026-09-28T00:00:00Z", **kwargs)


def test_config_carries_the_sections_the_plan_requires(tmp_path):
    """§10：生成器公式、配额、RL 设置、预算、评估规则、固定包版本。"""
    doc = _config(tmp_path)
    assert set(doc) >= {
        "schema_version", "generated_at", "generators", "cap_ratios", "sizes",
        "quotas", "real", "stress", "budgets", "evaluation", "packages", "runtime",
    }


def test_config_packages_are_parsed_from_requirements_not_hardcoded(tmp_path):
    """硬编码的版本会在依赖升级后悄悄说谎。"""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==9.9.9\n\n# 注释\npandas==1.2.3\n", encoding="utf-8")
    doc = _config(tmp_path, requirements=req)
    assert doc["packages"]["numpy"] == "9.9.9"
    assert doc["packages"]["pandas"] == "1.2.3"
    assert "注释" not in doc["packages"]


def test_config_packages_follow_an_edited_requirements_file(tmp_path):
    """改一个版本号，产物必须跟着变——证明它真是解析出来的。"""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.1.1\n", encoding="utf-8")
    assert _config(tmp_path, requirements=req)["packages"]["numpy"] == "1.1.1"
    req.write_text("numpy==2.2.2\n", encoding="utf-8")
    assert _config(tmp_path, requirements=req)["packages"]["numpy"] == "2.2.2"


def test_config_rejects_an_unpinned_requirement(tmp_path):
    """半钉住的依赖就是复现性漏洞，宁可报错也不静默记录。"""
    req = tmp_path / "requirements.txt"
    req.write_text("numpy>=1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numpy"):
        M.build_config(out_dir=tmp_path, generated_at="x", requirements=req)


def test_config_records_the_frozen_quotas_and_sizes(tmp_path):
    doc = _config(tmp_path)
    assert doc["quotas"]["synthetic"] == dict(spec.SYNTHETIC_QUOTA)
    assert doc["quotas"]["real"] == dict(spec.REAL_QUOTA)
    assert doc["sizes"] == list(spec.SIZES)
    assert doc["cap_ratios"] == list(spec.ALPHAS)
    assert doc["real"]["split_fractions"] == list(spec.SPLIT_DATE_FRACTIONS)


def test_config_e4_generator_uses_the_variant_parameters(tmp_path):
    doc = _config(tmp_path)
    assert doc["generators"]["skewed_e4"]["k"] == spec.E4_SKEW_K
    assert doc["generators"]["skewed_e4"]["heavy_weight"] == list(
        spec.E4_SKEW_HEAVY_WEIGHT_RANGE
    )
    assert doc["generators"]["skewed_e4"]["heavy_density"] == list(
        spec.E4_SKEW_HEAVY_DENSITY_RANGE
    )


def test_config_records_the_frozen_families_unchanged(tmp_path):
    """主语料的 skewed 参数必须与 spec 一致——E4 是变体，不是改版。"""
    skewed = _config(tmp_path)["generators"]["skewed"]
    assert skewed["heavy_weight"] == list(spec.SKEW_HEAVY_WEIGHT_RANGE)
    assert skewed["heavy_density"] == list(spec.SKEW_HEAVY_DENSITY_RANGE)
    assert skewed["base_weight"] == list(spec.SKEW_BASE_WEIGHT_RANGE)
    assert skewed["base_density"] == list(spec.SKEW_BASE_DENSITY_RANGE)


def test_config_records_the_training_budget_and_evaluation_rules(tmp_path):
    doc = _config(tmp_path)
    assert doc["budgets"]["training_transitions"] == 300000
    assert doc["budgets"]["training_seeds"] == [0, 1, 2]
    assert doc["evaluation"]["methods"] == ["random", "greedy", "dqn"]
    assert doc["evaluation"]["random_trials_per_instance"] == 10
    assert doc["evaluation"]["evaluation_seed"] == 6001
    assert doc["evaluation"]["bootstrap"]["resamples"] == 1000


def test_write_config_roundtrips_as_yaml(tmp_path):
    doc = _config(tmp_path)
    path = tmp_path / "experiment_config.yaml"
    M.write_config(path, doc)
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert yaml.safe_load(raw.decode("utf-8")) == doc


def test_write_config_is_byte_stable_and_sorts_keys(tmp_path):
    doc = _config(tmp_path)
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    M.write_config(a, doc)
    M.write_config(b, doc)
    assert a.read_bytes() == b.read_bytes()
    text = a.read_text(encoding="utf-8")
    assert text.index("cap_ratios") < text.index("generators") < text.index("packages")


def test_write_config_keeps_chinese_readable(tmp_path):
    """allow_unicode=True：中文标签不该被转义成 \\uXXXX 而变得不可读。"""
    path = tmp_path / "c.yaml"
    M.write_config(path, {"note": "训练预算"})
    assert "训练预算" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_manifest.py -p no:cacheprovider --basetemp="$TEMP/ap_t6" -q 2>&1 | tail -4
```

预期：**`12 failed, 6 passed`**（共 18 条；下面是实测的逐条分布，不是估计）。

失败分三类：

- **10 条是同一个 `TypeError: build_config() got an unexpected keyword argument 'requirements'`**
  ——Task 5 交付的 `build_config(out_dir, generated_at)` 只有两个形参，而本任务的测试
  要传第三个。**注意这不是 `KeyError`**：异常在进入任何字典之前就抛出了，所以
  "config 的一整批 `KeyError`" 这种读法在本状态下根本不可达。8 条 `test_config_*`
  加 `test_write_config_roundtrips_as_yaml`、`test_write_config_is_byte_stable_and_sorts_keys`
  都走 `_config()` 助手，因而全部撞在这一个 `TypeError` 上。
- **1 条 `AssertionError`**：`test_manifest_has_the_sections_the_plan_requires` 的集合
  比较报 `Extra items in the right set: 'split_policy', 'splits', 'streams', 'output_dir'`
  ——是缺键的集合断言，不是 `KeyError: 'split_policy'`。
- **1 条 `KeyError: 'streams'`**：`test_streams_record_both_the_key_and_the_declared_master_seed`。

**会通过的 6 条，与失败的一样重要**，别以为"字段类全红"就对了：

- `test_write_manifest_uses_lf_and_no_bom`、`test_write_manifest_is_byte_stable_for_the_same_input`
  ——写出函数在 Task 5 已定稿，本任务不动它们；
- `test_manifest_carries_the_provenance_it_was_given`、`test_tail_thresholds_are_carried_through_unchanged`、
  `test_manifest_is_json_serialisable`——这三条只看 Task 5 已有的字段，所以**本来就该绿**；
- `test_write_config_keeps_chinese_readable`——它直接调 `write_config`，不经过 `_config()`，
  所以躲开了那个 `TypeError`。

这是预期的，不是"测试写错了"；实现者**不必为让它们变红而改动任何东西**，
也**不该**为了让 `test_write_config_*` 变绿去改 `write_config`——它要的那第三个
形参正是本任务 Step 3 要给 `build_config` 加上的东西。

- [ ] **Step 3: 补齐 `manifest.py`**

`upstream_provenance` 与 `write_manifest` / `write_config` 保持 Task 5 的样子不动；
本步只增长 `build_manifest`（加 `splits` / `split_policy` / `streams` / `output_dir`）
与 `build_config`（加生成器、配额、预算、评估、包版本），并新增 `parse_packages`、
补上 `_boundaries` / `_splits` / `_eligible_dates` 三个纯函数。**下方整段替换该文件**：

```python
"""协议产物：``data_manifest.json`` 与 ``experiment_config.yaml``。

两者都放在**仓库根目录并纳入版本库**：``datasets/`` 整体被 gitignore，把审计
凭据放在那里等于换一台机器就无从证明数据未被调换。

本模块不读语料、不碰数据装配，只做两件事：从上游产物里**摘出**审计字段
（:func:`upstream_provenance`），以及把调用方给的事实**排版**成两个文件。
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from experiment.data.spec import (
    ALPHAS,
    CLEANING_RULES,
    E3_NAME,
    E3_REAL_PER_ALPHA,
    E3_SIZE,
    E3_SYNTHETIC_PER_FAMILY,
    E4_COUNT_PER_ALPHA,
    E4_NAME,
    E4_SEED_TOKEN,
    E4_SIZE,
    E4_SKEW_HEAVY_DENSITY_RANGE,
    E4_SKEW_HEAVY_WEIGHT_RANGE,
    E4_SKEW_K,
    LOTS_COLUMNS,
    REAL_QUOTA,
    REAL_SEED_TOKEN,
    REPO_ROOT,
    SCHEMA_VERSION,
    SKEW_BASE_DENSITY_RANGE,
    SKEW_BASE_WEIGHT_RANGE,
    SKEW_FRACTION,
    SKEW_HEAVY_DENSITY_RANGE,
    SKEW_HEAVY_WEIGHT_RANGE,
    SIZES,
    SPLIT_DATE_FRACTIONS,
    STRESS_MASTER_SEEDS,
    STRESS_SIZE_STREAM,
    STRESS_TAIL_STREAM,
    SYNTHETIC_QUOTA,
    TAIL_QUANTILE,
)

#: 训练预算，逐项转录自 EXPERIMENT_PLAN.md §6 的表格与正文。
#: 刻意原样保留计划里的数值：后续阶段应当读 YAML 而不是回头读 Markdown。
_BUDGETS: dict[str, object] = {
    "batch_size": 64,
    "epsilon": {"final": 0.05, "linear_decay_transitions": 150000, "initial": 1.0},
    "grad_clip_global_norm": 10,
    "learning_rate": 0.0003,
    "loss": "huber",
    "optimizer": "adam",
    "pilot_transitions": 50000,
    "replay_capacity": 100000,
    "replay_source_balance": {"real": 32, "synthetic": 32},
    "target_copy_transitions": 2000,
    "training_seeds": [0, 1, 2],
    "training_transitions": 300000,
    "update_every_transitions": 4,
    "validation_every_transitions": 25000,
    "warmup_transitions": 5000,
}

#: 评估口径，转录自 EXPERIMENT_PLAN.md §5（三种方法的定义）与 §7（参考解与指标）。
_EVALUATION: dict[str, object] = {
    "aggregate": "one-half synthetic mean + one-half real mean",
    "bootstrap": {
        "paired": True,
        "resamples": 1000,
        "unit": "whole source dates for real; whole synthetic instances within cells",
    },
    "evaluation_seed": 6001,
    "greedy": "sort by descending v/w; ties by lower w then stable item ID",
    "metrics": [
        "feasibility",
        "exact_optimality_gap",
        "bound_based_gap",
        "relative_improvement",
        "speed",
        "learning_cost",
        "robustness",
    ],
    "methods": ["random", "greedy", "dqn"],
    "random_trials_per_instance": 10,
    "reference": {
        "cp_sat_limits": {"seed": 0, "time_seconds": 10, "workers": 1},
        "general": "OR-Tools CP-SAT with binary item variables",
        "note": "an evaluation service, not a fourth competing method",
        "small_cases": "integer dynamic programming",
    },
    "rl": "argmax feasible Q-value with exploration disabled; exact ties resolved by skipping",
    "trivial_case": "empty solution when no item fits; excluded from percentage denominators",
}

#: EXPERIMENT_PLAN.md §6 的**正文**（相对表格而言）里定下的环境与训练规则。
#: 表格进 :data:`_BUDGETS`，正文进这里：两者都是转录，少记一半，后续阶段就得
#: 回头读 Markdown，而那份 Markdown 不是被冻结、被校验的产物。
_RL_ENVIRONMENT: dict[str, object] = {
    "action_masking": (
        "skip is always valid; infeasible include actions are masked in exploration, "
        "inference, and the maximum used in the learning target"
    ),
    "actions": (
        "include subtracts the resource requirement and earns the item's value; "
        "skip earns zero"
    ),
    "budget_escalation": {
        "applies_to": "all three seeds",
        "condition": "if 300000 transitions is insufficient and time allows",
        "means": "a validation-only decision, recorded",
        "to_transitions": 1000000,
    },
    "budget_freeze": "fix the final budget before viewing test metrics",
    "checkpoint_selection": {
        "basis": (
            "mean validation score V/U, where U is a fixed valid reference upper "
            "bound for that instance"
        ),
        "certified_optimum": (
            "U = V*; trivial zero-optimum cases are excluded from the ratio and "
            "counted separately"
        ),
        "never_select_on": "test performance",
        "source_weighting": (
            "weight sources equally; report cell-macro scores if quotas changed"
        ),
        "tie_break": "the earlier checkpoint",
    },
    "episode": "each item is considered exactly once",
    "gamma": 1,
    "network": {
        "encoder": (
            "shared two-layer, 64-unit ReLU over each remaining item's normalized "
            "(w_i/C, v_i/sum(v))"
        ),
        "excluded_from_inputs": "IDs and source labels are not prediction features",
        "head": "128-unit ReLU layer with two Q-value outputs",
        "inputs_after_pooling": [
            "mean and max pooled embeddings",
            "the current item's normalized pair",
            "remaining_capacity/C",
            "remaining_count/n",
            "log(1+n)",
        ],
        "pooling": (
            "only unprocessed items, including the current item; processed items "
            "and padding excluded"
        ),
        "terminal": "terminal states bypass next-state evaluation",
    },
    "reward_shaping": (
        "for learning only, reward = v_i / sum_j(v_j) when including an item; final "
        "selections are evaluated in the original units"
    ),
    "sampling": (
        "sample repeatedly from the fixed training corpus; a balanced block takes one "
        "instance per available (source, size) pair in random order, with capacity "
        "ratios and synthetic families cycled evenly across blocks; cells marked "
        "unavailable in the manifest are excluded"
    ),
    "state": "all item requirements/values, the current index, and the remaining capacity",
    "termination": "after the last item the episode terminates and the bootstrap term is zero",
}


def _display_path(path: Path) -> str:
    """路径在产物里的写法：仓库内记相对形式，仓库外保持绝对。

    绝对路径会把 ``C:/work/学校/...`` 这类本机信息写进**纳入版本库**的凭据里：
    换一台机器复核时，它既指向一个不存在的目录树，也无法与复核者自己的工作区
    比对。相对路径则恰好是复核者 `git status` 里看到的那个字符串。

    用 POSIX 分隔符而不是本机分隔符：两个产物都是 JSON/YAML，读者可能在任何
    平台上，反斜杠在 JSON 里还要转义，肉眼比对时是个陷阱。
    """
    path = Path(path)
    if path.is_relative_to(REPO_ROOT):
        return path.relative_to(REPO_ROOT).as_posix()
    return path.as_posix()


def _file_fact(path: Path) -> dict:
    """一个文件的路径与摘要；文件不存在时摘要为 ``None``（不假装有）。"""
    path = Path(path)
    if not path.is_file():
        return {"bytes": None, "path": _display_path(path), "sha256": None}
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": _display_path(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def upstream_provenance(lots_path: Path, generation_dir: Path) -> dict:
    """从上游两个阶段的产物里**摘出**审计字段。

    刻意读上游 JSON 的**内容**而不是把数值复制一份：复制来的常量会随上游
    变化而说谎，而本项目的病灶正是「常量对常量」。

    上游布局由前两个阶段固定：采集报告是批次表的同目录兄弟
    ``collect_report.json``，生成清单是 ``generation_dir/manifest.json``。
    """
    lots_path = Path(lots_path)
    report_path = lots_path.parent / "collect_report.json"
    generation_manifest = Path(generation_dir) / "manifest.json"

    report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    )
    generation = (
        json.loads(generation_manifest.read_text(encoding="utf-8"))
        if generation_manifest.is_file()
        else {}
    )
    source = report.get("source", {})

    return {
        "cleaning_rules": list(CLEANING_RULES),
        "collect_report": _file_fact(report_path),
        "generation_manifest": {
            **_file_fact(generation_manifest),
            "generator_fingerprint": generation.get("generator_fingerprint"),
        },
        "lots_clean": {
            **_file_fact(lots_path),
            "columns": list(LOTS_COLUMNS),
            "rows": report.get("kept_rows"),
        },
        "units": {"value": "value_pence", "weight": "quantity_w"},
        "workbook": {
            "bytes": source.get("xlsx_bytes"),
            "sha256": source.get("xlsx_sha256"),
            "url": source.get("url"),
        },
    }


def _boundaries(lot_index) -> dict[str, int]:
    """两个日期边界，以**整数天数**记录。

    写成两个独立数字而不是再存一份 0.85，是为了让校验者不必重做浮点下取整。
    """
    if lot_index is None:
        return {"train_end_exclusive": 0, "validation_end_exclusive": 0}
    train = len(lot_index.date_splits["train"])
    return {
        "train_end_exclusive": train,
        "validation_end_exclusive": train + len(lot_index.date_splits["validation"]),
    }


def _splits(lot_index) -> dict[str, dict]:
    """每个 split 的实测规模。

    实测占比**不必等于**日期占比（§4 明说），所以要存下来供人核对，
    而不是断言相等。
    """
    if lot_index is None:
        return {}
    total_lots = len(lot_index.lots)
    total_invoices = len(lot_index.first_dates)
    out: dict[str, dict] = {}
    for split, days in lot_index.date_splits.items():
        day_set = set(days)
        lots = sum(len(pool) for pool in lot_index.pools[split].values())
        invoices = sum(1 for day in lot_index.first_dates.values() if day in day_set)
        out[split] = {
            "date_from": days[0] if days else None,
            "date_to": days[-1] if days else None,
            "dates": len(days),
            "invoices": invoices,
            "invoices_share": invoices / total_invoices if total_invoices else 0.0,
            "lots": lots,
            "lots_share": lots / total_lots if total_lots else 0.0,
        }
    return out


def _eligible_dates(lot_index, sizes: Sequence[int]) -> dict[str, dict[str, int]]:
    """每个 split 中批次数 ≥ n 的日期个数，按 n 分列。"""
    if lot_index is None:
        return {}
    return {
        split: {str(n): len(lot_index.eligible(split, n)) for n in sizes}
        for split in lot_index.date_splits
    }


def build_manifest(
    cells: Sequence[dict],
    counts: Mapping[str, dict],
    dedup: Mapping[str, int],
    files: Sequence[dict],
    generated_at: str,
    lot_index,
    out_dir: Path,
    provenance: Mapping,
    schema_version: int,
    stress: Mapping[str, dict],
    tail: Mapping[str, object],
) -> dict:
    """汇总审计 manifest。每一项都能在事后被独立复核。"""
    return {
        "cells": list(cells),
        "counts": {split: dict(value) for split, value in counts.items()},
        "dedup": dict(dedup),
        "files": list(files),
        "generated_at": generated_at,
        "output_dir": _display_path(out_dir),
        "provenance": dict(provenance),
        "schema_version": schema_version,
        "split_policy": {
            "basis": "invoice earliest transaction calendar date",
            "boundaries": _boundaries(lot_index),
            "eligible_dates": _eligible_dates(lot_index, list(SIZES) + [E3_SIZE]),
            "fractions": list(SPLIT_DATE_FRACTIONS),
        },
        "splits": _splits(lot_index),
        "streams": {
            "core": {"seed_key": {"family": REAL_SEED_TOKEN, "split": "<per-split name>"}},
            "size": {
                "dataset": E3_NAME,
                "declared_master_seed": STRESS_MASTER_SEEDS["size"],
                "seed_key": {
                    "family": f"<family> | {REAL_SEED_TOKEN}",
                    "split": STRESS_SIZE_STREAM,
                },
            },
            "tail": {
                "dataset": E4_NAME,
                "declared_master_seed": STRESS_MASTER_SEEDS["tail"],
                "seed_key": {"family": E4_SEED_TOKEN, "split": STRESS_TAIL_STREAM},
            },
        },
        "stress": {name: dict(value) for name, value in stress.items()},
        "tail_thresholds": dict(tail),
    }


def write_manifest(path: Path, manifest: Mapping) -> None:
    """写出 manifest：缩进 2、键排序、LF、无 BOM、末尾一个换行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def parse_packages(requirements: Path) -> dict[str, str]:
    """从 ``requirements.txt`` 解析固定版本。

    刻意解析而不是硬编码：硬编码就会在依赖升级后悄悄说谎。

    Raises:
        ValueError: 出现没有 ``==`` 的依赖行。半钉住的依赖就是复现性漏洞。
    """
    packages: dict[str, str] = {}
    for raw in Path(requirements).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep or not version.strip():
            raise ValueError(
                f"requirements.txt 中的 {line!r} 没有钉住版本；"
                "本项目的复现性依赖全部依赖项精确固定"
            )
        packages[name.strip()] = version.strip()
    return packages


def build_config(
    out_dir: Path,
    generated_at: str,
    requirements: Path | None = None,
) -> dict:
    """``experiment_config.yaml`` 的内容：转录计划的公式、配额、预算与评估设置。

    生成器与评估段是**转录**而不是重算：让后续阶段读 YAML 而不是回头读
    Markdown。``packages`` 与 ``runtime`` 是仅有的两项从磁盘实测的，
    因为硬编码的版本会在升级后说谎。
    """
    requirements = (
        REPO_ROOT / "requirements.txt" if requirements is None else Path(requirements)
    )
    return {
        "budgets": dict(_BUDGETS),
        "cap_ratios": list(ALPHAS),
        "evaluation": dict(_EVALUATION),
        "generated_at": generated_at,
        "generators": {
            "correlated": {
                "e": "U(-10,10)",
                "value": "max(1, w + e)",
                "weight": "U(1,100)",
            },
            "normal": {
                "clip": [1, 100],
                "distribution": "Normal(50, 15^2)",
                "mean": 50,
                "rounding": "half-up to nearest integer",
                "sd": 15,
            },
            "skewed": {
                "base_density": list(SKEW_BASE_DENSITY_RANGE),
                "base_weight": list(SKEW_BASE_WEIGHT_RANGE),
                "heavy_density": list(SKEW_HEAVY_DENSITY_RANGE),
                "heavy_weight": list(SKEW_HEAVY_WEIGHT_RANGE),
                "k": f"max(1, floor({SKEW_FRACTION} * n + 0.5))",
                "value": "w * d",
            },
            "skewed_e4": {
                "base_density": list(SKEW_BASE_DENSITY_RANGE),
                "base_weight": list(SKEW_BASE_WEIGHT_RANGE),
                "heavy_density": list(E4_SKEW_HEAVY_DENSITY_RANGE),
                "heavy_weight": list(E4_SKEW_HEAVY_WEIGHT_RANGE),
                "k": E4_SKEW_K,
                "note": "E4 variant; the frozen 'skewed' family is unchanged",
                "value": "w * d",
            },
            "uniform": {"value": "U(1,100)", "weight": "U(1,100)"},
        },
        "packages": parse_packages(requirements),
        "quotas": {"real": dict(REAL_QUOTA), "synthetic": dict(SYNTHETIC_QUOTA)},
        "real": {"split_fractions": list(SPLIT_DATE_FRACTIONS)},
        "rl_environment": dict(_RL_ENVIRONMENT),
        "runtime": {"python": platform.python_version()},
        "schema_version": SCHEMA_VERSION,
        "sizes": list(SIZES),
        "stress": {
            E3_NAME: {
                "n": E3_SIZE,
                "real_per_alpha": E3_REAL_PER_ALPHA,
                "synthetic_per_family": E3_SYNTHETIC_PER_FAMILY,
            },
            E4_NAME: {
                "families": [E4_SEED_TOKEN],
                "n": E4_SIZE,
                "per_alpha": E4_COUNT_PER_ALPHA,
                "quantile_for_e4b": TAIL_QUANTILE,
            },
        },
    }


def write_config(path: Path, config: Mapping) -> None:
    """写出 YAML：键排序、不转义中文、块状风格、LF、无 BOM。

    ``sort_keys=True`` 保证字典键序稳定；``allow_unicode=True`` 避免中文被写成
    ``\\uXXXX`` 而让文件不可读。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        dict(config), allow_unicode=True, default_flow_style=False, sort_keys=True
    )
    path.write_bytes(payload.encode("utf-8"))
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_manifest.py -p no:cacheprovider --basetemp="$TEMP/ap_t6" -q 2>&1 | tail -3
```

预期：`18 passed`。

- [ ] **Step 5: 全量测试**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t6all" -q 2>&1 | tail -3
```

预期：`285 passed`，0 failed、0 skipped。

- [ ] **Step 6: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/assemble/manifest.py tests/test_assemble_manifest.py && git commit -m "feat(data): 审计 manifest 与实验配置两个协议产物"
```

---

## Task 7: `cli.py` —— 命令行入口

**Files:**
- Create: `experiment/data/assemble/cli.py`
- Test: `tests/test_assemble_cli.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_assemble_cli.py`：

```python
"""命令行入口的参数与退出码。"""

import json

import pytest

from experiment.data.assemble import cli

CELL = "20x0.25"
BASE_ARGV = ["--cell", CELL, "--synthetic-quota", "1", "--real-quota", "4", "--no-stress"]


def _argv(tmp_path, lots_csv, fixture_corpus) -> list[str]:
    return [
        "cli",
        "--out", str(tmp_path / "processed"),
        "--lots", str(lots_csv),
        "--generation", str(fixture_corpus),
        "--manifest", str(tmp_path / "data_manifest.json"),
        "--config", str(tmp_path / "experiment_config.yaml"),
        *BASE_ARGV,
    ]


def test_main_writes_every_artifact_and_exits_zero(
    tmp_path, lots_csv, fixture_corpus, monkeypatch
):
    monkeypatch.setattr("sys.argv", _argv(tmp_path, lots_csv, fixture_corpus))
    assert cli.main() == 0
    assert (tmp_path / "processed" / "train" / "instances.jsonl").is_file()
    assert (tmp_path / "processed" / "validation" / "instances.jsonl").is_file()
    assert (tmp_path / "processed" / "test" / "instances.jsonl").is_file()
    doc = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["counts"]["train"] == {"synthetic": 4, "real": 4, "total": 8}


def test_main_reports_the_counts_it_wrote(
    tmp_path, lots_csv, fixture_corpus, monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", _argv(tmp_path, lots_csv, fixture_corpus))
    cli.main()
    printed = capsys.readouterr().out
    assert "train" in printed
    assert "50:50" in printed
    # 每个核心 split 都是 4 合成 + 4 真实
    assert printed.count("合成 4 + 真实 4 = 8") == 3


def test_main_skips_stress_when_asked(tmp_path, lots_csv, fixture_corpus, monkeypatch):
    monkeypatch.setattr("sys.argv", _argv(tmp_path, lots_csv, fixture_corpus))
    cli.main()
    assert not (tmp_path / "processed" / "stress").exists()


def test_parse_cell_accepts_the_plan_spelling():
    assert cli.parse_cell("20x0.25") == (20, 0.25)
    assert cli.parse_cell("200x0.75") == (200, 0.75)


def test_parse_cell_rejects_garbage():
    with pytest.raises(Exception):
        cli.parse_cell("twenty")
    with pytest.raises(Exception):
        cli.parse_cell("20")


def test_default_paths_come_from_spec():
    from experiment.data import spec

    args = cli.build_parser().parse_args([])
    assert args.out == spec.DATASETS_PROCESSED
    assert args.manifest == spec.DATA_MANIFEST_PATH
    assert args.config == spec.EXPERIMENT_CONFIG_PATH
    assert args.lots == spec.REAL_DIR / "lots_clean.csv"
    assert args.generation == spec.GENERATION_DIR
    assert args.cell is None
    assert args.no_stress is False
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_cli.py -p no:cacheprovider --basetemp="$TEMP/ap_t7" -q 2>&1 | tail -4
```

预期：`ImportError: cannot import name 'cli' from 'experiment.data.assemble'`。

异常类型取决于导入写法：本任务的测试用的是 `from experiment.data.assemble import cli`，
所以报 `ImportError`（包里没有这个名字）；`import experiment.data.assemble.cli` 才会报
`ModuleNotFoundError`。两者说的是同一件事——模块还不存在——所以实现者看到 `ImportError`
是**对的**，不要为了凑 `ModuleNotFoundError` 去改测试的导入行。

- [ ] **Step 3: 写实现**

创建 `experiment/data/assemble/cli.py`：

```python
"""冻结语料装配入口。

用法::

    python -m experiment.data.assemble.cli
    python -m experiment.data.assemble.cli --out datasets/processed
    python -m experiment.data.assemble.cli --cell 20x0.25 --synthetic-quota 1 --no-stress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiment.data.assemble.merge import run_assembly
from experiment.data.spec import (
    DATA_MANIFEST_PATH,
    DATASETS_PROCESSED,
    EXPERIMENT_CONFIG_PATH,
    GENERATION_DIR,
    REAL_DIR,
    SPLITS,
)


def parse_cell(text: str) -> tuple[int, float]:
    """解析 ``20x0.25`` 形式的 cell 说明。

    用 ``x`` 而不是逗号：Windows 的 shell 与 argparse 对逗号分隔的多值参数
    处理各不相同，而 ``x`` 是一个不需要转义的整体记号。
    """
    size, sep, alpha = text.partition("x")
    if not sep:
        raise argparse.ArgumentTypeError(f"cell 需形如 20x0.25，收到 {text!r}")
    try:
        return int(size), float(alpha)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cell 需形如 20x0.25，收到 {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="装配冻结的混合背包语料")
    parser.add_argument(
        "--out", type=Path, default=DATASETS_PROCESSED, help="语料输出根目录"
    )
    parser.add_argument(
        "--lots", type=Path, default=REAL_DIR / "lots_clean.csv", help="清洗后的批次表"
    )
    parser.add_argument(
        "--generation", type=Path, default=GENERATION_DIR, help="合成语料根目录"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATA_MANIFEST_PATH, help="data_manifest.json 的位置"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_CONFIG_PATH,
        help="experiment_config.yaml 的位置",
    )
    parser.add_argument(
        "--cell",
        type=parse_cell,
        action="append",
        default=None,
        help="只装配指定的 n x alpha，可重复；默认全部 12 个",
    )
    parser.add_argument(
        "--synthetic-quota",
        type=int,
        default=None,
        help="覆盖每族每 cell 的合成条数（仅供小样本试跑）",
    )
    parser.add_argument(
        "--real-quota",
        type=int,
        default=None,
        help="覆盖每 cell 的真实条数（仅供小样本试跑）",
    )
    parser.add_argument("--no-stress", action="store_true", help="跳过 E3/E4")
    return parser


def main() -> int:
    # 与 generate/collect 的入口一致：重定向到文件时 Windows 用 ANSI 代码页
    # 编码 stdout，中文标签会抛 UnicodeEncodeError；此时语料已经写完，只会
    # 留下一个失败的退出码和空日志。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    synthetic_quota = (
        None if args.synthetic_quota is None else {s: args.synthetic_quota for s in SPLITS}
    )
    real_quota = None if args.real_quota is None else {s: args.real_quota for s in SPLITS}

    manifest = run_assembly(
        out_dir=args.out,
        lots_path=args.lots,
        generation_dir=args.generation,
        manifest_path=args.manifest,
        config_path=args.config,
        cells=args.cell,
        synthetic_quota=synthetic_quota,
        real_quota=real_quota,
        include_stress=not args.no_stress,
    )

    print(f"输出目录: {args.out}")
    print(f"manifest: {args.manifest}")
    print(f"config:   {args.config}")
    for split, count in manifest["counts"].items():
        print(
            f"  {split}: 合成 {count['synthetic']} + 真实 {count['real']} "
            f"= {count['total']}（逐 cell 50:50）"
        )
    for name, count in manifest["stress"].items():
        print(f"  {name}: 合成 {count['synthetic']} + 真实 {count['real']}")
    print(f"实例总数: {manifest['dedup']['total_instances']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_assemble_cli.py -p no:cacheprovider --basetemp="$TEMP/ap_t7" -q 2>&1 | tail -3
```

预期：`6 passed`。

- [ ] **Step 5: 全量测试**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t7all" -q 2>&1 | tail -3
```

预期：`291 passed`，0 failed、0 skipped。

- [ ] **Step 6: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/data/assemble/cli.py tests/test_assemble_cli.py && git commit -m "feat(data): 语料装配命令行入口"
```

---

## Task 8: 真实语料装配与验收

**Files:**
- Modify: `datasets/processed/`（产物，被 gitignore）
- Create: `data_manifest.json`（**入库**）
- Create: `experiment_config.yaml`（**入库**）

- [ ] **Step 1: 删掉上游遗留的空目录**

`datasets/processed/val/` 与项目其余部分（`SPLITS`、`datasets/raw/generation/validation`、§10）的 `validation` 不一致。**已实测确认 `train`/`test`/`val` 三个目录都是空的**，所以删 `val` 不损失任何数据。

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ls -A datasets/processed/val/ | wc -l && rmdir datasets/processed/val && echo "removed" && ls datasets/processed/
```

预期：先打印 `0`，然后 `removed`。若打印的不是 0，**停下来报告**，不要删。

- [ ] **Step 2: 跑全量测试，确认起点是绿的**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t8" -q 2>&1 | tail -3
```

预期：`307 passed`（291 + §0.9 的 16 条金样本），0 failed、**不得出现 skipped / xfailed / error**。若此处不是全绿，停下来报告，不要继续。

- [ ] **Step 3: 跑真实装配**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m experiment.data.assemble.cli; echo "EXIT=$?"
```

预期：退出码 0，打印三个 split 与两个应力集。**必须看到 `EXIT=0`**；只看到打印的成功信息不算通过。

- [ ] **Step 4: 核对行数**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && for f in train validation test; do printf "%-12s %s\n" "$f" "$(wc -l < datasets/processed/$f/instances.jsonl)"; done && for f in e3_size e4_tail; do printf "%-12s %s\n" "$f" "$(wc -l < datasets/processed/stress/$f/instances.jsonl)"; done
```

预期：`train 12000`、`validation 2400`、`test 2400`、`e3_size 600`、`e4_tail 300`。
**任何一项不符都是失败**，不要"大概对"。

- [ ] **Step 5: 独立复核**

用 Write 工具在**仓库之外**创建 `C:\Users\swj17\AppData\Local\Temp\ap_verify\verify.py`（**不要**在仓库里创建任何文件）：

```python
"""独立复核装配产物。只读，不写仓库。"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\work\学校\A.SPBU\Saint-peter-26\AI-2\project")
PROC = ROOT / "datasets" / "processed"
LOTS = ROOT / "datasets" / "raw" / "real" / "lots_clean.csv"

problems = []


def lines_of(path):
    if not path.is_file():
        problems.append(f"缺少文件 {path}")
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]


CORPUS = [PROC / s / "instances.jsonl" for s in ("train", "validation", "test")]
CORPUS += [
    PROC / "stress" / n / "instances.jsonl" for n in ("e3_size", "e4_tail")
]

# 1) 逐 cell 50:50
for split in ("train", "validation", "test"):
    per_cell = defaultdict(Counter)
    for line in lines_of(PROC / split / "instances.jsonl"):
        inst = json.loads(line)
        per_cell[(inst["n"], inst["alpha"])][inst["source"]] += 1
    if len(per_cell) != 12:
        problems.append(f"{split}: cell 数为 {len(per_cell)}，期望 12")
    for cell, counts in sorted(per_cell.items()):
        if counts["synthetic"] != counts["real"]:
            problems.append(
                f"{split} {cell}: 合成 {counts['synthetic']} != 真实 {counts['real']}"
            )
    print(f"{split}: {len(per_cell)} 个 cell 的逐 cell 50:50 检查完毕")

# 2) 全局唯一 + 不变量
seen = {}
checked = 0
for path in CORPUS:
    for line in lines_of(path):
        inst = json.loads(line)
        checked += 1
        h = inst["canonical_hash"]
        if h in seen:
            problems.append(f"哈希重复: {inst['instance_id']} 与 {seen[h]}")
        seen[h] = inst["instance_id"]
        if (inst["family"] is None) != (inst["source"] == "real"):
            problems.append(f"{inst['instance_id']}: family/source 不变量被破坏")
        if inst["family"] == "skewed_e4" and path.parent.name != "e4_tail":
            problems.append(f"{inst['instance_id']}: skewed_e4 出现在 {path}")
        if not (
            len(inst["weights"]) == len(inst["values"]) == len(inst["item_order"]) == inst["n"]
        ):
            problems.append(f"{inst['instance_id']}: 长度不一致")
        if min(inst["weights"]) <= 0 or min(inst["values"]) <= 0:
            problems.append(f"{inst['instance_id']}: 出现非正的 w 或 v")
print(f"实例 {checked} 条，唯一哈希 {len(seen)} 个")

# 3) 溯源回连
table = {}
with LOTS.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        table[int(row["source_row_id"])] = row

checked_back = 0
for split in ("train", "validation", "test"):
    for line in lines_of(PROC / split / "instances.jsonl"):
        if checked_back >= 500:
            break
        inst = json.loads(line)
        if inst["source"] != "real":
            continue
        for rid, w, v, inv in zip(
            inst["source_row_ids"], inst["weights"], inst["values"], inst["invoice_nos"]
        ):
            row = table[rid]
            if int(row["quantity_w"]) != w or int(row["value_pence"]) != v:
                problems.append(f"{inst['instance_id']} 行 {rid} 回连不符")
            if row["invoice_no"] != inv:
                problems.append(f"{inst['instance_id']} 行 {rid} 发票号不符")
            if row["invoice_date"][:10] != inst["source_date"]:
                problems.append(f"{inst['instance_id']} 行 {rid} 源日期不符")
        checked_back += 1
print(f"回连复核 {checked_back} 条真实实例")

# 4) 应力集：E3 真实只用 test 日期；E3/E4 与主集不相交
main_hashes = set()
for path in CORPUS[:3]:
    for line in lines_of(path):
        main_hashes.add(json.loads(line)["canonical_hash"])

# test 段的日期窗口取自 manifest——那是装配时记下的事实。不用主集 test 实例
# 「恰好抽到过」的日期集合来判定：那只是合格 test 日期的一个子集（46 个里抽
# 1200 次未必全覆盖），拿子集去判会把合法的 test 日期误报成违规。
manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
window_lo = manifest["splits"]["test"]["date_from"]
window_hi = manifest["splits"]["test"]["date_to"]
train_end = manifest["splits"]["train"]["date_to"]
print(f"test 日期窗口: {window_lo} .. {window_hi}  (train 止于 {train_end})")

for name, expected in (("e3_size", (300, 300)), ("e4_tail", (300, 0))):
    path = PROC / "stress" / name / "instances.jsonl"
    hashes = set()
    source_counts = Counter()
    for line in lines_of(path):
        inst = json.loads(line)
        hashes.add(inst["canonical_hash"])
        source_counts[inst["source"]] += 1
        if name == "e3_size" and inst["source"] == "real":
            day = inst["source_date"]
            if day <= train_end or not (window_lo <= day <= window_hi):
                problems.append(f"E3 真实用了非 test 日期: {day}")
    overlap = hashes & main_hashes
    if overlap:
        problems.append(f"{name} 与主集有 {len(overlap)} 个重复哈希")
    got = (source_counts["synthetic"], source_counts["real"])
    if got != expected:
        problems.append(f"{name} 配额为 {got}，期望 {expected}")
    print(f"{name}: 合成 {got[0]} + 真实 {got[1]}，与主集无交集")

print()
print(f"问题数: {len(problems)}")
for problem in problems[:20]:
    print("  ", problem)
sys.exit(1 if problems else 0)
```

运行：

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe "C:/Users/swj17/AppData/Local/Temp/ap_verify/verify.py"; echo "EXIT=$?"
```

预期：`问题数: 0`，`EXIT=0`。**必须看到 EXIT=0。**

- [ ] **Step 6: 复核可复现性——再跑一次，比对五个语料文件的字节**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -c "import hashlib,pathlib; [print(p.as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(pathlib.Path('datasets/processed').rglob('instances.jsonl'))]" > "$TEMP/ap_verify/before.txt" && ./venv/Scripts/python.exe -m experiment.data.assemble.cli > /dev/null; echo "EXIT=$?" && ./venv/Scripts/python.exe -c "import hashlib,pathlib; [print(p.as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(pathlib.Path('datasets/processed').rglob('instances.jsonl'))]" > "$TEMP/ap_verify/after.txt" && diff "$TEMP/ap_verify/before.txt" "$TEMP/ap_verify/after.txt" && echo "语料文件字节一致" && cat "$TEMP/ap_verify/after.txt"
```

预期：`EXIT=0`，最后打印「语料文件字节一致」并列出五行哈希。

**注意**：`data_manifest.json` 里的 `generated_at` 是时钟，所以 manifest 本身**不该**逐字节相同——只有五个 `instances.jsonl` 是。

- [ ] **Step 7: 确认两个协议产物在仓库根目录且会被 git 跟踪**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ls -l data_manifest.json experiment_config.yaml && git status --short && git check-ignore -v data_manifest.json experiment_config.yaml; echo "check-ignore EXIT=$? (期望 1 = 未被忽略)"
```

预期：两个文件存在；`git status --short` 显示它们为未跟踪的新文件（`??`）；`check-ignore` 退出码为 1。

- [ ] **Step 8: 人工核对 manifest 与 config 的关键字段**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -c "
import json
m = json.load(open('data_manifest.json', encoding='utf-8'))
print('counts      :', m['counts'])
print('stress      :', m['stress'])
print('dates       :', {k: v['dates'] for k, v in m['splits'].items()})
print('boundaries  :', m['split_policy']['boundaries'])
print('eligible200 :', {k: v['200'] for k, v in m['split_policy']['eligible_dates'].items()})
print('eligible500 :', {k: v['500'] for k, v in m['split_policy']['eligible_dates'].items()})
print('tail        :', m['tail_thresholds'])
print('dedup       :', m['dedup'])
print('workbook    :', json.dumps(m['provenance']['workbook'], ensure_ascii=False))
print('lots rows   :', m['provenance']['lots_clean']['rows'])
print('files       :', [(f['path'], f['lines']) for f in m['files']])
"
```

预期：`counts` 为 train 12000 / validation 2400 / test 2400；`stress` 为 e3_size 300+300、e4_tail 300+0；`dates` 为 213/46/46；`boundaries` 为 213/259；`eligible500` 为 209/46/46；`lots rows` 为 522504；`dedup.total_instances` 为 **17700**（= 12000+2400+2400 **+600+300**：`merge.py` 的 `total_instances` 在三个主 split 之外**把两个应力集也算进去**，见 `:252`；金样本同样印证——其夹具 `288+600+300=1188`，若不计 E4 则只有 888）；`files` 五行且行数与 Step 4 一致。

- [ ] **Step 9: 提交两个协议产物**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add data_manifest.json experiment_config.yaml && git commit -m "feat(data): 冻结混合语料与审计 manifest"
```

- [ ] **Step 10: 在全新 venv 中验证依赖可复现**

```bash
cd "$TEMP" && rm -rf ap_freshvenv && "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project/venv/Scripts/python.exe" -m venv ap_freshvenv && ap_freshvenv/Scripts/python.exe -m pip install -q -r "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project/requirements.txt" && cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && "$TEMP/ap_freshvenv/Scripts/python.exe" -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_t8fresh" -q 2>&1 | tail -3
```

预期：`307 passed`（291 + §0.9 的 16 条金样本），0 failed、0 skipped。

- [ ] **Step 11: 报告**

向控制者报告：五个语料文件的行数与 sha256、`data_manifest.json` 与 `experiment_config.yaml` 的 sha256、`问题数: 0` 的复核结果、逐字节复现结果、全新 venv 的测试结果，以及任何偏离。

---

## 完成之后

全部 8 个任务完成后，由控制者派一个最终的整体审查子代理（不是本计划的步骤），再走 `superpowers:finishing-a-development-branch`。
