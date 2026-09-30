# 子项目 B：两个基线（Random / Greedy）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结语料上产出 `baseline_results.csv`——Random 与 Greedy 两个基线的原始观测，3,300 个实例 × (10 次随机试验 + 1 次贪心) = **36,300 行**。

**Architecture:** 新建 `experiment/baselines/` 包，与 `experiment/eval/` 平级、互不依赖，只经由 `experiment.eval.{reference,metrics}` 的四个公开函数复用原语。三个模块各司其职：`seeds.py` 把 `(base, instance_id, trial)` 确定地映射成 64 位种子；`methods.py` 是**纯函数**——实例进、解出，不碰文件系统也不看时钟；`cli.py` 负责读配置、读语料、计时、落盘。计时只包住**单次**构造；列序与行序都是产物的一部分。

**Tech Stack:** Python 3.12、标准库 `hashlib` / `random` / `csv` / `argparse`、`PyYAML`（读 `experiment_config.yaml`）、`pytest`。**不引入新依赖。**

**依据：** `docs/superpowers/specs/2026-09-30-baselines-design.md`（下称"设计 §N"）。设计里对 `EXPERIMENT_PLAN.md` 的引用写作"计划 §N"。

---

## 前置事实（已实测，2026-09-30）

以下数字都由探针脚本在**真实冻结语料**上实测得到（`probe2.py` 锁定合成字面量、`probe3.py` 跑全语料）。测试里出现的每个期望值都出自这些记录，不是手算的。

| 事实 | 实测值 |
| --- | --- |
| 语料实例数 | 3,300（`test` 2,400 + `e3_size` 600 + `e4_tail` 300） |
| `instance_id` 唯一性 | 3,300/3,300，**0 重复** |
| `instance_id` 含 `\|` 的个数 | **0** |
| 带 `item_order` 的实例 | 3,300/3,300 |
| 每个 `instance_id` 都能在 `reference_results.csv` 里找到 | 0 缺失 |
| 产出行数 | 36,300 |
| 全部行的 `feasibility_violations` 之和 | **0** |
| 全部行里 `value > V*` 的个数 | **0** |
| Greedy 平均 `V/U` | **0.975118** |
| Random 平均 `V/U` | **0.559955** |
| 两法差额 | 0.415163（Greedy 严格更高） |
| 语料的 n 取值 | {20, 50, 100, 200, 500}，最小 20 |
| Random 抽到空集的试验数 | **0**（n≥20 ⇒ 全反面概率 ≤ 2⁻²⁰，33,000 次里期望约 0.03 次） |

> **为什么空集那一行要记。** 空集在合成夹具里会**真的出现**（见 Task 4），而在真实语料上出现 0 次。两者都对；但它们合起来说明一件事：产物里 `value` 为 0 的行**可能有也可能没有**，所以任何按 `value == 0` 数"平凡解"的统计都是错的（Task 5 有一条测试专门钉这一点）。

### 怎么跑测试：必须带 `--basetemp`

**本仓库当前无法使用 pytest 的默认 basetemp。** `%TEMP%\pytest-of-swj17` 这个目录已经
不可列举也不可写（实测：`ls` 报 Permission denied，`mkdir` 失败），它是用户的目录，**不
归本项目修复或删除**。在它之上跑测试的后果是**每一个用 `tmp_path` 的测试都报
PermissionError**——实测裸跑 `pytest -q` 得到 `224 passed, 126 errors`，其中 126 个错与
被测代码毫无关系。

```bash
# ✅ 本计划里每条 pytest 命令的形式
./venv/Scripts/python.exe -m pytest -q --basetemp="$TEMP/ai2-pytest"
# 得到 350 passed（B 之前）/ 398 passed（B 之后），0 skipped

# ❌ 不要这样跑：会把 126 个 PermissionError 当成自己的问题
./venv/Scripts/python.exe -m pytest -q
```

两点必须同时满足：**可写**，且**在仓库之外**。basetemp 若落在仓库里，`tmp_path` 会变成
`REPO_ROOT` 下的路径，而 `tests/test_assemble_golden.py::test_paths_are_posix_and_relative_inside_the_repo`
会因此翻成假失败（这是本项目早先踩过的坑）。`$TEMP/ai2-pytest` 两条都满足。

### 正向探针：这份计划的每一段代码都**跑过**

规矩是「派发之前先在临时树里探一遍」。这次探的方式是把本计划的**每一个代码 fence 逐字
抽出来**落到 `%TEMP%/probe-b/scratch`（仓库副本 + `datasets/`），所以探的就是计划里那段
代码，不是重新打一遍的版本。

实测结果：

| 检查 | 结果 |
| --- | --- |
| Task 1 改名后 A 的全套件 | **350 passed**（与改动前逐字相同） |
| Task 1 的残留检查 `\b_ratio_order\b` | 无输出（裸模式会假命中测试名，见 Task 1 Step 5） |
| 四个新单元测试文件 | **4 / 10 / 14 / 14 passed** |
| Task 6 全语料验收 | **6 passed**（真跑，不是 skipped） |
| 全套件 | **398 passed, 0 skipped** |
| Task 7 端到端 | 36,300 行、`wc -l` = 36,301、耗时 1.8 s |
| 覆写守卫 | 第二次裸跑退出码 1、文件未改动 |

**探针查出并已在本计划里修掉的四处缺陷**（都是计划自身的，不是被测代码的）：

1. **`grep` 残留检查不会静默**：裸 `_ratio_order` 会命中第 71 行的**测试名**（子串匹配）。
   → 改用 `\b` 词边界，并写明这次假命中。
2. **Greedy 测试数是 10 不是 9**：Greedy 有 **5** 个行为测试（不是 4 个），加 5 个参数化
   用例。→ 计划里所有相关数字改正（含全套件总数 397 → **398**）。
3. **全语料验收的集合断言写反了**：`set(by_instance) == set(reference)` 当场失败——
   `reference_results.csv` 覆盖**全部 17,700 个**实例（含 `train`/`validation`），而 B 只跑
   3,300 个，右边多出 14,400 个。→ 改成**包含**关系 `<=` 加 `len(...) == INSTANCES`。
4. **覆写守卫的中文提示在 GBK 控制台上是乱码**：`stdout` 重设了编码，`stderr` 没有，而本
   函数是本项目第一个往 `stderr` 写字的入口。→ 加上 `sys.stderr.reconfigure`，实测修复。

> 第 3 条是这次探针最值钱的一次发现：它是一条**会当场失败**的断言，而失败原因不在实现里，
> 在断言写错了参照集。若不探，实现者会花时间去找一个并不存在的"实例集错位"缺陷。

---

## 文件结构

| 动作 | 路径 | 职责 |
| --- | --- | --- |
| 创建 | `experiment/baselines/__init__.py` | 包说明，只有 docstring（镜像 `experiment/eval/__init__.py`） |
| 创建 | `experiment/baselines/seeds.py` | `trial_seed`：试验种子派生 |
| 创建 | `experiment/baselines/methods.py` | `BaselineSolution` / `greedy_solution` / `random_solution`：纯构造 |
| 创建 | `experiment/baselines/cli.py` | 读配置 → 跑语料 → 写 `baseline_results.csv` |
| 创建 | `tests/test_baselines_seeds.py` | 种子派生 |
| 创建 | `tests/test_baselines_greedy.py` | Greedy 的协议（5 个行为测试 + 1 个参数化一致性测试） |
| 创建 | `tests/test_baselines_random.py` | Random 的协议（5 个测试） |
| 创建 | `tests/test_baselines_cli.py` | 列序、行序、字节、覆写守卫、计时口径（14 个测试） |
| 创建 | `tests/test_baselines_corpus.py` | 设计 §10 的全语料验收（语料缺失时整体跳过） |
| 创建 | `baseline_results.csv` | 产物，仓库根目录，**入库**（Task 7 生成） |
| 修改 | `experiment/eval/reference.py:43,75,44-51` | `_ratio_order` → `ratio_order`（改 def 与 docstring，不改行为） |
| 修改 | `tests/test_eval_upper_bound.py:8,77-80` | 导入改名 + 4 条断言改名 |

---

## Task 1: `_ratio_order` 提升为公开的 `ratio_order`

**Files:**
- Modify: `experiment/eval/reference.py:43`（def 行）、`:44-51`（docstring）、`:75`（**唯一**的生产调用点）
- Modify: `tests/test_eval_upper_bound.py:8`（导入）、`:77-80`（4 条断言）

**背景：** A 的 `reference.py` 里有一个私有函数 `_ratio_order`，它的次序契约正是计划 §5 的 Greedy 次序（整数叉乘排 `v_i/w_i` 降序，同分取更小的 `w`，再取原索引升序）。B 的 Greedy 必须用**同一个**函数而不是抄一份：两份实现会在某次修订后分歧，而分歧的症状是"上界与贪心用了不同的次序"——一个既不报错也不崩溃、只在数字上体现的偏差（设计 §6）。

**只改名、不改行为。** 返回次序的契约一字不变。

**这不是 TDD 任务**：被测的行为已经存在且已被测试钉住，所以本任务的"测试"就是 A 的整套测试仍然是 350 passed。改名要做的只是不留残留。

- [ ] **Step 1: 改 def 行与 docstring**

`experiment/eval/reference.py` 第 43 行：

```python
def ratio_order(weights: Sequence[int], values: Sequence[int]) -> list[int]:
```

docstring 末尾（"使完整排序确定，U 不依赖 ``sort`` 的稳定性。"之后）**追加**一段，说明它为什么是公开的：

```python
    **公开是因为它同时是 Greedy 基线的次序**（`EXPERIMENT_PLAN.md` 第 5 节）：
    分数背包的贪心与经典 0/1 贪心按同一个比值序扫描。两份实现会在某次修订后
    分歧，而分歧的症状是「上界与贪心用了不同的次序」——一个不报错、不崩溃、
    只在数字上体现的偏差。见
    `docs/superpowers/specs/2026-09-30-baselines-design.md` §6。
```

- [ ] **Step 2: 改生产调用点**

`experiment/eval/reference.py:75`，在 `fractional_upper_bound` 内部：

```python
    order = ratio_order(weights, values)
```

- [ ] **Step 3: 改测试的导入与 4 条断言**

`tests/test_eval_upper_bound.py:8`：

```python
    ratio_order,
```

`tests/test_eval_upper_bound.py:77-80`——四行里的函数名都去掉下划线前缀：

```python
    assert ratio_order([1, 2], [2, 4]) == [0, 1]              # 同比值：w 小者先
    assert ratio_order([2, 1], [4, 2]) == [1, 0]              # 打乱输入，仍取 w 小的那件
    assert ratio_order([5, 5, 5], [7, 7, 7]) == [0, 1, 2]     # 全同：按原索引
    assert ratio_order([1, 3, 2], [3, 1, 2]) == [0, 2, 1]     # 比值 3、1/3、1
```

> 第 71 行的测试名 `test_ratio_order_breaks_ties_deterministically` **不要动**：它含有子串 `_ratio_order`（`test` + `_ratio_order` + …），但名字本身没有下划线前缀。

- [ ] **Step 4: 跑全套件，确认 350 passed / 0 skipped**

Run: `./venv/Scripts/python.exe -m pytest -q --basetemp="$TEMP/ai2-pytest"`

Expected: `350 passed`（与改动前逐字相同）。任何一条变了都说明改的是行为，不是名字。

- [ ] **Step 5: 确认代码里没有残留**

Run: `grep -rnE "\b_ratio_order\b" --include=*.py .`

Expected: **无输出**。

> **必须用词边界。** 裸 `grep -rn "_ratio_order" --include=*.py .` **不会**静默：它会命中
> `tests/test_eval_upper_bound.py:71` 的 `def test_ratio_order_breaks_ties_deterministically()`
> ——那里的 `test` + `_ratio_order` 拼出了这个子串，而这个名字（已确认）**不需要改**。
> 探针实测到了这次假命中。`\b` 在 `t` 与 `_` 之间不成立（两者都是词字符），所以加了边界的
> 模式只匹配真正的引用。

- [ ] **Step 6: 提交**

```bash
git add experiment/eval/reference.py tests/test_eval_upper_bound.py
git commit -m "refactor(eval): _ratio_order 提升为公开的 ratio_order，供 Greedy 基线共用"
```

---

## Task 2: 包骨架与种子派生

**Files:**
- Create: `experiment/baselines/__init__.py`
- Create: `experiment/baselines/seeds.py`
- Test: `tests/test_baselines_seeds.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_baselines_seeds.py`：

```python
"""试验种子的派生（设计 §4）。"""

import subprocess
import sys

from experiment.baselines.seeds import trial_seed
from experiment.data.spec import REPO_ROOT

#: 全部期望值由探针实测得到（probe2.py，2026-09-30）。
INSTANCE_ID = "syn-test-uniform-n20-a25-i0000"
BASE = 6001


def test_the_seed_is_the_measured_literal():
    """钉住算法本身：BLAKE2b（8 字节）、键 ``base|id|trial``、大端。"""
    assert trial_seed(INSTANCE_ID, 0, BASE) == 13096480787329443436
    assert trial_seed(INSTANCE_ID, 1, BASE) == 11617834981237195108
    assert trial_seed(INSTANCE_ID, 9, BASE) == 14240787302913443677


def test_seed_is_stable_across_processes():
    """**跨进程**稳定：全新解释器里给出同一个种子。

    进程内比较是测不出来的：``hash(str)`` 每个进程用不同的随机盐，同一个进程
    里当然自洽。用 ``hash()`` 的实现会让每一次运行都产出不同的
    ``baseline_results.csv``，而上一条测试照样通过。

    子进程的 ``cwd`` 设成仓库根目录，``python -c`` 才会把 ``experiment`` 放进
    ``sys.path``。
    """
    code = (
        "import sys\n"
        "from experiment.baselines.seeds import trial_seed\n"
        "sys.stdout.write(str(trial_seed(%r, 0, %d)))\n" % (INSTANCE_ID, BASE)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        cwd=str(REPO_ROOT),
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout == "13096480787329443436"


def test_every_component_changes_the_seed():
    """三个分量各自都进哈希。

    少进一个分量（比如忘了 ``trial``）的实现，会让十次试验抽到同一串硬币、
    十行数字完全相同——而上面两条测试仍然通过。
    """
    base = trial_seed(INSTANCE_ID, 0, BASE)
    assert trial_seed(INSTANCE_ID, 1, BASE) != base            # trial 进了
    assert trial_seed(INSTANCE_ID, 0, BASE + 1) != base        # base 进了
    assert trial_seed(INSTANCE_ID + "x", 0, BASE) != base      # instance_id 进了


def test_the_seed_uses_all_eight_digest_bytes():
    """种子用满 64 位，而不是只用低 32 位。

    若 ``digest_size`` 被改成 4，所有种子都小于 ``2**32``。3,300 实例 × 10 次
    试验 = 33,000 次抽取，在 2**32 空间里的生日碰撞概率约 **12%**，在 2**64**
    里约 3e-11；碰撞意味着两次"独立"试验抽到同一串硬币，取平均时它们是重复
    计数。

    用 1,000 次抽取的**最大值**而不是单个字面量：单个字面量只能证明这一个种子，
    最大值的量级才说明整个分布用满了空间。
    """
    seeds = [
        trial_seed(f"syn-test-uniform-n20-a25-i{i:04d}", trial, BASE)
        for i in range(100)
        for trial in range(10)
    ]
    assert len(seeds) == 1000
    assert max(seeds) >= 2**32
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_seeds.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: FAIL — `ModuleNotFoundError: No module named 'experiment.baselines'`

- [ ] **Step 3: 写包骨架**

创建 `experiment/baselines/__init__.py`：

```python
"""两个基线方法：Random 与 Greedy。

依据 `EXPERIMENT_PLAN.md` 第 5 节。本包与 ``experiment.eval`` **平级而不是
合并**：评测服务不是第四个参赛方法（`experiment/eval/__init__.py` 有详述），
而这里是参赛方法中用于对比 DQN 的那两个基线。两者混在一起正是那一节想避免的
混淆。
"""
```

- [ ] **Step 4: 写实现**

创建 `experiment/baselines/seeds.py`：

```python
"""试验种子的派生（设计 §4）。

一个随机试验的身份是三元组 ``(evaluation_seed, instance_id, trial)``。本模块
把它映射成一个确定的 64 位整数种子。

**刻意不用 Python 的 ``hash()``**：``hash(str)`` 每个进程用不同的随机盐
（PYTHONHASHSEED），进程内自洽、跨进程不稳定。而 ``baseline_results.csv`` 是要
入库的产物，"换个进程就变"等于复现性直接失效。BLAKE2b 没有这个问题。
"""

from __future__ import annotations

import hashlib

#: 摘要取 8 字节 → 种子恰好落在 64 位整数范围内，不必再掩码。
#:
#: 取 4 字节也"够用"，但 33,000 次抽取在 2**32 空间里的生日碰撞概率约 12%
#: （在 2**64 里约 3e-11）。碰撞意味着两次"独立"的试验抽到同一串硬币，
#: 取平均时它们是重复计数。取满 8 字节没有额外成本。
_DIGEST_BYTES = 8


def trial_seed(instance_id: str, trial: int, base: int) -> int:
    """``(base, instance_id, trial)`` → 稳定的 64 位种子。

    键序刻意写成 ``base | instance_id | trial``（与实参顺序不同）：三段用 ``|``
    分隔，而语料的 ``instance_id`` 不含 ``|``（实测 3,300/3,300），所以这个拼接
    不会让两段黏成一个。这条依赖由 ``tests/test_baselines_corpus.py`` 顺带断言。

    种子是**键的函数**，不依赖任何全局计数器，也不依赖语料的文件顺序：同一次
    试验在两次运行里、在两台机器上、在并行测试中都是同一个种子。
    """
    payload = "|".join((str(base), instance_id, str(trial)))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES)
    return int.from_bytes(digest.digest(), "big")
```

- [ ] **Step 5: 跑测试，确认通过**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_seeds.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: `4 passed`

- [ ] **Step 6: 提交**

```bash
git add experiment/baselines/__init__.py experiment/baselines/seeds.py tests/test_baselines_seeds.py
git commit -m "feat(baselines): 试验种子的内容寻址派生"
```

---

## Task 3: `methods.py` 的 Greedy

**Files:**
- Create: `experiment/baselines/methods.py`
- Test: `tests/test_baselines_greedy.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_baselines_greedy.py`：

```python
"""Greedy 基线（设计 §2.1、§11）。"""

import pytest

from experiment.baselines.methods import greedy_solution
from experiment.eval.reference import solve


def _instance(instance_id, capacity, values, weights):
    """按语料的字段形状造实例。

    ``item_order`` 必须给：它是实例的一个字段（设计 §2.2），Random 会读它。
    Greedy 不读它，但测试用的输入应当与真实输入**同一个形状**，否则这里验的
    就不是产物那条路径。
    """
    return {
        "capacity": capacity,
        "instance_id": instance_id,
        "item_order": list(range(len(weights))),
        "values": values,
        "weights": weights,
    }


def test_the_scan_follows_the_ratio_order_not_the_index_order():
    """高比值的那件先装，即使它的索引更大。

    ``w=[10,5]``、``v=[10,6]``、``C=10``：比值 1 与 1.2，``ratio_order = [1, 0]``。
      按比值：先装 1（w=5），剩 5 装不下 0 → value **6**。
      按索引：先装 0（w=10），容量用尽 → value 10（下面显式算出这个对照值）。

    期望值 6 因此同时钉住"排序发生了"和"排序用的是比值"，而不只是"排了序"。
    """
    instance = _instance("ratio-order", 10, [10, 6], [10, 5])
    solution = greedy_solution(instance)
    assert solution.selected_indices == (1,)
    assert solution.value == 6
    assert solution.weight_used == 5

    # 对照：按索引顺序扫描会得到 10。不写出这个数，「6」就可能只是一个被钉住的错值。
    remaining, index_value = 10, 0
    for i in range(2):
        if instance["weights"][i] <= remaining:
            index_value += instance["values"][i]
            remaining -= instance["weights"][i]
    assert index_value == 10


def test_a_rejected_item_does_not_stop_the_scan():
    """装不下就跳过，**不提前收工**（计划 §5 的 "continue after a rejected item"）。

    ``w=[10,20,3]``、``v=[60,100,3]``、``C=15``：比值序为 ``[0,1,2]``。装下 0
    （w=10，剩 5），1 装不下——**跳过**——再装下 2（w=3），得到 value **63**。
    把 ``continue`` 换成 ``break`` 会停在 60。
    """
    instance = _instance("continue", 15, [60, 100, 3], [10, 20, 3])
    solution = greedy_solution(instance)
    assert solution.selected_indices == (0, 2)
    assert solution.value == 63
    assert solution.weight_used == 13

    # 对照：把跳过换成收工会得到 60。
    remaining, break_value, taken = 15, 0, []
    for i in (0, 1, 2):
        if instance["weights"][i] > remaining:
            break
        taken.append(i)
        break_value += instance["values"][i]
        remaining -= instance["weights"][i]
    assert break_value == 60


def test_ties_are_broken_by_the_lower_weight():
    """比值并列时取 ``w`` 小的那件——**这条在 value 上就能看见**。

    ``w=[1,2]``、``v=[2,4]``、``C=2``：两件比值都是 2。取 ``w`` 小的那件（索引 0）
    后剩 1 容量，装不下索引 1 → value **2**。
    若把次级键写成"w 大者先"，会先装索引 1 → value **4**。下面的对照算出这个 4。
    """
    instance = _instance("tie-weight", 2, [2, 4], [1, 2])
    solution = greedy_solution(instance)
    assert solution.selected_indices == (0,)
    assert solution.value == 2

    # 对照：次级键方向写反会得到 4。
    remaining, wrong_value = 2, 0
    for i in (1, 0):
        if instance["weights"][i] <= remaining:
            wrong_value += instance["values"][i]
            remaining -= instance["weights"][i]
    assert wrong_value == 4


def test_ties_with_equal_weights_are_asserted_on_the_set_not_the_value():
    """比值并列**且** ``w`` 也相同：只能断言集合。

    ``v/w`` 相等且 ``w`` 相等 ⇒ ``v`` 也相等，三件物品**完全可互换**，装哪一件
    ``value`` 与 ``weight_used`` 都一模一样。这个次序差异只在 ``selected_indices``
    / ``item_set_hash`` 上可见——而 ``item_set_hash`` 正是入库产物的一列。

    所以这里必须断言集合：把次级键改成索引**降序**，下面那两条值断言仍然全绿，
    于是它会变成一条"声称在测次序、实际测不了"的断言。
    """
    instance = _instance("tie-all-equal", 5, [7, 7, 7], [5, 5, 5])
    solution = greedy_solution(instance)
    assert solution.selected_indices == (0,)
    # 值断言在这里是恒真的：三件可互换 ⇒ 取哪件都是 7 / 5。
    assert solution.value == 7
    assert solution.weight_used == 5


def test_greedy_can_be_strictly_beaten_by_the_exact_optimum():
    """经典的 0/1 反例，且**由参考解度量**。

    ``w=[10,20,30]``、``v=[60,100,120]``、``C=50``：比值序为 ``[0,1,2]``，装入
    0 与 1 后剩 20，装不下 30 → value **160**。而恰好的最优是 {1,2} = **220**。

    把 160 和 220 摆在一起，这条测试才说明"贪心不是精确解"；只断言 160 的话，
    它可能只是一个被钉住的错值。
    """
    instance = _instance("beaten", 50, [60, 100, 120], [10, 20, 30])
    assert greedy_solution(instance).value == 160
    assert solve(instance).value == 220


#: (name, capacity, values, weights)
CASES = [
    ("a", 10, [10, 6], [10, 5]),
    ("b", 15, [60, 100, 3], [10, 20, 3]),
    ("c", 50, [60, 100, 120], [10, 20, 30]),
    ("d", 3, [4, 5], [2, 6]),
    ("e", 100, [1, 2, 3], [1, 2, 3]),
]


@pytest.mark.parametrize("name,capacity,values,weights", CASES)
def test_the_three_returned_fields_agree_with_each_other(name, capacity, values, weights):
    """三个字段互相一致，且解可行。

    逐字段**重算**而不是信任实现：``weight_used`` 写成"集合重量"却存了别的东西
    （比如忘了在每次装入后扣减 ``remaining``）时，只有重算才发现得了。
    """
    instance = _instance(name, capacity, values, weights)
    solution = greedy_solution(instance)
    selected = list(solution.selected_indices)
    assert selected == sorted(set(selected))                     # 升序且无重复
    assert all(0 <= i < len(weights) for i in selected)          # 索引在范围内
    assert solution.value == sum(values[i] for i in selected)
    assert solution.weight_used == sum(weights[i] for i in selected)
    assert solution.weight_used <= capacity                      # 可行
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_greedy.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: FAIL — `ModuleNotFoundError: No module named 'experiment.baselines.methods'`

- [ ] **Step 3: 写实现**

创建 `experiment/baselines/methods.py`：

```python
"""两个基线方法的构造（设计 §2）。

本模块是**纯函数**：实例进、解出。不读文件、不写文件、不看时钟——计时归
``cli.py``。这样"方法是否正确"与"产物是否正确"可以分开测，两个方法也能在测试
里被直接调用，不必造配置文件或临时目录。
"""

from __future__ import annotations

import random
from typing import Mapping, NamedTuple, Sequence

from experiment.eval.reference import ratio_order


class BaselineSolution(NamedTuple):
    """一个基线的解：设计 §8 的三列原始观测。

    **不复用 A 的 ``ReferenceSolution``**：它带 ``upper_bound`` 与 ``certified``，
    基线产不出这两个量，硬塞进去只能填假值——而一个填了假值的字段迟早会被当成
    真值读。
    """

    #: 被选物品索引，**按 ID 升序**。
    #:
    #: 升序不是为了好看：``item_set_hash`` 对输入次序本就不敏感，而升序让"两次
    #: 运行逐格比对"这种最朴素的复核也能直接用。
    selected_indices: tuple[int, ...]

    #: 被选集合的价值总和，**原始单位**（不是奖励整形后的值）。
    value: int

    #: 被选集合占用的容量。可行的解满足 ``<= instance["capacity"]``。
    weight_used: int


def _totals(instance: Mapping, selected: Sequence[int]) -> tuple[int, int]:
    """由集合现算 ``(value, weight_used)``。

    刻意不把这两个数作为独立状态维护：三份数据各自漂移是本项目的旧病，而它们
    必须永远一致。
    """
    values, weights = instance["values"], instance["weights"]
    return (
        sum(values[i] for i in selected),
        sum(weights[i] for i in selected),
    )


def greedy_solution(instance: Mapping) -> BaselineSolution:
    """经典 0/1 贪心：按 ``v/w`` 降序扫描，装得下就装。

    依据 `EXPERIMENT_PLAN.md` 第 5 节。排序用 ``ratio_order``——**与分数背包上界
    ``U`` 共用同一个函数**，理由见设计 §6：两份实现会在某次修订后分歧，而症状是
    "上界与贪心用了不同的次序"。

    **装不下时跳过继续，不是停下。** ``break`` 是更"自然"的写法，而且它在多数
    实例上给出同样的答案——只有在一个轻物品被重物品挡在后面时才暴露。计划 §5
    专门写了 "continue after a rejected item"，所以它由合成实例钉住（本文件的
    ``test_a_rejected_item_does_not_stop_the_scan``）。

    **不使用 ``item_order``**：它自己排序。实例里那份存储排列只对 Random 有意义。
    """
    weights = instance["weights"]
    remaining = instance["capacity"]
    selected: list[int] = []
    for i in ratio_order(weights, instance["values"]):
        if weights[i] > remaining:
            continue
        selected.append(i)
        remaining -= weights[i]
    selected.sort()
    return BaselineSolution(tuple(selected), *_totals(instance, selected))
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_greedy.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: `10 passed`（5 个行为测试 + 5 个参数化一致性用例）

- [ ] **Step 5: 提交**

```bash
git add experiment/baselines/methods.py tests/test_baselines_greedy.py
git commit -m "feat(baselines): Greedy 基线，与上界共用 ratio_order 次序"
```

---

## Task 4: `methods.py` 的 Random

**Files:**
- Modify: `experiment/baselines/methods.py`（追加 `random_solution`）
- Test: `tests/test_baselines_random.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_baselines_random.py`：

```python
"""Random 基线（设计 §2.2、§11）。"""

import random

import pytest

from experiment.baselines.methods import random_solution
from experiment.baselines.seeds import trial_seed

#: 探针实测的夹具（probe2.py）。``item_order`` 刻意**不是**恒等排列：它把装不下
#: 的物品 3（w=20）排在能装下的 1、4 之前，这正是读法 (a) 与 (b) 分歧的地方——
#: (a) 跳过 3 不铸币，(b) 先为 3 掷一枚再丢弃，两条硬币流从此错位。
INSTANCE = {
    "capacity": 12,
    "instance_id": "syn-test-uniform-n20-a25-i0000",
    "item_order": [2, 5, 0, 3, 1, 4],
    "values": [7, 3, 9, 100, 8, 2],
    "weights": [4, 9, 5, 20, 7, 6],
}
BASE = 6001

#: 十次试验的实测结果（probe2.py 输出，逐字抄录）。
EXPECTED_SETS = [(5,), (2, 5), (1,), (0, 5), (0, 2), (), (2,), (2, 5), (2,), (2, 5)]
EXPECTED_VALUES = [2, 11, 3, 9, 16, 0, 9, 11, 9, 11]


def _seeded(trial):
    return trial_seed(INSTANCE["instance_id"], trial, BASE)


def _reading_b(instance, seed):
    """读法 (b)：**每件都铸币**，装不下才丢弃。

    实现在测试内部独立写一份，用来证明两种读法确实不同（见下一条测试）。
    """
    weights = instance["weights"]
    rng = random.Random(seed)
    remaining = instance["capacity"]
    selected: list[int] = []
    for i in instance["item_order"]:
        coin = rng.random() < 0.5
        if weights[i] <= remaining and coin:
            selected.append(i)
            remaining -= weights[i]
    selected.sort()
    return tuple(selected)


def test_ten_trials_reproduce_the_measured_golden():
    """十次试验逐次复现实测的集合与价值（确定性 + 传播）。

    这一条同时覆盖设计 §11 的两行：
      - **确定性**：同 ``(instance, seed)`` 每次调用得到同一个解（用了全局
        ``random`` 状态的实现会在这里漂移）；
      - **传播**：不同 ``trial`` 的结果**至少有一对不同**——最后那条断言显式
        检查这一点，否则"十行各记自己的观测值"就是空话。
    """
    solutions = [random_solution(INSTANCE, _seeded(t)) for t in range(10)]
    assert [s.selected_indices for s in solutions] == EXPECTED_SETS
    assert [s.value for s in solutions] == EXPECTED_VALUES
    assert len(set(EXPECTED_VALUES)) > 1


def test_a_coin_is_only_flipped_for_items_that_fit():
    """只有装得下的物品才铸币（设计 §2.2 的读法 (a)）。

    这类测试最容易写成"无法失败"的样子：两种读法都是确定的、都跨版本可复现，
    所以"同一 seed 给出同一结果"对**两者都成立**，区分不了它们。必须先证明两种
    读法在这个夹具上**真的分歧**，断言才有内容——所以下面的 `assert disagreements`
    不是客套，它没有内容时这条测试就该失败。
    """
    seeds = [_seeded(t) for t in range(200)]
    reading_a = [random_solution(INSTANCE, s).selected_indices for s in seeds]
    reading_b = [_reading_b(INSTANCE, s) for s in seeds]
    disagreements = [i for i in range(200) if reading_a[i] != reading_b[i]]
    assert disagreements, "夹具区分不了两种读法，这条测试没有内容"
    assert disagreements[0] == 2          # 探针实测：首次分歧在试验 2

    assert reading_a[:10] == EXPECTED_SETS   # 实现站在 (a) 这一边


def test_the_walk_follows_the_stored_item_order():
    """次序取自实例的 ``item_order``，而不是 ``range(n)``。

    "Hold the stored item permutation fixed"（计划 §5）指的就是这个字段：一次
    试验沿它走，而它在 10 次试验之间**固定不变**。反排它之后，同一个种子必须给出
    不同的解——若实现忽略该字段、改按索引 0..n-1 走，这条会当场失败。

    这类偏差不报错、不崩溃，只留下一份与计划不符的产物，所以必须由测试钉住。
    """
    reversed_instance = dict(INSTANCE, item_order=list(reversed(INSTANCE["item_order"])))
    changed = [
        t
        for t in range(50)
        if random_solution(INSTANCE, _seeded(t)).selected_indices
        != random_solution(reversed_instance, _seeded(t)).selected_indices
    ]
    assert len(changed) == 48        # 探针实测：前 50 次里 48 次不同


@pytest.mark.parametrize("trial", range(10))
def test_every_trial_is_feasible_and_its_fields_agree(trial):
    """每一次试验的三个字段互相一致，且解可行（逐字段重算）。"""
    solution = random_solution(INSTANCE, _seeded(trial))
    selected = list(solution.selected_indices)
    assert selected == sorted(set(selected))
    assert solution.value == sum(INSTANCE["values"][i] for i in selected)
    assert solution.weight_used == sum(INSTANCE["weights"][i] for i in selected)
    assert solution.weight_used <= INSTANCE["capacity"]


def test_an_all_tails_draw_is_a_legal_empty_solution():
    """十次全反面那次返回空集——**这是合法结果，不是错误**。

    它与计划 §5 说的"平凡零最优实例"（``V* = 0``，没有物品装得下）**不是一回事**：
    这里的实例 ``V*`` 是正的，只是这一次抽签一件都没装。

    所以产物里**可能**出现 ``value`` 为 0 的行（实测：真实语料上 0 行，因为最小的
    ``n`` 是 20，全反面概率 ≤ 2⁻²⁰；合成夹具上会出现 1 行）。任何按 ``value == 0``
    数"平凡解"的统计都会把这两件事混为一谈——``cli.main()`` 因此不打印那种计数，
    由 ``tests/test_baselines_cli.py`` 钉住。
    """
    empty = [
        s
        for s in (random_solution(INSTANCE, _seeded(t)) for t in range(10))
        if not s.selected_indices
    ]
    assert [s.value for s in empty] == [0]
    assert [s.weight_used for s in empty] == [0]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_random.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: FAIL — `ImportError: cannot import name 'random_solution'`

- [ ] **Step 3: 写实现**

在 `experiment/baselines/methods.py` **末尾追加**（`random` 已在 Task 3 的 import 里）：

```python
def random_solution(instance: Mapping, seed: int) -> BaselineSolution:
    """随机基线：沿实例自带的 ``item_order`` 走一遍，逐件掷硬币。

    ``item_order`` 是实例记录里的**存储排列**（计划 §5 的 "stored item
    permutation"）：`generate/corpus.py` 与 `assemble/{real,stress}.py` 都由一条
    独立的 order 随机流生成它。它在 10 次试验之间**固定不变**，所以一次试验是沿
    它走，而不是沿索引 ``0..n-1`` 走——后者产出的仍是一个确定的、可复现的 CSV，
    只是它不是计划描述的那个实验。

    **只有装得下的物品才铸币**（设计 §2.2 的读法 (a)）。读法 (b)（每件都掷、
    装不下才丢弃）同样确定、同样跨版本可复现，所以"种子可复现"这条验收**区分不了**
    两者；区分靠 ``tests/test_baselines_random.py`` 里的对照实现。

    **平均由 D 做，本函数不取最好**：计划 §5 的 "Average outcomes rather than
    selecting the best trial" 是一句禁令，best-of-ten 同样确定、同样可复现，
    只是数字好看得多。

    随机源的用法（设计 §2.2，三条都是可复现性的依据）：
      - 种子传 **int**，不传 ``str``/``bytes``：``random.seed`` 对 ``str``/``bytes``
        的哈希经历过 ``version=`` 语义变更；
      - 用 ``random()`` 而不是 ``getrandbits()``：标准库保证 ``random()`` 在给定
        兼容种子时**跨版本产出同一序列**；
      - 用**实例化**的 ``Random(seed)``，不用模块级的 ``random.seed()``：后者是
        全局状态，会把"这一次试验"绑定到调用顺序上，并让并行的测试互相污染。
    """
    weights = instance["weights"]
    rng = random.Random(seed)
    remaining = instance["capacity"]
    selected: list[int] = []
    for i in instance["item_order"]:
        if weights[i] > remaining:
            continue  # 装不下：跳过，**不铸币**
        if rng.random() < 0.5:  # 装得下才铸币
            selected.append(i)
            remaining -= weights[i]
    selected.sort()
    return BaselineSolution(tuple(selected), *_totals(instance, selected))
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_random.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: `14 passed`（4 + 10 个参数化用例）

- [ ] **Step 5: 提交**

```bash
git add experiment/baselines/methods.py tests/test_baselines_random.py
git commit -m "feat(baselines): Random 基线，沿 item_order 走且只为装得下的物品铸币"
```

---

## Task 5: `cli.py`——语料 → `baseline_results.csv`

**Files:**
- Create: `experiment/baselines/cli.py`
- Test: `tests/test_baselines_cli.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_baselines_cli.py`：

```python
"""``baseline_results.csv`` 的形状、行序、字节、覆写守卫与计时口径（设计 §5、§8、§11）。"""

import csv
import json
import random
import time
from pathlib import Path

import pytest
import yaml

from experiment.baselines import cli
from experiment.baselines.methods import random_solution
from experiment.baselines.seeds import trial_seed
from experiment.data.spec import DATASETS_PROCESSED, E3_NAME, E4_NAME, REPO_ROOT

#: 两个 test 实例的 id **字典序与它们在文件里的出现顺序相反**，这样"行序按
#: instance_id 升序"才有判别力。
TEST_INSTANCES = [
    ("b-second", {"capacity": 5, "item_order": [0, 1], "values": [7, 1], "weights": [5, 1]}),
    ("a-first", {"capacity": 9, "item_order": [2, 0, 1], "values": [10, 7, 9], "weights": [8, 4, 5]}),
]
E3_INSTANCES = [
    ("c-e3", {"capacity": 6, "item_order": [1, 0], "values": [3, 4], "weights": [2, 6]}),
]

#: 计时测试用的规模：要让"一次构造"明显高于计时器噪声，又不拖慢套件。
TIMING_N = 400


@pytest.fixture
def corpus(tmp_path):
    """手写的三文件语料：test 两条、E3 一条、E4 空。

    刻意手写而不是跑 `run_assembly`：这里要测的是 **CLI 的读取与落盘**，不是装配；
    手写让"读到了哪些文件"完全可控。

    E4 留成**空文件而不是缺失**：真实语料里它存在（300 个实例），空文件能顺带钉住
    "空子集贡献 0 行"，而缺失文件只会让实现撞上 FileNotFoundError。
    """
    root = tmp_path / "processed"
    layout = {
        root / "test" / "instances.jsonl": TEST_INSTANCES,
        root / "stress" / E3_NAME / "instances.jsonl": E3_INSTANCES,
        root / "stress" / E4_NAME / "instances.jsonl": [],
    }
    for path, items in layout.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(dict(body, instance_id=name), sort_keys=True)
            for name, body in items
        ]
        path.write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return root


def _write_config(path: Path, trials: int, seed: int = 6001) -> Path:
    path.write_bytes(
        yaml.safe_dump(
            {"evaluation": {"evaluation_seed": seed, "random_trials_per_instance": trials}},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        ).encode("utf-8")
    )
    return path


@pytest.fixture
def config(tmp_path):
    return _write_config(tmp_path / "experiment_config.yaml", trials=2)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _argv(corpus, config, out, *extra):
    return [
        "baseline-cli",
        "--corpus", str(corpus),
        "--config", str(config),
        "--out", str(out),
        *extra,
    ]


# ---- 形状、行序、字节 ----


def test_every_subset_is_read_and_the_row_order_is_fixed(corpus, config, tmp_path):
    out = tmp_path / "baseline_results.csv"
    summary = cli.run(corpus, out, evaluation_seed=6001, trials=2)
    # test 2 + E3 1 + E4 0 = 3 个实例 × (2 次随机 + 1 次贪心)
    assert summary["rows"] == 9
    assert summary["by_method"] == {"greedy": 3, "random": 6}
    assert [(r["instance_id"], r["method"], r["trial"]) for r in _rows(out)] == [
        ("a-first", "greedy", ""),
        ("a-first", "random", "0"),
        ("a-first", "random", "1"),
        ("b-second", "greedy", ""),
        ("b-second", "random", "0"),
        ("b-second", "random", "1"),
        ("c-e3", "greedy", ""),
        ("c-e3", "random", "0"),
        ("c-e3", "random", "1"),
    ]
    assert [r["split"] for r in _rows(out)] == [
        "test", "test", "test", "test", "test", "test",
        E3_NAME, E3_NAME, E3_NAME,
    ]


def test_column_order_and_bytes_are_fixed(corpus, config, tmp_path):
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=2)

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")

    with out.open(encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == list(cli.COLUMNS)


def test_greedy_has_one_row_per_instance_with_no_seed(corpus, config, tmp_path):
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=2)
    greedy = [r for r in _rows(out) if r["method"] == "greedy"]
    assert len(greedy) == 3
    assert {r["trial"] for r in greedy} == {""}
    assert {r["seed"] for r in greedy} == {""}


def test_the_recorded_seed_is_the_trial_seed_of_that_row(corpus, config, tmp_path):
    """``seed`` 列必须等于该行 ``(instance_id, trial)`` 与评估种子的派生值。

    写进产物就是为了让复核者**不必重跑**就能重算任意一行；若这一列装的是别的
    东西（比如试验号本身），它就没有这个用处。
    """
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=2)
    for row in _rows(out):
        if row["method"] == "greedy":
            continue
        expected = trial_seed(row["instance_id"], int(row["trial"]), 6001)
        assert row["seed"] == str(expected), row


def test_trials_are_ordered_numerically_not_lexicographically(corpus, tmp_path):
    """``trial`` 列按**数值**升序。

    按字符串排会把 "10" 排在 "2" 前面：0, 1, 10, 11, 2, …。而 10 次试验的试验号是
    0..9，**全是单字符**，字符串序与数值序恰好一样——所以这条测试必须用 **12** 次
    试验才具判别力。
    """
    config = _write_config(tmp_path / "c12.yaml", trials=12)
    out = tmp_path / "baseline_results.csv"
    cli.run(corpus, out, evaluation_seed=6001, trials=12)
    for instance_id in ("a-first", "b-second", "c-e3"):
        trials = [
            r["trial"]
            for r in _rows(out)
            if r["instance_id"] == instance_id and r["method"] == "random"
        ]
        assert trials == [str(t) for t in range(12)], instance_id


def test_only_the_latency_column_is_unstable(corpus, config, tmp_path):
    """两次运行除 ``latency_s`` 外逐格相同（设计 §5）。

    该列是墙钟测量。**任何把整份 CSV 一起哈希**的校验都会重蹈 Plan 3 的覆辙：当时
    一个存活探针把含临时路径与时钟的 manifest 一起哈希，两次纯净运行都不一致，
    于是每一个"存活"判定都是假的。比对必须排除这一列。
    """
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.run(corpus, first, evaluation_seed=6001, trials=2)
    cli.run(corpus, second, evaluation_seed=6001, trials=2)

    def without_latency(path: Path) -> list[list[str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        index = rows[0].index("latency_s")
        return [[c for i, c in enumerate(row) if i != index] for row in rows]

    assert without_latency(first) == without_latency(second)


# ---- 参数来源与默认值 ----


def test_read_config_reads_both_parameters(tmp_path):
    config = _write_config(tmp_path / "c.yaml", trials=7, seed=123)
    assert cli.read_config(config) == (123, 7)


def test_default_paths_point_at_the_repo_root():
    """默认产物落在仓库根目录（设计 §8）。

    这条钉的是"产物入库"这个决定：``datasets/`` 被 gitignore，默认写到那里等于
    生成的文件永远不会出现在提交里。
    """
    args = cli.build_parser().parse_args([])
    assert args.corpus == DATASETS_PROCESSED
    assert args.config == REPO_ROOT / "experiment_config.yaml"
    assert args.out == REPO_ROOT / "baseline_results.csv"
    assert args.force is False


# ---- 入口 ----


def test_the_trial_count_comes_from_the_config(corpus, tmp_path, monkeypatch):
    """行数随配置里的 trials 变：10 不是硬编码的。"""
    for trials in (2, 5):
        config = _write_config(tmp_path / f"c{trials}.yaml", trials=trials)
        out = tmp_path / f"t{trials}.csv"
        monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
        assert cli.main() == 0
        assert len(_rows(out)) == 3 * (trials + 1)


def test_main_writes_and_reports_where_it_wrote(corpus, config, tmp_path, monkeypatch, capsys):
    out = tmp_path / "baseline_results.csv"
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
    assert cli.main() == 0
    assert len(_rows(out)) == 9
    assert str(out) in capsys.readouterr().out


def test_main_does_not_label_zero_valued_rows_as_trivial(corpus, config, tmp_path, monkeypatch, capsys):
    """B 的汇总**不**打印"平凡解"计数。

    A 的入口有那一行（``value == 0`` 的行），在 A 那里它是对的。搬到这里就错了：
    Random 抽到全反面时返回空集、``value`` 为 0，那是**合法的一次抽样**，不是计划
    §5 说的"没有物品装得下的平凡零最优实例"。一个按 value 数的"平凡解"计数会把
    两者混为一谈。
    """
    out = tmp_path / "baseline_results.csv"
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
    assert cli.main() == 0
    printed = capsys.readouterr().out
    assert "平凡" not in printed
    assert str(out) in printed


def test_main_refuses_to_overwrite_without_force(corpus, config, tmp_path, monkeypatch, capsys):
    """目标已存在且没给 ``--force`` ⇒ 报错退出，且**文件逐字节未变**（设计 §8）。

    A 上真的发生过一次：裸跑 CLI 把已入库的 ``reference_results.csv`` 覆写成不同
    字节。只断言"退出码 1"不够——必须同时断言文件内容没变，因为一个"报了错但已经
    写了一半"的实现在退出码上看起来是对的。
    """
    out = tmp_path / "baseline_results.csv"
    out.write_bytes(b"sentinel\n")
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out))
    assert cli.main() == 1
    assert out.read_bytes() == b"sentinel\n"
    assert "force" in capsys.readouterr().err


def test_force_overwrites_and_writes_every_row(corpus, config, tmp_path, monkeypatch):
    out = tmp_path / "baseline_results.csv"
    out.write_bytes(b"sentinel\n")
    monkeypatch.setattr("sys.argv", _argv(corpus, config, out, "--force"))
    assert cli.main() == 0
    assert len(_rows(out)) == 9


# ---- 计时口径 ----


def _timing_instance() -> dict:
    rng = random.Random(20260930)
    weights = [rng.randint(1, 100) for _ in range(TIMING_N)]
    values = [rng.randint(1, 100) for _ in range(TIMING_N)]
    order = list(range(TIMING_N))
    rng.shuffle(order)
    return {
        "capacity": sum(weights) // 3,
        "instance_id": "timing-0001",
        "item_order": order,
        "values": values,
        "weights": weights,
    }


@pytest.fixture
def timing_corpus(tmp_path):
    """只有一个大实例的语料，专供计时测试。"""
    root = tmp_path / "processed"
    path = root / "test" / "instances.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(_timing_instance(), sort_keys=True) + "\n").encode("utf-8"))
    for parts in (("stress", E3_NAME), ("stress", E4_NAME)):
        empty = root.joinpath(*parts, "instances.jsonl")
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_bytes(b"")
    return root


def test_latency_is_one_construction_not_the_sum_of_ten(timing_corpus, tmp_path):
    """单行 ``latency_s`` 是**一次**构造的耗时，不是十次之和（设计 §5）。

    计划 §5 专门禁止了"把十次试验的时长当成一个解的时长上报"。十次之和会落在下面
    ``ten`` 的量级上，而单次约为它的 1/10；取 5 倍余量避开计时噪声。

    断言写成"``latency_s > 0``"是空转的——那样连"记的是十次之和"都挡不住。
    """
    out = tmp_path / "baseline_results.csv"
    cli.run(timing_corpus, out, evaluation_seed=6001, trials=10)

    instance = _timing_instance()
    start = time.perf_counter()
    for trial in range(10):
        random_solution(instance, trial_seed(instance["instance_id"], trial, 6001))
    ten = time.perf_counter() - start

    latencies = [float(r["latency_s"]) for r in _rows(out) if r["method"] == "random"]
    assert len(latencies) == 10
    assert max(latencies) < ten / 2
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_cli.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: FAIL — `ModuleNotFoundError: No module named 'experiment.baselines.cli'`

- [ ] **Step 3: 写实现**

创建 `experiment/baselines/cli.py`：

```python
"""``baseline_results.csv`` 的生成入口。

用法::

    python -m experiment.baselines.cli
    python -m experiment.baselines.cli --out %TEMP%/baseline_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml

from experiment.baselines.methods import BaselineSolution, greedy_solution, random_solution
from experiment.baselines.seeds import trial_seed
from experiment.data.spec import DATASETS_PROCESSED, E3_NAME, E4_NAME, REPO_ROOT
from experiment.eval.metrics import feasibility_violations
from experiment.eval.reference import item_set_hash

#: 产物放在**仓库根目录并入库**，与 reference_results.csv / data_manifest.json /
#: experiment_config.yaml 一致：datasets/ 整体被 gitignore，把审计凭据放在那里
#: 等于换台机器就无从证明数据未被调换。
DEFAULT_OUT: Path = REPO_ROOT / "baseline_results.csv"

#: 参数从 YAML 读，不硬编码。这正是 `assemble/manifest.py` 的 `build_config`
#: docstring 写明的意图——「让后续阶段读 YAML 而不是回头读 Markdown」：计划若再
#: 修订，只需重新生成 YAML。这两项目前在 Python 侧**没有**可导入的常量，只存在于
#: 转录进 YAML 的 `_EVALUATION` 里，所以读 YAML 也是唯一不必再抄一遍常量的选择。
DEFAULT_CONFIG: Path = REPO_ROOT / "experiment_config.yaml"

#: 列序即产物的一部分。
COLUMNS: tuple[str, ...] = (
    "instance_id",
    "split",
    "method",
    "trial",
    "seed",
    "value",
    "weight_used",
    "violations",
    "item_set_hash",
    "latency_s",
)

#: 语料的子集：只取 test 与两个 stress 集（设计 §1）。
#:
#: 刻意**不含** train / validation：计划 §8 的 E1–E4b 全部落在 test 与 stress 上
#: （E2 是 E1 的分解、不产生新数据，E4b 是 E1 的子集），train/validation 只服务于
#: C 的训练与检查点选择，基线不在其上报告。
SUBSETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("test", ("test",)),
    (E3_NAME, ("stress", E3_NAME)),
    (E4_NAME, ("stress", E4_NAME)),
)


def corpus_files(root: Path) -> list[tuple[str, Path]]:
    """三个子集的 ``(标签, 文件)``，顺序固定。

    与 ``experiment/eval/cli.py`` 的同名函数有两处**有意**不同：这里多一层标签
    （本产物有 ``split`` 列，A 的没有），且范围只取三个子集（见 ``SUBSETS``）。
    """
    root = Path(root)
    return [(label, root.joinpath(*parts, "instances.jsonl")) for label, parts in SUBSETS]


def read_instances(files: Sequence[tuple[str, Path]]) -> list[tuple[str, dict]]:
    """展开成 ``(标签, 实例)``，跳过空行。"""
    out: list[tuple[str, dict]] = []
    for label, path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append((label, json.loads(line)))
    return out


def read_config(path: Path) -> tuple[int, int]:
    """从 ``experiment_config.yaml`` 读 ``(evaluation_seed, trials)``。

    Raises:
        KeyError: 缺字段。让它在入口处当场炸掉，好过用一个默认值静默地跑完
            整份语料再产出一份参数不对的产物。
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    return (
        int(evaluation["evaluation_seed"]),
        int(evaluation["random_trials_per_instance"]),
    )


def resolve_out(path: Path, force: bool) -> Path:
    """落盘前的守卫：目标已存在且没给 ``--force`` 就拒绝。

    这不是礼貌。``--out`` 默认指向已入库的 ``baseline_results.csv``，而 ``latency_s``
    是墙钟量——裸跑会把一个已入库文件改写成不同字节、弄脏工作区，而这件事在 A 上
    **真的发生过**。拒绝比事后 ``git checkout`` 便宜。

    Raises:
        FileExistsError: 目标存在且 ``force`` 为假。
    """
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(path)
    return path


def _solution_row(
    label: str,
    method: str,
    instance: dict,
    solution: BaselineSolution,
    trial: int | None,
    seed: int | None,
    latency_s: float,
) -> dict:
    """一行产物。

    ``violations`` 由 ``feasibility_violations`` **算出来**而不是写死 "0"：一个写死的
    常量断言不了任何东西。它是**独立于构造过程**的复核——能抓到"扣容量写错但价值
    算对"这类自洽缺陷（设计 §7）。
    """
    violations = feasibility_violations(instance, solution.selected_indices)
    if violations:
        raise AssertionError(
            f"{instance['instance_id']} 的 {method} 解不可行：{violations} 条违例"
        )
    return {
        "instance_id": instance["instance_id"],
        "split": label,
        "method": method,
        "trial": "" if trial is None else str(trial),
        "seed": "" if seed is None else str(seed),
        "value": str(solution.value),
        "weight_used": str(solution.weight_used),
        "violations": str(violations),
        "item_set_hash": item_set_hash(solution.selected_indices),
        "latency_s": f"{latency_s:.6f}",
    }


def run(
    corpus_root: Path, out_path: Path, *, evaluation_seed: int, trials: int
) -> dict:
    """构造全部基线的解并落盘，返回一份汇总。

    **计时只包住单次构造调用**（设计 §5）：从进入方法到拿到 ``BaselineSolution``
    为止，不含读取实例、不含行构造与写盘。Random 的 10 行各记自己那一次的耗时——
    把十次之和写进每一行是这类产物最常见的错法，它会让 ``latency_s`` 不再是
    "一次构造多贵"，而 D 的 speed 指标正是拿它比三个方法。
    """
    rows: list[dict] = []
    by_method = {"greedy": 0, "random": 0}

    for label, instance in read_instances(corpus_files(corpus_root)):
        start = time.perf_counter()
        greedy = greedy_solution(instance)
        greedy_latency = time.perf_counter() - start
        rows.append(_solution_row(label, "greedy", instance, greedy, None, None, greedy_latency))
        by_method["greedy"] += 1

        for trial in range(trials):
            seed = trial_seed(instance["instance_id"], trial, evaluation_seed)
            start = time.perf_counter()
            solution = random_solution(instance, seed)
            latency = time.perf_counter() - start
            rows.append(_solution_row(label, "random", instance, solution, trial, seed, latency))
            by_method["random"] += 1

    # 行序是产物的一部分。``trial`` 按**数值**排：按字符串排会把 "10" 排到 "2"
    # 前面（"0" < "1" < "10" < "2"）——10 次试验时试验号恰好全是单字符，看不出
    # 来，所以这个坑由一个 12 次试验的测试钉住。贪心没有试验号，用 -1 让它排在该
    # 实例的随机行之前。
    rows.sort(key=lambda row: (row["instance_id"], row["method"], int(row["trial"] or -1)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" 关掉 Python 的换行翻译，lineterminator="\n" 固定为 LF。
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return {"by_method": by_method, "out": out_path, "rows": len(rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为冻结语料构造 Random 与 Greedy 两个基线"
    )
    parser.add_argument(
        "--corpus", type=Path, default=DATASETS_PROCESSED, help="冻结语料根目录"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="experiment_config.yaml 的位置"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="baseline_results.csv 的位置"
    )
    parser.add_argument("--force", action="store_true", help="目标已存在时也覆写")
    return parser


def main() -> int:
    # 与 generate/collect/assemble/eval 的入口一致：重定向到文件时 Windows 用 ANSI
    # 代码页编码 stdout，中文标签会抛 UnicodeEncodeError。
    #
    # stderr 一并重设：本函数是**本项目第一个往 stderr 写字**的入口（覆写守卫那两行），
    # 而重设之前它们在 GBK 控制台上是乱码——一条告诉用户「加 --force」的提示，读不出来
    # 就等于没有。探针实测过修复前后的两种输出。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    try:
        out_path = resolve_out(args.out, args.force)
    except FileExistsError:
        print(f"产物已存在，未覆写: {args.out}", file=sys.stderr)
        print("确要覆写请加 --force。", file=sys.stderr)
        return 1

    evaluation_seed, trials = read_config(args.config)
    start = time.perf_counter()
    summary = run(args.corpus, out_path, evaluation_seed=evaluation_seed, trials=trials)

    # 刻意**不**打印 A 的入口里那行"平凡解: N"：那里数的是 value == 0 的行，在 A
    # 是对的；在这里 Random 抽到全反面时会返回空集、value 为 0，那是合法的一次抽样，
    # 不是计划 §5 说的"没有物品装得下的平凡零最优实例"。
    print(f"语料根目录: {args.corpus}")
    print(f"产物:       {summary['out']}")
    print(f"实例行数:   {summary['rows']}")
    print(f"  greedy:   {summary['by_method']['greedy']}")
    print(f"  random:   {summary['by_method']['random']}")
    print(f"试验次数:   {trials}（来自 {args.config}）")
    print(f"墙钟总耗时: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试，确认通过**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_cli.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: `14 passed`

- [ ] **Step 5: 提交**

```bash
git add experiment/baselines/cli.py tests/test_baselines_cli.py
git commit -m "feat(baselines): baseline_results.csv 的生成入口与覆写守卫"
```

---

## Task 6: 全语料验收

**Files:**
- Test: `tests/test_baselines_corpus.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_baselines_corpus.py`：

```python
"""全语料验收（设计 §10）。

前置事实全部实测于 2026-09-30（探针 probe3.py）：3,300 实例 / 36,300 行 /
violations 合计 0 / 无行的 value 超过 V* / Greedy 平均 V/U 0.975118、
Random 平均 V/U 0.559955。
"""

import csv
from fractions import Fraction

import pytest

from experiment.baselines import cli
from experiment.data.spec import DATASETS_PROCESSED, REPO_ROOT

#: `datasets/` 被 gitignore，临时副本里没有它。本项目的完成标准是**真仓库里
#: 0 skipped**，而这个仓库里语料是在的，所以这条测试在本机真跑；在没有语料的副本
#: 里它跳过而不是失败。
CORPUS_TEST_FILE = DATASETS_PROCESSED / "test" / "instances.jsonl"
REFERENCE_CSV = REPO_ROOT / "reference_results.csv"

pytestmark = pytest.mark.skipif(
    not CORPUS_TEST_FILE.is_file() or not REFERENCE_CSV.is_file(),
    reason="冻结语料或参考解不在位（datasets/ 被 gitignore）；在本仓库里会真跑",
)

SEED = 6001
TRIALS = 10
INSTANCES = 3_300
ROWS = INSTANCES * (TRIALS + 1)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """在**真实语料**上跑一遍入口，产物写到临时目录，再把 CSV 读回来。

    写出去再读回来，顺带把序列化路径也验了。
    """
    out = tmp_path_factory.mktemp("baselines") / "baseline_results.csv"
    summary = cli.run(DATASETS_PROCESSED, out, evaluation_seed=SEED, trials=TRIALS)
    with out.open(encoding="utf-8", newline="") as handle:
        return summary, list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def reference():
    """A 的参考解：``instance_id -> (V*, U)``。"""
    with REFERENCE_CSV.open(encoding="utf-8", newline="") as handle:
        return {
            row["instance_id"]: (int(row["value"]), Fraction(row["upper_bound"]))
            for row in csv.DictReader(handle)
        }


def test_the_corpus_is_the_size_this_plan_was_written_against():
    """语料规模漂移会静默改变下面每一条断言的分母，所以先钉住它。"""
    instances = cli.read_instances(cli.corpus_files(DATASETS_PROCESSED))
    assert len(instances) == INSTANCES
    assert {label for label, _ in instances} == {"test", "e3_size", "e4_tail"}


def test_the_row_count_is_instances_times_eleven(results):
    summary, rows = results
    assert summary["rows"] == ROWS
    assert len(rows) == ROWS
    assert summary["by_method"] == {"greedy": INSTANCES, "random": INSTANCES * TRIALS}


def test_every_instance_is_present_once_per_method_and_trial(results, reference):
    """每个实例恰好 10 条 random + 1 条 greedy，且它的参考解存在。

    **与 `reference` 的关系是包含，不是相等。** `reference_results.csv` 覆盖**全部
    17,700 个实例**（含 `train` / `validation`），而 B 只跑 3,300 个——写成
    ``set(by_instance) == set(reference)`` 会当场失败，右边多出 14,400 个。正向探针
    实测到了这一次失败。所以是：B 的集合**是**参考解集合的子集，且大小正好是 3,300。
    """
    _, rows = results
    by_instance: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], {}).setdefault(row["method"], []).append(
            row["trial"]
        )
    assert set(by_instance) <= set(reference), "有实例找不到参考解"
    assert len(by_instance) == INSTANCES, "B 的实例集大小不对"
    for instance_id, methods in by_instance.items():
        assert sorted(methods["random"], key=int) == [str(t) for t in range(TRIALS)], instance_id
        assert methods["greedy"] == [""], instance_id


def test_no_instance_id_contains_the_key_separator(results):
    """设计 §4 的种子键用 ``|`` 拼接，这就要求各分量不含 ``|``。"""
    _, rows = results
    assert [r["instance_id"] for r in rows if "|" in r["instance_id"]] == []


def test_every_row_is_feasible_and_no_value_exceeds_the_optimum(results, reference):
    """两个**独立计算**的产物之间的一致性。

    这一条**不是**"贪心写对了"的证明：任何**可行**选择的 value 都不可能超过 ``V*``
    ——那是 ``V*`` 的定义。它挡的是**跨子系统的不一致**：实例集错位、参考解列读错、
    ``value`` 从错误的数组求和。

    排序方向写反、``break`` 代替 ``continue`` 这类缺陷**能通过这一条**（它们给出的
    解仍然可行，只是更差），由 `tests/test_baselines_greedy.py` 的合成实例钉住；
    下一条测试补的正是这个缺口。
    """
    _, rows = results
    for row in rows:
        assert int(row["value"]) <= reference[row["instance_id"]][0], row
        assert row["violations"] == "0", row


def test_greedy_average_quality_is_strictly_above_random(results, reference):
    """Greedy 的平均 ``V/U`` **严格高于** Random 的平均 ``V/U``。

    这是上一条补不上的那个缺口：排序反了或提前 ``break`` 的贪心仍然**可行**，
    只是**更差**——在 3,300 个实例的平均上会掉向 Random 的 0.5 掷币。

    只断言严格不等式、**不写死阈值**：阈值会在语料修订后变成一条假失败，而不等式
    不会。实测证据记在这里供人核对：Greedy **0.975118**、Random **0.559955**，
    差额 0.415163。
    """
    _, rows = results
    scored: dict[str, list[Fraction]] = {"greedy": [], "random": []}
    for row in rows:
        upper = reference[row["instance_id"]][1]
        scored[row["method"]].append(Fraction(int(row["value"])) / upper)

    greedy = sum(scored["greedy"]) / len(scored["greedy"])
    random_ = sum(scored["random"]) / len(scored["random"])
    assert greedy > random_
```

- [ ] **Step 2: 跑测试，确认通过**

Run: `./venv/Scripts/python.exe -m pytest tests/test_baselines_corpus.py -q --basetemp="$TEMP/ai2-pytest"`

Expected: `6 passed`（**不是** skipped —— 本仓库里 `datasets/` 在位）

- [ ] **Step 3: 跑全套件**

Run: `./venv/Scripts/python.exe -m pytest -q --basetemp="$TEMP/ai2-pytest"`

Expected: `398 passed`（A 的 350 + 种子 4 + Greedy 10 + Random 14 + CLI 14 + 语料 6），**0 skipped**

> 若这个数对不上，先看 `-q` 的最后一行而不是猜：多出来的多半是本计划之外的东西。
> 这六个分项都在正向探针里实测过（`4 / 10 / 14 / 14 / 6`），不是加出来的。

- [ ] **Step 4: 提交**

```bash
git add tests/test_baselines_corpus.py
git commit -m "test(baselines): 全语料验收——规模、可行性、与参考解的一致性、贪心质量"
```

---

## Task 7: 在真实语料上生成并入库存物

**Files:**
- Create: `baseline_results.csv`（仓库根目录）

- [ ] **Step 1: 跑入口**

Run: `./venv/Scripts/python.exe -m experiment.baselines.cli`

Expected（行数必须逐字对上）：

```
语料根目录: .../datasets/processed
产物:       .../baseline_results.csv
实例行数:   36300
  greedy:   3300
  random:   33000
试验次数:   10（来自 .../experiment_config.yaml）
墙钟总耗时: ...
```

- [ ] **Step 2: 核对产物**

Run: `wc -l baseline_results.csv`

Expected: `36301`（36,300 行 + 表头）

Run: `git status --short baseline_results.csv`

Expected: `?? baseline_results.csv`（未跟踪——目标原先不存在，所以**没有**触发覆写守卫）

- [ ] **Step 3: 提交**

```bash
git add baseline_results.csv
git commit -m "feat(baselines): 在冻结语料上生成 baseline_results.csv（36,300 行）"
```

- [ ] **Step 4: 记下产物的复现性质**

**`baseline_results.csv` 逐字节不可复现**，与 `reference_results.csv` 同理：`latency_s`
是墙钟测量（设计 §5）。要复核就比对**除 `latency_s` 以外的列**。

这一点已经写进设计 §12，这里只需要确认执行时没有踩到它：

- ❌ 不要"再跑一次然后 `git diff` 看是否一致"——`latency_s` 必然不同。
- ✅ 要复核就重跑一次、把 `--out` 指到仓库外的临时路径（**且该路径下已有文件时必须
  加 `--force`**，否则入口会按设计拒绝），再比对除 `latency_s` 外的列。

> **B 的覆写守卫是 A 那次事故的直接产物。** A 上发生过一次"裸跑 CLI 覆写已入库的
> `reference_results.csv`、弄脏工作区"。B 的入口在目标已存在时报错退出、不改动文件，
> 所以本任务的 Step 1 之后，任何不带 `--force` 的裸跑都不会再改动这份产物。

---

## 自检

**1. 规格覆盖。** 逐节对设计文档：

| 设计节 | 落在哪个任务 |
| --- | --- |
| §1 范围（3,300 实例 / 36,300 行 / 独立产物） | Task 5（`SUBSETS`）、Task 6（规模断言）、Task 7（产物） |
| §2.1 Greedy（`ratio_order`、跳过继续） | Task 3（两个方法性测试 + 两个对照变体） |
| §2.2 Random（`item_order`、读法 (a)、平均非最好、`Random(seed)` 三条用法） | Task 4（`test_the_walk_follows_the_stored_item_order`、`test_a_coin_is_only_flipped_for_items_that_fit`、`test_ten_trials_reproduce_the_measured_golden`） |
| §3 模块结构（三个模块、纯函数、`BaselineSolution` 字段与顺序） | Task 2–5（各文件的实现 fence） |
| §4 种子派生（BLAKE2b-8、键序、三分量缺一不可、不含 `|`） | Task 2（4 个测试）、Task 6（`test_no_instance_id_contains_the_key_separator`） |
| §5 计时口径（单次、6 位小数、不含读盘） | Task 5（`run` 里的计时位置 + `test_latency_is_one_construction_not_the_sum_of_ten`） |
| §6 `ratio_order` 提升为公开 | Task 1 |
| §7 复用 A 的三个原语 | Task 1（`ratio_order`，公开）、Task 5（`item_set_hash`、`feasibility_violations`） |
| §8 产物列、行序、参数只读一次、覆写守卫 | Task 5（`COLUMNS`、`run` 的排序、`read_config`、`resolve_out`） |
| §9 与 计划 §10 的差额（不含 `normalized_quality`） | 不产出该列，`COLUMNS` 里就没有它 |
| §10 全语料验收的 6 条断言 | Task 6（6 个测试，逐条对应） |
| §11 各测试文件的职责表 | Task 2–6 的测试文件；计时那一条在 Task 5 |
| §12 刻意不做 | 没有对应任务，即正确 |

**一处要说明的不对称：** 设计 §7 的两个原语 `item_set_hash` 与
`feasibility_violations` 都只在 Task 5（`cli.py`）里用，`methods.py` 不 import
它们——因为它们是**产物列**的性质，不是解的构造的性质。这与 §3 的模块划分一致
（`methods.py` 是纯构造、不读文件也不看时钟）。

**2. 占位符扫描。** 计划里没有 TBD / TODO / "类似 Task N" / 无代码的代码步骤。每个
断言都带**实测的字面量或对照变体**，没有"断言它大于 0"这类空转。

**3. 类型与名字一致性。** 跨任务核对：

- `BaselineSolution.selected_indices` 是 `tuple[int, ...]`：Task 3 定义，Task 3/4/5
  三处使用都按元组（`tuple(selected)`、`== (1,)`、`item_set_hash` 接受 `Sequence`）。
- `trial_seed(instance_id, trial, base)` 三参、实参顺序固定：Task 2 定义，Task 4
  （`_seeded`）、Task 5（`run` 与测试）调用的都是这个顺序。
- `cli.run(corpus_root, out_path, *, evaluation_seed, trials)`：Task 5 定义，Task 5
  的测试与 Task 6 的 fixture 都用这两个关键字实参。
- `cli.read_config(path) -> tuple[int, int]` 返回 `(seed, trials)`：Task 5 的
  `main()` 里按 `evaluation_seed, trials = read_config(...)` 解包，测试
  `assert cli.read_config(config) == (123, 7)` 也是这个顺序。
- `cli.COLUMNS` 是 `tuple[str, ...]`，测试里 `list(cli.COLUMNS)` 转成 list 再与
  `csv.reader` 的表头比较。
- `SUBSETS` 是 `tuple[tuple[str, tuple[str, ...]], ...]`：`corpus_files` 用
  `root.joinpath(*parts, "instances.jsonl")` 展开。
- 三个模块名 `seeds` / `methods` / `cli` 与 `experiment/eval/` 的命名习惯一致。

**修正一处（写完计划后回看发现的）：** 设计 §6 的影响面表原本把
`tests/test_eval_upper_bound.py:71-80` 记成"5 处引用（含测试名）"，实际需要改的只有
**4 条断言**——第 71 行被检索命中只是因为测试名含有子串。设计文档已同步改正
（`edff57b`）。

---

## 执行顺序与依赖

Task 1 → 2 → 3 → 4 → 5 → 6 → 7 串行。

- Task 1 是其余全部的前置：`methods.py` 从 `experiment.eval.reference` import
  `ratio_order`，改名前它不存在。
- Task 3 是 Task 4 的前置：`random_solution` 追加到同一个文件，且复用 `_totals`。
- Task 5 是 Task 6 的前置：语料测试调 `cli.run`。
- Task 7 是最后一步：它写的是入库产物，其正确性由 Task 6 的验收守护。
