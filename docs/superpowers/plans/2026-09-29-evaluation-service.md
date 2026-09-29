# 评测服务（参考解与指标原语）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付精确参考求解器（整数 DP）与一组单实例指标原语，并产出仓库根目录的 `reference_results.csv`。

**Architecture:** 新建 `experiment/eval/` 包，与 `data/` 平级但不属于它——`EXPERIMENT_PLAN.md` §7 明说参考解「an evaluation service, not a fourth competing method」。数据流单向：`reference.py`（DP、分数背包上界、重建与自校验）→ `metrics.py`（单实例指标原语）→ `cli.py`（读冻结语料、写出 `reference_results.csv`）。不引入任何新依赖。

**Tech Stack:** Python 3.12、numpy（已在 `requirements.txt`）、`fractions.Fraction`（标准库）、pytest。

**依据:** `docs/superpowers/specs/2026-09-29-evaluation-service-design.md`（已提交，`6074da0` + 精度修正 `282d885`）。

---

## 0. 执行者必读

### 0.1 环境

- 工作目录：`C:\work\学校\A.SPBU\Saint-peter-26\AI-2\project`
- **必须用 venv 的解释器**：`./venv/Scripts/python.exe`。裸 `python` 上没有 pytest，也没有 numpy。
- **`--basetemp` 是强制的，且必须指向仓库之外**：
  `./venv/Scripts/python.exe -m pytest tests/... -p no:cacheprovider --basetemp="$TEMP/ap_ev_<task>" -q`
  仓库内的临时目录会被 git 看见；默认的 `%TEMP%\pytest-of-<user>` 里有别的会话留下的、**不得触碰**的内容。
- **绝对不要删除、修复或接管 `C:\Users\swj17\AppData\Local\Temp\pytest-of-swj17`。**
- 控制台是 GBK 代码页，中文常显示成乱码。**判断内容正确性时比对哈希或字节，不要比对屏幕上的字符。**
- Bash 工具每次调用后工作目录会重置；命令里写绝对路径或用 `cd ... && ...`。
- 行尾必须是 LF、不带 BOM。`.gitattributes` 已有 `*.csv text eol=lf` 与 `*.py/*.json/*.md` 的同等规则，无需改动。
- 跑整套测试要给足超时：**Bash 工具的默认超时是 120 000 ms，而本仓库整套约 180 s**，必须显式传 `timeout: 600000`。

### 0.2 分支与基线

当前在 `main`。**动手前先由控制者建好 `feat/evaluation-service` 并切过去**，不要在 `main` 上提交。

基线（实测于 2026-09-29，合并 Plan 3 之后）：`307 passed`，0 failed、0 skipped，约 182 s。

### 0.3 不可修改的文件

- `experiment/data/spec.py` **一行都不动**，**包括追加**。理由：`generate/corpus.py` 的 `generator_fingerprint()` 会把 `spec.py` 的**字节**纳入哈希，而该指纹已写进入库的 `data_manifest.json`。追加一个常量就会让「重新生成是否一致」这个复核动作失真。本计划需要的新路径常量因此写在 `eval/cli.py` 里。
- `experiment/data/generate/*`、`experiment/data/collect/*`、`experiment/data/assemble/*` 全部只读。
- `tests/` 下**本计划新建之外的一切既有文件**（`tests/conftest.py` 也在内）。清单以 `git ls-files tests/` 为准，不要凭记忆列举。
- 本计划**只允许新建**：`experiment/eval/{__init__,reference,metrics,cli}.py`、`tests/test_eval_{upper_bound,reference,metrics,e0,golden,properties,cli}.py`、`tests/golden/reference_solutions.json`、`reference_results.csv`；以及 Task 8 对 `EXPERIMENT_PLAN.md` §7 的追加。

### 0.4 诊断纪律

- **永远不要为了验证一条测试是否真会失败去改仓库里的真文件。** 只能在仓库外写 scratch 脚本、用 `monkeypatch`，或复制一份再改。探针一律写在 `$TEMP` 下，不许往仓库里写任何临时文件。
- **`.pyc` 会让变异测试说谎。** CPython 的 `.pyc` 头把源文件 mtime 存成**整秒**并按 mtime + size 校验，所以同一秒内一次**等长**改写会被当成旧字节码复用，变异体静默存活。跑探针一律 `python -B` 或设 `PYTHONDONTWRITEBYTECODE=1`。
- **一次只改一个变量。** 本项目已经因为「同时改了两样东西」产出过假信号。
- **本计划里所有期望值都是探针实测出来的，不是推导出来的。** 若某个 Step 的期望与实际不符：**停下来报告，不要改期望值去迁就代码**。改期望值等于把栅栏拆掉。
- 每个 Task 结束时 `git status --short` 必须干净（除被 ignore 的 `__pycache__`）。

### 0.5 已实测的数值（写断言与验收时会用到）

来自仓库外的设计探针（`$TEMP/probe_evaldesign/probe.py` + `probe2.py`，实现与本计划 Task 1/2 的栅栏逐字相同）：

| 事实 | 实测值 |
| --- | --- |
| 全语料 `solve`（**含**重建表）总耗时 | **16.0 s**（numpy DP + `(n+1)×(C+1)` int32 表） |
| 全语料实例数 | **17,700**（12000 + 2400 + 2400 + 600 + 300） |
| 全语料 `U ≥ V*` 违例 | **0** |
| 全语料平凡零最优（`V*=0`） | **0** |
| 最坏单实例（n=500、C=21201） | 0.02 s |
| 100 个随机 n=15 实例（solve + 全枚举） | 0.26 s |
| 单次 `solve` 峰值内存（n=500、C=21201） | `501×21202×4 B ≈ 42 MB`，逐实例分配并释放 |

**不可达格不是哨兵本身，而是哨兵加若干价值。** 实测 `_dp_table([3],[5],10)` 的 `table[-1, 10] = -536870907`，而哨兵是 `-536870912`。它仍是极大的负数，因此永远赢不了 `maximum`、也永远不等于最优值——但**断言 `== 哨兵` 会失败**。Task 2 的断言写成「为负」而不是「等于哨兵」。

**同比值的两件物品，`U` 与次序无关。** 比值完全相等时价值与重量成正比，任何装法给出同一个 `U`。所以设计 §5 的次级排序键**无法通过 `U` 来测**，必须在 `_ratio_order` 上直接测（Task 1）。

### 0.6 可复用的既有纯函数（不要重写）

```python
from experiment.data.generate.corpus import build_instance   # (split, family, n, alpha, index, attempt=0) -> dict
from experiment.data.spec import DATASETS_PROCESSED, E3_NAME, E4_NAME, REPO_ROOT, SPLITS
```

`build_instance` 的 `n` 是**自由参数、无最小尺寸约束**：冻结语料的 `SIZES = (20,50,100,200)` 不含 `n≤15`，但直接以 `n=15`、split 名 `"e0"` 调用即可得到确定性种子。这样 E0 既不新写生成器，也不改动 `generate/*`。

冻结语料的落盘布局（实测）：

```
datasets/processed/{train,validation,test}/instances.jsonl
datasets/processed/stress/{e3_size,e4_tail}/instances.jsonl
```

---

## 1. 文件结构

| 文件 | 职责 |
| --- | --- |
| `experiment/eval/__init__.py` | 包说明。参考解是**评测服务**，不是第四个参赛方法。 |
| `experiment/eval/reference.py` | 整数 DP、分数背包上界 `U`、重建与自校验、`ReferenceSolution`。 |
| `experiment/eval/metrics.py` | 单实例指标原语：可行性违例、两种间隙、归一化质量、平凡判定。 |
| `experiment/eval/cli.py` | 读五个语料文件，写出仓库根目录的 `reference_results.csv`。 |
| `tests/test_eval_upper_bound.py` | `U` 的精确性、精确排序、存储口径。 |
| `tests/test_eval_reference.py` | DP、重建、自校验、平凡解、价值总和预算。 |
| `tests/test_eval_metrics.py` | 五个指标原语。 |
| `tests/test_eval_e0.py` | E0：全子集枚举对照。 |
| `tests/test_eval_golden.py` | 金样本回归。 |
| `tests/test_eval_properties.py` | 生产形状语料上的不变量。 |
| `tests/test_eval_cli.py` | CSV 的列序、行序、字节与「只有时间列不稳定」。 |
| `tests/golden/reference_solutions.json` | 5 个内联小实例的金样本。 |
| `reference_results.csv` | 入库产物，仓库根目录。 |

跨实例聚合、表格、图、1000 次配对 bootstrap **不在本计划内**，属子项目 D。

---

## Task 1: 分数背包上界 `U`

**Files:**
- Create: `experiment/eval/__init__.py`
- Create: `experiment/eval/reference.py`
- Test: `tests/test_eval_upper_bound.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_eval_upper_bound.py`：

```python
"""分数背包上界 U 与它的存储口径（设计 §5、§9）。"""

from fractions import Fraction

import pytest

from experiment.eval.reference import (
    _ratio_order,
    format_upper_bound,
    fractional_upper_bound,
)

#: (weights, values, capacity, 精确的 U)。每一行的期望值都由设计探针实测得到。
CASES = [
    ([6, 8], [12, 8], 10, Fraction(16)),        # 装下一件 + 4/8 件
    ([1, 2, 3], [1, 2, 3], 3, Fraction(3)),     # 三件比值全为 1
    ([5, 5], [6, 7], 5, Fraction(7)),           # 比值降序，取大的那件
    ([4, 6], [4, 6], 6, Fraction(6)),           # 比值并列
    ([3, 2], [1, 5], 4, Fraction(17, 3)),       # 分数项约不尽
    ([10, 20], [5, 6], 9, Fraction(9, 2)),      # 一件都整件装不下，U 仍为正
]


@pytest.mark.parametrize("weights,values,capacity,expected", CASES)
def test_upper_bound_is_exact(weights, values, capacity, expected):
    assert fractional_upper_bound(weights, values, capacity) == expected


def test_ordering_uses_exact_ratios_not_float_division():
    """构造一对**浮点相等、有理数不等**的比值，钉住「按真实比值排序」。

    A = (w=1, v=1)：比值精确为 1。
    B = (w=2**53, v=2**53+1)：比值精确为 1 + 2**-53，但浮点除法看不见它——
      ``2**53 + 1`` 本身就超出 double 的整数精度，先被舍入成 ``2**53``，
      商于是恰好是 1.0。

    容量取 ``2**53``，两件里只能整件装下一件。
    按真实比值应先取 B，U = 2**53 + 1，恰好等于整数最优值。
    按浮点比值排序时 A 与 B 并列，次级键（w 小者先）把 A 排到前面：先占掉
    1 单位容量，B 只剩 2**53 - 1 可装，U 比 V* **小 1**——上界性质当场破裂。
    """
    weights = [1, 2**53]
    values = [1, 2**53 + 1]
    capacity = 2**53
    optimum = 2**53 + 1

    # 前提：这一对确实是「浮点相等、有理数不等」。
    assert float(values[1]) / float(weights[1]) == 1.0
    assert float(values[0]) / float(weights[0]) == 1.0
    assert Fraction(values[1], weights[1]) > Fraction(values[0], weights[0])

    def float_ordered_greedy() -> Fraction:
        """把同一套贪心换成浮点排序，用来**先把靶子打出来**。"""
        order = sorted(range(2), key=lambda i: (-(values[i] / weights[i]), weights[i], i))
        remaining, total = capacity, Fraction(0)
        for i in order:
            if remaining <= 0:
                break
            if weights[i] <= remaining:
                total += values[i]
                remaining -= weights[i]
            else:
                total += Fraction(values[i] * remaining, weights[i])
                remaining = 0
        return total

    assert float_ordered_greedy() < optimum
    assert fractional_upper_bound(weights, values, capacity) == Fraction(optimum)


def test_ratio_order_breaks_ties_deterministically():
    """同比值时次级键使排序**完全确定**，U 因此不依赖 sort 的稳定性（设计 §5）。

    次级键无法通过 U 来测：比值完全相等时价值与重量成正比，任何装法都给出
    同一个 U。所以它必须在排序函数本身上测。
    """
    assert _ratio_order([1, 2], [2, 4]) == [0, 1]              # 同比值：w 小者先
    assert _ratio_order([2, 1], [4, 2]) == [1, 0]              # 打乱输入，仍取 w 小的那件
    assert _ratio_order([5, 5, 5], [7, 7, 7]) == [0, 1, 2]     # 全同：按原索引
    assert _ratio_order([1, 3, 2], [3, 1, 2]) == [0, 2, 1]     # 比值 3、1/3、1


def test_storage_rounds_up_to_six_decimals():
    """向上取整而非四舍五入：存储后的值必须仍是一个合法上界（设计 §9）。"""
    assert format_upper_bound(Fraction(10, 3)) == "3.333334"
    assert format_upper_bound(Fraction(17, 3)) == "5.666667"
    assert format_upper_bound(Fraction(66, 7)) == "9.428572"
    assert format_upper_bound(Fraction(16)) == "16.000000"
    assert format_upper_bound(Fraction(9, 2)) == "4.500000"


@pytest.mark.parametrize("weights,values,capacity,expected", CASES)
def test_stored_upper_bound_is_still_an_upper_bound(weights, values, capacity, expected):
    """既钉方向（存储值 ≥ 精确值），也钉紧度（最多高 1e-6）。

    只钉方向会让「无脑放大一个数量级」也通过。
    """
    stored = Fraction(format_upper_bound(expected))
    assert stored >= expected
    assert stored - expected < Fraction(1, 10**6)
```

- [ ] **Step 2: 跑测试确认它失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_upper_bound.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t1" -q 2>&1 | tail -5
```

Expected: 收集阶段失败，`ModuleNotFoundError: No module named 'experiment.eval'`（或 `ImportError: cannot import name '_ratio_order'`）。

- [ ] **Step 3: 写实现**

创建 `experiment/eval/__init__.py`：

```python
"""评测服务：精确参考解与单实例指标原语。

依据 `EXPERIMENT_PLAN.md` 第 7 节。本包**不是**第四个参赛方法——计划第 7 节的
原话是「This is an evaluation service, not a fourth competing method」。把它和
三个参赛方法放进同一个包，正是那一节想避免的混淆。
"""
```

创建 `experiment/eval/reference.py`：

```python
"""精确参考解：整数 DP、分数背包上界、重建与自校验。

依据 `EXPERIMENT_PLAN.md` 第 7 节与设计文档第 4、5、6 节。
"""

from __future__ import annotations

from fractions import Fraction
from functools import cmp_to_key
from typing import Sequence

#: 把 U 写成十进制字符串时的放大倍数（6 位小数）。
_UB_SCALE = 1_000_000


def _ratio_order(weights: Sequence[int], values: Sequence[int]) -> list[int]:
    """按 ``v_i / w_i`` 降序返回物品索引，**用精确整数叉乘**。

    走浮点除法是错的：两个不同的有理比值可能舍入到同一个 double，比较结果就
    不再反映真实次序，贪心于是可能给出**小于**真分数最优的 U，进而 ``U < V*``
    ——上界性质被破坏，而 `EXPERIMENT_PLAN.md` 第 6 节的 ``V/U`` 随之失真。

    次级键依次取更小的 ``w``、再取原索引升序，使完整排序确定，U 不依赖
    ``sort`` 的稳定性。
    """

    def compare(i: int, j: int) -> int:
        left = values[i] * weights[j]
        right = values[j] * weights[i]
        if left != right:
            return -1 if left > right else 1
        if weights[i] != weights[j]:
            return -1 if weights[i] < weights[j] else 1
        return (i > j) - (i < j)

    return sorted(range(len(weights)), key=cmp_to_key(compare))


def fractional_upper_bound(
    weights: Sequence[int], values: Sequence[int], capacity: int
) -> Fraction:
    """分数背包上界 U，**精确**计算。

    按真实比值降序贪心：装得下就整件装入，第一件装不下的按剩余容量取分数。
    ``v`` 与 ``w`` 都是整数，所以 U 是一个有理数，可以用 ``Fraction`` 精确表示；
    这使 ``U >= V*`` 是**定理**而不是容差问题，``V/U <= 1`` 由构造保证。
    """
    order = _ratio_order(weights, values)
    remaining = capacity
    total = Fraction(0)
    for i in order:
        if remaining <= 0:
            break
        wi, vi = weights[i], values[i]
        if wi <= remaining:
            total += vi
            remaining -= wi
        else:
            total += Fraction(vi * remaining, wi)
            remaining = 0
    return total


def format_upper_bound(upper_bound: Fraction) -> str:
    """把 U 写成**向上取整**到 6 位小数的十进制字符串。

    向上取整而非四舍五入或下取整：存储后的值必须仍是一个合法上界。

    用整数运算而不是 ``decimal``：decimal 的上下文精度会变成一个隐式的输入，
    而这里没有理由引入它。
    """
    scaled = upper_bound * _UB_SCALE
    whole = -(-scaled.numerator // scaled.denominator)  # 向上取整的整数除法
    return f"{whole // _UB_SCALE}.{whole % _UB_SCALE:06d}"
```

- [ ] **Step 4: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_upper_bound.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t1" -q 2>&1 | tail -3
```

Expected: `15 passed`（`test_upper_bound_is_exact` 与 `test_stored_upper_bound_is_still_an_upper_bound` 各展开 6 条，其余 3 条）。

- [ ] **Step 5: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/eval/__init__.py experiment/eval/reference.py tests/test_eval_upper_bound.py && git commit -m "feat(eval): 精确分数背包上界与整数叉乘排序"
```

---

## Task 2: 整数 DP、重建与自校验

**Files:**
- Modify: `experiment/eval/reference.py`
- Test: `tests/test_eval_reference.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_eval_reference.py`：

```python
"""参考解：整数 DP、重建与自校验（设计 §4、§6）。"""

from fractions import Fraction

import pytest

from experiment.eval.reference import (
    ReferenceSolution,
    _dp_table,
    _verify_reconstruction,
    item_set_hash,
    solve,
)

THREE_ITEM = {"capacity": 9, "values": [10, 7, 9], "weights": [8, 4, 5]}


def _instance(weights, values, capacity):
    return {"capacity": capacity, "values": list(values), "weights": list(weights)}


def test_three_item_example():
    """1 号单件价值最大（10），但它挤掉 2 号 + 3 号（价值 16），最优取后者。"""
    result = solve(THREE_ITEM)
    assert isinstance(result, ReferenceSolution)
    assert result.value == 16
    assert result.selected_mask == [False, True, True]
    assert result.selected_indices == [1, 2]
    assert result.certified is True
    assert result.upper_bound == Fraction(16)
    assert result.runtime_s >= 0.0


def test_answer_is_the_best_cell_not_the_full_capacity_cell():
    """最优解**用不满**容量时，``table[-1][capacity]`` 是哨兵派生的负数。

    DP 的语义是「恰好用掉 c」，所以用不满时那一格不可达。本设计的第一个探针
    脚本正是取了那一格，返回 -2.3e18；而它自带的「V* ≤ sum(values)」校验对
    负数恒真，**没能发现**。

    注意断言写「为负」而不是「等于哨兵」：哨兵会被 ``+ v`` 抬起来（实测
    ``_dp_table([3],[5],10)`` 的 ``table[-1, 10]`` 是 -536870907，而哨兵本身
    是 -536870912）。它仍是极大的负数，永远赢不了 maximum。
    """
    table = _dp_table([3], [5], 10)
    assert int(table[-1, 10]) < 0
    assert int(table[-1].max()) == 5

    result = solve(_instance([3], [5], capacity=10))
    assert result.value == 5
    assert result.selected_mask == [True]


def test_no_item_fits_gives_the_empty_solution():
    """平凡解：一件都装不下时返回空集、V* = 0。

    冻结语料上这个分支**永不触发**（实测 0/17700，设计 §8），所以只有合成
    实例能覆盖它。注意此时 U 仍为正：分数松弛可以切开物品。
    """
    result = solve(_instance([10, 20], [5, 6], capacity=9))
    assert result.value == 0
    assert result.selected_indices == []
    assert result.selected_mask == [False, False]
    assert result.upper_bound == Fraction(9, 2)


def test_all_empty_selections_share_one_hash():
    """空集就是空集：两个平凡实例的哈希相同是**正确**的，不是碰撞。

    ``item_set_hash`` 只覆盖选择结果，不覆盖实例身份——那是 ``canonical_hash``
    的事（设计 §4）。
    """
    first = solve(_instance([10], [5], capacity=9))
    second = solve(_instance([100, 200], [7, 8], capacity=50))
    assert first.item_set_hash == second.item_set_hash == item_set_hash([])


def test_self_check_rejects_every_bad_reconstruction():
    """强制自校验的三条拒绝路径（设计 §6）。

    直接测这个纯函数，是因为「自校验真的会拒绝」在正常路径上永远观察不到——
    只写「返回的集合合法」等于没测这段代码。
    """
    _verify_reconstruction([1, 1], [1, 1], 2, 2, [0, 1])  # 合法：不抛

    with pytest.raises(ReferenceError, match="重复"):
        _verify_reconstruction([1, 1], [1, 1], 2, 2, [0, 0])
    with pytest.raises(ReferenceError, match="越界"):
        _verify_reconstruction([1, 1], [1, 1], 2, 2, [0, 5])
    with pytest.raises(ReferenceError, match="不可行"):
        _verify_reconstruction([1, 1], [1, 1], 1, 2, [0, 1])
    with pytest.raises(ReferenceError, match="最优值"):
        _verify_reconstruction([1, 1], [1, 1], 2, 2, [0])


def test_value_budget_is_set_by_the_sentinel_not_by_int32():
    """阈值的解是 ``|哨兵| = 2**29``，**不是** int32 的一半。

    DP 里不可达的格子拿到的是 ``哨兵 + 沿路径的价值``，它必须保持负数，否则
    会伪装成一个合法价值。一条路径至多把每件物品加一次，所以能累加的上限就是
    所有正价值之和：

    - 总和恰为 ``2**29`` 时，``哨兵 + v`` **恰好是 0**——与「空集价值 0」
      无法区分；
    - 再大一点就是**正数**，``dp.max()`` 于是可能返回一个来自不可行路径的
      幻影最优值。

    所以阈值必须**恰好**是 ``2**29``：多一分则危险，少一分则无谓地拒绝。
    若有人把它放宽到 ``2**30``（「int32 的一半」这个看起来更自然的值），
    本测试的第一条断言就会失败。

    设计 §4 只写了「max sum(values) = 17,232,254，远在 int32 内」——那句话
    论证的是**不溢出**，而这里要守住的是**保持为负**，两者不是同一件事，且后者
    的界更紧（低三十倍）。
    """
    with pytest.raises(ValueError, match="超出预算"):
        solve(_instance([1], [2**29], capacity=1))

    # 恰好在阈值下方：接受，且不可达的格子仍是负数（不是 0、不是正数）。
    table = _dp_table([1], [2**29 - 1], 3)
    assert table[-1].tolist() == [0, 2**29 - 1, -1, -1]
    assert int(table[-1, 2]) == -1
    assert int(table[-1].max()) == 2**29 - 1


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="长度"):
        solve({"capacity": 1, "values": [1], "weights": [1, 2]})
```

- [ ] **Step 2: 跑测试确认它失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_reference.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t2" -q 2>&1 | tail -5
```

Expected: 收集阶段失败，`ImportError: cannot import name 'ReferenceSolution'`。

- [ ] **Step 3: 写实现**

在 `experiment/eval/reference.py` 顶部补上导入与常量（放在已有的 `_UB_SCALE` 旁）：

```python
import hashlib
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
```

```python
#: 不可达状态的哨兵。工作数组与重建表统一用 int32。
_SENTINEL = -(2**29)

#: 价值总和的预算，**由哨兵决定而不是由 int32 的位宽决定**。
#:
#: DP 里不可达的格子会拿到 ``哨兵 + 沿路径的若干价值``。这种格子必须保持
#: **负数**，否则它会看起来像一个合法的价值：
#:
#:   - 价值总和达到 ``2**29`` 时，``哨兵 + v`` 恰好是 **0**——与「空集价值 0」
#:     无法区分；
#:   - 再大一点（如 ``2**30 - 1``）会得到 **正数**，于是 ``dp.max()`` 可能返回
#:     一个来自不可行路径的幻影最优值。
#:
#: 任何一条路径至多把每件物品加一次，所以路径上能累加的最大值就是所有正价值
#: 之和。于是「哨兵保持为负」的充要条件是 ``价值总和 < 2**29 = |哨兵|``。
#: 这个阈值恰好是哨兵的绝对值——不是巧合，它就是这条不变量的解。
#:
#: 依据设计文档第 4 节的数值界：冻结语料 max sum(values) = 17,232,254，
#: 比阈值低三十倍。这条界不是装饰，它由 `_dp_table` 在每次求解时实际校验
#: ——把假设写成断言，假设才可被检验。
_VALUE_SUM_LIMIT = -_SENTINEL  # 2**29
```

在文件末尾追加：

```python
def _dp_table(
    weights: Sequence[int], values: Sequence[int], capacity: int
) -> np.ndarray:
    """``(n+1) x (capacity+1)`` 的 int32 表，供取值与重建共用。

    ``table[i][c]`` = **恰好**用掉容量 ``c``、只考虑前 ``i`` 件物品时的最优
    价值；不可达置哨兵。``table[0][0] = 0``，其余为哨兵。

    最终答案取 ``table[-1].max()`` 而**不是** ``table[-1][capacity]``。

    内存：最坏情形 ``500 x 21202 x 4 B ≈ 42 MB``，逐实例分配并释放。若将来
    内存吃紧，升级路径是 Hirschberg 分治重建（``O(C)`` 空间）；**现在不做**
    ——42 MB 不是问题，为一个不存在的问题写分治重建是过度设计。
    """
    n = len(weights)
    if len(values) != n:
        raise ValueError(f"weights 与 values 长度不一致: {n} vs {len(values)}")
    if capacity < 0:
        raise ValueError(f"容量必须非负，收到 {capacity}")

    w = np.asarray(weights, dtype=np.int64)
    v = np.asarray(values, dtype=np.int64)
    positive = int(v[v > 0].sum()) if v.size else 0
    if positive >= _VALUE_SUM_LIMIT:
        raise ValueError(
            f"价值总和 {positive} 超出预算 {_VALUE_SUM_LIMIT}（= |哨兵|）；"
            "超过它，不可达格子就不再是负数，会伪装成合法价值。"
            "冻结语料实测的上界是 17,232,254"
        )

    table = np.full((n + 1, capacity + 1), _SENTINEL, dtype=np.int32)
    table[0, 0] = 0
    for i, (wi, vi) in enumerate(zip(w.tolist(), v.tolist())):
        row = table[i + 1]
        row[:] = table[i]  # 不选第 i 件
        if wi <= capacity:
            np.maximum(row[wi:], table[i][: capacity + 1 - wi] + vi, out=row[wi:])
    return table


def _reconstruct(
    table: np.ndarray, weights: Sequence[int], capacity: int, value: int
) -> list[int]:
    """从 DP 表回推一组达到 ``value`` 的物品索引。

    起点取**最小**的那个达到 ``value`` 的容量，使结果确定。逐件往前判定
    「这件用没用」：两行相等就说明不用它也能达到同样的值，一律判为**没用**，
    既正确又使并列时的结果确定。
    """
    n = len(weights)
    column = int(np.flatnonzero(table[n] == value)[0])
    selected: list[int] = []
    for i in range(n - 1, -1, -1):
        if table[i + 1, column] != table[i, column]:
            selected.append(i)
            column -= weights[i]
    return sorted(selected)


def _verify_reconstruction(
    weights: Sequence[int],
    values: Sequence[int],
    capacity: int,
    value: int,
    selected: Sequence[int],
) -> None:
    """强制自校验：集合必须**可行**且**价值恰好等于 V***。失败即抛错、不写行。

    第二条是 ``n`` 较大时唯一可用的强校验：``n<=15`` 有全枚举对照（E0），
    ``n=500`` 没有。两条合起来还顺带钉住了重建代码本身——只验证「值等于
    V*」会漏掉重建错误，而两条一起则要求结果**既可行又达到最优值**。
    """
    if len(set(selected)) != len(selected):
        raise ReferenceError(f"重建出的集合有重复物品: {list(selected)}")
    if any(not 0 <= i < len(weights) for i in selected):
        raise ReferenceError(f"重建出的索引越界: {list(selected)}")
    total_w = sum(weights[i] for i in selected)
    if total_w > capacity:
        raise ReferenceError(f"重建出的集合不可行: 重量 {total_w} > 容量 {capacity}")
    total_v = sum(values[i] for i in selected)
    if total_v != value:
        raise ReferenceError(f"重建出的集合价值 {total_v} != 最优值 {value}")


def item_set_hash(selected_indices: Sequence[int]) -> str:
    """被选**集合**的规范哈希。

    先排序再拼接，因此与传入次序无关（设计 §12）。只覆盖选择结果、不覆盖
    价值——这样「值对但集合错」的缺陷不会被哈希掩盖。刻意**不复用**
    ``generate.hashing.canonical_hash``：那是对 ``(n, capacity, weights,
    values)`` 的**实例身份**哈希，语义不同，混用会让审计时分不清「实例变了」
    与「解变了」。
    """
    payload = "|".join(str(i) for i in sorted(selected_indices))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()


@dataclass
class ReferenceSolution:
    """一个实例的精确参考解（设计 §4）。"""

    certified: bool
    item_set_hash: str
    runtime_s: float
    selected_mask: list[bool]
    upper_bound: Fraction
    value: int

    @property
    def selected_indices(self) -> list[int]:
        """由 ``selected_mask`` 导出。

        刻意**不存第二份**：两个字段会各自漂移，而它们必须永远一致。
        """
        return [i for i, used in enumerate(self.selected_mask) if used]


def solve(instance: Mapping) -> ReferenceSolution:
    """精确求解一个实例：V*、U、被选集合与自校验（设计 §4）。

    DP 是精确的整数算法，因此 ``certified`` 恒为 ``True``；保留该字段是为了
    符合 `EXPERIMENT_PLAN.md` 第 10 节对产物列的称呼。
    """
    weights = instance["weights"]
    values = instance["values"]
    capacity = instance["capacity"]

    start = time.perf_counter()
    table = _dp_table(weights, values, capacity)
    value = int(table[-1].max())
    selected = _reconstruct(table, weights, capacity, value)
    _verify_reconstruction(weights, values, capacity, value, selected)
    upper_bound = fractional_upper_bound(weights, values, capacity)
    runtime_s = time.perf_counter() - start

    mask = [False] * len(weights)
    for i in selected:
        mask[i] = True
    return ReferenceSolution(
        certified=True,
        item_set_hash=item_set_hash(selected),
        runtime_s=runtime_s,
        selected_mask=mask,
        upper_bound=upper_bound,
        value=value,
    )
```

- [ ] **Step 3b: 给 `format_upper_bound` 补上前置条件说明**

Task 1 的代码评审发现：`format_upper_bound` 的 docstring 说「向上取整」，但 `-(-n // d)` 只在**非负**时才是向上取整。实测 `format_upper_bound(Fraction(-1,3))` 返回 `-1.666667`，而 `-1/3` 的六位小数上界应为 `-0.333333`——是一个**静默的错值**。

本项目里 `U >= 0` 由构造保证（权重非负 ⇒ `remaining >= 0`；价值非负 ⇒ 每个加数非负），所以**不设运行时断言**——那是为一个不可能出现的状态写守卫。改为把前置条件写进 docstring。把 `format_upper_bound` 的 docstring 替换为：

```python
    """把 U 写成**向上取整**到 6 位小数的十进制字符串。

    向上取整而非四舍五入或下取整：存储后的值必须仍是一个合法上界。

    用整数运算而不是 ``decimal``：decimal 的上下文精度会变成一个隐式的输入，
    而这里没有理由引入它。

    **前置条件：``upper_bound >= 0``。** ``-(-n // d)`` 只在非负时才有「向上
    取整」的含义；负数会得到一个静默的错值（实测
    ``format_upper_bound(Fraction(-1, 3))`` 返回 ``-1.666667``，而 ``-1/3`` 的
    六位小数上界是 ``-0.333333``）。本项目里 ``U >= 0`` 由构造保证（权重与价值
    都非负），所以这里不设运行时断言——那是为一个不可能出现的状态写守卫；改为
    把前置条件写在这里，让调用方看得见。
    """
```

只改 docstring，一行代码都不动。

- [ ] **Step 4: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_reference.py tests/test_eval_upper_bound.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t2" -q 2>&1 | tail -3
```

Expected: `22 passed`（Task 2 的 7 条 + Task 1 的 15 条）。

- [ ] **Step 5: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/eval/reference.py tests/test_eval_reference.py && git commit -m "feat(eval): 整数 DP、集合重建与强制自校验"
```

---

## Task 3: 单实例指标原语

**Files:**
- Create: `experiment/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_eval_metrics.py`：

```python
"""单实例指标原语（`EXPERIMENT_PLAN.md` 第 7 节）。"""

from fractions import Fraction

import pytest

from experiment.eval.metrics import (
    bound_gap,
    exact_gap,
    feasibility_violations,
    is_trivial_zero,
    normalized_quality,
)
from experiment.eval.reference import solve

INSTANCE = {"capacity": 5, "values": [1, 1, 1], "weights": [3, 3, 3]}


def test_feasibility_counts_each_kind_of_violation_separately():
    """返回的是**条数**而不是布尔值，三类问题才不会塌缩成一个 False。"""
    assert feasibility_violations(INSTANCE, [0, 1]) == 0
    assert feasibility_violations(INSTANCE, [0, 0]) == 1        # 重复选择
    assert feasibility_violations(INSTANCE, [0, 7]) == 1        # 选了不存在的物品
    assert feasibility_violations(INSTANCE, [0, 1, 2]) == 1     # 超容量（9 > 5）
    assert feasibility_violations(INSTANCE, [0, 9, 0, 0]) == 3  # 越界 1 + 重复 2
    assert feasibility_violations(INSTANCE, []) == 0


def test_gaps_use_the_plan_percentage_scale():
    assert exact_gap(90, 100) == pytest.approx(10.0)
    assert bound_gap(90, Fraction(120)) == pytest.approx(25.0)
    assert normalized_quality(90, Fraction(120)) == pytest.approx(0.75)


def test_zero_denominators_raise_instead_of_returning_nan():
    """静默的 nan 会一路污染到汇总表（设计 §7）。"""
    with pytest.raises(ValueError, match="精确间隙"):
        exact_gap(0, 0)
    with pytest.raises(ValueError, match="上界"):
        bound_gap(0, Fraction(0))
    with pytest.raises(ValueError, match="归一化"):
        normalized_quality(0, Fraction(0))


def test_normalized_quality_never_exceeds_one():
    """``V/U <= 1`` 由构造保证（U >= V*），这正是它适合做检查点分数的原因。"""
    result = solve({"capacity": 9, "values": [10, 7, 9], "weights": [8, 4, 5]})
    assert 0.0 <= normalized_quality(result.value, result.upper_bound) <= 1.0


def test_trivial_is_about_fitting_not_about_value():
    """定义是「没有物品装得下」，不是「V* == 0」。

    两者在本语料上等价（价值恒 >= 1），但不是同一个命题。
    """
    assert is_trivial_zero({"capacity": 9, "weights": [10, 20], "values": [5, 6]})
    assert not is_trivial_zero({"capacity": 10, "weights": [10, 20], "values": [5, 6]})
    # 价值为 0 但装得下：V* 也是 0，却**不是**平凡实例。
    assert not is_trivial_zero({"capacity": 10, "weights": [1], "values": [0]})
```

- [ ] **Step 2: 跑测试确认它失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_metrics.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t3" -q 2>&1 | tail -5
```

Expected: `ModuleNotFoundError: No module named 'experiment.eval.metrics'`。

- [ ] **Step 3: 写实现**

创建 `experiment/eval/metrics.py`：

```python
"""单实例指标原语（`EXPERIMENT_PLAN.md` 第 7 节）。

只做**单实例**的计算；跨实例聚合、表格、图与 1000 次配对 bootstrap 属子项目 D。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence


def feasibility_violations(instance: Mapping, selected_indices: Sequence[int]) -> int:
    """返回选择的**违例条数**，0 表示合法。

    刻意返回条数而不是布尔值：这样「选了不存在的物品」「重复选择」「超容量」
    三类问题可以分别计数，而不是塌缩成一个 False。
    """
    weights = instance["weights"]
    capacity = instance["capacity"]
    n = len(weights)

    violations = 0
    seen: set[int] = set()
    for i in selected_indices:
        if not 0 <= i < n:
            violations += 1
        if i in seen:
            violations += 1
        seen.add(i)

    total = sum(weights[i] for i in selected_indices if 0 <= i < n)
    if total > capacity:
        violations += 1
    return violations


def exact_gap(value: float, optimum: float) -> float:
    """精确最优间隙 ``100*(V*-V)/V*``。

    分母为 0 时抛错而不是返回 ``inf``/``nan``：调用方必须先按
    `EXPERIMENT_PLAN.md` 第 8 节剔除平凡解。
    """
    if optimum == 0:
        raise ValueError(
            "精确间隙的分母 V* 为 0；平凡解必须先按 EXPERIMENT_PLAN.md 第 8 节剔除"
        )
    return 100.0 * (optimum - value) / optimum


def bound_gap(value: float, upper_bound: Fraction | float) -> float:
    """基于上界的间隙 ``100*(U-V)/U``。"""
    ub = Fraction(upper_bound)
    if ub == 0:
        raise ValueError("基于上界的间隙分母 U 为 0")
    return 100.0 * float((ub - Fraction(value)) / ub)


def normalized_quality(value: float, upper_bound: Fraction | float) -> float:
    """归一化质量 ``V/U``，供 `EXPERIMENT_PLAN.md` 第 6 节的检查点选择。"""
    ub = Fraction(upper_bound)
    if ub == 0:
        raise ValueError("归一化质量的分母 U 为 0")
    return float(Fraction(value) / ub)


def is_trivial_zero(instance: Mapping) -> bool:
    """没有任何物品装得下。

    按 `EXPERIMENT_PLAN.md` 第 7 节的**定义**（「if no item fits」）实现，而不是
    「V* == 0」：两者在本语料上等价（价值恒 >= 1），但不是同一个命题——若出现
    价值为 0 的物品，V* == 0 而本函数仍为 False。

    空物品列表返回 True：``all`` 对空序列为真，而「没有物品装得下」确实成立。
    这是有意的边界语义，不是「在空列表上空转」的那类断言。
    """
    capacity = instance["capacity"]
    return all(w > capacity for w in instance["weights"])
```

- [ ] **Step 4: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_metrics.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t3" -q 2>&1 | tail -3
```

Expected: `5 passed`。

- [ ] **Step 5: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/eval/metrics.py tests/test_eval_metrics.py && git commit -m "feat(eval): 单实例指标原语与零分母的显式失败"
```

---

## Task 4: E0 —— 与全子集枚举对照

**Files:**
- Test: `tests/test_eval_e0.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_eval_e0.py`：

```python
"""E0：与全子集枚举对照（`EXPERIMENT_PLAN.md` 第 8 节）。

实例来源复用冻结的 ``build_instance``：它的 ``n`` 是自由参数、无最小尺寸约束，
取专用 split 名 ``e0`` 即得到确定性种子。这样既不新写生成器，也不改动
``experiment/data/generate/*``（冻结目录）。
"""

import numpy as np
import pytest

from experiment.data.generate.corpus import build_instance
from experiment.eval.reference import solve

E0_N = 15
E0_COUNT = 100
FAMILIES = ("uniform", "normal", "correlated", "skewed")
ALPHAS = (0.25, 0.50, 0.75)


def brute_force_best(weights, values, capacity) -> int:
    """枚举全部 ``2**n`` 个子集，返回可行子集的最大价值。

    与 DP **算法不同**，所以两者的分歧是真正的信号，而不是同一份实现的回声。
    空子集必然可行（重量 0），故 ``total_v[...]`` 永不为空。
    """
    n = len(weights)
    bits = (np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
    total_w = bits @ np.asarray(weights, dtype=np.int64)
    total_v = bits @ np.asarray(values, dtype=np.int64)
    return int(total_v[total_w <= capacity].max())


def optimal_subsets(weights, values, capacity) -> list[list[int]]:
    """列出**全部**达到最优的索引集合，用来确认手工例子的最优解唯一。"""
    best, winners = -1, []
    for mask in range(1 << len(weights)):
        chosen = [i for i in range(len(weights)) if mask >> i & 1]
        if sum(weights[i] for i in chosen) > capacity:
            continue
        total = sum(values[i] for i in chosen)
        if total > best:
            best, winners = total, [chosen]
        elif total == best:
            winners.append(chosen)
    return winners


def e0_instances() -> list[dict]:
    """100 个 ``n=15`` 实例，覆盖四个族与三个 alpha。

    用普通函数而不是夹具：它没有 I/O，且下面两个测试必须看到**同一批**实例，
    否则「值对照」与「集合对照」可能落在不同的实例上。
    """
    return [
        build_instance(
            "e0",
            FAMILIES[index % len(FAMILIES)],
            E0_N,
            ALPHAS[(index // len(FAMILIES)) % len(ALPHAS)],
            index,
        )
        for index in range(E0_COUNT)
    ]


@pytest.fixture(scope="module")
def e0() -> list[tuple[dict, int, object]]:
    """跑一次：每项是 ``(实例, 枚举最优值, 参考解)``。"""
    return [
        (
            instance,
            brute_force_best(
                instance["weights"], instance["values"], instance["capacity"]
            ),
            solve(instance),
        )
        for instance in e0_instances()
    ]


def test_dp_value_matches_full_enumeration(e0):
    for instance, expected, result in e0:
        assert result.value == expected, instance["instance_id"]


def test_rebuilt_set_is_feasible_and_reaches_the_enumerated_optimum(e0):
    """不只比值，还比**集合**：只比值会漏掉「值对但集合错」。"""
    for instance, expected, result in e0:
        weights, values = instance["weights"], instance["values"]
        chosen = result.selected_indices
        assert len(set(chosen)) == len(chosen), instance["instance_id"]
        assert (
            sum(weights[i] for i in chosen) <= instance["capacity"]
        ), instance["instance_id"]
        assert sum(values[i] for i in chosen) == expected, instance["instance_id"]
        assert result.selected_mask.count(True) == len(chosen), instance["instance_id"]


def test_the_three_item_example_has_a_unique_optimum():
    """`EXPERIMENT_PLAN.md` 第 8 节要求的三物品例子。

    最优解**唯一**是有意构造的：若有两组同值的最优集合，「选错集合」这个缺陷
    就会被值的相等掩盖。
    """
    weights, values, capacity = [8, 4, 5], [10, 7, 9], 9
    assert optimal_subsets(weights, values, capacity) == [[1, 2]]
    result = solve(
        {"capacity": capacity, "values": values, "weights": weights}
    )
    assert result.selected_indices == [1, 2]
```

- [ ] **Step 2: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_e0.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t4" -q 2>&1 | tail -3
```

Expected: `3 passed`。实测整套约 1 s（探针测得 100 个 n=15 实例的 solve + 全枚举共 0.26 s，`build_instance` 的生成开销另计）。

> 本任务没有「先红」的一步：它测的是 Task 2 已经交付的求解器，写下来就该是绿的。若它**红**了，说明 Task 2 的实现有问题，回去修 Task 2，不要在这里放宽断言。

- [ ] **Step 3: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add tests/test_eval_e0.py && git commit -m "test(eval): E0 全子集枚举对照与唯一最优的三物品例子"
```

---

## Task 5: 金样本

**Files:**
- Create: `tests/golden/reference_solutions.json`
- Test: `tests/test_eval_golden.py`

- [ ] **Step 1: 用仓库外的脚本生成金样本**

在 `$TEMP/probe_evaldesign/make_golden.py` 写下并运行：

```python
"""生成 tests/golden/reference_solutions.json 的内容（仓库外运行）。"""

import json
import sys
from pathlib import Path

REPO = Path(r"C:\work\学校\A.SPBU\Saint-peter-26\AI-2\project")
sys.path.insert(0, str(REPO))

from experiment.eval.reference import format_upper_bound, solve  # noqa: E402

CASES = [
    ("three_item_unique_optimum", [8, 4, 5], [10, 7, 9], 9),
    ("optimum_leaves_slack", [6, 7], [6, 7], 10),
    ("tied_optimum_prefers_skipping", [5, 5], [7, 7], 5),
    ("trivial_no_item_fits", [10, 20], [5, 6], 9),
    ("fractional_fill_bound", [3, 2], [1, 5], 4),
]

instances = []
for name, weights, values, capacity in CASES:
    result = solve(
        {"capacity": capacity, "values": values, "weights": weights}
    )
    instances.append(
        {
            "capacity": capacity,
            "item_set_hash": result.item_set_hash,
            "name": name,
            "selected_indices": result.selected_indices,
            "upper_bound": format_upper_bound(result.upper_bound),
            "value": result.value,
            "values": values,
            "weights": weights,
        }
    )

payload = json.dumps({"instances": instances}, indent=2, sort_keys=True, ensure_ascii=False)
print(payload)
(Path(__file__).with_name("golden.json")).write_text(payload + "\n", encoding="utf-8", newline="")
```

```bash
"/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project/venv/Scripts/python.exe" -B "$TEMP/probe_evaldesign/make_golden.py"
```

- [ ] **Step 2: 逐条核对，不符就停下报告**

把脚本输出与下表逐格比对。**每一格都是设计探针实测出来的**；任何一格对不上，说明 Task 1/2 的实现与探针不同——停下来报告，**不要改这张表**。

| name | weights | values | capacity | V* | selected | `upper_bound` |
| --- | --- | --- | --- | --- | --- | --- |
| `three_item_unique_optimum` | [8,4,5] | [10,7,9] | 9 | 16 | [1,2] | `16.000000` |
| `optimum_leaves_slack` | [6,7] | [6,7] | 10 | 7 | [1] | `10.000000` |
| `tied_optimum_prefers_skipping` | [5,5] | [7,7] | 5 | 7 | [0] | `7.000000` |
| `trivial_no_item_fits` | [10,20] | [5,6] | 9 | 0 | [] | `4.500000` |
| `fractional_fill_bound` | [3,2] | [1,5] | 4 | 5 | [1] | `5.666667` |

`item_set_hash` 的期望值：

| name | hash |
| --- | --- |
| `three_item_unique_optimum` | `df2d4757898e0ca107991603786c25d0d7f86e6e363bff9041b11b1e48933a0b` |
| `optimum_leaves_slack` | `92cdf578c47085a5992256f0dcf97d0b19f1f1c9de4d5fe30c3ace6191b6e5db` |
| `tied_optimum_prefers_skipping` | `0fd923ca5e7218c4ba3c3801c26a617ecdbfdaebb9c76ce2eca166e7855efbb8` |
| `trivial_no_item_fits` | `0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8` |
| `fractional_fill_bound` | `92cdf578c47085a5992256f0dcf97d0b19f1f1c9de4d5fe30c3ace6191b6e5db` |

两条需要理解、不要「修好」的事：

- **第 2 条与第 5 条的哈希相同。** 两者都选中了索引 `[1]`，而哈希只覆盖选择结果、不覆盖实例身份（设计 §4）。这是**正确**的行为，不是碰撞。
- **第 4 条的哈希是空字符串的 blake2b-256 摘要。** 平凡解选中空集，因此所有平凡实例共享同一个哈希。同样正确。

- [ ] **Step 3: 落盘金样本**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && mkdir -p tests/golden && cp "$TEMP/probe_evaldesign/golden.json" tests/golden/reference_solutions.json && ./venv/Scripts/python.exe -c "import pathlib; raw=pathlib.Path('tests/golden/reference_solutions.json').read_bytes(); print('bom', raw[:3]==b'\xef\xbb\xbf', 'cr', b'\r' in raw, 'tail', raw[-2:])"
```

Expected: `bom False cr False tail b'}\n'`。

- [ ] **Step 4: 写测试**

创建 `tests/test_eval_golden.py`：

```python
"""金样本：钉住求解器的回归（设计 §12）。

金样本**自包含**（内联实例数据），不依赖被 gitignore 的语料：参考解的正确性
不应该只有在拿到 522,504 行语料时才可验证。

本模块**没有**「缺失即生成」的回退：那种回退会让金样本在文件丢失时静默通过，
而本项目的验收要求全绿且无 skipped。
"""

import json
from fractions import Fraction
from pathlib import Path

from experiment.eval.reference import (
    format_upper_bound,
    fractional_upper_bound,
    solve,
)

GOLDEN = Path(__file__).parent / "golden" / "reference_solutions.json"

#: 每条记录必须**恰好**有这些键：多一个意味着有人塞进了未钉住的字段（比如
#: ``runtime_s`` 这种不可复现的墙钟量），少一个意味着钉住退化了。
FIELDS = frozenset(
    {
        "capacity",
        "item_set_hash",
        "name",
        "selected_indices",
        "upper_bound",
        "value",
        "values",
        "weights",
    }
)


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _instance(case: dict) -> dict:
    return {
        "capacity": case["capacity"],
        "values": case["values"],
        "weights": case["weights"],
    }


def test_golden_carries_exactly_the_pinned_fields():
    assert len(_golden()["instances"]) == 5
    for case in _golden()["instances"]:
        assert set(case) == FIELDS, case["name"]


def test_solver_reproduces_the_golden():
    for case in _golden()["instances"]:
        result = solve(_instance(case))
        assert result.value == case["value"], case["name"]
        assert result.selected_indices == case["selected_indices"], case["name"]
        assert result.item_set_hash == case["item_set_hash"], case["name"]
        assert format_upper_bound(result.upper_bound) == case["upper_bound"], case["name"]


def test_stored_upper_bounds_are_upward_rounded_and_tight():
    """金样本里的 U 是十进制字符串（JSON 没有 ``Fraction``），口径与 CSV 相同。

    既钉方向（存储值 >= 精确值），也钉紧度（最多高 1e-6）——只钉方向会让
    「无脑放大」也通过。
    """
    for case in _golden()["instances"]:
        exact = fractional_upper_bound(
            case["weights"], case["values"], case["capacity"]
        )
        stored = Fraction(case["upper_bound"])
        assert stored >= exact, case["name"]
        assert stored - exact < Fraction(1, 10**6), case["name"]
        assert stored >= case["value"], case["name"]
```

- [ ] **Step 5: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_golden.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t5" -q 2>&1 | tail -3
```

Expected: `3 passed`。

- [ ] **Step 6: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add tests/golden/reference_solutions.json tests/test_eval_golden.py && git commit -m "test(eval): 内联 5 例金样本锁定参考解回归"
```

---

## Task 6: 生产形状语料上的不变量

**Files:**
- Test: `tests/test_eval_properties.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_eval_properties.py`：

```python
"""性质测试：生产形状语料上的不变量（设计 §12）。

为什么用**生成器产出的实例**而不是直接读 ``datasets/``：该目录被 gitignore，
新克隆上不存在，依赖它会让测试静默跳过或静默通过——这正是本项目的验收不
允许的。真要验证「冻结语料上 U >= V* 处处成立」，那是 Task 7 的 CLI 验收
步骤：它在真实语料上跑一遍并写进报告，而不是伪装成一条 CI 断言。

用 ``quota_override`` 把每 cell 压到 1 条，但**族、n、alpha 一个不少**，因此
容量量级与真实语料同阶（n=200 时上万格）——这正是「上界算成下界」或「浮点
排序」会露馅的地方。
"""

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from experiment.data.generate.corpus import run_generation
from experiment.data.spec import SPLITS
from experiment.eval.reference import fractional_upper_bound, item_set_hash, solve

#: 4 族 x 4 个 n x 3 个 alpha x 3 个 split
EXPECTED_INSTANCES = 144


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory) -> list[dict]:
    out = tmp_path_factory.mktemp("eval_gen")
    run_generation(out_dir=out, quota_override={split: 1 for split in SPLITS})
    instances: list[dict] = []
    for split in SPLITS:
        for cell in sorted((out / split).glob("*.jsonl")):
            for line in cell.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    instances.append(json.loads(line))
    return instances


def _solve_with(instance: Mapping, weights, values):
    """用**替代的**物品数据求解，容量与 ``instance_id`` 仍取自原实例。"""
    return solve(
        {
            "capacity": instance["capacity"],
            "instance_id": instance["instance_id"],
            "values": values,
            "weights": weights,
        }
    )


def test_upper_bound_dominates_the_optimum(generated_instances):
    assert len(generated_instances) == EXPECTED_INSTANCES
    for instance in generated_instances:
        result = solve(instance)
        upper = fractional_upper_bound(
            instance["weights"], instance["values"], instance["capacity"]
        )
        assert result.upper_bound == upper, instance["instance_id"]
        assert upper >= result.value, instance["instance_id"]


def test_reconstructed_sets_are_feasible_and_optimal(generated_instances):
    for instance in generated_instances:
        weights, values = instance["weights"], instance["values"]
        result = solve(instance)
        chosen = result.selected_indices
        assert len(set(chosen)) == len(chosen), instance["instance_id"]
        assert (
            sum(weights[i] for i in chosen) <= instance["capacity"]
        ), instance["instance_id"]
        assert sum(values[i] for i in chosen) == result.value, instance["instance_id"]


def test_upper_bound_and_optimum_are_permutation_invariant(generated_instances):
    """``U`` 与 ``V*`` 是物品**多重集**的函数，与存储顺序无关。

    刻意**不**断言 ``item_set_hash`` 也不变：打乱物品顺序会把最优解映射成
    另一组索引，哈希因此本就应该变（并列最优时尤甚）。
    """
    rng = np.random.default_rng(20260929)
    for instance in generated_instances:
        weights, values = instance["weights"], instance["values"]
        capacity = instance["capacity"]
        permutation = rng.permutation(len(weights)).tolist()
        shuffled_w = [weights[i] for i in permutation]
        shuffled_v = [values[i] for i in permutation]

        assert (
            fractional_upper_bound(shuffled_w, shuffled_v, capacity)
            == fractional_upper_bound(weights, values, capacity)
        ), instance["instance_id"]
        assert (
            _solve_with(instance, shuffled_w, shuffled_v).value
            == solve(instance).value
        ), instance["instance_id"]


def test_item_set_hash_depends_on_the_set_not_the_list_order():
    """哈希先排序再拼接，因此与传入次序无关（设计 §12）。"""
    assert item_set_hash([2, 0, 1]) == item_set_hash([0, 1, 2])
    assert item_set_hash([5, 3]) == item_set_hash([3, 5])
    assert item_set_hash([0, 1]) != item_set_hash([0, 2])
    assert item_set_hash([1]) != item_set_hash([1, 1])


def test_adversarial_shapes_keep_the_bound_and_the_reconstruction_valid():
    """边界形状：容量 0、一件都整件装不下、同比值并列、零重量的物品。

    这些形状真实语料里一个都没有（实测平凡解 0/17700），所以必须合成。
    """
    instances = [
        {"capacity": 0, "values": [1, 2], "weights": [1, 2]},
        {"capacity": 1, "values": [5, 6], "weights": [2, 3]},
        {"capacity": 6, "values": [1, 2, 3], "weights": [1, 2, 3]},
        {"capacity": 5, "values": [7, 7], "weights": [5, 5]},
        {"capacity": 10, "values": [0, 0], "weights": [1, 2]},
        {"capacity": 3, "values": [4], "weights": [0]},
    ]
    for instance in instances:
        result = solve(instance)
        upper = fractional_upper_bound(
            instance["weights"], instance["values"], instance["capacity"]
        )
        assert upper >= result.value, instance
        chosen = result.selected_indices
        assert (
            sum(instance["weights"][i] for i in chosen) <= instance["capacity"]
        ), instance
        assert sum(instance["values"][i] for i in chosen) == result.value, instance
```

- [ ] **Step 2: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_properties.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t6" -q 2>&1 | tail -3
```

Expected: `5 passed`，约 5–15 s（144 个实例的生成 + 数百次 `solve`，其中 n=200 的实例每次要建 201×41329 的表）。

- [ ] **Step 3: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add tests/test_eval_properties.py && git commit -m "test(eval): 生产形状语料上的上界性质与排列不变性"
```

---

## Task 7: `reference_results.csv`

**Files:**
- Create: `experiment/eval/cli.py`
- Test: `tests/test_eval_cli.py`
- Produce: `reference_results.csv`（仓库根目录，入库）

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_eval_cli.py`：

```python
"""``reference_results.csv`` 的形状与字节（设计 §9）。"""

import csv
import json
from fractions import Fraction
from pathlib import Path

import pytest

from experiment.data.spec import E3_NAME, E4_NAME
from experiment.eval import cli

#: (instance_id, 实例体)。两个 id 的字典序与它们在语料里的出现顺序相反，
#: 这样「行序按 instance_id 升序」才有判别力。
INSTANCES = [
    ("b-second", {"capacity": 5, "values": [7, 1], "weights": [5, 1]}),
    ("a-first", {"capacity": 9, "values": [10, 7, 9], "weights": [8, 4, 5]}),
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """手写的五文件语料：前三个 split 各一/零条，E3 一条，E4 空。

    刻意手写而不是跑 ``run_assembly``：这里要测的是 **CLI 的读取与落盘**，
    不是装配；手写让「读到了哪些文件」完全可控。
    """
    root = tmp_path / "processed"
    layout = {
        root / "train" / "instances.jsonl": [INSTANCES[0]],
        root / "validation" / "instances.jsonl": [INSTANCES[1]],
        root / "test" / "instances.jsonl": [],
        root / "stress" / E3_NAME / "instances.jsonl": [INSTANCES[0]],
        root / "stress" / E4_NAME / "instances.jsonl": [],
    }
    for path, items in layout.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "capacity": body["capacity"],
                    "instance_id": name,
                    "values": body["values"],
                    "weights": body["weights"],
                },
                sort_keys=True,
            )
            for name, body in items
        ]
        path.write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return root


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_corpus_file_is_read_and_the_row_order_is_fixed(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    summary = cli.run(corpus, out)
    # train 1 + validation 1 + test 0 + E3 1 + E4 0
    assert summary["rows"] == 3
    # 按 instance_id 升序，与文件顺序无关：a-first 在 validation 里，却排在前面
    assert [row["instance_id"] for row in _rows(out)] == [
        "a-first",
        "b-second",
        "b-second",
    ]


def test_column_order_and_bytes_are_fixed(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    cli.run(corpus, out)

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")

    with out.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(cli.COLUMNS)


def test_certified_is_lowercase_true_everywhere(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    cli.run(corpus, out)
    assert {row["certified"] for row in _rows(out)} == {"true"}


def test_upper_bound_is_an_upward_rounded_decimal_string(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    cli.run(corpus, out)
    for row in _rows(out):
        integer, _, decimals = row["upper_bound"].partition(".")
        assert integer and len(decimals) == 6, row
        assert Fraction(row["upper_bound"]) >= int(row["value"]), row


def test_only_the_time_column_is_unstable(corpus, tmp_path):
    """两次运行除 ``solver_time_s`` 外逐格相同（设计 §9）。

    该列是墙钟测量。**任何把整份 CSV 一起哈希**的校验都会重蹈 Plan 3 的覆辙：
    当时一个存活探针把含临时路径与时钟的 manifest 一起哈希，两次纯净运行都
    不一致，于是每一个「存活」判定都是假的。比对必须排除这一列。
    """
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.run(corpus, first)
    cli.run(corpus, second)

    def without_time(path: Path) -> list[list[str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        index = rows[0].index("solver_time_s")
        return [[cell for i, cell in enumerate(row) if i != index] for row in rows]

    assert without_time(first) == without_time(second)
```

- [ ] **Step 2: 跑测试确认它失败**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_cli.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t7" -q 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'cli' from 'experiment.eval'`。

- [ ] **Step 3: 写实现**

创建 `experiment/eval/cli.py`：

```python
"""``reference_results.csv`` 的生成入口。

用法::

    python -m experiment.eval.cli
    python -m experiment.eval.cli --corpus datasets/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from experiment.data.spec import (
    DATASETS_PROCESSED,
    E3_NAME,
    E4_NAME,
    REPO_ROOT,
    SPLITS,
)
from experiment.eval.reference import format_upper_bound, solve

#: 产物放在**仓库根目录并入库**，与 data_manifest.json / experiment_config.yaml
#: 一致：datasets/ 整体被 gitignore，把审计凭据放在那里等于换台机器就无从证明
#: 数据未被调换。
#:
#: 刻意**不**写进 spec.py：generate/corpus.py 的 generator_fingerprint() 会把
#: spec.py 的字节纳入哈希，而该指纹已写进入库的 data_manifest.json；往里加一个
#: 常量会让「重新生成是否一致」这个复核动作失真。
DEFAULT_OUT: Path = REPO_ROOT / "reference_results.csv"

#: 列序即产物的一部分。
COLUMNS: tuple[str, ...] = (
    "instance_id",
    "certified",
    "value",
    "upper_bound",
    "item_set_hash",
    "solver_time_s",
)


def corpus_files(root: Path) -> list[Path]:
    """语料的五个文件，顺序固定：三个 split 之后是 E3、E4。"""
    root = Path(root)
    files = [root / split / "instances.jsonl" for split in SPLITS]
    files.append(root / "stress" / E3_NAME / "instances.jsonl")
    files.append(root / "stress" / E4_NAME / "instances.jsonl")
    return files


def read_instances(files: Sequence[Path]) -> list[dict]:
    instances: list[dict] = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    instances.append(json.loads(line))
    return instances


def run(corpus_root: Path, out_path: Path) -> dict:
    """求解全部实例并落盘，返回一份汇总。

    途中会校验 ``U >= V*`` 并对违例直接报错——这是设计 §12 的「上界性质」
    在**真实语料**上的执行点；CI 里跑的是同阶形状的合成语料（测试不读被
    gitignore 的 datasets/）。
    """
    rows = []
    for instance in read_instances(corpus_files(corpus_root)):
        result = solve(instance)
        if result.upper_bound < result.value:
            raise AssertionError(
                f"{instance['instance_id']}: 上界 {result.upper_bound} "
                f"< 最优值 {result.value}"
            )
        rows.append(
            {
                "certified": "true" if result.certified else "false",
                "instance_id": instance["instance_id"],
                "item_set_hash": result.item_set_hash,
                "solver_time_s": f"{result.runtime_s:.6f}",
                "upper_bound": format_upper_bound(result.upper_bound),
                "value": str(result.value),
            }
        )

    # 行序是产物的一部分：按 instance_id 升序，确定且与语料的文件顺序无关。
    rows.sort(key=lambda row: row["instance_id"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" 关掉 Python 的换行翻译，lineterminator="\n" 固定为 LF。
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "total_solver_time_s": sum(float(row["solver_time_s"]) for row in rows),
        "trivial_zero": sum(1 for row in rows if row["value"] == "0"),
        "out": out_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为冻结语料生成精确参考解与上界")
    parser.add_argument(
        "--corpus", type=Path, default=DATASETS_PROCESSED, help="冻结语料根目录"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="reference_results.csv 的位置"
    )
    return parser


def main() -> int:
    # 与 generate/collect/assemble 的入口一致：重定向到文件时 Windows 用 ANSI
    # 代码页编码 stdout，中文标签会抛 UnicodeEncodeError。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    start = time.perf_counter()
    summary = run(args.corpus, args.out)

    print(f"语料根目录: {args.corpus}")
    print(f"产物:       {summary['out']}")
    print(f"实例数:     {summary['rows']}")
    print(f"平凡解:     {summary['trivial_zero']}（已从百分比分母剔除）")
    print(f"求解总耗时: {summary['total_solver_time_s']:.1f} s")
    print(f"墙钟总耗时: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认它通过**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -m pytest tests/test_eval_cli.py -p no:cacheprovider --basetemp="$TEMP/ap_ev_t7" -q 2>&1 | tail -3
```

Expected: `5 passed`。

- [ ] **Step 5: 在真实冻结语料上做验收运行**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && ./venv/Scripts/python.exe -B -m experiment.eval.cli
```

Expected（数值来自设计探针的实测，允许机器差异）：

```
实例数:     17700
平凡解:     0（已从百分比分母剔除）
求解总耗时: ~16 s
```

若 `U >= V*` 有任何违例，`run()` 会直接抛 `AssertionError`——那说明实现有 bug，停下来报告。

- [ ] **Step 6: 独立复核产物**

在仓库外写 `$TEMP/probe_evaldesign/verify_csv.py` 并运行：

```python
"""独立复核 reference_results.csv（仓库外运行，只读）。"""

import csv
from fractions import Fraction
from pathlib import Path

CSV = Path(r"C:\work\学校\A.SPBU\Saint-peter-26\AI-2\project\reference_results.csv")

with CSV.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

assert list(rows[0]) == [
    "instance_id",
    "certified",
    "value",
    "upper_bound",
    "item_set_hash",
    "solver_time_s",
], list(rows[0])

ids = [row["instance_id"] for row in rows]
assert ids == sorted(ids), "row order is not instance_id ascending"
assert len(set(ids)) == len(ids), "duplicate instance_id"

bad = [row["instance_id"] for row in rows if Fraction(row["upper_bound"]) < int(row["value"])]
print("rows:", len(rows))
print("U < V* rows:", len(bad))
print("trivial (V*=0):", sum(1 for row in rows if row["value"] == "0"))
print("distinct item_set_hash:", len({row["item_set_hash"] for row in rows}))
print("certified values:", {row["certified"] for row in rows})
print("total solver time: %.1f s" % sum(float(row["solver_time_s"]) for row in rows))
print("zero-length item_set_hash:", sum(1 for row in rows if not row["item_set_hash"]))
```

```bash
"/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project/venv/Scripts/python.exe" -B "$TEMP/probe_evaldesign/verify_csv.py"
```

Expected: `rows: 17700`、`U < V* rows: 0`、`trivial (V*=0): 0`、`certified values: {'true'}`、`zero-length item_set_hash: 0`。

- [ ] **Step 7: 确认入库字节是 LF 且整套测试仍全绿**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add reference_results.csv && git ls-files --eol reference_results.csv && ./venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp="$TEMP/ap_ev_t7all" -q 2>&1 | tail -3
```

`git ls-files --eol` 必须显示 `i/lf`（索引区 LF）。`.gitattributes` 已有 `*.csv text eol=lf`，若仍显示 `i/crlf`，说明该规则没生效——停下来报告，不要手工改文件绕过。

Expected: `350 passed`，0 failed、0 skipped（基线 307 + 本计划新增 43 条：Task 1 的 15 条、Task 2 的 7 条、Task 3 的 5 条、Task 4 的 3 条、Task 5 的 3 条、Task 6 的 5 条、Task 7 的 5 条。以实际收集数为准，但 **0 failed 与 0 skipped 是硬要求**）。

- [ ] **Step 8: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add experiment/eval/cli.py tests/test_eval_cli.py reference_results.csv && git commit -m "feat(eval): reference_results.csv 与真实语料的验收运行"
```

---

## Task 8: `EXPERIMENT_PLAN.md` §7 的修订说明

**Files:**
- Modify: `EXPERIMENT_PLAN.md`（在第 7 节末尾、`## 8. Experiment Matrix` 之前**追加**一段）

- [ ] **Step 1: 追加修订说明**

在 `EXPERIMENT_PLAN.md` 中，找到第 7 节最后一段的结尾：

```
... Report training-seed variation separately from data-sampling uncertainty.
```

在它之后、`## 8. Experiment Matrix` 之前，插入下面这段（**不删原文**，读者要能看到决策过程）：

```markdown
> **Revision (2026-09-29) — the reference solver is integer DP, not CP-SAT.**
>
> The paragraph above says "integer dynamic programming for small cases ... or
> OR-Tools CP-SAT ... for the general case". Measured on the frozen corpus, the
> general case never arrives: the largest capacity is 41,328 (n=200; 21,201 at
> n=500) and exact DP over all 17,700 instances takes 5.2 s for values alone and
> 16.0 s including the reconstruction tables, on one machine. Every instance is
> therefore solvable exactly, and `certified` is `true` for all of them.
>
> Consequences, recorded rather than silently absorbed:
>
> - OR-Tools is **not** a dependency. The 10-second CP-SAT timeout, the `L`/`U`
>   dual track, the certification rate, and the bound-based gap as a *fallback*
>   never occur in the implementation. The `certified` column exists and is
>   always `true`, so the artifact still matches this section's column list.
> - The metric table above **stays in force**, with the exact optimality gap as
>   the primary measure.
> - `U` is still computed and still load-bearing: section 6 selects checkpoints
>   on the mean validation score `V/U`. It is computed exactly, with
>   `fractions.Fraction` and integer cross-multiplication for the ratio ordering,
>   so that `U >= V*` is a theorem rather than a tolerance question.
> - Trivial cases (no item fits) do occur in principle but not in this corpus:
>   0 of 17,700 instances have `V* = 0`. That branch is covered by synthetic
>   tests only, and the frequency is reported as zero rather than assumed away.
```

- [ ] **Step 2: 确认原文与第 8 节都还在**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git diff --stat EXPERIMENT_PLAN.md && grep -c "^## " EXPERIMENT_PLAN.md && grep -n "^## 8\. Experiment Matrix" EXPERIMENT_PLAN.md && sed -n '/^## 7\./,/^## 8\./p' EXPERIMENT_PLAN.md | grep -c "OR-Tools CP-SAT"
```

Expected: 只有插入（`--stat` 显示若干 insertions、**0 deletions**）；`^## ` 计数仍为 **10**；第 8 节仍在；第 7 节里 `OR-Tools CP-SAT` 仍出现 **1** 次（原文未删）。

- [ ] **Step 3: 提交**

```bash
cd "/c/work/学校/A.SPBU/Saint-peter-26/AI-2/project" && git add EXPERIMENT_PLAN.md && git commit -m "docs(plan): 第 7 节记录「一律用整数 DP」的实测依据与后果"
```

---

## 完成标准

- 全部 Task 提交完毕，`git status --short` 干净。
- 整套测试全绿且 **0 skipped**。
- `reference_results.csv` 在仓库根目录、已入库、行数 17,701（含表头）、`U < V*` 零行。
- `EXPERIMENT_PLAN.md` 第 7 节原文未删，修订说明在末尾。
- `experiment/data/**` 与既有 `tests/**` 一行未改（`git diff --stat main -- experiment/data tests` 只应显示新建文件）。

之后进入 `superpowers:subagent-driven-development` 的收尾流程：整份实现的最终审查 → `superpowers:finishing-a-development-branch`。
