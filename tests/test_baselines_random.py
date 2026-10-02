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
