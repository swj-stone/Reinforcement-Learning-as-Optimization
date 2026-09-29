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
