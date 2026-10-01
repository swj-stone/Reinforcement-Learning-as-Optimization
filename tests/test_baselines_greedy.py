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
