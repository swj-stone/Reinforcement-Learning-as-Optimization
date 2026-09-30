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

#: 三个实例把三类违例**分开**：容量充裕的那个只可能踩「重复」与「越界」，
#: 容量吃紧的那个让「超容量」单独成立，``WRAP`` 则专门区分求和守卫。
#:
#: 初稿只用一个 ``capacity=5, weights=[3,3,3]`` 的实例，六条断言里错了三条——
#: 手算时忘了超容量会与另两类**叠加**：``[0,1]`` 的重量是 6 > 5，``[0,0]`` 是
#: 重复 1 + 超容量 1 = 2，``[0,9,0,0]`` 是越界 1 + 重复 2 + 超容量 1 = 4。
#: 拆成两个实例后，每条期望值都只有一个来源。
TIGHT = {"capacity": 5, "values": [1, 1, 1], "weights": [3, 3, 3]}
LOOSE = {"capacity": 100, "values": [1, 1, 1], "weights": [3, 3, 3]}
#: 最后一件的重量（3）**大于**容量（2）：只有「负索引绕回末尾」才会把总和推过
#: 容量，因此这是唯一能区分 ``0 <= i < n`` 与 ``i < n`` 两版求和守卫的构造。
WRAP = {"capacity": 2, "values": [1, 1, 1], "weights": [1, 1, 3]}


def test_feasibility_counts_each_kind_of_violation_separately():
    """返回的是**条数**而不是布尔值，三类问题才不会塌缩成一个 False。"""
    # 容量充裕：只测「重复」与「越界」
    assert feasibility_violations(LOOSE, [0, 1, 2]) == 0
    assert feasibility_violations(LOOSE, []) == 0
    assert feasibility_violations(LOOSE, [0, 0]) == 1        # 重复选择
    assert feasibility_violations(LOOSE, [0, 7]) == 1        # 选了不存在的物品
    assert feasibility_violations(LOOSE, [7, 8]) == 2        # 两次越界
    assert feasibility_violations(LOOSE, [0, 9, 0, 0]) == 3  # 越界 1 + 重复 2

    # 容量吃紧：超容量各算一条，并会与上面两类叠加
    assert feasibility_violations(TIGHT, [0, 1]) == 1        # 6 > 5
    assert feasibility_violations(TIGHT, [0, 1, 2]) == 1     # 9 > 5
    assert feasibility_violations(TIGHT, [0, 0]) == 2        # 重复 1 + 超容量 1

    # 同一个越界索引重复出现会**同时**计入「越界」与「重复」——它确实是两个
    # 问题。这里显式钉住该语义，免得日后被当成 bug「修掉」。
    assert feasibility_violations(LOOSE, [7, 7]) == 3

    # 负索引**不会**绕回末尾去参与容量求和。``0 <= i < n`` 这个守卫做两件事：
    # 挡住 IndexError，以及挡住 ``weights[-1]`` 被悄悄计入。前者由上面的
    # ``[7, 8]`` 覆盖；后者**只有本实例能覆盖**——探针实测：上面十条断言在两版
    # 守卫下取值完全相同，也就是说只写它们，守卫退化成 ``i < n`` 也照样全绿，
    # 而那时 ``selected_indices=[-1]`` 会把最后一件的重量算进容量。
    assert feasibility_violations(WRAP, [-1]) == 1  # 越界 1；权重不计入


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
