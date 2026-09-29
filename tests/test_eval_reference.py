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
