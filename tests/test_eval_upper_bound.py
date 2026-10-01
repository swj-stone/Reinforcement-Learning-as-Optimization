"""分数背包上界 U 与它的存储口径（设计 §5、§9）。"""

from fractions import Fraction

import pytest

from experiment.eval.reference import (
    ratio_order,
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
    assert ratio_order([1, 2], [2, 4]) == [0, 1]              # 同比值：w 小者先
    assert ratio_order([2, 1], [4, 2]) == [1, 0]              # 打乱输入，仍取 w 小的那件
    assert ratio_order([5, 5, 5], [7, 7, 7]) == [0, 1, 2]     # 全同：按原索引
    assert ratio_order([1, 3, 2], [3, 1, 2]) == [0, 2, 1]     # 比值 3、1/3、1


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
