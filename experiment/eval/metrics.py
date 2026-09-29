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

    三类**独立计数、可以叠加**：同一个越界索引出现两次会计入 1 条越界 + 1 条
    重复，一个超容量的选择若同时含重复物品则再 +1。这不是 bug——它确实是三个
    问题——但它是个容易被「修掉」的语义，所以由测试显式钉住。
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
