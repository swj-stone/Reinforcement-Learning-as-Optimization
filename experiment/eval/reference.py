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
