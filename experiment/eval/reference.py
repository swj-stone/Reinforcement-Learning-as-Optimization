"""精确参考解：整数 DP、分数背包上界、重建与自校验。

依据 `EXPERIMENT_PLAN.md` 第 7 节与设计文档第 4、5、6 节。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Mapping, Sequence

import numpy as np

#: 把 U 写成十进制字符串时的放大倍数（6 位小数）。
_UB_SCALE = 1_000_000

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


def ratio_order(weights: Sequence[int], values: Sequence[int]) -> list[int]:
    """按 ``v_i / w_i`` 降序返回物品索引，**用精确整数叉乘**。

    走浮点除法是错的：两个不同的有理比值可能舍入到同一个 double，比较结果就
    不再反映真实次序，贪心于是可能给出**小于**真分数最优的 U，进而 ``U < V*``
    ——上界性质被破坏，而 `EXPERIMENT_PLAN.md` 第 6 节的 ``V/U`` 随之失真。

    次级键依次取更小的 ``w``、再取原索引升序，使完整排序确定，U 不依赖
    ``sort`` 的稳定性。

    **公开是因为它同时是 Greedy 基线的次序**（`EXPERIMENT_PLAN.md` 第 5 节）：
    分数背包的贪心与经典 0/1 贪心按同一个比值序扫描。两份实现会在某次修订后
    分歧，而分歧的症状是「上界与贪心用了不同的次序」——一个不报错、不崩溃、
    只在数字上体现的偏差。见
    `docs/superpowers/specs/2026-09-30-baselines-design.md` §6。
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
    order = ratio_order(weights, values)
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

    **前置条件：``upper_bound >= 0``。** ``-(-n // d)`` 只在非负时才有「向上
    取整」的含义；负数会得到一个静默的错值（实测
    ``format_upper_bound(Fraction(-1, 3))`` 返回 ``-1.666667``，而 ``-1/3`` 的
    六位小数上界是 ``-0.333333``）。本项目里 ``U >= 0`` 由构造保证（权重与价值
    都非负），所以这里不设运行时断言——那是为一个不可能出现的状态写守卫；改为
    把前置条件写在这里，让调用方看得见。
    """
    scaled = upper_bound * _UB_SCALE
    whole = -(-scaled.numerator // scaled.denominator)  # 向上取整的整数除法
    return f"{whole // _UB_SCALE}.{whole % _UB_SCALE:06d}"


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
