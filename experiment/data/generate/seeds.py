"""内容寻址的种子派生。

设计取舍见设计文档第 5 节：用哈希而非位置式的 ``SeedSequence.spawn()``，
使种子与调用顺序无关，语料冻结后不会因为别处新增维度而整体平移。
"""

from __future__ import annotations

import hashlib

from experiment.data.spec import alpha_key

#: 相互隔离的抽样流。``items`` 抽权重与价值/参数，``order`` 抽物品排列。
STREAMS: tuple[str, ...] = ("items", "order")

_MASK64 = (1 << 64) - 1


def derive_seed(
    split: str,
    family: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
    stream: str = "items",
) -> int:
    """由实例的完整身份派生一个 64 位种子。

    Args:
        split: ``train`` / ``validation`` / ``test``
        family: 四个族之一
        n: 物品数
        alpha: 容量比
        index: 该 cell 内的实例序号
        attempt: 重抽次数，跨 split 哈希冲突时递增
        stream: ``items`` 或 ``order``

    Returns:
        ``[0, 2**64)`` 内的整数，可直接交给 ``numpy.random.default_rng``。
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown stream: {stream!r}; expected one of {STREAMS}")

    key = "|".join(
        (
            split,
            family,
            str(n),
            alpha_key(alpha),
            str(index),
            str(attempt),
            stream,
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK64
