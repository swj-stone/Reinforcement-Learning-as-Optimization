"""真实实例构造：从按日候选池里无放回抽样成 0/1 背包实例。

物品 = 一条批次（不可分割的履约批次），资源 = ``quantity_w``，
价值 = ``value_pence``，容量 = ``floor(alpha * sum(w))``。
依据 EXPERIMENT_PLAN.md 第 3.1 节与第 4 节。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from experiment.data.assemble.lots import Lot, LotIndex
from experiment.data.generate.families import capacity_for, compute_stats
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import MAX_RESAMPLE_ATTEMPT, REAL_SEED_TOKEN, alpha_token


def real_instance_id(name: str, n: int, alpha: float, index: int) -> str:
    """如 ``real-train-n20-a25-i0007``、``real-e3_size-n500-a75-i0000``。

    ``name`` 是落盘数据集名（核心集用 split 本身，E3 用 ``e3_size``），
    与种子键里的 split 槽位是两回事——应力集里二者不同。
    """
    return f"real-{name}-n{n}-{alpha_token(alpha)}-i{index:04d}"


def sample_lots(
    lot_index: LotIndex,
    date_split: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
    seed_stream: str | None = None,
) -> tuple[str, list[Lot]]:
    """选出一个日期与该日池内的 ``n`` 个批次。

    **rng 调用次序是冻结语义**：先均匀取一天，再在该日池内无放回抽 ``n`` 个。
    调换这两步会改变全部真实实例的内容。

    Args:
        date_split: 从哪个 split 的日期池里抽（决定候选池）
        seed_stream: 种子键的 split 槽位；``None`` 表示与 ``date_split`` 同名。
            E3 真实用它把种子挪到保留流上，同时仍从 test 的日期池里抽。
    """
    stream_key = date_split if seed_stream is None else seed_stream
    days = lot_index.eligible(date_split, n)
    if not days:
        raise ValueError(
            f"{date_split} 中没有任何日期能提供 n={n} 个批次；"
            "第 4 节禁止在实例内复制行凑数"
        )
    rng = np.random.default_rng(
        derive_seed(stream_key, REAL_SEED_TOKEN, n, alpha, index, attempt, "items")
    )
    day = days[int(rng.integers(len(days)))]
    pool = lot_index.pools[date_split][day]
    picked = rng.choice(len(pool), size=n, replace=False)
    return day, [pool[int(position)] for position in picked]


def build_real_instance(
    lot_index: LotIndex,
    name: str,
    date_split: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
    seed_stream: str | None = None,
) -> dict:
    """构造一个真实实例的完整记录。

    ``items`` 流决定抽哪些批次与它们的顺序，``order`` 流独立决定物品排列；
    两条流互不影响，与合成实例的约定完全一致。
    """
    stream_key = date_split if seed_stream is None else seed_stream
    day, lots = sample_lots(lot_index, date_split, n, alpha, index, attempt, seed_stream)

    weights = np.array([lot.quantity_w for lot in lots], dtype=np.int64)
    values = np.array([lot.value_pence for lot in lots], dtype=np.int64)
    capacity = capacity_for(alpha, weights)

    order_rng = np.random.default_rng(
        derive_seed(stream_key, REAL_SEED_TOKEN, n, alpha, index, attempt, "order")
    )

    return {
        "alpha": alpha,
        "canonical_hash": canonical_hash(n, capacity, weights, values),
        "capacity": capacity,
        "family": None,
        "instance_id": real_instance_id(name, n, alpha, index),
        "invoice_nos": [lot.invoice_no for lot in lots],
        "item_order": order_rng.permutation(n).tolist(),
        "n": n,
        "resample_attempt": attempt,
        "seed": derive_seed(stream_key, REAL_SEED_TOKEN, n, alpha, index, attempt, "items"),
        "source": "real",
        "source_date": day,
        "source_row_ids": [lot.source_row_id for lot in lots],
        "stats": compute_stats(weights, values),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }


def fill_cell(
    build: Callable[[int], dict],
    quota: int,
    seen: set[str],
    max_attempt: int = MAX_RESAMPLE_ATTEMPT,
) -> list[dict]:
    """反复调用 ``build(attempt)``，直到取满 ``quota`` 条哈希互不相同的实例。

    配额不因重抽改变：重抽只换实例内容，不减少条数。``seen`` 就地更新，
    因此它可以是跨 cell、跨 split 的全局集合。

    Raises:
        RuntimeError: ``attempt`` 超过 ``max_attempt`` 仍未取得唯一哈希。
    """
    instances: list[dict] = []
    for _ in range(quota):
        attempt = 0
        while True:
            instance = build(attempt)
            if instance["canonical_hash"] not in seen:
                break
            attempt += 1
            if attempt > max_attempt:
                raise RuntimeError(
                    f"resample exceeded {max_attempt} attempts for {instance['instance_id']}"
                )
        seen.add(instance["canonical_hash"])
        instances.append(instance)
    return instances


def build_real_cell(
    lot_index: LotIndex,
    name: str,
    date_split: str,
    n: int,
    alpha: float,
    quota: int,
    seen: set[str],
    max_attempt: int = MAX_RESAMPLE_ATTEMPT,
    seed_stream: str | None = None,
) -> list[dict]:
    """构造一个真实 cell 的全部实例。

    同一下标的重抽只递增 ``attempt``——与 ``generate.corpus.build_cell`` 的
    语义一致：重抽改内容，不改身份。
    """
    instances: list[dict] = []
    for index in range(quota):
        instances.extend(
            fill_cell(
                lambda attempt, index=index: build_real_instance(
                    lot_index, name, date_split, n, alpha, index, attempt, seed_stream
                ),
                quota=1,
                seen=seen,
                max_attempt=max_attempt,
            )
        )
    return instances
