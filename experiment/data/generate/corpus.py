"""cell 编排、配额、跨 split 去重与落盘。

本模块只做编排，不知道任一族的抽样公式。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment.data.generate.families import (
    capacity_for,
    compute_stats,
    generate_items,
)
from experiment.data.generate.hashing import canonical_hash
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    MAX_RESAMPLE_ATTEMPT,
    REPO_ROOT,
    SCHEMA_VERSION,
    alpha_token,
    all_cells,
)


def instance_id(split: str, family: str, n: int, alpha: float, index: int) -> str:
    """如 ``syn-train-uniform-n20-a25-i0007``。"""
    return f"syn-{split}-{family}-n{n}-{alpha_token(alpha)}-i{index:04d}"


def build_instance(
    split: str,
    family: str,
    n: int,
    alpha: float,
    index: int,
    attempt: int = 0,
) -> dict:
    """构造一个实例的完整记录。

    ``items`` 流产生权重与价值，``order`` 流独立产生物品排列；两条流互不
    影响，因此在 items 流中增减抽样不会改变已冻结的排列。
    """
    seed = derive_seed(split, family, n, alpha, index, attempt, "items")
    weights, values = generate_items(family, n, seed)
    capacity = capacity_for(alpha, weights)

    order_rng = np.random.default_rng(
        derive_seed(split, family, n, alpha, index, attempt, "order")
    )
    item_order = order_rng.permutation(n).tolist()

    return {
        "alpha": alpha,
        "canonical_hash": canonical_hash(n, capacity, weights, values),
        "capacity": capacity,
        "family": family,
        "instance_id": instance_id(split, family, n, alpha, index),
        "item_order": item_order,
        "n": n,
        "resample_attempt": attempt,
        "seed": seed,
        "source": "synthetic",
        "stats": compute_stats(weights, values),
        "values": values.tolist(),
        "weights": weights.tolist(),
    }


def build_cell(
    split: str,
    family: str,
    n: int,
    alpha: float,
    quota: int,
    seen: set[str],
    max_attempt: int = MAX_RESAMPLE_ATTEMPT,
) -> list[dict]:
    """构造一个 cell 的全部实例，并就地更新跨 split 的已用哈希集合。

    哈希冲突时递增 ``attempt`` 重派生，直到哈希未被占用。配额不因重抽改变：
    重抽只换实例内容，不减少条数。

    Raises:
        RuntimeError: ``attempt`` 超过 ``max_attempt`` 仍未取得唯一哈希。
            宁可报错也不死循环。
    """
    instances: list[dict] = []
    for index in range(quota):
        attempt = 0
        while True:
            instance = build_instance(split, family, n, alpha, index, attempt)
            if instance["canonical_hash"] not in seen:
                break
            attempt += 1
            if attempt > max_attempt:
                raise RuntimeError(
                    f"resample exceeded {max_attempt} attempts for "
                    f"{split}/{family}/n={n}/alpha={alpha}/index={index}"
                )
        seen.add(instance["canonical_hash"])
        instances.append(instance)
    return instances
