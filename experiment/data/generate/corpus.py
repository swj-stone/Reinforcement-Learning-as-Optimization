"""cell 编排、配额、跨 split 去重与落盘。

本模块只做编排，不知道任一族的抽样公式。
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
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
    ALPHAS,
    FAMILIES,
    GENERATION_DIR,
    MASTER_SEEDS,
    MAX_RESAMPLE_ATTEMPT,
    REPO_ROOT,
    SCHEMA_VERSION,
    SIZES,
    SPLITS,
    SYNTHETIC_QUOTA,
    alpha_token,
    all_cells,
    cell_filename,
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


def write_cell(path: Path, instances: list[dict]) -> str:
    """把实例写成 JSONL，返回内容的 SHA-256。

    统一的序列化约定：``sort_keys=True`` 保证键序稳定，LF 换行，末尾留一个
    换行符。三者共同保证同一份实例集合永远产出同样的字节。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(instance, sort_keys=True, ensure_ascii=False)
        for instance in instances
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def generator_fingerprint() -> str:
    """生成器代码与运行环境的合并哈希，用于事后诊断「输出为何变了」。

    必须同时纳入**运行环境**，而不只是 .py 文件：numpy 升级会改变抽样与浮点
    实现，CPython 升级会改变 dict 顺序与随机数实现。若只哈希源码，那么最
    容易被忽略、也最难排查的那类原因（「代码没动，结果却变了」）恰好是
    指纹唯一检测不到的。
    """
    digest = hashlib.sha256()
    sources = [
        Path(__file__),
        Path(__file__).with_name("families.py"),
        Path(__file__).with_name("hashing.py"),
        Path(__file__).with_name("seeds.py"),
        REPO_ROOT / "experiment" / "data" / "spec.py",
    ]
    for source in sorted(sources):
        digest.update(source.read_bytes())

    requirements = REPO_ROOT / "requirements.txt"
    if requirements.is_file():
        digest.update(requirements.read_bytes())

    digest.update(f"python={platform.python_version()}".encode("utf-8"))
    digest.update(f"numpy={np.__version__}".encode("utf-8"))
    return digest.hexdigest()


def run_generation(
    out_dir: Path = GENERATION_DIR,
    quota_override: dict[str, int] | None = None,
) -> dict:
    """生成全部 cell 与 manifest。

    遍历顺序是冻结语义：split -> family -> n -> alpha -> index。一个实例只
    可能与先于它的实例冲突，因此单遍生成即可判定，无需二次扫描。

    Args:
        out_dir: 输出根目录，默认 ``datasets/raw/generation``
        quota_override: 覆盖每 cell 配额，仅供测试用小样本
    """
    quotas = quota_override or SYNTHETIC_QUOTA
    seen: set[str] = set()
    cells: list[dict] = []
    total_resamples = 0
    max_attempt_seen = 0

    for split in SPLITS:
        quota = quotas[split]
        for family, n, alpha in all_cells():
            instances = build_cell(split, family, n, alpha, quota, seen)
            relative = Path(split) / cell_filename(family, n, alpha)
            digest = write_cell(out_dir / relative, instances)

            attempts = [i["resample_attempt"] for i in instances]
            total_resamples += sum(attempts)
            max_attempt_seen = max(max_attempt_seen, max(attempts))

            cells.append(
                {
                    "alpha": alpha,
                    "count": len(instances),
                    "family": family,
                    "n": n,
                    "path": relative.as_posix(),
                    "resample_attempts": sum(attempts),
                    "sha256": digest,
                    "split": split,
                }
            )

    manifest = {
        "cells": cells,
        "dedup": {
            "collisions_resampled": total_resamples,
            "max_resample_attempt": max_attempt_seen,
            "total_instances": len(seen),
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_fingerprint": generator_fingerprint(),
        "schema_version": SCHEMA_VERSION,
        "spec": {
            "alphas": list(ALPHAS),
            "families": list(FAMILIES),
            "master_seeds": dict(MASTER_SEEDS),
            "quota": dict(quotas),
            "sizes": list(SIZES),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
