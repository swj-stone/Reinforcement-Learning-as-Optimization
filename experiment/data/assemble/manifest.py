"""协议产物：``data_manifest.json`` 与 ``experiment_config.yaml``。

两者都放在**仓库根目录并纳入版本库**：``datasets/`` 整体被 gitignore，把审计
凭据放在那里等于换一台机器就无从证明数据未被调换。

本模块不读语料、不碰数据装配，只做两件事：从上游产物里**摘出**审计字段
（:func:`upstream_provenance`），以及把调用方给的事实**排版**成两个文件。
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from experiment.data.spec import (
    ALPHAS,
    CLEANING_RULES,
    E3_NAME,
    E3_REAL_PER_ALPHA,
    E3_SIZE,
    E3_SYNTHETIC_PER_FAMILY,
    E4_COUNT_PER_ALPHA,
    E4_NAME,
    E4_SEED_TOKEN,
    E4_SIZE,
    E4_SKEW_HEAVY_DENSITY_RANGE,
    E4_SKEW_HEAVY_WEIGHT_RANGE,
    E4_SKEW_K,
    LOTS_COLUMNS,
    REAL_QUOTA,
    REAL_SEED_TOKEN,
    REPO_ROOT,
    SCHEMA_VERSION,
    SKEW_BASE_DENSITY_RANGE,
    SKEW_BASE_WEIGHT_RANGE,
    SKEW_FRACTION,
    SKEW_HEAVY_DENSITY_RANGE,
    SKEW_HEAVY_WEIGHT_RANGE,
    SIZES,
    SPLIT_DATE_FRACTIONS,
    STRESS_MASTER_SEEDS,
    STRESS_SIZE_STREAM,
    STRESS_TAIL_STREAM,
    SYNTHETIC_QUOTA,
    TAIL_QUANTILE,
)

#: 训练预算，逐项转录自 EXPERIMENT_PLAN.md §6 的表格与正文。
#: 刻意原样保留计划里的数值：后续阶段应当读 YAML 而不是回头读 Markdown。
_BUDGETS: dict[str, object] = {
    "batch_size": 64,
    "epsilon": {"final": 0.05, "linear_decay_transitions": 150000, "initial": 1.0},
    "grad_clip_global_norm": 10,
    "learning_rate": 0.0003,
    "loss": "huber",
    "optimizer": "adam",
    "pilot_transitions": 50000,
    "replay_capacity": 100000,
    "replay_source_balance": {"real": 32, "synthetic": 32},
    "target_copy_transitions": 2000,
    "training_seeds": [0, 1, 2],
    "training_transitions": 300000,
    "update_every_transitions": 4,
    "validation_every_transitions": 25000,
    "warmup_transitions": 5000,
}

#: 评估口径，转录自 EXPERIMENT_PLAN.md §5（三种方法的定义）与 §7（参考解与指标）。
_EVALUATION: dict[str, object] = {
    "aggregate": "one-half synthetic mean + one-half real mean",
    "bootstrap": {
        "paired": True,
        "resamples": 1000,
        "unit": "whole source dates for real; whole synthetic instances within cells",
    },
    "evaluation_seed": 6001,
    "greedy": "sort by descending v/w; ties by lower w then stable item ID",
    "metrics": [
        "feasibility",
        "exact_optimality_gap",
        "bound_based_gap",
        "relative_improvement",
        "speed",
        "learning_cost",
        "robustness",
    ],
    "methods": ["random", "greedy", "dqn"],
    "random_trials_per_instance": 10,
    "reference": {
        "cp_sat_limits": {"seed": 0, "time_seconds": 10, "workers": 1},
        "general": "OR-Tools CP-SAT with binary item variables",
        "note": "an evaluation service, not a fourth competing method",
        "small_cases": "integer dynamic programming",
    },
    "rl": "argmax feasible Q-value with exploration disabled; exact ties resolved by skipping",
    "trivial_case": "empty solution when no item fits; excluded from percentage denominators",
}


def _file_fact(path: Path) -> dict:
    """一个文件的路径与摘要；文件不存在时摘要为 ``None``（不假装有）。"""
    path = Path(path)
    if not path.is_file():
        return {"bytes": None, "path": path.as_posix(), "sha256": None}
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def upstream_provenance(lots_path: Path, generation_dir: Path) -> dict:
    """从上游两个阶段的产物里**摘出**审计字段。

    刻意读上游 JSON 的**内容**而不是把数值复制一份：复制来的常量会随上游
    变化而说谎，而本项目的病灶正是「常量对常量」。

    上游布局由前两个阶段固定：采集报告是批次表的同目录兄弟
    ``collect_report.json``，生成清单是 ``generation_dir/manifest.json``。
    """
    lots_path = Path(lots_path)
    report_path = lots_path.parent / "collect_report.json"
    generation_manifest = Path(generation_dir) / "manifest.json"

    report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    )
    generation = (
        json.loads(generation_manifest.read_text(encoding="utf-8"))
        if generation_manifest.is_file()
        else {}
    )
    source = report.get("source", {})

    return {
        "cleaning_rules": list(CLEANING_RULES),
        "collect_report": _file_fact(report_path),
        "generation_manifest": {
            **_file_fact(generation_manifest),
            "generator_fingerprint": generation.get("generator_fingerprint"),
        },
        "lots_clean": {
            **_file_fact(lots_path),
            "columns": list(LOTS_COLUMNS),
            "rows": report.get("kept_rows"),
        },
        "units": {"value": "value_pence", "weight": "quantity_w"},
        "workbook": {
            "bytes": source.get("xlsx_bytes"),
            "sha256": source.get("xlsx_sha256"),
            "url": source.get("url"),
        },
    }


def _boundaries(lot_index) -> dict[str, int]:
    """两个日期边界，以**整数天数**记录。

    写成两个独立数字而不是再存一份 0.85，是为了让校验者不必重做浮点下取整。
    """
    if lot_index is None:
        return {"train_end_exclusive": 0, "validation_end_exclusive": 0}
    train = len(lot_index.date_splits["train"])
    return {
        "train_end_exclusive": train,
        "validation_end_exclusive": train + len(lot_index.date_splits["validation"]),
    }


def _splits(lot_index) -> dict[str, dict]:
    """每个 split 的实测规模。

    实测占比**不必等于**日期占比（§4 明说），所以要存下来供人核对，
    而不是断言相等。
    """
    if lot_index is None:
        return {}
    total_lots = len(lot_index.lots)
    total_invoices = len(lot_index.first_dates)
    out: dict[str, dict] = {}
    for split, days in lot_index.date_splits.items():
        day_set = set(days)
        lots = sum(len(pool) for pool in lot_index.pools[split].values())
        invoices = sum(1 for day in lot_index.first_dates.values() if day in day_set)
        out[split] = {
            "date_from": days[0] if days else None,
            "date_to": days[-1] if days else None,
            "dates": len(days),
            "invoices": invoices,
            "invoices_share": invoices / total_invoices if total_invoices else 0.0,
            "lots": lots,
            "lots_share": lots / total_lots if total_lots else 0.0,
        }
    return out


def _eligible_dates(lot_index, sizes: Sequence[int]) -> dict[str, dict[str, int]]:
    """每个 split 中批次数 ≥ n 的日期个数，按 n 分列。"""
    if lot_index is None:
        return {}
    return {
        split: {str(n): len(lot_index.eligible(split, n)) for n in sizes}
        for split in lot_index.date_splits
    }


def build_manifest(
    cells: Sequence[dict],
    counts: Mapping[str, dict],
    dedup: Mapping[str, int],
    files: Sequence[dict],
    generated_at: str,
    lot_index,
    out_dir: Path,
    provenance: Mapping,
    schema_version: int,
    stress: Mapping[str, dict],
    tail: Mapping[str, object],
) -> dict:
    """汇总审计 manifest。每一项都能在事后被独立复核。"""
    return {
        "cells": list(cells),
        "counts": {split: dict(value) for split, value in counts.items()},
        "dedup": dict(dedup),
        "files": list(files),
        "generated_at": generated_at,
        "output_dir": (
            Path(out_dir).relative_to(REPO_ROOT).as_posix()
            if Path(out_dir).is_relative_to(REPO_ROOT)
            else str(out_dir)
        ),
        "provenance": dict(provenance),
        "schema_version": schema_version,
        "split_policy": {
            "basis": "invoice earliest transaction calendar date",
            "boundaries": _boundaries(lot_index),
            "eligible_dates": _eligible_dates(lot_index, list(SIZES) + [E3_SIZE]),
            "fractions": list(SPLIT_DATE_FRACTIONS),
        },
        "splits": _splits(lot_index),
        "streams": {
            "core": {"seed_key": {"family": REAL_SEED_TOKEN, "split": "<per-split name>"}},
            "size": {
                "dataset": E3_NAME,
                "declared_master_seed": STRESS_MASTER_SEEDS["size"],
                "seed_key": {
                    "family": f"<family> | {REAL_SEED_TOKEN}",
                    "split": STRESS_SIZE_STREAM,
                },
            },
            "tail": {
                "dataset": E4_NAME,
                "declared_master_seed": STRESS_MASTER_SEEDS["tail"],
                "seed_key": {"family": E4_SEED_TOKEN, "split": STRESS_TAIL_STREAM},
            },
        },
        "stress": {name: dict(value) for name, value in stress.items()},
        "tail_thresholds": dict(tail),
    }


def write_manifest(path: Path, manifest: Mapping) -> None:
    """写出 manifest：缩进 2、键排序、LF、无 BOM、末尾一个换行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def parse_packages(requirements: Path) -> dict[str, str]:
    """从 ``requirements.txt`` 解析固定版本。

    刻意解析而不是硬编码：硬编码就会在依赖升级后悄悄说谎。

    Raises:
        ValueError: 出现没有 ``==`` 的依赖行。半钉住的依赖就是复现性漏洞。
    """
    packages: dict[str, str] = {}
    for raw in Path(requirements).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep or not version.strip():
            raise ValueError(
                f"requirements.txt 中的 {line!r} 没有钉住版本；"
                "本项目的复现性依赖全部依赖项精确固定"
            )
        packages[name.strip()] = version.strip()
    return packages


def build_config(
    out_dir: Path,
    generated_at: str,
    requirements: Path | None = None,
) -> dict:
    """``experiment_config.yaml`` 的内容：转录计划的公式、配额、预算与评估设置。

    生成器与评估段是**转录**而不是重算：让后续阶段读 YAML 而不是回头读
    Markdown。``packages`` 与 ``runtime`` 是仅有的两项从磁盘实测的，
    因为硬编码的版本会在升级后说谎。
    """
    requirements = (
        REPO_ROOT / "requirements.txt" if requirements is None else Path(requirements)
    )
    return {
        "budgets": dict(_BUDGETS),
        "cap_ratios": list(ALPHAS),
        "evaluation": dict(_EVALUATION),
        "generated_at": generated_at,
        "generators": {
            "correlated": {
                "e": "U(-10,10)",
                "value": "max(1, w + e)",
                "weight": "U(1,100)",
            },
            "normal": {
                "clip": [1, 100],
                "distribution": "Normal(50, 15^2)",
                "mean": 50,
                "rounding": "half-up to nearest integer",
                "sd": 15,
            },
            "skewed": {
                "base_density": list(SKEW_BASE_DENSITY_RANGE),
                "base_weight": list(SKEW_BASE_WEIGHT_RANGE),
                "heavy_density": list(SKEW_HEAVY_DENSITY_RANGE),
                "heavy_weight": list(SKEW_HEAVY_WEIGHT_RANGE),
                "k": f"max(1, floor({SKEW_FRACTION} * n + 0.5))",
                "value": "w * d",
            },
            "skewed_e4": {
                "base_density": list(SKEW_BASE_DENSITY_RANGE),
                "base_weight": list(SKEW_BASE_WEIGHT_RANGE),
                "heavy_density": list(E4_SKEW_HEAVY_DENSITY_RANGE),
                "heavy_weight": list(E4_SKEW_HEAVY_WEIGHT_RANGE),
                "k": E4_SKEW_K,
                "note": "E4 variant; the frozen 'skewed' family is unchanged",
                "value": "w * d",
            },
            "uniform": {"value": "U(1,100)", "weight": "U(1,100)"},
        },
        "packages": parse_packages(requirements),
        "quotas": {"real": dict(REAL_QUOTA), "synthetic": dict(SYNTHETIC_QUOTA)},
        "real": {"split_fractions": list(SPLIT_DATE_FRACTIONS)},
        "runtime": {"python": platform.python_version()},
        "schema_version": SCHEMA_VERSION,
        "sizes": list(SIZES),
        "stress": {
            E3_NAME: {
                "n": E3_SIZE,
                "real_per_alpha": E3_REAL_PER_ALPHA,
                "synthetic_per_family": E3_SYNTHETIC_PER_FAMILY,
            },
            E4_NAME: {
                "families": [E4_SEED_TOKEN],
                "n": E4_SIZE,
                "per_alpha": E4_COUNT_PER_ALPHA,
                "quantile_for_e4b": TAIL_QUANTILE,
            },
        },
    }


def write_config(path: Path, config: Mapping) -> None:
    """写出 YAML：键排序、不转义中文、块状风格、LF、无 BOM。

    ``sort_keys=True`` 保证字典键序稳定；``allow_unicode=True`` 避免中文被写成
    ``\\uXXXX`` 而让文件不可读。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        dict(config), allow_unicode=True, default_flow_style=False, sort_keys=True
    )
    path.write_bytes(payload.encode("utf-8"))
