"""协议产物：``data_manifest.json`` 与 ``experiment_config.yaml``。

本模块不读语料、不碰数据装配，只做两件事：从上游产物里**摘出**审计字段
（:func:`upstream_provenance`），以及把调用方给的事实**排版**成两个文件。

Task 5 交付 :func:`upstream_provenance` 与两个写出函数；manifest 的派生段
（``splits`` / ``split_policy`` / ``streams`` / ``output_dir``）与完整的 config
由 Task 6 补齐——那时装配流程已经跑通，可以拿真实产物校对。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from experiment.data.spec import CLEANING_RULES, LOTS_COLUMNS


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
    """把调用方给的事实原样带出。派生段由 Task 6 补齐。"""
    return {
        "cells": list(cells),
        "counts": {split: dict(value) for split, value in counts.items()},
        "dedup": dict(dedup),
        "files": list(files),
        "generated_at": generated_at,
        "provenance": dict(provenance),
        "schema_version": schema_version,
        "stress": {name: dict(value) for name, value in stress.items()},
        "tail_thresholds": dict(tail),
    }


def write_manifest(path: Path, manifest: Mapping) -> None:
    """写出 manifest：缩进 2、键排序、LF、无 BOM、末尾一个换行。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def build_config(out_dir: Path, generated_at: str) -> dict:
    """Task 6 补齐生成器、配额、预算与评估段。"""
    return {"generated_at": generated_at, "schema_version": 1}


def write_config(path: Path, config: Mapping) -> None:
    """写出 YAML：键排序、不转义中文、块状风格、LF、无 BOM。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        dict(config), allow_unicode=True, default_flow_style=False, sort_keys=True
    )
    path.write_bytes(payload.encode("utf-8"))
