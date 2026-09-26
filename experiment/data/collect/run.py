"""采集端到端编排：下载 -> 读取 -> 清洗 -> 落盘 -> 写报告。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiment.data.collect.clean import apply_cleaning, build_lots, write_lots_csv
from experiment.data.collect.download import ensure_workbook, sha256_file
from experiment.data.spec import (
    CLEANING_RULES,
    LOTS_COLUMNS,
    REAL_DIR,
    SCHEMA_VERSION,
    SOURCE_COLUMNS,
    UCI_SHEET_NAME,
    UCI_URL,
    UCI_XLSX_BYTES,
    UCI_XLSX_NAME,
    UCI_ZIP_NAME,
)


def read_workbook(path: Path) -> pd.DataFrame:
    """读取工作簿，并校验列集合与规格一致。

    列名或列序变了就说明上游改了结构，此时继续往下算只会产出看似正常的错数据。
    """
    frame = pd.read_excel(path, sheet_name=UCI_SHEET_NAME, engine="openpyxl")
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"工作簿列与规格不符：期望 {SOURCE_COLUMNS}，实际 {tuple(frame.columns)}"
        )
    return frame


def build_report(
    raw_rows: int,
    cleaning: dict,
    zip_path: Path,
    xlsx_path: Path,
    lots_path: Path,
) -> dict:
    """汇总审计报告。每一项都可在事后独立复核。"""
    dropped = cleaning["dropped_by_reason"]
    return {
        "cleaning": {
            "dropped_by_reason": dropped,
            "identity_holds": raw_rows
            == cleaning["kept_rows"] + sum(dropped.values()),
            "rules": list(CLEANING_RULES),
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kept_rows": cleaning["kept_rows"],
        "lots_columns": list(LOTS_COLUMNS),
        "raw_rows": raw_rows,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "url": UCI_URL,
            "xlsx_bytes": UCI_XLSX_BYTES,
            "xlsx_sha256": sha256_file(xlsx_path),
            "zip_sha256": sha256_file(zip_path) if zip_path.is_file() else None,
            "lots_sha256": sha256_file(lots_path),
        },
    }


def run_collection(
    out_dir: Path = REAL_DIR,
    workbook: Path | None = None,
    force_download: bool = False,
) -> dict:
    """执行完整采集流程，返回审计报告。

    Args:
        out_dir: 输出目录，默认 ``datasets/raw/real``
        workbook: 已有的工作簿路径；为 ``None`` 时才走下载
        force_download: 忽略已存在的工作簿，强制重新下载
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if workbook is None:
        zip_path, xlsx_path = ensure_workbook(out_dir, force=force_download)
    else:
        xlsx_path = Path(workbook)
        zip_path = out_dir / UCI_ZIP_NAME

    raw = read_workbook(xlsx_path)
    cleaned, cleaning = apply_cleaning(raw)
    lots = build_lots(cleaned)

    lots_path = out_dir / "lots_clean.csv"
    write_lots_csv(lots_path, lots)

    report = build_report(len(raw), cleaning, zip_path, xlsx_path, lots_path)
    (out_dir / "collect_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
