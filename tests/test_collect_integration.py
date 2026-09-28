"""真实工作簿的端到端校验。

未下载工作簿时整体跳过，因此不联网也能跑绿。先执行
``python -m experiment.data.collect.cli`` 再跑本文件即可得到真实校验。
"""

import json

import pytest

from experiment.data.collect.download import sha256_file
from experiment.data.collect.run import read_workbook, run_collection
from experiment.data.spec import (
    CLEANING_RULES,
    LOTS_COLUMNS,
    REAL_DIR,
    SOURCE_COLUMNS,
    UCI_XLSX_NAME,
)

WORKBOOK = REAL_DIR / UCI_XLSX_NAME

pytestmark = pytest.mark.skipif(
    not WORKBOOK.is_file(), reason="真实工作簿尚未下载，先运行 CLI"
)


def test_read_workbook_has_the_source_columns():
    frame = read_workbook(WORKBOOK)
    assert tuple(frame.columns) == SOURCE_COLUMNS
    assert len(frame) > 500_000


def test_run_collection_writes_the_outputs_of_a_supplied_workbook(tmp_path):
    """传了 workbook 就不下载，因此只落盘清洗产物与审计报告。

    四个文件齐全的形态属于下载路径：那时 ensure_workbook 会把 zip 与 xlsx
    一并写进同一个 out_dir（Task 8 用 CLI 验收这一点）。本文件刻意走离线
    路径，所以不该要求那两个下载件——build_report 里 zip_sha256 允许为
    None，正是为这条路径留的。
    """
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    for name in ("lots_clean.csv", "collect_report.json"):
        assert (tmp_path / name).is_file(), name
    assert report["kept_rows"] > 0


def test_report_identity_holds(tmp_path):
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    dropped = sum(report["cleaning"]["dropped_by_reason"].values())
    assert report["raw_rows"] == report["kept_rows"] + dropped


def test_report_rule_keys_are_complete(tmp_path):
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    assert tuple(report["cleaning"]["dropped_by_reason"]) == CLEANING_RULES


def test_report_records_source_provenance(tmp_path):
    """报告里的来源信息必须是对磁盘上这份文件**实测**的结果。

    只断言「长度 64」是恒真的：任何 64 字符串都能通过，常量回填也能通过。
    这里改为与本地文件实测值比对，报告一旦记录虚构的字节数或摘要就会失败。
    """
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    assert report["source"]["url"].startswith("https://archive.ics.uci.edu/")
    assert report["source"]["xlsx_bytes"] == WORKBOOK.stat().st_size
    assert report["source"]["xlsx_sha256"] == sha256_file(WORKBOOK)
    assert len(report["source"]["lots_sha256"]) == 64


def test_kept_lots_survive_the_zero_pence_rule(tmp_path):
    """幸存批次的价值必须全部为正——否则规则 9 没生效。"""
    import pandas as pd
    report = run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    lots = pd.read_csv(tmp_path / "lots_clean.csv")
    assert (lots["value_pence"] > 0).all()
    assert (lots["quantity_w"] > 0).all()


def test_kept_lots_have_valid_stock_codes(tmp_path):
    import pandas as pd
    from experiment.data.spec import STOCK_CODE_RE
    run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    lots = pd.read_csv(tmp_path / "lots_clean.csv", dtype={"stock_code": str})
    assert lots["stock_code"].str.match(STOCK_CODE_RE).all()


def test_kept_lots_have_no_cancellation_invoices(tmp_path):
    import pandas as pd
    run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    lots = pd.read_csv(tmp_path / "lots_clean.csv", dtype={"invoice_no": str})
    assert not lots["invoice_no"].str.upper().str.startswith("C").any()


def test_kept_lots_columns_are_the_frozen_set(tmp_path):
    import pandas as pd
    run_collection(out_dir=tmp_path, workbook=WORKBOOK)
    header = (tmp_path / "lots_clean.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(LOTS_COLUMNS)
