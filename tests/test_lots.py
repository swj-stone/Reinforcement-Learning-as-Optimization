import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiment.data.collect.clean import apply_cleaning, build_lots, write_lots_csv
from experiment.data.spec import LOTS_COLUMNS, SOURCE_COLUMNS


@pytest.fixture
def lots() -> pd.DataFrame:
    rows = [
        (536365, "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        (536369, "22139", "RETROSPOT", 2, "2010-12-01 09:34:00", 1.25, None, "United Kingdom"),
    ]
    frame, _ = apply_cleaning(pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)))
    return build_lots(frame)


def test_lots_has_exactly_the_frozen_columns(lots):
    assert tuple(lots.columns) == LOTS_COLUMNS


def test_customer_and_descriptive_text_are_excluded(lots):
    """EXPERIMENT_PLAN.md 第 3.1 节：客户标识与描述文本不进模型特征。"""
    for column in ("CustomerID", "customer_id", "Description", "description", "Country"):
        assert column not in lots.columns


def test_lots_values(lots):
    assert lots["invoice_no"].tolist() == ["536365", "536369"]
    assert lots["stock_code"].tolist() == ["85123A", "22139"]
    assert lots["quantity_w"].tolist() == [6, 2]
    assert lots["value_pence"].tolist() == [1530, 250]


def test_invoice_date_is_iso_text(lots):
    assert lots["invoice_date"].tolist() == [
        "2010-12-01 08:26:00", "2010-12-01 09:34:00"
    ]


def test_source_row_id_is_preserved(lots):
    assert lots["source_row_id"].tolist() == [0, 1]


def test_write_lots_csv_creates_lf_terminated_file(tmp_path, lots):
    path = tmp_path / "lots_clean.csv"
    write_lots_csv(path, lots)
    raw = path.read_bytes()
    assert b"\r" not in raw, "必须是 LF 换行，否则字节级不可复现"
    assert raw.endswith(b"\n")


def test_write_lots_csv_roundtrips(tmp_path, lots):
    path = tmp_path / "lots_clean.csv"
    write_lots_csv(path, lots)
    back = pd.read_csv(path, dtype={"invoice_no": str, "stock_code": str})
    assert tuple(back.columns) == LOTS_COLUMNS
    assert back["value_pence"].tolist() == [1530, 250]
    assert back["invoice_no"].tolist() == ["536365", "536369"]
    assert back["invoice_date"].tolist() == lots["invoice_date"].tolist()


def test_write_lots_csv_is_byte_identical_on_rerun(tmp_path, lots):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    write_lots_csv(a, lots)
    write_lots_csv(b, lots)
    assert a.read_bytes() == b.read_bytes()


def test_write_lots_csv_creates_parent_dirs(tmp_path, lots):
    path = tmp_path / "nested" / "deeper" / "lots_clean.csv"
    write_lots_csv(path, lots)
    assert path.is_file()
