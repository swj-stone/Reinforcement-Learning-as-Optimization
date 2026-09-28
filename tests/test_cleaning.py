"""八条规则的逐条归因测试。

样本表是手搓的：每一行恰好触发一条规则（或全部通过），因此丢弃计数是
精确值而不是范围。这样一旦规则顺序被改动，测试会立刻指出是哪一条。
"""

from decimal import Decimal

import pandas as pd
import pytest

from experiment.data.collect.clean import apply_cleaning, to_pence
from experiment.data.spec import CLEANING_RULES, SOURCE_COLUMNS

KEPT_ID = 0
KEPT_ID_2 = 1
KEPT_ID_MISSING_CUSTOMER = 5
CANCELLATION_ID = 2
NONPOSITIVE_QTY_ID = 3
NONPOSITIVE_PRICE_ID = 4
MISSING_STOCK_ID = 6
BAD_TIMESTAMP_ID = 7
FRACTIONAL_QTY_ID = 8
NONNUMERIC_QTY_ID = 9
BAD_STOCK_ID = 10
DUPLICATE_ID = 11
ZERO_PENCE_ID = 12


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = [
        (536365, "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        (536365, "71053", "WHITE METAL LANTERN", 6, "2010-12-01 08:26:00", 3.39, 17850.0, "United Kingdom"),
        ("C536379", "D", "Discount", -1, "2010-12-01 09:41:00", 27.50, 14527.0, "United Kingdom"),
        (536367, "84879", "ASSORTED COLOUR BIRD", -4, "2010-12-01 09:34:00", 1.69, 13047.0, "United Kingdom"),
        (536368, "84879", "ADJUSTABLE", 1, "2010-12-01 09:34:00", 0.0, 13047.0, "United Kingdom"),
        (536369, "22139", "RETROSPOT", 2, "2010-12-01 09:34:00", 1.25, None, "United Kingdom"),
        (536370, None, "?", 1, "2010-12-01 09:34:00", 1.0, 13047.0, "United Kingdom"),
        (536371, "22139", "RETROSPOT", 1, "not-a-date", 1.25, 13047.0, "United Kingdom"),
        (536372, "22139", "RETROSPOT", 1.5, "2010-12-01 09:34:00", 1.25, 13047.0, "United Kingdom"),
        (536373, "22139", "RETROSPOT", "abc", "2010-12-01 09:34:00", 1.25, 13047.0, "United Kingdom"),
        (536374, "ABC", "RETROSPOT", 1, "2010-12-01 09:34:00", 1.25, 13047.0, "United Kingdom"),
        (536365, "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        (536375, "22139", "RETROSPOT", 1, "2010-12-01 09:34:00", 0.001, 13047.0, "United Kingdom"),
    ]
    return pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))


@pytest.fixture
def dropped(raw_frame) -> dict[str, int]:
    _, report = apply_cleaning(raw_frame)
    return report["dropped_by_reason"]


@pytest.fixture
def kept(raw_frame) -> pd.DataFrame:
    frame, _ = apply_cleaning(raw_frame)
    return frame


def test_source_row_id_is_the_original_position(raw_frame):
    frame, _ = apply_cleaning(raw_frame)
    assert frame["source_row_id"].tolist() == [KEPT_ID, KEPT_ID_2, KEPT_ID_MISSING_CUSTOMER]


def test_cancellation_is_dropped(dropped):
    assert dropped["cancellation_invoice"] == 1


def test_cancellation_is_attributed_before_nonpositive(dropped):
    """C 开头的发票同时数量为负；归因必须落在取消发票上，而不是数量非正。"""
    assert dropped["cancellation_invoice"] == 1
    assert dropped["nonpositive_quantity_or_price"] == 2


def test_nonpositive_quantity_and_price_are_dropped(dropped):
    assert dropped["nonpositive_quantity_or_price"] == 2


def test_missing_required_field_is_dropped(dropped):
    assert dropped["missing_required_field"] == 1


def test_missing_customer_id_is_not_a_missing_required_field(dropped, kept):
    """CustomerID 允许为空：样本里那一行必须在幸存者中。"""
    assert dropped["missing_required_field"] == 1
    assert KEPT_ID_MISSING_CUSTOMER in kept["source_row_id"].tolist()


def test_report_keys_are_the_frozen_rules(raw_frame):
    _, report = apply_cleaning(raw_frame)
    assert tuple(report["dropped_by_reason"]) == CLEANING_RULES
    assert report["raw_rows"] == 13
    assert report["kept_rows"] == 3


def test_empty_frame_is_handled():
    frame, report = apply_cleaning(pd.DataFrame(columns=list(SOURCE_COLUMNS)))
    assert frame.empty
    assert report["raw_rows"] == 0
    assert report["kept_rows"] == 0
    assert all(count == 0 for count in report["dropped_by_reason"].values())


def test_invalid_timestamp_is_dropped(dropped):
    assert dropped["invalid_timestamp"] == 1


def test_fractional_quantity_is_dropped_not_rounded(dropped, kept):
    """1.5 件必须被丢弃，而不是静默取整成 1 或 2。"""
    assert dropped["non_integer_quantity"] == 2
    assert FRACTIONAL_QTY_ID not in kept["source_row_id"].tolist()


def test_nonnumeric_quantity_is_dropped(dropped):
    """非数值数量归入「非整数数量」，而不是「字段缺失」。"""
    assert dropped["non_integer_quantity"] == 2
    assert dropped["missing_required_field"] == 1


def test_invalid_stock_code_is_dropped(dropped):
    assert dropped["invalid_stock_code"] == 1


def test_duplicate_record_keeps_the_smallest_source_row_id(dropped, kept):
    assert dropped["exact_duplicate_record"] == 1
    assert DUPLICATE_ID not in kept["source_row_id"].tolist()
    assert KEPT_ID in kept["source_row_id"].tolist()


def test_zero_value_pence_is_dropped(dropped):
    assert dropped["zero_value_pence"] == 1


def test_identity_holds(raw_frame):
    """总账恒等式：原始行数 = 幸存行数 + 各规则丢弃数之和。"""
    _, report = apply_cleaning(raw_frame)
    dropped = report["dropped_by_reason"]
    assert sum(dropped.values()) == 10
    assert report["kept_rows"] == 3
    assert report["raw_rows"] == 13
    assert report["raw_rows"] == report["kept_rows"] + sum(dropped.values())


def test_kept_rows_are_exactly_the_expected_three(kept):
    assert kept["source_row_id"].tolist() == [
        KEPT_ID, KEPT_ID_2, KEPT_ID_MISSING_CUSTOMER
    ]


def test_kept_rows_carry_computed_pence(kept):
    by_id = kept.set_index("source_row_id")
    assert by_id.loc[KEPT_ID, "value_pence"] == 1530      # 2.55 * 6
    assert by_id.loc[KEPT_ID_2, "value_pence"] == 2034    # 3.39 * 6
    assert by_id.loc[KEPT_ID_MISSING_CUSTOMER, "value_pence"] == 250   # 1.25 * 2


def test_extreme_value_is_preserved_not_winsorised():
    """设计文档第 9 节：正异常值一律保留，不按任何分位数裁剪。

    取真实语料里的最大单笔（数量 80995 × 单价 2.08）：极值行必须原样幸存，
    价值恰为 16846960 分——既不封顶、也不截尾、更不被当作离群点丢弃。
    """
    rows = [
        (536365, "85123A", "MEDIUM CERAMIC TOP STORAGE JAR", 80995,
         "2010-12-01 08:26:00", 2.08, 17850.0, "United Kingdom"),
    ]
    frame, report = apply_cleaning(pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)))
    assert report["kept_rows"] == 1
    assert report["dropped_by_reason"]["zero_value_pence"] == 0
    assert frame["source_row_id"].tolist() == [0]
    assert frame["quantity_w"].tolist() == [80995]
    # 16846960 是手算的独立锚点；to_pence 一侧记录它正是规格给出的换算。
    assert frame["value_pence"].tolist() == [16846960]
    assert frame["value_pence"].tolist() == [to_pence(Decimal("2.08"), 80995)]


def test_quantity_w_is_integer_typed(kept):
    assert kept["quantity_w"].tolist() == [6, 6, 2]
    assert all(isinstance(x, int) for x in kept["quantity_w"].tolist())


def test_duplicate_is_judged_among_survivors_only():
    """一条已被前序规则丢弃的行，不应把它后面的同内容行「带走」。

    构造：第 0 行是取消发票，第 1 行与它逐字段相同。第 0 行按规则 2 丢弃；
    第 1 行若按「与原始表比较」会被判重复而丢弃，按「只与幸存者比较」则应保留。
    流水线语义要求后者。
    """
    rows = [
        ("C536379", "85123A", "X", -1, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
        ("C536379", "85123A", "X", -1, "2010-12-01 08:26:00", 2.55, 17850.0, "United Kingdom"),
    ]
    frame, report = apply_cleaning(pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)))
    assert report["dropped_by_reason"]["cancellation_invoice"] == 2
    assert report["dropped_by_reason"]["exact_duplicate_record"] == 0
    assert report["kept_rows"] == 0
