"""日期切分、整票随最早日期、候选池与合格日期、E4b 阈值。"""

from decimal import Decimal

import pytest

from experiment.data.assemble import lots as L
from tests.conftest import FIXTURE_LOTS_PER_DAY, write_lots_csv


# --- 切分 -------------------------------------------------------------------


def test_split_dates_floors_both_boundaries():
    """第 4 节要求两个边界都下取整。

    取 13 个日期：floor(0.70*13)=9、floor(0.85*13)=11，故 train 9 天、
    validation 2 天、test 2 天。若实现用了 round 或 ceil 就会不同。
    """
    days = [f"2011-01-{d:02d}" for d in range(1, 14)]
    split = L.split_dates(days)
    assert len(split["train"]) == 9
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    assert split["train"][0] == "2011-01-01"
    assert split["test"][-1] == "2011-01-13"


def test_split_dates_are_disjoint_and_cover_everything():
    days = [f"2011-02-{d:02d}" for d in range(1, 29)]
    split = L.split_dates(days)
    flat = split["train"] + split["validation"] + split["test"]
    assert len(flat) == len(set(flat)) == 28
    assert set(flat) == set(days)
    assert max(split["train"]) < min(split["validation"]) < min(split["test"])


def test_split_dates_deduplicates_and_sorts_input():
    split = L.split_dates(["2011-01-03", "2011-01-01", "2011-01-03", "2011-01-02"])
    assert split["train"] + split["validation"] + split["test"] == [
        "2011-01-01",
        "2011-01-02",
        "2011-01-03",
    ]


def test_split_dates_can_leave_a_partition_empty():
    """3 天：floor(0.70*3)=2、floor(0.85*3)=2，validation 为空。"""
    split = L.split_dates(["2011-01-01", "2011-01-02", "2011-01-03"])
    assert split["train"] == ["2011-01-01", "2011-01-02"]
    assert split["validation"] == []
    assert split["test"] == ["2011-01-03"]


# --- 发票的最早日期 ---------------------------------------------------------


def _midnight_csv(tmp_path):
    rows = [
        (0, 900001, "2011-03-02", 3, "1.00", 300),
        (1, 900001, "2011-03-03", 4, "1.00", 400),
        (2, 900002, "2011-03-03", 5, "1.00", 500),
    ]
    return write_lots_csv(tmp_path / "mid.csv", rows)


def test_invoice_spanning_midnight_takes_its_earliest_date(tmp_path):
    """第 4 节：发票归到它最早一笔交易的日历日，即使跨午夜。"""
    index = L.load_index(_midnight_csv(tmp_path), fractions=(0.0, 0.0))
    assert index.first_dates["900001"] == "2011-03-02"
    assert index.first_dates["900002"] == "2011-03-03"
    assert index.date_splits["test"] == ["2011-03-02", "2011-03-03"]


def test_all_lots_of_an_invoice_land_in_the_earliest_days_pool(tmp_path):
    """跨午夜的发票，两笔都必须进「最早那天」的池子，而不是各自那天。"""
    index = L.load_index(_midnight_csv(tmp_path), fractions=(0.0, 0.0))
    assert [lot.source_row_id for lot in index.pools["test"]["2011-03-02"]] == [0, 1]
    assert [lot.source_row_id for lot in index.pools["test"]["2011-03-03"]] == [2]


# --- 候选池与合格日期 -------------------------------------------------------


def test_pool_preserves_file_row_order(lots_csv):
    index = L.load_index(lots_csv)
    for pool in index.pools["train"].values():
        ids = [lot.source_row_id for lot in pool]
        assert ids == sorted(ids)
        assert len(pool) == FIXTURE_LOTS_PER_DAY


def test_eligible_dates_excludes_pools_smaller_than_n(lots_csv):
    """第 4 节：不足 n 个批次的日期不合格；不得在实例内复制行凑数。"""
    index = L.load_index(lots_csv)
    assert len(index.eligible("train", FIXTURE_LOTS_PER_DAY)) == 7      # 每天正好 24 条
    assert index.eligible("train", FIXTURE_LOTS_PER_DAY + 1) == []      # 一天也凑不出 25 条
    assert index.eligible("train", 4) == index.date_splits["train"]


def test_eligible_dates_are_sorted(lots_csv):
    index = L.load_index(lots_csv)
    days = index.eligible("test", 4)
    assert days == sorted(days)
    assert len(days) == 2


def test_load_index_rejects_a_wrong_column_set(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="列"):
        L.load_index(path)


# --- E4b 阈值 ---------------------------------------------------------------


def test_tail_thresholds_ignore_rows_they_were_not_given(tmp_path):
    """阈值必须只由**传入的**批次定，不是由整张表定。

    构造两个日期：早的那天（会成为 train）quantity 全在 1..3，晚的那天
    （会成为 test）全是 1000。用 train 段算阈值必然落在 1..3；用全表算就会
    跳到 1000。两次调用一对比，「只看传入的行」就是一条会失败的性质——
    若实现偷偷去读了全表，第一次调用就会背叛自己。
    """
    rows = [(i, 700000 + i, "2011-01-01", 1 + i % 3, "1.00", 100 * (1 + i % 3)) for i in range(70)]
    rows += [(70 + i, 700070 + i, "2011-02-01", 1000, "9.99", 999000) for i in range(30)]
    index = L.load_index(write_lots_csv(tmp_path / "t.csv", rows))

    train_lots = [lot for pool in index.pools["train"].values() for lot in pool]
    assert len(train_lots) == 70 and len(index.lots) == 100

    on_train = L.tail_thresholds(train_lots, quantile=0.99)
    on_all = L.tail_thresholds(index.lots, quantile=0.99)
    assert int(on_train["quantity_w"]) <= 3      # 没被 1000 拉走
    assert int(on_all["quantity_w"]) == 1000     # 全表算就会是 1000
    assert on_train["basis"] == "train-split lots only"
    assert on_train["quantile"] == 0.99


def test_tail_thresholds_are_deterministic_and_exact_for_prices():
    lots = [
        L.Lot(i, f"7{i:05d}", "2011-01-01", 1, Decimal(p), 100)
        for i, p in enumerate(["1.00", "2.00", "3.00", "4.00", "5.00"])
    ]
    assert L.tail_thresholds(lots, quantile=0.99) == L.tail_thresholds(lots, quantile=0.99)
    # 最近秩法：ceil(0.99*5)-1 = 4，即最大值
    assert L.tail_thresholds(lots, quantile=0.99)["unit_price"] == "5.00"


def test_tail_thresholds_reject_an_empty_input():
    with pytest.raises(ValueError):
        L.tail_thresholds([])


def test_quantile_index_is_nearest_rank_and_clamped():
    assert L.quantile_index(5, 0.99) == 4
    assert L.quantile_index(100, 0.99) == 98
    assert L.quantile_index(1, 0.99) == 0
    assert L.quantile_index(3, 0.0) == 0
