"""把 ``lots_clean.csv`` 变成按日期分区的候选池。

只做数据变换：读表、定发票的最早日期、切分日期、建池、筛合格日期、算阈值。
不抽样、不落盘、不持全局状态。

依据 EXPERIMENT_PLAN.md 第 4 节。
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple, Sequence

from experiment.data.spec import (
    LOTS_COLUMNS,
    SPLITS,
    SPLIT_DATE_FRACTIONS,
    TAIL_QUANTILE,
)


class Lot(NamedTuple):
    """一条可用的批次。

    ``unit_price`` 保留 ``Decimal``：它是 E4b 阈值的输入，而阈值会被冻进
    manifest 并用于事后标记，浮点表示会让「同一份数据两次算出不同阈值」
    成为可能。``invoice_no`` 保留字符串：它是标识，不是数量。
    """

    source_row_id: int
    invoice_no: str
    invoice_date: str
    quantity_w: int
    unit_price: Decimal
    value_pence: int


class LotIndex(NamedTuple):
    """一次装表的全部派生结果，供 real.py 与 merge.py 消费。"""

    lots: list[Lot]
    first_dates: dict[str, str]
    date_splits: dict[str, list[str]]
    pools: dict[str, dict[str, list[Lot]]]

    def eligible(self, split: str, n: int) -> list[str]:
        """该 split 中批次数不少于 ``n`` 的日期，按日期升序。

        第 4 节：一个日期不足 ``n`` 个可用批次即不合格，且**不得在实例内
        复制行来凑数**——所以这里是硬筛选，不是「尽量」。
        """
        return sorted(day for day, pool in self.pools[split].items() if len(pool) >= n)


def read_lots(path: Path) -> list[Lot]:
    """按文件行序读回批次表，并校验列集合与规格一致。

    列名或列序变了就说明上游改了结构，此时继续往下算只会产出看似正常的
    错数据。与 ``collect.run.read_workbook`` 同一套做法。
    """
    lots: list[Lot] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != LOTS_COLUMNS:
            raise ValueError(f"批次表列与规格不符：期望 {LOTS_COLUMNS}，实际 {header}")
        for row in reader:
            lots.append(
                Lot(
                    source_row_id=int(row["source_row_id"]),
                    invoice_no=row["invoice_no"],
                    invoice_date=row["invoice_date"],
                    quantity_w=int(row["quantity_w"]),
                    unit_price=Decimal(row["unit_price"]),
                    value_pence=int(row["value_pence"]),
                )
            )
    return lots


def invoice_first_dates(lots: Sequence[Lot]) -> dict[str, str]:
    """发票号 -> 该发票最早一笔交易的日历日。

    一张发票可能跨午夜（第 4 节）。全部批次随发票走，所以先算出发票的归属日，
    再据此给批次定日期。
    """
    first: dict[str, str] = {}
    for lot in lots:
        day = lot.invoice_date[:10]
        current = first.get(lot.invoice_no)
        if current is None or day < current:
            first[lot.invoice_no] = day
    return first


def split_dates(
    days: Sequence[str],
    fractions: tuple[float, float] = SPLIT_DATE_FRACTIONS,
) -> dict[str, list[str]]:
    """把去重排序后的日期切成 train/validation/test（第 4 节）。

    两个边界都**下取整**：``floor(f0*D)`` 与 ``floor(f1*D)``。后者是
    validation 的**上界（开区间）**，不是它的天数——validation 的天数是两者之差。

    Args:
        days: 全部发票的最早日期，可含重复、可不排序
        fractions: 两个边界占比，默认取冻结常量；测试可传 ``(0.0, 0.0)``
            把全部日期归入 test
    """
    ordered = sorted(set(days))
    if not ordered:
        return {split: [] for split in SPLITS}
    low = math.floor(fractions[0] * len(ordered))
    high = math.floor(fractions[1] * len(ordered))
    return {
        "train": ordered[:low],
        "validation": ordered[low:high],
        "test": ordered[high:],
    }


def pools_by_split(
    lots: Sequence[Lot],
    first_dates: dict[str, str],
    date_splits: dict[str, list[str]],
) -> dict[str, dict[str, list[Lot]]]:
    """``split -> 发票归属日 -> 该日全部批次``，池内保持文件行序。

    **桶键是发票的最早日期，不是批次自己的 ``invoice_date``**（第 4 节 /
    设计文档 6.1：全部 lot 随发票走）。跨午夜的发票两笔都要落进「最早那天」
    的池子；若按批次自己的日期分桶，池键就会与 ``date_splits`` 的日期集合
    脱节，``eligible`` 会报出这个 split 并不拥有的日期。

    行序是冻结语义：抽样用 rng 在池内取下标，池的顺序一变，抽出的批次就变。
    """
    split_of_date = {day: split for split, days in date_splits.items() for day in days}
    pools: dict[str, dict[str, list[Lot]]] = {split: defaultdict(list) for split in SPLITS}
    for lot in lots:
        day = first_dates[lot.invoice_no]
        pools[split_of_date[day]][day].append(lot)
    return {split: dict(by_day) for split, by_day in pools.items()}


def load_index(
    path: Path,
    fractions: tuple[float, float] = SPLIT_DATE_FRACTIONS,
) -> LotIndex:
    """一次装表：读表、定发票日期、切分、建池。"""
    lots = read_lots(path)
    first_dates = invoice_first_dates(lots)
    date_splits = split_dates(list(first_dates.values()), fractions)
    pools = pools_by_split(lots, first_dates, date_splits)
    return LotIndex(lots, first_dates, date_splits, pools)


def quantile_index(count: int, quantile: float) -> int:
    """最近秩法的下标，钳到 ``[0, count-1]``。

    显式定义而非用 ``numpy.percentile``：后者的插值方式随版本与参数变化，
    而阈值要跨机器复现。``ceil(q*n)-1`` 是确定性的整数运算。
    """
    if count <= 0:
        raise ValueError("count must be positive")
    return max(0, min(count - 1, math.ceil(quantile * count) - 1))


def tail_thresholds(
    lots: Sequence[Lot],
    quantile: float = TAIL_QUANTILE,
) -> dict[str, object]:
    """E4b 的极值阈值：给定批次的 ``quantity_w`` 与 ``unit_price`` 的分位数。

    第 8 节硬要求阈值**只用训练记录**定。本函数只认传入的批次，调用方负责
    只传训练集那部分——这一点由
    ``test_tail_thresholds_ignore_rows_they_were_not_given`` 钉住。

    价格用 ``Decimal`` 排序而非 float：阈值会被冻进 manifest，浮点会让
    「同一份数据两次算出不同阈值」成为可能。返回值一律是字符串，JSON 无损。
    """
    if not lots:
        raise ValueError("tail_thresholds needs at least one lot")
    weights = sorted(lot.quantity_w for lot in lots)
    prices = sorted(lot.unit_price for lot in lots)
    return {
        "basis": "train-split lots only",
        "quantile": quantile,
        "quantity_w": str(weights[quantile_index(len(weights), quantile)]),
        "unit_price": str(prices[quantile_index(len(prices), quantile)]),
    }
