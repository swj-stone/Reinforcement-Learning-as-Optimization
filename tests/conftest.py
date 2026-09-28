"""装配阶段的共享夹具。

夹具全部在 tmp_path 内自建，**不读 datasets/**：该目录被 gitignore，新克隆上
不存在，依赖它会让测试在别人机器上静默跳过或静默通过。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from experiment.data.generate.corpus import run_generation

LOTS_HEADER = (
    "source_row_id",
    "invoice_no",
    "stock_code",
    "invoice_date",
    "quantity_w",
    "unit_price",
    "value_pence",
)

#: 夹具批次表的天数与每天批次数。每天 24 条 ≥ 20，使 (n=20, alpha) 这个
#: **真实存在**的 cell 在密闭夹具上可装配（生成语料的 n 只取 20/50/100/200）。
FIXTURE_DAYS = 10
FIXTURE_LOTS_PER_DAY = 24

#: 夹具批次表的总行数，与随附的 collect_report.json 的 kept_rows 一致
FIXTURE_ROWS = FIXTURE_DAYS * FIXTURE_LOTS_PER_DAY


def write_lots_csv(path: Path, rows: list[tuple]) -> Path:
    """把 ``(row_id, invoice_no, day, qty, unit_price, value_pence)`` 写成批次表。

    列序与 ``spec.LOTS_COLUMNS`` 一致，日期统一补 `` 08:00:00``。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(LOTS_HEADER)]
    for row_id, invoice_no, day, qty, price, pence in rows:
        lines.append(f"{row_id},{invoice_no},85123A,{day} 08:00:00,{qty},{price},{pence}")
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


@pytest.fixture
def lots_csv(tmp_path: Path) -> Path:
    """10 天、每天 24 条的批次表，日期与行号的对应关系可预测。

    切分边界（第 4 节）：``floor(0.70*10)=7``、``floor(0.85*10)=8``，故
    train 7 天、validation 1 天、test 2 天。

    同目录下另写一份 `collect_report.json`，因为装配会拿它的 ``kept_rows``
    与批次表的实际行数对账——那是一条真实的上游产物一致性检查，夹具必须
    让它有得查。
    """
    rows = []
    row_id = 0
    for day_index in range(FIXTURE_DAYS):
        day = (date(2011, 1, 1) + timedelta(days=day_index)).isoformat()
        for slot in range(FIXTURE_LOTS_PER_DAY):
            rows.append(
                (
                    row_id,
                    100000 + day_index * 100 + slot,
                    day,
                    slot + 1,          # quantity_w 1..24
                    "2.50",
                    (slot + 1) * 250,  # value_pence = quantity_w * 250
                )
            )
            row_id += 1
    path = write_lots_csv(tmp_path / "lots_clean.csv", rows)
    (tmp_path / "collect_report.json").write_bytes(
        json.dumps(
            {
                "kept_rows": FIXTURE_ROWS,
                "schema_version": 1,
                "source": {
                    "url": "https://example.invalid/fixture.zip",
                    "xlsx_bytes": 123,
                    "xlsx_sha256": "0" * 64,
                },
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return path


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> Path:
    """每 cell 1 条的小型合成语料，供 merge 的密闭测试使用。

    走既有的 ``run_generation``，因此 cell 集合与真实语料完全一致
    （n ∈ SIZES，每 split 48 个文件）——测试里出现的 n 必须取自 SIZES。
    """
    out = tmp_path / "generation"
    run_generation(out_dir=out, quota_override={"train": 1, "validation": 1, "test": 1})
    return out
