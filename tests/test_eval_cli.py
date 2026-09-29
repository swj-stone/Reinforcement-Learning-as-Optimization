"""``reference_results.csv`` 的形状与字节（设计 §9）。"""

import csv
import json
from fractions import Fraction
from pathlib import Path

import pytest

from experiment.data.spec import E3_NAME, E4_NAME
from experiment.eval import cli

#: (instance_id, 实例体)。两个 id 的字典序与它们在语料里的出现顺序相反，
#: 这样「行序按 instance_id 升序」才有判别力。
INSTANCES = [
    ("b-second", {"capacity": 5, "values": [7, 1], "weights": [5, 1]}),
    ("a-first", {"capacity": 9, "values": [10, 7, 9], "weights": [8, 4, 5]}),
]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """手写的五文件语料：前三个 split 各一/零条，E3 一条，E4 空。

    刻意手写而不是跑 ``run_assembly``：这里要测的是 **CLI 的读取与落盘**，
    不是装配；手写让「读到了哪些文件」完全可控。
    """
    root = tmp_path / "processed"
    layout = {
        root / "train" / "instances.jsonl": [INSTANCES[0]],
        root / "validation" / "instances.jsonl": [INSTANCES[1]],
        root / "test" / "instances.jsonl": [],
        root / "stress" / E3_NAME / "instances.jsonl": [INSTANCES[0]],
        root / "stress" / E4_NAME / "instances.jsonl": [],
    }
    for path, items in layout.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "capacity": body["capacity"],
                    "instance_id": name,
                    "values": body["values"],
                    "weights": body["weights"],
                },
                sort_keys=True,
            )
            for name, body in items
        ]
        path.write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return root


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_corpus_file_is_read_and_the_row_order_is_fixed(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    summary = cli.run(corpus, out)
    # train 1 + validation 1 + test 0 + E3 1 + E4 0
    assert summary["rows"] == 3
    # 按 instance_id 升序，与文件顺序无关：a-first 在 validation 里，却排在前面
    assert [row["instance_id"] for row in _rows(out)] == [
        "a-first",
        "b-second",
        "b-second",
    ]


def test_column_order_and_bytes_are_fixed(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    cli.run(corpus, out)

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")

    with out.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == list(cli.COLUMNS)


def test_certified_is_lowercase_true_everywhere(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    cli.run(corpus, out)
    assert {row["certified"] for row in _rows(out)} == {"true"}


def test_upper_bound_is_an_upward_rounded_decimal_string(corpus, tmp_path):
    out = tmp_path / "reference_results.csv"
    cli.run(corpus, out)
    for row in _rows(out):
        integer, _, decimals = row["upper_bound"].partition(".")
        assert integer and len(decimals) == 6, row
        assert Fraction(row["upper_bound"]) >= int(row["value"]), row


def test_only_the_time_column_is_unstable(corpus, tmp_path):
    """两次运行除 ``solver_time_s`` 外逐格相同（设计 §9）。

    该列是墙钟测量。**任何把整份 CSV 一起哈希**的校验都会重蹈 Plan 3 的覆辙：
    当时一个存活探针把含临时路径与时钟的 manifest 一起哈希，两次纯净运行都
    不一致，于是每一个「存活」判定都是假的。比对必须排除这一列。
    """
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    cli.run(corpus, first)
    cli.run(corpus, second)

    def without_time(path: Path) -> list[list[str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        index = rows[0].index("solver_time_s")
        return [[cell for i, cell in enumerate(row) if i != index] for row in rows]

    assert without_time(first) == without_time(second)
