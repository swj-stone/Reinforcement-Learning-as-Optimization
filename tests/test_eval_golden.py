"""金样本：钉住求解器的回归（设计 §12）。

金样本**自包含**（内联实例数据），不依赖被 gitignore 的语料：参考解的正确性
不应该只有在拿到 522,504 行语料时才可验证。

本模块**没有**「缺失即生成」的回退：那种回退会让金样本在文件丢失时静默通过，
而本项目的验收要求全绿且无 skipped。
"""

import json
from fractions import Fraction
from pathlib import Path

from experiment.eval.reference import (
    format_upper_bound,
    fractional_upper_bound,
    solve,
)

GOLDEN = Path(__file__).parent / "golden" / "reference_solutions.json"

#: 每条记录必须**恰好**有这些键：多一个意味着有人塞进了未钉住的字段（比如
#: ``runtime_s`` 这种不可复现的墙钟量），少一个意味着钉住退化了。
FIELDS = frozenset(
    {
        "capacity",
        "item_set_hash",
        "name",
        "selected_indices",
        "upper_bound",
        "value",
        "values",
        "weights",
    }
)


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _instance(case: dict) -> dict:
    return {
        "capacity": case["capacity"],
        "values": case["values"],
        "weights": case["weights"],
    }


def test_golden_carries_exactly_the_pinned_fields():
    assert len(_golden()["instances"]) == 5
    for case in _golden()["instances"]:
        assert set(case) == FIELDS, case["name"]


def test_solver_reproduces_the_golden():
    for case in _golden()["instances"]:
        result = solve(_instance(case))
        assert result.value == case["value"], case["name"]
        assert result.selected_indices == case["selected_indices"], case["name"]
        assert result.item_set_hash == case["item_set_hash"], case["name"]
        assert format_upper_bound(result.upper_bound) == case["upper_bound"], case["name"]


def test_stored_upper_bounds_are_upward_rounded_and_tight():
    """金样本里的 U 是十进制字符串（JSON 没有 ``Fraction``），口径与 CSV 相同。

    既钉方向（存储值 >= 精确值），也钉紧度（最多高 1e-6）——只钉方向会让
    「无脑放大」也通过。
    """
    for case in _golden()["instances"]:
        exact = fractional_upper_bound(
            case["weights"], case["values"], case["capacity"]
        )
        stored = Fraction(case["upper_bound"])
        assert stored >= exact, case["name"]
        assert stored - exact < Fraction(1, 10**6), case["name"]
        assert stored >= case["value"], case["name"]
