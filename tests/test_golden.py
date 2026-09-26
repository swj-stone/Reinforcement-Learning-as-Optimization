"""金样本：把已冻结的语义钉死，任何改动都会被立刻发现。

金样本文件在首次运行时落地并由人工核对后入库。它锁定的是
(seed, weights, values, capacity, item_order) 这组内容，因此任何
会改变语料的重构（改抽样顺序、改取整方向、改种子派生）都会让它失败。
"""

import json
from pathlib import Path

import pytest

from experiment.data.generate import corpus
from experiment.data.spec import alpha_token

GOLDEN_DIR = Path(__file__).parent / "golden"

CASES = [
    ("train", "uniform", 20, 0.25, 0),
    ("train", "normal", 50, 0.50, 3),
    ("train", "correlated", 100, 0.75, 7),
    ("train", "skewed", 200, 0.25, 11),
    ("validation", "skewed", 200, 0.75, 4),
    ("test", "uniform", 50, 0.50, 24),
]


def _golden_path(split, family, n, alpha, index):
    """复用 spec.alpha_token，避免文件名约定在两处各写一遍而漂移。"""
    return GOLDEN_DIR / f"{split}_{family}_n{n}_{alpha_token(alpha)}_i{index}.json"


@pytest.mark.parametrize("split, family, n, alpha, index", CASES)
def test_matches_golden(split, family, n, alpha, index):
    path = _golden_path(split, family, n, alpha, index)
    built = corpus.build_instance(split, family, n, alpha, index)

    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(built, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        pytest.skip(f"金样本首次生成，请人工核对后入库: {path.name}")

    assert built == json.loads(path.read_text(encoding="utf-8"))


def test_golden_weights_are_integers():
    for split, family, n, alpha, index in CASES:
        path = _golden_path(split, family, n, alpha, index)
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(isinstance(x, int) for x in data["weights"])
        assert all(isinstance(x, int) for x in data["values"])
