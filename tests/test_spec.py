import math
from pathlib import Path

from experiment.data import spec


def test_families_are_the_four_frozen_ones():
    assert spec.FAMILIES == ("uniform", "normal", "correlated", "skewed")


def test_sizes_and_alphas():
    assert spec.SIZES == (20, 50, 100, 200)
    assert spec.ALPHAS == (0.25, 0.50, 0.75)


def test_master_seeds():
    assert spec.MASTER_SEEDS == {"train": 1001, "validation": 2001, "test": 3001}


def test_cell_count_is_48():
    assert len(spec.FAMILIES) * len(spec.SIZES) * len(spec.ALPHAS) == 48


def test_all_cells_order_is_frozen():
    """顺序即去重时的判定顺序，写错嵌套序别的测试全绿、语义却已改变。"""
    cells = spec.all_cells()
    assert len(cells) == 48
    assert len(set(cells)) == 48
    assert cells[0] == ("uniform", 20, 0.25)
    assert cells[-1] == ("skewed", 200, 0.75)
    assert [c[0] for c in cells[::12]] == list(spec.FAMILIES)
    # n 居中、alpha 在内层且变化最快。只钉首尾元素抓不到 n/alpha 互换：
    # 两种排列的两端取值恰好相同，必须钉住内层的完整排列。
    assert [c[1] for c in cells[:12]] == [20, 20, 20, 50, 50, 50, 100, 100, 100, 200, 200, 200]
    assert [c[2] for c in cells[:12]] == [0.25, 0.50, 0.75] * 4


def test_quota_math_matches_plan():
    assert spec.SYNTHETIC_QUOTA == {"train": 125, "validation": 25, "test": 25}
    assert 48 * spec.SYNTHETIC_QUOTA["train"] == 6000
    assert 48 * spec.SYNTHETIC_QUOTA["validation"] == 1200
    assert 48 * spec.SYNTHETIC_QUOTA["test"] == 1200


def test_alpha_key_is_stable_and_two_decimals():
    assert spec.alpha_key(0.25) == "0.25"
    assert spec.alpha_key(0.50) == "0.50"
    assert spec.alpha_key(0.75) == "0.75"


def test_alpha_token_matches_filename_convention():
    assert spec.alpha_token(0.25) == "a25"
    assert spec.alpha_token(0.50) == "a50"
    assert spec.alpha_token(0.75) == "a75"


def test_output_dirs_are_under_datasets_raw():
    assert spec.GENERATION_DIR.parent == spec.DATASETS_RAW
    assert spec.DATASETS_RAW.name == "raw"
    assert spec.DATASETS_RAW.parent.name == "datasets"
    # 钉死 REPO_ROOT 的层级：上面三条只看目录名，parents[2] 若误写成 parents[1]，
    # DATASETS_RAW 会落到 experiment/datasets/raw，三条依然全绿。
    assert (spec.REPO_ROOT / "pyproject.toml").is_file()
    assert (spec.REPO_ROOT / "experiment" / "data" / "spec.py").is_file()


def test_skew_k_per_size():
    assert [spec.skew_k(n) for n in spec.SIZES] == [1, 3, 5, 10]


def test_skew_k_is_floor_of_half_offset():
    for n in (20, 50, 100, 200):
        assert spec.skew_k(n) == max(1, math.floor(0.05 * n + 0.5))
