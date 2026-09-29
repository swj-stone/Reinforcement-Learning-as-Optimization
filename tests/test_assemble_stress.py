"""E4 变体生成器、E3/E4 的形状与配额、流隔离。"""

import numpy as np
import pytest

from experiment.data.assemble import lots as L
from experiment.data.assemble import stress as S
from experiment.data.generate.families import generate_items
from experiment.data.generate.seeds import derive_seed
from experiment.data.spec import (
    ALPHAS,
    E3_NAME,
    E3_SIZE,
    E4_NAME,
    E4_SEED_TOKEN,
    E4_SIZE,
    E4_SKEW_HEAVY_DENSITY_RANGE,
    E4_SKEW_HEAVY_WEIGHT_RANGE,
    E4_SKEW_K,
    FAMILIES,
    REAL_SEED_TOKEN,
    SKEW_HEAVY_DENSITY_RANGE,
    SKEW_HEAVY_WEIGHT_RANGE,
    SPLITS,
    STRESS_SIZE_STREAM,
    STRESS_TAIL_STREAM,
    skew_k,
)

N = 20


@pytest.fixture
def index(lots_csv):
    return L.load_index(lots_csv)


# --- E4 变体生成器 ----------------------------------------------------------


def test_e4_variant_has_exactly_k_heavy_weights():
    """第 8 节：k=2 个稀有权重落在 U(500,2000)，其余落在 U(1,20)。

    这是精确计数，不是统计——它同时钉住 k 与权重范围两件事。
    """
    w, _ = S.generate_skewed_e4(N, seed=12345)
    lo, hi = E4_SKEW_HEAVY_WEIGHT_RANGE
    heavy = [int(x) for x in w if lo <= x <= hi]
    light = [int(x) for x in w if x not in heavy]
    assert len(heavy) == E4_SKEW_K == 2
    assert all(1 <= x <= 20 for x in light)
    assert len(light) == N - E4_SKEW_K


def test_e4_variant_has_exactly_k_heavy_densities():
    """密度无法从 v=w*d 反推，所以生成器多返回一个数组供直接检验。"""
    _, _, d = S.generate_skewed_e4(N, seed=999, with_density=True)
    lo, hi = E4_SKEW_HEAVY_DENSITY_RANGE
    heavy = [int(x) for x in d if lo <= x <= hi]
    assert len(heavy) == E4_SKEW_K
    assert all(1 <= x <= 5 for x in d if x not in heavy)


def test_e4_variant_ranges_are_strictly_wider_than_the_frozen_family():
    """变体必须真的更强，否则 E4 就不是「更强的不平衡」。

    三条各管一件事：稀有权的下界抬高、上界抬高、稀有密度的下界高过主语料
    稀有密度的上界。主语料的 skewed 半边已被金样本钉死，这些数字只能是变体
    自己的，不能与主语料相等。
    """
    assert E4_SKEW_HEAVY_WEIGHT_RANGE[0] > SKEW_HEAVY_WEIGHT_RANGE[0]   # 500 > 200
    assert E4_SKEW_HEAVY_WEIGHT_RANGE[1] > SKEW_HEAVY_WEIGHT_RANGE[1]   # 2000 > 500
    assert E4_SKEW_HEAVY_DENSITY_RANGE[0] > SKEW_HEAVY_DENSITY_RANGE[1]  # 100 > 60


def test_e4_variant_is_not_the_frozen_skewed_generator():
    """同一个种子下，变体与既有族必须产出不同的数组。

    若二者相同，说明变体只是把 _skewed 抄了一遍却没换参数。
    """
    seed = 4242
    w_e4, v_e4 = S.generate_skewed_e4(N, seed)
    w_sk, v_sk = generate_items("skewed", N, seed)
    assert not np.array_equal(w_e4, w_sk)
    assert not np.array_equal(v_e4, v_sk)


def test_e4_variant_values_are_weight_times_density():
    w, v, d = S.generate_skewed_e4(N, seed=777, with_density=True)
    assert np.array_equal(v, (w * d).astype(np.int64))


def test_e4_variant_is_deterministic():
    a = S.generate_skewed_e4(N, 31337)
    b = S.generate_skewed_e4(N, 31337)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_frozen_skewed_generator_is_untouched():
    """本任务不得改动 generate/families.py 的 _skewed。

    单抽一个种子直接数稀有项——若有人顺手改了主语料，这里会亮。
    """
    w, _ = generate_items("skewed", N, seed=1)
    lo, hi = SKEW_HEAVY_WEIGHT_RANGE
    heavy = [int(x) for x in w if lo <= x <= hi]
    assert len(heavy) == skew_k(N)


# --- 流隔离 -----------------------------------------------------------------


def test_e4_seed_does_not_collide_with_any_generation_seed():
    """第 8 节「保留独立流」的可观测形式。

    若 E4 复用 split/family 记号，它就会与主集某些实例共用种子。
    """
    e4_seed = derive_seed(STRESS_TAIL_STREAM, E4_SEED_TOKEN, E4_SIZE, 0.25, 0, 0, "items")
    for split in SPLITS:
        for family in FAMILIES:
            assert e4_seed != derive_seed(split, family, E4_SIZE, 0.25, 0, 0, "items")
        assert e4_seed != derive_seed(split, REAL_SEED_TOKEN, E4_SIZE, 0.25, 0, 0, "items")


def test_e3_seed_does_not_collide_with_any_generation_seed():
    e3_seed = derive_seed(STRESS_SIZE_STREAM, REAL_SEED_TOKEN, E3_SIZE, 0.25, 0, 0, "items")
    for split in SPLITS:
        for family in FAMILIES:
            assert e3_seed != derive_seed(split, family, E3_SIZE, 0.25, 0, 0, "items")
        assert e3_seed != derive_seed(split, REAL_SEED_TOKEN, E3_SIZE, 0.25, 0, 0, "items")


# --- E3 合成 ----------------------------------------------------------------


def test_e3_synthetic_shapes_and_labels():
    inst = S.build_e3_synthetic("uniform", 0.25, 0, n=N)
    assert inst["n"] == N
    assert inst["source"] == "synthetic"
    assert inst["family"] == "uniform"
    assert inst["instance_id"] == f"syn-{E3_NAME}-uniform-n{N}-a25-i0000"
    assert len(inst["weights"]) == len(inst["values"]) == N
    assert sorted(inst["item_order"]) == list(range(N))


def test_e3_synthetic_uses_the_size_stream():
    inst = S.build_e3_synthetic("normal", 0.50, 3, n=N)
    assert inst["seed"] == derive_seed(STRESS_SIZE_STREAM, "normal", N, 0.50, 3, 0, "items")


def test_e3_synthetic_defaults_to_the_frozen_size():
    assert S.build_e3_synthetic("uniform", 0.25, 0)["n"] == E3_SIZE == 500


# --- E4 合成 ----------------------------------------------------------------


def test_e4_synthetic_shapes_and_labels():
    inst = S.build_e4_synthetic(0.25, 0, n=N)
    assert inst["n"] == N
    assert inst["source"] == "synthetic"
    assert inst["family"] == E4_SEED_TOKEN
    assert inst["instance_id"] == f"syn-{E4_NAME}-{E4_SEED_TOKEN}-n{N}-a25-i0000"


def test_e4_family_token_is_not_a_frozen_family():
    """下游按 FAMILIES 枚举族的代码必须自动跳过 E4。"""
    assert E4_SEED_TOKEN not in FAMILIES


def test_e4_synthetic_uses_the_tail_stream():
    inst = S.build_e4_synthetic(0.75, 5, n=N)
    assert inst["seed"] == derive_seed(STRESS_TAIL_STREAM, E4_SEED_TOKEN, N, 0.75, 5, 0, "items")


def test_e4_synthetic_carries_exactly_k_heavy_items():
    """从产出的记录本身也能数出 2 个极值——端到端的第二重确认。"""
    inst = S.build_e4_synthetic(0.25, 0, n=N)
    lo, hi = E4_SKEW_HEAVY_WEIGHT_RANGE
    assert sum(1 for w in inst["weights"] if lo <= w <= hi) == E4_SKEW_K


def test_e4_synthetic_defaults_to_the_frozen_size():
    assert S.build_e4_synthetic(0.25, 0)["n"] == E4_SIZE == 200


# --- E3 真实 ----------------------------------------------------------------


def test_e3_real_draws_from_the_test_date_pool(index):
    inst = S.build_e3_real(index, 0.25, 0, n=N)
    assert inst["source"] == "real"
    assert inst["family"] is None
    assert inst["source_date"] in index.eligible("test", N)
    assert inst["instance_id"] == f"real-{E3_NAME}-n{N}-a25-i0000"


def test_e3_real_uses_the_size_stream(index):
    inst = S.build_e3_real(index, 0.50, 2, n=N)
    assert inst["seed"] == derive_seed(STRESS_SIZE_STREAM, REAL_SEED_TOKEN, N, 0.50, 2, 0, "items")


# --- 两个应力集的整体 -------------------------------------------------------


def test_build_e3_matches_the_frozen_quotas(index):
    insts = S.build_e3(index, n=N, synthetic_per_family=2, real_per_alpha=3)
    assert sum(1 for i in insts if i["source"] == "synthetic") == len(FAMILIES) * len(ALPHAS) * 2
    assert sum(1 for i in insts if i["source"] == "real") == len(ALPHAS) * 3


def test_build_e3_without_an_index_emits_no_real_instances():
    """没有批次索引就没有真实半边，但合成半边必须**照样足额**产出。

    只断言 ``all(source == "synthetic")`` 会被空列表满足，因此先钉住条数。
    """
    insts = S.build_e3(None, n=N, synthetic_per_family=1, real_per_alpha=3)
    assert len(insts) == len(FAMILIES) * len(ALPHAS) * 1
    assert all(i["source"] == "synthetic" for i in insts)


def test_build_e4_matches_the_frozen_quota():
    insts = S.build_e4(n=N, per_alpha=2)
    assert len(insts) == len(ALPHAS) * 2
    assert all(i["source"] == "synthetic" for i in insts)
    assert all(i["family"] == E4_SEED_TOKEN for i in insts)


def test_build_e3_and_e4_hashes_are_globally_unique():
    """空结果会让两条唯一性断言同时空转，所以先钉住总数。

    ``build_e3(None, ...)`` 只有合成半边：4 族 × 3 比率 × 2 条；E4 是 3 比率 × 2 条。
    """
    seen: set[str] = set()
    insts = S.build_e3(None, seen=seen, n=N, synthetic_per_family=2, real_per_alpha=2)
    insts += S.build_e4(seen=seen, n=N, per_alpha=2)
    assert len(insts) == len(FAMILIES) * len(ALPHAS) * 2 + len(ALPHAS) * 2
    hashes = [i["canonical_hash"] for i in insts]
    assert len(set(hashes)) == len(hashes)
    assert set(hashes) == seen
