import numpy as np
import pytest

from experiment.data.generate import corpus


def test_instance_id_format():
    assert (
        corpus.instance_id("train", "uniform", 20, 0.25, 7)
        == "syn-train-uniform-n20-a25-i0007"
    )


def test_instance_id_pads_index_to_four_digits():
    assert corpus.instance_id("test", "skewed", 200, 0.75, 0).endswith("i0000")


def test_build_instance_has_the_frozen_fields():
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    assert set(inst) == {
        "alpha", "canonical_hash", "capacity", "family", "instance_id",
        "item_order", "n", "resample_attempt", "seed", "source", "stats",
        "values", "weights",
    }


def test_build_instance_basic_values():
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    assert inst["source"] == "synthetic"
    assert inst["family"] == "uniform"
    assert inst["n"] == 20
    assert inst["alpha"] == 0.25
    assert len(inst["weights"]) == 20
    assert len(inst["values"]) == 20
    assert inst["resample_attempt"] == 0


def test_item_order_is_a_permutation_of_range_n():
    inst = corpus.build_instance("train", "uniform", 50, 0.50, 3)
    assert sorted(inst["item_order"]) == list(range(50))


def test_build_instance_is_deterministic():
    a = corpus.build_instance("train", "correlated", 100, 0.75, 5)
    b = corpus.build_instance("train", "correlated", 100, 0.75, 5)
    assert a == b


def test_capacity_matches_formula():
    import math
    inst = corpus.build_instance("train", "skewed", 200, 0.25, 1)
    assert inst["capacity"] == math.floor(0.25 * sum(inst["weights"]))


def test_different_splits_give_different_instances():
    a = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    b = corpus.build_instance("test", "uniform", 20, 0.25, 0)
    assert a["canonical_hash"] != b["canonical_hash"]


def test_attempt_changes_the_instance():
    a = corpus.build_instance("train", "uniform", 20, 0.25, 0, attempt=0)
    b = corpus.build_instance("train", "uniform", 20, 0.25, 0, attempt=1)
    assert a["canonical_hash"] != b["canonical_hash"]


def test_seed_is_recorded_and_matches_derivation():
    from experiment.data.generate.seeds import derive_seed
    inst = corpus.build_instance("validation", "normal", 50, 0.50, 9)
    assert inst["seed"] == derive_seed("validation", "normal", 50, 0.50, 9, 0, "items")


def test_item_order_stream_is_isolated_from_items_stream():
    """排列必须来自 order 流，而不是 items 流用剩的 rng 状态。"""
    from experiment.data.generate.seeds import derive_seed
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    rng = np.random.default_rng(
        derive_seed("train", "uniform", 20, 0.25, 0, 0, "order")
    )
    assert inst["item_order"] == rng.permutation(20).tolist()


def test_build_cell_returns_exactly_quota_instances():
    seen = set()
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 125, seen)
    assert len(instances) == 125


def test_build_cell_records_every_hash_in_seen():
    seen = set()
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 10, seen)
    assert {i["canonical_hash"] for i in instances} == seen


def test_build_cell_has_no_internal_duplicates():
    seen = set()
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 125, seen)
    hashes = [i["canonical_hash"] for i in instances]
    assert len(set(hashes)) == len(hashes)


def test_build_cell_resamples_when_hash_already_seen():
    """把某个已知哈希预先塞进 seen，该实例必须被重抽掉。"""
    known = corpus.build_instance("train", "uniform", 20, 0.25, 0)["canonical_hash"]
    seen = {known}
    instances = corpus.build_cell("train", "uniform", 20, 0.25, 1, seen)
    assert instances[0]["canonical_hash"] != known
    assert instances[0]["resample_attempt"] >= 1


def test_build_cell_is_deterministic_across_calls():
    a = corpus.build_cell("test", "skewed", 50, 0.75, 8, set())
    b = corpus.build_cell("test", "skewed", 50, 0.75, 8, set())
    assert a == b


def test_build_cell_shares_seen_across_cells():
    seen = set()
    first = corpus.build_cell("train", "uniform", 20, 0.25, 20, seen)
    corpus.build_cell("train", "uniform", 20, 0.50, 20, seen)
    assert seen.issuperset(i["canonical_hash"] for i in first)


def test_resample_cap_raises_instead_of_looping_forever():
    """把上限设为 0，则任何一次碰撞都立刻触顶——用于验证死循环防护存在。

    刻意不用 max_attempt=3：那需要人为造出连续 4 次碰撞，代价高且脆弱。
    """
    seen = set()
    inst = corpus.build_instance("train", "uniform", 20, 0.25, 0)
    seen.add(inst["canonical_hash"])
    with pytest.raises(RuntimeError, match="resample"):
        corpus.build_cell("train", "uniform", 20, 0.25, 1, seen, max_attempt=0)
