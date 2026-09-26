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
