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


import hashlib
import json
from pathlib import Path


def _one_instance(split="train"):
    return corpus.build_instance(split, "uniform", 20, 0.25, 0)


def test_write_cell_creates_lf_terminated_jsonl(tmp_path):
    path = tmp_path / "cell.jsonl"
    digest = corpus.write_cell(path, [_one_instance()])
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r" not in raw, "必须是 LF 换行，否则字节级不可复现"
    assert digest == hashlib.sha256(raw).hexdigest()


def test_write_cell_roundtrips(tmp_path):
    path = tmp_path / "cell.jsonl"
    instance = _one_instance()
    corpus.write_cell(path, [instance])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == instance


def test_write_cell_is_byte_identical_on_rerun(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    corpus.write_cell(a, [_one_instance()])
    corpus.write_cell(b, [_one_instance()])
    assert a.read_bytes() == b.read_bytes()


def test_run_generation_writes_every_cell_and_manifest(tmp_path):
    manifest = corpus.run_generation(out_dir=tmp_path, quota_override={"train": 2, "validation": 1, "test": 1})
    assert (tmp_path / "manifest.json").is_file()
    assert len(manifest["cells"]) == 48 * 3
    for split in ("train", "validation", "test"):
        expected = {"train": 2, "validation": 1, "test": 1}[split]
        for cell in [c for c in manifest["cells"] if c["split"] == split]:
            assert cell["count"] == expected


def test_manifest_records_actual_file_hashes(tmp_path):
    manifest = corpus.run_generation(out_dir=tmp_path, quota_override={"train": 1, "validation": 1, "test": 1})
    for cell in manifest["cells"]:
        raw = (tmp_path / cell["path"]).read_bytes()
        assert cell["sha256"] == hashlib.sha256(raw).hexdigest()


def test_manifest_global_hashes_are_unique(tmp_path):
    corpus.run_generation(out_dir=tmp_path, quota_override={"train": 3, "validation": 2, "test": 2})
    hashes = []
    for path in sorted(tmp_path.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            hashes.append(json.loads(line)["canonical_hash"])
    assert len(set(hashes)) == len(hashes), "跨 split 出现重复实例"


def test_run_generation_is_idempotent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    corpus.run_generation(out_dir=first, quota_override={"train": 2, "validation": 1, "test": 1})
    corpus.run_generation(out_dir=second, quota_override={"train": 2, "validation": 1, "test": 1})
    for path in sorted(first.rglob("*.jsonl")):
        twin = second / path.relative_to(first)
        assert path.read_bytes() == twin.read_bytes(), f"{path.name} 不可复现"


def test_manifest_schema_fields(tmp_path):
    manifest = corpus.run_generation(out_dir=tmp_path, quota_override={"train": 1, "validation": 1, "test": 1})
    assert manifest["schema_version"] == 1
    assert set(manifest) == {
        "cells", "dedup", "generated_at", "generator_fingerprint",
        "schema_version", "spec",
    }
    assert set(manifest["dedup"]) == {
        "collisions_resampled", "max_resample_attempt", "total_instances",
    }
    assert set(manifest["spec"]) == {
        "alphas", "families", "master_seeds", "quota", "sizes",
    }


def test_generator_fingerprint_is_stable_hex():
    first = corpus.generator_fingerprint()
    assert first == corpus.generator_fingerprint()
    assert len(first) == 64
    assert all(c in "0123456789abcdef" for c in first)


def test_manifest_fingerprint_matches_recomputation(tmp_path):
    manifest = corpus.run_generation(
        out_dir=tmp_path, quota_override={"train": 1, "validation": 1, "test": 1}
    )
    assert manifest["generator_fingerprint"] == corpus.generator_fingerprint()


def test_fingerprint_covers_the_runtime_not_just_source(monkeypatch):
    """指纹必须随运行环境变化。

    只哈希 .py 文件的实现会在这里失败——而那正是最需要诊断的情形：
    代码一行未改、numpy 升级了，输出却变了。
    """
    baseline = corpus.generator_fingerprint()
    monkeypatch.setattr(corpus.platform, "python_version", lambda: "9.9.9")
    assert corpus.generator_fingerprint() != baseline
