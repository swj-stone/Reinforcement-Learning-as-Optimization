"""Small, self-contained follow-up holdouts; never open the frozen real corpus."""
from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiment.data.assemble.lots import load_index
from experiment.data.assemble.real import build_real_instance
from experiment.data.assemble.stress import build_e3_synthetic, build_e4_synthetic
from experiment.data.generate.corpus import build_instance
from experiment.data.generate.hashing import canonical_hash
from experiment.data.spec import LOTS_COLUMNS


SMALL_QUOTAS = {"main_synthetic_per_cell": 1, "main_real_per_cell": 4,
                "e3_synthetic_per_cell": 1, "e3_real_per_cell": 4,
                "e4_per_alpha": 1}


def module():
    path = Path(__file__).resolve().parents[1] / "portfolio" / "holdout.py"
    assert path.is_file(), "The follow-up holdout builder has not been implemented"
    spec = importlib.util.spec_from_file_location("holdout_under_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "fixture-repo"
    lots = root / "datasets/raw/real/lots_clean.csv"
    lots.parent.mkdir(parents=True)
    with lots.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(LOTS_COLUMNS)
        # More than 500 rows per date permits distinct n=500 combinations.
        for day in range(10):
            day_string = (date(2011, 1, 1) + timedelta(days=day)).isoformat()
            for slot in range(530):
                quantity = 1 + (slot * 7 + day * 13) % 200
                price = 1 + slot % 23
                writer.writerow([day * 530 + slot, f"I{day}-{slot}", "12345",
                                 day_string + " 09:00:00", quantity, price, quantity * price * 100])
    index = load_index(lots)
    original = {}
    for split in ("train", "validation", "test"):
        original[f"{split}/instances.jsonl"] = [
            build_instance(split, "uniform", 20, .25, 0),
            build_real_instance(index, split, split, 20, .25, 0),
        ]
    original["stress/e3_size/instances.jsonl"] = [
        build_e3_synthetic("normal", .5, 0),
        build_real_instance(index, "e3_size", "test", 500, .5, 0, seed_stream="stream_size"),
    ]
    original["stress/e4_tail/instances.jsonl"] = [build_e4_synthetic(.5, 0)]
    files = []
    for relative, rows in original.items():
        path = root / "datasets/processed" / relative
        write_jsonl(path, rows)
        files.append(dict(path=relative, lines=len(rows), sha256=sha(path), bytes=path.stat().st_size))
    manifest = dict(files=files, provenance={"lots_clean": {
        "path": "datasets/raw/real/lots_clean.csv", "sha256": sha(lots), "rows": 5300}},
        split_policy={"fractions": [.7, .85]},
        tail_thresholds={"basis": "train-split lots only", "quantile": .99,
                         "quantity_w": "200", "unit_price": "23"})
    (root / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_fresh_namespaces_are_deterministic_disjoint_and_source_balanced(repo, tmp_path):
    holdout = module()
    before = {p.relative_to(repo).as_posix(): sha(p) for p in repo.rglob("*") if p.is_file()}
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = holdout.build_holdout(first, repo, quotas=SMALL_QUOTAS)
    repeated = holdout.build_holdout(second, repo, quotas=SMALL_QUOTAS)
    assert manifest == repeated
    assert {p.relative_to(first).as_posix(): p.read_bytes() for p in first.rglob("*") if p.is_file()} == {
        p.relative_to(second).as_posix(): p.read_bytes() for p in second.rglob("*") if p.is_file()}
    expected = {"main": {"synthetic": 48, "real": 48, "total": 96},
                "e3_size": {"synthetic": 12, "real": 12, "total": 24},
                "e4_tail": {"synthetic": 3, "real": 0, "total": 3}}
    assert manifest["counts"] == expected
    assert manifest["namespaces"] == {"main": "portfolio_v2_test",
                                      "e3_size": "portfolio_v2_test/e3", "e4_tail": "portfolio_v2_test/e4"}
    original = [r for p in (repo / "datasets/processed").rglob("*.jsonl") for r in read_records(p)]
    old_hashes = {r["canonical_hash"] for r in original}
    records = [r for suite in expected for r in read_records(first / suite / "instances.jsonl")]
    assert len(records) == len({r["instance_id"] for r in records}) == len({r["canonical_hash"] for r in records}) == 123
    assert not old_hashes & {r["canonical_hash"] for r in records}
    for record in records:
        assert record["canonical_hash"] == canonical_hash(record["n"], record["capacity"], np.array(record["weights"]), np.array(record["values"]))
        assert sorted(record["item_order"]) == list(range(record["n"]))
        if record["source"] == "real":
            assert record["source_date"] in {"2011-01-09", "2011-01-10"}
            assert len(set(record["source_row_ids"])) == record["n"]
    for suite in ("main", "e3_size"):
        suite_records = read_records(first / suite / "instances.jsonl")
        for n, alpha in {(r["n"], r["alpha"]) for r in suite_records}:
            cell = [r for r in suite_records if r["n"] == n and r["alpha"] == alpha]
            assert sum(r["source"] == "synthetic" for r in cell) == sum(r["source"] == "real" for r in cell) == 4
    overlap = manifest["overlap"]["all_new_real"]
    for split in ("train", "validation"):
        assert overlap["original_source_pools"][split] == {"source_rows": 0, "source_dates": 0, "invoices": 0}
        assert overlap["original_instances"][split] == {"source_rows": 0, "source_dates": 0, "invoices": 0}
    assert overlap["original_instances"]["test"]["source_rows"] > 0
    assert overlap["original_source_pools"]["test"]["source_rows"] == overlap["unique_source_rows"]
    assert manifest["independent_business_sample"] is False
    for fact in manifest["files"]:
        assert sha(first / fact["path"]) == fact["sha256"]
        assert len(read_records(first / fact["path"])) == fact["lines"]
    assert {p.relative_to(repo).as_posix(): sha(p) for p in repo.rglob("*") if p.is_file()} == before


def test_existing_output_is_refused_before_reading_inputs(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        module().build_holdout(out, tmp_path / "missing-repo")
    assert marker.read_text() == "keep"


def test_collision_with_original_stress_is_resampled(repo, tmp_path):
    # Seed a known first new main candidate into the old stress partition.
    collision = build_instance("portfolio_v2_test", "uniform", 20, .25, 0)
    path = repo / "datasets/processed/stress/e4_tail/instances.jsonl"
    rows = read_records(path)
    rows.append(dict(collision, instance_id="old-stress-collision"))
    write_jsonl(path, rows)
    manifest_path = repo / "data_manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    fact = next(f for f in original_manifest["files"] if f["path"] == "stress/e4_tail/instances.jsonl")
    fact.update(sha256=sha(path), lines=len(rows), bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(original_manifest))
    out = tmp_path / "collision"
    manifest = module().build_holdout(out, repo, quotas=SMALL_QUOTAS)
    first = read_records(out / "main/instances.jsonl")[0]
    assert first["resample_attempt"] == 1
    assert first["canonical_hash"] != collision["canonical_hash"]
    assert manifest["dedup"]["collisions_resampled"] >= 1
    assert manifest["dedup"]["original_instances"] == 10


@pytest.mark.parametrize("changed", ["corpus", "lots"])
def test_changed_original_inputs_rejected_without_output(repo, tmp_path, changed):
    path = repo / ("datasets/processed/train/instances.jsonl" if changed == "corpus" else "datasets/raw/real/lots_clean.csv")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    out = tmp_path / "bad-input"
    with pytest.raises(ValueError, match="SHA-256"):
        module().build_holdout(out, repo, quotas=SMALL_QUOTAS)
    assert not out.exists()


def test_ineligible_size_fails_without_creating_partial_output(repo, tmp_path):
    lots = repo / "datasets/raw/real/lots_clean.csv"
    with lots.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    kept = [rows[0]] + [r for r in rows[1:] if int(r[0]) % 530 < 200]
    with lots.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(kept)
    manifest_path = repo / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["lots_clean"].update(sha256=sha(lots), rows=2000)
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "too-small"
    with pytest.raises(ValueError, match="500"):
        module().build_holdout(out, repo, quotas=SMALL_QUOTAS)
    assert not out.exists()
