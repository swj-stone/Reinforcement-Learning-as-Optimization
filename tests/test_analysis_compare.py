"""Comparison fixtures never open the frozen evaluation corpus or train models."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.compare import aggregate_instances, bootstrap_differences, load_results, run, sha256, summarize


def digest(indices):
    return hashlib.blake2b("|".join(map(str, sorted(indices))).encode(), digest_size=32).hexdigest()


def trial_seed(key, trial):
    return int.from_bytes(hashlib.blake2b(f"6001|{key}|{trial}".encode(), digest_size=8).digest(), "big")


def row(key, method, indices, *, trial="", seed="", value=None, latency=1):
    value = sum([4, 6][i] for i in indices) if value is None else value
    result = dict(instance_id=key, split="test", method=method, trial=str(trial), seed=str(seed),
                  value=str(value), weight_used=str(len(indices)), violations="0",
                  item_set_hash=digest(indices), latency_s=str(latency))
    if method == "dqn":
        result.update(selected_indices=json.dumps(indices), checkpoint_sha256=str(seed) * 64,
                      checkpoint_step="300000", run_kind="full")
    return result


@pytest.fixture
def fixture():
    metadata = {
        "s": dict(instance_id="s", split="test", source="synthetic", family="uniform", n=2,
                  alpha=.5, capacity=2, weights=[1, 1], values=[4, 6], e4b_tail=False),
        "r": dict(instance_id="r", split="test", source="real", family=None, n=2,
                  alpha=.5, capacity=2, weights=[1, 1], values=[4, 6], source_date="2011-12-01",
                  e4b_tail=True),
    }
    references = {key: dict(value=10, upper_bound=12, certified=True) for key in metadata}
    baseline = [row(key, "greedy", [1]) for key in metadata]
    for key in metadata:
        # A best-of-ten bug returns 10; averaging must return 1 and 9 respectively.
        for trial in range(10):
            selected = [0, 1] if (trial == 0 if key == "s" else trial != 0) else []
            baseline.append(row(key, "random", selected, trial=trial, seed=trial_seed(key, trial),
                                latency=100 if trial == 0 else 1))
    rl = [[row(key, "dqn", [0] if seed == 0 else [1] if seed == 1 else [0, 1], seed=seed)
           for key in metadata] for seed in range(3)]
    return metadata, references, baseline, rl


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_random_is_averaged_before_quality_and_trial_latency_is_not_summed(fixture):
    metadata, references, baseline, rl = fixture
    records = aggregate_instances(metadata, references, baseline, rl)
    random = {r["instance_id"]: r for r in records if r["method"] == "random"}
    assert random["s"]["value"] == 1
    assert random["r"]["value"] == 9
    tables = summarize(records)
    overall = next(r for r in tables if r["suite"] == "E1" and r["group"] == "overall"
                   and r["method"] == "random")
    assert overall["normalized_quality_mean"] == pytest.approx(5 / 12)
    assert overall["exact_gap_pct_mean"] == 50
    assert overall["latency_median_s"] == 1
    assert overall["latency_p95_s"] == 100
    assert overall["latency_observation_count"] == 20


def test_dqn_seed_mean_retains_seed_uncertainty_and_reference_denominators(fixture):
    records = aggregate_instances(*fixture)
    rows = summarize(records)
    summary = next(r for r in rows if r["suite"] == "E1" and r["group"] == "overall"
                   and r["method"] == "dqn" and r["seed"] == "mean")
    mean_instance = next(r for r in records if r["method"] == "dqn" and r["seed"] == "mean")
    assert mean_instance["value"] == pytest.approx(20 / 3)
    assert summary["normalized_quality_mean"] == pytest.approx(5 / 9)
    assert summary["exact_quality_mean"] == pytest.approx(2 / 3)
    assert summary["exact_gap_pct_mean"] == pytest.approx(100 / 3)
    assert summary["exact_gap_pct_seed_sd"] == pytest.approx(np.std([60, 40, 0], ddof=1))
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["tie_rate"] == pytest.approx(1 / 3)
    assert summary["loss_rate"] == pytest.approx(1 / 3)
    assert summary["relative_improvement_pct_mean"] == pytest.approx(100 / 9)


def test_equal_source_weighting_does_not_weight_large_real_pool_more(fixture):
    metadata, references, baseline, rl = fixture
    records = aggregate_instances(metadata, references, baseline, rl)
    records += [dict(r, instance_id="r2") for r in records if r["instance_id"] == "r"]
    rows = summarize(records)
    overall = next(r for r in rows if r["suite"] == "E1" and r["group"] == "overall"
                   and r["method"] == "random")
    assert overall["exact_gap_pct_mean"] == 50


def test_bootstrap_preserves_pairing_real_date_clusters_and_synthetic_cells():
    metadata = [dict(source="real", source_date="A", family=None, n=20, alpha=.5)]
    metadata += [dict(source="real", source_date="B", family=None, n=50, alpha=.5)] * 3
    metadata += [dict(source="synthetic", family=f, n=20, alpha=.5) for f in ("uniform", "skewed")]
    greedy = np.array([1, 20, 30, 40, 50, 60.])
    candidates = {"constant": greedy + 7, "cluster": greedy + np.array([0, 100, 100, 100, 0, 0])}
    result = bootstrap_differences(metadata, greedy, candidates, resamples=1000, seed=7001)
    assert result["constant"]["ci_low"] == 7
    assert result["constant"]["ci_high"] == 7
    assert result["cluster"]["estimate"] == 37.5
    assert set(result["cluster"]["samples"]) == {0., 37.5, 50.}
    assert result == bootstrap_differences(metadata, greedy, candidates, resamples=1000, seed=7001)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate"), ("missing", "coverage"), ("extra", "coverage"),
    ("random_trial", "trial"), ("selected_value", "selected value"),
    ("selected_hash", "item_set_hash"), ("selected_weight", "selected weight"),
    ("latency_nan", "latency"), ("wrong_split", "split"), ("wrong_seed", "seed"),
])
def test_input_validation_rejects_bad_joins_and_inconsistent_rl(tmp_path, fixture, mutation, match):
    metadata, references, baseline, rl = copy.deepcopy(fixture)
    if mutation == "duplicate":
        rl[0].append(rl[0][0])
    elif mutation == "missing":
        rl[0].pop()
    elif mutation == "extra":
        rl[0].append(dict(rl[0][0], instance_id="unknown"))
    elif mutation == "random_trial":
        baseline.pop()
    elif mutation == "selected_value":
        rl[0][0]["value"] = "5"
    elif mutation == "selected_hash":
        rl[0][0]["item_set_hash"] = "a" * 64
    elif mutation == "selected_weight":
        rl[0][0]["weight_used"] = "0"
    elif mutation == "latency_nan":
        rl[0][0]["latency_s"] = "nan"
    elif mutation == "wrong_split":
        rl[0][0]["split"] = "e3_size"
    elif mutation == "wrong_seed":
        rl[1][0]["seed"] = "0"
    baseline_path = tmp_path / "baseline.csv"
    write_csv(baseline_path, baseline)
    paths = []
    for seed, rows in enumerate(rl):
        path = tmp_path / f"rl{seed}.csv"
        write_csv(path, rows)
        paths.append(path)
    with pytest.raises(ValueError, match=match):
        load_results(baseline_path, paths, metadata, references)


def test_trivial_cases_are_counted_and_excluded_from_percentage_denominators(fixture):
    metadata, references, baseline, rl = copy.deepcopy(fixture)
    metadata["s"]["capacity"] = 0
    references["s"] = dict(value=0, upper_bound=0, certified=True)
    for result in baseline + sum(rl, []):
        if result["instance_id"] == "s":
            result.update(value="0", weight_used="0", item_set_hash=digest([]))
            if result["method"] == "dqn":
                result["selected_indices"] = "[]"
    records = aggregate_instances(metadata, references, baseline, rl)
    rows = summarize(records)
    source = next(r for r in rows if r["suite"] == "E1" and r["group"] == "source"
                  and r["group_value"] == "synthetic" and r["method"] == "greedy")
    assert source["trivial_count"] == 1
    assert source["exact_gap_pct_mean"] is None
    assert source["normalized_quality_mean"] is None
    assert source["feasible_rate"] == 1
    mixed = next(r for r in rows if r["suite"] == "E1" and r["group"] == "overall" and r["method"] == "greedy")
    assert mixed["exact_gap_pct_mean"] is None  # A missing source denominator cannot silently become 100% real.


def test_declared_infeasible_selections_are_counted_not_dropped(tmp_path, fixture):
    metadata, references, baseline, rl = copy.deepcopy(fixture)
    bad = rl[0][0]
    bad.update(selected_indices="[0,0,1]", value="14", weight_used="3", violations="2",
               item_set_hash=digest([0, 0, 1]))
    baseline_path = tmp_path / "baseline.csv"
    write_csv(baseline_path, baseline)
    paths = []
    for seed, rows in enumerate(rl):
        path = tmp_path / f"rl{seed}.csv"
        write_csv(path, rows)
        paths.append(path)
    checked_baseline, checked_runs = load_results(baseline_path, paths, metadata, references)
    records = aggregate_instances(metadata, references, checked_baseline, checked_runs)
    output = next(r for r in summarize(records) if r["suite"] == "E1" and r["group"] == "overall"
                  and r["method"] == "dqn" and r["seed"] == "0")
    assert output["instance_count"] == 2
    assert output["failure_instance_count"] == 1
    assert output["violation_count"] == 2
    assert output["feasible_rate"] == .5


def test_bootstrap_cannot_reweight_a_trivial_only_source():
    meta = [dict(source="real", source_date="A"), dict(source="synthetic", family="u", n=20, alpha=.5)]
    result = bootstrap_differences(meta, [10., np.nan], {"0": [5., np.nan]}, resamples=3)
    assert result["0"]["estimate"] is None
    assert result["0"]["ci_low"] is None
    assert result["0"]["valid_resamples"] == 0


def end_to_end_fixture(tmp_path):
    """Small frozen files with every real schema field the comparison consumes."""
    corpus = tmp_path / "corpus"
    files, all_instances = [], []
    for split, partition, sources in [("test", "test", ["synthetic", "real"]),
                                      ("e3_size", "stress/e3_size", ["synthetic", "real"]),
                                      ("e4_tail", "stress/e4_tail", ["synthetic"])]:
        records = []
        for source in sources:
            key = f"{split}-{source}"
            record = dict(instance_id=key, source=source, family="uniform" if source == "synthetic" else None,
                          n=2, alpha=.5, capacity=2, weights=[1, 1], values=[4, 6], item_order=[0, 1],
                          canonical_hash=hashlib.sha256(key.encode()).hexdigest())
            if source == "real":
                record.update(source_date="2011-12-01", source_row_ids=[0, 1])
            records.append(record)
            all_instances.append(dict(record, split=split))
        path = corpus / partition / "instances.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        files.append(dict(path=f"{partition}/instances.jsonl", sha256=sha256(path), lines=len(records)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(files=files, counts={"test": dict(real=1, synthetic=1, total=2)},
        stress={"e3_size": dict(real=1, synthetic=1), "e4_tail": dict(real=0, synthetic=1)},
        tail_thresholds=dict(quantity_w="100", unit_price="0.05", basis="train-split lots only", quantile=.99))))
    references = tmp_path / "references.csv"
    write_csv(references, [dict(instance_id=r["instance_id"], certified="true", value=10, upper_bound="12.0",
                               item_set_hash=digest([0, 1]), solver_time_s="0.001") for r in all_instances])
    config = tmp_path / "config.yaml"
    config.write_text("budgets:\n  training_seeds: [0, 1, 2]\n  training_transitions: 300000\n"
                      "evaluation:\n  evaluation_seed: 6001\n  random_trials_per_instance: 10\n")
    lots = tmp_path / "lots.csv"
    write_csv(lots, [dict(source_row_id=i, quantity_w=1, unit_price=price, value_pence=value)
                     for i, price, value in [(0, ".04", 4), (1, ".06", 6)]])
    baseline = []
    for instance in all_instances:
        key, split = instance["instance_id"], instance["split"]
        baseline.append(dict(row(key, "greedy", [1]), split=split))
        baseline += [dict(row(key, "random", [] if t else [0, 1], trial=t, seed=trial_seed(key, t)), split=split)
                     for t in range(10)]
    baseline_path = tmp_path / "baseline.csv"
    write_csv(baseline_path, baseline)
    paths = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}" / "rl_results.csv"
        path.parent.mkdir()
        # A completed training run may validly select its initial validation checkpoint.
        rows = [dict(row(r["instance_id"], "dqn", [0, 1], seed=seed), split=r["split"], checkpoint_step="0")
                for r in all_instances]
        write_csv(path, rows)
        evaluation = dict(rows=len(rows), seed=seed, checkpoint_sha256=str(seed)*64, checkpoint_step=0,
            run_kind="full", provenance=dict(manifest_sha256=sha256(manifest), reference_sha256=sha256(references),
                config_sha256=sha256(config), corpus_sha256={f["path"]: f["sha256"] for f in files}),
            training=dict(requested_transitions=300000, actual_transitions=300004),
            evaluation_code_sha256={"code.py": "a"*64})
        path.with_name("evaluation.json").write_text(json.dumps(evaluation))
        paths.append(path)
    return dict(baseline=baseline_path, rl=paths, out=tmp_path / "comparison", corpus=corpus,
                manifest=manifest, references=references, config=config, lots=lots,
                expected_counts={"test": 2, "e3_size": 2, "e4_tail": 1})


def test_end_to_end_artifacts_include_fixed_model_uncertainty_and_initial_checkpoint(tmp_path):
    inputs = end_to_end_fixture(tmp_path)
    summary = run(**inputs)
    assert summary["protocol"]["bootstrap_seed"] == 7001
    primary = summary["primary"]
    tail = next(r for r in primary if r["suite"] == "E4b" and r["method"] == "dqn" and r["seed"] == "mean")
    assert tail["instance_count"] == 1
    assert tail["exact_gap_pct_mean"] == 0
    assert tail["exact_gap_pct_seed_sd"] == 0
    assert len(summary["paired_bootstrap"]) == 16
    assert all(row["estimate"] == -40 and row["ci_low"] == -40 and row["ci_high"] == -40
               for row in summary["paired_bootstrap"])
    e4_ci = next(r for r in summary["paired_bootstrap"] if r["suite"] == "E4")
    tail_ci = next(r for r in summary["paired_bootstrap"] if r["suite"] == "E4b")
    assert e4_ci["real_date_clusters"] == 0 and e4_ci["synthetic_cells"] == 1
    assert tail_ci["real_date_clusters"] == 1 and tail_ci["synthetic_cells"] == 0
    assert sorted(path.name for path in inputs["out"].iterdir()) == ["instances.csv", "paired_bootstrap.csv", "summary.csv", "summary.json"]
    assert len(list(csv.DictReader((inputs["out"] / "instances.csv").open()))) == 30
    assert "value_mean" not in tail  # monetary and synthetic units must never be mixed in aggregate tables
    first_bytes = {p.name: p.read_bytes() for p in inputs["out"].iterdir()}
    other = dict(inputs, out=tmp_path / "second")
    run(**other)
    assert first_bytes == {p.name: p.read_bytes() for p in other["out"].iterdir()}
    with pytest.raises(FileExistsError):
        run(**inputs)


@pytest.mark.parametrize("mutation,match", [("hash", "SHA-256"), ("count", "count"),
    ("metadata", "provenance"), ("incomplete", "incomplete"), ("lots", "source lot")])
def test_end_to_end_rejects_changed_frozen_inputs_before_writing(tmp_path, mutation, match):
    inputs = end_to_end_fixture(tmp_path)
    if mutation == "hash":
        with (inputs["corpus"] / "test/instances.jsonl").open("a") as handle:
            handle.write("\n")
    elif mutation == "count":
        inputs["expected_counts"]["test"] = 3
    elif mutation in {"metadata", "incomplete"}:
        path = inputs["rl"][0].with_name("evaluation.json")
        info = json.loads(path.read_text())
        if mutation == "metadata":
            info["provenance"]["config_sha256"] = "f" * 64
        else:
            info["training"]["actual_transitions"] = 299999
        path.write_text(json.dumps(info))
    else:
        inputs["lots"].write_text(inputs["lots"].read_text().replace(".06", ".07"))
    with pytest.raises(ValueError, match=match):
        run(**inputs)
    assert not inputs["out"].exists()


def test_e4b_uses_strict_original_price_threshold_not_greater_equal(tmp_path):
    inputs = end_to_end_fixture(tmp_path)
    path = inputs["manifest"]
    manifest = json.loads(path.read_text())
    manifest["tail_thresholds"]["unit_price"] = ".06"
    path.write_text(json.dumps(manifest))
    for result_path in inputs["rl"]:
        path = result_path.with_name("evaluation.json")
        info = json.loads(path.read_text())
        info["provenance"]["manifest_sha256"] = sha256(inputs["manifest"])
        path.write_text(json.dumps(info))
    result = run(**inputs)
    assert not any(row["suite"] == "E4b" for row in result["primary"])
