"""Small portfolio report fixtures; no actual holdout or models are opened."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from portfolio.report import RAW_METHODS, aggregate, load_evaluation, report


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync_artifact_hashes(directory):
    path = directory / "evaluation.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["artifacts"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ("instances.jsonl", "results.csv", "references.json")}
    path.write_text(json.dumps(metadata), encoding="utf-8")


def fixture_evaluation(tmp_path):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    specs = [("s", "main", "synthetic"), ("r1", "main", "real"), ("r2", "main", "real"),
             ("s3", "e3_size", "synthetic"), ("r3", "e3_size", "real"), ("s4", "e4_tail", "synthetic")]
    instances = [dict(instance_id=key, split=split, source=source, family="uniform" if source == "synthetic" else None,
        n=2, alpha=.5, weights=[1, 1], values=[4, 6], capacity=2, item_order=[0, 1],
        source_date="2011-12-01" if source == "real" else None) for key, split, source in specs]
    (directory / "instances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in instances), encoding="utf-8")
    rows = []
    for instance in instances:
        key = instance["instance_id"]
        for method in RAW_METHODS:
            selected, arm = [1], method
            latency = 2
            if method.startswith("random_trial"):
                trial = int(method.removeprefix("random_trial"))
                selected = [0, 1] if (trial == 0 if instance["source"] == "synthetic" else trial != 0) else []
                arm, latency = "", 100 if trial == 0 else 1
            elif method == "value_first" or method == "portfolio_best":
                selected, arm = [0, 1], "value_first"
            elif method.startswith("selector_seed"):
                seed = int(method[-1])
                selected = [1] if seed < 2 else [0, 1]
                arm = "greedy_density" if seed < 2 else "value_first"
            elif method.startswith("guarded_seed"):
                seed = int(method[-1])
                selected = [1] if seed < 2 else [0, 1]
                arm = "greedy_density" if seed < 2 else "value_first"
            elif method == "fixed_best":
                selected, arm = [1], "greedy_density"
            rows.append(dict(instance_id=key, split=instance["split"], method=method,
                value=sum(instance["values"][i] for i in selected), weight_used=len(selected),
                selected_indices=json.dumps(selected), latency_s=latency, chosen_arm=arm,
                reference_value=10, upper_bound=12, certified="true"))
    write_csv(directory / "results.csv", rows)
    (directory / "references.json").write_text(json.dumps(dict(records=[dict(instance_id=r["instance_id"],
        value=10, upper_bound="12", certified=True, selected_indices=[0, 1]) for r in instances])), encoding="utf-8")
    (directory / "evaluation.json").write_text(json.dumps(dict(
        config=dict(bootstrap_seed=8101, bootstrap_resamples=1000), fixture=True,
        status="completed", rows=len(rows), instances=len(instances), fixed_best="greedy_density")), encoding="utf-8")
    sync_artifact_hashes(directory)
    return directory


def primary(result, split, method):
    return next(row for row in result["primary"] if row["split"] == split and row["method"] == method)


def test_report_averages_random_before_equal_source_quality_and_keeps_trial_latency(tmp_path):
    directory = fixture_evaluation(tmp_path)
    result = report(evaluation=directory, out=tmp_path / "report", figures=False)
    row = primary(result, "main", "random")
    # Synthetic random mean=1, real mean=9; 50:50 source quality =5/10 despite 1:2 counts.
    assert row["exact_gap_pct_mean"] == 50
    assert row["normalized_quality_mean"] == pytest.approx(5 / 12)
    assert row["latency_median_s"] == 1
    assert row["latency_p95_s"] == 100
    assert row["latency_observation_count"] == 30
    assert row["synthetic_count"] == 1 and row["real_count"] == 2
    assert primary(result, "e3_size", "random")["instance_count"] == 2
    assert primary(result, "e4_tail", "random")["real_count"] == 0


def test_seed_means_regret_and_paired_interval_sign_are_explicit(tmp_path):
    directory = fixture_evaluation(tmp_path)
    result = report(evaluation=directory, out=tmp_path / "report", figures=False)
    row = primary(result, "main", "selector_mean")
    assert row["exact_gap_pct_mean"] == pytest.approx(80 / 3)
    assert row["exact_gap_pct_seed_sd"] == pytest.approx(40 / (3 ** .5))
    assert row["regret_pp_to_portfolio_mean"] == pytest.approx(80 / 3)
    assert row["relative_improvement_pct_mean"] == pytest.approx(200 / 9)
    assert row["win_rate"] == pytest.approx(1 / 3)
    assert row["tie_rate"] == pytest.approx(2 / 3)
    assert row["latency_observation_count"] == 9
    cis = [r for r in result["paired_bootstrap"] if r["split"] == "main" and r["candidate_method"] == "selector_mean"]
    assert {r["reference_method"] for r in cis} == {"greedy_density", "fixed_best"}
    assert all(r["estimate"] == pytest.approx(-40 / 3) and r["ci_low"] == pytest.approx(-40 / 3)
               and r["ci_high"] == pytest.approx(-40 / 3) for r in cis)
    assert all(r["bootstrap_seed"] == 8101 for r in cis)
    family = [r for r in result["rows"] if r["group"] == "family"]
    assert family and all(r["real_count"] == 0 for r in family)
    chosen = [r for r in result["chosen_arm_counts"] if r["split"] == "main" and r["method"] == "selector_seed2"]
    assert chosen == [dict(split="main", method="selector_seed2", chosen_arm="value_first", count=3, share=1.)]


@pytest.mark.parametrize("mutation,match", [("duplicate", "duplicate"), ("missing", "coverage"),
    ("extra_id", "coverage"), ("bad_value", "value"), ("bad_weight", "weight"),
    ("duplicate_index", "selected"), ("bad_reference", "reference"), ("bad_split", "split"),
    ("bad_arm", "chosen_arm"), ("nan_time", "latency")])
def test_invalid_results_fail_before_creating_report(tmp_path, mutation, match):
    directory = fixture_evaluation(tmp_path)
    with (directory / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mutation == "duplicate": rows.append(rows[0])
    elif mutation == "missing": rows.pop()
    elif mutation == "extra_id": rows.append(dict(rows[0], instance_id="absent"))
    elif mutation == "bad_value": rows[0]["value"] = "7"
    elif mutation == "bad_weight": rows[0]["weight_used"] = "2"
    elif mutation == "duplicate_index": rows[0]["selected_indices"] = "[1,1]"
    elif mutation == "bad_reference": rows[0]["reference_value"] = "11"
    elif mutation == "bad_split": rows[0]["split"] = "e4_tail"
    elif mutation == "nan_time": rows[0]["latency_s"] = "nan"
    else:
        next(row for row in rows if row["method"] == "selector_seed2")["chosen_arm"] = "light_first"
    write_csv(directory / "results.csv", rows)
    sync_artifact_hashes(directory)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match=match):
        report(evaluation=directory, out=output, figures=False)
    assert not output.exists()


def test_outputs_refuse_overwrite_and_include_two_figure_pairs(tmp_path):
    directory = fixture_evaluation(tmp_path)
    out = tmp_path / "report"
    result = report(evaluation=directory, out=out)
    for name in ("summary.csv", "summary.json", "instances.csv", "paired_bootstrap.csv", "chosen_arms.csv", "tables.md",
                 "main_exact_gap.png", "main_exact_gap.svg", "selector_action_share.png", "selector_action_share.svg"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0
    assert "Conditional on fitted models" in (out / "tables.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        report(evaluation=directory, out=out)


def test_group_metric_arrays_never_reuse_another_splits_instances(tmp_path):
    directory = fixture_evaluation(tmp_path)
    instances, rows, metadata = load_evaluation(directory)
    data = aggregate(instances, rows)
    main = [r for r in data if r["split"] == "main"]
    assert len({r["instance_id"] for r in main}) == 3
    assert {r["instance_id"] for r in main} == {"s", "r1", "r2"}
    assert len([r for r in main if r["method"] == "random"]) == 3


@pytest.mark.parametrize("change,match", [("hash", "SHA-256"), ("status", "completed"),
    ("rows", "row count"), ("fixed", "fixed_best"), ("reference", "reference")])
def test_evaluation_metadata_and_exact_reference_file_are_verified(tmp_path, change, match):
    directory = fixture_evaluation(tmp_path)
    path = directory / "evaluation.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if change == "hash":
        with (directory / "results.csv").open("a") as handle:
            handle.write("\n")
    elif change == "status":
        metadata["status"] = "running"
    elif change == "rows":
        metadata["rows"] += 1
    elif change == "fixed":
        metadata["fixed_best"] = "value_first"
    else:
        reference_path = directory / "references.json"
        data = json.loads(reference_path.read_text(encoding="utf-8"))
        data["records"][0]["value"] = 11
        reference_path.write_text(json.dumps(data), encoding="utf-8")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    if change == "reference":
        sync_artifact_hashes(directory)
    with pytest.raises(ValueError, match=match):
        report(evaluation=directory, out=tmp_path / "report", figures=False)
