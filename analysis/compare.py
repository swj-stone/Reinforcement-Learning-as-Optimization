"""Strict, paired comparison of frozen baseline results and three trained DQNs.

Run ``python -m analysis.compare --baseline baseline_results.csv --rl PATH PATH PATH
--out NEWDIR``. No solver, policy evaluation or training is performed here.
Bootstrap seed 7001 is fixed before test inspection; intervals condition on the
three trained models and do not estimate training-seed uncertainty.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = {"test": "test", "e3_size": "stress/e3_size", "e4_tail": "stress/e4_tail"}
EXPECTED_COUNTS = {"test": 2400, "e3_size": 600, "e4_tail": 300}
BOOTSTRAP_SEED = 7001
RESAMPLES = 1000
BASE_COLUMNS = {"instance_id", "split", "method", "trial", "seed", "value", "weight_used",
                "violations", "item_set_hash", "latency_s"}
RL_COLUMNS = BASE_COLUMNS | {"selected_indices", "checkpoint_sha256", "checkpoint_step", "run_kind"}
SEEDS = (0, 1, 2)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value)):
        raise ValueError(f"invalid integer {label}: {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"invalid {label}: {value!r}")
    return result


def _number(value, label):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {label}: {value!r}")
    return result


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid {label}")
    return value


def _read_csv(path, required):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV header: {path}")
        if not required <= set(reader.fieldnames):
            raise ValueError(f"missing columns in {path}: {sorted(required - set(reader.fieldnames))}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed CSV row: {path}")
    return rows


def _coverage(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label} coverage mismatch: missing={sorted(set(expected)-set(actual))[:3]}, "
                         f"extra={sorted(set(actual)-set(expected))[:3]}")


def load_results(baseline_path, rl_paths, instances, references, *, evaluation_seed=6001):
    """Validate exact joins and all declared outputs before any aggregation.

    Baselines do not contain selected indices: their stored violation counts and
    capacity totals can be checked, but their item hashes cannot be reconstructed.
    DQN selections are independently reconstructed from the shared corpus.
    Correctly declared infeasible solutions remain present as failures.
    """
    if len(rl_paths) != 3:
        raise ValueError("exactly three RL result files required (seeds 0, 1, 2)")
    baseline = _read_csv(baseline_path, BASE_COLUMNS)
    runs = [_read_csv(path, RL_COLUMNS) for path in rl_paths]
    for rows in [baseline, *runs]:
        seen = set()
        for row in rows:
            key = (row["instance_id"], row["method"], row["trial"])
            if key in seen:
                raise ValueError(f"duplicate result: {key}")
            seen.add(key)
        _coverage({row["instance_id"] for row in rows}, instances, "result ID")
        for row in rows:
            key = row["instance_id"]
            instance = instances[key]
            if row["split"] != instance["split"]:
                raise ValueError(f"split mismatch: {key}")
            for column in ("value", "weight_used", "violations"):
                _integer(row[column], f"{key} {column}")
            _number(row["latency_s"], f"{key} latency")
            _digest(row["item_set_hash"], "item_set_hash")
            if int(row["weight_used"]) > instance["capacity"] and int(row["violations"]) == 0:
                raise ValueError(f"unreported capacity violation: {key}")
            if not int(row["violations"]) and int(row["value"]) > references[key]["value"]:
                raise ValueError(f"feasible value exceeds certified optimum: {key}")
            if references[key]["value"] == 0 and (int(row["value"]) or int(row["weight_used"])):
                raise ValueError(f"nonempty trivial solution: {key}")
    grouped = defaultdict(list)
    for row in baseline:
        if row["method"] not in {"random", "greedy"}:
            raise ValueError(f"unexpected baseline method: {row['method']}")
        grouped[row["instance_id"], row["method"]].append(row)
        if row["method"] == "greedy":
            if row["trial"] or row["seed"]:
                raise ValueError("greedy trial and seed must be empty")
        else:
            trial = _integer(row["trial"], "random trial")
            payload = f"{evaluation_seed}|{row['instance_id']}|{trial}".encode()
            expected_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
            if _integer(row["seed"], "random seed") != expected_seed:
                raise ValueError(f"random seed mismatch: {row['instance_id']}")
    for key in instances:
        if len(grouped[key, "greedy"]) != 1:
            raise ValueError(f"greedy coverage mismatch: {key}")
        trials = [int(row["trial"]) for row in grouped[key, "random"]]
        if sorted(trials) != list(range(10)):
            raise ValueError(f"random trial coverage mismatch: {key}")
    seen_seeds = set()
    for rows in runs:
        seeds = {_integer(row["seed"], "RL seed") for row in rows}
        if len(seeds) != 1 or not seeds <= set(SEEDS) or seeds & seen_seeds:
            raise ValueError("RL seed coverage must be one distinct run each of 0, 1, 2")
        seen_seeds |= seeds
        checkpoint_fields = {(row["checkpoint_sha256"], row["checkpoint_step"], row["run_kind"])
                             for row in rows}
        if len(checkpoint_fields) != 1:
            raise ValueError("mixed checkpoint metadata within RL run")
        for row in rows:
            key = row["instance_id"]
            if row["method"] != "dqn" or row["trial"] or row["run_kind"] != "full":
                raise ValueError("RL requires full dqn results and empty trial")
            _digest(row["checkpoint_sha256"], "checkpoint_sha256")
            _integer(row["checkpoint_step"], "checkpoint step")
            selected = json.loads(row["selected_indices"])
            instance = instances[key]
            if not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < instance["n"]
                                                      for i in selected):
                raise ValueError(f"invalid selected indices: {key}")
            value = sum(instance["values"][i] for i in selected)
            weight = sum(instance["weights"][i] for i in selected)
            if value != int(row["value"]):
                raise ValueError(f"selected value mismatch: {key}")
            if weight != int(row["weight_used"]):
                raise ValueError(f"selected weight mismatch: {key}")
            expected_hash = hashlib.blake2b("|".join(map(str, sorted(selected))).encode(),
                                            digest_size=32).hexdigest()
            if expected_hash != row["item_set_hash"]:
                raise ValueError(f"selected item_set_hash mismatch: {key}")
            violations = len(selected) - len(set(selected)) + int(weight > instance["capacity"])
            if violations != int(row["violations"]):
                raise ValueError(f"selected violations mismatch: {key}")
    return baseline, sorted(runs, key=lambda rows: int(rows[0]["seed"]))


def aggregate_instances(instances, references, baseline, runs):
    """Average random trials first; retain per-seed DQN and an arithmetic seed mean."""
    grouped = defaultdict(list)
    for row in baseline + [row for run in runs for row in run]:
        seed = row["seed"] if row["method"] == "dqn" else ""
        grouped[row["instance_id"], row["method"], str(seed)].append(row)
    records = []
    for key, instance in sorted(instances.items()):
        reference = references[key]
        optimum, upper = reference["value"], float(reference["upper_bound"])
        greedy = float(grouped[key, "greedy", ""][0]["value"])
        dqn = []
        for method, seed in [("greedy", ""), ("random", ""), *[("dqn", str(s)) for s in SEEDS]]:
            rows = grouped[key, method, seed]
            value = statistics.mean(float(row["value"]) for row in rows)
            record = {k: instance.get(k) for k in ("instance_id", "split", "source", "family", "n", "alpha",
                                                   "source_date", "e4b_tail")}
            record.update(method=method, seed=seed, value=value,
                weight_used=statistics.mean(float(row["weight_used"]) for row in rows),
                optimum=optimum, upper_bound=upper, certified=bool(reference["certified"]),
                trivial=optimum == 0, normalized_quality=value / upper if optimum > 0 and upper > 0 else None,
                exact_quality=value / optimum if optimum > 0 else None,
                exact_gap_pct=100 * (optimum - value) / optimum if optimum > 0 else None,
                bound_gap_pct=100 * (upper - value) / upper if optimum > 0 and upper > 0 else None,
                relative_improvement_pct=100 * (value - greedy) / greedy if method == "dqn" and greedy > 0 else None,
                win=float(value > greedy) if method == "dqn" and optimum > 0 else None,
                tie=float(value == greedy) if method == "dqn" and optimum > 0 else None,
                loss=float(value < greedy) if method == "dqn" and optimum > 0 else None,
                feasible_rate=statistics.mean(int(row["violations"]) == 0 for row in rows),
                violation_count=sum(int(row["violations"]) for row in rows),
                failed_observations=sum(int(row["violations"]) > 0 for row in rows), observation_count=len(rows),
                _latencies=[float(row["latency_s"]) for row in rows])
            records.append(record)
            if method == "dqn":
                dqn.append(record)
        mean = dict(dqn[0], seed="mean")
        for field in ("value", "weight_used", "normalized_quality", "exact_quality", "exact_gap_pct", "bound_gap_pct",
                      "relative_improvement_pct", "win", "tie", "loss", "feasible_rate"):
            values = [record[field] for record in dqn]
            mean[field] = statistics.mean(values) if all(v is not None for v in values) else None
        for field in ("violation_count", "failed_observations", "observation_count"):
            mean[field] = sum(record[field] for record in dqn)
        mean["_latencies"] = [v for record in dqn for v in record["_latencies"]]
        records.append(mean)
    return records


def _mean(rows, field):
    """Equal weight to each represented source, then equal instances per source."""
    sources = defaultdict(list)
    for row in rows:
        sources[row["source"]]  # Retain an explicitly represented source even if its denominator is zero.
        if row[field] is not None:
            sources[row["source"]].append(row[field])
    return statistics.mean(statistics.mean(values) for values in sources.values()) if sources and all(sources.values()) else None


def _weighted_median(rows, field):
    selected = [row for row in rows if row[field] is not None]
    counts = Counter(row["source"] for row in selected)
    if not counts or set(counts) != {row["source"] for row in rows}:
        return None
    weighted = sorted((row[field], 1 / (len(counts) * counts[row["source"]])) for row in selected)
    cumulative = 0.
    for index, (value, weight) in enumerate(weighted):
        cumulative += weight
        if math.isclose(cumulative, .5, abs_tol=1e-12) and index + 1 < len(weighted):
            return (value + weighted[index + 1][0]) / 2
        if cumulative > .5:
            return value
    return weighted[-1][0]


def suite_records(records):
    return {"E1": [r for r in records if r["split"] == "test"],
            "E3": [r for r in records if r["split"] == "e3_size"],
            "E4": [r for r in records if r["split"] == "e4_tail"],
            "E4b": [r for r in records if r["split"] == "test" and r["source"] == "real" and r["e4b_tail"]]}


def _summary_row(rows, suite, group, group_value):
    first = rows[0]
    latency = [v for row in rows for v in row["_latencies"]]
    result = dict(suite=suite, group=group, group_value=group_value, method=first["method"], seed=first["seed"],
        instance_count=len(rows), synthetic_count=sum(r["source"] == "synthetic" for r in rows),
        real_count=sum(r["source"] == "real" for r in rows),
        source_date_count=len({r["source_date"] for r in rows if r["source"] == "real"}),
        trivial_count=sum(r["trivial"] for r in rows), certification_rate=_mean(rows, "certified"),
        feasible_rate=_mean(rows, "feasible_rate"), failure_instance_count=sum(r["failed_observations"] > 0 for r in rows),
        failed_observations=sum(r["failed_observations"] for r in rows),
        observation_count=sum(r["observation_count"] for r in rows),
        violation_count=sum(r["violation_count"] for r in rows),
        latency_median_s=float(np.median(latency)), latency_p95_s=float(np.percentile(latency, 95)),
        latency_observation_count=len(latency))
    for field in ("normalized_quality", "exact_quality", "exact_gap_pct", "bound_gap_pct", "relative_improvement_pct"):
        result[field + "_mean"] = _mean(rows, field)
        result[field + "_median"] = _weighted_median(rows, field)
    for field in ("win", "tie", "loss"):
        result[field + "_rate"] = _mean(rows, field)
    return result


def summarize(records):
    tables = []
    for suite, suite_rows in suite_records(records).items():
        groups = defaultdict(list)
        for row in suite_rows:
            dimensions = [("overall", "all"), ("source", row["source"]), ("n", str(row["n"])),
                          ("alpha", str(row["alpha"])), ("source_n", f"{row['source']}|{row['n']}"),
                          ("cell", f"{row['source']}|{row['family'] or 'real'}|{row['n']}|{row['alpha']}")]
            if row["source"] == "synthetic":
                dimensions.append(("family", row["family"]))
            if suite == "E1" and row["source"] == "real":
                dimensions.append(("real_tail", str(bool(row["e4b_tail"])).lower()))
            for group, value in dimensions:
                groups[group, value, row["method"], row["seed"]].append(row)
        local = {}
        for (group, value, method, seed), rows in sorted(groups.items()):
            item = _summary_row(rows, suite, group, value)
            local[group, value, method, seed] = item
            tables.append(item)
        for (group, value, method, seed), item in local.items():
            if method != "dqn" or seed != "mean":
                continue
            for metric in ("normalized_quality_mean", "exact_quality_mean", "exact_gap_pct_mean", "relative_improvement_pct_mean",
                           "win_rate", "tie_rate", "loss_rate", "feasible_rate"):
                values = [local[group, value, method, str(s)][metric] for s in SEEDS]
                name = metric.removesuffix("_mean") + "_seed_sd"
                item[name] = statistics.stdev(values) if all(v is not None for v in values) else None
    return tables


def bootstrap_differences(metadata, greedy, candidates, *, resamples=RESAMPLES, seed=BOOTSTRAP_SEED):
    """Paired percentile CI for candidate minus greedy exact gap, in percentage points.

    The same sampled indices multiply every method. Real clusters span every cell
    on a sampled source date; synthetic strata are family/n/alpha cells. Source
    means use all sampled instances (including cluster multiplicity), then receive
    equal weights. Trivial zero-optimum instances have NaN gaps and no denominator.
    """
    if resamples < 1 or not metadata:
        raise ValueError("bootstrap requires records and positive resamples")
    greedy = np.asarray(greedy, dtype=float)
    labels = list(candidates)
    deltas = np.column_stack([np.asarray(candidates[label], dtype=float) - greedy for label in labels])
    if deltas.shape[0] != len(metadata) or greedy.shape != (len(metadata),):
        raise ValueError("bootstrap method arrays must align with metadata")
    if np.isinf(deltas).any() or np.any(np.isnan(deltas) != np.isnan(deltas[:, :1])):
        raise ValueError("bootstrap has nonfinite or unpaired metric coverage")
    rng = np.random.default_rng(seed)
    real = defaultdict(list)
    synthetic = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["source"] == "real":
            if not row.get("source_date"):
                raise ValueError("real bootstrap record missing source_date")
            real[row["source_date"]].append(index)
        elif row["source"] == "synthetic":
            synthetic[row["family"], row["n"], row["alpha"]].append(index)
        else:
            raise ValueError("unknown bootstrap source")
    real_groups = [np.array(real[key]) for key in sorted(real)]
    syn_groups = [np.array(synthetic[key]) for key in sorted(synthetic)]
    def source_mean(indices):
        values = deltas[indices]
        valid = ~np.isnan(values[:, 0])
        return np.mean(values[valid], axis=0) if valid.any() else np.full(len(labels), np.nan)
    source_indices = [[i for group in groups for i in group] for groups in (real_groups, syn_groups) if groups]
    estimate_values = [source_mean(indices) for indices in source_indices]
    estimate = np.mean(estimate_values, axis=0)
    samples = []
    skipped = 0
    for _ in range(resamples):
        means = []
        if real_groups:
            selected = rng.integers(len(real_groups), size=len(real_groups))
            means.append(source_mean(np.concatenate([real_groups[i] for i in selected])))
        if syn_groups:
            selected = np.concatenate([group[rng.integers(len(group), size=len(group))] for group in syn_groups])
            means.append(source_mean(selected))
        # Do not silently change the estimand when a source draw has no nontrivial case.
        if any(np.isnan(mean).any() for mean in means):
            skipped += 1
            continue
        samples.append(np.mean(means, axis=0))
    result = {}
    for index, label in enumerate(labels):
        values = [float(sample[index]) for sample in samples]
        result[label] = {"estimate": float(estimate[index]) if math.isfinite(estimate[index]) else None,
            "ci_low": float(np.percentile(values, 2.5)) if values else None,
            "ci_high": float(np.percentile(values, 97.5)) if values else None,
            "samples": values, "resamples": resamples, "valid_resamples": len(values),
            "skipped_resamples": skipped, "bootstrap_seed": seed,
            "real_date_clusters": len(real_groups), "synthetic_cells": len(syn_groups)}
    return result


def _load_inputs(corpus, manifest_path, references_path, lots_path, *, expected_counts=EXPECTED_COUNTS):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    facts = {entry["path"]: entry for entry in manifest["files"]}
    if len(facts) != len(manifest["files"]):
        raise ValueError("duplicate manifest paths")
    instances, hashes, counts = {}, {}, {}
    canonical = set()
    for split, partition in PARTITIONS.items():
        relative = f"{partition}/instances.jsonl"
        fact = facts[relative]
        path = Path(corpus) / relative
        hashes[relative] = sha256(path)
        if hashes[relative] != fact["sha256"]:
            raise ValueError(f"corpus SHA-256 mismatch: {relative}")
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if len(records) != fact["lines"] or len(records) != expected_counts[split]:
            raise ValueError(f"unexpected corpus count for {split}")
        counts[split] = dict(Counter(record["source"] for record in records))
        frozen = manifest["counts"]["test"] if split == "test" else manifest["stress"][split]
        for source in ("synthetic", "real"):
            if counts[split].get(source, 0) != frozen[source]:
                raise ValueError(f"manifest source count mismatch: {split}/{source}")
        if split != "e4_tail" and counts[split].get("real") != counts[split].get("synthetic"):
            raise ValueError(f"source imbalance: {split}")
        if split == "e4_tail" and counts[split].get("real", 0):
            raise ValueError("E4 must be synthetic-only")
        for record in records:
            key = record["instance_id"]
            if key in instances or record["canonical_hash"] in canonical:
                raise ValueError(f"duplicate corpus ID or canonical hash: {key}")
            canonical.add(record["canonical_hash"])
            n = _integer(record["n"], "n", 1)
            if len(record["weights"]) != n or len(record["values"]) != n:
                raise ValueError(f"corpus item lengths mismatch: {key}")
            for value in record["weights"] + record["values"]:
                _integer(value, "item", 1)
            _integer(record["capacity"], "capacity")
            if record["source"] == "real" and not record.get("source_date"):
                raise ValueError(f"real source_date missing: {key}")
            if record["source"] not in {"real", "synthetic"}:
                raise ValueError(f"invalid source: {key}")
            record.update(split=split, e4b_tail=False)
            instances[key] = record
    references, seen = {}, set()
    for row in _read_csv(references_path, {"instance_id", "certified", "value", "upper_bound", "item_set_hash", "solver_time_s"}):
        key = row["instance_id"]
        if key in seen:
            raise ValueError(f"duplicate reference ID: {key}")
        seen.add(key)
        if key not in instances:
            continue
        optimum = _integer(row["value"], "reference value")
        upper = Fraction(row["upper_bound"])
        if row["certified"] != "true" or upper < optimum:
            raise ValueError(f"invalid or uncertified reference: {key}")
        if (optimum == 0) != all(w > instances[key]["capacity"] for w in instances[key]["weights"]):
            raise ValueError(f"reference trivial mismatch: {key}")
        _digest(row["item_set_hash"], "reference item_set_hash")
        _number(row["solver_time_s"], "reference solver time")
        references[key] = dict(value=optimum, upper_bound=upper, certified=True)
    _coverage(references, instances, "reference ID")
    tail = manifest["tail_thresholds"]
    quantity_threshold, price_threshold = Decimal(tail["quantity_w"]), Decimal(tail["unit_price"])
    required_rows = {str(row_id) for instance in instances.values() if instance["source"] == "real"
                     and instance["split"] == "test" for row_id in instance["source_row_ids"]}
    lots = {}
    with Path(lots_path).open(encoding="utf-8", newline="") as handle:
        for lot in csv.DictReader(handle):
            key = lot["source_row_id"]
            if key in required_rows:
                if key in lots:
                    raise ValueError(f"duplicate source row: {key}")
                lots[key] = lot
    _coverage(lots, required_rows, "E4b source row")
    for instance in instances.values():
        if instance["source"] != "real" or instance["split"] != "test":
            continue
        if len(instance["source_row_ids"]) != instance["n"]:
            raise ValueError("source row count mismatch")
        for i, row_id in enumerate(instance["source_row_ids"]):
            lot = lots[str(row_id)]
            weight, value = int(lot["quantity_w"]), int(lot["value_pence"])
            price = Decimal(lot["unit_price"])
            expected_value = int((Decimal(weight) * price * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            if (weight, value) != (instance["weights"][i], instance["values"][i]) or value != expected_value:
                raise ValueError(f"source lot does not match corpus input: {row_id}")
            if weight > quantity_threshold or price > price_threshold:
                instance["e4b_tail"] = True
    provenance = {"manifest_sha256": sha256(manifest_path), "reference_sha256": sha256(references_path),
                  "corpus_sha256": hashes, "lots_sha256": sha256(lots_path)}
    return instances, references, provenance, counts, tail


def _validate_evaluations(paths, runs, provenance, config):
    evaluations = []
    for path, rows in zip(paths, runs):
        info = json.loads(Path(path).with_name("evaluation.json").read_text(encoding="utf-8"))
        if info["rows"] != len(rows) or info["seed"] != int(rows[0]["seed"]):
            raise ValueError("evaluation metadata row count or seed mismatch")
        for key in ("checkpoint_sha256", "checkpoint_step", "run_kind"):
            if str(info[key]) != rows[0][key]:
                raise ValueError(f"evaluation metadata mismatch: {key}")
        for key in ("manifest_sha256", "reference_sha256", "config_sha256", "corpus_sha256"):
            if info["provenance"][key] != provenance[key]:
                raise ValueError(f"evaluation provenance mismatch: {key}")
        training = info["training"]
        if training["requested_transitions"] != config["budgets"]["training_transitions"]:
            raise ValueError("RL training budget differs from frozen configuration")
        if training["actual_transitions"] < training["requested_transitions"]:
            raise ValueError("incomplete RL training budget")
        if int(info["checkpoint_step"]) > training["actual_transitions"]:
            raise ValueError("checkpoint step exceeds completed training")
        evaluations.append(info)
    if len({json.dumps(info["evaluation_code_sha256"], sort_keys=True) for info in evaluations}) != 1:
        raise ValueError("evaluation code differs across seeds")
    return evaluations


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row if not key.startswith("_")))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if not key.startswith("_")} for row in rows)


def run(*, baseline, rl, out, corpus=ROOT / "datasets/processed", manifest=ROOT / "data_manifest.json",
        references=ROOT / "reference_results.csv", config=ROOT / "experiment_config.yaml",
        lots=ROOT / "datasets/raw/real/lots_clean.csv", expected_counts=EXPECTED_COUNTS):
    """Validate inputs and save deterministic summaries in a new directory."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    config_data = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if config_data["evaluation"]["random_trials_per_instance"] != 10 or config_data["budgets"]["training_seeds"] != list(SEEDS):
        raise ValueError("configuration does not match fixed 10-trial/three-seed protocol")
    instances, refs, provenance, counts, thresholds = _load_inputs(corpus, manifest, references, lots,
                                                                  expected_counts=expected_counts)
    provenance["config_sha256"] = sha256(config)
    baseline_rows, runs = load_results(baseline, rl, instances, refs,
                                       evaluation_seed=config_data["evaluation"]["evaluation_seed"])
    paths_by_seed = {int(_read_csv(path, RL_COLUMNS)[0]["seed"]): Path(path) for path in rl}
    ordered_paths = [paths_by_seed[seed] for seed in SEEDS]
    evaluations = _validate_evaluations(ordered_paths, runs, provenance, config_data)
    provenance["baseline_sha256"] = sha256(baseline)
    provenance["rl_sha256"] = {str(seed): sha256(path) for seed, path in zip(SEEDS, ordered_paths)}
    provenance["evaluation_sha256"] = {str(seed): sha256(path.with_name("evaluation.json"))
                                       for seed, path in zip(SEEDS, ordered_paths)}
    provenance["analysis_code_sha256"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    records = aggregate_instances(instances, refs, baseline_rows, runs)
    tables = summarize(records)
    bootstrap = []
    for suite, rows in suite_records(records).items():
        if not rows:
            continue
        keyed = {(r["instance_id"], r["method"], r["seed"]): r for r in rows}
        ids = sorted({r["instance_id"] for r in rows})
        meta = [instances[key] for key in ids]
        def values(method, seed):
            return np.array([keyed[key, method, seed]["exact_gap_pct"]
                             if keyed[key, method, seed]["exact_gap_pct"] is not None else np.nan for key in ids])
        results = bootstrap_differences(meta, values("greedy", ""),
                                       {seed: values("dqn", seed) for seed in ("0", "1", "2", "mean")})
        for seed, result in results.items():
            result.pop("samples")
            bootstrap.append(dict(suite=suite, seed=seed, metric="dqn_minus_greedy_exact_gap_pct", **result))
    primary = [row for row in tables if row["group"] == "overall"]
    summary = {"schema_version": 1, "protocol": {
        "quality": "normalized_quality = V / fractional upper_bound U; exact_quality = V / certified V*; exact gap = 100*(1-exact_quality)",
        "primary_weighting": "equal source means for E1/E3; synthetic only E4; real only E4b",
        "random": "arithmetic mean of 10 trials within each instance, never best-of-ten",
        "dqn_mean": "arithmetic mean across seeds 0, 1, 2; win/tie/loss average per-seed rates",
        "training_seed_sd": "sample standard deviation (ddof=1) of three source-balanced seed estimates",
        "median": "weighted empirical median with equal represented-source mass; DQN mean row takes the median of per-instance seed averages, not the mean of per-seed medians",
        "latency": "pooled observed single-solution calls; random per trial; DQN mean pools single-run seed timings descriptively, not ensemble cost",
        "bootstrap": "paired percentile 95% CI conditional on trained models; whole real dates across cells, synthetic instances within family/n/alpha cells; equal source means in each draw",
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": RESAMPLES,
        "bootstrap_delta": "DQN exact gap minus greedy exact gap, percentage points; lower is better",
        "infeasible": "retained and counted as failures; quality is reported as recorded and must be interpreted with feasibility",
        "trivial": "counted; excluded from percentage denominators; a source with no eligible denominator makes its mixed percentage undefined, and such bootstrap draws are skipped and counted",
        "e4b": "main-test subset; strict quantity_w or original UnitPrice above training 99th percentile, joined by source_row_ids",
        "e4b_thresholds": thresholds,
        "baseline_selection_validation": "CSV omits indices; only declared feasibility, capacity totals, value bounds, trial identities and hashes' format are verifiable",
        "e4b_and_e3_dependence": "E4b overlaps E1; E3 may reuse test source rows/dates; intervals are suite-specific, not independent studies"},
        "provenance": provenance, "counts": counts, "evaluations": evaluations,
        "primary": primary, "paired_bootstrap": bootstrap,
        "training_seed_variation": [{k: v for k, v in row.items() if k in {"suite", "group", "group_value"} or k.endswith("_seed_sd")}
                                    for row in primary if row["method"] == "dqn" and row["seed"] == "mean"]}
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "instances.csv", records)
    _write_csv(out / "summary.csv", tables)
    _write_csv(out / "paired_bootstrap.csv", bootstrap)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                                                allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rl", type=Path, nargs=3, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "datasets/processed")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_manifest.json")
    parser.add_argument("--references", type=Path, default=ROOT / "reference_results.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.yaml")
    parser.add_argument("--lots", type=Path, default=ROOT / "datasets/raw/real/lots_clean.csv")
    args = parser.parse_args(argv)
    try:
        result = run(**vars(args))
        print(f"Saved {len(result['primary'])} primary summary rows to {args.out}")
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
