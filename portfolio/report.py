"""Validate and summarize saved portfolio evaluations, without running policies.

Run ``python -m portfolio.report --evaluation EVALUATION_DIR --out NEW_DIR``.
Bootstrap seed 8101 and 1,000 replicates are frozen in portfolio_config.json.
Intervals condition on fitted models; training-seed sample SD is separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

from analysis.compare import bootstrap_differences
from portfolio.strategies import CANDIDATE_NAMES


ARMS = (*CANDIDATE_NAMES, "dqn_seed0")
DIRECT_METHODS = (*CANDIDATE_NAMES, "dqn_seed0", "dqn_seed1", "dqn_seed2")
RAW_METHODS = (*DIRECT_METHODS, *(f"random_trial{i}" for i in range(10)), "fixed_best",
               *(f"selector_seed{i}" for i in range(3)), *(f"guarded_seed{i}" for i in range(3)), "portfolio_best")
SEED_GROUPS = {f"{name}_mean": tuple(f"{name}_seed{i}" for i in range(3)) for name in ("dqn", "selector", "guarded")}
SPLITS = ("main", "e3_size", "e4_tail")
BOOTSTRAP_SEED, RESAMPLES = 8101, 1000
COLUMNS = {"instance_id", "split", "method", "value", "weight_used", "selected_indices", "latency_s",
           "chosen_arm", "reference_value", "upper_bound", "certified"}


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _int(value, label):
    if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 0:
        raise ValueError(f"invalid nonnegative integer {label}")
    return int(value)


def _coverage(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(f"{label} coverage mismatch: missing={sorted(set(expected)-set(actual))[:3]}, "
                         f"extra={sorted(set(actual)-set(expected))[:3]}")


def load_evaluation(evaluation):
    """Reconstruct every selected set and require a complete instance/method join."""
    directory = Path(evaluation)
    metadata = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("evaluation status must be completed")
    artifacts = metadata.get("artifacts", {})
    for name in ("instances.jsonl", "results.csv"):
        if artifacts.get(name) != _hash(directory / name):
            raise ValueError(f"evaluation artifact SHA-256 mismatch: {name}")
    reference_path = directory / "references.json"
    if (reference_path.exists() or "references.json" in artifacts) and artifacts.get("references.json") != _hash(reference_path):
        raise ValueError("evaluation artifact SHA-256 mismatch: references.json")
    instances = {}
    with (directory / "instances.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            instance = json.loads(line)
            key = instance["instance_id"]
            if key in instances:
                raise ValueError(f"duplicate instance ID: {key}")
            if instance["split"] not in SPLITS or instance["source"] not in {"real", "synthetic"}:
                raise ValueError(f"invalid split/source: {key}")
            if instance["split"] == "e4_tail" and instance["source"] != "synthetic":
                raise ValueError("e4_tail must be synthetic-only")
            n = _int(instance["n"], "n")
            if len(instance["weights"]) != n or len(instance["values"]) != n:
                raise ValueError(f"item-array length mismatch: {key}")
            for value in instance["weights"] + instance["values"]:
                if _int(value, "item") == 0:
                    raise ValueError(f"nonpositive item: {key}")
            _int(instance["capacity"], "capacity")
            if instance["source"] == "real" and not instance.get("source_date"):
                raise ValueError(f"missing real source_date: {key}")
            if instance["source"] == "synthetic" and not instance.get("family"):
                raise ValueError(f"missing synthetic family: {key}")
            instances[key] = instance
    if not instances:
        raise ValueError("empty instance corpus")
    with (directory / "results.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not COLUMNS <= set(reader.fieldnames):
            raise ValueError("missing or duplicate results CSV columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed results CSV row")
    keyed = {}
    for row in rows:
        key = (row["instance_id"], row["method"])
        if key in keyed:
            raise ValueError(f"duplicate result row: {key}")
        keyed[key] = row
    _coverage(keyed, {(key, method) for key in instances for method in RAW_METHODS}, "result")
    if metadata.get("rows") != len(rows) or metadata.get("instances") != len(instances):
        raise ValueError("evaluation metadata row count or instance count mismatch")
    config = metadata.get("config", {})
    for key, expected in (("arms", list(ARMS)), ("seeds", [0, 1, 2]), ("random_trials", 10)):
        if key in config and config[key] != expected:
            raise ValueError(f"evaluation config mismatch: {key}")
    if "holdout" in config:
        for split in SPLITS:
            if config["holdout"][split] != sum(instance["split"] == split for instance in instances.values()):
                raise ValueError(f"evaluation holdout count mismatch: {split}")
    references = {}
    for (key, method), row in keyed.items():
        instance = instances[key]
        if row["split"] != instance["split"]:
            raise ValueError(f"split mismatch: {key}")
        selected = json.loads(row["selected_indices"])
        if not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < instance["n"] for i in selected):
            raise ValueError(f"invalid selected indices: {key}/{method}")
        if len(set(selected)) != len(selected):
            raise ValueError(f"duplicate selected index: {key}/{method}")
        value, weight = _int(row["value"], "value"), _int(row["weight_used"], "weight")
        if sum(instance["values"][i] for i in selected) != value:
            raise ValueError(f"selected value mismatch: {key}/{method}")
        if sum(instance["weights"][i] for i in selected) != weight:
            raise ValueError(f"selected weight mismatch: {key}/{method}")
        if weight > instance["capacity"]:
            raise ValueError(f"infeasible selected weight: {key}/{method}")
        latency = float(row["latency_s"])
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"invalid latency: {key}/{method}")
        optimum, upper = _int(row["reference_value"], "reference value"), Fraction(row["upper_bound"])
        if row["certified"].lower() != "true" or upper < optimum or value > optimum:
            raise ValueError(f"invalid reference/certification/value bound: {key}")
        reference = (optimum, upper)
        if key in references and references[key] != reference:
            raise ValueError(f"inconsistent reference across methods: {key}")
        references[key] = reference
        if (optimum == 0) != all(w > instance["capacity"] for w in instance["weights"]):
            raise ValueError(f"reference trivial-case mismatch: {key}")
        row.update(value=value, weight_used=weight, selected_indices=tuple(sorted(selected)), latency_s=latency,
                   reference_value=optimum, upper_bound=upper, certified=True)
    for (key, method), row in keyed.items():
        arm = row["chosen_arm"]
        if method in DIRECT_METHODS:
            if arm and arm != method:
                raise ValueError(f"chosen_arm mismatch for direct method: {key}/{method}")
        elif not method.startswith("random_trial"):
            if arm not in ARMS:
                raise ValueError(f"invalid chosen_arm: {key}/{method}")
            source = keyed[key, arm]
            if (row["value"], row["weight_used"], row["selected_indices"]) != (source["value"], source["weight_used"], source["selected_indices"]):
                raise ValueError(f"chosen_arm solution mismatch: {key}/{method}")
        if method == "portfolio_best" and row["value"] != max(keyed[key, arm]["value"] for arm in ARMS):
            raise ValueError(f"portfolio_best does not match candidate maximum: {key}")
        if method.startswith("guarded_seed") and row["value"] < keyed[key, "greedy_density"]["value"]:
            raise ValueError(f"guarded result regresses below greedy: {key}")
    fixed_arms = {row["chosen_arm"] for row in rows if row["method"] == "fixed_best"}
    if fixed_arms != {metadata.get("fixed_best")}:
        raise ValueError("fixed_best must match the single validation-selected arm in metadata")
    if reference_path.exists():
        exact = {}
        for record in json.loads(reference_path.read_text(encoding="utf-8"))["records"]:
            key = record["instance_id"]
            if key in exact:
                raise ValueError(f"duplicate exact reference ID: {key}")
            exact[key] = record
        _coverage(exact, instances, "exact reference")
        for key, record in exact.items():
            instance = instances[key]
            selected = record["selected_indices"]
            if not isinstance(selected, list) or any(type(i) is not int or not 0 <= i < instance["n"] for i in selected):
                raise ValueError(f"invalid reference selected indices: {key}")
            if len(set(selected)) != len(selected) or sum(instance["weights"][i] for i in selected) > instance["capacity"]:
                raise ValueError(f"infeasible reference selection: {key}")
            optimum, upper = _int(record["value"], "exact reference value"), Fraction(record["upper_bound"])
            if record["certified"] is not True or sum(instance["values"][i] for i in selected) != optimum or upper < optimum:
                raise ValueError(f"invalid exact reference: {key}")
            if optimum != references[key][0] or float(upper) != float(references[key][1]):
                raise ValueError(f"CSV/exact reference mismatch: {key}")
            for method in RAW_METHODS:
                keyed[key, method]["upper_bound"] = upper
    return instances, rows, metadata


def aggregate(instances, rows):
    """Collapse random trials per instance; add descriptive seed means after metrics."""
    keyed = {(r["instance_id"], r["method"]): r for r in rows}
    output = []
    for key, instance in sorted(instances.items()):
        local = {}
        methods = [m for m in RAW_METHODS if not m.startswith("random_trial")] + ["random"]
        for method in methods:
            observations = [keyed[key, f"random_trial{i}"] for i in range(10)] if method == "random" else [keyed[key, method]]
            value = statistics.mean(row["value"] for row in observations)
            reference = observations[0]
            optimum, upper = reference["reference_value"], float(reference["upper_bound"])
            greedy, portfolio = keyed[key, "greedy_density"]["value"], keyed[key, "portfolio_best"]["value"]
            record = {field: instance.get(field) for field in ("instance_id", "split", "source", "family", "n", "alpha", "source_date")}
            record.update(method=method, value=value, optimum=optimum, upper_bound=upper,
                trivial=optimum == 0, exact_gap_pct=100 * (1 - value / optimum) if optimum else None,
                exact_quality=value / optimum if optimum else None,
                normalized_quality=value / upper if optimum and upper else None,
                regret_pp_to_portfolio=100 * (portfolio - value) / optimum if optimum else None,
                relative_improvement_pct=100 * (value - greedy) / greedy if greedy else None,
                win=float(value > greedy) if optimum else None, tie=float(value == greedy) if optimum else None,
                loss=float(value < greedy) if optimum else None, feasible_rate=1.,
                _latencies=[row["latency_s"] for row in observations], observation_count=len(observations))
            local[method] = record
            output.append(record)
        for mean_name, seeds in SEED_GROUPS.items():
            seed_rows = [local[name] for name in seeds]
            record = dict(seed_rows[0], method=mean_name)
            for field in ("value", "exact_gap_pct", "exact_quality", "normalized_quality", "regret_pp_to_portfolio",
                          "relative_improvement_pct", "win", "tie", "loss", "feasible_rate"):
                values = [row[field] for row in seed_rows]
                record[field] = statistics.mean(values) if all(value is not None for value in values) else None
            record["_latencies"] = [value for row in seed_rows for value in row["_latencies"]]
            record["observation_count"] = 3
            output.append(record)
    return output


def _balanced_mean(rows, field):
    sources = defaultdict(list)
    for row in rows:
        sources[row["source"]]
        if row[field] is not None:
            sources[row["source"]].append(row[field])
    return statistics.mean(statistics.mean(values) for values in sources.values()) if sources and all(sources.values()) else None


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        dimensions = [("overall", "all"), ("source", row["source"]), ("n", str(row["n"])), ("alpha", str(row["alpha"]))]
        if row["source"] == "synthetic":
            dimensions.append(("family", row["family"]))
        for group, value in dimensions:
            groups[row["split"], group, value, row["method"]].append(row)
    results = {}
    for (split, group, value, method), rows in sorted(groups.items()):
        latencies = [latency for row in rows for latency in row["_latencies"]]
        result = dict(split=split, group=group, group_value=value, method=method, instance_count=len(rows),
            real_count=sum(row["source"] == "real" for row in rows), synthetic_count=sum(row["source"] == "synthetic" for row in rows),
            source_date_count=len({row["source_date"] for row in rows if row["source"] == "real"}),
            trivial_count=sum(row["trivial"] for row in rows), feasible_rate=_balanced_mean(rows, "feasible_rate"),
            latency_median_s=float(np.median(latencies)), latency_p95_s=float(np.percentile(latencies, 95)),
            latency_observation_count=len(latencies), observation_count=sum(row["observation_count"] for row in rows))
        for metric in ("exact_gap_pct", "exact_quality", "normalized_quality", "regret_pp_to_portfolio", "relative_improvement_pct"):
            result[metric + "_mean"] = _balanced_mean(rows, metric)
        for metric in ("win", "tie", "loss"):
            result[metric + "_rate"] = _balanced_mean(rows, metric)
        results[split, group, value, method] = result
    for (split, group, value, method), result in results.items():
        if method not in SEED_GROUPS:
            continue
        for metric in ("exact_gap_pct", "exact_quality", "normalized_quality", "regret_pp_to_portfolio", "relative_improvement_pct"):
            values = [results[split, group, value, seed][metric + "_mean"] for seed in SEED_GROUPS[method]]
            result[metric + "_seed_sd"] = statistics.stdev(values) if all(v is not None for v in values) else None
    return list(results.values())


def paired_intervals(instances, records):
    output = []
    keyed = {(row["instance_id"], row["method"]): row for row in records}
    for split in SPLITS:
        ids = sorted(key for key, instance in instances.items() if instance["split"] == split)
        if not ids:
            continue
        metadata = [instances[key] for key in ids]
        def gaps(method):
            return [keyed[key, method]["exact_gap_pct"] if keyed[key, method]["exact_gap_pct"] is not None else np.nan for key in ids]
        comparisons = {"greedy_density": ["selector_seed0", "selector_seed1", "selector_seed2", "selector_mean",
                                           "guarded_mean", "fixed_best", "portfolio_best"],
                       "fixed_best": ["selector_mean"]}
        for reference, candidates in comparisons.items():
            values = bootstrap_differences(metadata, gaps(reference), {name: gaps(name) for name in candidates},
                                           seed=BOOTSTRAP_SEED, resamples=RESAMPLES)
            for name, result in values.items():
                result.pop("samples")
                output.append(dict(split=split, candidate_method=name, reference_method=reference,
                                   metric="candidate_minus_reference_exact_gap_pp", **result))
    return output


def chosen_arm_counts(rows):
    counts = Counter((row["split"], row["method"], row["chosen_arm"]) for row in rows if row["chosen_arm"])
    totals = Counter()
    for (split, method, arm), count in counts.items():
        totals[split, method] += count
    return [dict(split=split, method=method, chosen_arm=arm, count=count, share=count / totals[split, method])
            for (split, method, arm), count in sorted(counts.items())]


def _write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row if not key.startswith("_")))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if not key.startswith("_")} for row in rows)


def _format(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(summary):
    lines = ["# Portfolio comparison tables", "", "All quality means give equal weight to each represented source. "
             "Exact gap and regret are percentage points; lower is better.", "",
             "| Split | Method | Exact gap (%) | V/U | Regret to portfolio (pp) | Median ms | p95 ms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["primary"]:
        lines.append(f"| {row['split']} | {row['method']} | {_format(row['exact_gap_pct_mean'])} | "
                     f"{_format(row['normalized_quality_mean'], 5)} | {_format(row['regret_pp_to_portfolio_mean'])} | "
                     f"{_format(row['latency_median_s'] * 1000)} | {_format(row['latency_p95_s'] * 1000)} |")
    lines += ["", "Conditional on fitted models: paired 95% data-sampling intervals below use 1,000 bootstrap "
              "resamples (seed 8101). They are distinct from training-seed sample SD in summary.csv.", "",
              "| Split | Candidate | Reference | Gap difference (pp) | 95% interval |",
              "| --- | --- | --- | ---: | --- |"]
    for row in summary["paired_bootstrap"]:
        lines.append(f"| {row['split']} | {row['candidate_method']} | {row['reference_method']} | "
                     f"{_format(row['estimate'])} | [{_format(row['ci_low'])}, {_format(row['ci_high'])}] |")
    lines += ["", "Random quality averages ten trials within each instance; its latency distribution contains all "
              "single-trial timings. Seed-mean rows pool timings descriptively and do not represent an ensemble cost.", "",
              "The portfolio maximum is the best result from the eight candidate strategies, not a globally optimal "
              "solver. It pays for all candidates. Regret may be negative for a DQN seed outside these eight arms.", ""]
    return "\n".join(lines)


def _figures(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "portfolio-report-8101", "font.size": 10})
    methods = ["random", "greedy_density", "dqn_mean", "fixed_best", "selector_mean", "guarded_mean", "portfolio_best"]
    main = {row["method"]: row for row in summary["primary"] if row["split"] == "main"}
    labels = [name for name in methods if name in main and main[name]["exact_gap_pct_mean"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), layout="constrained")
    colors = dict(zip(methods, ["#9a9a9a", "#4477aa", "#cc6677", "#aa4499", "#228833", "#66ccee", "#997700"]))
    friendly = dict(zip(methods, ["Random", "Greedy", "DQN mean", "Best fixed", "Selector mean", "Guarded mean", "Portfolio best"]))
    for axis, displayed, title in [(axes[0], labels, "All comparisons"),
                                   (axes[1], [name for name in labels if name not in {"random", "dqn_mean"}], "Greedy and portfolio methods")]:
        values = [main[name]["exact_gap_pct_mean"] for name in displayed]
        bars = axis.barh([friendly[name] for name in displayed], values, color=[colors[name] for name in displayed])
        axis.invert_yaxis()
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        axis.set(xlabel="Mean exact gap (%) — lower is better", title=title)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        if values and max(values) > 0:
            axis.set_xlim(right=max(values) * 1.23)
    fig.suptitle("Main holdout: equal source weight")
    for extension in ("png", "svg"):
        fig.savefig(out / f"main_exact_gap.{extension}", dpi=180, metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)
    selectors = [f"selector_seed{i}" for i in range(3)]
    shares = {(row["method"], row["chosen_arm"]): row["share"] for row in summary["chosen_arm_counts"] if row["split"] == "main"}
    fig, ax = plt.subplots(figsize=(9.2, 4.9), layout="constrained")
    left = np.zeros(3)
    colors = plt.get_cmap("tab10")
    for index, arm in enumerate(ARMS):
        values = np.array([shares.get((method, arm), 0.) for method in selectors])
        if not values.any():
            continue
        ax.barh(selectors, values * 100, left=left * 100, label=arm, color=colors(index))
        left += values
    ax.set(xlim=(0, 100), xlabel="Observed selection share (%)", title="Main holdout: learned selector actions")
    ax.invert_yaxis()
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower center", bbox_to_anchor=(.5, -.33), ncol=3, frameon=False, fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(out / f"selector_action_share.{extension}", dpi=180, metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)


def report(*, evaluation, out, figures=True):
    directory, out = Path(evaluation), Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    instances, rows, metadata = load_evaluation(directory)
    embedded_config = metadata.get("config", {})
    if embedded_config.get("bootstrap_seed", BOOTSTRAP_SEED) != BOOTSTRAP_SEED or embedded_config.get("bootstrap_resamples", RESAMPLES) != RESAMPLES:
        raise ValueError("evaluation bootstrap configuration differs from frozen report protocol")
    records = aggregate(instances, rows)
    table = summarize(records)
    summary = dict(schema_version=1, protocol={
        "quality": "exact_gap_pct=100*(1-V/V*); normalized_quality=V/U; exact_quality=V/V*",
        "regret": "100*(V_portfolio_best-V)/V*, in percentage points; portfolio comprises eight arms and excludes DQN seeds 1/2",
        "weighting": "equal source means separately within each split/group; synthetic-only E4",
        "random": "10-trial arithmetic mean per instance, never best-of-ten; all per-trial timings retained",
        "seed_means": "arithmetic mean of per-instance seed metrics; W/T/L rates average per-seed comparisons; latency pools original single-run observations",
        "seed_sd": "sample SD (ddof=1) of the three source-balanced model estimates, separate from sampling CI",
        "bootstrap": "paired percentile 95% sampling CI conditional on fitted models; real whole dates across cells, synthetic instances within family/n/alpha cells; equal source means each draw",
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": RESAMPLES,
        "interval_sign": "candidate exact gap minus reference exact gap; negative favors candidate",
        "feasibility": "all selected sets independently reconstructed; malformed/infeasible rows stop reporting before any output",
        "trivial": "zero-optimum cases counted but excluded from percentage denominators; undefined whole-source denominators stay undefined",
        "timing": "saved empirical single-instance construction timings; no simulated or inferred timing; portfolio and guarded costs are taken from evaluator",
        "action_counts": "observed raw-method decisions; no invented action for descriptive mean rows"},
        provenance={name: _hash(directory / name) for name in ("instances.jsonl", "results.csv", "evaluation.json")},
        evaluation=metadata, counts={split: dict(Counter(i["source"] for i in instances.values() if i["split"] == split)) for split in SPLITS},
        rows=table, primary=[r for r in table if r["group"] == "overall"],
        paired_bootstrap=paired_intervals(instances, records), chosen_arm_counts=chosen_arm_counts(rows))
    summary["provenance"]["report_code_sha256"] = _hash(Path(__file__))
    summary["provenance"]["bootstrap_code_sha256"] = _hash(Path(__file__).resolve().parents[1] / "analysis/compare.py")
    if (directory / "references.json").exists():
        summary["provenance"]["references.json"] = _hash(directory / "references.json")
    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "summary.csv", table)
    _write_csv(out / "instances.csv", records)
    _write_csv(out / "paired_bootstrap.csv", summary["paired_bootstrap"])
    _write_csv(out / "chosen_arms.csv", summary["chosen_arm_counts"])
    (out / "tables.md").write_text(_markdown(summary), encoding="utf-8")
    if figures:
        _figures(out, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                                                allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = report(evaluation=args.evaluation, out=args.out, figures=not args.no_figures)
        print(f"Saved {len(summary['primary'])} primary portfolio rows to {args.out}")
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
        print(f"Portfolio report failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
