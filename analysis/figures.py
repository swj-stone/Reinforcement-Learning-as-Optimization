"""Create report-ready static figures from formal comparison summaries.

The module intentionally consumes only saved formal artifacts.  It never trains a
model, evaluates a checkpoint, or fabricates a missing experimental result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SUMMARY_FIELDS = {
    "suite",
    "group",
    "group_value",
    "method",
    "seed",
    "normalized_quality_mean",
    "exact_gap_pct_mean",
    "latency_median_s",
}
VALIDATION_FIELDS = {"step", "score", "new_best"}
METHOD_COLORS = {"random": "#8c8c8c", "greedy": "#2c7fb8", "dqn": "#d95f02"}
METHOD_LABELS = {"random": "Random (10-trial mean)", "greedy": "Greedy", "dqn": "DQN"}
FIGURE_STEMS = (
    "validation_learning_curves",
    "e1_exact_gap_by_source",
    "e2_exact_gap_by_size",
    "stress_exact_gap",
    "quality_vs_runtime",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value in ("", None):
        raise ValueError(f"missing {field} in summary row: {row}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field}={value!r} in summary row") from exc


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def load_summary(path: Path) -> list[dict[str, str]]:
    """Load the comparison agent's long-form summary without coercing its labels."""
    return _read_csv(path, SUMMARY_FIELDS)


def load_validation(run_dirs: Iterable[Path]) -> dict[str, list[dict[str, str]]]:
    curves: dict[str, list[dict[str, str]]] = {}
    for run_dir in run_dirs:
        path = Path(run_dir) / "validation.csv"
        rows = _read_csv(path, VALIDATION_FIELDS)
        curves[Path(run_dir).name] = rows
    if not curves:
        raise ValueError("at least one --runs directory is required")
    return curves


def _mean_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Use DQN's descriptive seed mean; never present it as an ensemble policy."""
    out = []
    for row in rows:
        if row["method"] != "dqn" or row["seed"] == "mean":
            out.append(row)
    return out


def _rows_for(rows: list[dict[str, str]], suite: str, group: str) -> list[dict[str, str]]:
    selected = _mean_rows(r for r in rows if r["suite"] == suite and r["group"] == group)
    if not selected:
        raise ValueError(f"summary contains no {suite}/{group} rows needed for a figure")
    return selected


def _by_value_method(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["method"] not in METHOD_LABELS:
            continue
        if row["method"] == "dqn" and row["seed"] not in ("mean", ""):
            continue
        grouped[row["group_value"]][row["method"]] = row
    return grouped


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> list[Path]:
    outputs = [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"]
    fig.savefig(outputs[0], dpi=150, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    plt.close(fig)
    return outputs


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def plot_validation_learning_curves(curves: dict[str, list[dict[str, str]]], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    for name, rows in sorted(curves.items()):
        steps = [_float(row, "step") for row in rows]
        scores = [_float(row, "score") for row in rows]
        line = ax.plot(steps, scores, marker="o", markersize=3.4, linewidth=1.8, label=name)[0]
        best = max(rows, key=lambda row: _float(row, "score"))
        ax.scatter([_float(best, "step")], [_float(best, "score")],
                   marker="*", s=130, color=line.get_color(), edgecolor="white", zorder=3)
    ax.plot([], [], marker="*", linestyle="none", markersize=10, color="#333333", label="Selected checkpoint")
    ax.set(title="Validation learning curves", xlabel="Environment transitions", ylabel="Validation V/U")
    ax.legend(title="Run", frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, "validation_learning_curves", out_dir)


def plot_e1_source_gaps(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    grouped = _by_value_method(_rows_for(rows, "E1", "source"))
    sources = [source for source in ("synthetic", "real") if source in grouped]
    if not sources:
        raise ValueError("E1/source summary has no synthetic or real rows")
    fig, axes = plt.subplots(1, len(sources), figsize=(5.2 * len(sources), 4.6), layout="constrained", squeeze=False, sharey=True)
    for ax, source in zip(axes[0], sources):
        methods = grouped[source]
        labels = [METHOD_LABELS[m] for m in ("random", "greedy", "dqn") if m in methods]
        values = [_float(methods[m], "exact_gap_pct_mean") for m in ("random", "greedy", "dqn") if m in methods]
        colors = [METHOD_COLORS[m] for m in ("random", "greedy", "dqn") if m in methods]
        ax.bar(labels, values, color=colors, zorder=2)
        dqn_points = [
            _float(row, "exact_gap_pct_mean")
            for row in rows
            if row["suite"] == "E1" and row["group"] == "source" and row["group_value"] == source
            and row["method"] == "dqn" and row["seed"] not in ("", "mean")
        ]
        if dqn_points and "dqn" in methods:
            dqn_x = labels.index(METHOD_LABELS["dqn"])
            ax.scatter([dqn_x] * len(dqn_points), dqn_points, color="#4d4d4d", s=25, zorder=3, label="DQN seeds")
            ax.legend(frameon=False, fontsize=8)
        ax.set(title=f"E1: {source}", ylabel="Mean exact gap (%)")
        ax.tick_params(axis="x", labelrotation=18)
        _style(ax)
    fig.suptitle("Source-specific exact optimality gap\nDQN mean is descriptive across three policies")
    return _save(fig, "e1_exact_gap_by_source", out_dir)


def plot_e2_gap_by_size(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    data = _by_value_method(_rows_for(rows, "E1", "source_n"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained", sharey=True)
    for ax, source in zip(axes, ("synthetic", "real")):
        sizes = sorted(int(key.split("|")[1]) for key in data if key.startswith(source + "|"))
        if sizes != [20, 50, 100, 200]:
            raise ValueError(f"incomplete E1/source_n groups for {source}")
        for method in ("random", "greedy", "dqn"):
            values = [_float(data[f"{source}|{size}"][method], "exact_gap_pct_mean") for size in sizes]
            label = METHOD_LABELS[method] + (" (seed mean)" if method == "dqn" else "")
            ax.plot(sizes, values, marker="o", linewidth=1.9, color=METHOD_COLORS[method], label=label)
            if method == "dqn":
                per_size = [[_float(r, "exact_gap_pct_mean") for r in rows
                             if r["suite"] == "E1" and r["group"] == "source_n"
                             and r["group_value"] == f"{source}|{size}" and r["method"] == "dqn"
                             and r["seed"] in ("0", "1", "2")] for size in sizes]
                if any(len(v) != 3 for v in per_size):
                    raise ValueError("missing DQN seed for size plot")
                ax.fill_between(sizes, [min(v) for v in per_size], [max(v) for v in per_size],
                                color=METHOD_COLORS[method], alpha=.14, label="DQN seed range")
        ax.set(title="Synthetic" if source == "synthetic" else "Real-derived", xlabel="Items per instance")
        ax.set_xticks(sizes)
        _style(ax)
    axes[0].set_ylabel("Mean exact gap (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("E2: gap by size — shaded range is seed variation, not a confidence interval")
    return _save(fig, "e2_exact_gap_by_size", out_dir)


def _stress_rows(rows: list[dict[str, str]], suite: str) -> dict[str, dict[str, dict[str, str]]]:
    candidates = _mean_rows(row for row in rows if row["suite"] == suite)
    preferred = [row for row in candidates if row["group"] == "overall"]
    if len(preferred) != 3 or any(row["group_value"] != "all" for row in preferred):
        raise ValueError(f"missing complete {suite}/overall/all summary")
    return _by_value_method(preferred)


def plot_stress_gaps(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    panels = [("E4", "E4: extreme synthetic skew\nn=200; 1% rare-index sets"),
              ("E4b", "E4b: real tail subset\n691 instances already in E1")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained", sharey=True)
    families = _by_value_method(_rows_for(rows, "E1", "family"))
    names = [name for name in ("uniform", "normal", "correlated", "skewed") if name in families]
    if len(names) != 4:
        raise ValueError("missing synthetic family summary")
    for index, method in enumerate(("random", "greedy", "dqn")):
        axes[0].bar([i + (index - 1) * .25 for i in range(4)],
                    [_float(families[name][method], "exact_gap_pct_mean") for name in names],
                    width=.24, color=METHOD_COLORS[method], label="DQN" if method == "dqn" else method.capitalize(), zorder=2)
    axes[0].set_xticks(range(4), [name.capitalize() for name in names], rotation=22)
    axes[0].set(title="E1: synthetic families\nAll main sizes and capacity ratios", ylabel="Mean exact gap (%)")
    axes[0].legend(frameon=False, fontsize=8)
    _style(axes[0])
    for ax, (suite, title) in zip(axes[1:], panels):
        data = _stress_rows(rows, suite)
        if not data:
            raise ValueError(f"summary contains no {suite} rows needed for stress figure")
        methods = data["all"]
        present = [method for method in ("random", "greedy", "dqn") if method in methods]
        bars = ax.bar(
            ["DQN" if method == "dqn" else method.capitalize() for method in present],
            [_float(methods[method], "exact_gap_pct_mean") for method in present],
            color=[METHOD_COLORS[method] for method in present], zorder=2,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.margins(y=.15)
        ax.set(title=title, ylabel="Mean exact gap (%)")
        ax.tick_params(axis="x", labelrotation=18)
        _style(ax)
    fig.suptitle("Distribution and tail effects — DQN bars average three policies")
    return _save(fig, "stress_exact_gap", out_dir)


def plot_quality_vs_runtime(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    candidates = _mean_rows(row for row in rows if row["suite"] == "E1" and row["group"] == "overall")
    if not candidates:
        raise ValueError("summary contains no E1/overall rows needed for quality-runtime figure")
    fig, ax = plt.subplots(figsize=(6.7, 4.6), layout="constrained")
    for row in candidates:
        method = row["method"]
        if method not in METHOD_LABELS:
            continue
        runtime = _float(row, "latency_median_s")
        if runtime <= 0:
            raise ValueError(f"nonpositive median latency for {method}: {runtime}")
        gap = _float(row, "exact_gap_pct_mean")
        ax.scatter(runtime, gap, s=85, color=METHOD_COLORS[method], zorder=3)
        text = METHOD_LABELS[method] + ("\n(descriptive mean)" if method == "dqn" else "")
        offset = (-8, 8) if method == "dqn" else (8, 8)
        ax.annotate(text, (runtime, gap), xytext=offset, ha="right" if method == "dqn" else "left",
                    textcoords="offset points", fontsize=8)
    ax.set(title="E1 quality versus runtime", xlabel="Median single-instance latency (s, log scale)", ylabel="Mean exact gap (%)")
    ax.set_xscale("log")
    ax.margins(x=.18, y=.25)
    ax.set_ylim(bottom=0)
    _style(ax)
    return _save(fig, "quality_vs_runtime", out_dir)


def generate(analysis_dir: Path, run_dirs: Iterable[Path], out_dir: Path, *, force: bool = False) -> dict:
    """Generate all five figure pairs and a checksum-based provenance record."""
    analysis_dir, out_dir = Path(analysis_dir), Path(out_dir)
    summary_path = analysis_dir / "summary.csv"
    summary = load_summary(summary_path)
    curves = load_validation(run_dirs)
    expected = [out_dir / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "svg")]
    expected.append(out_dir / "figures.json")
    exists = [path for path in expected if path.exists()]
    if exists and not force:
        raise FileExistsError(f"refusing to overwrite: {', '.join(str(path) for path in exists)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs += plot_validation_learning_curves(curves, out_dir)
    outputs += plot_e1_source_gaps(summary, out_dir)
    outputs += plot_e2_gap_by_size(summary, out_dir)
    outputs += plot_stress_gaps(summary, out_dir)
    outputs += plot_quality_vs_runtime(summary, out_dir)
    inputs = {str(summary_path): _sha256(summary_path)}
    for run_dir in run_dirs:
        path = Path(run_dir) / "validation.csv"
        inputs[str(path)] = _sha256(path)
    record = {
        "generator_sha256": _sha256(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "inputs_sha256": inputs,
        "outputs_sha256": {str(path): _sha256(path) for path in outputs},
        "schema": "formal-comparison-summary-v1",
        "figures": [path.name for path in outputs],
    }
    provenance = out_dir / "figures.json"
    provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate static figures from formal comparison outputs")
    parser.add_argument("--analysis", type=Path, required=True, help="formal comparison directory")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="DQN run directories")
    parser.add_argument("--out", type=Path, required=True, help="output figure directory")
    parser.add_argument("--force", action="store_true", help="overwrite existing figure files")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    record = generate(args.analysis, args.runs, args.out, force=args.force)
    print(f"Generated {len(record['figures']) // 2} figure pairs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
