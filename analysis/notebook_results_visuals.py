"""Auditable distribution, condition and cost figures for ``report.ipynb``.

Public plot helpers only read compact artifacts or caller-supplied DataFrames,
return a Figure, and restore Matplotlib settings. They never save or display.
Rebuild the two compact tables from frozen evaluations with::

    python -m analysis.notebook_results_visuals --build-data --drafts DIRECTORY

That explicit CLI also records input/output SHA-256 hashes and checks its
per-instance calculations against every corresponding published main group.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analysis.notebook_visuals import COLORS, GRID, INK, MUTED, NAVY, PALE, STYLE, TEAL

__all__ = ["gap_distribution_figure", "condition_map_figure", "runtime_quality_figure",
           "stress_benchmark_figure", "verify_visual_artifacts"]

PRIMARY = ("random", "greedy_density", "dqn_mean", "selector_mean")
LABELS = {"random": "Random · mean of 10 trials", "greedy_density": "Density greedy",
          "dqn_mean": "Original DQN · mean of 3 seeds", "selector_mean": "RL selector · mean of 3 seeds",
          "fixed_best": "Best fixed strategy", "portfolio_best": "Best of all 8 candidates"}
SHORT_LABELS = {"random": "Random", "greedy_density": "Density greedy", "dqn_mean": "Original DQN",
                "selector_mean": "RL selector", "fixed_best": "Best fixed", "portfolio_best": "All 8 candidates"}
ARTIFACTS = ("visual_gap_distribution.csv", "visual_condition_map.csv")
SEQUENTIAL = LinearSegmentedColormap.from_list("gap_teal", ["#F1F7F6", "#B6D9D3", "#479E96", "#13544F"])
IMPROVEMENT = LinearSegmentedColormap.from_list("improvement", ["#D4816E", "#F8F7F4", "#12877F"])


def _styled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context({**STYLE, "font.size": 11}):
            return function(*args, **kwargs)
    return wrapped


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_visual_artifacts(root):
    """Verify the portable derived tables, without requiring private raw results."""
    bundle = Path(root) / "assets/notebook"
    manifest = json.loads((bundle / "visual_provenance.json").read_text(encoding="utf-8"))
    for name in ARTIFACTS:
        if _hash(bundle / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Derived visual artifact hash mismatch: {name}")
    return manifest


def _table(data, name):
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    verify_visual_artifacts(data)
    return pd.read_csv(Path(data) / "assets/notebook" / name)


def _overall(summary, split, methods):
    rows = summary.loc[summary["split"].eq(split) & summary["group"].eq("overall")]
    if rows["method"].duplicated().any():
        raise ValueError("Duplicate overall method rows")
    rows = rows.set_index("method").loc[list(methods)].copy()
    values = rows["exact_gap_pct_mean"].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise ValueError("Exact gaps must be finite percentages in [0, 100]")
    return rows


def _title(fig, title, subtitle):
    fig.text(.045, .963, title, fontsize=21, weight="bold", va="top", color=NAVY)
    fig.text(.045, .899, subtitle, fontsize=11.4, va="top", color=MUTED)


@_styled
def gap_distribution_figure(data):
    """Plot exact weighted empirical CDFs and counts after averaging trials/seeds.

    ``data`` is the repository root or the compact distribution DataFrame. The
    CDF is the mean of the real and synthetic empirical CDFs; it is not a CDF
    of individual trial outcomes or of an executable three-model ensemble.
    """
    frame = _table(data, "visual_gap_distribution.csv")
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(1, 2, left=.065, right=.975, top=.79, bottom=.265,
                          width_ratios=[1.36, 1], wspace=.24)
    ax, table_ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _title(fig, "How often is a solution close to the exact optimum?",
           "Main holdout · 2,400 instances · empirical distributions after trial / seed averaging")
    cell_text = []
    means = {}
    max_gap = float(frame["gap_pct"].max())
    for method in PRIMARY:
        local = frame.loc[frame.method.eq(method)].sort_values("gap_pct")
        gaps = local.gap_pct.to_numpy(float)
        counts = local[["count_real", "count_synthetic"]].to_numpy(int)
        totals = counts.sum(axis=0)
        if (totals != [1200, 1200]).any() or (gaps < 0).any() or (gaps > 100).any():
            raise ValueError("Main distributions must have 1,200 valid instances per source")
        weighted = (counts / totals).mean(axis=1)
        cdf = weighted.cumsum()
        ax.step(np.r_[0, gaps, max_gap], np.r_[0, cdf, 1], where="post",
                color=COLORS[method], linewidth=2.5, label=SHORT_LABELS[method],
                linestyle="--" if method == "random" else "-")
        total_counts = counts.sum(axis=1)
        means[method] = float(np.dot(gaps, weighted))
        counts_at = [int(total_counts[gaps <= 1e-12].sum()),
                     int(total_counts[gaps <= 1 + 1e-12].sum()),
                     int(total_counts[gaps > 5 + 1e-12].sum())]
        cell_text.append([f"{value:,}\n{value / 24:.1f}%" for value in counts_at])
    ax.set(xscale="symlog", xlim=(0, min(100, max_gap * 1.03)), ylim=(0, 1.025),
           xlabel="Exact optimality gap (%)  ·  lower is better", ylabel="Share of instances at or below gap")
    ax.set_xscale("symlog", linthresh=.1, linscale=.45)
    ax.set_xticks([0, .1, .5, 1, 5, 20, 60], ["0", "0.1", "0.5", "1", "5", "20", "60"])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axvspan(0, 1, color=TEAL, alpha=.045)
    ax.axvline(1, color=TEAL, alpha=.4, linewidth=1.1, linestyle=":")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Higher curve = more near-optimal instances", fontsize=12.5, weight="bold", pad=16)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    table_ax.axis("off")
    table_ax.set_title("Reliability counts out of 2,400", fontsize=12.5, weight="bold", pad=16)
    table = table_ax.table(cellText=[[SHORT_LABELS[method], *row] for method, row in zip(PRIMARY, cell_text)],
                          colLabels=["Method", "Exact", "Gap ≤ 1%", "Gap > 5%"], cellLoc="center",
                          bbox=[0, .16, 1, .82], colWidths=[.32, .225, .23, .225])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(3)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set(color="white", weight="bold", fontsize=10.5)
        elif c == 0:
            cell.set_facecolor("white")
            cell.get_text().set(color=COLORS[PRIMARY[r - 1]], weight="bold", fontsize=10.5)
        else:
            cell.set_facecolor("#E4F2EF" if r == 4 else PALE)
    table_ax.text(.5, .06, "Counts refer to averaged instance gaps.\nThey are descriptive; no confidence bands implied.",
                  ha="center", va="center", fontsize=10.3, color=MUTED)
    for index, method in enumerate(PRIMARY):
        x = .068 + index * .236
        fig.text(x, .171, SHORT_LABELS[method], color=COLORS[method], fontsize=11, weight="bold")
        fig.text(x, .127, f"{means[method]:.3f}%", color=COLORS[method], fontsize=23, weight="bold")
        fig.text(x + .105, .128, "mean gap", color=MUTED, fontsize=10)
    fig.text(.065, .057, "CDF gives equal weight to real and synthetic sources (1,200 each). Random: 10-trial mean; DQN / selector: 3-seed mean.",
             fontsize=10, color=MUTED)
    fig.text(.065, .025, "Gap axis is linear from 0 to 0.1%, then logarithmic. Exact means zero gap after averaging; seed means are not deployed ensembles.",
             fontsize=10, color=MUTED)
    return fig


def _annotated_matrix(ax, values, *, cmap, norm, formatter, row_labels=None, column_labels=None):
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    for (r, c), value in np.ndenumerate(values):
        rgba = image.cmap(image.norm(value))
        luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
        ax.text(c, r, formatter(value), ha="center", va="center", fontsize=12,
                weight="bold", color="white" if luminance < .56 else NAVY)
    ax.set_xticks(np.arange(values.shape[1]), column_labels or [])
    ax.set_yticks(np.arange(values.shape[0]), row_labels or [])
    ax.set_xticks(np.arange(values.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


@_styled
def condition_map_figure(data):
    """Joint source × alpha × size map; each cell contains 100 paired instances."""
    frame = _table(data, "visual_condition_map.csv")
    keys = [(source, alpha, n) for source in ("synthetic", "real")
            for alpha in (.25, .5, .75) for n in (20, 50, 100, 200)]
    frame = frame.set_index(["source", "alpha", "n"]).loc[keys]
    if len(frame) != 24 or not frame.instance_count.eq(100).all():
        raise ValueError("Condition map requires 24 cells of 100 matched instances")
    improvement = frame.improvement_pp.to_numpy(float).reshape(6, 4)
    gaps = frame.selector_gap_pct.to_numpy(float).reshape(6, 4)
    if (gaps < 0).any() or not np.isfinite(gaps).all():
        raise ValueError("Invalid selector gaps")
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.0))
    fig.subplots_adjust(left=.145, right=.945, bottom=.23, top=.79, wspace=.27)
    _title(fig, "Where does the selector help — and where does error remain?",
           "Main holdout · all 24 source × capacity × size cells · 100 matched instances in every cell")
    rows = [f"{source}\nα = {alpha:.2f}" for source in ("Synthetic", "Real") for alpha in (.25, .5, .75)]
    labels = ["20", "50", "100", "200"]
    limit = max(float(np.abs(improvement).max()), .01)
    image1 = _annotated_matrix(axes[0], improvement, cmap=IMPROVEMENT, norm=Normalize(-limit, limit),
                               formatter=lambda value: f"{value:+.2f}", row_labels=rows, column_labels=labels)
    image2 = _annotated_matrix(axes[1], gaps, cmap=SEQUENTIAL, norm=Normalize(0, max(float(gaps.max()), .01)),
                               formatter=lambda value: f"{value:.3f}", row_labels=[""] * 6, column_labels=labels)
    for ax, title in zip(axes, ("Gap reduction vs greedy (percentage points)", "Remaining selector gap (%)")):
        ax.set_title(title, fontsize=12.5, pad=18, weight="bold")
        ax.set_xlabel("Number of items, n", labelpad=12)
        ax.axhline(2.5, color="white", linewidth=8)
    axes[0].text(-.35, 1.035, "Source / capacity fraction", transform=axes[0].transAxes,
                 fontsize=10, color=MUTED)
    for index, (image, label) in enumerate(((image1, "Positive = selector improves on greedy"),
                                           (image2, "Lower = closer to the exact optimum"))):
        cax = fig.add_axes([.16 + index * .44, .125, .33, .018])
        colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        colorbar.set_label(label, fontsize=10, labelpad=6)
        colorbar.ax.tick_params(labelsize=9)
        colorbar.outline.set_visible(False)
    fig.text(.145, .035, "Each cell averages the same instances: greedy gap − mean selector gap over 3 fitted seeds. No cell is selected by performance.",
             fontsize=10, color=MUTED)
    return fig


@_styled
def stress_benchmark_figure(summary):
    """Compare three holdouts on a common gap color scale, with control rows."""
    methods = (*PRIMARY, "fixed_best", "portfolio_best")
    splits = ("main", "e3_size", "e4_tail")
    values = np.column_stack([_overall(summary, split, methods).exact_gap_pct_mean for split in splits])
    fig, ax = plt.subplots(figsize=(13.6, 7.8))
    fig.subplots_adjust(left=.23, right=.84, bottom=.20, top=.77)
    _title(fig, "Does solution quality survive a change in the test conditions?",
           "Every cell is a mean exact gap (%) · common color scale across all methods and holdouts")
    image = _annotated_matrix(ax, values, cmap=SEQUENTIAL, norm=LogNorm(.005, 50),
                               formatter=lambda v: f"{v:.4f}" if v < .1 else f"{v:.3f}",
                               row_labels=[SHORT_LABELS[m] for m in methods],
                               column_labels=["Main\n2,400 instances", "Unseen n = 500\n600 instances", "Extreme-tail stress\n300 instances"])
    ax.xaxis.tick_top()
    ax.axhline(3.5, color="white", linewidth=12)
    for index, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(COLORS[methods[index]])
        tick.set_weight("bold")
    ax.text(-.03, 1 - 4 / 6, "Controls", transform=ax.transAxes, fontsize=9, color=MUTED,
            va="center", ha="right", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    cax = fig.add_axes([.877, .26, .018, .45])
    colorbar = fig.colorbar(image, cax=cax, ticks=[.01, .1, 1, 10, 50])
    colorbar.ax.set_yticklabels(["0.01", "0.1", "1", "10", "50"])
    colorbar.set_label("Mean gap (%) · log color scale", labelpad=12, fontsize=10)
    colorbar.outline.set_visible(False)
    fig.text(.23, .135, "Main and size-shift means: 50% real + 50% synthetic. Tail stress: synthetic only, n = 200.", fontsize=10.5, color=MUTED)
    fig.text(.23, .092, "Random uses 10-trial instance means; DQN / selector average 3 seeds. Read the numbers: colors are logarithmic.", fontsize=10.3, color=MUTED)
    fig.text(.23, .049, "Controls: fixed strategy selected on validation; all-candidate row runs eight solvers. Neither row is the RL selector score.", fontsize=10.3, color=MUTED)
    return fig


@_styled
def runtime_quality_figure(summary):
    """Rank measured median–p95 latency intervals beside quality and call scope."""
    methods = (*PRIMARY, "fixed_best", "portfolio_best")
    rows = _overall(summary, "main", methods).sort_values("latency_median_s")
    median = rows.latency_median_s.to_numpy(float) * 1000
    p95 = rows.latency_p95_s.to_numpy(float) * 1000
    if not np.isfinite(median).all() or not np.isfinite(p95).all() or (median <= 0).any() or (p95 < median).any():
        raise ValueError("Expected positive measured median and p95 latencies")
    fig = plt.figure(figsize=(16, 7.5))
    ax = fig.add_axes([.20, .275, .38, .50])
    info = fig.add_axes([.61, .275, .36, .50], sharey=ax)
    _title(fig, "What computation is paid for each solution?",
           "Main holdout · warmed local CPU calls · median-to-p95 intervals, sorted by median")
    y = np.arange(len(rows))
    for i, (method, row) in enumerate(rows.iterrows()):
        color = COLORS[method]
        ax.hlines(i, median[i], p95[i], linewidth=5, color=color, alpha=.52)
        ax.plot(median[i], i, "o", color=color, markersize=9)
        ax.plot(p95[i], i, "|", color=color, markersize=15, markeredgewidth=2)
        for axis in (ax, info):
            if i % 2 == 0:
                axis.axhspan(i - .46, i + .46, color=PALE, zorder=-5)
        info.text(.015, i, f"{median[i]:.3f} → {p95[i]:.3f}", va="center", fontsize=11, color=INK)
        info.text(.37, i, f"{row.exact_gap_pct_mean:.3f}%", va="center", fontsize=12, color=color, weight="bold")
    scopes = {"random": "1 randomized trial", "greedy_density": "1 density-sorted greedy pass",
              "dqn_mean": "1 sequential DQN policy", "selector_mean": "Context + 1 selected solver",
              "fixed_best": "1 fixed exchange-repair solver", "portfolio_best": "Run 8 solvers; retain best"}
    for i, method in enumerate(rows.index):
        info.text(.60, i, scopes[method], va="center", fontsize=10.4, color=INK)
    ax.set(xscale="log", xlim=(median.min() * .65, p95.max() * 1.5),
           ylim=(len(rows) - .5, -.65), yticks=y, yticklabels=[SHORT_LABELS[m] for m in rows.index],
           xlabel="Measured construction time (ms, logarithmic)")
    ax.set_xticks([.01, .1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=15)
    for label, method in zip(ax.get_yticklabels(), rows.index):
        label.set_color(COLORS[method])
        label.set_weight("bold")
    info.set(xlim=(0, 1.12))
    info.axis("off")
    for x, title in ((.015, "Median → p95, ms"), (.37, "Mean gap"), (.60, "Work in each timed call")):
        info.text(x, -.87, title, weight="bold", fontsize=10.8, color=NAVY)
    ax.legend(handles=[Line2D([0], [0], color=NAVY, marker="o", linestyle="none", label="Median"),
                       Line2D([0], [0], color=NAVY, marker="|", markersize=13, linestyle="none", label="95th percentile")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.text(.07, .165, "Timing includes context and solution construction; excludes I/O, checkpoint loading, exact DP, and the later solution audit.",
             fontsize=10.5, color=MUTED)
    fig.text(.07, .116, "Random quality averages 10 trials; its time is per trial. Seed-mean rows pool measured single-policy calls; they are not ensemble costs.",
             fontsize=10.5, color=MUTED)
    fig.text(.07, .067, "One PyTorch CPU thread; local single-pass measurements. Intervals describe observed call latency, not confidence intervals or training cost.",
             fontsize=10.5, color=MUTED)
    return fig


def build_visual_artifacts(root):
    """Explicit data preparation; no model runs and no mutation of frozen files."""
    root = Path(root)
    directory, bundle = root / "results/portfolio-v2-evaluation", root / "assets/notebook"
    metadata = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    inputs = [directory / "results.csv", directory / "instances.jsonl", directory / "evaluation.json",
              root / "portfolio_report_tables/summary.csv"]
    for name in ("results.csv", "instances.jsonl"):
        if _hash(directory / name) != metadata["artifacts"][name]:
            raise ValueError(f"Frozen evaluation hash mismatch: {name}")
    fields = ("instance_id", "split", "source", "family", "n", "alpha")
    with (directory / "instances.jsonl").open(encoding="utf-8") as handle:
        instances = pd.DataFrame([{key: record.get(key) for key in fields}
                                  for line in handle if line.strip() for record in [json.loads(line)]])
    instances = instances.loc[instances.split.eq("main")].set_index("instance_id")
    if len(instances) != 2400 or instances.index.duplicated().any():
        raise ValueError("Expected 2,400 unique main instances")
    raw = pd.read_csv(directory / "results.csv", usecols=["instance_id", "split", "method", "value", "reference_value"])
    raw = raw.loc[raw.split.eq("main")]
    if raw.groupby("instance_id").reference_value.nunique().ne(1).any():
        raise ValueError("Inconsistent exact reference")
    values = raw.pivot(index="instance_id", columns="method", values="value").reindex(instances.index)
    optimum = raw.groupby("instance_id").reference_value.first().reindex(instances.index)
    if optimum.le(0).any() or values.isna().any().any() or values.gt(optimum, axis="index").any().any():
        raise ValueError("Invalid or incomplete exact-reference results")
    instance_gaps = instances.copy()
    instance_gaps["random"] = 100 * (1 - values[[f"random_trial{i}" for i in range(10)]].mean(axis=1) / optimum)
    instance_gaps["greedy_density"] = 100 * (1 - values.greedy_density / optimum)
    for method in ("dqn", "selector"):
        instance_gaps[f"{method}_mean"] = (100 * (1 - values[[f"{method}_seed{i}" for i in range(3)]].div(optimum, axis=0))).mean(axis=1)
    summary = pd.read_csv(root / "portfolio_report_tables/summary.csv")
    checks = []
    for method in PRIMARY:
        expected_rows = summary.loc[summary.split.eq("main") & summary.method.eq(method)]
        for row in expected_rows.itertuples():
            local = instance_gaps
            if row.group != "overall":
                expected = float(row.group_value) if row.group in ("n", "alpha") else row.group_value
                local = local.loc[local[row.group].eq(expected)]
            measured = float(local.groupby("source")[method].mean().mean())
            difference = abs(measured - row.exact_gap_pct_mean)
            if len(local) != row.instance_count or difference > 1e-11:
                raise ValueError(f"Published summary disagreement for {method}/{row.group}/{row.group_value}")
            checks.append(difference)
    distributions = []
    statistics = {}
    for method in PRIMARY:
        counts = instance_gaps.groupby([method, "source"]).size().unstack(fill_value=0).reindex(columns=["real", "synthetic"], fill_value=0)
        local = counts.rename_axis("gap_pct").reset_index().rename(columns={"real": "count_real", "synthetic": "count_synthetic"})
        local.insert(0, "method", method)
        distributions.append(local)
        series = instance_gaps[method]
        statistics[method] = {"mean_gap_pct": float(instance_gaps.groupby("source")[method].mean().mean()),
                              "instance_count": len(series), "exact_count": int(series.le(1e-12).sum()),
                              "gap_at_most_1pct_count": int(series.le(1 + 1e-12).sum()),
                              "gap_above_5pct_count": int(series.gt(5 + 1e-12).sum()),
                              "max_gap_pct": float(series.max())}
    distribution = pd.concat(distributions, ignore_index=True)
    instance_gaps["improvement_pp"] = instance_gaps.greedy_density - instance_gaps.selector_mean
    conditions = instance_gaps.groupby(["source", "alpha", "n"], sort=True).agg(
        instance_count=("selector_mean", "size"), selector_gap_pct=("selector_mean", "mean"),
        greedy_gap_pct=("greedy_density", "mean"), improvement_pp=("improvement_pp", "mean")).reset_index()
    bundle.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(ARTIFACTS, (distribution, conditions)):
        frame.to_csv(bundle / name, index=False, float_format="%.16g", lineterminator="\n")
    manifest = {"schema_version": 1, "scope": "main follow-up holdout", "split": "main", "instance_count": 2400,
                "source_counts": {"real": 1200, "synthetic": 1200},
                "metric": "100 * (1 - value / exact optimum)",
                "aggregation": "Random averages ten trial values per instance before gap; DQN and selector average three seed gaps per instance. CDF averages empirical source CDFs equally.",
                "condition_map": "24 exhaustive source x alpha x n cells, 100 paired instances each; improvement is greedy gap minus 3-seed mean selector gap.",
                "cdf_interpretation": "Distribution of per-instance means, not individual trial/seed outcomes; exact means all averaged runs attained exact value. Counts are descriptive, not independent Bernoulli confidence intervals.",
                "source_files": {str(path.relative_to(root)).replace("\\", "/"): _hash(path) for path in inputs},
                "build_code_sha256": _hash(Path(__file__)),
                "validation": {"published_summary_rows_checked": len(checks), "max_abs_difference_pp": max(checks),
                               "condition_cell_count": len(conditions), "condition_cell_size": 100},
                "method_statistics": statistics,
                "artifacts": {name: {"sha256": _hash(bundle / name), "rows": len(frame), "bytes": (bundle / name).stat().st_size}
                              for name, frame in zip(ARTIFACTS, (distribution, conditions))}}
    (bundle / "visual_provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_visual_artifacts(root)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--drafts", type=Path)
    args = parser.parse_args(argv)
    manifest = build_visual_artifacts(args.root) if args.build_data else verify_visual_artifacts(args.root)
    if args.drafts:
        args.drafts.mkdir(parents=True, exist_ok=True)
        summary = pd.read_csv(args.root / "portfolio_report_tables/summary.csv")
        for name, figure in (("gap_distribution", gap_distribution_figure(args.root)),
                             ("condition_map", condition_map_figure(args.root)),
                             ("stress_benchmark", stress_benchmark_figure(summary)),
                             ("runtime_quality", runtime_quality_figure(summary))):
            figure.savefig(args.drafts / f"{name}.png", dpi=145)
            plt.close(figure)
    print(json.dumps({"validation": manifest["validation"], "method_statistics": manifest["method_statistics"],
                      "artifacts": manifest["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
