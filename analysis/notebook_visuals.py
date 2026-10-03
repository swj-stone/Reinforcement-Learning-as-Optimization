"""Presentation figures for the classroom notebook.

Every public helper returns a Matplotlib Figure. Nothing is displayed or saved,
and global plotting settings are restored before returning. The caller owns
``plt.show()``, ``figure.savefig(...)`` and figure lifetime.
"""
from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["workflow_figure", "knapsack_example_figure", "learning_architecture_figure",
           "main_results_figure", "stress_results_figure", "training_figure"]

NAVY = "#23384D"
TEAL = "#12877F"
ORANGE = "#D77B25"
INK = "#34485B"
MUTED = "#667789"
PALE = "#F1F5F7"
GRID = "#E1E8EC"
COLORS = {"random": "#8A96A3", "greedy_density": NAVY, "dqn_mean": ORANGE,
          "selector_mean": TEAL, "fixed_best": "#726385", "guarded_mean": "#61B6AB",
          "portfolio_best": "#385D7C"}
LABELS = {"random": "Random\n10-trial mean", "greedy_density": "Density greedy", "dqn_mean": "DQN\n3-seed mean",
          "selector_mean": "Learned selector\n3-seed mean", "fixed_best": "Best fixed strategy",
          "guarded_mean": "Greedy-guarded selector", "portfolio_best": "Best of all 8 candidates"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 12, "text.color": NAVY,
         "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": NAVY,
         "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
         "savefig.facecolor": "white", "svg.hashsalt": "knapsack-classroom-figures"}


def _styled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context(STYLE):
            return function(*args, **kwargs)
    return wrapped


def _diagram(title, figsize=(15, 6.2)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([.025, .055, .95, .855])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(.5, .965, title, ha="center", va="top", fontsize=21, weight="bold", color=NAVY)
    return fig, ax


def _card(ax, x, y, width, height, title, body, *, accent=NAVY, title_size=12, body_size=10.5):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.015",
                          facecolor=PALE, edgecolor=accent, linewidth=1.25)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * .78, title, ha="center", va="center", weight="bold",
            color=accent, fontsize=title_size, linespacing=1.15)
    ax.text(x + width / 2, y + height * .33, body, ha="center", va="center", color=INK,
            fontsize=body_size, linespacing=1.4)


def _arrow(ax, start, end, *, color=MUTED, style="-", scale=13):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale,
                                linewidth=1.6, color=color, linestyle=style))


def _route(ax, points, *, color=MUTED, dashed=False):
    path = Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.4, color=color, linestyle="--" if dashed else "-"))


@_styled
def workflow_figure():
    """Show the full experimental workflow and the distinct roles of each split."""
    fig, ax = _diagram("From business records to an auditable comparison", (16, 5.8))
    steps = [
        ("Build data", "4 synthetic families\n+ real retail lots"),
        ("Freeze splits", "Source dates first\n50:50 instance mix"),
        ("Solve\nreferences", "Exact optimum V*\nFractional bound U"),
        ("Run\nbaselines", "Random × 10\nDensity greedy"),
        ("Train DQN", "Include / skip\nValidation checkpoints"),
        ("Train\nselector", "Choose 1 of 8 solvers\nSelected-action rewards"),
        ("Evaluate\n& report", "Fresh held-out cases\nPaired uncertainty"),
    ]
    width, gap, start = .122, .020, .012
    for index, (title, body) in enumerate(steps):
        x = start + index * (width + gap)
        accent = TEAL if index in (4, 5) else NAVY
        _card(ax, x, .36, width, .35, title, body, accent=accent, title_size=11.5, body_size=10)
        ax.add_patch(Circle((x + width / 2, .80), .029, color=accent))
        ax.text(x + width / 2, .80, str(index + 1), ha="center", va="center", color="white", weight="bold", fontsize=12)
        if index < len(steps) - 1:
            _arrow(ax, (x + width + .004, .535), (x + width + gap - .004, .535), scale=11)
    ax.text(.5, .272, "References measure solution quality; exact test optima are never policy inputs.",
            ha="center", va="center", fontsize=12, color=MUTED)
    for x, heading, body, accent in [
        (.027, "TRAIN", "Fit the network and learn\nempirical strategy rewards", NAVY),
        (.362, "VALIDATION", "Select checkpoints and\nthe best fixed strategy", TEAL),
        (.697, "TEST", "Freeze policies; measure once\nNo feedback or retuning", ORANGE),
    ]:
        _card(ax, x, .02, .276, .16, heading, body, accent=accent, title_size=11, body_size=10.5)
    return fig


def _selection(instance, selected):
    indices = tuple(selected)
    n = len(instance["weights"])
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Selected indices must be unique original item indices")
    weight = sum(instance["weights"][i] for i in indices)
    value = sum(instance["values"][i] for i in indices)
    if weight > instance["capacity"]:
        raise ValueError("Cannot illustrate an infeasible selection as feasible")
    return tuple(sorted(indices)), weight, value


@_styled
def knapsack_example_figure(instance, greedy_indices, optimal_indices):
    """Display a small classroom instance and two capacity-feasible selections.

    The optimal indices are supplied by the caller's exact solver; this helper
    does not solve the problem. Limit examples to 14 items so labels stay readable.
    """
    weights, values = instance["weights"], instance["values"]
    if len(weights) != len(values) or not 0 < len(weights) <= 14 or any(w <= 0 for w in weights):
        raise ValueError("Use a teaching instance with 1–14 positive-weight items and matching values")
    greedy = _selection(instance, greedy_indices)
    optimal = _selection(instance, optimal_indices)
    fig, (table_ax, bar_ax) = plt.subplots(1, 2, figsize=(13.6, 6.2), gridspec_kw={"width_ratios": [1.08, 1.15]})
    fig.subplots_adjust(top=.80, bottom=.26, left=.035, right=.965, wspace=.33)
    fig.suptitle("A 0/1 knapsack: item choices change the final value", fontsize=21, weight="bold", y=.955)
    table_ax.axis("off")
    table_ax.set_title("Candidate items", fontsize=15, weight="bold", pad=15)
    colors = [NAVY, TEAL, ORANGE, "#7A6B91", "#5F91B0", "#AA815E"]
    rows = [[str(i), f"{w:g}", f"{values[i]:g}", f"{values[i] / w:.2f}",
             "yes" if i in greedy[0] else "—", "yes" if i in optimal[0] else "—"] for i, w in enumerate(weights)]
    table = table_ax.table(cellText=rows, colLabels=["Item", "Weight", "Value", "v / w", "Greedy", "Optimal"],
                           cellLoc="center", colWidths=[.12, .16, .16, .16, .20, .20], bbox=[0, .08, 1, .83])
    table.auto_set_font_size(False)
    table.set_fontsize(11 if len(weights) <= 8 else 9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 0:
            cell.set_facecolor(colors[(r - 1) % len(colors)])
            cell.get_text().set_color("white")
        else:
            cell.set_facecolor(PALE if r % 2 else "#FAFCFD")
            if c == 4 and r - 1 in greedy[0]:
                cell.get_text().set_weight("bold")
            if c == 5 and r - 1 in optimal[0]:
                cell.get_text().set_color(TEAL)
                cell.get_text().set_weight("bold")
    capacity = instance["capacity"]
    for y, (indices, weight, value) in zip([1.05, .05], [greedy, optimal]):
        bar_ax.barh(y, capacity, color=PALE, height=.34, edgecolor=GRID)
        offset = 0
        for index in indices:
            bar_ax.barh(y, weights[index], left=offset, height=.34, color=colors[index % len(colors)], edgecolor="white", linewidth=2)
            if weights[index] >= max(capacity, 1) * .075:
                bar_ax.text(offset + weights[index] / 2, y, str(index), ha="center", va="center", color="white", weight="bold", fontsize=11)
            offset += weights[index]
        bar_ax.text(0, y + .29, f"Value = {value:g}   |   Weight = {weight:g} / {capacity:g}",
                    ha="left", va="center", fontsize=12, weight="bold")
    bar_ax.axvline(capacity, color=MUTED, linewidth=1.3, linestyle="--")
    bar_ax.set(xlim=(0, max(capacity, 1) * 1.06), ylim=(-.43, 1.62), yticks=[1.05, .05], yticklabels=["Greedy", "Exact optimum"],
               xlabel="Capacity used (stacked by original item index)")
    bar_ax.set_title("Same capacity, different combinations", fontsize=15, weight="bold", pad=15)
    bar_ax.grid(axis="x", color=GRID, alpha=.65)
    bar_ax.set_axisbelow(True)
    fig.text(.5, .105, "Each item is indivisible: accept it once or reject it. The pale area is unused capacity.",
             ha="center", fontsize=12, color=INK)
    fig.text(.5, .055, "A high value-to-weight ratio can be locally attractive and still block a better combination.",
             ha="center", fontsize=11, color=MUTED)
    return fig


@_styled
def learning_architecture_figure():
    """Contrast multi-step value learning with one-step contextual strategy choice."""
    fig, ax = _diagram("Two learning problems — two different decision levels", (15, 8.0))
    ax.plot([.505, .505], [.10, .96], color=GRID, linewidth=1.6)
    ax.text(.25, .955, "Sequential DQN", ha="center", va="top", fontsize=17, color=ORANGE, weight="bold")
    ax.text(.25, .905, "One include / skip decision per item", ha="center", fontsize=11.5, color=MUTED)
    ax.text(.76, .955, "Contextual strategy selector", ha="center", va="top", fontsize=17, color=TEAL, weight="bold")
    ax.text(.76, .905, "One complete-solver choice per instance", ha="center", fontsize=11.5, color=MUTED)
    for x, title, body, y, accent in [
        (.068, "Observed state", "Remaining items + current item\nRemaining capacity", .68, ORANGE),
        (.068, "Q-network", "Shared item encoder + mean/max pooling\nMasked Q(include) and Q(skip)", .44, ORANGE),
        (.068, "Knapsack environment", "Include consumes capacity\nMove to the next stored-order item", .20, NAVY),
        (.585, "Input-derived context", "Size, capacity fraction, density variation\nMaximum weight / capacity", .68, TEAL),
        (.585, "Contextual bandit", "Empirical action-value estimates\nChoose one strategy", .44, TEAL),
        (.585, "Execution layer: 8 candidate solvers", "7 deterministic heuristics\n+ 1 frozen DQN policy", .20, NAVY),
    ]:
        _card(ax, x, y, .345, .155, title, body, accent=accent, title_size=11.8, body_size=10.5)
    for x, action in [(.2405, "include / skip"), (.7575, "one selected strategy")]:
        _arrow(ax, (x, .676), (x, .600))
        _arrow(ax, (x, .436), (x, .360))
        ax.text(x + .012, .395, action, fontsize=10, color=MUTED, va="center")
    _route(ax, [(.418, .273), (.474, .273), (.474, .757), (.418, .757)], color=ORANGE)
    ax.text(.458, .525, "next state", rotation=90, ha="center", va="center", fontsize=10, color=ORANGE)
    _route(ax, [(.063, .274), (.026, .274), (.026, .513), (.063, .513)], color=ORANGE, dashed=True)
    ax.text(.013, .395, "item reward\n(training)", rotation=90, ha="center", va="center", fontsize=9, color=ORANGE)
    _route(ax, [(.935, .273), (.978, .273), (.978, .514), (.935, .514)], color=TEAL, dashed=True)
    ax.text(.963, .395, "reward V/U\n(training only)", rotation=90, ha="center", va="center", fontsize=9, color=TEAL)
    ax.text(.25, .106, "TRAIN: replay and masked TD updates\nTEST: fixed network; no exploration or learning",
            ha="center", va="center", fontsize=11, color=INK, linespacing=1.5)
    ax.text(.757, .106, "TRAIN: only the chosen action receives feedback\nTEST: fixed selector; no reward feedback or learning",
            ha="center", va="center", fontsize=11, color=INK, linespacing=1.5)
    fig.text(.5, .025, "The selector learns which solver to use. Its deterministic heuristic actions do not learn new parameters.",
             ha="center", fontsize=11, color=MUTED)
    return fig


def _overall(summary: "pd.DataFrame", split, methods):
    required = {"split", "group", "method", "exact_gap_pct_mean"}
    if not required <= set(summary.columns):
        raise ValueError(f"Summary is missing columns: {sorted(required - set(summary.columns))}")
    frame = summary.loc[(summary["split"] == split) & (summary["group"] == "overall")]
    rows = {}
    for method in methods:
        matches = frame.loc[frame["method"] == method]
        if len(matches) != 1:
            raise ValueError(f"Expected one overall {split}/{method} row, received {len(matches)}")
        row = matches.iloc[0]
        value = float(row["exact_gap_pct_mean"])
        if not np.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid exact gap for {split}/{method}")
        rows[method] = row
    return rows


def _gap_text(value):
    return f"{value:.4f}" if value < .1 else f"{value:.3f}"


def _gap_bars(ax, methods, rows, *, seed_sd=False):
    values = np.array([float(rows[method]["exact_gap_pct_mean"]) for method in methods])
    bars = ax.barh(range(len(methods)), values, color=[COLORS[method] for method in methods], height=.60)
    ax.set_yticks(range(len(methods)), [LABELS[method] for method in methods])
    ax.invert_yaxis()
    errors = np.zeros(len(methods))
    has_sd = np.zeros(len(methods), dtype=bool)
    if seed_sd:
        for index, method in enumerate(methods):
            sd = rows[method].get("exact_gap_pct_seed_sd", np.nan)
            if sd is not None and np.isfinite(float(sd)):
                errors[index] = float(sd)
                has_sd[index] = True
        if has_sd.any():
            ax.errorbar(values[has_sd], np.arange(len(methods))[has_sd], xerr=errors[has_sd],
                        fmt="none", ecolor=INK, capsize=4, linewidth=1.4)
    right = max(float(np.max(values + errors)), .001) * 1.26
    ax.set(xlim=(0, right), xlabel="Mean exact optimality gap (%)\nLower is better")
    for index, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + errors[index] + right * .018, bar.get_y() + bar.get_height() / 2, _gap_text(value),
                va="center", ha="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.xaxis.set_major_locator(MaxNLocator(5))


@_styled
def main_results_figure(summary: "pd.DataFrame"):
    """Show the four main methods; keep optional controls in a separate panel."""
    primary = ["random", "greedy_density", "dqn_mean", "selector_mean"]
    rows = _overall(summary, "main", primary)
    available = set(summary.loc[(summary["split"] == "main") & (summary["group"] == "overall"), "method"])
    controls = [method for method in ("fixed_best", "guarded_mean", "portfolio_best") if method in available]
    if controls:
        controls.insert(1, "selector_mean")
        rows.update(_overall(summary, "main", controls))
    fig, axes = plt.subplots(1, 2 if controls else 1, figsize=(14.8 if controls else 9.5, 6.5), squeeze=False)
    fig.subplots_adjust(left=.135 if controls else .24, right=.965, bottom=.28, top=.78, wspace=.72)
    _gap_bars(axes[0, 0], primary, rows, seed_sd=True)
    axes[0, 0].set_title("Primary comparison", fontsize=15, weight="bold", pad=16)
    if controls:
        _gap_bars(axes[0, 1], controls, rows, seed_sd=True)
        axes[0, 1].set_title("Controls — separate linear scale", fontsize=14, weight="bold", pad=16)
    fig.suptitle("Main holdout: what changed in solution quality?", fontsize=21, weight="bold", y=.955)
    fig.text(.5, .87, "Equal weight to synthetic and real-instance means; every number comes from saved measurements.",
             ha="center", fontsize=12, color=MUTED)
    fig.text(.5, .105, "Whiskers show training-seed SD where available (3 seeds), not data-sampling confidence intervals.",
             ha="center", fontsize=10.7, color=MUTED)
    fig.text(.5, .05, "Controls have different costs: the portfolio maximum runs all eight candidates; the guarded selector also pays for greedy.",
             ha="center", fontsize=10.7, color=MUTED)
    return fig


@_styled
def stress_results_figure(summary: "pd.DataFrame"):
    """Compare size and tail robustness with explicit independent linear scales."""
    methods = ["greedy_density", "dqn_mean", "selector_mean"]
    titles = ["Main holdout\n50:50 synthetic / real", "Unseen size: n = 500\n50:50 synthetic / real", "Extreme-tail stress\nSynthetic only; n = 200"]
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 5.8))
    fig.subplots_adjust(left=.10, right=.975, bottom=.31, top=.72, wspace=.79)
    for ax, split, title in zip(axes, ("main", "e3_size", "e4_tail"), titles):
        _gap_bars(ax, methods, _overall(summary, split, methods))
        ax.set_title(title, fontsize=13, weight="bold", pad=15)
    fig.suptitle("Robustness across size and distribution shifts", fontsize=21, weight="bold", y=.955)
    fig.text(.5, .12, "Panels use independent linear scales. Read the annotated percentages; colors identify the same methods throughout.",
             ha="center", fontsize=11, color=MUTED)
    fig.text(.5, .055, "Random is omitted here for clarity. DQN and selector rows average their three training seeds.",
             ha="center", fontsize=11, color=MUTED)
    return fig


@_styled
def training_figure(training: dict):
    """Plot the three bandit validation histories and validation-best fixed control."""
    runs = sorted(training["runs"], key=lambda run: run["seed"])
    if [run["seed"] for run in runs] != [0, 1, 2]:
        raise ValueError("Expected one training history for each seed 0, 1, 2")
    fixed = training["fixed_best"]
    fixed_quality = float(training["fixed_validation_scores"][fixed])
    fig, ax = plt.subplots(figsize=(13.1, 6.3))
    fig.subplots_adjust(left=.10, right=.96, bottom=.21, top=.78)
    all_values = [fixed_quality]
    for run, color in zip(runs, (NAVY, TEAL, ORANGE)):
        history = run["history"]
        epochs = np.array([point["epoch"] for point in history])
        quality = np.array([point["validation_quality"] for point in history], dtype=float)
        if not len(history) or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
            raise ValueError("Validation quality must be a finite V/U score in [0, 1]")
        ax.plot(epochs, quality, marker="o", markersize=5, linewidth=2.2, color=color, label=f"Selector seed {run['seed']}")
        chosen = [point for point in history if point["step"] == run["best_step"]]
        if chosen:
            ax.scatter([chosen[0]["epoch"]], [chosen[0]["validation_quality"]], marker="*", s=210,
                       color=color, edgecolor="white", linewidth=.8, zorder=5)
        all_values.extend(quality)
    ax.axhline(fixed_quality, color=MUTED, linestyle="--", linewidth=1.7, label=f"Best fixed control: {fixed}")
    spread = max(max(all_values) - min(all_values), .0001)
    ax.set(ylim=(max(0, min(all_values) - spread * .16), min(1, max(all_values) + spread * .34)),
           xlabel="Training epoch", ylabel="Source-balanced validation mean V/U\nHigher is better; vertical scale is zoomed")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(color=GRID, alpha=.8)
    ax.legend(loc="upper left", ncol=2, fontsize=10.5, frameon=False)
    fig.suptitle("Learning to choose a solver from experience", fontsize=21, weight="bold", y=.955)
    fig.text(.5, .866, "Validation curves for three independently trained contextual-bandit seeds", ha="center", fontsize=12, color=MUTED)
    fig.text(.5, .105, "Stars mark selected checkpoints. The dashed control is selected on the same validation set.",
             ha="center", fontsize=11, color=MUTED)
    fig.text(.5, .048, "Held-out test scores never enter these curves or checkpoint selection.", ha="center", fontsize=11, color=MUTED)
    return fig
