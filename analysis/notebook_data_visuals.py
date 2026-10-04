"""High-information data figures for the seminar report notebook.

The cleaning and split plots use retained audit/manifest data.  Distribution
fingerprints use a fixed illustrative generator draw and are never presented as
test or evaluation evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch

from experiment.data.generate.families import generate_items


NAVY = "#183a5a"
TEAL = "#147d78"
ORANGE = "#e67e22"
PURPLE = "#7652a5"
INK = "#1f2933"
PALE = "#eaf1f5"
FAMILY_COLORS = {"uniform": NAVY, "normal": TEAL, "correlated": ORANGE, "skewed": PURPLE}

NARRATIONS = {
    "cleaning": (
        "Most UCI Online Retail rows survived the documented cleaning pipeline. "
        "The reason view uses a deliberately magnified scale because removals are only a small share of input."
    ),
    "splits": (
        "Dates determine chronological train, validation, and test partitions; lots are a separate count. "
        "The complete-instance tiles show the 50:50 synthetic/real mixture used by each split."
    ),
    "fingerprints": (
        "These are fixed illustrative draws from the four generators, not held-out observations or performance results. "
        "They show how the spread of value density changes the information available to a density-greedy rule."
    ),
}


def load_json(path: Path) -> dict:
    """Read a retained JSON artifact for use in a notebook cell."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finish(fig: plt.Figure) -> plt.Figure:
    fig.set_facecolor("white")
    return fig


def _styled(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                             'text.color': INK, 'axes.titlecolor': INK}):
            return function(*args, **kwargs)
    return wrapped


@_styled
def figure_cleaning_stream(audit: Mapping) -> plt.Figure:
    """Show the retained cleaning stream and magnified first-hit removal reasons."""
    raw = int(audit["raw_rows"])
    kept = int(audit["kept_rows"])
    removed = raw - kept
    reasons = audit["cleaning"]["dropped_by_reason"]
    if removed <= 0 or sum(int(value) for value in reasons.values()) != removed:
        raise ValueError("cleaning audit does not reconcile raw, kept, and removed counts")

    fig = plt.figure(figsize=(12, 6.8))
    fig.text(.05, .955, 'Follow the records: keep valid lots, explain every removal',
             fontsize=19, weight='bold')
    fig.text(.05, .9, f'{raw:,} source rows → {kept:,} usable lots + {removed:,} removed rows',
             color='#667789', fontsize=12)
    flow = fig.add_axes([.055, .56, .89, .23])
    flow.set(xlim=(0, 1), ylim=(0, 1)); flow.axis('off')
    kept_share = kept / raw
    flow.add_patch(Rectangle((0, .3), kept_share, .47, color=TEAL, linewidth=0))
    flow.add_patch(Rectangle((kept_share, .3), 1-kept_share, .47, color=ORANGE, linewidth=0))
    flow.text(kept_share / 2, .54, f'RETAINED  {kept:,}  |  {kept_share:.2%}',
              ha='center', va='center', color='white', weight='bold', fontsize=14)
    flow.text(1, .92, f'Removed {removed/raw:.2%}', color=ORANGE, ha='right', fontsize=11, weight='bold')
    detail = fig.add_axes([.36, .13, .585, .34])
    detail.set(xlim=(0, 1), ylim=(0, 1)); detail.axis('off')
    labels = {'cancellation_invoice': 'Cancellations', 'exact_duplicate_record': 'Exact duplicates',
              'nonpositive_quantity_or_price': 'Nonpositive\nquantity / price', 'invalid_stock_code': 'Invalid stock code'}
    nonzero = sorted([(k, int(v)) for k, v in reasons.items() if v], key=lambda kv: -kv[1])
    # Binary slice layout: rectangle areas equal each category's share of removals.
    boxes = []
    x = 0.
    for k, value in nonzero[:2]:
        width = value / removed
        boxes.append((k, value, x, 0., width, 1.)); x += width
    remaining_width = 1-x
    y = 0.
    for k, value in nonzero[2:]:
        height = value / removed / remaining_width
        boxes.append((k, value, x, y, remaining_width, height)); y += height
    assert abs(sum(w*h for _, _, _, _, w, h in boxes)-1) < 1e-12
    for (k, value, x, y, w, h), color in zip(boxes, ['#BA6124', '#886693', '#507493', '#647A52']):
        detail.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='white', linewidth=3))
        detail.text(x+w/2, y+h/2, f'{labels.get(k,k)}\n{value:,}\n{value/removed:.1%} of removals',
                    ha='center', va='center', color='white', fontsize=10.5, weight='bold', linespacing=1.5)
    fig.add_artist(ConnectionPatch(xyA=(kept_share, .3), coordsA=flow.transData,
                                  xyB=(0, 1), coordsB=detail.transData, color=ORANGE, linestyle='--', linewidth=1))
    fig.add_artist(ConnectionPatch(xyA=(1, .3), coordsA=flow.transData,
                                  xyB=(1, 1), coordsB=detail.transData, color=ORANGE, linestyle='--', linewidth=1))
    fig.text(.055, .365, 'ZOOM INTO\nREMOVED ROWS', color=ORANGE, weight='bold', fontsize=14, linespacing=1.4)
    fig.text(.055, .23, 'Each area is proportional\nto its share of removals.\nOne first-failing rule per row.',
             color='#667789', fontsize=11, linespacing=1.6)
    fig.text(.055, .047, 'Top: share of all 541,909 rows. Bottom: magnified composition of 19,405 removals; a different denominator.',
             fontsize=10.5, color='#667789')
    return _finish(fig)


@_styled
def figure_split_and_mixture(manifest: Mapping) -> plt.Figure:
    """Connect date chronology, lot counts, and complete-instance 50:50 mixing."""
    splits = manifest["splits"]
    counts = manifest["counts"]
    order = ("train", "validation", "test")
    colors = {"train": NAVY, "validation": TEAL, "test": ORANGE}
    fig = plt.figure(figsize=(12, 7.3))
    fig.text(.05, .955, 'Separate business time first; build balanced instances second', fontsize=19, weight='bold')
    fig.text(.05, .9, 'The time split acts on whole invoice dates. The 50:50 mixture acts on complete knapsacks.', fontsize=11, color='#667789')
    timeline = fig.add_axes([.065, .64, .87, .18])
    for split in order:
        record = splits[split]
        start = datetime.fromisoformat(record["date_from"])
        end = datetime.fromisoformat(record["date_to"])
        timeline.plot([start, end + timedelta(days=1)], [0, 0], lw=24, solid_capstyle="butt", color=colors[split])
        midpoint = start + (end - start) / 2
        timeline.text(midpoint, 0.20, split.title(), ha="center", va="bottom", color=colors[split], fontweight="bold")
    timeline.set(ylim=(-.35, .65), yticks=[])
    timeline.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    timeline.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    timeline.spines[["top", "right", "left", "bottom"]].set_visible(False)
    timeline.grid(axis="x", color="#d9e1e6", lw=0.7)
    tiles = fig.add_axes([.045, .115, .91, .42])
    tiles.set_xlim(0, 1)
    tiles.set_ylim(0, 1)
    tiles.axis("off")
    for column, split in enumerate(order):
        x = .005 + column * .337
        record = splits[split]
        synthetic, real = counts[split]["synthetic"], counts[split]["real"]
        total = counts[split]["total"]
        assert synthetic == real and synthetic + real == total
        tiles.add_patch(FancyBboxPatch((x, .02), .315, .94, boxstyle='round,pad=.003,rounding_size=.025', facecolor=PALE, edgecolor='none'))
        center=x+.1575
        tiles.text(center, .88, split.upper(), ha='center', va='center', weight='bold', color=colors[split], fontsize=13)
        tiles.text(center, .75, f"{record['date_from']} → {record['date_to']}", ha='center', fontsize=10)
        tiles.text(center, .61, f"{record['dates']} source dates | {record['lots']:,} lots", ha='center', fontsize=10)
        for offset, color, label, value in [(0, PURPLE, 'Synthetic', synthetic), (.142, TEAL, 'Real-derived', real)]:
            tiles.add_patch(Rectangle((x+.015+offset, .21), .142, .27, facecolor=color, edgecolor='white'))
            tiles.text(x+.086+offset, .345, f'{value:,}\n{label}', ha='center', va='center', color='white', fontsize=11, linespacing=1.5)
        tiles.text(center, .11, f'{total:,} complete instances', ha='center', va='center', weight='bold', fontsize=11)
    fig.text(.05, .04, 'Calendar distance is shown above; card sizes are not population scales. No invoice or source row crosses the three partitions.', color='#667789', fontsize=10)
    return _finish(fig)


@_styled
def figure_distribution_fingerprints(n: int = 1000, seed: int = 42) -> plt.Figure:
    """Plot ranked value-density fingerprints for fixed illustrative generator draws."""
    if n < 20:
        raise ValueError("n must be at least 20 for stable distribution fingerprints")
    descriptions = {
        "uniform": "independent value and weight",
        "normal": "concentrated around typical items",
        "correlated": "value tracks weight; densities cluster",
        "skewed": "rare high-density steps and unequal values",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
    percentile = np.linspace(0, 100, n)
    for ax, family in zip(axes.flat, FAMILY_COLORS):
        weights, values = generate_items(family, n=n, seed=seed)
        density = np.sort(values / weights)[::-1]
        median = float(np.median(density))
        ax.plot(percentile, density, color=FAMILY_COLORS[family], lw=2.2)
        ax.fill_between(percentile, density, median, color=FAMILY_COLORS[family], alpha=0.13)
        ax.axhline(median, color="#6b7280", lw=0.9, ls="--")
        ax.set_title(f"{family.title()}\n{descriptions[family]}", fontsize=11, pad=10)
        ax.set(ylabel='Value / weight (common log scale)', yscale='log', ylim=(.008, 150))
        top_value_share = np.sort(values)[-max(1, int(np.ceil(.05*n))):].sum() / values.sum()
        ax.text(0.98, 0.96, f"median = {median:.2f} | max / median = {density[0] / median:.1f}×\nTop 5% by value hold {top_value_share:.1%} of all value", transform=ax.transAxes, ha="right", va="top", fontsize=9.2, color=INK, bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.85})
        ax.grid(axis="y", color="#d9e1e6", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel('Item percentile: highest → lowest density', fontsize=10)
    fig.suptitle(f'How unequal is the value available per unit of resource?\nIllustrative generator draws: {n:,} items per family, seed {seed}; shared log axes', color=INK, fontweight='bold', fontsize=15)
    return _finish(fig)


def save_draft_figures(audit: Mapping, manifest: Mapping, out_dir: Path = Path("results/notebook-v2-visuals")) -> dict[str, Path]:
    """Save the three notebook draft PNGs and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "data-cleaning-stream": figure_cleaning_stream(audit),
        "data-split-mixture": figure_split_and_mixture(manifest),
        "data-distribution-fingerprints": figure_distribution_fingerprints(),
    }
    paths: dict[str, Path] = {}
    for name, figure in figures.items():
        path = out_dir / f"{name}.png"
        figure.savefig(path, dpi=150, facecolor="white")
        plt.close(figure)
        paths[name] = path
    return paths
