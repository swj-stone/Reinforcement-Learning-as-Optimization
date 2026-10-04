"""Inspect a saved selector's input-to-action map, without updating its policy."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

from portfolio.bandit import Bandit


def policy_atlas_figure(saved_selector):
    """One cell per possible context; gray marks contexts unseen in training.

    Colored cells show the actual frozen argmax action, not measured test quality.
    Unseen contexts still use the global-mean fallback in the solver; that fallback
    is intentionally hidden here so training coverage is visible.
    """
    model = Bandit.from_dict(saved_selector['model'])
    expected = {'n': [30, 75, 150], 'capacity_fraction': [.375, .625],
                'density_cv': [.5, 1.5], 'max_weight_over_capacity': [.1, .5]}
    assert saved_selector['config']['context_bins'] == expected
    descriptions = {
        'greedy_density': ('G', 'Density greedy', '#536A85'),
        'value_first': ('V', 'Value first', '#B9744D'),
        'light_first': ('L', 'Light first', '#BCAD66'),
        'density_sqrt': ('Q', 'Value / sqrt(weight)', '#816BAA'),
        'one_e_threshold': ('T', '1/e threshold', '#CF657A'),
        'seeded_greedy8': ('S', 'Seeded greedy', '#DA8F31'),
        'swap_repair8': ('R', 'Exchange repair', '#168C82'),
        'dqn_seed0': ('D', 'Frozen DQN', '#4E98B9'),
    }
    visited = {key for key, counts in model.counts.items() if int(counts.sum()) > 0}
    actions = sorted({model.predict(key) for key in visited})
    colors = ['#E8EDF1'] + [descriptions[model.arms[a]][2] for a in actions]
    color_index = {action: i + 1 for i, action in enumerate(actions)}
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, len(colors) + .5), cmap.N)
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': '#23384D',
             'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(style):
        fig, axes = plt.subplots(4, 3, figsize=(12, 10.5))
        fig.subplots_adjust(left=.15, right=.96, bottom=.19, top=.84, hspace=.27, wspace=.22)
        fig.suptitle('The learned policy is a map, not a single universal rule',
                     x=.5, y=.98, fontsize=19, weight='bold')
        fig.text(.5, .935, 'Saved selector seed 1 | 108 possible input contexts | actual frozen action choices',
                 ha='center', color='#667789', fontsize=11)
        capacity_labels = ['Tight capacity\nC / total weight < 0.375',
                           'Medium capacity\n0.375 to < 0.625',
                           'Loose capacity\nC / total weight ≥ 0.625']
        size_labels = ['Small\nn < 30', 'Medium\n30 ≤ n < 75',
                       'Large\n75 ≤ n < 150', 'Largest bin\nn ≥ 150']
        for ni in range(4):
            for ci in range(3):
                ax = axes[ni, ci]
                values = np.zeros((3, 3), dtype=int)
                for di in range(3):
                    for wi in range(3):
                        key = (ni, ci, di, wi)
                        if key in visited:
                            action = model.predict(key)
                            values[di, wi] = color_index[action]
                            ax.text(wi, di, descriptions[model.arms[action]][0],
                                    ha='center', va='center', color='white', fontsize=13, weight='bold')
                        else:
                            ax.text(wi, di, '—', ha='center', va='center', color='#9AA9B4')
                ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')
                ax.set_xticks([0, 1, 2], ['<0.1', '0.1–<0.5', '≥0.5'] if ni == 3 else [])
                ax.set_yticks([0, 1, 2], ['<0.5', '0.5–<1.5', '≥1.5'] if ci == 0 else [])
                ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
                ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
                ax.grid(which='minor', color='white', linewidth=3)
                ax.tick_params(which='both', length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if ni == 0:
                    ax.set_title(capacity_labels[ci], fontsize=11, pad=14)
                if ci == 0:
                    ax.set_ylabel(size_labels[ni], fontsize=11, rotation=0, labelpad=66, va='center')
        fig.text(.5, .901, 'Inside each tile: density variation (CV) increases from the top row to the bottom row',
                 ha='center', fontsize=10, color='#667789')
        fig.text(.56, .135, 'Within each tile: largest item weight / capacity increases →',
                 ha='center', fontsize=11)
        handles = [Patch(facecolor=descriptions[model.arms[a]][2],
                         label=f'{descriptions[model.arms[a]][0]} = {descriptions[model.arms[a]][1]}')
                   for a in actions]
        handles.append(Patch(facecolor=colors[0], label='Unseen context (fallback not shown)'))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .075),
                   frameon=False, ncol=min(3, len(handles)), fontsize=10)
        observations = sum(int(counts.sum()) for counts in model.counts.values())
        fig.text(.5, .028,
                 f'{len(visited)} / 108 contexts observed in {observations:,} selected-action updates. '
                 'Colors describe policy choices, not test performance.\n'
                 'Test sizes 200 and 500 share the bottom row; bin membership alone cannot guarantee generalization.',
                 ha='center', va='center', fontsize=10, color='#667789', linespacing=1.5)
    return fig
