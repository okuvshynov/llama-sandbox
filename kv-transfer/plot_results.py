#!/usr/bin/env python3
"""Generate static PNG charts from kv-transfer CSV results."""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

QUANT_ORDER = [
    'ud-iq1_m', 'ud-iq2_xxs', 'ud-q2_k_xl', 'ud-iq3_xss', 'ud-iq3_xxs',
    'ud-q3_k_xl', 'ud-iq4_xs', 'ud-q4_k_xl', 'ud-q5_k_xl', 'ud-q6_k_xl', 'ud-q8_k_xl',
]

COLORS_TARGET = '#d04040'
COLORS_HANDOFF = '#2070c0'

QUANT_COLORS = [
    '#e03030', '#e07840', '#d0a030', '#40a050', '#2090a0',
    '#2070c0', '#6050c0', '#a040a0', '#888', '#333',
]

MEDIUM_PROMPT_THRESHOLD = 1500
XLARGE_PROMPT_THRESHOLD = 5000

THEMES = {
    'light': {
        'bg': '#ffffff',
        'text': '#333333',
        'text_light': '#555555',
        'grid': '#dddddd',
        'diagonal': '#bbbbbb',
    },
    'dark': {
        'bg': '#212121',
        'text': '#dcdcdc',
        'text_light': '#ababab',
        'grid': '#444444',
        'diagonal': '#666666',
    },
}


def apply_theme(fig, ax, theme):
    t = THEMES[theme]
    fig.patch.set_facecolor(t['bg'])
    ax.set_facecolor(t['bg'])
    ax.title.set_color(t['text'])
    ax.xaxis.label.set_color(t['text'])
    ax.yaxis.label.set_color(t['text'])
    ax.tick_params(colors=t['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(t['grid'])


def quant_sort_key(q):
    try:
        return QUANT_ORDER.index(q)
    except ValueError:
        return 999


def model_short_name(dirname):
    parts = dirname.split('-ud-')
    return parts[0] if len(parts) > 1 else dirname


def read_summary(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'prompt': r['prompt'],
                'target_model': r['target_model'],
                'n_prompt': int(r['n_prompt']),
                'kl_target': float(r['kl_target']),
                'kl_handoff': float(r['kl_handoff']),
            })
    return rows


def prompt_marker(n_prompt):
    if n_prompt >= XLARGE_PROMPT_THRESHOLD:
        return 'D'
    elif n_prompt >= MEDIUM_PROMPT_THRESHOLD:
        return 's'
    return 'o'


def plot_combined(rows, output_path, title, theme='light'):
    """Horizontal box plot per quant with individual points overlaid.

    Each quant gets two rows: target (red) and handoff (blue).
    Box shows quartiles, individual points are shaped by prompt length.
    """
    t = THEMES[theme]
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Group by quant
    by_quant = defaultdict(list)
    for r in rows:
        by_quant[r['target_model']].append(r)

    quants = sorted(by_quant.keys(), key=quant_sort_key)
    n_quants = len(quants)

    # Each quant gets 2 rows (target + handoff), with spacing between quants
    row_height = 0.35
    quant_gap = 0.3
    fig_height = max(4, n_quants * (2 * row_height + quant_gap) + 1.5)

    fig, ax = plt.subplots(figsize=(10, fig_height))
    apply_theme(fig, ax, theme)

    y_positions = []  # (y_target, y_handoff) per quant
    y_labels = []
    y_label_positions = []

    for i, q in enumerate(quants):
        base_y = i * (2 * row_height + quant_gap)
        y_target = base_y
        y_handoff = base_y + row_height
        y_positions.append((y_target, y_handoff))
        y_label_positions.append(base_y + row_height / 2)
        y_labels.append(q)

    # Draw box plots and points
    for i, q in enumerate(quants):
        rs = by_quant[q]
        y_t, y_h = y_positions[i]

        for y_pos, key, color in [(y_t, 'kl_target', COLORS_TARGET),
                                   (y_h, 'kl_handoff', COLORS_HANDOFF)]:
            values = [r[key] for r in rs if r[key] > 0]
            if not values:
                continue

            # Box plot
            bp = ax.boxplot([values], positions=[y_pos], vert=False,
                            widths=row_height * 0.8, patch_artist=True,
                            boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color),
                            medianprops=dict(color=color, linewidth=2),
                            whiskerprops=dict(color=color),
                            capprops=dict(color=color),
                            flierprops=dict(marker='none'))

            # Individual points with jitter
            np.random.seed(42)
            jitter = np.random.uniform(-row_height * 0.25, row_height * 0.25, len(rs))
            for j, r in enumerate(rs):
                v = r[key]
                if v <= 0:
                    continue
                m = prompt_marker(r['n_prompt'])
                ax.scatter(v, y_pos + jitter[j], marker=m, s=25,
                           c=color, alpha=0.8, edgecolors=color,
                           linewidths=0.5, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('KL divergence (log scale, lower is better)')
    ax.set_title(title, fontsize=11)

    ax.set_yticks(y_label_positions)
    ax.set_yticklabels(y_labels, fontfamily='monospace', fontsize=9)
    ax.invert_yaxis()

    # Grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)

    # Legend
    handles = [
        Patch(facecolor=COLORS_TARGET, alpha=0.5, label='Target (no transfer)'),
        Patch(facecolor=COLORS_HANDOFF, alpha=0.5, label='Handoff (KV transfer)'),
        Line2D([0], [0], marker='o', color=t['bg'], markerfacecolor=t['text_light'],
               markersize=7, label='short prompt'),
        Line2D([0], [0], marker='s', color=t['bg'], markerfacecolor=t['text_light'],
               markersize=7, label='medium prompt'),
        Line2D([0], [0], marker='D', color=t['bg'], markerfacecolor=t['text_light'],
               markersize=7, label='xlarge prompt'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              facecolor=t['bg'], edgecolor=t['grid'], labelcolor=t['text'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f'  {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='./results')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write PNG files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for name in sorted(os.listdir(args.results_dir)):
        summary_path = os.path.join(args.results_dir, name, 'summary.csv')
        if not os.path.isfile(summary_path):
            continue

        short = model_short_name(name)
        print(f'{short}:')
        rows = read_summary(summary_path)

        for theme in ('light', 'dark'):
            suffix = f'-{theme}' if theme == 'dark' else ''
            plot_combined(rows,
                          os.path.join(args.output_dir, f'kl-{short}{suffix}.png'),
                          f'KL divergence — {short}', theme=theme)


if __name__ == '__main__':
    main()
