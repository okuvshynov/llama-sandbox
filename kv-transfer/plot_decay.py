#!/usr/bin/env python3
"""Generate static PNG decay charts from kv-transfer decay CSV results.

Shows how the KV transfer benefit (KL ratio = handoff/target) changes
across generation position. Each line is a quant level, x-axis is
token window, y-axis is KL ratio (lower = more benefit from handoff).
"""

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

QUANT_COLORS = [
    '#e03030', '#e07840', '#d0a030', '#40a050', '#2090a0',
    '#2070c0', '#6050c0', '#a040a0', '#888', '#333',
]

THEMES = {
    'light': {
        'bg': '#ffffff',
        'text': '#333333',
        'text_light': '#555555',
        'grid': '#dddddd',
        'parity': '#bbbbbb',
    },
    'dark': {
        'bg': '#212121',
        'text': '#dcdcdc',
        'text_light': '#ababab',
        'grid': '#444444',
        'parity': '#666666',
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


def read_decay(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'prompt': r['prompt'],
                'target': r['target'],
                'n_prompt': int(r['n_prompt']),
                'window_start': int(r['window_start']),
                'window_end': int(r['window_end']),
                'kl_target': float(r['kl_target']),
                'kl_handoff': float(r['kl_handoff']),
            })
    return rows


def aggregate_decay(rows):
    """Average kl_target and kl_handoff across prompts per (quant, window),
    then compute ratio. This gives ratio-of-means, not mean-of-ratios."""
    by_key = defaultdict(lambda: {'kl_target': [], 'kl_handoff': []})
    for r in rows:
        key = (r['target'], r['window_start'])
        by_key[key]['kl_target'].append(r['kl_target'])
        by_key[key]['kl_handoff'].append(r['kl_handoff'])

    result = defaultdict(dict)
    for (target, window_start), vals in by_key.items():
        mean_t = np.mean(vals['kl_target'])
        mean_h = np.mean(vals['kl_handoff'])
        result[target][window_start] = {
            'kl_target': mean_t,
            'kl_handoff': mean_h,
            'kl_ratio': mean_h / mean_t if mean_t > 0 else 0.0,
        }
    return result


def plot_decay(rows, output_path, title, theme='light'):
    """KL ratio decay curve per quant, ratio of means across prompts."""
    t = THEMES[theme]

    agg = aggregate_decay(rows)
    quants = sorted(agg.keys(), key=quant_sort_key)
    qcolor = {q: QUANT_COLORS[i % len(QUANT_COLORS)] for i, q in enumerate(quants)}

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_theme(fig, ax, theme)

    window_size = rows[0]['window_end'] - rows[0]['window_start']

    for q in quants:
        windows = sorted(agg[q].keys())
        ratios = [agg[q][w]['kl_ratio'] for w in windows]
        x = [w + window_size / 2 for w in windows]
        ax.plot(x, ratios, color=qcolor[q], linewidth=2, label=q, marker='o', markersize=4)

    # parity line at 1.0
    ax.axhline(y=1.0, color=t['parity'], linestyle='--', linewidth=1, zorder=0)

    ax.set_xlabel('Generation token position')
    ax.set_ylabel('KL ratio (handoff / target, lower = more benefit)')
    ax.set_title(title, fontsize=11)
    ax.set_ylim(bottom=0)

    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)

    ax.legend(fontsize=8, loc='upper left',
              facecolor=t['bg'], edgecolor=t['grid'], labelcolor=t['text'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f'  {output_path}')


def plot_decay_spread(rows, output_path, title, theme='light'):
    """KL ratio decay with per-prompt spread (shaded band) per quant.

    Center line uses ratio-of-means (robust). Band shows IQR of
    per-prompt ratios (shows prompt-to-prompt variation).
    """
    t = THEMES[theme]

    agg = aggregate_decay(rows)

    # also collect per-prompt ratios for the spread band
    per_prompt = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ratio = r['kl_handoff'] / r['kl_target'] if r['kl_target'] > 0 else 0.0
        per_prompt[r['target']][r['window_start']].append(ratio)

    quants = sorted(agg.keys(), key=quant_sort_key)
    qcolor = {q: QUANT_COLORS[i % len(QUANT_COLORS)] for i, q in enumerate(quants)}

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_theme(fig, ax, theme)

    window_size = rows[0]['window_end'] - rows[0]['window_start']

    for q in quants:
        windows = sorted(agg[q].keys())
        x = [w + window_size / 2 for w in windows]
        means = [agg[q][w]['kl_ratio'] for w in windows]
        p25 = [np.percentile(per_prompt[q][w], 25) for w in windows]
        p75 = [np.percentile(per_prompt[q][w], 75) for w in windows]

        ax.plot(x, means, color=qcolor[q], linewidth=2, label=q)
        ax.fill_between(x, p25, p75, color=qcolor[q], alpha=0.15)

    ax.axhline(y=1.0, color=t['parity'], linestyle='--', linewidth=1, zorder=0)

    ax.set_xlabel('Generation token position')
    ax.set_ylabel('KL ratio (handoff / target)')
    ax.set_title(title, fontsize=11)
    ax.set_ylim(bottom=0)

    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)

    ax.legend(fontsize=8, loc='upper left',
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
        decay_path = os.path.join(args.results_dir, name, 'decay.csv')
        if not os.path.isfile(decay_path):
            continue

        short = model_short_name(name)
        print(f'{short}:')
        rows = read_decay(decay_path)

        for theme in ('light', 'dark'):
            suffix = f'-{theme}' if theme == 'dark' else ''
            plot_decay(rows,
                       os.path.join(args.output_dir, f'decay-{short}{suffix}.png'),
                       f'KL ratio decay — {short}', theme=theme)
            plot_decay_spread(rows,
                              os.path.join(args.output_dir, f'decay-spread-{short}{suffix}.png'),
                              f'KL ratio decay (with spread) — {short}', theme=theme)


if __name__ == '__main__':
    main()
