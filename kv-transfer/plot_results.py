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

LONG_PROMPT_THRESHOLD = 1500


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


def plot_bar(rows, output_path, title):
    """Horizontal bar chart: all quant×{target,handoff} bars sorted by KL value."""
    # Aggregate by quant
    by_quant = defaultdict(list)
    for r in rows:
        by_quant[r['target_model']].append(r)

    entries = []
    for q, rs in by_quant.items():
        n = len(rs)
        kl_t = sum(r['kl_target'] for r in rs) / n
        kl_h = sum(r['kl_handoff'] for r in rs) / n
        entries.append((kl_t, f'{q}', COLORS_TARGET))
        entries.append((kl_h, f'{q} + handoff', COLORS_HANDOFF))

    # Sort by value (lowest = best at top after inversion)
    entries.sort(key=lambda e: e[0], reverse=True)

    values = [e[0] for e in entries]
    labels = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(8, max(3, len(entries) * 0.35)))
    ax.barh(range(len(entries)), values, color=colors, height=0.7)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontfamily='monospace', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('KL divergence (log scale, lower is better)')
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()

    # Value labels
    for i, v in enumerate(values):
        ax.text(v * 1.1, i, f'{v:.3f}', va='center', fontsize=8, color='#555')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS_TARGET, label='Target (no transfer)'),
        Patch(facecolor=COLORS_HANDOFF, label='Handoff (KV transfer)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  {output_path}')


def plot_scatter(rows, output_path, title):
    """Scatter: KL(target) vs KL(handoff), colored by quant, shape by prompt length."""
    quants = sorted(set(r['target_model'] for r in rows), key=quant_sort_key)
    qcolor = {q: QUANT_COLORS[i % len(QUANT_COLORS)] for i, q in enumerate(quants)}

    fig, ax = plt.subplots(figsize=(6, 6))

    for r in rows:
        if r['kl_target'] <= 0 or r['kl_handoff'] <= 0:
            continue
        color = qcolor[r['target_model']]
        marker = 's' if r['n_prompt'] >= LONG_PROMPT_THRESHOLD else 'o'
        ax.scatter(r['kl_target'], r['kl_handoff'],
                   c=color, marker=marker, s=40, alpha=0.75,
                   edgecolors=color, linewidths=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Diagonal — draw after setting scale so limits are correct
    all_kl = [r['kl_target'] for r in rows if r['kl_target'] > 0] + \
             [r['kl_handoff'] for r in rows if r['kl_handoff'] > 0]
    lo = min(all_kl) * 0.5
    hi = max(all_kl) * 2.0
    ax.plot([lo, hi], [lo, hi], '--', color='#bbb', linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel('KL(target)')
    ax.set_ylabel('KL(handoff)')
    ax.set_title(title, fontsize=11)

    # Legend: quant colors
    from matplotlib.lines import Line2D
    handles = []
    for q in quants:
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=qcolor[q], markersize=8,
                              label=q))
    handles.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#999', markersize=8,
                          label='short prompt'))
    handles.append(Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='#999', markersize=8,
                          label='long prompt'))
    ax.legend(handles=handles, fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

        plot_bar(rows,
                 os.path.join(args.output_dir, f'kl-bar-{short}.png'),
                 f'KL divergence — {short}')

        plot_scatter(rows,
                     os.path.join(args.output_dir, f'kl-scatter-{short}.png'),
                     f'Per-prompt KL — {short}')


if __name__ == '__main__':
    main()
