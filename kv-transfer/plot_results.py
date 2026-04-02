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


def plot_bar(rows, output_path, title, theme='light'):
    """Horizontal bar chart: all quant×{target,handoff} bars sorted by KL value."""
    t = THEMES[theme]
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
    apply_theme(fig, ax, theme)
    ax.barh(range(len(entries)), values, color=colors, height=0.7)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontfamily='monospace', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('KL divergence (log scale, lower is better)')
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()

    # Value labels
    for i, v in enumerate(values):
        ax.text(v * 1.1, i, f'{v:.3f}', va='center', fontsize=8, color=t['text_light'])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS_TARGET, label='Target (no transfer)'),
        Patch(facecolor=COLORS_HANDOFF, label='Handoff (KV transfer)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              facecolor=t['bg'], edgecolor=t['grid'], labelcolor=t['text'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=t['bg'])
    plt.close()
    print(f'  {output_path}')


def plot_scatter(rows, output_path, title, theme='light'):
    """Scatter: KL(target) vs KL(handoff), colored by quant, shape by prompt length."""
    t = THEMES[theme]
    quants = sorted(set(r['target_model'] for r in rows), key=quant_sort_key)
    qcolor = {q: QUANT_COLORS[i % len(QUANT_COLORS)] for i, q in enumerate(quants)}

    fig, ax = plt.subplots(figsize=(6, 6))
    apply_theme(fig, ax, theme)

    for r in rows:
        if r['kl_target'] <= 0 or r['kl_handoff'] <= 0:
            continue
        color = qcolor[r['target_model']]
        if r['n_prompt'] >= XLARGE_PROMPT_THRESHOLD:
            marker, ms = 'D', 40
        elif r['n_prompt'] >= MEDIUM_PROMPT_THRESHOLD:
            marker, ms = 's', 40
        else:
            marker, ms = 'o', 40
        ax.scatter(r['kl_target'], r['kl_handoff'],
                   c=color, marker=marker, s=ms, alpha=0.75,
                   edgecolors=color, linewidths=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Diagonal — draw after setting scale so limits are correct
    all_kl = [r['kl_target'] for r in rows if r['kl_target'] > 0] + \
             [r['kl_handoff'] for r in rows if r['kl_handoff'] > 0]
    lo = min(all_kl) * 0.5
    hi = max(all_kl) * 2.0
    ax.plot([lo, hi], [lo, hi], '--', color=t['diagonal'], linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel('KL(target)')
    ax.set_ylabel('KL(handoff)')
    ax.set_title(title, fontsize=11)

    # Legend: quant colors
    from matplotlib.lines import Line2D
    handles = []
    for q in quants:
        handles.append(Line2D([0], [0], marker='o', color=t['bg'],
                              markerfacecolor=qcolor[q], markersize=8,
                              label=q))
    handles.append(Line2D([0], [0], marker='o', color=t['bg'],
                          markerfacecolor='#999', markersize=8,
                          label='short prompt'))
    handles.append(Line2D([0], [0], marker='s', color=t['bg'],
                          markerfacecolor='#999', markersize=8,
                          label='medium prompt'))
    handles.append(Line2D([0], [0], marker='D', color=t['bg'],
                          markerfacecolor='#999', markersize=8,
                          label='xlarge prompt'))
    ax.legend(handles=handles, fontsize=7, loc='upper left',
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
            plot_bar(rows,
                     os.path.join(args.output_dir, f'kl-bar-{short}{suffix}.png'),
                     f'KL divergence — {short}', theme=theme)

            plot_scatter(rows,
                         os.path.join(args.output_dir, f'kl-scatter-{short}{suffix}.png'),
                         f'Per-prompt KL — {short}', theme=theme)


if __name__ == '__main__':
    main()
