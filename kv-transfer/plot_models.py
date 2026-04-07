#!/usr/bin/env python3
"""Generate static PNG charts comparing KV transfer decay across models.

Chart 1: One line per model, KL ratio (handoff/target) averaged across
all quant levels, with IQR spread band showing variation across quants
and prompts. Shows which model families benefit most from KV transfer
and how the benefit decays with generation length.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = [
    '#e03030', '#e07840', '#d0a030', '#40a050', '#2090a0',
    '#2070c0', '#6050c0', '#a040a0', '#c06080', '#888',
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


def plot_model_comparison(all_models, output_path, title, theme='light'):
    """One line per model, KL ratio aggregated across all quants.

    Center line: ratio-of-means(kl_handoff, kl_target) across all
    prompts and quants per window.
    Band: IQR of per-(prompt, quant) ratios — shows combined variation.
    """
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    apply_theme(fig, ax, theme)

    for i, (model_name, rows) in enumerate(all_models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        # collect per-window: mean KL target, mean KL handoff (across all quants + prompts)
        by_window_agg = defaultdict(lambda: {'kl_target': 0.0, 'kl_handoff': 0.0, 'count': 0})
        # also per-(prompt, quant) ratios for spread
        by_window_ratios = defaultdict(list)

        window_size = rows[0]['window_end'] - rows[0]['window_start']

        for r in rows:
            w = r['window_start']
            by_window_agg[w]['kl_target'] += r['kl_target']
            by_window_agg[w]['kl_handoff'] += r['kl_handoff']
            by_window_agg[w]['count'] += 1
            ratio = r['kl_handoff'] / r['kl_target'] if r['kl_target'] > 0 else 0.0
            by_window_ratios[w].append(ratio)

        windows = sorted(by_window_agg.keys())
        x = [w + window_size / 2 for w in windows]

        # ratio of means (robust center line)
        means = []
        for w in windows:
            d = by_window_agg[w]
            mean_t = d['kl_target'] / d['count']
            mean_h = d['kl_handoff'] / d['count']
            means.append(mean_h / mean_t if mean_t > 0 else 0.0)

        p25 = [np.percentile(by_window_ratios[w], 25) for w in windows]
        p75 = [np.percentile(by_window_ratios[w], 75) for w in windows]

        ax.plot(x, means, color=color, linewidth=2, label=model_name)
        ax.fill_between(x, p25, p75, color=color, alpha=0.12)

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


def plot_model_comparison_by_size(all_models, output_path, title, size_filter, theme='light'):
    """Same as plot_model_comparison but filtered by prompt size category."""
    MEDIUM_THRESHOLD = 1500
    LARGE_THRESHOLD = 5000

    def prompt_size(n_prompt):
        if n_prompt >= LARGE_THRESHOLD:
            return 'large'
        if n_prompt >= MEDIUM_THRESHOLD:
            return 'medium'
        return 'small'

    filtered = []
    for model_name, rows in all_models:
        f_rows = [r for r in rows if prompt_size(r['n_prompt']) == size_filter]
        if f_rows:
            filtered.append((model_name, f_rows))

    if not filtered:
        return

    plot_model_comparison(filtered, output_path, title, theme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='./results')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write PNG files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load all models
    all_models = []
    for name in sorted(os.listdir(args.results_dir)):
        decay_path = os.path.join(args.results_dir, name, 'decay.csv')
        if not os.path.isfile(decay_path):
            continue
        short = model_short_name(name)
        rows = read_decay(decay_path)
        if rows:
            all_models.append((short, rows))

    if not all_models:
        print('No decay data found')
        return

    print(f'Found {len(all_models)} models: {", ".join(m for m, _ in all_models)}')
    print()

    # chart 1: all models compared, all quants aggregated
    print('Model comparison (all quants aggregated):')
    for theme in ('light', 'dark'):
        suffix = f'-{theme}' if theme == 'dark' else ''
        plot_model_comparison(
            all_models,
            os.path.join(args.output_dir, f'models-decay{suffix}.png'),
            'KV transfer decay — all models (averaged across quants)',
            theme=theme,
        )

    # chart 2-4: by prompt size
    for size in ('small', 'medium', 'large'):
        print(f'\nModel comparison ({size} prompts only):')
        for theme in ('light', 'dark'):
            suffix = f'-{theme}' if theme == 'dark' else ''
            plot_model_comparison_by_size(
                all_models,
                os.path.join(args.output_dir, f'models-decay-{size}{suffix}.png'),
                f'KV transfer decay — all models ({size} prompts only)',
                size,
                theme=theme,
            )


if __name__ == '__main__':
    main()
