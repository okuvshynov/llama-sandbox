#!/usr/bin/env python3
"""Plot per-quant MCC ECDF for a llama-variance results JSONL.

Compile-error rows (no `mcc` field) are treated as MCC = -1, so a single
ECDF curve summarizes both compile rate and corpus quality.
"""

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#e03030', '#2070c0', '#40a050', '#d0a030', '#a040a0', '#888']


def quant_label(model_path: str) -> str:
    name = model_path.rsplit('/', 1)[-1]
    for suffix in ('.gguf',):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def load(path: str) -> dict[str, list[float]]:
    by_quant: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mcc = row.get('mcc', -1.0)
            if mcc is None:
                mcc = -1.0
            by_quant[quant_label(row['model'])].append(float(mcc))
    return by_quant


def plot(by_quant: dict[str, list[float]], out: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, vals) in enumerate(sorted(by_quant.items())):
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        xs = np.concatenate(([-1.05], xs, [1.0]))
        ys = np.concatenate(([0.0], ys, [1.0]))
        ax.step(xs, ys, where='post',
                color=COLORS[i % len(COLORS)],
                label=f'{label}  (n={len(vals)})',
                linewidth=1.8)

    ax.set_xlim(-1.05, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('MCC threshold $x$  (compile errors treated as $-1$)')
    ax.set_ylabel(r'$P(\mathrm{MCC} \leq x)$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='#999', linewidth=0.7, linestyle='--')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input',  default='results/res_reasoning_off.jsonl')
    p.add_argument('--output', default='results/mcc_cdf.png')
    p.add_argument('--title',  default='MCC ECDF — compile errors as $-1$')
    args = p.parse_args()

    by_quant = load(args.input)
    if not by_quant:
        raise SystemExit(f'no rows loaded from {args.input}')
    plot(by_quant, args.output, args.title)


if __name__ == '__main__':
    main()
