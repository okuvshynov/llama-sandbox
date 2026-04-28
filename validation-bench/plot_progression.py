#!/usr/bin/env python
"""Per-turn MCC progression slopegraph for a hand-picked set of slugs.

For each slug, plots the mean "best MCC after N turns" across attempts
(N = 1..max_turns). Lines that climb steeply across turns benefit from the
multi-turn submit-and-fix loop; flat-from-N=1 lines are one-shotters.

Infra failures (a turn with no submission row, or an error row) are *not*
folded into the MCC mean — those attempts simply don't contribute to that
turn's mean. Marker face-alpha at each (slug, turn) is scaled by the
fraction of attempts that had produced a submission by that turn, so
faint markers = many attempts hadn't submitted yet. A small "k/n"
annotation appears under markers where the fraction is < 1.

Usage:
    plot_progression.py --slugs SLUG [SLUG ...]
    plot_progression.py --task toml-1.0-lua --slugs SLUG1,SLUG2,...
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_progression(results_file: Path, task: str, slugs: list[str], max_turns: int
                     ) -> dict[str, np.ndarray]:
    """Read results.jsonl → {slug: array of shape (n_attempts, max_turns)} where
    cell [a, t] = best MCC seen by attempt a after turns 0..t (inclusive),
    or NaN if no successful submission has happened in that window. Error
    rows and missing turns are both treated as "no submission this turn"."""
    slug_set = set(slugs)
    per_turn: dict[tuple[str, str], list[float]] = {}
    for line in results_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") != task:
            continue
        slug = r.get("slug")
        if slug not in slug_set:
            continue
        turn = r.get("turn")
        if turn is None or turn >= max_turns:
            continue
        key = (r["attempt_id"], slug)
        if key not in per_turn:
            per_turn[key] = [np.nan] * max_turns
        mcc = r.get("mcc")
        if mcc is not None:
            per_turn[key][turn] = mcc

    by_slug: dict[str, list[list[float]]] = defaultdict(list)
    for (_, slug), turns in per_turn.items():
        best = -np.inf
        bests: list[float] = []
        for t in turns:
            if not np.isnan(t):
                best = max(best, t)
            bests.append(np.nan if best == -np.inf else best)
        by_slug[slug].append(bests)

    return {s: np.array(by_slug[s]) for s in slugs if s in by_slug}


def _twelve_distinct_colors(n: int) -> list:
    """tab20 even indices (10 saturated colors) then odd indices for >10 lines."""
    cmap = plt.get_cmap("tab20")
    even = [cmap(i) for i in range(0, 20, 2)]
    odd = [cmap(i) for i in range(1, 20, 2)]
    return (even + odd)[:n]


def plot_progression(progression: dict[str, np.ndarray], task: str,
                     slugs: list[str], output: Path, max_turns: int):
    available = [s for s in slugs if s in progression]
    missing = [s for s in slugs if s not in progression]
    for s in missing:
        print(f"  no data for {s} on task {task}, skipping")
    if not available:
        print(f"No matching data for any requested slug.")
        return

    colors = _twelve_distinct_colors(len(available))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.arange(1, max_turns + 1)

    # First pass: draw lines + per-turn markers; collect end-of-line points
    # so labels can be staggered below.
    endpoints: list[tuple[str, int, float, float, tuple]] = []
    for slug, color in zip(available, colors):
        attempts = progression[slug]
        n = attempts.shape[0]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # nanmean over an all-NaN column (turn N where 0 attempts had
            # submitted) returns NaN with a RuntimeWarning — that's the
            # signal we *want* on this plot, not an error.
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_mcc = np.nanmean(attempts, axis=0)
        submit_rate = np.sum(~np.isnan(attempts), axis=0) / n

        ax.plot(xs, mean_mcc, color=color, linewidth=1.8, alpha=0.85, zorder=2)
        for x, y, rate in zip(xs, mean_mcc, submit_rate):
            if np.isnan(y):
                continue
            ax.scatter([x], [y], color=color, s=70,
                       alpha=0.25 + 0.75 * rate,
                       edgecolors="white", linewidths=0.8, zorder=3)
            if rate < 1.0:
                ax.text(x, y - 0.015,
                        f"{int(round(rate * n))}/{n}",
                        ha="center", va="top", fontsize=7, color=color)

        last_idx = max(i for i, v in enumerate(mean_mcc) if not np.isnan(v))
        endpoints.append((slug, n, float(xs[last_idx]),
                          float(mean_mcc[last_idx]), color))

    # Stagger end-of-line labels: sort by y desc, then nudge down whenever a
    # label would land within MIN_GAP of the previous one. A short leader line
    # connects the dot at the actual endpoint to the staggered label.
    MIN_GAP = 0.035
    LABEL_X = max_turns + 0.25
    endpoints.sort(key=lambda e: -e[3])
    last_y = float("inf")
    for slug, n, ex, ey, color in endpoints:
        ly = min(ey, last_y - MIN_GAP)
        last_y = ly
        ax.plot([ex, LABEL_X - 0.05], [ey, ly], color=color, linewidth=0.6,
                alpha=0.5, zorder=1)
        ax.text(LABEL_X, ly, f"{slug} (n={n})",
                fontsize=8, color=color, va="center")

    ax.set_xlim(0.5, max_turns + 3.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(xs)
    ax.set_xlabel("N (turns elapsed): mean best-MCC across attempts that had submitted by turn N")
    ax.set_ylabel("Mean best MCC")
    ax.set_title(f"Per-turn progression — {task}")
    ax.axhline(0, color="#888", linewidth=0.5, zorder=1)
    ax.grid(alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser(
        description="Plot per-turn MCC progression for a hand-picked set of slugs.")
    p.add_argument("--task", default="toml-1.0-cpp",
                   help="Task name (default: toml-1.0-cpp)")
    p.add_argument("--slugs", nargs="+", required=True,
                   help="Slugs to plot. Space-separated or comma-separated (or both).")
    p.add_argument("--max-turns", type=int, default=5,
                   help="Number of turns to plot (default: 5)")
    p.add_argument("--results", default=None,
                   help="Path to results.jsonl (default: results/results.jsonl)")
    p.add_argument("--output", default=None,
                   help="Output image path (default: plots/progression-<task>.png)")
    args = p.parse_args()

    slugs: list[str] = []
    for s in args.slugs:
        slugs.extend(part for part in s.split(",") if part)

    here = Path(__file__).parent
    results_file = Path(args.results) if args.results else here / "results" / "results.jsonl"
    output = Path(args.output) if args.output else here / "plots" / f"progression-{args.task}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    progression = load_progression(results_file, args.task, slugs, args.max_turns)
    plot_progression(progression, args.task, slugs, output, args.max_turns)


if __name__ == "__main__":
    main()
