#!/usr/bin/env python
"""Per-turn MCC tier distribution for a hand-picked set of slugs.

Each (slug, turn) cell shows what fraction of attempts have reached which
quality tier by turn N. Tiers are coarse, principled bins on cumulative
best MCC:

    perfect : MCC == 1.0          (runner breaks on perfect, qualitatively different)
    strong  : 0.95 <= MCC < 1.0   (≥ 95% test pass equivalent)
    medium  : 0.75 <= MCC < 0.95  (decent classifier, clear room to improve)
    low     : 0.5 <= MCC < 0.75   (below medium but visibly above random)
    weak    : MCC < 0.5           (poor signal — also catches failures / no submission)

Failures (compile error, no submission yet, attempt ran out of budget mid-way)
collapse into the weak tier rather than being a separate visual category.
This trades the failure-type breakdown of plot_progression.py for a much
cleaner read: a "mostly green" row reliably reaches perfect; "mostly red"
reliably fails; mixed rows are interesting.

Vertical stacked bar per (slug, turn): tier segments stack bottom-up
(weak at the floor, perfect at the top) so the green band "rises" as a
model improves across turns. Slug labels on the y-axis are colored by
weight family — closed-weight (gpt-/claude-/anthropic-/o[1-4]-) in blue,
open-weight (everything else) in green — matching plot_scores.py.

Usage:
    plot_tiers.py --slugs SLUG [SLUG ...]
    plot_tiers.py --task toml-1.0-lua --slugs SLUG1,SLUG2,...
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


TIER_PERFECT = "perfect"
TIER_STRONG = "strong"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_WEAK = "weak"
TIER_ORDER = [TIER_PERFECT, TIER_STRONG, TIER_MEDIUM, TIER_LOW, TIER_WEAK]
TIER_COLORS = {
    TIER_PERFECT: "#2c7a2c",  # dark green
    TIER_STRONG:  "#94d495",  # light green
    TIER_MEDIUM:  "#f0d264",  # yellow
    TIER_LOW:     "#e8965a",  # orange
    TIER_WEAK:    "#d65454",  # red
}
TIER_LABELS = {
    TIER_PERFECT: "perfect (= 1.0)",
    TIER_STRONG:  "strong (≥ 0.95)",
    TIER_MEDIUM:  "medium (≥ 0.75)",
    TIER_LOW:     "low (≥ 0.5)",
    TIER_WEAK:    "weak (< 0.5)",
}

# Same convention as plot_scores.py: prefixes that identify closed-weight
# (vendor-served-only) models. Everything else is open-weight.
CLOSED_WEIGHT_PREFIXES = ("gpt-", "claude-", "anthropic-", "o1-", "o3-", "o4-")
COLOR_CLOSED = "#4C72B0"  # blue
COLOR_OPEN   = "#55A868"  # green


def is_closed_weight(slug: str) -> bool:
    return any(slug.startswith(p) for p in CLOSED_WEIGHT_PREFIXES)


def tier_of(mcc: float) -> str:
    """Classify a cumulative-best MCC into a tier. NaN ('no submission yet')
    folds into weak — that's the whole point of this encoding."""
    if np.isnan(mcc):
        return TIER_WEAK
    if mcc >= 1.0 - 1e-9:
        return TIER_PERFECT
    if mcc >= 0.95:
        return TIER_STRONG
    if mcc >= 0.75:
        return TIER_MEDIUM
    if mcc >= 0.5:
        return TIER_LOW
    return TIER_WEAK


def load_cumulative_best(results_file: Path, task: str, slugs: list[str],
                         max_turns: int) -> dict[str, np.ndarray]:
    """Read results.jsonl → {slug: array of shape (n_attempts, max_turns)}.
    Cell [a, t] = best MCC seen by attempt a after turns 0..t (inclusive),
    or NaN if no successful submission has happened in that window. Error
    rows count as 'no submission this turn' (cumulative best unchanged)."""
    slug_set = set(slugs)
    per_attempt: dict[tuple[str, str], list[float]] = {}
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
        if key not in per_attempt:
            per_attempt[key] = [np.nan] * max_turns
        if (mcc := r.get("mcc")) is not None:
            per_attempt[key][turn] = mcc

    by_slug: dict[str, list[list[float]]] = defaultdict(list)
    for (_, slug), turns in per_attempt.items():
        best = -np.inf
        bests: list[float] = []
        for t in turns:
            if not np.isnan(t):
                best = max(best, t)
            bests.append(np.nan if best == -np.inf else best)
        by_slug[slug].append(bests)

    return {s: np.array(by_slug[s]) for s in slugs if s in by_slug}


def plot_tiers(progression: dict[str, np.ndarray], task: str, slugs: list[str],
               output: Path, max_turns: int):
    available = [s for s in slugs if s in progression]
    missing_slugs = [s for s in slugs if s not in progression]
    for s in missing_slugs:
        print(f"  no data for {s} on task {task}, skipping")
    if not available:
        print("No matching data for any requested slug.")
        return

    # Sort by mean cumulative best-MCC at the last turn, descending —
    # strongest at the top, weakest at the bottom.
    def _avg_at_last(slug: str) -> float:
        col = progression[slug][:, -1]
        col = col[~np.isnan(col)]
        return float(np.mean(col)) if len(col) else -np.inf
    sorted_slugs = sorted(available, key=lambda s: -_avg_at_last(s))

    fig, ax = plt.subplots(figsize=(14, max(4, len(sorted_slugs) * 0.55 + 2.0)))

    BAR_HEIGHT = 0.85
    BAR_WIDTH = 0.55
    STACK_ORDER = list(reversed(TIER_ORDER))  # weak first → bottom of bar
    for row_idx, slug in enumerate(sorted_slugs):
        attempts = progression[slug]
        n = attempts.shape[0]
        for turn in range(max_turns):
            counts = {t: 0 for t in TIER_ORDER}
            for a in range(n):
                counts[tier_of(attempts[a, turn])] += 1
            y_bottom = row_idx - BAR_HEIGHT / 2
            cur = y_bottom
            for t in STACK_ORDER:
                h = (counts[t] / n) * BAR_HEIGHT
                if h > 0:
                    ax.bar(turn + 1, h, bottom=cur, width=BAR_WIDTH,
                           color=TIER_COLORS[t], edgecolor="white",
                           linewidth=0.4)
                cur += h

    ax.set_yticks(range(len(sorted_slugs)))
    ax.set_yticklabels(sorted_slugs, fontsize=12)
    # Color y-tick labels by weight family — matches plot_scores.py.
    for label, slug in zip(ax.get_yticklabels(), sorted_slugs):
        label.set_color(COLOR_CLOSED if is_closed_weight(slug) else COLOR_OPEN)

    ax.set_xticks(np.arange(1, max_turns + 1))
    ax.set_xlim(0.4, max_turns + 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Turn N")
    ax.set_title(f"Per-turn MCC tier distribution — {task}")
    ax.tick_params(axis="y", length=0)
    ax.spines[["right", "top"]].set_visible(False)

    # Tier legend at the bottom (horizontal); weight-family legend below it.
    tier_handles = [Patch(facecolor=TIER_COLORS[t], label=TIER_LABELS[t])
                    for t in TIER_ORDER]
    family_handles = [
        Patch(facecolor=COLOR_CLOSED, label="closed-weight (slug label)"),
        Patch(facecolor=COLOR_OPEN,   label="open-weight (slug label)"),
    ]
    leg1 = ax.legend(handles=tier_handles, loc="upper center",
                     bbox_to_anchor=(0.5, -0.08), fontsize=11, framealpha=0.95,
                     ncol=5, title="MCC tier at turn N", title_fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=family_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=11, framealpha=0.95,
              ncol=2)

    fig.subplots_adjust(left=0.27, right=0.98, top=0.93, bottom=0.22)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser(
        description="Per-turn MCC tier-distribution chart (single panel).")
    p.add_argument("--task", default="toml-1.0-cpp",
                   help="Task name (default: toml-1.0-cpp)")
    p.add_argument("--slugs", nargs="+", required=True,
                   help="Slugs to plot. Space-separated or comma-separated.")
    p.add_argument("--max-turns", type=int, default=5,
                   help="Number of turns to plot (default: 5)")
    p.add_argument("--results", default=None,
                   help="Path to results.jsonl (default: results/results.jsonl)")
    p.add_argument("--output", default=None,
                   help="Output image path (default: plots/tiers-<task>.png)")
    args = p.parse_args()

    slugs: list[str] = []
    for s in args.slugs:
        slugs.extend(part for part in s.split(",") if part)

    here = Path(__file__).parent
    results_file = Path(args.results) if args.results else here / "results" / "results.jsonl"
    output = Path(args.output) if args.output else here / "plots" / f"tiers-{args.task}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    progression = load_cumulative_best(results_file, args.task, slugs, args.max_turns)
    plot_tiers(progression, args.task, slugs, output, args.max_turns)


if __name__ == "__main__":
    main()
