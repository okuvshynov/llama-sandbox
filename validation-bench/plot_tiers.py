#!/usr/bin/env python
"""Per-turn MCC tier distribution for a hand-picked set of slugs.

Each (slug, turn) cell shows what fraction of attempts have reached which
quality tier by turn N. Tiers are coarse, principled bins on cumulative
best MCC:

    perfect : MCC == 1.0          (runner breaks on perfect, qualitatively different)
    strong  : 0.95 <= MCC < 1.0   (≥ 95% test pass equivalent)
    medium  : 0.75 <= MCC < 0.95  (decent classifier, clear room to improve)
    weak    : MCC < 0.75          (poor signal — also catches infra failures and no submission)

Failures (compile error, no submission yet, attempt ran out of budget mid-way)
collapse into the weak tier rather than being a separate visual category.
This trades the failure-type breakdown of plot_progression.py for a much
cleaner read: a "mostly green" row reliably reaches perfect; "mostly red"
reliably fails; mixed rows are interesting.

Top panel: mean cumulative best-MCC trajectory (same as plot_progression.py)
for context. Bottom panel: per-(slug, turn) stacked horizontal bar of tier
fractions, perfect→weak left-to-right.

Usage:
    plot_tiers.py --slugs SLUG [SLUG ...]
    plot_tiers.py --task toml-1.0-lua --slugs SLUG1,SLUG2,...
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


TIER_PERFECT = "perfect"
TIER_STRONG = "strong"
TIER_MEDIUM = "medium"
TIER_WEAK = "weak"
TIER_ORDER = [TIER_PERFECT, TIER_STRONG, TIER_MEDIUM, TIER_WEAK]
TIER_COLORS = {
    TIER_PERFECT: "#2c7a2c",  # dark green
    TIER_STRONG:  "#94d495",  # light green
    TIER_MEDIUM:  "#f0d264",  # yellow
    TIER_WEAK:    "#d65454",  # red
}
TIER_LABELS = {
    TIER_PERFECT: "perfect (MCC = 1.0)",
    TIER_STRONG:  "strong (0.95 ≤ MCC < 1.0)",
    TIER_MEDIUM:  "medium (0.75 ≤ MCC < 0.95)",
    TIER_WEAK:    "weak (MCC < 0.75 / failure)",
}


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


def _twelve_distinct_colors(n: int) -> list:
    cmap = plt.get_cmap("tab20")
    even = [cmap(i) for i in range(0, 20, 2)]
    odd = [cmap(i) for i in range(1, 20, 2)]
    return (even + odd)[:n]


def plot_tiers(progression: dict[str, np.ndarray], task: str, slugs: list[str],
               output: Path, max_turns: int):
    available = [s for s in slugs if s in progression]
    missing_slugs = [s for s in slugs if s not in progression]
    for s in missing_slugs:
        print(f"  no data for {s} on task {task}, skipping")
    if not available:
        print("No matching data for any requested slug.")
        return

    color_map = dict(zip(available, _twelve_distinct_colors(len(available))))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 3], "hspace": 0.10},
    )
    xs = np.arange(1, max_turns + 1)

    # ===== TOP PANEL: mean trajectory =====
    endpoints: list[tuple[str, int, float, float, tuple]] = []
    for slug in available:
        attempts = progression[slug]
        n = attempts.shape[0]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_mcc = np.nanmean(attempts, axis=0)
        submit_rate = np.sum(~np.isnan(attempts), axis=0) / n
        color = color_map[slug]

        ax_top.plot(xs, mean_mcc, color=color, linewidth=1.8, alpha=0.85, zorder=2)
        for x, y, rate in zip(xs, mean_mcc, submit_rate):
            if np.isnan(y):
                continue
            ax_top.scatter([x], [y], color=color, s=70,
                           alpha=0.25 + 0.75 * rate,
                           edgecolors="white", linewidths=0.8, zorder=3)

        last_idx = max(i for i, v in enumerate(mean_mcc) if not np.isnan(v))
        endpoints.append((slug, n, float(xs[last_idx]),
                          float(mean_mcc[last_idx]), color))

    MIN_GAP = 0.035
    LABEL_X = max_turns + 0.25
    endpoints.sort(key=lambda e: -e[3])
    last_y = float("inf")
    for slug, n, ex, ey, color in endpoints:
        ly = min(ey, last_y - MIN_GAP)
        last_y = ly
        ax_top.plot([ex, LABEL_X - 0.05], [ey, ly], color=color, linewidth=0.6,
                    alpha=0.5, zorder=1)
        ax_top.text(LABEL_X, ly, f"{slug} (n={n})",
                    fontsize=8, color=color, va="center")

    ax_top.set_xlim(0.5, max_turns + 3.5)
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.set_xticks(xs)
    ax_top.set_ylabel("Mean best MCC")
    ax_top.set_title(f"Per-turn MCC tier distribution — {task}")
    ax_top.axhline(0, color="#888", linewidth=0.5, zorder=1)
    # Faint horizontal bands at the tier boundaries to anchor the eye.
    for boundary in (0.75, 0.95, 1.0):
        ax_top.axhline(boundary, color="#bbb", linewidth=0.5,
                       linestyle="--", zorder=1)
    ax_top.grid(alpha=0.3, zorder=0)

    # ===== BOTTOM PANEL: per-(slug, turn) tier distribution =====
    def _avg_at_last(slug: str) -> float:
        col = progression[slug][:, -1]
        col = col[~np.isnan(col)]
        return float(np.mean(col)) if len(col) else -np.inf

    bot_slugs = sorted(available, key=lambda s: -_avg_at_last(s))

    BAR_HEIGHT = 0.78
    BAR_WIDTH = 0.85
    for row_idx, slug in enumerate(bot_slugs):
        attempts = progression[slug]
        n = attempts.shape[0]
        for turn in range(max_turns):
            counts = {t: 0 for t in TIER_ORDER}
            for a in range(n):
                counts[tier_of(attempts[a, turn])] += 1
            x_left = (turn + 1) - BAR_WIDTH / 2
            cur = x_left
            for t in TIER_ORDER:
                w = (counts[t] / n) * BAR_WIDTH
                if w > 0:
                    ax_bot.barh(row_idx, w, left=cur, height=BAR_HEIGHT,
                                color=TIER_COLORS[t], edgecolor="white",
                                linewidth=0.4)
                cur += w

    ax_bot.set_yticks(range(len(bot_slugs)))
    ax_bot.set_yticklabels(
        [f"{s} (n={len(progression[s])})" for s in bot_slugs], fontsize=8)
    ax_bot.set_xticks(xs)
    ax_bot.set_xlim(0.5, max_turns + 3.5)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("Turn N")
    ax_bot.tick_params(axis="y", length=0)
    ax_bot.spines[["right", "top"]].set_visible(False)

    legend_elements = [
        Patch(facecolor=TIER_COLORS[t], label=TIER_LABELS[t])
        for t in TIER_ORDER
    ]
    ax_bot.legend(handles=legend_elements, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.95,
                  ncol=1, title="MCC tier at turn N")

    fig.subplots_adjust(left=0.21, right=0.97, top=0.94, bottom=0.06)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser(
        description="Two-panel per-turn MCC tier-distribution chart.")
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
