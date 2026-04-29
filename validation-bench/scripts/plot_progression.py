#!/usr/bin/env python
"""Per-turn MCC progression for a hand-picked set of slugs (two-panel chart).

Top panel: mean "best MCC after N turns" across attempts (N = 1..max_turns).
Lines that climb steeply benefit from the multi-turn submit-and-fix loop;
flat-from-N=1 lines are one-shotters.

Bottom panel: per-slug, per-turn outcome breakdown — what fraction of
attempts had which outcome at each turn. Categories:
- submitted: this turn produced an MCC (regardless of value).
- compile error: this turn's submission failed to build/parse.
- completed early: no row this turn because a prior turn already hit
  MCC=1.0 and the loop broke (this is a *good* outcome).
- not submitted: no row this turn and no prior perfect submission. Lumps
  several causes (model overthought, ran out of max_tokens, api_error
  mid-attempt, never called the submit tool, etc.) — these are not
  separable from results.jsonl alone.

The top panel's mean is taken only over attempts whose cumulative best
is non-NaN (i.e. that have submitted at least once by turn N), so the
two failure modes "model produced a bad answer" and "model produced no
answer" are kept apart. The bottom panel surfaces the second one.

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
from matplotlib.patches import Patch


OUTCOME_SUBMITTED = "submitted"
OUTCOME_ERROR = "error"
OUTCOME_DONE = "done"
OUTCOME_MISSING = "missing"
OUTCOME_ORDER = [OUTCOME_SUBMITTED, OUTCOME_ERROR, OUTCOME_DONE, OUTCOME_MISSING]
OUTCOME_COLORS = {
    OUTCOME_SUBMITTED: "#5cb85c",  # green
    OUTCOME_ERROR:     "#f0ad4e",  # orange
    OUTCOME_DONE:      "#5bc0de",  # sky blue (early-completed: good)
    OUTCOME_MISSING:   "#cccccc",  # gray
}
OUTCOME_LABELS = {
    OUTCOME_SUBMITTED: "submitted",
    OUTCOME_ERROR:     "compile error",
    OUTCOME_DONE:      "completed early (perfect)",
    OUTCOME_MISSING:   "not submitted",
}


def load_progression(results_file: Path, task: str, slugs: list[str], max_turns: int
                     ) -> dict[str, tuple[np.ndarray, list[list[str]]]]:
    """Read results.jsonl → {slug: (mcc_matrix, outcomes_list)}.

    mcc_matrix: shape (n_attempts, max_turns), best-MCC-after-N for each
                attempt (NaN until first submission).
    outcomes_list: per-attempt list of length max_turns, each element one
                   of OUTCOME_*.
    """
    slug_set = set(slugs)
    seen: dict[tuple[str, str], dict] = {}
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
        rec = seen.get(key)
        if rec is None:
            rec = {"mccs": [np.nan] * max_turns,
                   "errs": [None] * max_turns}
            seen[key] = rec
        if (mcc := r.get("mcc")) is not None:
            rec["mccs"][turn] = mcc
        if (err := r.get("error")) is not None:
            rec["errs"][turn] = err

    by_slug_mcc: dict[str, list[list[float]]] = defaultdict(list)
    by_slug_outcomes: dict[str, list[list[str]]] = defaultdict(list)
    for (_, slug), rec in seen.items():
        mccs = rec["mccs"]
        errs = rec["errs"]

        # Cumulative best so the top-panel line carries forward after a
        # successful submission, including past the early-break point.
        best = -np.inf
        bests: list[float] = []
        for m in mccs:
            if not np.isnan(m):
                best = max(best, m)
            bests.append(np.nan if best == -np.inf else best)
        by_slug_mcc[slug].append(bests)

        # Earliest turn that hit MCC=1.0 — once that happens, the runner
        # breaks the attempt loop, so subsequent missing rows are *good*
        # outcomes ("completed early"), not failures.
        won_at = next((i for i, m in enumerate(mccs)
                       if not np.isnan(m) and m >= 0.999999), None)

        outcomes: list[str] = []
        for t in range(max_turns):
            if not np.isnan(mccs[t]):
                outcomes.append(OUTCOME_SUBMITTED)
            elif errs[t] is not None:
                outcomes.append(OUTCOME_ERROR)
            elif won_at is not None and t > won_at:
                outcomes.append(OUTCOME_DONE)
            else:
                outcomes.append(OUTCOME_MISSING)
        by_slug_outcomes[slug].append(outcomes)

    return {
        s: (np.array(by_slug_mcc[s]), by_slug_outcomes[s])
        for s in slugs if s in by_slug_mcc
    }


def _twelve_distinct_colors(n: int) -> list:
    """tab20 even indices (10 saturated colors) then odd indices for >10 lines."""
    cmap = plt.get_cmap("tab20")
    even = [cmap(i) for i in range(0, 20, 2)]
    odd = [cmap(i) for i in range(1, 20, 2)]
    return (even + odd)[:n]


def plot_progression(progression: dict, task: str, slugs: list[str],
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
        gridspec_kw={"height_ratios": [3, 2.4], "hspace": 0.12},
    )
    xs = np.arange(1, max_turns + 1)

    # ===== TOP PANEL: MCC trajectory =====
    endpoints: list[tuple[str, int, float, float, tuple]] = []
    for slug in available:
        attempts_mcc, _ = progression[slug]
        n = attempts_mcc.shape[0]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_mcc = np.nanmean(attempts_mcc, axis=0)
        submit_rate = np.sum(~np.isnan(attempts_mcc), axis=0) / n
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

    # Stagger end-of-line labels so dense top-cluster doesn't overlap.
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
    ax_top.set_ylabel("Mean best MCC (over submitted attempts)")
    ax_top.set_title(f"Per-turn progression — {task}")
    ax_top.axhline(0, color="#888", linewidth=0.5, zorder=1)
    ax_top.grid(alpha=0.3, zorder=0)

    # ===== BOTTOM PANEL: per-(slug, turn) outcome breakdown =====
    # Sort rows by mean best-MCC at the last turn (descending) so the
    # reader scans top→bottom from strongest to weakest.
    def _avg_at_last(slug: str) -> float:
        col = progression[slug][0][:, -1]
        col = col[~np.isnan(col)]
        return float(np.mean(col)) if len(col) else -np.inf

    bot_slugs = sorted(available, key=lambda s: -_avg_at_last(s))

    BAR_HEIGHT = 0.78
    BAR_WIDTH = 0.85
    for row_idx, slug in enumerate(bot_slugs):
        _, outcomes_list = progression[slug]
        n = len(outcomes_list)
        for turn in range(max_turns):
            counts = {oc: 0 for oc in OUTCOME_ORDER}
            for outcomes in outcomes_list:
                counts[outcomes[turn]] += 1
            x_left = (turn + 1) - BAR_WIDTH / 2
            cur = x_left
            for oc in OUTCOME_ORDER:
                w = (counts[oc] / n) * BAR_WIDTH
                if w > 0:
                    ax_bot.barh(row_idx, w, left=cur, height=BAR_HEIGHT,
                                color=OUTCOME_COLORS[oc], edgecolor="white",
                                linewidth=0.4)
                cur += w

    ax_bot.set_yticks(range(len(bot_slugs)))
    ax_bot.set_yticklabels(
        [f"{s} (n={len(progression[s][1])})" for s in bot_slugs], fontsize=8)
    ax_bot.set_xticks(xs)
    ax_bot.set_xlim(0.5, max_turns + 3.5)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("Turn N")
    ax_bot.tick_params(axis="y", length=0)
    ax_bot.spines[["right", "top"]].set_visible(False)

    legend_elements = [
        Patch(facecolor=OUTCOME_COLORS[oc], label=OUTCOME_LABELS[oc])
        for oc in OUTCOME_ORDER
    ]
    ax_bot.legend(handles=legend_elements, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), fontsize=8, framealpha=0.95,
                  ncol=1, title="outcome at turn N")

    # tight_layout stumbles on the staggered top-panel labels (which sit
    # outside the axes); fix margins explicitly.
    fig.subplots_adjust(left=0.21, right=0.97, top=0.94, bottom=0.06)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser(
        description="Two-panel per-turn MCC progression chart.")
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

    here = Path(__file__).resolve().parent.parent  # validation-bench/
    results_file = Path(args.results) if args.results else here / "results" / "results.jsonl"
    output = Path(args.output) if args.output else here / "results" / "plots" / f"progression-{args.task}.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    progression = load_progression(results_file, args.task, slugs, args.max_turns)
    plot_progression(progression, args.task, slugs, output, args.max_turns)


if __name__ == "__main__":
    main()
