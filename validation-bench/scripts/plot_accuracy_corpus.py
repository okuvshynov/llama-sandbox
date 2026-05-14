#!/usr/bin/env python
"""Render a static PNG of Accuracy curves with corpus-only CI bands.

Mirrors the Streamlit `Accuracy (corpus)` page (viewer.py) for a fixed
slug/task selection so the figure can be embedded in writeups or
shared without firing up Streamlit.

Reuses `build_leaderboard` from viewer.py — same point estimates, same
1-step bootstrap CI (Stage 2 only, no task resampling). Interpretation:
"if I reran this same sweep on the same corpus with fresh attempts,
the macro-avg would land in [lo, hi] in 95% of reruns."

Usage:
    python plot_accuracy_corpus.py [--results PATH] [--out PATH]
                                   [--no-intersection] [--ci-B N]

Edit SLUGS / TASKS below to change the selection.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress the "No runtime found" warning Streamlit's cache emits when
# we import viewer.py outside a Streamlit session. Doesn't affect any
# computation — load_results is the only @cache_data-decorated symbol
# and we don't call it from this script.
warnings.filterwarnings("ignore", message="No runtime found")

# Reuse viewer.py's aggregation. Path-insert lets us import a sibling
# module without making `scripts/` a package.
sys.path.insert(0, str(Path(__file__).parent))
from viewer import build_leaderboard, _tau_col  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RESULTS = (Path(__file__).parent.parent / "results" / "results.jsonl")
DEFAULT_OUT = (Path(__file__).parent.parent / "results" / "plots"
               / "accuracy_corpus.png")

SLUGS = [
    "anthropic-claude-opus-4-7-adaptive",
    "anthropic-claude-sonnet-4-6-enabled",
    "deepseek-v4-pro-thinking",
    "fireworks-glm-5p1",
    "gpt-5.5-xhigh",
    "moonshot-kimi-k2.6-thinking",
]

TASKS = [
    "hcl-2-cpp17", "hcl-2-nospec-cpp17", "hcl-2-nospec-zig", "hcl-2-zig",
    "toml-1.0-cpp17", "toml-1.0-nospec-cpp17",
    "toml-1.0-nospec-zig", "toml-1.0-zig",
    "yaml-1.2-cpp17", "yaml-1.2-nospec-cpp17",
    "yaml-1.2-nospec-zig", "yaml-1.2-zig",
]

# 101-point grid (0.01 step) — same resolution the Accuracy (corpus)
# Streamlit page uses; the 0.9-1.0 region is where models actually
# differentiate, and 21 points there is too coarse.
TAUS = [round(0.01 * i, 2) for i in range(101)]


def _short(slug: str) -> str:
    """Display label: trim provider prefix while keeping the last
    distinguishing segment (e.g. '-adaptive', '-thinking')."""
    parts = slug.split("-")
    # Drop the provider/model prefix; keep the trailing variant token
    # plus the model series so e.g.
    #   anthropic-claude-opus-4-7-adaptive  → opus-4-7-adaptive
    #   moonshot-kimi-k2.6-thinking         → kimi-k2.6-thinking
    if parts[0] in {"anthropic", "moonshot", "fireworks", "accounts"}:
        return "-".join(parts[1:])[len("claude-"):] if parts[0] == "anthropic" \
               else "-".join(parts[1:])
    return slug


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Mid-write tail line — skip.
                continue
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                   help=f"Path to results.jsonl (default: {DEFAULT_RESULTS})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output PNG path (default: {DEFAULT_OUT})")
    p.add_argument("--no-intersection", action="store_true",
                   help="Don't restrict to tasks every selected model has "
                        "attempted (default: intersection ON for fair "
                        "cross-model comparison).")
    p.add_argument("--ci-B", type=int, default=500,
                   help="Bootstrap iterations for CI (default: 500).")
    p.add_argument("--shrinkage", action="store_true",
                   help="Use Beta-Binomial shrinkage instead of raw "
                        "macro-average (off by default).")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if not args.results.exists():
        print(f"error: results file not found: {args.results}", file=sys.stderr)
        return 1

    df = load_results(args.results)
    print(f"Loaded {len(df):,} rows from {args.results}")

    # Synthesize the per-row accuracy column (same logic as
    # render_accuracy_corpus_curves_page in viewer.py): mean of
    # (TP + TN) / (TP + FN + FP + TN), NaN-safe for non-scored rows.
    cm_cols = ["tp", "fn", "fp", "tn"]
    if not all(c in df.columns for c in cm_cols):
        print(f"error: missing confusion-matrix columns; need {cm_cols}",
              file=sys.stderr)
        return 1
    cm_total = df[cm_cols].sum(axis=1, min_count=4)
    df["accuracy"] = np.where(cm_total > 0,
                              (df["tp"] + df["tn"]) / cm_total,
                              np.nan)

    lb, lb_lo, lb_hi, tasks_used = build_leaderboard(
        df, SLUGS, TASKS, TAUS,
        intersection=not args.no_intersection,
        shrunken=args.shrinkage,
        ci_B=args.ci_B,
        score_col="accuracy",
        ci_corpus_only=True,
    )

    if not tasks_used:
        print("error: no tasks survived the intersection filter — try "
              "--no-intersection or trim the slug set.", file=sys.stderr)
        return 1

    print(f"Aggregated over {len(tasks_used)} task(s): "
          f"{', '.join(tasks_used)}")
    print(f"Models in plot: {len(lb)}")

    # Sort rows by point estimate at τ=0.5 desc — visually puts the
    # strongest model on top of the legend.
    sort_key = _tau_col(0.50)
    lb_sorted_idx = lb.sort_values(sort_key, ascending=False,
                                   na_position="last").index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # tab10 has 10 distinct colors; we use 6. Reproducible across runs.
    palette = plt.colormaps["tab10"].colors

    # Tabular summary — point estimates of P(accuracy ≥ τ) at the
    # τ values most worth reading off the curves: 0.9 (high-reliability),
    # 0.99 (near-perfect), 1.0 (perfect validator). τ=0.5 is excluded
    # because on a balanced valid/invalid corpus a constant classifier
    # trivially scores 0.5, so P(acc≥0.5) is uninformative noise.
    # Printed both as stdout (for piping/copying) and as a Markdown-
    # compatible block for direct paste into writeups.
    summary_taus = [0.90, 0.99, 1.00]
    print("\nHeadline pass rates (point estimates):")
    header = ("model".ljust(40)
              + "  ".join(f"P(acc≥{t:.2f})".rjust(11) for t in summary_taus)
              + "  support")
    print(header)
    print("-" * len(header))

    summary_rows: list[dict] = []
    for color_i, idx in enumerate(lb_sorted_idx):
        row = lb.loc[idx]
        slug = row["slug"]
        support = row["support"]
        points = np.array([row[_tau_col(t)] for t in TAUS], dtype=float)
        los = np.array([lb_lo.loc[idx][_tau_col(t)] for t in TAUS], dtype=float)
        his = np.array([lb_hi.loc[idx][_tau_col(t)] for t in TAUS], dtype=float)
        color = palette[color_i % len(palette)]
        ax.fill_between(TAUS, los, his, alpha=0.16, color=color, linewidth=0)
        ax.plot(TAUS, points, color=color, linewidth=2.0, label=_short(slug))

        vals = [float(row[_tau_col(t)]) for t in summary_taus]
        print(_short(slug).ljust(40)
              + "  ".join(f"{v:>11.3f}" for v in vals)
              + f"  {support}")
        summary_rows.append({
            "model": _short(slug), "slug": slug,
            **{f"P(acc≥{t:.2f})": v for t, v in zip(summary_taus, vals)},
            "support": support,
        })

    # Markdown variant — paste-ready for the writeup. Same numbers,
    # pipe-delimited so GitHub / Notion / pandoc render it as a table.
    print("\nMarkdown:")
    md_header = ("| model | "
                 + " | ".join(f"P(acc≥{t:.2f})" for t in summary_taus)
                 + " | support |")
    md_sep = "|" + "|".join(["---"] * (len(summary_taus) + 2)) + "|"
    print(md_header)
    print(md_sep)
    for r in summary_rows:
        # build_leaderboard formats support as "n_cells | n_attempts",
        # which collides with the Markdown pipe delimiter — swap to "/"
        # just for the rendered table.
        support_md = str(r["support"]).replace(" | ", " / ")
        cells = ([r["model"]]
                 + [f"{r[f'P(acc≥{t:.2f})']:.3f}" for t in summary_taus]
                 + [support_md])
        print("| " + " | ".join(cells) + " |")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold τ (fraction of tests correct)")
    ax.set_ylabel("P(accuracy ≥ τ)")
    ax.set_title(
        "Accuracy curves (corpus-only CI) — P(test pass rate ≥ τ) with rerun CI\n"
        f"{len(tasks_used)} tasks, "
        f"{'intersection' if not args.no_intersection else 'union'}, "
        f"{'shrunk' if args.shrinkage else 'raw'} macro-avg, "
        f"B={args.ci_B} • bands are 95% bootstrap CI under same-corpus rerun",
        fontsize=11,
    )
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92,
              title="model", title_fontsize=9)

    # Tiny task-list footer so the figure is self-contained.
    footer = "tasks: " + ", ".join(tasks_used)
    if len(footer) > 140:
        footer = footer[:137] + "..."
    fig.text(0.5, 0.005, footer, ha="center", va="bottom",
             fontsize=7, color="#555")

    plt.tight_layout(rect=(0, 0.02, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
