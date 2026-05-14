#!/usr/bin/env python
"""Render static PNGs + tables comparing validator difficulty by spec type.

Sibling of plot_cpp17_vs_zig.py, but the line dimension is the 3 spec
roots — hcl-2 vs toml-1.0 vs yaml-1.2 — instead of the 2 envs.

Two figures:

  by_spec_type_headline.png   — Single panel. Three macro accuracy
                                curves (one per spec root) pooled
                                across all (model, env, spec-variant)
                                cells, with 95% corpus-only bootstrap
                                CI bands.

  by_spec_type_breakdown.png  — 2×2 panel grid. Rows = env (cpp17 /
                                zig), columns = spec variant (with-
                                spec / nospec). Each panel overlays
                                the 3 spec-root curves with CI bands,
                                averaged across the same model set.

Both use the corpus-only CI mode (Stage 2 only, no task resampling) —
same interpretation as the Streamlit `Accuracy (corpus)` page: "if I
reran this same sweep on the same corpus with fresh attempts, the
macro-avg lands in [lo, hi] in 95% of reruns".

Stdout prints two tables in ASCII + Markdown:
  * Aggregate P(acc≥τ) per spec root at τ ∈ {0.9, 0.99, 1.0}.
  * Breakdown — same, sliced per (env, spec-variant).

Usage:
    python plot_by_spec_type.py [--results PATH] [--out-dir PATH]
                                [--ci-B N] [--dpi N]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="No runtime found")

sys.path.insert(0, str(Path(__file__).parent))
from viewer import _per_cell_counts, _bootstrap_macro_avg_ci  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_RESULTS = Path(__file__).parent.parent / "results" / "results.jsonl"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "results" / "plots"

SLUGS = [
    "anthropic-claude-opus-4-7-adaptive",
    "anthropic-claude-sonnet-4-6-enabled",
    "deepseek-v4-pro-thinking",
    "fireworks-glm-5p1",
    "gpt-5.5-xhigh",
    "moonshot-kimi-k2.6-thinking",
]

SPEC_ROOTS = ["hcl-2", "toml-1.0", "yaml-1.2"]
SPEC_VARIANTS = ["", "-nospec"]
ENVS = ["cpp17", "zig"]

TAUS = [round(0.01 * i, 2) for i in range(101)]
SUMMARY_TAUS = [0.90, 0.99, 1.00]

# One color per spec root — tab10's blue / orange / green, all
# color-vision-friendly.
PALETTE = {"hcl-2": "#2ca02c", "toml-1.0": "#1f77b4", "yaml-1.2": "#ff7f0e"}


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
                continue
    return pd.DataFrame(rows)


def add_accuracy_col(df: pd.DataFrame) -> pd.DataFrame:
    cm_cols = ["tp", "fn", "fp", "tn"]
    cm_total = df[cm_cols].sum(axis=1, min_count=4)
    df["accuracy"] = np.where(cm_total > 0,
                              (df["tp"] + df["tn"]) / cm_total,
                              np.nan)
    return df


def _cells_for(df: pd.DataFrame, slugs: list[str], tasks: list[str],
               taus: list[float]) -> list[tuple]:
    """Per-cell (n_t, {τ: k_t}) tuples for every (slug, task) with data."""
    sub = df[df["slug"].isin(slugs) & df["task"].isin(tasks)]
    out = []
    for slug in slugs:
        for task in tasks:
            cell_df = sub[(sub.slug == slug) & (sub.task == task)]
            if cell_df.empty:
                continue
            out.append(_per_cell_counts(cell_df, taus, score_col="accuracy"))
    return out


def _macro_with_ci(cells: list[tuple], taus: list[float],
                   B: int, seed_base: int
                   ) -> tuple[dict, dict, dict]:
    """Returns ({τ: point}, {τ: lo}, {τ: hi}) under corpus-only bootstrap."""
    if not cells:
        nan = {t: float("nan") for t in taus}
        return nan, dict(nan), dict(nan)
    n_arr = [n for n, _ in cells]
    point: dict[float, float] = {}
    lo: dict[float, float] = {}
    hi: dict[float, float] = {}
    for ti, tau in enumerate(taus):
        k_arr = [k_at[tau] for _, k_at in cells]
        rates = [k / n if n > 0 else 0.0 for k, n in zip(k_arr, n_arr)]
        point[tau] = sum(rates) / len(rates)
        l, h = _bootstrap_macro_avg_ci(
            k_arr, n_arr, shrunken=False, B=B, ci_level=0.95,
            seed=seed_base + ti, corpus_only=True,
        )
        lo[tau] = l
        hi[tau] = h
    return point, lo, hi


def _slugs_with_all(df: pd.DataFrame, slugs: list[str],
                    tasks: list[str]) -> list[str]:
    """Slugs that have ≥1 attempt on EVERY task in `tasks`. Keeps the
    pooled curves apples-to-apples — same model set behind each line."""
    out = []
    for s in slugs:
        if all(not df[(df.slug == s) & (df.task == t)].empty for t in tasks):
            out.append(s)
    return out


def plot_headline(curves: dict, n_cells: dict, out_path: Path,
                  dpi: int) -> None:
    """curves[root] = (point, (lo, hi)); n_cells[root] = int."""
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for root in SPEC_ROOTS:
        if root not in curves:
            continue
        point, (lo, hi) = curves[root]
        ys = [point[t] for t in TAUS]
        los = [lo[t] for t in TAUS]
        his = [hi[t] for t in TAUS]
        color = PALETTE[root]
        ax.fill_between(TAUS, los, his, alpha=0.16, color=color, linewidth=0)
        ax.plot(TAUS, ys, color=color, linewidth=2.2,
                label=f"{root}  ({n_cells.get(root, 0)} cells)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold τ (fraction of tests correct)")
    ax.set_ylabel("P(accuracy ≥ τ)")
    ax.set_title(
        "Validator difficulty by spec type — macro accuracy curves with corpus-only CI\n"
        f"{len(SLUGS)} models × {len(ENVS)} envs × {len(SPEC_VARIANTS)} spec-variants, "
        "bands are 95% bootstrap CI under same-corpus rerun",
        fontsize=11,
    )
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92,
              title="spec root", title_fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(panel_data: dict, out_path: Path, dpi: int) -> None:
    """panel_data[(env, variant)] = {root: (point, (lo, hi), n_cells)}."""
    fig, axes = plt.subplots(len(ENVS), len(SPEC_VARIANTS),
                             figsize=(11, 8), sharex=True, sharey=True)
    for i, env in enumerate(ENVS):
        for j, variant in enumerate(SPEC_VARIANTS):
            ax = axes[i, j]
            curves = panel_data.get((env, variant), {})
            for root in SPEC_ROOTS:
                if root not in curves:
                    continue
                point, (lo, hi), n = curves[root]
                ys = [point[t] for t in TAUS]
                los = [lo[t] for t in TAUS]
                his = [hi[t] for t in TAUS]
                color = PALETTE[root]
                ax.fill_between(TAUS, los, his, alpha=0.16,
                                color=color, linewidth=0)
                ax.plot(TAUS, ys, color=color, linewidth=1.8,
                        label=f"{root} (n={n})")
            variant_label = "with-spec" if variant == "" else "nospec"
            ax.set_title(f"{env} · {variant_label}", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.4)
            if i == len(ENVS) - 1:
                ax.set_xlabel("τ")
            if j == 0:
                ax.set_ylabel("P(acc ≥ τ)")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.suptitle(
        "Validator difficulty by spec type — per env × spec-variant, corpus-only CI",
        fontsize=12, y=0.997,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_aggregate_table(curves: dict, summary_taus: list[float]) -> None:
    print("\n=== Aggregate: P(accuracy ≥ τ) by spec root "
          "(macro across all envs × spec-variants × models) ===")
    header = f"{'spec root':<12s}  " + "  ".join(
        f"P(acc≥{t:.2f})".rjust(11) for t in summary_taus)
    print(header)
    print("-" * len(header))
    md = ["| spec root | " + " | ".join(
        f"P(acc≥{t:.2f})" for t in summary_taus) + " |",
        "|" + "|".join(["---"] * (len(summary_taus) + 1)) + "|"]
    for root in SPEC_ROOTS:
        point = curves.get(root, ({},))[0]
        vals = [point.get(t, float("nan")) for t in summary_taus]
        print(f"{root:<12s}  " + "  ".join(f"{v:>11.3f}" for v in vals))
        md.append(f"| {root} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    print("\nMarkdown:")
    print("\n".join(md))


def print_breakdown_table(panel_data: dict, summary_taus: list[float]) -> None:
    print(f"\n=== Breakdown: P(accuracy ≥ τ) by (env, spec-variant, spec root) ===")
    header = (f"{'env':<6s} {'variant':<10s} {'spec root':<11s}  "
              + "  ".join(f"P(acc≥{t:.2f})".rjust(11) for t in summary_taus))
    print(header)
    print("-" * len(header))
    md = ["| env | variant | spec root | " + " | ".join(
        f"P(acc≥{t:.2f})" for t in summary_taus) + " |",
        "|" + "|".join(["---"] * (3 + len(summary_taus))) + "|"]
    for env in ENVS:
        for variant in SPEC_VARIANTS:
            variant_label = "with-spec" if variant == "" else "nospec"
            curves = panel_data.get((env, variant), {})
            for root in SPEC_ROOTS:
                point = curves.get(root, ({},))[0] if root in curves else {}
                vals = [point.get(t, float("nan")) for t in summary_taus]
                print(f"{env:<6s} {variant_label:<10s} {root:<11s}  "
                      + "  ".join(f"{v:>11.3f}" for v in vals))
                md.append(f"| {env} | {variant_label} | {root} | "
                          + " | ".join(f"{v:.3f}" for v in vals) + " |")
    print("\nMarkdown:")
    print("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--ci-B", type=int, default=500)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if not args.results.exists():
        print(f"error: {args.results} not found", file=sys.stderr)
        return 1

    df = add_accuracy_col(load_results(args.results))
    print(f"Loaded {len(df):,} rows from {args.results}")

    # ===== HEADLINE: one curve per spec root, pooled across env ×
    # spec-variant. Per-root intersection so the same model set backs
    # every cell in that root's pool.
    headline_curves: dict = {}
    headline_n: dict = {}
    for root in SPEC_ROOTS:
        tasks = [f"{root}{v}-{e}" for v in SPEC_VARIANTS for e in ENVS]
        slugs_used = _slugs_with_all(df, SLUGS, tasks)
        cells = _cells_for(df, slugs_used, tasks, TAUS)
        pt, lo, hi = _macro_with_ci(
            cells, TAUS, B=args.ci_B, seed_base=10_000 + hash(root) % 9000)
        headline_curves[root] = (pt, (lo, hi))
        headline_n[root] = len(cells)
        print(f"Headline — {root:<10s}: {len(cells)} cells "
              f"({len(tasks)} tasks × {len(slugs_used)} slugs)")

    headline_path = args.out_dir / "by_spec_type_headline.png"
    plot_headline(headline_curves, headline_n, headline_path, args.dpi)
    print(f"Wrote {headline_path}")

    print_aggregate_table(headline_curves, SUMMARY_TAUS)

    # ===== BREAKDOWN: 2×2 grid, panel = (env, spec-variant), 3 lines.
    # Per-panel intersection across the 3 spec roots so the panel's
    # lines share a model set.
    panel_data: dict = {}
    for env in ENVS:
        for variant in SPEC_VARIANTS:
            panel_tasks = [f"{r}{variant}-{env}" for r in SPEC_ROOTS]
            slugs_used = _slugs_with_all(df, SLUGS, panel_tasks)
            curves: dict = {}
            for root in SPEC_ROOTS:
                task = f"{root}{variant}-{env}"
                cells = _cells_for(df, slugs_used, [task], TAUS)
                if not cells:
                    continue
                pt, lo, hi = _macro_with_ci(
                    cells, TAUS, B=args.ci_B,
                    seed_base=40_000 + (hash((env, variant, root)) & 0xFFFF))
                curves[root] = (pt, (lo, hi), len(cells))
            panel_data[(env, variant)] = curves

    breakdown_path = args.out_dir / "by_spec_type_breakdown.png"
    plot_breakdown(panel_data, breakdown_path, args.dpi)
    print(f"\nWrote {breakdown_path}")

    # Drop the n_cells element so the table helper sees the same
    # (point, (lo, hi)) shape the headline curves use.
    panel_for_table = {
        key: {root: (c[0], c[1]) for root, c in curves.items()}
        for key, curves in panel_data.items()
    }
    print_breakdown_table(panel_for_table, SUMMARY_TAUS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
