#!/usr/bin/env python
"""Render static PNGs + tables comparing C++17 vs Zig validator difficulty.

Two figures:

  cpp17_vs_zig_headline.png  — Single panel. Two macro accuracy curves
                               (cpp17 vs zig) pooled across all
                               (model, spec) cells in the selection,
                               with 95% corpus-only bootstrap CI bands.

  cpp17_vs_zig_per_spec.png  — 3×2 panel grid (one panel per spec):
                               hcl-2 / hcl-2-nospec / toml-1.0 /
                               toml-1.0-nospec / yaml-1.2 /
                               yaml-1.2-nospec. Each panel overlays
                               cpp17 vs zig macro curves with CI bands
                               averaged across the same model set.

Both use the corpus-only CI mode (Stage 2 only, no task resampling) —
same interpretation as the Streamlit `Accuracy (corpus)` page.

Stdout prints two tables in ASCII + Markdown:
  * Aggregate Δ at τ ∈ {0.9, 0.99, 1.0} — for the headline.
  * Per-spec Δ at the same τ values — for the per-spec view.

Usage:
    python plot_cpp17_vs_zig.py [--results PATH] [--out-dir PATH]
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

# Spec/env labels expanded out so the order is deterministic and the
# task names match what's in results.jsonl.
ALL_SPECS = [f"{r}{v}" for r in SPEC_ROOTS for v in SPEC_VARIANTS]
ALL_TASKS = [f"{s}-{e}" for s in ALL_SPECS for e in ENVS]

TAUS = [round(0.01 * i, 2) for i in range(101)]
SUMMARY_TAUS = [0.90, 0.99, 1.00]

# Match the Streamlit `Accuracy (corpus)` palette intent: blue for
# cpp17, red for zig. Both are color-vision-friendly.
PALETTE = {"cpp17": "#1f77b4", "zig": "#d62728"}


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


def _slugs_with_both(df: pd.DataFrame, slugs: list[str], spec: str
                     ) -> list[str]:
    """Slugs that have ≥1 attempt on BOTH (spec, cpp17) AND (spec, zig).
    Used to keep the per-spec panels apples-to-apples — the same model
    set on both env curves."""
    cpp_task = f"{spec}-cpp17"
    zig_task = f"{spec}-zig"
    out = []
    for s in slugs:
        has_cpp = not df[(df.slug == s) & (df.task == cpp_task)].empty
        has_zig = not df[(df.slug == s) & (df.task == zig_task)].empty
        if has_cpp and has_zig:
            out.append(s)
    return out


def _filter_to_intersection(df: pd.DataFrame, slugs: list[str],
                            tasks: list[str]) -> list[str]:
    """Tasks where every slug has ≥1 attempt — same semantics as the
    Streamlit pages."""
    present = (df.groupby(["slug", "task"]).size()
               .unstack(fill_value=0) > 0)
    present = present.reindex(index=slugs, columns=tasks, fill_value=False)
    return [t for t in tasks if bool(present[t].all())]


def plot_headline(point_by_env: dict, ci_by_env: dict,
                  n_cells_by_env: dict, tasks_by_env: dict,
                  out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for env in ENVS:
        point = point_by_env[env]
        lo, hi = ci_by_env[env]
        ys = [point[t] for t in TAUS]
        los = [lo[t] for t in TAUS]
        his = [hi[t] for t in TAUS]
        color = PALETTE[env]
        ax.fill_between(TAUS, los, his, alpha=0.18, color=color, linewidth=0)
        ax.plot(TAUS, ys, color=color, linewidth=2.2,
                label=f"{env}  ({n_cells_by_env[env]} cells, "
                      f"{len(tasks_by_env[env])} tasks)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold τ (fraction of tests correct)")
    ax.set_ylabel("P(accuracy ≥ τ)")
    ax.set_title(
        "C++17 vs Zig — macro accuracy curves with corpus-only CI\n"
        f"{len(SLUGS)} models, raw macro-avg, bands are 95% bootstrap CI under same-corpus rerun",
        fontsize=11,
    )
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92,
              title="environment", title_fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_per_spec(spec_data: dict, out_path: Path, dpi: int) -> None:
    """spec_data[spec] = (env_results, n_per_env)
    env_results[env] = (point, (lo, hi)); n_per_env[env] = int."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5),
                             sharex=True, sharey=True)
    for i, root in enumerate(SPEC_ROOTS):
        for j, variant in enumerate(SPEC_VARIANTS):
            spec = f"{root}{variant}"
            ax = axes[i, j]
            env_results, n_per_env = spec_data.get(spec, ({}, {}))
            for env in ENVS:
                if env not in env_results:
                    continue
                point, (lo, hi) = env_results[env]
                ys = [point[t] for t in TAUS]
                los = [lo[t] for t in TAUS]
                his = [hi[t] for t in TAUS]
                color = PALETTE[env]
                ax.fill_between(TAUS, los, his, alpha=0.18,
                                color=color, linewidth=0)
                ax.plot(TAUS, ys, color=color, linewidth=1.8,
                        label=f"{env} (n={n_per_env.get(env, 0)})")
            ax.set_title(spec, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.4)
            if i == len(SPEC_ROOTS) - 1:
                ax.set_xlabel("τ")
            if j == 0:
                ax.set_ylabel("P(acc ≥ τ)")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.suptitle(
        "C++17 vs Zig — per-spec macro accuracy curves with corpus-only CI",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_aggregate_table(point_by_env: dict, summary_taus: list[float],
                          tasks_by_env: dict) -> None:
    print("\n=== Aggregate: C++17 vs Zig (macro across all selected cells) ===")
    print(f"{'metric':<13s}  {'cpp17':>9s}  {'zig':>9s}  {'Δ (cpp−zig)':>13s}")
    print("-" * 50)
    md = ["| metric | cpp17 | zig | Δ (cpp17 − zig) |",
          "|---|---|---|---|"]
    for tau in summary_taus:
        c = point_by_env["cpp17"][tau]
        z = point_by_env["zig"][tau]
        d = c - z
        print(f"P(acc≥{tau:.2f})    {c:>9.3f}  {z:>9.3f}  {d:>+13.3f}")
        md.append(f"| P(acc≥{tau:.2f}) | {c:.3f} | {z:.3f} | {d:+.3f} |")
    print(f"\ncpp17 tasks: {', '.join(tasks_by_env['cpp17'])}")
    print(f"zig tasks:   {', '.join(tasks_by_env['zig'])}")
    print("\nMarkdown:")
    print("\n".join(md))


def print_per_spec_table(spec_data: dict, summary_taus: list[float]) -> None:
    print(f"\n=== Per-spec breakdown at τ ∈ "
          f"{{{', '.join(f'{t:.2f}' for t in summary_taus)}}} ===")
    header = f"{'spec':<18s}  " + "  ".join(
        f"{'cpp17':>7s} {'zig':>6s} {'Δ':>7s}  @τ={t:.2f}"
        for t in summary_taus
    )
    print(header)
    print("-" * len(header))

    md_cols = ["spec"]
    for t in summary_taus:
        md_cols += [f"cpp17 (τ={t:.2f})", f"zig (τ={t:.2f})", f"Δ (τ={t:.2f})"]
    md = ["| " + " | ".join(md_cols) + " |",
          "|" + "|".join(["---"] * len(md_cols)) + "|"]

    for spec in ALL_SPECS:
        env_results = spec_data.get(spec, ({}, {}))[0]
        ascii_parts = []
        md_cells = [spec]
        for t in summary_taus:
            c = (env_results.get("cpp17", ({},))[0].get(t)
                 if "cpp17" in env_results else float("nan"))
            z = (env_results.get("zig", ({},))[0].get(t)
                 if "zig" in env_results else float("nan"))
            d = (c - z) if (c is not None and z is not None
                            and not np.isnan(c) and not np.isnan(z)) else float("nan")
            ascii_parts.append(f"{c:>7.3f} {z:>6.3f} {d:>+7.3f}        ")
            md_cells += [f"{c:.3f}", f"{z:.3f}", f"{d:+.3f}"]
        print(f"{spec:<18s}  " + "  ".join(p.rstrip() for p in ascii_parts))
        md.append("| " + " | ".join(md_cells) + " |")
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

    # ===== HEADLINE: pool across all spec roots per env =====
    cpp17_tasks_all = [t for t in ALL_TASKS if t.endswith("-cpp17")]
    zig_tasks_all = [t for t in ALL_TASKS if t.endswith("-zig")]
    # Intersection per env: every selected slug must have data on every
    # task in the env's task set. Keeps the macro apples-to-apples.
    cpp17_tasks = _filter_to_intersection(df, SLUGS, cpp17_tasks_all)
    zig_tasks = _filter_to_intersection(df, SLUGS, zig_tasks_all)

    cpp17_cells = _cells_for(df, SLUGS, cpp17_tasks, TAUS)
    zig_cells = _cells_for(df, SLUGS, zig_tasks, TAUS)
    print(f"Headline cells — cpp17: {len(cpp17_cells)} "
          f"(over {len(cpp17_tasks)} tasks × {len(SLUGS)} slugs)")
    print(f"Headline cells — zig:   {len(zig_cells)} "
          f"(over {len(zig_tasks)} tasks × {len(SLUGS)} slugs)")

    cpp17_point, cpp17_lo, cpp17_hi = _macro_with_ci(
        cpp17_cells, TAUS, B=args.ci_B, seed_base=10_000)
    zig_point, zig_lo, zig_hi = _macro_with_ci(
        zig_cells, TAUS, B=args.ci_B, seed_base=20_000)

    headline_path = args.out_dir / "cpp17_vs_zig_headline.png"
    plot_headline(
        {"cpp17": cpp17_point, "zig": zig_point},
        {"cpp17": (cpp17_lo, cpp17_hi), "zig": (zig_lo, zig_hi)},
        {"cpp17": len(cpp17_cells), "zig": len(zig_cells)},
        {"cpp17": cpp17_tasks, "zig": zig_tasks},
        headline_path, args.dpi,
    )
    print(f"Wrote {headline_path}")

    print_aggregate_table(
        {"cpp17": cpp17_point, "zig": zig_point}, SUMMARY_TAUS,
        {"cpp17": cpp17_tasks, "zig": zig_tasks},
    )

    # ===== PER-SPEC: one panel per spec, intersected per spec =====
    spec_data: dict = {}
    for spec in ALL_SPECS:
        # Use only slugs that have data on BOTH cpp17 and zig variants
        # so the panel's two curves are based on the same model set.
        slugs_used = _slugs_with_both(df, SLUGS, spec)
        if not slugs_used:
            continue
        env_results: dict = {}
        n_per_env: dict = {}
        for env in ENVS:
            task = f"{spec}-{env}"
            cells = _cells_for(df, slugs_used, [task], TAUS)
            if not cells:
                continue
            pt, lo, hi = _macro_with_ci(
                cells, TAUS, B=args.ci_B,
                seed_base=30_000 + (hash((spec, env)) & 0xFFFF),
            )
            env_results[env] = (pt, (lo, hi))
            n_per_env[env] = len(cells)
        spec_data[spec] = (env_results, n_per_env)

    per_spec_path = args.out_dir / "cpp17_vs_zig_per_spec.png"
    plot_per_spec(spec_data, per_spec_path, args.dpi)
    print(f"\nWrote {per_spec_path}")

    print_per_spec_table(spec_data, SUMMARY_TAUS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
