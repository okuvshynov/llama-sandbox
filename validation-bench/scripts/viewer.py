"""Interactive results viewer for validation-bench.

Usage:
    streamlit run scripts/viewer.py -- <path-to-results.jsonl>
    streamlit run scripts/viewer.py                 # defaults to results/results.jsonl

First-page surface: a slug × task matrix with toggleable slugs / envs /
tasks. Cell statistic matches matrix-xan.sh / vb-may-10.sh exactly —
best-MCC-per-attempt aggregated as `[min; max], avg=A, n=K/N`, where
K = attempts with at least one scored turn, N = every attempt the
model started. Future pages (per-attempt drill-down, progression
plots, sweep monitor) slot in via Streamlit's pages/ pattern.

Data is re-read every 10 seconds via @st.cache_data(ttl=10), so the
matrix updates as a sweep adds rows to results.jsonl. Partial lines
at the file tail (from in-flight writes) are tolerated.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results" / "results.jsonl"

# --- column-order presets (match vb-may-10's hierarchy) -----------------

def _task_sort_key_hierarchical(task: str) -> tuple:
    """Sort key for the (spec → spec-presence → env) hierarchy.

    Produces an order like:
      toml-1.0-cpp17, toml-1.0-zig, toml-1.0-nospec-cpp17, toml-1.0-nospec-zig,
      yaml-1.2-cpp17, yaml-1.2-zig, yaml-1.2-nospec-cpp17, yaml-1.2-nospec-zig.
    """
    # Split out the env suffix (last hyphenated token) for ordering.
    parts = task.rsplit("-", 1)
    spec_part = parts[0] if len(parts) == 2 else task
    env_part = parts[1] if len(parts) == 2 else ""
    is_nospec = "-nospec" in spec_part
    spec_root = spec_part.replace("-nospec", "")
    return (spec_root, is_nospec, env_part)


COL_ORDERINGS = {
    "alphabetical": lambda t: (t,),
    "spec → spec/nospec → env": _task_sort_key_hierarchical,
}


# --- data loading --------------------------------------------------------

@st.cache_data(ttl=10)
def load_results(path: str) -> pd.DataFrame:
    """Read results.jsonl line-by-line; drop a partially-written tail line
    if one is present (the harness appends mid-sweep)."""
    rows = []
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    with p.open() as f:
        text = f.read()
    lines = text.splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            import json
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Partial final line during in-flight write — safe to skip.
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


# --- cell aggregation ----------------------------------------------------

def aggregate_cell(group: pd.DataFrame) -> dict:
    """For one (slug, task) group, compute the cell statistic.

    Returns dict with mean / min / max / n_scored / n_total / text.
    Mirrors vb-may-10.sh's `[mn; mx], avg=A, n=K/N` format exactly.
    """
    n_total = group["attempt_id"].nunique()
    if n_total == 0:
        return {"mean": None, "min": None, "max": None,
                "n_scored": 0, "n_total": 0, "text": "<empty>"}
    # Best MCC per attempt — only attempts with at least one scored turn.
    with_mcc = group.dropna(subset=["mcc"])
    if with_mcc.empty:
        return {"mean": None, "min": None, "max": None,
                "n_scored": 0, "n_total": n_total,
                "text": f"n=0/{n_total} (no mcc)"}
    best_per_attempt = with_mcc.groupby("attempt_id")["mcc"].max()
    mn, mx = best_per_attempt.min(), best_per_attempt.max()
    avg = best_per_attempt.mean()
    n_scored = len(best_per_attempt)
    return {
        "mean": avg, "min": mn, "max": mx,
        "n_scored": n_scored, "n_total": n_total,
        "text": f"[{mn:.3f}; {mx:.3f}], avg={avg:.3f}, n={n_scored}/{n_total}",
    }


def build_matrix(df: pd.DataFrame, slugs: list, tasks: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (text_matrix, mean_matrix) — first for display, second for
    background-gradient coloring. Both indexed by slug, columns are tasks
    in the order provided."""
    sub = df[df["slug"].isin(slugs) & df["task"].isin(tasks)]
    text_rows = {}
    mean_rows = {}
    for slug in slugs:
        text_rows[slug] = {}
        mean_rows[slug] = {}
        for task in tasks:
            cell_df = sub[(sub["slug"] == slug) & (sub["task"] == task)]
            cell = aggregate_cell(cell_df)
            text_rows[slug][task] = cell["text"]
            mean_rows[slug][task] = cell["mean"]
    text_matrix = pd.DataFrame(text_rows).T.reindex(columns=tasks)
    mean_matrix = pd.DataFrame(mean_rows).T.reindex(columns=tasks)
    return text_matrix, mean_matrix


# --- UI ------------------------------------------------------------------

def main():
    # Argv after `--` is the source path.
    source = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RESULTS)

    st.set_page_config(page_title="validation-bench viewer",
                       layout="wide", initial_sidebar_state="expanded")

    # Sidebar: source metadata.
    with st.sidebar:
        st.markdown("## validation-bench")
        st.markdown(f"**Source:** `{source}`")
        if st.button("Refresh data", help="Force re-read (otherwise ~10s TTL)"):
            load_results.clear()

        df = load_results(source)
        if df.empty:
            st.error(f"No data at {source}")
            return

        st.markdown(f"**Rows:** {len(df):,}")
        st.markdown(f"**Attempts:** {df['attempt_id'].nunique():,}")
        st.markdown(f"**Slugs:** {df['slug'].nunique()}")
        st.markdown(f"**Tasks:** {df['task'].nunique()}")
        if "env" in df.columns:
            st.markdown(f"**Envs:** {df['env'].nunique()}")

    # Main: tabs (only Matrix for v1; placeholders for future pages).
    tab_matrix, tab_attempts = st.tabs(["Matrix", "Per-attempt (TODO)"])

    with tab_matrix:
        st.markdown("### Cross-(slug, task) matrix")
        st.caption(
            "Cells show `[min; max], avg=mean, n=K/N` where K = attempts "
            "with at least one scored turn, N = every attempt started. "
            "Background gradient colors by mean MCC (red −1 ↔ green +1)."
        )

        # Discover dimensions from the data.
        all_slugs = sorted(df["slug"].unique())
        all_envs = sorted(df["env"].dropna().unique()) if "env" in df.columns else []
        all_tasks = sorted(df["task"].unique())

        # Filter widgets — three independent multi-selects.
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs)
        with c2:
            sel_envs = st.multiselect("Envs", all_envs, default=all_envs)
        with c3:
            # Default task list = tasks matching the selected envs.
            task_candidates = [
                t for t in all_tasks
                if any(t.endswith(f"-{e}") or t == e for e in sel_envs)
            ] if sel_envs else all_tasks
            sel_tasks = st.multiselect("Tasks", all_tasks, default=task_candidates)

        # Ordering.
        c4, c5 = st.columns(2)
        with c4:
            col_order_name = st.selectbox(
                "Column order", list(COL_ORDERINGS.keys()),
                index=1,  # default to hierarchical
            )
        with c5:
            row_order = st.selectbox(
                "Row order",
                ["alphabetical", "by mean best-MCC (desc)"],
                index=0,
            )

        if not sel_slugs or not sel_tasks:
            st.warning("Select at least one slug and one task.")
            return

        # Sort tasks by chosen ordering.
        col_key = COL_ORDERINGS[col_order_name]
        ordered_tasks = sorted(sel_tasks, key=col_key)

        # Build the matrix.
        text_matrix, mean_matrix = build_matrix(df, sel_slugs, ordered_tasks)

        # Row order.
        if row_order.startswith("by mean"):
            score = mean_matrix.mean(axis=1, skipna=True).fillna(-2)
            ordered_slugs = score.sort_values(ascending=False).index.tolist()
        else:
            ordered_slugs = sorted(sel_slugs)
        text_matrix = text_matrix.reindex(ordered_slugs)
        mean_matrix = mean_matrix.reindex(ordered_slugs)

        # Render: text content with background gradient driven by mean.
        styled = text_matrix.style.apply(
            lambda _col: [
                f"background-color: {_mcc_color(v)}" if pd.notna(v) else ""
                for v in mean_matrix[_col.name]
            ],
            axis=0,
        )
        st.dataframe(styled, use_container_width=True, height=min(35 * (len(ordered_slugs) + 1) + 10, 800))

    with tab_attempts:
        st.markdown("### Per-attempt drill-down")
        st.caption("Coming soon — click a cell in Matrix to see attempt-level details.")


def _mcc_color(mcc: float) -> str:
    """Red-yellow-green gradient over the MCC range [-1, +1]. Light shade
    so the cell text stays readable in either Streamlit theme."""
    if mcc is None or pd.isna(mcc):
        return "transparent"
    # Map -1 → red, 0 → yellow, +1 → green. Use HSL for smooth interpolation.
    # Hue: 0 (red) → 60 (yellow) → 120 (green).
    clamped = max(-1.0, min(1.0, float(mcc)))
    hue = (clamped + 1.0) * 60.0  # -1 → 0, 0 → 60, +1 → 120
    return f"hsl({hue:.0f}, 65%, 80%)"


if __name__ == "__main__":
    main()
