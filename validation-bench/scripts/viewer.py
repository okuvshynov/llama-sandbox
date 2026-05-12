"""Interactive results viewer for validation-bench.

Usage:
    streamlit run scripts/viewer.py -- <path-to-results.jsonl>
    streamlit run scripts/viewer.py                 # defaults to results/results.jsonl

Pages (selectable in the sidebar):
  - Matrix: slug × task cell matrix with toggleable slugs and tasks.
    Cell statistic matches matrix-xan.sh / vb-may-10.sh exactly —
    `[min; max], avg=A, n=K/N`. Drill-in widget below the matrix
    navigates to the Cell page for a chosen (slug, task).
  - P(MCC≥τ): same matrix layout, but each cell is the fraction of
    attempts whose best per-turn MCC reaches a configurable threshold τ.
    Attempts that never produced a scored turn (e.g. all compile errors)
    count as fail — sidesteps the "what MCC do we impute for a non-
    compiling validator" question by reframing as "did this combination
    deliver a usable validator at all?".
  - Leaderboard: one row per model, one column per τ in a fine grid
    (0.00, 0.05, ..., 1.00 — step 0.05, 21 columns), plus a `support`
    column ("n_cells | n_attempts") so thin-coverage rows are visible.
    Two aggregation modes via radio:
      - "raw macro-avg": (1/T)·Σ K_t/N_t — every cell counts equally
        regardless of N_t.
      - "Beta-Binomial shrinkage": per-(model,τ) latent Beta(α,β)
        prior fit by MLE; small-N cells get pulled toward the model's
        mean rate. Models with too few cells fall back to raw rates.
    Optional 95% CI checkbox computes a two-stage parametric bootstrap
    (resample cells × resample within-cell binomially, B=500) and
    renders cells as "point [lo, hi]". Sort + colour stay on the
    point estimate. Toggle restricts to tasks where every selected
    model has data (default on) so cross-model rows are apples-to-apples.
  - Curves: same data as Leaderboard rendered as a line chart — one
    line per model (point estimate), with a translucent shaded band
    showing the 95% bootstrap CI. X axis is τ ∈ [0, 1]; Y axis is
    P(MCC ≥ τ) ∈ [0, 1]. Same filter / mode / intersection controls
    as Leaderboard. Click a model in the legend to highlight it (other
    models fade); useful when many bands overlap.
  - Cell: all attempts in one (slug, task), with per-turn MCC trend
    + sortable table. Selecting a row + clicking "View attempt →"
    navigates to the Attempt page.
  - Attempt: per-turn source code + compile output + test failures
    for one (slug, task, attempt_id). Reads artifacts from
    VB_DATA_DIR (default ~/.vb-data; same convention as the harness).

Data is re-read every 10 seconds via @st.cache_data(ttl=10), so views
update as a sweep adds rows to results.jsonl. Partial lines at the file
tail (from in-flight writes) are tolerated.
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import betabinom


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results" / "results.jsonl"
VB_DATA_DIR = Path(os.environ.get("VB_DATA_DIR") or (Path.home() / ".vb-data"))


# --- env → syntax-highlight language hint -------------------------------

ENV_LANG = {
    "cpp17": "cpp",
    "go": "go",
    "zig": "zig",
    "d": "d",
    "lua": "lua",
    "erlang": "erlang",
}

PAGES = ["Matrix", "P(MCC≥τ)", "Leaderboard", "Curves", "Cell", "Attempt"]

LEADERBOARD_TAUS = [round(0.05 * i, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00


# --- session-state init -------------------------------------------------

def init_session_state() -> None:
    # Note: we use `current_page` (not `page`) as the source of truth so the
    # sidebar radio widget can have an auto-generated key without owning the
    # navigation state. Streamlit forbids programmatic writes to a keyed
    # widget's session_state slot after the widget has rendered — we hit
    # that error trying to set `st.session_state.page` from a drill-in
    # button below the radio. Separating them avoids the collision.
    defaults = {
        "current_page": "Matrix",
        "sel_slug": None,
        "sel_task": None,
        "sel_attempt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go_to(page: str, **kwargs) -> None:
    """Set the current page (+ any extra session-state fields) and rerun."""
    st.session_state.current_page = page
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()

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


# --- threshold (P(MCC ≥ τ)) aggregation ---------------------------------

def aggregate_cell_prob(group: pd.DataFrame, tau: float) -> dict:
    """For one (slug, task) group, compute P(best per-attempt MCC ≥ τ).

    Numerator: attempts whose best per-turn MCC is ≥ τ.
    Denominator: every attempt that started — including ones whose every
    turn failed to score (all compile_error / no_submissions). Those count
    as fail at any τ ≥ 0, which is the whole point of this metric (a
    non-compiling validator is a failure, not a missing data point).
    """
    n_total = group["attempt_id"].nunique()
    if n_total == 0:
        return {"p": None, "n_pass": 0, "n_total": 0, "text": "<empty>"}
    with_mcc = group.dropna(subset=["mcc"])
    if with_mcc.empty:
        return {"p": 0.0, "n_pass": 0, "n_total": n_total,
                "text": f"0.00 (0/{n_total})"}
    best_per_attempt = with_mcc.groupby("attempt_id")["mcc"].max()
    n_pass = int((best_per_attempt >= tau).sum())
    p = n_pass / n_total
    return {"p": p, "n_pass": n_pass, "n_total": n_total,
            "text": f"{p:.2f} ({n_pass}/{n_total})"}


def build_prob_matrix(df: pd.DataFrame, slugs: list, tasks: list,
                      tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (text_matrix, p_matrix) — the latter for gradient coloring."""
    sub = df[df["slug"].isin(slugs) & df["task"].isin(tasks)]
    text_rows: dict = {}
    p_rows: dict = {}
    for slug in slugs:
        text_rows[slug] = {}
        p_rows[slug] = {}
        for task in tasks:
            cell_df = sub[(sub["slug"] == slug) & (sub["task"] == task)]
            cell = aggregate_cell_prob(cell_df, tau)
            text_rows[slug][task] = cell["text"]
            p_rows[slug][task] = cell["p"]
    text_matrix = pd.DataFrame(text_rows).T.reindex(columns=tasks)
    p_matrix = pd.DataFrame(p_rows).T.reindex(columns=tasks)
    return text_matrix, p_matrix


def _prob_color(p: float) -> str:
    """Red 0 → yellow 0.5 → green 1 gradient. Same lightness as _mcc_color
    so the two matrices look visually consistent."""
    if p is None or pd.isna(p):
        return "transparent"
    clamped = max(0.0, min(1.0, float(p)))
    hue = clamped * 120.0
    return f"hsl({hue:.0f}, 65%, 80%)"


# --- per-attempt tab -----------------------------------------------------

# Parses "(<reason>, expected '<verdict>')" out of a FAIL line. The reason
# itself can contain a single nested parenthesised clause (e.g.
# "timeout (no verdict printed)"), so we accept one optional `(...)` inside.
_FAIL_REASON_RE = re.compile(r"\(([^()]*(?:\([^()]*\)[^()]*)?), expected '")


def _categorize_fail_line(line: str) -> str:
    m = _FAIL_REASON_RE.search(line)
    if not m:
        return "unparseable"
    reason = m.group(1)
    # Collapse digit sequences so e.g. `exit=124` and `exit=137` bucket
    # together. Same heuristic the matrix-xan failure analysis uses.
    return re.sub(r"\d+", "N", reason)


def _read_text(path: Path) -> str | None:
    """Best-effort read; returns None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        return path.read_text(errors="replace")
    except Exception:
        return None


def _attempt_summary_label(aid: str, rows: pd.DataFrame) -> str:
    """Compact dropdown label: short_id (slug × task, best=X.XXX or 'all errs')."""
    rows = rows.sort_values("turn")
    first = rows.iloc[0]
    mccs = rows["mcc"].dropna() if "mcc" in rows.columns else pd.Series([], dtype=float)
    if len(mccs):
        tag = f"best={mccs.max():+.3f}"
    else:
        tag = "all errs"
    short_aid = aid.rsplit("-", 1)[-1] if "-" in aid else aid[-12:]
    return f"{short_aid}  ({first['slug']} × {first['task']}, {tag})"


def render_per_attempt_tab(df: pd.DataFrame, preselected_aid: str | None = None) -> None:
    st.markdown("### Per-attempt drill-down")
    st.caption(
        "Pick (slug, task, attempt) — then expand each turn to see the "
        "submitted source, compiler output, and test failures. "
        f"Artifacts read from `{VB_DATA_DIR}` (override with VB_DATA_DIR env var)."
    )

    # If a pre-selected attempt_id was passed in (from the Cell page's
    # "View attempt →" drill-in), pre-fill the filters to match it so the
    # user lands directly on the requested attempt.
    if preselected_aid:
        match = df[df["attempt_id"] == preselected_aid]
        if not match.empty:
            r0 = match.iloc[0]
            preselected_slug = r0["slug"]
            preselected_task = r0["task"]
        else:
            preselected_aid = None
            preselected_slug = preselected_task = None
    else:
        preselected_slug = preselected_task = None

    # Cascading filters: slug → task → attempt.
    all_slugs = sorted(df["slug"].dropna().unique())
    c1, c2 = st.columns(2)
    with c1:
        slug_opts = ["(any)"] + all_slugs
        default_idx = slug_opts.index(preselected_slug) if preselected_slug in slug_opts else 0
        sel_slug = st.selectbox("Slug", slug_opts, index=default_idx)
    sub = df if sel_slug == "(any)" else df[df["slug"] == sel_slug]
    all_tasks = sorted(sub["task"].dropna().unique())
    with c2:
        task_opts = ["(any)"] + all_tasks
        default_idx = task_opts.index(preselected_task) if preselected_task in task_opts else 0
        sel_task = st.selectbox("Task", task_opts, index=default_idx)
    sub = sub if sel_task == "(any)" else sub[sub["task"] == sel_task]

    # Attempts sorted by timestamp (newest first) so recent runs are easy to find.
    if "attempt_timestamp" in sub.columns:
        attempts_df = sub.drop_duplicates("attempt_id").sort_values(
            "attempt_timestamp", ascending=False)
    else:
        attempts_df = sub.drop_duplicates("attempt_id")
    attempts = attempts_df["attempt_id"].tolist()

    if not attempts:
        st.warning("No attempts match the current filters.")
        return

    default_attempt_idx = attempts.index(preselected_aid) if preselected_aid in attempts else 0
    sel_aid = st.selectbox(
        f"Attempt ({len(attempts)} match)",
        attempts,
        index=default_attempt_idx,
        format_func=lambda aid: _attempt_summary_label(aid, sub[sub["attempt_id"] == aid]),
    )
    if not sel_aid:
        return

    rows = sub[sub["attempt_id"] == sel_aid].sort_values("turn").reset_index(drop=True)
    first = rows.iloc[0]

    # --- attempt summary block ---
    st.markdown(f"**attempt_id:** `{sel_aid}`")
    meta_l, meta_r = st.columns(2)
    with meta_l:
        st.markdown(f"**slug:** {first['slug']}")
        st.markdown(f"**task:** {first['task']} (env={first.get('env','?')})")
        if "model" in rows.columns:
            st.markdown(f"**model:** `{first.get('model','?')}`")
    with meta_r:
        if "attempt_timestamp" in rows.columns:
            st.markdown(f"**timestamp:** {first.get('attempt_timestamp','?')}")
        if "attempt_elapsed_seconds" in rows.columns:
            elapsed = first.get("attempt_elapsed_seconds")
            if pd.notna(elapsed):
                st.markdown(f"**elapsed:** {elapsed:.1f}s")
        if "vb_version" in rows.columns:
            st.markdown(f"**vb_version:** {first.get('vb_version','?')}")

    # Per-turn one-liner trend.
    trend = []
    for _, r in rows.iterrows():
        turn = int(r["turn"])
        if pd.notna(r.get("mcc")):
            trend.append(f"t{turn}: {r['mcc']:+.3f}")
        else:
            trend.append(f"t{turn}: {r.get('error','?')}")
    st.markdown("**Per-turn:** " + " · ".join(trend))

    # Artifact directory.
    attempt_dir = VB_DATA_DIR / sel_aid
    if not attempt_dir.is_dir():
        st.warning(
            f"Artifact dir not found: `{attempt_dir}`. Per-turn details unavailable. "
            "(If you have artifacts under a different path, set the VB_DATA_DIR env var.)"
        )
        return
    st.caption(f"Artifacts: `{attempt_dir}`")
    st.divider()

    # --- per-turn details ---
    # Submission dirs (`submissions/1`, `/2`, ...) are numbered by the
    # harness on every accepted `submit` tool call — independent of the
    # `turn` field, which counts conversational turns. They diverge when
    # a turn produces zero submissions (e.g. model replies in plain text
    # with no tool call) or more than one. Each results.jsonl row equals
    # one submission, written in submission order — so the K-th row (after
    # sorting by turn) maps to `submissions/K`.
    lang = ENV_LANG.get(first.get("env", ""), "text")
    for sub_idx, (_, r) in enumerate(rows.iterrows(), start=1):
        turn = int(r["turn"])
        sub_dir = attempt_dir / "submissions" / str(sub_idx)

        # Header
        head_suffix = f" (submission {sub_idx})" if sub_idx != turn else ""
        if pd.notna(r.get("mcc")):
            cm_parts = [f"TP={int(r.get('tp',0))}",
                        f"FN={int(r.get('fn',0))}",
                        f"FP={int(r.get('fp',0))}",
                        f"TN={int(r.get('tn',0))}"]
            head = f"### Turn {turn}{head_suffix} — MCC={r['mcc']:+.4f} ({' '.join(cm_parts)})"
        else:
            head = f"### Turn {turn}{head_suffix} — ⚠️ {r.get('error','?')}"
        st.markdown(head)

        if not sub_dir.is_dir():
            st.caption(f"_No submission dir at `{sub_dir.relative_to(VB_DATA_DIR)}`._")
            continue

        # Source.
        src_paths = sorted(sub_dir.glob("solution.*"))
        if src_paths:
            src_path = src_paths[0]
            src_text = _read_text(src_path) or ""
            lines = src_text.count("\n") + 1
            with st.expander(f"📄 `{src_path.name}` — {len(src_text):,} chars, {lines:,} lines"):
                st.code(src_text, language=lang)
        else:
            st.caption("_No source file saved._")

        # Compiler output (show only if non-empty; auto-expand on real errors).
        comp_text = _read_text(sub_dir / "compiler.txt") or ""
        if comp_text.strip():
            with st.expander(f"⚠️ Compiler output ({len(comp_text)} bytes)", expanded=True):
                st.code(comp_text, language="text")

        # Test failures.
        tests_text = _read_text(sub_dir / "tests.txt") or ""
        fail_lines = [l for l in tests_text.splitlines() if l.startswith("FAIL")]
        if fail_lines:
            cats = Counter(_categorize_fail_line(l) for l in fail_lines)
            top = ", ".join(f"{n}× `{k}`" for k, n in cats.most_common(4))
            label = f"❌ {len(fail_lines)} test failures · {top}"
            with st.expander(label):
                # Show all failure categories first (sorted desc), then the
                # full raw list. Categories handle the case where the user
                # just wants the shape, not every individual line.
                st.markdown("**Failure categories** (top 20):")
                cat_df = pd.DataFrame(
                    [(k, n) for k, n in cats.most_common(20)],
                    columns=["reason", "count"],
                )
                st.dataframe(cat_df, hide_index=True, use_container_width=True)
                st.markdown("**Full failure list:**")
                st.code(tests_text, language="text")

        st.divider()


# --- Matrix page ---------------------------------------------------------

def render_matrix_page(df: pd.DataFrame) -> None:
    st.markdown("### Cross-(slug, task) matrix")
    st.caption(
        "Cells show `[min; max], avg=mean, n=K/N` where K = attempts "
        "with at least one scored turn, N = every attempt started. "
        "Background gradient colors by mean MCC (red −1 ↔ green +1)."
    )

    # Discover dimensions from the data.
    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    # Filter widgets — two multi-selects (slugs, tasks). The data model
    # is task = (env, spec); env is a substring of task, so a separate
    # env filter would be redundant and confusing.
    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs)
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks)

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

    col_key = COL_ORDERINGS[col_order_name]
    ordered_tasks = sorted(sel_tasks, key=col_key)
    text_matrix, mean_matrix = build_matrix(df, sel_slugs, ordered_tasks)

    if row_order.startswith("by mean"):
        score = mean_matrix.mean(axis=1, skipna=True).fillna(-2)
        ordered_slugs = score.sort_values(ascending=False).index.tolist()
    else:
        ordered_slugs = sorted(sel_slugs)
    text_matrix = text_matrix.reindex(ordered_slugs)
    mean_matrix = mean_matrix.reindex(ordered_slugs)

    styled = text_matrix.style.apply(
        lambda _col: [
            f"background-color: {_mcc_color(v)}" if pd.notna(v) else ""
            for v in mean_matrix[_col.name]
        ],
        axis=0,
    )

    # --- Click-to-drill: row + column selection identifies a cell. ---
    # st.dataframe doesn't support single-cell click directly, but it does
    # accept a list of selection modes. Clicking a row selects the slug;
    # clicking a column header selects the task. Together they identify
    # the (slug, task) cell to drill into.
    event = st.dataframe(
        styled,
        use_container_width=True,
        height=min(35 * (len(ordered_slugs) + 1) + 10, 800),
        selection_mode=["single-row", "single-column"],
        on_select="rerun",
        key="matrix_table",
    )

    sel_rows = event.selection.rows if hasattr(event, "selection") else []
    sel_cols = event.selection.columns if hasattr(event, "selection") else []
    selected_slug = ordered_slugs[sel_rows[0]] if sel_rows else None
    selected_task = sel_cols[0] if sel_cols else None

    st.markdown("---")
    if selected_slug and selected_task:
        st.markdown(f"**Selected cell:** `{selected_slug}` × `{selected_task}`")
        st.caption("Click 'View cell →' to see all attempts in this (slug, task).")
        if st.button("View cell →", use_container_width=False):
            _go_to("Cell", sel_slug=selected_slug, sel_task=selected_task)
    elif selected_slug:
        st.caption(f"Row selected: `{selected_slug}`. Click a **column header** to pick a task.")
    elif selected_task:
        st.caption(f"Column selected: `{selected_task}`. Click any **row** to pick a slug.")
    else:
        st.caption("Click any row + any column header in the matrix to pick a cell, then 'View cell →'.")


# --- P(MCC ≥ τ) page -----------------------------------------------------

def render_threshold_matrix_page(df: pd.DataFrame) -> None:
    st.markdown("### Threshold matrix — P(best MCC ≥ τ)")
    st.caption(
        "Cells show `P (n_pass/n_total)` where n_pass = attempts whose best "
        "per-turn MCC reaches τ, n_total = every attempt that started. "
        "Attempts that produced no scored turn (all compile errors / "
        "no submissions) count as fail. Background gradient by P "
        "(red 0 ↔ green 1)."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    tau = st.slider(
        "Threshold τ (best per-attempt MCC must reach this to count as pass)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help=("τ=0 → strictly any model that scored at all; "
              "τ=0.5 → substantively useful; τ=0.9 → near-perfect; "
              "τ=1.0 → only perfect MCC counts."),
        key="thresh_tau",
    )

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="thresh_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="thresh_tasks")

    c4, c5 = st.columns(2)
    with c4:
        col_order_name = st.selectbox(
            "Column order", list(COL_ORDERINGS.keys()),
            index=1, key="thresh_col_order",
        )
    with c5:
        row_order = st.selectbox(
            "Row order",
            ["alphabetical", "by mean P (desc)"],
            index=0, key="thresh_row_order",
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    col_key = COL_ORDERINGS[col_order_name]
    ordered_tasks = sorted(sel_tasks, key=col_key)
    text_matrix, p_matrix = build_prob_matrix(df, sel_slugs, ordered_tasks, tau)

    if row_order.startswith("by mean"):
        score = p_matrix.mean(axis=1, skipna=True).fillna(-1)
        ordered_slugs = score.sort_values(ascending=False).index.tolist()
    else:
        ordered_slugs = sorted(sel_slugs)
    text_matrix = text_matrix.reindex(ordered_slugs)
    p_matrix = p_matrix.reindex(ordered_slugs)

    styled = text_matrix.style.apply(
        lambda _col: [
            f"background-color: {_prob_color(v)}" if pd.notna(v) else ""
            for v in p_matrix[_col.name]
        ],
        axis=0,
    )

    event = st.dataframe(
        styled,
        use_container_width=True,
        height=min(35 * (len(ordered_slugs) + 1) + 10, 800),
        selection_mode=["single-row", "single-column"],
        on_select="rerun",
        key="thresh_matrix_table",
    )

    sel_rows = event.selection.rows if hasattr(event, "selection") else []
    sel_cols = event.selection.columns if hasattr(event, "selection") else []
    selected_slug = ordered_slugs[sel_rows[0]] if sel_rows else None
    selected_task = sel_cols[0] if sel_cols else None

    st.markdown("---")
    if selected_slug and selected_task:
        st.markdown(f"**Selected cell:** `{selected_slug}` × `{selected_task}`")
        st.caption("Click 'View cell →' to see all attempts in this (slug, task).")
        if st.button("View cell →", use_container_width=False, key="thresh_view_cell"):
            _go_to("Cell", sel_slug=selected_slug, sel_task=selected_task)
    elif selected_slug:
        st.caption(f"Row selected: `{selected_slug}`. Click a **column header** to pick a task.")
    elif selected_task:
        st.caption(f"Column selected: `{selected_task}`. Click any **row** to pick a slug.")
    else:
        st.caption("Click any row + any column header in the matrix to pick a cell, then 'View cell →'.")


# --- Leaderboard page ----------------------------------------------------
#
# Aggregates per-cell pass rates into one number per (model, τ). Three
# choices were considered:
#   1. Macro-average:  (1/T) · Σ_t K_t/N_t — every task counts equally,
#      regardless of how many attempts that cell happened to accumulate.
#      Matches the framing "expected pass rate over the population of
#      tasks I care about". Trade: a cell with N=1 carries the same
#      weight as N=10.
#   2. Micro-average:  Σ K_t / Σ N_t — pooled. Cells you swept more
#      deeply (often for accidental reasons) dominate. Wrong here.
#   3. Beta-binomial shrinkage: per-model latent Beta prior, per-cell
#      rates pulled toward the model mean by Empirical Bayes. Proper
#      CIs free. Worth doing once N=3-vs-11 spread visibly distorts
#      rankings — punted to v2.
# v1 implements (1) only, with an "intersection" toggle so that cross-
# model rows are aggregated over the same task set (apples-to-apples).

def _per_cell_counts(group: pd.DataFrame, taus: list[float]
                     ) -> tuple[int, dict[float, int]]:
    """For one (slug, task) cell, return (N_total, {τ: K_passed}).

    K_passed = attempts whose best per-turn MCC ≥ τ.
    N_total  = every attempt that started (non-scored attempts count
               toward N but never toward K, so non-compile = fail at
               any τ ≥ 0).
    """
    n_total = int(group["attempt_id"].nunique())
    if n_total == 0:
        return 0, {tau: 0 for tau in taus}
    with_mcc = group.dropna(subset=["mcc"])
    if with_mcc.empty:
        return n_total, {tau: 0 for tau in taus}
    best = with_mcc.groupby("attempt_id")["mcc"].max()
    return n_total, {tau: int((best >= tau).sum()) for tau in taus}


def fit_beta_binomial(k_arr: list[int], n_arr: list[int],
                      max_concentration: float = 1000.0
                      ) -> tuple[float, float] | None:
    """MLE fit of (α, β) for the per-model latent Beta prior on cell rates.

    Model:
      p_t       ~ Beta(α, β)            (per-cell true rate)
      K_t | p_t ~ Binomial(N_t, p_t)    (observed passes)
      ⇒ marginal: K_t ~ BetaBinomial(N_t, α, β)

    Returns None when the fit is degenerate — caller should fall back to
    raw rates. Degenerate cases:
      - fewer than 2 cells (no between-cell variance to estimate);
      - all observed rates identical (no signal for the prior shape);
      - optimizer fails to converge.

    Initialization: method-of-moments on the observed cell rates, with
    a floor on the variance to prevent ν → ∞ on near-uniform data.
    Optimizer: Nelder-Mead on log(α), log(β) — keeps params positive
    without bounds-handling, robust to small-N likelihood landscapes.
    A concentration cap (α + β ≤ max_concentration) keeps the optimizer
    from running away when all cells are 0/0 or all 1/1.
    """
    k = np.asarray(k_arr, dtype=float)
    n = np.asarray(n_arr, dtype=float)
    if len(k) < 2:
        return None
    p_hat = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    if np.allclose(p_hat, p_hat[0]):
        return None  # no between-cell variance — prior is unidentifiable

    mu = float(np.mean(p_hat))
    var = float(np.var(p_hat, ddof=1))
    var = max(var, 1e-3)
    nu0 = max(mu * (1 - mu) / var - 1, 0.5)
    alpha0 = float(np.clip(mu * nu0, 0.1, 100.0))
    beta0 = float(np.clip((1 - mu) * nu0, 0.1, 100.0))

    def neg_log_lik(log_params):
        alpha, beta = np.exp(log_params)
        if alpha + beta > max_concentration:
            return 1e10
        return -float(np.sum(betabinom.logpmf(k, n, alpha, beta)))

    res = minimize(neg_log_lik, np.log([alpha0, beta0]),
                   method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 500})
    if not res.success:
        return None
    alpha, beta = (float(x) for x in np.exp(res.x))
    return alpha, beta


def _shrunken_rate(k: int, n: int, alpha: float, beta: float) -> float:
    """Posterior mean of p_t under prior Beta(α, β) and observation
    K_t=k, N_t=n. The K=0/N=0 case never arises since cells with no
    attempts are excluded earlier in build_leaderboard."""
    return (alpha + k) / (alpha + beta + n)


def _bootstrap_macro_avg_ci(k_arr: list[int], n_arr: list[int], *,
                            shrunken: bool,
                            alpha: float | None = None,
                            beta: float | None = None,
                            B: int = 500, ci_level: float = 0.95,
                            seed: int = 0) -> tuple[float, float]:
    """Two-stage parametric bootstrap CI for the macro-average of per-cell
    pass rates. Used for both raw and shrunken aggregation.

    Stage 1: resample T cells with replacement. Captures uncertainty
             from "which tasks happened to be in our sample" — usually
             the dominant source of variance in our T=3..9 regime.
    Stage 2: for each resampled cell, draw K* ~ Binomial(N, K/N).
             Captures within-cell binomial noise, which matters more
             when N is small (e.g. N=2 or N=3 cells).

    For shrunken=True, the supplied (α, β) are held FIXED across
    bootstrap iterations rather than refit per iteration. This is
    deliberate: refitting would multiply runtime by the MLE cost (~20ms
    per fit × B × |τ| × |slugs| would push the page into multi-minute
    redraws), and the dominant uncertainty is in the cells, not the
    prior shape. Reasonable for a v1 visualization; would matter more
    if the prior fit itself were unstable.

    Returns (lo, hi) at the requested CI level (default 95%).
    Falls back to raw bootstrap if shrunken=True but α/β are None
    (degenerate fit on the original data).
    """
    T = len(k_arr)
    if T == 0:
        return float("nan"), float("nan")
    if shrunken and (alpha is None or beta is None):
        shrunken = False

    k = np.asarray(k_arr, dtype=int)
    n = np.asarray(n_arr, dtype=int)
    rng = np.random.default_rng(seed)

    # Vectorize the bootstrap loop: draw all B×T cell indices at once,
    # then a single Binomial draw per cell-iteration. Reshape and reduce.
    idx = rng.integers(0, T, size=(B, T))
    n_b = n[idx]
    k_b = k[idx]
    # Within-cell binomial resample using observed cell rate.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_obs = np.where(n_b > 0, k_b / np.maximum(n_b, 1), 0.0)
    k_star = rng.binomial(n_b, p_obs)

    if shrunken:
        rates = (alpha + k_star) / (alpha + beta + n_b)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.where(n_b > 0, k_star / np.maximum(n_b, 1), 0.0)

    macro = rates.mean(axis=1)
    half = (1 - ci_level) / 2 * 100
    lo, hi = np.percentile(macro, [half, 100 - half])
    return float(lo), float(hi)


def build_leaderboard(df: pd.DataFrame, slugs: list[str], tasks: list[str],
                      taus: list[float], intersection: bool,
                      shrunken: bool = False,
                      ci_B: int = 0, ci_level: float = 0.95,
                      ) -> tuple[pd.DataFrame, pd.DataFrame | None,
                                 pd.DataFrame | None, list[str]]:
    """Returns (leaderboard_df, tasks_used).

    Returns (point_df, lo_df, hi_df, tasks_used).

    point_df has one row per slug, columns:
      slug, support ("cells | attempts" — e.g. "7 | 45"), then one
      column per τ ("≥0.00", "≥0.05", ...).
    tasks_used is the actual task set after intersection filtering, so
    the page can show what was aggregated.

    When shrunken=False (default): each τ entry is the macro-average of
    raw per-cell pass rates — `(1/T) · Σ_t K_t/N_t`.

    When shrunken=True: for each (model, τ), fit a per-model latent
    Beta(α, β) prior on cell rates via Empirical Bayes (MLE on the
    Beta-Binomial marginal), then macro-average the *shrunken* rates
    (α + K_t)/(α + β + N_t). Cells with small N get pulled toward the
    model's mean rate; cells with large N stay near observed. Models
    with too few cells (or all-equal cell rates) for a meaningful
    prior fit fall back to raw rates per-τ — fit_beta_binomial returns
    None in those cases.

    When ci_B > 0: also compute (1-ci_level) percentile CIs via
    _bootstrap_macro_avg_ci using ci_B iterations. lo_df and hi_df
    are returned alongside point_df, with the same row index and τ
    columns. When ci_B == 0 (default), lo_df and hi_df are None — the
    common path stays free of bootstrap cost.

    Note: support is a single text column rather than two int columns
    because (a) cells and attempts are highly correlated context numbers
    rather than independently sortable metrics, and (b) the page already
    has 21 τ columns — every saved column matters for readability.
    """
    sub = df[df["slug"].isin(slugs) & df["task"].isin(tasks)]

    # cell_present[slug, task] = True iff that (slug, task) has ≥1 row.
    if sub.empty:
        cell_present = pd.DataFrame(False, index=slugs, columns=tasks)
    else:
        cell_present = (sub.groupby(["slug", "task"]).size().unstack(fill_value=0) > 0)
        cell_present = cell_present.reindex(index=slugs, columns=tasks, fill_value=False)

    if intersection:
        # Only tasks where every selected slug has at least one attempt.
        tasks_used = [t for t in tasks if bool(cell_present[t].all())]
    else:
        tasks_used = list(tasks)

    tau_cols = [_tau_col(tau) for tau in taus]
    rows: list[dict] = []
    lo_rows: list[dict] = []
    hi_rows: list[dict] = []
    for slug_idx, slug in enumerate(slugs):
        slug_tasks = [t for t in tasks_used if bool(cell_present.loc[slug, t])]
        if not slug_tasks:
            rows.append({"slug": slug, "support": "0 | 0",
                         **{c: float("nan") for c in tau_cols}})
            lo_rows.append({c: float("nan") for c in tau_cols})
            hi_rows.append({c: float("nan") for c in tau_cols})
            continue
        # Collect per-cell (N_t, {τ: K_t}) once per cell.
        cell_counts = []
        for task in slug_tasks:
            cell_df = sub[(sub["slug"] == slug) & (sub["task"] == task)]
            cell_counts.append(_per_cell_counts(cell_df, taus))
        n_attempts_total = sum(n for n, _ in cell_counts)

        macro: dict[str, float] = {}
        lo_d: dict[str, float] = {}
        hi_d: dict[str, float] = {}
        for tau_idx, tau in enumerate(taus):
            k_arr = [k_at[tau] for _, k_at in cell_counts]
            n_arr = [n for n, _ in cell_counts]
            alpha = beta = None
            if shrunken:
                fit = fit_beta_binomial(k_arr, n_arr)
                if fit is not None:
                    alpha, beta = fit
                    rates = [_shrunken_rate(k, n, alpha, beta)
                             for k, n in zip(k_arr, n_arr)]
                else:
                    rates = [k / n if n > 0 else 0.0
                             for k, n in zip(k_arr, n_arr)]
            else:
                rates = [k / n if n > 0 else 0.0
                         for k, n in zip(k_arr, n_arr)]
            col = _tau_col(tau)
            macro[col] = sum(rates) / len(rates)
            if ci_B > 0:
                # Per-(slug, τ) seed so the bootstrap is reproducible
                # and independent across cells (avoids correlated draws
                # if the same RNG state were reused).
                seed = 1_000_000 * slug_idx + tau_idx
                lo, hi = _bootstrap_macro_avg_ci(
                    k_arr, n_arr,
                    shrunken=shrunken, alpha=alpha, beta=beta,
                    B=ci_B, ci_level=ci_level, seed=seed,
                )
                lo_d[col] = lo
                hi_d[col] = hi

        rows.append({"slug": slug,
                     "support": f"{len(slug_tasks)} | {n_attempts_total}",
                     **macro})
        lo_rows.append(lo_d)
        hi_rows.append(hi_d)

    point_df = pd.DataFrame(rows)
    if ci_B > 0:
        lo_df = pd.DataFrame(lo_rows)
        hi_df = pd.DataFrame(hi_rows)
    else:
        lo_df = hi_df = None
    return point_df, lo_df, hi_df, tasks_used


def _tau_col(tau: float) -> str:
    return f"≥{tau:.2f}"


def render_leaderboard_page(df: pd.DataFrame) -> None:
    st.markdown("### Leaderboard — macro-averaged P(MCC ≥ τ) per model")
    st.caption(
        "One row per model. Each τ column is the model's aggregated pass "
        "rate at threshold τ — either the raw macro-average of per-cell "
        "rates `(1/T)·Σ K_t/N_t`, or a Beta-Binomial shrunk version where "
        "each model gets a per-τ latent Beta(α,β) prior fit by MLE and "
        "small-N cells are pulled toward the model's mean. Toggle "
        "'Show 95% CIs' to render cells as `point [lo, hi]` from a "
        "two-stage parametric bootstrap (resample cells × resample "
        "within-cell binomially). Toggle 'restrict to intersection' to "
        "limit tasks to those every selected model has attempted, so "
        "rows are apples-to-apples. The `support` column shows "
        "`n_cells | n_attempts` — how many tasks contributed and the "
        "total attempts those cells summed to."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="lb_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="lb_tasks")

    c4, c5, c6, c7 = st.columns([2, 2, 1, 1])
    with c4:
        intersection = st.checkbox(
            "Restrict to intersection (only tasks every selected model has attempted)",
            value=True, key="lb_intersection",
            help=("Off → each model is averaged over whatever tasks it has data "
                  "for, so rows aren't directly comparable. The `support` "
                  "column shows the per-model task/attempt count either way."),
        )
    with c5:
        mode = st.radio(
            "Aggregation",
            ["raw macro-avg", "Beta-Binomial shrinkage"],
            index=0, horizontal=True, key="lb_mode",
            help=("Raw: (1/T)·Σ K_t/N_t — every cell weighed equally regardless "
                  "of N_t.  Shrunken: per-model latent Beta(α,β) prior fit by "
                  "MLE; small-N cells get pulled toward the model's average. "
                  "Shrinkage is per-τ (each threshold gets its own prior)."),
        )
    with c6:
        show_ci = st.checkbox(
            "Show 95% CIs", value=False, key="lb_show_ci",
            help=("Two-stage parametric bootstrap (B=500): resample cells with "
                  "replacement, then resample within-cell binomially. CI "
                  "captures uncertainty in 'which tasks happened to be sampled' "
                  "+ within-cell binomial noise. Cell text becomes "
                  "'point [lo, hi]'; sort and color still use the point estimate."),
        )
    with c7:
        sort_options = [_tau_col(t) for t in LEADERBOARD_TAUS] + ["slug"]
        default_sort = _tau_col(0.50)
        sort_col = st.selectbox(
            "Sort by", sort_options,
            index=sort_options.index(default_sort),
            key="lb_sort",
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    lb, lb_lo, lb_hi, tasks_used = build_leaderboard(
        df, sel_slugs, sel_tasks, LEADERBOARD_TAUS, intersection,
        shrunken=(mode == "Beta-Binomial shrinkage"),
        ci_B=500 if show_ci else 0,
    )

    if intersection and not tasks_used:
        st.warning(
            "No tasks are common to every selected model — try unchecking "
            "'restrict to intersection', or removing the model with the "
            "thinnest coverage."
        )
        return

    # Sort the point dataframe; CIs follow via reindex so colors / values
    # stay aligned with the sorted display.
    if sort_col == "slug":
        order = lb.sort_values("slug").index
    else:
        order = lb.sort_values(sort_col, ascending=False,
                               na_position="last").index
    lb = lb.reindex(order).reset_index(drop=True)
    if lb_lo is not None:
        lb_lo = lb_lo.reindex(order).reset_index(drop=True)
        lb_hi = lb_hi.reindex(order).reset_index(drop=True)

    st.markdown(f"**Aggregated over {len(tasks_used)} task(s):** "
                + (", ".join(f"`{t}`" for t in tasks_used) if tasks_used else "_none_"))

    tau_cols = [_tau_col(t) for t in LEADERBOARD_TAUS]
    if show_ci:
        # Build a display dataframe whose τ cells are "0.91 [0.83, 0.96]"
        # strings. The original `lb` dataframe stays around as the source
        # of truth for cell coloring (point estimate → colour), since the
        # styled apply lambda below indexes into it directly.
        display = lb.copy()
        for col in tau_cols:
            display[col] = [
                f"{p:.2f} [{lo:.2f}, {hi:.2f}]" if pd.notna(p) else "—"
                for p, lo, hi in zip(lb[col], lb_lo[col], lb_hi[col])
            ]
        styled = display.style.apply(
            lambda c: [
                f"background-color: {_prob_color(v)}" if pd.notna(v) else ""
                for v in lb[c.name]
            ] if c.name in tau_cols else [""] * len(c),
            axis=0,
        )
    else:
        styled = lb.style.apply(
            lambda c: [
                f"background-color: {_prob_color(v)}" if pd.notna(v) else ""
                for v in c
            ] if c.name in tau_cols else [""] * len(c),
            axis=0,
        ).format({c: "{:.2f}" for c in tau_cols}, na_rep="—")

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(lb) + 1) + 10, 600),
        key="lb_table",
    )


# --- Cell page -----------------------------------------------------------

def _per_turn_trend(rows: pd.DataFrame) -> str:
    """Compact text sparkline: '+0.123 ▸ err ▸ +0.215 ▸ +0.220'."""
    rows = rows.sort_values("turn")
    parts = []
    for _, r in rows.iterrows():
        if pd.notna(r.get("mcc")):
            parts.append(f"{r['mcc']:+.3f}")
        else:
            parts.append("err")
    return " ▸ ".join(parts)


def _attempts_table(cell_df: pd.DataFrame) -> pd.DataFrame:
    """One row per attempt in the cell. Columns:
       attempt, timestamp, best_mcc, n_turns, n_errs, per_turn_trend,
       model, attempt_id."""
    rows = []
    for aid, group in cell_df.groupby("attempt_id"):
        group = group.sort_values("turn")
        first = group.iloc[0]
        mccs = group["mcc"].dropna() if "mcc" in group.columns else pd.Series([], dtype=float)
        best = float(mccs.max()) if len(mccs) else None
        n_turns = len(group)
        n_errs = int(group.get("error", pd.Series([None] * len(group))).notna().sum())
        ts = first.get("attempt_timestamp", "")
        # Truncate "2026-05-10T11:31:42.123456+00:00" → "2026-05-10 11:31"
        if isinstance(ts, str) and len(ts) >= 16:
            ts_short = ts[:10] + " " + ts[11:16]
        else:
            ts_short = str(ts)
        short_aid = aid.rsplit("-", 1)[-1] if "-" in aid else aid[-12:]
        rows.append({
            "attempt": short_aid,
            "timestamp": ts_short,
            "best_mcc": best,
            "n_turns": n_turns,
            "n_errs": n_errs,
            "per_turn_trend": _per_turn_trend(group),
            "model": first.get("model", ""),
            "attempt_id": aid,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Default sort: best_mcc desc, then timestamp desc.
        out = out.sort_values(["best_mcc", "timestamp"],
                              ascending=[False, False],
                              na_position="last").reset_index(drop=True)
    return out


# --- Curves page ---------------------------------------------------------

def render_curves_page(df: pd.DataFrame) -> None:
    st.markdown("### Reliability curves — P(MCC ≥ τ) per model with 95% CI")
    st.caption(
        "Same numbers as the Leaderboard, plotted as a line chart. Each "
        "model gets one line (point estimate of P(MCC ≥ τ)) plus a "
        "translucent shaded band for its 95% bootstrap CI. The shape of "
        "the curve says something the table can't: a flat-then-cliff "
        "profile means 'usually near-perfect, rarely exactly perfect', "
        "vs a smooth slope meaning 'gradually degrading reliability'. "
        "Click any model in the legend to fade the others."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="curves_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="curves_tasks")

    c3, c4 = st.columns([2, 2])
    with c3:
        intersection = st.checkbox(
            "Restrict to intersection (only tasks every selected model has attempted)",
            value=True, key="curves_intersection",
            help=("Off → each model is averaged over whatever tasks it has data "
                  "for, so curves aren't directly comparable."),
        )
    with c4:
        mode = st.radio(
            "Aggregation",
            ["raw macro-avg", "Beta-Binomial shrinkage"],
            index=0, horizontal=True, key="curves_mode",
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    lb, lb_lo, lb_hi, tasks_used = build_leaderboard(
        df, sel_slugs, sel_tasks, LEADERBOARD_TAUS, intersection,
        shrunken=(mode == "Beta-Binomial shrinkage"),
        ci_B=500,
    )

    if intersection and not tasks_used:
        st.warning(
            "No tasks are common to every selected model — try unchecking "
            "'restrict to intersection', or removing the model with the "
            "thinnest coverage."
        )
        return

    # Reshape (slug × τ-columns) → long-form (slug, τ, point, lo, hi).
    # Altair wants tidy long-form; one row per (slug, τ) data point.
    long_rows: list[dict] = []
    for i, row in lb.iterrows():
        slug = row["slug"]
        for tau in LEADERBOARD_TAUS:
            col = _tau_col(tau)
            long_rows.append({
                "slug": slug,
                "tau": tau,
                "point": row[col],
                "lo": lb_lo.iloc[i][col],
                "hi": lb_hi.iloc[i][col],
            })
    long_df = pd.DataFrame(long_rows).dropna(subset=["point"])

    st.markdown(f"**Aggregated over {len(tasks_used)} task(s):** "
                + (", ".join(f"`{t}`" for t in tasks_used) if tasks_used else "_none_"))

    # Legend-click selection: clicking a model in the legend fades the
    # others. Using mark_point so the selection works on the line color
    # encoding (`fields=['slug']`); applied to both layers so they fade
    # together.
    selection = alt.selection_point(fields=["slug"], bind="legend")

    band = (
        alt.Chart(long_df)
        .mark_area(opacity=0.18)
        .encode(
            x=alt.X("tau:Q",
                    title="threshold τ",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("lo:Q",
                    title="P(MCC ≥ τ)",
                    scale=alt.Scale(domain=[0, 1])),
            y2="hi:Q",
            color=alt.Color("slug:N", legend=alt.Legend(title="model")),
            opacity=alt.condition(selection, alt.value(0.18), alt.value(0.03)),
        )
    )

    line = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(size=20))
        .encode(
            x="tau:Q",
            y=alt.Y("point:Q", scale=alt.Scale(domain=[0, 1])),
            color="slug:N",
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("slug:N", title="model"),
                alt.Tooltip("tau:Q", title="τ", format=".2f"),
                alt.Tooltip("point:Q", title="P", format=".3f"),
                alt.Tooltip("lo:Q", title="lo", format=".3f"),
                alt.Tooltip("hi:Q", title="hi", format=".3f"),
            ],
        )
    )

    chart = (band + line).add_params(selection).properties(
        height=500
    ).interactive(bind_x=False, bind_y=False)

    st.altair_chart(chart, use_container_width=True)


def render_cell_page(df: pd.DataFrame) -> None:
    st.markdown("### Cell view — attempts in one (slug, task)")
    st.caption(
        "Sortable list of all attempts in the chosen cell. "
        "Select a row + click 'View attempt →' to drill into per-turn detail."
    )

    all_slugs = sorted(df["slug"].dropna().unique())
    all_tasks = sorted(df["task"].dropna().unique())

    c1, c2 = st.columns(2)
    with c1:
        default_slug = (st.session_state.sel_slug
                        if st.session_state.sel_slug in all_slugs
                        else all_slugs[0] if all_slugs else None)
        sel_slug = st.selectbox(
            "Slug", all_slugs,
            index=all_slugs.index(default_slug) if default_slug else 0,
        )
    with c2:
        # Filter task options to those that have data for this slug.
        valid_tasks = sorted(df[df["slug"] == sel_slug]["task"].unique()) if sel_slug else []
        if not valid_tasks:
            st.warning(f"No data for slug `{sel_slug}`.")
            return
        default_task = (st.session_state.sel_task
                        if st.session_state.sel_task in valid_tasks
                        else valid_tasks[0])
        sel_task = st.selectbox(
            "Task", valid_tasks,
            index=valid_tasks.index(default_task) if default_task else 0,
        )

    cell_df = df[(df["slug"] == sel_slug) & (df["task"] == sel_task)]
    if cell_df.empty:
        st.warning(f"No data for ({sel_slug}, {sel_task}).")
        return

    # Cell summary line (mirror Matrix's cell text).
    cell_stat = aggregate_cell(cell_df)
    st.markdown(f"**Cell:** `{sel_slug}` × `{sel_task}` — {cell_stat['text']}")

    # Attempts table with row selection.
    table = _attempts_table(cell_df)

    # Hide the full attempt_id by default (it's wide); keep it last column
    # so it's available on horizontal scroll.
    display_cols = ["attempt", "timestamp", "best_mcc", "n_turns",
                    "n_errs", "per_turn_trend", "model", "attempt_id"]
    table = table[display_cols]

    event = st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(table) + 1) + 10, 600),
        selection_mode="single-row",
        on_select="rerun",
        key="cell_attempts_table",
    )

    # Drill-in button — enabled only when a row is selected.
    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    if selected_rows:
        row_idx = selected_rows[0]
        selected_aid = table.iloc[row_idx]["attempt_id"]
        st.markdown(f"Selected: `{selected_aid}`")
        if st.button("View attempt →"):
            _go_to("Attempt", sel_attempt=selected_aid)
    else:
        st.caption("Click a row in the table above to enable 'View attempt →'.")


# --- Attempt page wrapper -----------------------------------------------

def render_attempt_page(df: pd.DataFrame) -> None:
    """Wrap render_per_attempt_tab so a pre-selected attempt from
    session_state is honored on entry (e.g. when the user clicked
    'View attempt →' on the Cell page)."""
    render_per_attempt_tab(df, preselected_aid=st.session_state.sel_attempt)


# --- main dispatcher ----------------------------------------------------

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RESULTS)

    st.set_page_config(page_title="validation-bench viewer",
                       layout="wide", initial_sidebar_state="expanded")
    init_session_state()

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

        st.markdown("---")
        # Radio is unkeyed so it doesn't own `current_page` in session_state.
        # We drive the displayed selection via `index=` and react to user
        # changes by rerunning.
        page_idx = PAGES.index(st.session_state.current_page)
        new_page = st.radio("Page", PAGES, index=page_idx)
        if new_page != st.session_state.current_page:
            st.session_state.current_page = new_page
            st.rerun()

    page = st.session_state.current_page
    if page == "Matrix":
        render_matrix_page(df)
    elif page == "P(MCC≥τ)":
        render_threshold_matrix_page(df)
    elif page == "Leaderboard":
        render_leaderboard_page(df)
    elif page == "Curves":
        render_curves_page(df)
    elif page == "Cell":
        render_cell_page(df)
    elif page == "Attempt":
        render_attempt_page(df)


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
