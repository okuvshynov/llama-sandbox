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
  - Accuracy: same chart shape as Curves but the underlying score is
    per-attempt accuracy (TP+TN)/(TP+FN+FP+TN) — "fraction of test
    cases classified correctly" — instead of MCC. Useful when you
    care about raw test pass rate rather than the imbalance-adjusted
    MCC; less informative on heavily imbalanced corpora (e.g. an
    "always say valid" classifier scores accuracy ≈ valid_fraction
    while MCC stays ≈ 0).
  - Distribution: bar chart of best-per-attempt MCC counts, bucketed
    into a single "≤0 / failed" group plus 10 right-closed bins of
    width 0.1. Group-by selects how the bars are colored: model
    (slug), spec (e.g. toml-1.0, hcl-2-nospec), spec/nospec (whether
    the spec body was embedded in the prompt), or environment
    (cpp17, zig, ...). Normalize toggle switches between raw counts
    and within-group %.
  - Variance: simulates "you only get one attempt per cell" — for
    each (model, task) cell, picks ONE random attempt and counts as
    pass iff its best-per-turn MCC ≥ τ. The plot is a trace of N
    independent random samples: X axis = seed, Y axis = pass count,
    one line per model. Variance shows up directly as line wiggle —
    flat lines = stable models whose ranking doesn't depend on
    luck; jagged lines = models whose pass count swings by 3-5+
    depending on which attempt got drawn. Correlated jumps across
    several lines = "lucky seed" affecting many models; uncorrelated
    jumps = independent per-model sampling variance.
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
from scipy.stats import betabinom, binom


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

PAGES = ["Matrix", "P(MCC≥τ)", "Leaderboard", "Turn budget", "Curves",
         "Accuracy", "Distribution", "Variance", "Saturation",
         "Pair compare", "Cell", "Attempt"]

DISTRIBUTION_BUCKETS = (
    ["≤0 / failed"]
    + [f"({i / 10:.1f}, {(i + 1) / 10:.1f}]" for i in range(10)]
)

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

def _per_cell_counts(group: pd.DataFrame, taus: list[float],
                     score_col: str = "mcc",
                     max_turns: int | None = None,
                     ) -> tuple[int, dict[float, int]]:
    """For one (slug, task) cell, return (N_total, {τ: K_passed}).

    K_passed = attempts whose best per-turn `score_col` value ≥ τ.
    N_total  = every attempt that started (non-scored attempts count
               toward N but never toward K, so non-compile = fail at
               any τ ≥ 0).

    score_col defaults to "mcc" but can be any per-row numeric column —
    e.g. an accuracy column derived from (tp + tn) / (tp + fn + fp + tn).
    Callers responsible for ensuring the column exists in `group`.

    max_turns: when set, only rows with `turn < max_turns` contribute to
    each attempt's best — i.e. "what would this cell look like if the
    model had only been given max_turns shots?". N_total still reflects
    every attempt that started, so an attempt whose in-budget turns all
    failed to score counts as a fail at any τ ≥ 0 (cap monotonicity:
    raising the cap can only raise pass rates, never lower them).
    """
    n_total = int(group["attempt_id"].nunique())
    if n_total == 0:
        return 0, {tau: 0 for tau in taus}
    filt = group if max_turns is None else group[group["turn"] < max_turns]
    scored = filt.dropna(subset=[score_col])
    if scored.empty:
        return n_total, {tau: 0 for tau in taus}
    best = scored.groupby("attempt_id")[score_col].max()
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
                      score_col: str = "mcc",
                      max_turns: int | None = None,
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
            cell_counts.append(_per_cell_counts(cell_df, taus,
                                                score_col=score_col,
                                                max_turns=max_turns))
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


# --- Turn budget page ----------------------------------------------------
# Same table as Leaderboard but with a per-attempt turn cap. The cap K
# means "only turns 0..K-1 count toward each attempt's best MCC" — i.e.
# what would the model's pass rate look like if it had only been given
# K turns to converge? Denominators stay based on all started attempts
# so the cap is monotone: raising K can only raise pass rates.
#
# Note on K=1 (one-shot): the task prompts explicitly tell the model it
# will be allowed to resubmit, so K=1 systematically under-rates models
# that hedge on the first turn intending to refine. Useful as a lower
# bound, but the K=2..max range is where the interesting comparisons
# live. Also: ~14% of attempts have no turn=0 row (model produced no
# parsable submission on the first shot); those count as fails at K=1.

def render_turn_budget_page(df: pd.DataFrame) -> None:
    st.markdown("### Turn budget — macro-averaged P(MCC ≥ τ) under a turn cap")
    st.caption(
        "Same metric as Leaderboard, but with a per-attempt turn cap. "
        "K=3 answers 'what would each model's pass rate look like if it "
        "only had 3 turns to converge?' — useful for separating models "
        "that succeed on turn 0 from ones that lean on the resubmission "
        "loop. Denominators are unchanged (all started attempts), so the "
        "table is monotone in K: raising the cap can only raise rates. "
        "K=1 (one-shot) is shown but is misleading — the prompt tells "
        "the model it can resubmit, so models that hedge early are "
        "underrated; ~14% of attempts produce no parsable submission on "
        "turn 0 and therefore can't pass at K=1 regardless of skill."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    # Detect observed turn range from data. Turns are 0-indexed; a value
    # of K in the selector means "rows with turn < K count", so the
    # natural maximum is max_turn_observed + 1.
    if "turn" in df.columns and df["turn"].notna().any():
        max_turn_observed = int(df["turn"].max())
    else:
        max_turn_observed = 4
    max_cap = max_turn_observed + 1

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="tb_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="tb_tasks")

    c3, c4, c5, c6 = st.columns([1, 2, 2, 1])
    with c3:
        # Options: 1..max_cap, plus "all" (no filter). "all" is the
        # default — matches the Leaderboard page exactly so users can
        # confirm the new page reproduces the old numbers before
        # lowering the cap.
        cap_options = [str(k) for k in range(1, max_cap + 1)] + ["all"]
        cap_choice = st.selectbox(
            "Turn budget K", cap_options,
            index=len(cap_options) - 1, key="tb_cap",
            help=("K = number of turns the model is allowed. Only rows "
                  "with turn < K contribute to each attempt's best MCC. "
                  "'all' applies no cap and reproduces the Leaderboard."),
        )
        max_turns_filter = None if cap_choice == "all" else int(cap_choice)
    with c4:
        intersection = st.checkbox(
            "Restrict to intersection (only tasks every selected model has attempted)",
            value=True, key="tb_intersection",
            help=("Off → each model is averaged over whatever tasks it has data "
                  "for, so rows aren't directly comparable."),
        )
    with c5:
        mode = st.radio(
            "Aggregation",
            ["raw macro-avg", "Beta-Binomial shrinkage"],
            index=0, horizontal=True, key="tb_mode",
            help=("Same options as the Leaderboard page — see there for "
                  "the math. Shrinkage is per-τ, applied after the turn "
                  "cap, so the prior is fit on cap-aware cell rates."),
        )
    with c6:
        show_ci = st.checkbox(
            "Show 95% CIs", value=False, key="tb_show_ci",
            help=("Two-stage parametric bootstrap (B=500). Same as "
                  "Leaderboard — the CI captures task-resampling and "
                  "within-cell binomial noise after the cap is applied."),
        )

    c7, _ = st.columns([1, 4])
    with c7:
        sort_options = [_tau_col(t) for t in LEADERBOARD_TAUS] + ["slug"]
        default_sort = _tau_col(0.50)
        sort_col = st.selectbox(
            "Sort by", sort_options,
            index=sort_options.index(default_sort),
            key="tb_sort",
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    lb, lb_lo, lb_hi, tasks_used = build_leaderboard(
        df, sel_slugs, sel_tasks, LEADERBOARD_TAUS, intersection,
        shrunken=(mode == "Beta-Binomial shrinkage"),
        ci_B=500 if show_ci else 0,
        max_turns=max_turns_filter,
    )

    if intersection and not tasks_used:
        st.warning(
            "No tasks are common to every selected model — try unchecking "
            "'restrict to intersection', or removing the model with the "
            "thinnest coverage."
        )
        return

    if sort_col == "slug":
        order = lb.sort_values("slug").index
    else:
        order = lb.sort_values(sort_col, ascending=False,
                               na_position="last").index
    lb = lb.reindex(order).reset_index(drop=True)
    if lb_lo is not None:
        lb_lo = lb_lo.reindex(order).reset_index(drop=True)
        lb_hi = lb_hi.reindex(order).reset_index(drop=True)

    cap_label = "no cap" if max_turns_filter is None else f"K={max_turns_filter}"
    st.markdown(
        f"**Turn budget:** {cap_label} &nbsp;·&nbsp; "
        f"**Aggregated over {len(tasks_used)} task(s):** "
        + (", ".join(f"`{t}`" for t in tasks_used) if tasks_used else "_none_")
    )

    tau_cols = [_tau_col(t) for t in LEADERBOARD_TAUS]
    if show_ci:
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
        key="tb_table",
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


# --- Accuracy curves page -----------------------------------------------

def render_accuracy_curves_page(df: pd.DataFrame) -> None:
    st.markdown("### Accuracy curves — P(test pass rate ≥ τ) per model with 95% CI")
    st.caption(
        "Same chart shape as the Curves page, but the underlying score "
        "is per-attempt accuracy `(TP + TN) / (TP + FN + FP + TN)` — "
        "the fraction of test cases this attempt classified correctly. "
        "X axis is τ ∈ [0, 1] (fraction correct); Y axis is the "
        "macro-averaged P(accuracy ≥ τ) across the filtered tasks, "
        "with a translucent band for the 95% bootstrap CI. Caveat: "
        "accuracy can flatter trivial classifiers on imbalanced corpora "
        "— an 'always valid' validator scores accuracy ≈ valid_fraction "
        "while MCC stays ≈ 0. Use Curves (MCC) for the imbalance-"
        "adjusted view; this page for the 'how many tests right' view."
    )

    # Compute the accuracy column from the per-row confusion matrix.
    # NaN-safe: rows with no scored cm (compile errors, no submissions)
    # don't have tp/fn/fp/tn populated, so the sum + division naturally
    # produces NaN, which dropna() then filters out in _per_cell_counts.
    df = df.copy()
    cm_cols = ["tp", "fn", "fp", "tn"]
    if not all(c in df.columns for c in cm_cols):
        st.error(f"Missing confusion-matrix columns in results.jsonl: "
                 f"need {cm_cols}, have {list(df.columns)}")
        return
    cm_total = df[cm_cols].sum(axis=1, min_count=4)
    df["accuracy"] = np.where(cm_total > 0,
                              (df["tp"] + df["tn"]) / cm_total,
                              np.nan)

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="acc_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="acc_tasks")

    c3, c4 = st.columns([2, 2])
    with c3:
        intersection = st.checkbox(
            "Restrict to intersection (only tasks every selected model has attempted)",
            value=True, key="acc_intersection",
        )
    with c4:
        mode = st.radio(
            "Aggregation",
            ["raw macro-avg", "Beta-Binomial shrinkage"],
            index=0, horizontal=True, key="acc_mode",
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    lb, lb_lo, lb_hi, tasks_used = build_leaderboard(
        df, sel_slugs, sel_tasks, LEADERBOARD_TAUS, intersection,
        shrunken=(mode == "Beta-Binomial shrinkage"),
        ci_B=500,
        score_col="accuracy",
    )

    if intersection and not tasks_used:
        st.warning(
            "No tasks are common to every selected model — try unchecking "
            "'restrict to intersection', or removing the model with the "
            "thinnest coverage."
        )
        return

    # Reshape (slug × τ-columns) → long-form (slug, τ, point, lo, hi).
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

    selection = alt.selection_point(fields=["slug"], bind="legend")

    band = (
        alt.Chart(long_df)
        .mark_area(opacity=0.18)
        .encode(
            x=alt.X("tau:Q",
                    title="threshold τ (fraction of tests correct)",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("lo:Q",
                    title="P(accuracy ≥ τ)",
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


# --- Distribution page ---------------------------------------------------

def _bucket_for_mcc(best_mcc: float | None) -> str:
    """Map a best-per-attempt MCC to its bucket label.

    No-MCC attempts (compile errors / no_submissions) and any MCC ≤ 0
    collapse into the single "≤0 / failed" bucket — they're all "the
    model didn't produce a usable validator." Above 0 we use 10
    right-closed bins of width 0.1; MCC=1.0 lands in (0.9, 1.0].
    """
    if best_mcc is None or pd.isna(best_mcc) or best_mcc <= 0:
        return DISTRIBUTION_BUCKETS[0]
    # ceil(mcc * 10) / 10 → upper edge of right-closed bin
    upper = min(1.0, np.ceil(best_mcc * 10) / 10)
    lower = upper - 0.1
    return f"({lower:.1f}, {upper:.1f}]"


def _task_to_env(task: str) -> str:
    """Last hyphen-separated token. Works for every existing env name
    (cpp17, zig, d, go, lua, erlang) since none of them contain '-'."""
    return task.rsplit("-", 1)[-1] if "-" in task else task


def _task_to_spec(task: str) -> str:
    """Everything before the env suffix. e.g. 'hcl-2-nospec-zig' →
    'hcl-2-nospec'."""
    return task.rsplit("-", 1)[0] if "-" in task else task


def _task_to_spec_or_nospec(task: str) -> str:
    """Coarse: did this task embed the spec text in the prompt or not?"""
    return "nospec" if "-nospec-" in f"-{task}-" or task.endswith("-nospec") else "with-spec"


GROUP_BY_OPTIONS = {
    "model": lambda row: row["slug"],
    "spec": lambda row: _task_to_spec(row["task"]),
    "spec/nospec": lambda row: _task_to_spec_or_nospec(row["task"]),
    "environment": lambda row: _task_to_env(row["task"]),
}


def render_distribution_page(df: pd.DataFrame) -> None:
    st.markdown("### Score distribution — best-per-attempt MCC, bucketed")
    st.caption(
        "Each attempt contributes one count. The score plotted is the "
        "best MCC across that attempt's turns. Attempts that never "
        "produced a scored turn (compile errors, no submissions, …) "
        "and attempts with best MCC ≤ 0 (worse-than-coin-flip or "
        "exactly random) collapse into the single leftmost bucket — "
        "the operational definition is 'didn't produce a usable "
        "validator.' Above 0, bins are right-closed of width 0.1; MCC=1.0 "
        "lands in (0.9, 1.0]. Group-by colors the bars; normalize toggles "
        "between raw counts and within-group percentages."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="dist_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="dist_tasks")

    c3, c4 = st.columns([3, 2])
    with c3:
        group_by = st.radio(
            "Group by",
            list(GROUP_BY_OPTIONS.keys()),
            index=0, horizontal=True, key="dist_groupby",
        )
    with c4:
        normalize = st.checkbox(
            "Show as % (normalized within group)", value=False,
            key="dist_normalize",
            help=("Off → raw attempt counts. On → each group sums to 100%. "
                  "Normalize when groups have very different sizes (e.g. "
                  "10 attempts for one slug vs 50 for another) so the "
                  "shape comparison isn't dominated by sample size."),
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    sub = df[df["slug"].isin(sel_slugs) & df["task"].isin(sel_tasks)]

    # One row per attempt: best MCC across turns + group + bucket.
    rows: list[dict] = []
    for (slug, task, aid), g in sub.groupby(["slug", "task", "attempt_id"]):
        mccs = g["mcc"].dropna()
        best = float(mccs.max()) if not mccs.empty else None
        meta = {"slug": slug, "task": task, "attempt_id": aid}
        rows.append({
            **meta,
            "best_mcc": best,
            "bucket": _bucket_for_mcc(best),
            "group": GROUP_BY_OPTIONS[group_by]({"slug": slug, "task": task}),
        })
    if not rows:
        st.warning("No attempts match the current filters.")
        return
    attempts = pd.DataFrame(rows)

    # (group, bucket) → count. Ensure every (group, bucket) cell exists
    # so empty buckets render as 0-height (otherwise Altair drops them
    # from the layout and bars between non-empty buckets shift around).
    counts = (attempts.groupby(["group", "bucket"]).size()
              .reset_index(name="count"))
    all_groups = sorted(attempts["group"].unique())
    full_grid = pd.MultiIndex.from_product(
        [all_groups, DISTRIBUTION_BUCKETS], names=["group", "bucket"]
    ).to_frame(index=False)
    counts = full_grid.merge(counts, on=["group", "bucket"], how="left").fillna({"count": 0})
    counts["count"] = counts["count"].astype(int)
    if normalize:
        totals = counts.groupby("group")["count"].transform("sum")
        # Avoid div-by-zero if a group somehow has no attempts post-filter.
        counts["pct"] = np.where(totals > 0, counts["count"] / totals * 100, 0.0)

    # Summary line above the chart so the user sees the support.
    n_attempts = len(attempts)
    n_failed = int((attempts["bucket"] == DISTRIBUTION_BUCKETS[0]).sum())
    st.markdown(f"**Attempts:** {n_attempts:,} total, {n_failed:,} in "
                f"`≤0 / failed` ({n_failed / n_attempts * 100:.1f}%)")

    y_field = "pct:Q" if normalize else "count:Q"
    y_title = "% of attempts (within group)" if normalize else "attempts"
    tooltip = [
        alt.Tooltip("group:N", title=group_by),
        alt.Tooltip("bucket:O", title="bucket"),
        alt.Tooltip("count:Q", title="count"),
    ]
    if normalize:
        tooltip.append(alt.Tooltip("pct:Q", title="%", format=".1f"))

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("bucket:O", sort=DISTRIBUTION_BUCKETS,
                    title="best-MCC bucket"),
            y=alt.Y(y_field, title=y_title),
            color=alt.Color("group:N",
                            legend=alt.Legend(title=group_by)),
            xOffset=alt.XOffset("group:N"),
            tooltip=tooltip,
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


# --- Variance page ------------------------------------------------------

def _attempt_score_table(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-attempt best MCC table, including attempts that produced no
    scored turn (best=NaN, treated as fail at any τ ≥ 0)."""
    all_attempts = (sub.groupby(["slug", "task", "attempt_id"])
                    .size().reset_index(name="n_turns"))
    best = (sub.dropna(subset=["mcc"])
            .groupby(["slug", "task", "attempt_id"])["mcc"]
            .max().reset_index())
    return all_attempts.merge(best, on=["slug", "task", "attempt_id"], how="left")


def _build_cell_arrays(per_attempt: pd.DataFrame, slugs: list[str], tasks: list[str]
                       ) -> dict[tuple[str, str], np.ndarray]:
    """For each (slug, task), a numpy array of best-MCC values per
    attempt. NaN → -inf (so any threshold rejects). Used by both the
    single-sample and bootstrap-distribution code paths so per-cell
    pandas filtering only happens once."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for slug in slugs:
        for task in tasks:
            arr = per_attempt[(per_attempt.slug == slug)
                              & (per_attempt.task == task)]["mcc"].to_numpy()
            if len(arr) == 0:
                continue
            arr = np.where(np.isnan(arr), -np.inf, arr)
            out[(slug, task)] = arr
    return out


def render_variance_page(df: pd.DataFrame) -> None:
    st.markdown("### Single-attempt variance — pass count per model across N independent random samples")
    st.caption(
        "For each random sample (seed), each (model, task) cell gets ONE "
        "randomly-picked attempt; the attempt counts as a pass iff its "
        "best-per-turn MCC ≥ τ. The chart traces the per-model pass count "
        "as a function of seed — flat lines mean the model's ranking "
        "doesn't depend on which attempt got drawn (Opus typical), wiggly "
        "lines mean the count swings by several tasks based on luck "
        "(mid-tier typical). Correlated jumps across multiple lines at "
        "the same seed = a 'lucky seed' boosting many models together; "
        "uncorrelated jumps = independent per-model variance."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="var_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="var_tasks")

    c3, c4, c5 = st.columns([3, 2, 2])
    with c3:
        tau = st.slider(
            "Threshold τ (best per-attempt MCC must reach this to count as pass)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            key="var_tau",
        )
    with c4:
        intersection = st.checkbox(
            "Restrict to intersection (same task set across models)",
            value=True, key="var_intersection",
        )
    with c5:
        n_samples = st.slider(
            "Samples (X-axis range)",
            min_value=20, max_value=500, value=100, step=10,
            key="var_n_samples",
            help=("Number of independent random samplings to plot. Each "
                  "sample = one random pick per (model, task) cell at the "
                  "same seed. More samples → smoother variance picture, "
                  "but more visual noise on the chart and slightly slower "
                  "to redraw."),
        )

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    sub = df[df["slug"].isin(sel_slugs) & df["task"].isin(sel_tasks)]
    per_attempt = _attempt_score_table(sub)

    # Determine task set per intersection toggle.
    cell_present = (sub.groupby(["slug", "task"]).size()
                    .unstack(fill_value=0) > 0)
    cell_present = cell_present.reindex(index=sel_slugs, columns=sel_tasks,
                                        fill_value=False)
    if intersection:
        tasks_used = [t for t in sel_tasks if bool(cell_present[t].all())]
    else:
        tasks_used = list(sel_tasks)

    if intersection and not tasks_used:
        st.warning(
            "No tasks are common to every selected model — try unchecking "
            "intersection, or removing the model with the thinnest coverage."
        )
        return

    cell_arrays = _build_cell_arrays(per_attempt, sel_slugs, tasks_used)

    # --- Run N samples, accumulate (seed, slug) → pass count.
    # Each seed gets its own RNG so traces are reproducible *and* a "lucky
    # seed" pattern (correlated jumps across slugs) can emerge from shared
    # randomness within a seed. Vectorized over tasks within each slug ×
    # seed; total cost is N × |slugs| × |tasks| ~= 100 × 11 × 11 ≈ 12k
    # numpy index ops, microseconds each.
    trace_rows: list[dict] = []
    for seed in range(int(n_samples)):
        rng = np.random.default_rng(seed)
        for slug in sel_slugs:
            passed = 0
            total = 0
            for task in tasks_used:
                arr = cell_arrays.get((slug, task))
                if arr is None:
                    continue
                total += 1
                if arr[rng.integers(0, len(arr))] >= tau:
                    passed += 1
            trace_rows.append({"seed": seed, "slug": slug,
                               "passed": passed, "total": total})
    trace_df = pd.DataFrame(trace_rows)

    total_tasks = (trace_df.groupby("slug")["total"].first().max()
                   if not trace_df.empty else 0)
    means = (trace_df.groupby("slug")["passed"].mean()
             .sort_values(ascending=False))
    st.markdown(
        f"**τ = {tau:.2f}, samples = {n_samples}, tasks evaluated per "
        f"model: {total_tasks}** (intersection ON → same task set "
        f"everywhere; intersection OFF → per-model coverage, see "
        f"`total` in tooltip). Mean pass count per model over the "
        f"{n_samples} samples: "
        + ", ".join(f"`{s.rsplit('-', 1)[-1]}={m:.1f}`"
                    for s, m in means.items())
    )

    # Color order = mean pass count desc, so legend matches the visual
    # ranking.
    slug_order = means.index.tolist()
    selection = alt.selection_point(fields=["slug"], bind="legend")

    chart = (
        alt.Chart(trace_df)
        .mark_line(opacity=0.85, strokeWidth=1.5)
        .encode(
            x=alt.X("seed:Q", title="random sample (seed)"),
            y=alt.Y("passed:Q", title="passed (out of total)",
                    scale=alt.Scale(domain=[0, max(1, int(total_tasks))])),
            color=alt.Color("slug:N", sort=slug_order,
                            legend=alt.Legend(title="model")),
            opacity=alt.condition(selection, alt.value(0.85), alt.value(0.12)),
            tooltip=[
                alt.Tooltip("slug:N", title="model"),
                alt.Tooltip("seed:Q", title="seed"),
                alt.Tooltip("passed:Q", title="passed"),
                alt.Tooltip("total:Q", title="total"),
            ],
        )
        .add_params(selection)
        .properties(height=460)
        .interactive(bind_y=False)
    )

    st.altair_chart(chart, use_container_width=True)


# --- Saturation page ----------------------------------------------------
# Empirical "diminishing returns on more attempts per cell". The macro-avg
# SE decomposes as Var(p̂) ≈ (1/T)·(Var(p_t) + E[p_t(1-p_t)/N_t]):
# the second term shrinks as 1/N, the first only shrinks with more T.
# This page subsamples N' attempts (without replacement) from each cell
# S times, computes the macro-avg per subsample, and reports the empirical
# SE — recombined in quadrature with the across-cell SE to give a "total
# SE if you'd actually only run N' attempts" estimate. The dashed line
# per slug is the across-cell asymptote: the irreducible part that more
# attempts can never fix.
#
# Caveats baked in:
#   - The across-cell SE uses observed K_t/N_t as proxies for true cell
#     rates. This slightly overestimates Var(p_t) for small N_t (the
#     within-cell sampling noise leaks into the cross-cell sample
#     variance). v1 ships without a bias correction; for N_t ≥ 5 the
#     inflation is modest and the visual story is unchanged.
#   - Subsampling assumes attempts within a cell are exchangeable. If
#     attempts share strong prompt-seeded modes the empirical SE will
#     undershoot what a fresh sweep would show.
#   - Only cells with N_t ≥ N'_max qualify, so the task set is fixed
#     across the X-axis (per intersection toggle). Cells that don't
#     qualify drop out of T entirely for the plot.

def render_saturation_page(df: pd.DataFrame) -> None:
    st.markdown("### Saturation — predicted SE of macro-avg P(MCC ≥ τ) vs attempts-per-cell N′")
    st.caption(
        "How much do you gain from more attempts per cell? Y-axis is "
        "`total SE = √(within² + across²)`. The within term is the "
        "empirical SE from subsampling N′ attempts without replacement "
        "from each cell, S times. The across term is estimated from "
        "Var(K_t/N_t) across cells and is irreducible without more "
        "tasks — shown as the dashed asymptote per slug. The 'reasonable "
        "N′' for a model is where the solid line has visibly merged onto "
        "its dashed asymptote: beyond that point, more attempts per cell "
        "buy you essentially nothing; only broader spec/env coverage will. "
        "Note: 'reasonable N′' is τ-dependent — it's largest where the "
        "model's per-cell rates cluster near 0.5 (max p(1-p)), and "
        "smaller at extremes where cells are mostly 0 or 1."
    )

    all_slugs = sorted(df["slug"].unique())
    all_tasks = sorted(df["task"].unique())

    c1, c2 = st.columns(2)
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="sat_slugs")
    with c2:
        sel_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks,
                                   key="sat_tasks")

    if not sel_slugs or not sel_tasks:
        st.warning("Select at least one slug and one task.")
        return

    sub = df[df["slug"].isin(sel_slugs) & df["task"].isin(sel_tasks)]
    per_attempt = _attempt_score_table(sub)
    if per_attempt.empty:
        st.warning("No attempts in the selected slug × task region.")
        return

    cell_n_series = per_attempt.groupby(["slug", "task"]).size()
    max_possible_np = int(cell_n_series.max())
    if max_possible_np < 2:
        st.warning("Need at least one cell with N_t ≥ 2 for a saturation curve.")
        return

    c3, c4, c5, c6 = st.columns([2, 2, 2, 2])
    with c3:
        tau = st.slider(
            "Threshold τ", min_value=0.0, max_value=1.0,
            value=0.5, step=0.05, key="sat_tau",
            help=("Per-attempt pass = best-per-turn MCC ≥ τ. The saturation "
                  "point is τ-dependent — try shifting τ to see how the "
                  "'reasonable N′' moves."),
        )
    with c4:
        np_max = st.slider(
            "X-axis max N′", min_value=2, max_value=max_possible_np,
            value=min(max_possible_np, 8), step=1, key="sat_np_max",
            help=("Cells with fewer attempts than this are excluded so the "
                  "task set is fixed across the X-axis. Higher = longer "
                  "curve, fewer qualifying cells; lower = shorter curve, "
                  "more cells."),
        )
    with c5:
        n_subsamples = st.slider(
            "Subsample iterations S",
            min_value=50, max_value=1000, value=200, step=50,
            key="sat_n_subsamples",
            help=("How many subsamples per (model, N′) point. Higher S → "
                  "smoother curve, costlier redraw. 200 is usually enough "
                  "for the visual story."),
        )
    with c6:
        intersection = st.checkbox(
            "Restrict to intersection",
            value=True, key="sat_intersection",
            help=("ON → only tasks where EVERY selected slug has at least "
                  "N_t ≥ N′_max; same task set across slugs, fair "
                  "comparison. OFF → each slug uses its own qualifying "
                  "task subset."),
        )

    cell_n_df = (cell_n_series.unstack(fill_value=0)
                 .reindex(index=sel_slugs, columns=sel_tasks, fill_value=0))
    qualifies = cell_n_df >= np_max
    if intersection:
        common = [t for t in sel_tasks if bool(qualifies[t].all())]
        if not common:
            st.warning(
                f"No tasks have N_t ≥ {np_max} for every selected slug. "
                f"Lower 'X-axis max N′' or remove the slug with the "
                f"thinnest coverage."
            )
            return
        tasks_per_slug = {slug: common for slug in sel_slugs}
    else:
        tasks_per_slug = {
            slug: [t for t in sel_tasks if bool(qualifies.loc[slug, t])]
            for slug in sel_slugs
        }

    rng = np.random.default_rng(0)
    np_range = list(range(1, np_max + 1))
    curve_rows: list[dict] = []
    asym_rows: list[dict] = []

    for slug in sel_slugs:
        tasks_used = tasks_per_slug[slug]
        if not tasks_used:
            continue
        # Per-cell attempt arrays. NaN best-MCC → -inf so it fails any τ ≥ 0.
        cell_arrays: list[np.ndarray] = []
        for task in tasks_used:
            arr = per_attempt[(per_attempt.slug == slug)
                              & (per_attempt.task == task)]["mcc"].to_numpy()
            arr = np.where(np.isnan(arr), -np.inf, arr)
            cell_arrays.append(arr)
        T = len(cell_arrays)
        if T == 0:
            continue

        # Across-cell variance contribution to the macro-avg SE: Var(p̂_t)/T.
        # ddof=1 for sample variance; T==1 has no across-cell variance.
        p_hat = np.array([float((arr >= tau).mean()) for arr in cell_arrays])
        across_var = float(np.var(p_hat, ddof=1) / T) if T >= 2 else 0.0
        across_se = float(np.sqrt(across_var))

        # Within-cell SE per N' via subsampling without replacement.
        # argsort-of-random gives independent permutations; first N'
        # columns are the subsampled indices.
        for np_ in np_range:
            rates = np.empty((n_subsamples, T))
            for ti, arr in enumerate(cell_arrays):
                L = len(arr)
                idxs = np.argsort(rng.random((n_subsamples, L)), axis=1)[:, :np_]
                rates[:, ti] = (arr[idxs] >= tau).mean(axis=1)
            macro = rates.mean(axis=1)
            within_se = (float(np.std(macro, ddof=1))
                         if n_subsamples > 1 else 0.0)
            total_se = float(np.sqrt(within_se ** 2 + across_var))
            curve_rows.append({"slug": slug, "np": np_,
                               "within_se": within_se,
                               "across_se": across_se,
                               "total_se": total_se, "T": T})
        asym_rows.append({"slug": slug, "across_se": across_se, "T": T})

    if not curve_rows:
        st.warning(
            "No qualifying cells in any selected slug at this N′_max. "
            "Lower 'X-axis max N′'."
        )
        return

    curve_df = pd.DataFrame(curve_rows)
    asym_df = pd.DataFrame(asym_rows)

    T_per_slug = asym_df.set_index("slug")["T"].to_dict()
    st.markdown(
        f"**τ = {tau:.2f}, S = {n_subsamples}, N′ ∈ [1, {np_max}]** &nbsp;·&nbsp; "
        f"tasks per model (qualifying cells only): "
        + ", ".join(f"`{s.rsplit('-', 1)[-1]}: T={t}`"
                    for s, t in T_per_slug.items())
    )

    # Legend ordered by total_se at N'=1 desc — most-improvable-by-more-attempts on top.
    se_at_1 = (curve_df[curve_df.np == 1].set_index("slug")["total_se"]
               .sort_values(ascending=False))
    slug_order = se_at_1.index.tolist()

    selection = alt.selection_point(fields=["slug"], bind="legend")

    line = (
        alt.Chart(curve_df)
        .mark_line(point=alt.OverlayMarkDef(size=30), strokeWidth=2)
        .encode(
            x=alt.X("np:Q", title="attempts per cell N′",
                    scale=alt.Scale(domain=[1, np_max])),
            y=alt.Y("total_se:Q",
                    title="total SE of macro-avg P(MCC ≥ τ)",
                    scale=alt.Scale(zero=True)),
            color=alt.Color("slug:N", sort=slug_order,
                            legend=alt.Legend(title="model")),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("slug:N", title="model"),
                alt.Tooltip("np:Q", title="N′"),
                alt.Tooltip("total_se:Q", title="total SE", format=".4f"),
                alt.Tooltip("within_se:Q", title="within SE", format=".4f"),
                alt.Tooltip("across_se:Q", title="across SE", format=".4f"),
                alt.Tooltip("T:Q", title="T (tasks)"),
            ],
        )
    )

    # Dashed asymptote per slug: the across-cell SE the curve approaches.
    asymptote = (
        alt.Chart(asym_df)
        .mark_rule(strokeDash=[4, 4], strokeWidth=1.5)
        .encode(
            y="across_se:Q",
            color=alt.Color("slug:N", sort=slug_order, legend=None),
            opacity=alt.condition(selection, alt.value(0.55), alt.value(0.08)),
            tooltip=[
                alt.Tooltip("slug:N", title="model"),
                alt.Tooltip("across_se:Q", title="across SE (asymptote)",
                            format=".4f"),
                alt.Tooltip("T:Q", title="T (tasks)"),
            ],
        )
    )

    chart = ((asymptote + line).add_params(selection)
             .properties(height=480)
             .interactive(bind_x=False, bind_y=False))

    st.altair_chart(chart, use_container_width=True)

    # Companion table: total SE per (slug × N′). Useful for reading off
    # the exact saturation point when the chart's visual flattening is
    # too subtle to eyeball.
    st.markdown("---")
    st.markdown("**Total SE table** (rows = slug, columns = N′):")
    table = (curve_df.pivot(index="slug", columns="np", values="total_se")
             .reindex(slug_order))
    table.columns = [f"N′={c}" for c in table.columns]
    # Append the asymptote as a final column for quick eyeballing of the gap.
    asym_lookup = asym_df.set_index("slug")["across_se"]
    table["asymptote"] = asym_lookup.reindex(slug_order)
    styled = table.style.format("{:.4f}", na_rep="—")
    st.dataframe(styled, use_container_width=True)


# --- Pair compare page --------------------------------------------------
# Paired (model, spec) comparison of pass rates between two envs. The
# question this answers: "for the same model and spec, does env_a make
# the task systematically easier or harder than env_b?". Spec is held
# fixed so spec-difficulty doesn't confound; the model is held fixed so
# model strength doesn't confound; the env varies, isolating its effect.
#
# Caveats baked into the page:
#   - At low τ (e.g. 0.5) many pairs are ceiling-bound (P=1 in both
#     envs) → Δ=0 → ties dominate the sign test. Raising τ converts
#     ties into measurable differences and is usually the right move
#     for env-comparison.
#   - Some pairs have very small N in one env (e.g. N_zig=1). Those
#     P estimates are wildly noisy. Point size is scaled by min(N_a,
#     N_b) so confident pairs are visually dominant.
#   - The bootstrap CI resamples (slug, spec) pairs with replacement —
#     same caveat as the Leaderboard CIs: it quantifies precision of
#     the mean Δ over the observed pair-set, not generalization to
#     unseen spec/env combinations.

def render_pair_compare_page(df: pd.DataFrame) -> None:
    st.markdown("### Pair compare — paired P(MCC ≥ τ) for env_a vs env_b across (model, spec) cells")
    st.caption(
        "Scatter of per-cell pass rates: X = P(MCC ≥ τ) in env_a, "
        "Y = same in env_b. One point per (model, spec) pair where "
        "both envs have ≥ 1 attempt. The dashed diagonal is 'no "
        "difference'; points **above** the diagonal mean env_b is "
        "easier, points **below** mean env_a is easier. Point size = "
        "min(N_a, N_b) so confident pairs dominate. Summary below: "
        "sign-test, mean signed Δ, bootstrap 95% CI on the mean. "
        "Raise τ to convert ceiling-bound ties (both envs at 1.0) into "
        "measurable differences — env-comparison signal at τ=0.5 is "
        "usually swamped by ties when both envs saturate."
    )

    if "env" not in df.columns or "spec" not in df.columns:
        st.error("`env` and `spec` columns required — Pair compare needs them.")
        return

    all_envs = sorted(df["env"].dropna().unique())
    if len(all_envs) < 2:
        st.warning("Need at least 2 envs in the data to pair-compare.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        env_a_default = "cpp17" if "cpp17" in all_envs else all_envs[0]
        env_a = st.selectbox("env_a (X-axis)", all_envs,
                             index=all_envs.index(env_a_default),
                             key="pc_env_a")
    with c2:
        env_b_options = [e for e in all_envs if e != env_a]
        env_b_default = "zig" if "zig" in env_b_options else env_b_options[0]
        env_b = st.selectbox("env_b (Y-axis)", env_b_options,
                             index=env_b_options.index(env_b_default),
                             key="pc_env_b")
    with c3:
        tau = st.slider("Threshold τ", 0.0, 1.0, 0.5, step=0.05, key="pc_tau",
                        help=("Per-attempt pass = best-per-turn MCC ≥ τ. "
                              "Higher τ converts ceiling-bound ties (both "
                              "envs at 1.0) into measurable differences."))
    with c4:
        group_by = st.selectbox(
            "Color by", ["spec", "spec-root", "slug", "none"],
            index=0, key="pc_group",
            help=("spec → full spec (yaml-1.2 vs yaml-1.2-nospec separate). "
                  "spec-root → collapse nospec variants onto their root. "
                  "slug → one color per model. none → all gray."),
        )

    sub = df[df["env"].isin([env_a, env_b])]
    if sub.empty:
        st.warning(f"No data for env_a={env_a} or env_b={env_b}.")
        return

    all_slugs = sorted(sub["slug"].unique())
    all_specs = sorted(sub["spec"].unique())

    c5, c6 = st.columns(2)
    with c5:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs,
                                   key="pc_slugs")
    with c6:
        sel_specs = st.multiselect("Specs", all_specs, default=all_specs,
                                   key="pc_specs")

    if not sel_slugs or not sel_specs:
        st.warning("Select at least one slug and one spec.")
        return

    sub = sub[sub["slug"].isin(sel_slugs) & sub["spec"].isin(sel_specs)]
    if sub.empty:
        st.warning("No rows match the current slug/spec/env selection.")
        return

    # Per-attempt best MCC (reuses the shared helper). Env/spec are
    # attached via task → env/spec lookup since _attempt_score_table
    # carries neither.
    per_attempt = _attempt_score_table(sub)
    task_env = sub.drop_duplicates("task").set_index("task")["env"].to_dict()
    task_spec = sub.drop_duplicates("task").set_index("task")["spec"].to_dict()
    per_attempt["env"] = per_attempt["task"].map(task_env)
    per_attempt["spec"] = per_attempt["task"].map(task_spec)

    # Per (slug, spec, env): pass count, attempt count, pass rate at τ.
    def _stats(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        scored = g["mcc"].dropna()
        n_pass = int((scored >= tau).sum())
        return pd.Series({"n": n, "n_pass": n_pass,
                          "p": n_pass / n if n > 0 else float("nan")})

    cells = (per_attempt.groupby(["slug", "spec", "env"], group_keys=False)
             .apply(_stats).reset_index())
    # Pivot to wide: one row per (slug, spec), columns for each env's p and n.
    p_wide = cells.pivot(index=["slug", "spec"], columns="env", values="p")
    n_wide = cells.pivot(index=["slug", "spec"], columns="env", values="n")
    if env_a not in p_wide.columns or env_b not in p_wide.columns:
        st.warning(
            f"No (slug, spec) pairs have data in BOTH {env_a} and {env_b} "
            "for the current selection."
        )
        return

    paired = pd.DataFrame({
        "p_a": p_wide[env_a],
        "p_b": p_wide[env_b],
        "n_a": n_wide[env_a],
        "n_b": n_wide[env_b],
    }).reset_index().dropna(subset=["p_a", "p_b"])

    if paired.empty:
        st.warning(
            f"No (slug, spec) pairs have data in BOTH {env_a} and {env_b}."
        )
        return

    # Δ > 0 ⇔ env_a easier (P higher); Δ < 0 ⇔ env_b easier.
    paired["delta"] = paired["p_a"] - paired["p_b"]
    paired["min_n"] = paired[["n_a", "n_b"]].min(axis=1).astype(int)

    if group_by == "spec":
        paired["group"] = paired["spec"]
    elif group_by == "spec-root":
        paired["group"] = paired["spec"].str.replace("-nospec", "", regex=False)
    elif group_by == "slug":
        paired["group"] = paired["slug"]
    else:
        paired["group"] = ""

    n_total = len(paired)
    EPS = 1e-9
    n_pos = int((paired["delta"] > EPS).sum())
    n_neg = int((paired["delta"] < -EPS).sum())
    n_zero = n_total - n_pos - n_neg
    mean_delta = float(paired["delta"].mean())

    # Bootstrap CI on the mean Δ — resample (slug, spec) pairs with
    # replacement. Doesn't model within-cell noise (small N pairs are
    # treated as point values); fine for v1 since the dominant variance
    # in our regime is the small number of pairs.
    rng = np.random.default_rng(0)
    B = 2000
    deltas = paired["delta"].to_numpy()
    boot_means = np.array([deltas[rng.integers(0, n_total, n_total)].mean()
                           for _ in range(B)])
    lo, hi = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))

    # Exact two-sided sign test on non-tie pairs. n_eff=0 → undefined.
    n_eff = n_pos + n_neg
    if n_eff > 0:
        k = max(n_pos, n_neg)
        upper = float(binom.sf(k - 1, n_eff, 0.5))
        lower = float(binom.cdf(n_eff - k, n_eff, 0.5))
        p_two = float(min(upper + lower, 1.0))
        sign_txt = f"sign-test p (excl. ties, n_eff={n_eff}): **{p_two:.3f}**"
    else:
        sign_txt = "sign-test undefined (no non-tie pairs)"

    if lo > 0:
        verdict = f"`{env_a}` advantage — CI excludes 0"
    elif hi < 0:
        verdict = f"`{env_b}` advantage — CI excludes 0"
    else:
        verdict = "CI includes 0 — no detectable directional difference"

    st.markdown(
        f"**τ = {tau:.2f}** &nbsp;·&nbsp; **{n_total}** paired (slug, spec) cells "
        f"&nbsp;·&nbsp; Δ = P(`{env_a}`) − P(`{env_b}`). "
        f"Above-diagonal = `{env_b}` easier; below-diagonal = `{env_a}` easier."
    )
    st.markdown(
        f"**Sign:** `{env_a}` > `{env_b}`: **{n_pos}** &nbsp;·&nbsp; "
        f"`{env_a}` < `{env_b}`: **{n_neg}** &nbsp;·&nbsp; "
        f"ties: **{n_zero}** &nbsp;·&nbsp; {sign_txt}"
    )
    st.markdown(
        f"**Mean Δ:** **{mean_delta:+.3f}** &nbsp;·&nbsp; "
        f"**95% bootstrap CI** (B={B}, resample over pairs): "
        f"**[{lo:+.3f}, {hi:+.3f}]** &nbsp;·&nbsp; _{verdict}_"
    )

    # Jitter to break stacking at corners (0,0), (1,1), (0,1), (1,0)
    # where many cells pile up at low τ. Same RNG state regardless of
    # B above (advances internally; that's fine — visual jitter only).
    JITTER = 0.012
    paired["x_j"] = paired["p_a"] + rng.uniform(-JITTER, JITTER, size=n_total)
    paired["y_j"] = paired["p_b"] + rng.uniform(-JITTER, JITTER, size=n_total)

    diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    diag_chart = (
        alt.Chart(diag)
        .mark_line(strokeDash=[4, 4], color="gray", strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )

    use_color = group_by != "none"
    selection = alt.selection_point(fields=["group"], bind="legend")
    color_enc = (alt.Color("group:N", legend=alt.Legend(title=group_by))
                 if use_color else alt.value("#5b6772"))
    opacity_enc = (alt.condition(selection, alt.value(0.8), alt.value(0.12))
                   if use_color else alt.value(0.75))

    points = (
        alt.Chart(paired)
        .mark_circle()
        .encode(
            x=alt.X("x_j:Q", title=f"P({env_a})",
                    scale=alt.Scale(domain=[-0.05, 1.05])),
            y=alt.Y("y_j:Q", title=f"P({env_b})",
                    scale=alt.Scale(domain=[-0.05, 1.05])),
            size=alt.Size("min_n:Q", title="min(N_a, N_b)",
                          scale=alt.Scale(range=[40, 400])),
            color=color_enc,
            opacity=opacity_enc,
            tooltip=[
                alt.Tooltip("slug:N", title="model"),
                alt.Tooltip("spec:N", title="spec"),
                alt.Tooltip("p_a:Q", title=f"P({env_a})", format=".3f"),
                alt.Tooltip("p_b:Q", title=f"P({env_b})", format=".3f"),
                alt.Tooltip("n_a:Q", title=f"N({env_a})"),
                alt.Tooltip("n_b:Q", title=f"N({env_b})"),
                alt.Tooltip("delta:Q", title="Δ", format="+.3f"),
            ],
        )
    )
    if use_color:
        points = points.add_params(selection)

    chart = ((diag_chart + points)
             .properties(height=520)
             .interactive(bind_x=False, bind_y=False))
    st.altair_chart(chart, use_container_width=True)

    # Per-spec breakdown — surfaces whether the cpp17/zig signal is
    # universal or spec-dependent (the answer is usually spec-dependent).
    st.markdown("---")
    st.markdown(
        f"**Per-spec breakdown** — mean Δ over models, "
        f"Δ = P(`{env_a}`) − P(`{env_b}`):"
    )
    per_spec = (paired.groupby("spec")
                .agg(
                    n_models=("slug", "count"),
                    mean_delta=("delta", "mean"),
                    sd_delta=("delta", lambda x: float(x.std(ddof=1))
                              if len(x) > 1 else 0.0),
                    n_pos=("delta", lambda x: int((x > EPS).sum())),
                    n_zero=("delta", lambda x: int((x.abs() <= EPS).sum())),
                    n_neg=("delta", lambda x: int((x < -EPS).sum())),
                )
                .reset_index()
                .sort_values("mean_delta", ascending=False))
    styled = per_spec.style.format({
        "mean_delta": "{:+.3f}",
        "sd_delta": "{:.3f}",
    })
    st.dataframe(styled, hide_index=True, use_container_width=True)


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
    elif page == "Turn budget":
        render_turn_budget_page(df)
    elif page == "Curves":
        render_curves_page(df)
    elif page == "Accuracy":
        render_accuracy_curves_page(df)
    elif page == "Distribution":
        render_distribution_page(df)
    elif page == "Variance":
        render_variance_page(df)
    elif page == "Saturation":
        render_saturation_page(df)
    elif page == "Pair compare":
        render_pair_compare_page(df)
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
