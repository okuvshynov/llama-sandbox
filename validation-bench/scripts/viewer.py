"""Interactive results viewer for validation-bench.

Usage:
    streamlit run scripts/viewer.py -- <path-to-results.jsonl>
    streamlit run scripts/viewer.py                 # defaults to results/results.jsonl

Pages (selectable in the sidebar):
  - Matrix: slug × task cell matrix with toggleable slugs/envs/tasks.
    Cell statistic matches matrix-xan.sh / vb-may-10.sh exactly —
    `[min; max], avg=A, n=K/N`. Drill-in widget below the matrix
    navigates to the Cell page for a chosen (slug, task).
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

import pandas as pd
import streamlit as st


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

PAGES = ["Matrix", "Cell", "Attempt"]


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
    all_envs = sorted(df["env"].dropna().unique()) if "env" in df.columns else []
    all_tasks = sorted(df["task"].unique())

    # Filter widgets — three independent multi-selects.
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        sel_slugs = st.multiselect("Slugs", all_slugs, default=all_slugs)
    with c2:
        sel_envs = st.multiselect("Envs", all_envs, default=all_envs)
    with c3:
        task_candidates = [
            t for t in all_tasks
            if any(t.endswith(f"-{e}") or t == e for e in sel_envs)
        ] if sel_envs else all_tasks
        sel_tasks = st.multiselect("Tasks", all_tasks, default=task_candidates)

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
