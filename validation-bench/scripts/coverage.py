#!/usr/bin/env python
"""Print a (slug × task) attempt-count coverage matrix from results.jsonl.

An "attempt" is one distinct attempt_id; each attempt typically produces
N rows in results.jsonl (one per turn), so attempts-per-(slug, task) is
NOT the same as row count. Slugs and tasks that have at least one
attempt in the dataset are listed; combinations with zero attempts show
as "." for visual contrast.

Examples:

    python scripts/coverage.py
    python scripts/coverage.py --transpose         # tasks as rows
    python scripts/coverage.py --format markdown   # paste-into-PR shape
    python scripts/coverage.py --results other.jsonl
"""

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results" / "results.jsonl"


def render_text(row_axis: list[str], col_axis: list[str],
                counts: dict[tuple[str, str], int],
                row_label: str, col_label: str) -> None:
    row_w = max(len(row_label), max(len(r) for r in row_axis))
    col_w = {c: max(3, len(c)) for c in col_axis}
    total_w = max(5, len("total"))

    header_cells = [row_label.ljust(row_w)]
    header_cells += [c.rjust(col_w[c]) for c in col_axis]
    header_cells.append("total".rjust(total_w))
    header = "  ".join(header_cells)
    sep = "-" * len(header)

    print(header)
    print(sep)
    col_totals = {c: 0 for c in col_axis}
    for r in row_axis:
        cells = [r.ljust(row_w)]
        row_total = 0
        for c in col_axis:
            n = counts.get((r, c), 0)
            cells.append((str(n) if n else ".").rjust(col_w[c]))
            row_total += n
            col_totals[c] += n
        cells.append(str(row_total).rjust(total_w))
        print("  ".join(cells))

    print(sep)
    grand = sum(col_totals.values())
    cells = ["total".ljust(row_w)]
    cells += [str(col_totals[c]).rjust(col_w[c]) for c in col_axis]
    cells.append(str(grand).rjust(total_w))
    print("  ".join(cells))


def render_markdown(row_axis: list[str], col_axis: list[str],
                    counts: dict[tuple[str, str], int],
                    row_label: str, col_label: str) -> None:
    rows: list[list[str]] = []
    rows.append([row_label] + col_axis + ["total"])
    col_totals = {c: 0 for c in col_axis}
    for r in row_axis:
        line = [r]
        row_total = 0
        for c in col_axis:
            n = counts.get((r, c), 0)
            line.append(str(n) if n else "—")
            row_total += n
            col_totals[c] += n
        line.append(str(row_total))
        rows.append(line)
    grand = sum(col_totals.values())
    rows.append(["total"] + [str(col_totals[c]) for c in col_axis] + [str(grand)])

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt(r: list[str]) -> str:
        return "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |"

    print(fmt(rows[0]))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows[1:]:
        print(fmt(r))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                   help=f"Path to results.jsonl (default: {DEFAULT_RESULTS})")
    p.add_argument("--format", choices=["text", "markdown"], default="text",
                   help="Output format (default: text)")
    p.add_argument("--transpose", action="store_true",
                   help="Tasks as rows, slugs as columns (default: slugs as rows)")
    args = p.parse_args()

    if not args.results.exists():
        raise SystemExit(f"results file not found: {args.results}")

    pairs: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for line in args.results.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        slug = r.get("slug") or r["model"]
        pairs[(r["task"], slug)].add(r["attempt_id"])

    counts = {pair: len(aids) for pair, aids in pairs.items()}
    tasks = sorted({t for t, _ in counts})
    slugs = sorted({s for _, s in counts})

    if args.transpose:
        # Tasks as rows, slugs as columns. Counts are keyed (task, slug)
        # already, so render_*(row_axis=tasks, col_axis=slugs, counts).
        render = render_markdown if args.format == "markdown" else render_text
        render(tasks, slugs, counts, row_label="task \\ slug", col_label="slug")
    else:
        # Default: slugs as rows, tasks as columns. Re-key counts to (slug, task).
        keyed = {(s, t): n for (t, s), n in counts.items()}
        render = render_markdown if args.format == "markdown" else render_text
        render(slugs, tasks, keyed, row_label="slug \\ task", col_label="task")


if __name__ == "__main__":
    main()
