#!/usr/bin/env python
"""One-shot migration: backfill `spec` and `env` fields onto pre-0.0.4 result rows.

vb_version 0.0.4 introduced `spec` (what's being implemented, e.g. "toml-1.0")
and `env` (implementation language family, e.g. "cpp") as first-class fields
in every result row, so analysis tools can aggregate "all C++ runs across
specs" or "all TOML 1.0 runs across envs" with a single filter instead of
listing every composite task name.

Older rows lack these fields. The mapping is mechanical (derivable from
the existing `task` field), so this script reads each results.jsonl file,
adds the two fields based on the static map below, and rewrites the file.

Idempotent: rows that already have spec+env are passed through unchanged.

Usage:
    python scripts/migrate_results_spec_env.py results/results.jsonl [more.jsonl...]
"""

import argparse
import json
import sys
from pathlib import Path


# Composite task → (spec, env). Add new entries here as new tasks land
# pre-cutover (i.e. while tasks/<name>/ directories still exist).
TASK_MAPPING = {
    "toml-1.0-cpp":         ("toml-1.0",         "cpp"),
    "toml-1.0-cpp-nospec":  ("toml-1.0-nospec",  "cpp"),
    "toml-1.0-lua":         ("toml-1.0",         "lua"),
    "toml-1.1-cpp":         ("toml-1.1",         "cpp"),
    "toml-1.1-cpp-nospec":  ("toml-1.1-nospec",  "cpp"),
    "lua-5.4-cpp":          ("lua-5.4",          "cpp"),
}


def migrate_file(path: Path) -> tuple[int, int, int]:
    """Returns (touched, already_had_fields, unmapped). Rewrites in place."""
    touched = already = unmapped = 0
    out_lines: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        r = json.loads(line)
        if "spec" in r and "env" in r:
            already += 1
            out_lines.append(line)
            continue
        task = r.get("task")
        mapping = TASK_MAPPING.get(task)
        if mapping is None:
            unmapped += 1
            out_lines.append(line)
            continue
        spec, env = mapping
        # Insert spec/env right after `task` to match the new row shape.
        new_r = {}
        for k, v in r.items():
            new_r[k] = v
            if k == "task":
                new_r["spec"] = spec
                new_r["env"] = env
        out_lines.append(json.dumps(new_r, ensure_ascii=False))
        touched += 1

    path.write_text("\n".join(out_lines) + "\n")
    return touched, already, unmapped


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("files", nargs="+", type=Path,
                   help="Result JSONL files to migrate in place")
    args = p.parse_args()

    total_touched = total_already = total_unmapped = 0
    for f in args.files:
        if not f.exists():
            print(f"  SKIP {f}: not found", file=sys.stderr)
            continue
        t, a, u = migrate_file(f)
        total_touched += t
        total_already += a
        total_unmapped += u
        print(f"  {f}: touched={t} already_had={a} unmapped={u}")

    print(f"\nTotal: touched={total_touched} already_had={total_already} "
          f"unmapped={total_unmapped}")
    if total_unmapped:
        print("WARNING: some rows had a `task` value not in TASK_MAPPING — "
              "they were written through unchanged.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
