#!/usr/bin/env python
"""One-shot migration: cut over each tasks/<name>/ to the composed shape.

After this script runs, each task directory holds only the per-cell content
that's genuinely unique to that (spec, env) tuple — its preamble. Everything
else (env config, spec body) is centralized in envs/<env>/ and specs/<spec>/.

Before:
    tasks/<name>/
      task.json    # full env meta inline (language, docker_image, ...) + spec + env
      prompt.txt   # ~15-line preamble + "---" + ~1000-line spec body
      tests/, tests.jsonl, corpus/, ...

After:
    tasks/<name>/
      task.json    # just {"spec": "...", "env": "..."}
      preamble.md  # the ~15-line per-cell preamble (extracted)
      tests/, tests.jsonl, corpus/, ...   # unchanged

Idempotent: skips a task whose task.json is already in composed form.
"""

import argparse
import json
import sys
from pathlib import Path


SEPARATOR = "\n\n---\n\n"


def migrate_task(task_dir: Path) -> str:
    """Returns 'migrated' / 'already' / 'error: ...'."""
    task_json_path = task_dir / "task.json"
    if not task_json_path.exists():
        return f"error: missing {task_json_path}"
    data = json.loads(task_json_path.read_text())

    # Already in composed shape if the env-specific fields are absent.
    if "language" not in data and {"spec", "env"} <= set(data.keys()):
        return "already"

    if "spec" not in data or "env" not in data:
        return "error: task.json missing 'spec' and/or 'env' (run step 1 migration first)"

    prompt_path = task_dir / "prompt.txt"
    preamble_path = task_dir / "preamble.md"

    if not prompt_path.exists():
        return f"error: missing {prompt_path}"

    full = prompt_path.read_text()
    if SEPARATOR in full:
        preamble, body = full.split(SEPARATOR, 1)
    else:
        # nospec-style prompt: no spec body, the whole thing is the preamble.
        preamble, body = full, None

    # Write the new files.
    preamble_path.write_text(preamble)
    new_data = {"spec": data["spec"], "env": data["env"]}
    task_json_path.write_text(json.dumps(new_data, indent=2) + "\n")
    prompt_path.unlink()

    return "migrated"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--tasks-dir", default="tasks", type=Path)
    args = p.parse_args()

    if not args.tasks_dir.is_dir():
        print(f"tasks dir not found: {args.tasks_dir}", file=sys.stderr)
        sys.exit(1)

    n_migrated = n_already = n_error = 0
    for d in sorted(args.tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "task.json").exists():
            continue
        result = migrate_task(d)
        if result == "migrated":
            n_migrated += 1
            print(f"  migrated: {d.name}")
        elif result == "already":
            n_already += 1
            print(f"  already:  {d.name}")
        else:
            n_error += 1
            print(f"  ERROR:    {d.name}: {result}", file=sys.stderr)

    print(f"\nTotal: migrated={n_migrated} already={n_already} errors={n_error}")
    if n_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
