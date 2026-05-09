#!/usr/bin/env python
"""Replay a saved submission through a fresh sandbox.

Reads a saved solution.<ext> from VB_DATA_DIR/<attempt_id>/submissions/<turn+1>/,
spins up a fresh sandbox using the (composed) task config, and runs the full
test corpus. Prints recorded vs replayed scores. Does not write anywhere.

Useful when an env's docker image / Dockerfile changes (e.g. the
busybox-timeout PID-leak fix that landed 2026-05-09) and you want to
verify whether previously-recorded MCC values reflect the model or the
infra. The attempt_id format `<task>_<slug>_<timestamp>-<suffix>` tells
us the task; the saved submissions dir gives us the source.

Usage:
    scripts/rerun_submission.py <attempt_id> [--turn N]
    scripts/rerun_submission.py <attempt_id> --turn 2 --vb-data-dir /path

Examples:
    scripts/rerun_submission.py \\
        toml-1.0-zig_anthropic-claude-opus-4-7-adaptive_20260509-121623-1831e76b
        # replays turn 0 (submission 1)

    scripts/rerun_submission.py \\
        yaml-1.2-nospec-zig_gpt-5.5-xhigh_20260509-043453-768cc66a --turn 2
        # replays turn 2 (submission 3) — useful when turn 0/1 had compile errors
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from composer import load_task
from validation_bench_lib import Sandbox, handle_submit


def discover_task(attempt_id: str, tasks_root: Path) -> str:
    """Find the longest task name in tasks_root that prefixes attempt_id.

    Attempt-id format is `<task>_<slug>_<timestamp>-<suffix>`. Both task
    and slug can contain hyphens, so the underscore that separates them
    from the timestamp is unambiguous, but the boundary between task and
    slug isn't. Prefix-matching against the actual tasks/ directory is
    the cheapest reliable way to recover the task.
    """
    candidates = sorted(
        (d.name for d in tasks_root.iterdir() if d.is_dir()),
        key=len, reverse=True,
    )
    for task in candidates:
        if attempt_id.startswith(task + "_"):
            return task
    raise SystemExit(
        f"Could not match attempt_id prefix to any data/tasks/ entry: {attempt_id}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Replay a saved submission through a fresh sandbox.",
    )
    ap.add_argument("attempt_id",
                    help="Attempt directory name under VB_DATA_DIR (the same "
                         "string stored in results.jsonl's `attempt_id` field).")
    ap.add_argument("--turn", type=int, default=0,
                    help="Which turn to replay (0-indexed). Submission dir is "
                         "turn+1. Default: 0.")
    ap.add_argument("--vb-data-dir",
                    default=os.environ.get("VB_DATA_DIR")
                            or str(Path.home() / ".vb-data"),
                    help="VB_DATA_DIR root (where attempt subdirs live). "
                         "Default: $VB_DATA_DIR or ~/.vb-data")
    args = ap.parse_args()

    vb_data = Path(args.vb_data_dir)
    attempt_dir = vb_data / args.attempt_id
    if not attempt_dir.is_dir():
        raise SystemExit(f"Attempt dir not found: {attempt_dir}")

    sub_dir = attempt_dir / "submissions" / str(args.turn + 1)
    if not sub_dir.is_dir():
        raise SystemExit(f"Submission dir not found: {sub_dir}")

    task = discover_task(args.attempt_id, HERE / "data" / "tasks")
    config, _ = load_task(HERE / "data" / "tasks" / task)

    src_filename = config.source_filename  # e.g. "solution.zig", "solution.go"
    src_path = sub_dir / src_filename
    if not src_path.exists():
        raise SystemExit(f"Source not found: {src_path}")
    src = src_path.read_text()

    spec_root = HERE / "data" / "specs" / config.spec
    tests = [json.loads(l) for l in (spec_root / "tests.jsonl").open()]

    print(f"attempt:    {args.attempt_id}")
    print(f"task:       {task}  (env={config.env}, docker_image={config.docker_image})")
    print(f"turn:       {args.turn}  (submission dir: {sub_dir.relative_to(vb_data)})")
    print(f"source:     {src_path.name}, {len(src):,} chars")
    print(f"corpus:     {len(tests):,} tests")
    print()

    sandbox = Sandbox(config)
    sandbox.start()
    try:
        res = handle_submit(src, tests, sandbox, spec_root)
    finally:
        sandbox.stop()

    if not res.compiled:
        print(f"REPLAY: compile_error")
        compiler = (res.compiler_output or "").strip()
        if compiler:
            print("--- compiler output ---")
            print(compiler[:2000])
        return

    m = res.matrix
    total = m.tp + m.fn + m.fp + m.tn
    passed = m.tp + m.tn
    print(f"REPLAY: MCC={m.mcc:+.4f}  TP={m.tp} FN={m.fn} FP={m.fp} TN={m.tn}  passed={passed}/{total}")
    print(f"        prepare={res.prepare_seconds:.1f}s  tests={res.tests_seconds:.1f}s")


if __name__ == "__main__":
    main()
