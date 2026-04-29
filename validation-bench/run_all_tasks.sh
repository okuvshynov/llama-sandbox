#!/usr/bin/env bash
# Run every task in tasks/<name>/ against one runner+model. Output goes to
# a separate results dir under results/scratch/ (gitignored) so the sweep
# doesn't co-mingle with the main results/results.jsonl dataset until
# you've decided what's worth keeping.
#
# Usage:
#     ./run_all_tasks.sh <runner.py> <model> [extra runner args...]
#
# Examples:
#     ./run_all_tasks.sh validation_bench_anthropic.py claude-sonnet-4-6 \
#         --thinking enabled --thinking-budget 30000 --max-tokens 100000
#
#     ./run_all_tasks.sh validation_bench_openai.py gpt-5.5 --reasoning-effort high
#
# Override the results dir via the RESULTS_DIR env var:
#     RESULTS_DIR=results/scratch/test ./run_all_tasks.sh validation_bench_openai.py gpt-5.5
#
# Default is one attempt per task, 5 turns. Override either by appending
# `--n-attempts N` / `--max-turns N` to the extra args (later flags win).

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <runner.py> <model> [extra runner args...]" >&2
    exit 1
fi

RUNNER="$1"
MODEL="$2"
shift 2
RESULTS_DIR="${RESULTS_DIR:-results/scratch/sweep}"

cd "$(dirname "$0")"

if [ ! -f "$RUNNER" ]; then
    echo "Runner not found: $RUNNER" >&2
    exit 1
fi

for task_dir in data/tasks/*/; do
    task=$(basename "$task_dir")
    echo
    echo "=== $task ==="
    python "$RUNNER" --task "$task" --model "$MODEL" \
        --results-dir "$RESULTS_DIR" \
        --n-attempts 1 --max-turns 5 \
        "$@"
done

echo
echo "Done. Results in $RESULTS_DIR/results.jsonl"
