#!/usr/bin/env bash
# Per-task comparison of best-submission MCC per attempt, grouped by slug,
# with significance annotation against a baseline slug (Welch's t-test).
# Usage: scores-xan.sh [baseline_slug]   # default: fireworks-glm-5p1
# Requires xan and xan-dev (https://github.com/medialab/xan).
set -euo pipefail

BASELINE="${1:-fireworks-glm-5p1}"
RESULTS="$(dirname "$0")/results/results.jsonl"

tasks=$(xan from "$RESULTS" | xan groupby task 'sum(1) as n' | awk -F, 'NR>1 {print $1}')

for task in $tasks; do
  echo "=== $task (baseline: $BASELINE) ==="
  set +e
  xan-dev from "$RESULTS" \
    | xan-dev filter "task eq \"$task\"" \
    | xan-dev groupby attempt_id,slug,task 'max(mcc) as max_mcc' \
    | xan-dev sigtest -g slug -v max_mcc "$BASELINE" \
    | xan view --cols 200
  set -e
  echo
done
