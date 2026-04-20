#!/usr/bin/env bash
# Per-slug attempt counts across tasks — one column per task.
# Requires xan (https://github.com/medialab/xan).
set -euo pipefail

RESULTS="$(dirname "$0")/results/results.jsonl"

xan from "$RESULTS" \
  | xan groupby attempt_id,slug,task 'max(mcc) as max_mcc' \
  | xan groupby slug,task 'sum(1) as cnt' \
  | xan pivot task 'first(cnt)' -g slug \
  | xan view --cols 200
