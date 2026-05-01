#!/usr/bin/env bash
# Pivot results into a slug × task matrix. For each cell, "best-MCC-per-attempt"
# is aggregated across attempts as `[min; max], avg=A, n=N`.
# Usage: matrix-xan.sh [task_filter] [slug_filter]
#   task_filter, slug_filter are xan filter expressions; default "true" (no-op).
# Examples:
#   matrix-xan.sh                                            # everything
#   matrix-xan.sh 'endswith(task, "cpp17")'                  # cpp17 tasks
#   matrix-xan.sh 'endswith(task, "cpp17")' 'startswith(slug, "anthropic-")'
#   matrix-xan.sh 'true' 'slug eq "gpt-5.5-xhigh" or slug eq "gpt-5.5-high"'
# Requires xan (https://github.com/medialab/xan).
set -euo pipefail

TASK_FILTER="${1:-true}"
SLUG_FILTER="${2:-true}"
RESULTS="$(dirname "$0")/../results/results.jsonl"

xan from "$RESULTS" \
  | xan filter "$TASK_FILTER" \
  | xan filter "$SLUG_FILTER" \
  | xan groupby attempt_id,slug,task 'max(mcc) as max_mcc' \
  | xan filter 'len(max_mcc) > 0' \
  | xan groupby slug,task '
      min(max_mcc) as mn,
      max(max_mcc) as mx,
      mean(max_mcc) as avg,
      count() as n' \
  | xan map '
      fmt("[{}; {}], avg={}, n={}",
          slice(fmt("{}", round(mn,  0.001)), 0, 6),
          slice(fmt("{}", round(mx,  0.001)), 0, 6),
          slice(fmt("{}", round(avg, 0.001)), 0, 6),
          n) as cell' \
  | xan select slug,task,cell \
  | xan pivot task 'first(cell)' \
  | xan sort -s slug \
  | xan view --cols 300
