#!/usr/bin/env bash
# Pivot results into a slug × task matrix. For each cell, "best-MCC-per-attempt"
# is aggregated across attempts as `[min; max], avg=A, n=N`. By default, "best"
# is taken over all turns of an attempt; pass a third arg to restrict to the
# first K turns (turn is 0-indexed, so `'turn < 2'` = best of turns 0 and 1).
# Usage: matrix-xan.sh [task_filter] [slug_filter] [turn_filter]
#   filters are xan filter expressions; default "true" (no-op).
# Examples:
#   matrix-xan.sh                                            # everything, all turns
#   matrix-xan.sh 'endswith(task, "cpp17")'                  # cpp17 tasks, all turns
#   matrix-xan.sh 'endswith(task, "cpp17")' 'true' 'turn < 2'  # cpp17, first 2 turns only
#   matrix-xan.sh 'true' 'slug eq "gpt-5.5-xhigh"' 'turn < 1'  # one slug, first turn only
# Requires xan (https://github.com/medialab/xan).
set -euo pipefail

TASK_FILTER="${1:-true}"
SLUG_FILTER="${2:-true}"
TURN_FILTER="${3:-true}"
RESULTS="$(dirname "$0")/../results/results.jsonl"

xan from "$RESULTS" \
  | xan filter "$TASK_FILTER" \
  | xan filter "$SLUG_FILTER" \
  | xan filter "$TURN_FILTER" \
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
