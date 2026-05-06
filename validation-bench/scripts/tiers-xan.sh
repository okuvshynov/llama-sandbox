#!/usr/bin/env bash
# Per-slug fraction of attempts whose cumulative best MCC by turn T meets a
# threshold. Same logic as plot_tiers.py — cumulative best is max(mcc) over
# turns 0..T-1 within an attempt — but rendered as a sortable text table.
# Cells are P(best-by-turn-T >= threshold), one column per turn.
#
# Usage: tiers-xan.sh [task] [threshold] [slug_filter]
#   task        — task name (default: yaml-1.2-cpp17)
#   threshold   — MCC cutoff, inclusive (default: 0.75 → "medium" tier or better)
#   slug_filter — xan filter expression (default: true)
#
# Examples:
#   tiers-xan.sh                                                      # yaml, ≥0.75
#   tiers-xan.sh toml-1.0-cpp17 0.95                                  # toml, ≥0.95 ("strong")
#   tiers-xan.sh yaml-1.2-cpp17 0.5 'startswith(slug, "anthropic-")'  # ≥0.5, anthropic only
#   tiers-xan.sh toml-1.0-nospec-cpp17 1.0                            # exact perfect only
# Requires xan and xan-dev (https://github.com/medialab/xan).
set -euo pipefail

TASK="${1:-yaml-1.2-cpp17}"
THRESHOLD="${2:-0.75}"
SLUG_FILTER="${3:-true}"
RESULTS="$(dirname "$0")/../results/results.jsonl"

echo "=== $TASK: P(cumulative best MCC by turn T >= $THRESHOLD) ==="

xan-dev from "$RESULTS" \
  | xan-dev filter "task eq \"$TASK\"" \
  | xan-dev filter "$SLUG_FILTER" \
  | xan-dev groupby attempt_id,slug \
      'max(if(turn < 1, or(mcc, -1), -1)) as mcc_of_1,
       max(if(turn < 2, or(mcc, -1), -1)) as mcc_of_2,
       max(if(turn < 3, or(mcc, -1), -1)) as mcc_of_3,
       max(if(turn < 4, or(mcc, -1), -1)) as mcc_of_4,
       max(if(turn < 5, or(mcc, -1), -1)) as mcc_of_5' \
  | xan-dev groupby slug \
      "count() as n,
       mean(if(mcc_of_1 >= $THRESHOLD, 1, 0)) as p_t1,
       mean(if(mcc_of_2 >= $THRESHOLD, 1, 0)) as p_t2,
       mean(if(mcc_of_3 >= $THRESHOLD, 1, 0)) as p_t3,
       mean(if(mcc_of_4 >= $THRESHOLD, 1, 0)) as p_t4,
       mean(if(mcc_of_5 >= $THRESHOLD, 1, 0)) as p_t5" \
  | xan sort -N -R -s p_t5 \
  | xan transform p_t1,p_t2,p_t3,p_t4,p_t5 \
      'fmt("{}%", slice(fmt("{}", round(_ * 100, 0.1)), 0, 5))' \
  | xan view --cols 200
