#!/usr/bin/env bash
# Per-slug progression of best-MCC-after-N-turns within each attempt, averaged
# across attempts. mcc_of_N = max(mcc) over turns 0..N-1; lets you see whether
# a model one-shots (high mcc_of_1) or relies on the multi-turn submit loop to
# climb (mcc_of_5 >> mcc_of_1). Error rows (no mcc) count as -1 so a botched
# turn pulls the average down.
# Usage: progression-xan.sh [task]   # default: toml-1.0-cpp17
# Requires xan and xan-dev (https://github.com/medialab/xan).
set -euo pipefail

TASK="${1:-toml-1.0-cpp17}"
RESULTS="$(dirname "$0")/../results/results.jsonl"

echo "=== $TASK: avg best-MCC-after-N-turns per slug ==="

xan-dev from "$RESULTS" \
  | xan-dev filter "task eq \"$TASK\"" \
  | xan-dev groupby attempt_id,slug \
      'max(if(turn < 1, or(mcc, -1), -1)) as mcc_of_1,
       max(if(turn < 2, or(mcc, -1), -1)) as mcc_of_2,
       max(if(turn < 3, or(mcc, -1), -1)) as mcc_of_3,
       max(if(turn < 4, or(mcc, -1), -1)) as mcc_of_4,
       max(if(turn < 5, or(mcc, -1), -1)) as mcc_of_5' \
  | xan-dev groupby slug \
      'count() as n,
       mean(mcc_of_1) as avg_1,
       mean(mcc_of_2) as avg_2,
       mean(mcc_of_3) as avg_3,
       mean(mcc_of_4) as avg_4,
       mean(mcc_of_5) as avg_5' \
  | xan sort -N -R -s avg_5 \
  | xan transform avg_1,avg_2,avg_3,avg_4,avg_5 \
      'slice(fmt("{}", round(_, 0.0001)), 0, 7)' \
  | xan view --cols 200
