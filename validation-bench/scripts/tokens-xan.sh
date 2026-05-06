#!/usr/bin/env bash
# Per-slug cumulative token usage by turn T, averaged across attempts.
# Same denominator/prefix-shape as progression-xan.sh, but reporting token
# sums instead of best-MCC. For each attempt, in_of_K = sum(input_tokens)
# over turns 0..K-1; same for output_tokens. Mean across attempts per slug.
#
# Notes:
#   - Tokens are billing units. Summing input_tokens across turns is correct
#     for cost: each turn pays for its full context replay (provider charges
#     per call, not per unique token).
#   - Reasoning-token reporting is provider-dependent:
#       Anthropic: thinking tokens are folded into output_tokens
#       OpenAI / DeepSeek: thinking tokens are separate in reasoning_tokens
#     (not counted in output_tokens). So output_tokens across providers is
#     not strictly comparable for thinking-mode runs.
#
# Usage: tokens-xan.sh [task] [slug_filter]
#   task        — task name (default: toml-1.0-cpp17)
#   slug_filter — xan filter expression (default: true)
# Examples:
#   tokens-xan.sh toml-1.0-nospec-cpp17
#   tokens-xan.sh yaml-1.2-cpp17 'startswith(slug, "anthropic-")'
# Requires xan and xan-dev (https://github.com/medialab/xan).
set -euo pipefail

TASK="${1:-toml-1.0-cpp17}"
SLUG_FILTER="${2:-true}"
RESULTS="$(dirname "$0")/../results/results.jsonl"

build_table() {
  local field="$1"
  local label="$2"
  echo "=== $TASK: avg cumulative $label tokens by turn T (in K) ==="
  xan-dev from "$RESULTS" \
    | xan-dev filter "task eq \"$TASK\"" \
    | xan-dev filter "$SLUG_FILTER" \
    | xan-dev groupby attempt_id,slug \
        "sum(if(turn < 1, or($field, 0), 0)) as tk_1,
         sum(if(turn < 2, or($field, 0), 0)) as tk_2,
         sum(if(turn < 3, or($field, 0), 0)) as tk_3,
         sum(if(turn < 4, or($field, 0), 0)) as tk_4,
         sum(if(turn < 5, or($field, 0), 0)) as tk_5" \
    | xan-dev groupby slug \
        'count() as n,
         mean(tk_1) as t1,
         mean(tk_2) as t2,
         mean(tk_3) as t3,
         mean(tk_4) as t4,
         mean(tk_5) as t5' \
    | xan sort -N -R -s t5 \
    | xan transform t1,t2,t3,t4,t5 \
        'fmt("{}K", round(_ / 1000.0, 1))' \
    | xan view --cols 200
  echo
}

build_table input_tokens INPUT
build_table output_tokens OUTPUT
