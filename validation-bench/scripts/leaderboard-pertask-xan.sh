#!/usr/bin/env bash
# Per-task leaderboard for the 6-slug shortlist on the 4 cpp17 tasks.
# Mirrors leaderboard-xan.sh but stops at the per-(slug, task)
# aggregation level so the per-task spread is visible — useful for
# rendering sparkdots / dot-plots / variance summaries on top of the
# headline leaderboard cells.
#
# Output: csv with columns slug, task, a1, a2, a3, a4, a5, n_a
#
# Usage: leaderboard-pertask-xan.sh
# Requires xan and xan-dev (https://github.com/medialab/xan).
set -euo pipefail

RESULTS="$(dirname "$0")/../results/results.jsonl"

SLUGS='slug eq "anthropic-claude-opus-4-7-adaptive"
    or slug eq "anthropic-claude-sonnet-4-6-enabled"
    or slug eq "deepseek-v4-pro-thinking"
    or slug eq "fireworks-glm-5p1"
    or slug eq "gpt-5.5-xhigh"
    or slug eq "moonshot-kimi-k2.6-thinking"'

TASKS='task eq "yaml-1.2-cpp17"
    or task eq "yaml-1.2-nospec-cpp17"
    or task eq "toml-1.0-cpp17"
    or task eq "toml-1.0-nospec-cpp17"'

xan-dev from "$RESULTS" \
  | xan-dev filter "$SLUGS" \
  | xan-dev filter "$TASKS" \
  | xan-dev groupby attempt_id,slug,task \
      'max(if(turn < 1, or(mcc, -1), -1)) as mcc_of_1,
       max(if(turn < 2, or(mcc, -1), -1)) as mcc_of_2,
       max(if(turn < 3, or(mcc, -1), -1)) as mcc_of_3,
       max(if(turn < 4, or(mcc, -1), -1)) as mcc_of_4,
       max(if(turn < 5, or(mcc, -1), -1)) as mcc_of_5' \
  | xan-dev groupby slug,task \
      'mean(mcc_of_1) as a1,
       mean(mcc_of_2) as a2,
       mean(mcc_of_3) as a3,
       mean(mcc_of_4) as a4,
       mean(mcc_of_5) as a5,
       count() as n_a' \
  | xan sort -s slug,task \
  | xan transform a1,a2,a3,a4,a5 'round(_, 0.0001)'
