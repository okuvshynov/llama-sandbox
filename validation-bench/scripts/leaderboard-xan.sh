#!/usr/bin/env bash
# Cross-task leaderboard for the 6-slug shortlist on the 4 cpp17 tasks
# (toml/yaml × spec/nospec). Each cell = mean across tasks of the per-task
# avg cumulative best MCC by turn T. Per-task avgs are computed first so
# tasks with more attempts don't dominate.
#
# n_tasks shows how many of the 4 tasks the slug has any data for; cells
# are averaged over only those (not the full 4) — slugs missing data for
# a task get a smaller-denominator average. n_attempts is the total
# attempt count summed across present tasks.
#
# Usage: leaderboard-xan.sh
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

echo "=== Leaderboard: avg cumulative best MCC by turn T (mean across cpp17 tasks) ==="

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
  | xan-dev groupby slug \
      'mean(a1) as avg_1,
       mean(a2) as avg_2,
       mean(a3) as avg_3,
       mean(a4) as avg_4,
       mean(a5) as avg_5,
       count() as n_tasks,
       sum(n_a) as n_attempts' \
  | xan sort -N -R -s avg_5 \
  | xan transform avg_1,avg_2,avg_3,avg_4,avg_5 \
      'slice(fmt("{}", round(_, 0.0001)), 0, 7)' \
  | xan view --cols 200
