#!/usr/bin/env bash
# vb-may-10: hierarchically-ordered cross-env matrix for today's report.
#
# Same cell statistic as matrix-xan.sh — per (slug, task), best-MCC-per-
# attempt aggregated as `[min; max], avg=A, n=K/N`. Three differences:
#
#   1. Slug / task filters are hardcoded to a fixed shortlist (the 5
#      mid-tier slugs and the 8 cpp17 + zig task cells we're reporting
#      on today).
#   2. Columns are ordered HIERARCHICALLY rather than alphabetically:
#        primary  : spec family   (toml-1.0  <  yaml-1.2)
#        secondary: spec presence (with-spec <  nospec)
#        tertiary : env           (cpp17     <  zig)
#      → toml-1.0-cpp17, toml-1.0-zig, toml-1.0-nospec-cpp17,
#        toml-1.0-nospec-zig, yaml-1.2-cpp17, yaml-1.2-zig,
#        yaml-1.2-nospec-cpp17, yaml-1.2-nospec-zig.
#   3. Cell `n` is reported as `K/N` where:
#        K = n_scored — attempts where at least one turn produced a
#            non-null MCC (the same denominator matrix-xan.sh uses).
#        N = n_total  — every attempt the model started (including
#            attempts where every captured turn was a compile_error
#            or where the harness recorded an `error` field on every
#            row). Surfaces the "infra-truncated vs model-failed"
#            gap that matrix-xan hides; see the atomic-results plan
#            item #3 (attempt_status enum) discussion.
#      Empty cells (N=0) render as `<empty>`. All-errors cells
#      (K=0, N>0) render as `n=0/N (no mcc)` so they remain visible
#      in the hierarchy.
#
# Usage: vb-may-10.sh
# Requires xan (https://github.com/medialab/xan).
set -euo pipefail

RESULTS="$(dirname "$0")/../results/results.jsonl"

python3 <<EOF | xan view --cols 500
import json
from collections import defaultdict

SLUGS = [
    "anthropic-claude-sonnet-4-6-enabled",
    "deepseek-v4-pro-thinking",
    "fireworks-glm-5p1",
    "moonshot-kimi-k2.6-thinking",
    "qwen3.6-27b-q4_k_xl",
]

TASKS = [
    "toml-1.0-cpp17",
    "toml-1.0-zig",
    "toml-1.0-nospec-cpp17",
    "toml-1.0-nospec-zig",
    "yaml-1.2-cpp17",
    "yaml-1.2-zig",
    "yaml-1.2-nospec-cpp17",
    "yaml-1.2-nospec-zig",
]

# Track every attempt_id ever seen per (slug, task), AND the best MCC
# across turns for attempts that produced at least one. Two passes
# fused into one: when iterating rows, register the attempt_id in
# attempts_per_st (drives n_total), and update best_mcc[aid] when
# we see a non-null mcc (drives n_scored).
attempts_per_st = defaultdict(set)
best_mcc = {}

for line in open("$RESULTS"):
    r = json.loads(line)
    slug, task = r.get("slug"), r.get("task")
    if slug not in SLUGS or task not in TASKS:
        continue
    aid = r["attempt_id"]
    attempts_per_st[(slug, task)].add(aid)
    mcc = r.get("mcc")
    if mcc is None:
        continue
    prev = best_mcc.get(aid)
    if prev is None or mcc > prev:
        best_mcc[aid] = mcc


def fmt_cell(slug, task):
    aids = attempts_per_st.get((slug, task), set())
    n_total = len(aids)
    if n_total == 0:
        return "<empty>"
    scored = [best_mcc[a] for a in aids if a in best_mcc]
    n_scored = len(scored)
    if n_scored == 0:
        return f"n=0/{n_total} (no mcc)"
    mn, mx = min(scored), max(scored)
    avg = sum(scored) / n_scored
    return f"[{mn:.3f}; {mx:.3f}], avg={avg:.3f}, n={n_scored}/{n_total}"


# Emit CSV: header + one row per slug
header = ["slug"] + TASKS
print(",".join(header))
for slug in SLUGS:
    row = [slug] + [fmt_cell(slug, t) for t in TASKS]
    print(",".join(f'"{c}"' for c in row))
EOF
