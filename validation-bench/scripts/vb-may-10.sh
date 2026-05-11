#!/usr/bin/env bash
# vb-may-10: hierarchically-ordered cross-env matrix for today's report.
#
# Same cell statistic as matrix-xan.sh — per (slug, task), best-MCC-per-
# attempt aggregated as `[min; max], avg=A, n=N`. Two differences:
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
#      Empty cells are shown explicitly as `<empty>` (preserves the
#      slot in the hierarchy when no data exists yet).
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

# attempt_id -> {(slug, task): max_mcc}
best = defaultdict(lambda: (None, None, None))  # (slug, task, max_mcc)
for line in open("$RESULTS"):
    r = json.loads(line)
    if r["slug"] not in SLUGS or r["task"] not in TASKS:
        continue
    mcc = r.get("mcc")
    if mcc is None:
        continue
    key = r["attempt_id"]
    prev = best[key]
    if prev[2] is None or mcc > prev[2]:
        best[key] = (r["slug"], r["task"], mcc)

# (slug, task) -> [best_mcc per attempt]
cells = defaultdict(list)
for (slug, task, mx) in best.values():
    cells[(slug, task)].append(mx)


def fmt_cell(vs):
    if not vs:
        return "<empty>"
    mn, mx = min(vs), max(vs)
    avg = sum(vs) / len(vs)
    return f"[{mn:.3f}; {mx:.3f}], avg={avg:.3f}, n={len(vs)}"


# Emit CSV: header + one row per slug
header = ["slug"] + TASKS
print(",".join(header))
for slug in SLUGS:
    row = [slug] + [fmt_cell(cells.get((slug, t), [])) for t in TASKS]
    # Quote each field so embedded commas in the cell are safe.
    print(",".join(f'"{c}"' for c in row))
EOF
