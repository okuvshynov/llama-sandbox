# Held-out history (README step 72)

Every decode change as one point: its commit, a one-line summary, and the generation speed on the
8 held-out prompts (`results/spec-heldout/prompts.txt`, 300 tokens each, `--chat`, sclk 6), mean
and per prompt. Per prompt shows whether a change is universal (every prompt gains alike) or
speculation-specific (the prompts that accept many drafted tokens gain most).

- `points.tsv`: the plan, one row per point (step, commit, status, flags, source, summary).
- `history.tsv`, `history.html`: every point's numbers, written by `scripts/heldout_history.py`;
  the page charts the mean by step and each prompt's change from the previous point.
- `runs/<step>-<commit>/`: the reruns (`summary.txt`, each prompt's `.out` and `.err`).

## Points

Steps 48-82 marked `measured` were measured at the time with `scripts/spec-measure.sh` at sclk 6
(the source column names the directory). Earlier steps were measured at sclk 2 or 8 or with other settings, so they
are rerun at sclk 6 from their commits, each with the flags it had then:

| status | steps | settings |
|---|---|---|
| measured | 48, 50, 53, 57, 62, 66, 67, 68, 71, 73, 74, 80, 82, 84, 86, 88, 97, 98 | `--spec --canon` with the cache (200 loads/s at 48, 800/s every 8 tokens from 50) |
| rerun (tier 1) | 6, 7 | dense on GPU 0 (`--gpu 0 --rt`), every routed expert on the CPU |
| | 14, 18 | `--placement study30`: experts split per layer between GPUs and the CPU |
| | 20, 21a, 21b, 22, 23, 24, 25 | `--dspark --spec`: speculative decoding as it was built up |
| | 32, 45 | `--canon --cache-loads 200 --cache-wait`: the adaptive cache, then the -O3 fix |
| optional (tier 2) | 16, 17 | expert-path fixes 1 and 2 on their own (18 holds all three) |
| | 31 | `--canon` before the cache (its cost) |

Not in the history: step 4 (ggml's GPU path, removed in step 11; it needs the fork's HIP build),
steps without a decode change (prompt work 37-44, studies such as 72, 77, 83, 87, 90 and 92, the exit abort (93, 96: teardown only), docs), step 67's state is not a commit
(measured before its commit, which also holds stage 3), and 56, 59, 64, 65, 70, 75, 76, 78, 79, 81, 85, 91 (tried, not kept), 94 (step 70 traced).

Outputs: points from 32 on run `--canon` and must equal `results/cache-test/plain-canon-static`
("output same"). Earlier points use the fast numerics of their time, so their outputs differ
from that reference by design ("DIFFERS" is expected there); the token counts are the same
(300 per prompt).

## Running it

    scripts/heldout-rerun.sh --build rerun optional measured   # worktrees + builds, ~20 s each (done 2026-11-03)
    scripts/heldout-rerun.sh rerun optional measured           # the overnight run, ~4 h 50 min at sclk 6
    watch -n 60 scripts/heldout-status.sh                      # progress; STALLED? after 15 min without a write
    python3 scripts/heldout_history.py                         # history.tsv, history.html

The overnight run includes steps 48-71 again (`measured`, ~4 h; step 67 is skipped, not being a
commit), so every point is measured under one set of conditions; a complete rerun takes
precedence over the source directory, and the runs of the time stay as they are. It warms the
page cache first, runs each point's 8 prompts from its own worktree
(`~/.cache/ds-heldout-history/<commit>`), skips points already complete (so it can be stopped
and restarted), and prints each point's mean. Set the GPU clock to level 6 on all four cards
first. A subset: `scripts/heldout-rerun.sh 20 21a 32` (step labels) or `rerun` (a status).

Checked before the first night (2026-11-03): every point's commit builds (23 worktrees, ~20-40 s
each), and a 24-token smoke run of three of them works with today's model files: 69474c2 (the
oldest runtime, 10.0 t/s), 329d7d2 (the first speculation build) and 5ec0716 (the first cache
build).

## The run (2026-11-03)

14:10-19:00, all 26 points, no failures (`~/heldout-rerun.log`). Means: 6: 10.06, 7: 12.32,
14: 18.08, 18: 19.36, 20: 11.44, 25: 22.92, 31: 20.70, 32: 29.02, 45: 27.55, 48: 28.07,
50: 30.42, 74: 35.94 t/s (all points in `history.tsv` and README step 72). Steps 48-74 agree
with their earlier measurements within 0.2 t/s. Every point from 31 on gives "output same".
