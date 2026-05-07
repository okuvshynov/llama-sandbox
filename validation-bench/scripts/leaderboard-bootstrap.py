#!/usr/bin/env python3
"""
Bootstrap SE for the leaderboard cell estimator.

Estimator
---------
For task i in {1..k} (here k=4) with n_i attempts, define
  μ_i = (1/n_i) Σ_j cum_best_MCC_at_turn_T(attempt j of task i)
  θ̂  = (1/k) Σ_i μ_i

Each task contributes equally to the point estimate regardless of n_i.

True variance (closed form, assuming i.i.d. attempts within task)
----------------------------------------------------------------
  Var(μ_i) = σ_i² / n_i
  Var(θ̂)  = (1/k²) Σ_i σ_i² / n_i

Tasks with more attempts contribute *less variance* even though they
contribute *equally to the mean*. That's the asymmetry we want.

Bootstrap (preserving n_i)
--------------------------
We don't know σ_i². Use the empirical within-task distribution as the
plug-in. For b = 1..B: for each task i, draw n_i values WITH replacement
from {X_{i,1},…,X_{i,n_i}}; compute μ_i*; compute θ̂* = (1/k) Σ_i μ_i*.
SE = std({θ̂*}).

As B → ∞, (SE*)² → (1/k²) Σ_i σ̂_i² / n_i, the plug-in for the
closed-form SE. Bootstrap is preferred over the formula because it
handles asymmetric MCC distributions near ±1 honestly via percentile
CIs and trivially extends if the cell statistic changes.

Why NOT resample to a common N
------------------------------
Resampling N values per task gives bootstrap variance σ̂_i² / N — uniform
in N, regardless of how much was actually observed. Three failures:
  1. Wrong scaling: 1/N replaces 1/n_i, dropping the per-task weighting.
  2. Information isn't created by averaging more draws of σ̂_i² that's
     still computed from only n_i observations.
  3. Degenerate case: at n_i = 1, the only bootstrap draw is X_{i,1}
     itself → bootstrap variance = 0 at any N, while the true
     uncertainty is huge. Bootstrap can mirror the sampling distribution
     only when the resample size matches the actual sample size.

I/O
---
Mirrors the slug + task shortlist of leaderboard-xan.sh /
leaderboard-pertask-xan.sh. Outputs CSV: slug, a1..a5 (point estimate),
a1_se..a5_se (bootstrap SE).

Usage: leaderboard-bootstrap.py [results.jsonl]
       leaderboard-bootstrap.py --B 2000 --seed 12648430
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

SLUGS = [
    "anthropic-claude-opus-4-7-adaptive",
    "anthropic-claude-sonnet-4-6-enabled",
    "deepseek-v4-pro-thinking",
    "fireworks-glm-5p1",
    "gpt-5.5-xhigh",
    "moonshot-kimi-k2.6-thinking",
]
TASKS = [
    "toml-1.0-cpp17",
    "toml-1.0-nospec-cpp17",
    "yaml-1.2-cpp17",
    "yaml-1.2-nospec-cpp17",
]
TURNS = 5  # avg_1..avg_5


def cum_best_at(turn_vals, T):
    # Mirrors xan: max(if(turn < T, or(mcc, -1), -1))
    candidates = []
    for t, v in turn_vals.items():
        if t >= T:
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = -1.0
        candidates.append(v)
    return max(candidates) if candidates else -1.0


def load_per_task(path):
    """Returns by_st: (slug, task) -> list[ list[TURNS floats] ],
    one cumulative-best vector per attempt."""
    raw = defaultdict(dict)  # (slug, task, attempt_id) -> {turn -> mcc}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            slug, task = r.get("slug"), r.get("task")
            if slug not in SLUGS or task not in TASKS:
                continue
            attempt = r.get("attempt_id")
            turn = r.get("turn")
            if attempt is None or turn is None:
                continue
            raw[(slug, task, attempt)][int(turn)] = r.get("mcc")

    by_st = defaultdict(list)
    for (slug, task, _), turn_vals in raw.items():
        by_st[(slug, task)].append([cum_best_at(turn_vals, T) for T in range(1, TURNS + 1)])
    return by_st


def cell_stats(task_data, T_idx, B, rng):
    """Returns (point_estimate, bootstrap_SE) for one cell."""
    # task_data: list of len-k, each element is list of len-n_i of cum-best vectors len TURNS
    task_means = [
        sum(v[T_idx] for v in obs) / len(obs) for obs in task_data
    ]
    point = sum(task_means) / len(task_means)

    boot = []
    for _ in range(B):
        tm = []
        for obs in task_data:
            n = len(obs)
            s = 0.0
            for _ in range(n):
                s += obs[rng.randrange(n)][T_idx]
            tm.append(s / n)
        boot.append(sum(tm) / len(tm))

    mean_b = sum(boot) / len(boot)
    var_b = sum((x - mean_b) ** 2 for x in boot) / (len(boot) - 1)
    return point, math.sqrt(var_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default=str(Path(__file__).resolve().parent.parent / "results" / "results.jsonl"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    args = ap.parse_args()

    by_st = load_per_task(args.results)
    rng = random.Random(args.seed)

    header = ["slug"] + [f"a{T}" for T in range(1, TURNS + 1)] + [f"a{T}_se" for T in range(1, TURNS + 1)]
    print(",".join(header))

    for slug in SLUGS:
        task_data = []
        skip = False
        for task in TASKS:
            obs = by_st.get((slug, task))
            if not obs:
                print(f"WARN: no data for {slug} / {task}", file=sys.stderr)
                skip = True
                break
            task_data.append(obs)
        if skip:
            continue

        means = []
        ses = []
        for T_idx in range(TURNS):
            m, s = cell_stats(task_data, T_idx, args.B, rng)
            means.append(m)
            ses.append(s)

        cols = [slug] + [f"{m:.4f}" for m in means] + [f"{s:.4f}" for s in ses]
        print(",".join(cols))


if __name__ == "__main__":
    main()
