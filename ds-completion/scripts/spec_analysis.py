#!/usr/bin/env python3
"""Speculation in live runs, from --spec-log files (README step 48): how calibrated the DSpark
confidence head is, how the draft-length rule behaves over a run, how far it is from the best
choice per step in hindsight, and whether past success predicts acceptance (step 47's test).

    ~/venvs/ds-ref/bin/python scripts/spec_analysis.py [--max-steps N] results/spec-measure-sclk6/*.log

--max-steps: only each log's first N steps (e.g. before a greedy run falls into a repetition
loop, which the draft predicts perfectly and which would flatter every number).

Log lines (steps that drafted; backoff's plain steps are not logged):
    n_verified accepted verify_ms draft_ms conf[0..5)
Draft i was verified when i < n_verified; it was reached when drafts 0..i-1 were accepted
(accepted >= i), and accepted when accepted > i. Only verified drafts have a known outcome.

Hindsight: with each run's own least-squares verify cost a + b n, a step that accepted k < n of n
drafts would have done best verifying exactly k (k + 1 tokens for a + b k); a step that accepted
all n could have done at least as well as it did (more drafts might have been accepted: unknown),
so the hindsight figure is a lower bound on the best achievable tokens per ms.
"""
import sys
from pathlib import Path

import numpy as np


def read(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        s = line.split()
        rows.append((int(s[0]), int(s[1]), float(s[2]), float(s[3]), [float(x) for x in s[4:9]]))
    return rows


def sig(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def main():
    args = sys.argv[1:]
    max_steps = None
    if args and args[0] == "--max-steps":
        max_steps, args = int(args[1]), args[2:]
    runs = {Path(p).stem: read(p)[:max_steps] for p in args}
    runs = {k: v for k, v in runs.items() if v}
    print(f"# {len(runs)} runs: {' '.join(runs)}")

    # 1. calibration over verified drafts
    pred, obs, pos, who = [], [], [], []
    for name, rows in runs.items():
        for n, k, _, _, conf in rows:
            for i in range(n):
                if k < i:
                    break
                pred.append(sig(conf[i]))
                obs.append(1.0 if k > i else 0.0)
                pos.append(i)
                who.append(name)
    pred, obs, pos = np.array(pred), np.array(obs), np.array(pos)
    ll = -np.mean(obs * np.log(np.clip(pred, 1e-9, 1)) + (1 - obs) * np.log(np.clip(1 - pred, 1e-9, 1)))
    print(f"\n## 1. calibration: {len(obs)} verified drafts, log-loss {ll:.4f} (step 25's calibration prompts: 0.390)")
    print("sigmoid(conf) bin   n     predicted  observed")
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.01)]:
        m = (pred >= lo) & (pred < hi)
        if m.sum():
            print(f"  {lo:.1f}-{min(hi, 1):.1f}        {m.sum():5d}  {pred[m].mean():9.2f}  {obs[m].mean():8.2f}")
    print("by position:      " + "  ".join(f"{i}: {pred[pos == i].mean():.2f}/{obs[pos == i].mean():.2f} (n={int((pos == i).sum())})"
                                       for i in range(5) if (pos == i).sum()))
    print("by run (predicted/observed): " + "  ".join(
        f"{nm} {pred[np.array(who) == nm].mean():.2f}/{obs[np.array(who) == nm].mean():.2f}" for nm in runs))

    # 2. over a run, 3. hindsight
    print("\n## 2. per run: blocks of 25 logged steps (mean drafts verified / accepted, verify ms, draft ms)")
    print("## 3. tokens per ms: achieved vs the best per step in hindsight (lower bound)")
    tok_a = time_a = tok_h = time_h = 0.0
    for name, rows in runs.items():
        n = np.array([r[0] for r in rows], float)
        k = np.array([r[1] for r in rows], float)
        v = np.array([r[2] for r in rows])
        d = np.array([r[3] for r in rows])
        b, a = np.polyfit(n, v, 1) if len(set(n)) > 1 else (0.0, v.mean())
        blocks = []
        for s0 in range(0, len(rows), 25):
            sl = slice(s0, s0 + 25)
            blocks.append(f"{n[sl].mean():.1f}/{k[sl].mean():.1f} {v[sl].mean():.0f} {d[sl].mean():.0f}")
        achieved = (k + 1).sum() / (v + d).sum()
        best_n = np.where(k < n, k, n)  # all accepted: at least what was done
        hind = (best_n + 1).sum() / (d + a + b * best_n).sum()
        tok_a += (k + 1).sum()
        time_a += (v + d).sum()
        tok_h += (best_n + 1).sum()
        time_h += (d + a + b * best_n).sum()
        print(f"{name:12s} {len(rows):4d} steps, verify fit {a:5.1f} + {b:4.1f} n ms; tokens per ms {achieved * 1000:5.1f}/s achieved, "
              f">= {hind * 1000:5.1f}/s in hindsight ({achieved / hind:.0%})")
        print(f"             " + " | ".join(blocks))
    print(f"pooled: {tok_a / time_a * 1000:.1f}/s achieved, >= {tok_h / time_h * 1000:.1f}/s in hindsight "
          f"({(tok_a / time_a) / (tok_h / time_h):.0%})")

    # 4. momentum (step 47's test) over verified drafts: previous logged step's accepted fraction
    prev, y, c, names = [], [], [], []
    for name, rows in runs.items():
        frac = 0.5
        for n, k, _, _, conf in rows:
            for i in range(n):
                if k < i:
                    break
                prev.append(frac)
                y.append(1.0 if k > i else 0.0)
                c.append(conf[i])
                names.append(name)
            frac = k / n if n else frac
    prev, y, c, names = map(np.array, (prev, y, c, names))

    def fit(X, t):
        w = np.zeros(X.shape[1])
        for _ in range(50):
            p = sig(X @ w)
            H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(w))
            w -= np.linalg.solve(H, X.T @ (p - t))
        return w

    def loss(X, t, w):
        p = np.clip(sig(X @ w), 1e-9, 1 - 1e-9)
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

    print("\n## 4. momentum: log-loss on runs left out of the fit")
    for label, X in [("confidence", np.c_[np.ones_like(c), c]), ("+ previous step's accepted fraction", np.c_[np.ones_like(c), c, prev])]:
        out = sum(loss(X[names == nm], y[names == nm], fit(X[names != nm], y[names != nm])) for nm in runs)
        print(f"  {label:38s} {out / len(y):.4f}")


if __name__ == "__main__":
    main()
