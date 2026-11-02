#!/usr/bin/env python3
"""Does past speculation success predict acceptance beyond the DSpark confidence head? (README
step 47.) Logs: ds-completion --spec --spec-n 5 --spec-log (every draft verified, so acceptance
is known for every reached draft): per step `n_verified accepted verify_ms draft_ms conf[0..5)`.

    ~/venvs/ds-ref/bin/python scripts/spec_momentum.py 'results/spec-conf/*-n5.log'

Draft i is reached when drafts 0..i-1 were accepted. A logistic fit of acceptance on the head's
confidence, then with one "momentum" feature added (the previous step's accepted fraction, its
draft-0 surprise, or a running average); compared by log-loss on prompts left out of the fit.
Then, within bins of sigmoid(conf), acceptance after a poor vs a good previous step.
"""
import glob
import sys

import numpy as np


def fit(X, y, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, X.T @ (p - y))
    return w


def logloss(X, y, w):
    p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    rows = []  # prompt, position, conf, accepted, prev surprise, prev accepted fraction, running average
    for f in sorted(glob.glob(sys.argv[1])):
        name = f.split("/")[-1].rsplit("-n5", 1)[0]
        surprise, frac, ema = 0.0, 0.5, 0.5
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            s = line.split()
            k, conf = int(s[1]), [float(x) for x in s[4:9]]
            for i in range(5):
                if k < i:
                    break
                rows.append((name, i, conf[i], 1.0 if k > i else 0.0, surprise, frac, ema))
            surprise = (1.0 if k > 0 else 0.0) - 1 / (1 + np.exp(-conf[0]))
            frac = k / 5
            ema = 0.7 * ema + 0.3 * frac
    names = sorted({r[0] for r in rows})
    y = np.array([r[3] for r in rows])
    c = np.array([r[2] for r in rows])
    print(f"# {len(rows)} reached drafts over {len(names)} prompts: {' '.join(names)}")
    print(f"log-loss of sigmoid(conf) as is: {logloss(c[:, None], y, np.array([1.0])):.4f}")
    print(f"{'features':40s} {'coefficients':28s} {'in-sample':>9s} {'left-out prompts':>16s}")
    feats = [("confidence", None), ("+ previous step's draft-0 surprise", 4), ("+ previous step's accepted fraction", 5),
             ("+ running average of accepted fractions", 6)]
    for label, j in feats:
        X = np.c_[np.ones_like(c), c] if j is None else np.c_[np.ones_like(c), c, [r[j] for r in rows]]
        w = fit(X, y)
        out = 0.0
        for nm in names:
            m = np.array([r[0] == nm for r in rows])
            out += logloss(X[m], y[m], fit(X[~m], y[~m])) * m.sum()
        print(f"{label:40s} {str(np.round(w, 3)):28s} {logloss(X, y, w):9.4f} {out / len(y):16.4f}")
    p = 1 / (1 + np.exp(-c))
    frac = np.array([r[5] for r in rows])
    print("observed acceptance per sigmoid(conf) bin, after a poor (< 0.5 accepted) / good previous step:")
    for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        m = (p >= lo) & (p < hi)
        a, b = m & (frac < 0.5), m & (frac >= 0.5)
        print(f"  {lo:.1f}-{min(hi, 1):.1f}: predicted {p[m].mean():.2f}, after poor {y[a].mean():.2f} (n={a.sum()}), "
              f"after good {y[b].mean():.2f} (n={b.sum()})")


if __name__ == "__main__":
    main()
