#!/usr/bin/env python3
"""DSpark confidence head: calibration and an offline test of the draft-length rule.

    python3 scripts/spec_conf.py --cal 'DIR/*-n5.log' [--cost 'DIR/*-n[1-5].log'] [--plain-ms 48]

Logs come from ds-completion --spec --spec-log (with every draft verified for --cal): per
verify step `n_verified accepted verify_ms draft_ms conf[0..5)`.

Calibration: draft i is "reached" when drafts 0..i-1 were all accepted; among reached ones,
the observed acceptance rate per bin of sigmoid(conf[i]) (and per position).

Rule (what ds-completion --spec-auto does): with p_i = sigmoid(a * conf[i] + b) the chance of
draft i given the earlier ones, pick n in 0..5 maximizing
    (1 + p_0 + p_0 p_1 + ... + p_0..p_{n-1}) / (draft_ms + verify_ms(n + 1))
The offline test replays the logged steps: tokens 1 + min(accepted, n), time from the cost
table; it compares with every fixed n on the same steps (approximate: in a real run the step
boundaries, so the drafts, differ).
"""
import argparse
import glob
import math
from collections import defaultdict


def read(pattern):
    steps = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            steps.append((int(v[0]), int(v[1]), float(v[2]), float(v[3]), [float(x) for x in v[4:]], f))
    return steps


def sig(x):
    return 1 / (1 + math.exp(-x))


def fit_platt(pairs, iters=2000, lr=0.05):
    """logistic regression of the label on conf: p = sigmoid(a * conf + b)"""
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for c, y in pairs:
            p = sig(a * c + b)
            ga += (p - y) * c
            gb += p - y
        a -= lr * ga / len(pairs)
        b -= lr * gb / len(pairs)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--cost")
    ap.add_argument("--plain-ms", type=float, default=48.0, help="one plain decode step, ms")
    a = ap.parse_args()
    steps = read(a.cal)
    B = len(steps[0][4])
    print(f"# {len(steps)} verify steps (all {B} drafts verified) from {len({s[5] for s in steps})} logs")

    # calibration
    pairs, per_pos = [], defaultdict(list)
    for n, k, _, _, conf, _ in steps:
        for i in range(min(n, B)):
            if k < i:
                break  # not reached
            y = 1 if k >= i + 1 else 0
            pairs.append((conf[i], y))
            per_pos[i].append((conf[i], y))
    print("\nper position (reached only): n, observed acceptance, mean sigmoid(conf)")
    for i in sorted(per_pos):
        ps = per_pos[i]
        print(f"  draft {i}: {len(ps):4d}  observed {sum(y for _, y in ps) / len(ps):.3f}  "
              f"predicted {sum(sig(c) for c, _ in ps) / len(ps):.3f}")
    print("\nby sigmoid(conf) bin: n, observed acceptance")
    bins = defaultdict(list)
    for c, y in pairs:
        bins[min(int(sig(c) * 10), 9)].append(y)
    for bi in sorted(bins):
        ys = bins[bi]
        print(f"  [{bi / 10:.1f}, {(bi + 1) / 10:.1f}): {len(ys):4d}  observed {sum(ys) / len(ys):.3f}")
    # ranking quality: AUC
    pos = [c for c, y in pairs if y]
    neg = [c for c, y in pairs if not y]
    auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / max(1, len(pos) * len(neg))
    A, Bb = fit_platt(pairs)
    ll = lambda f: -sum(math.log(max(1e-9, f(c) if y else 1 - f(c))) for c, y in pairs) / len(pairs)
    print(f"\nAUC {auc:.3f}; log-loss raw sigmoid {ll(sig):.3f}, refit sigmoid({A:.3f} conf + {Bb:.3f}) {ll(lambda c: sig(A * c + Bb)):.3f}")

    if not a.cost:
        return
    # verify cost by tokens verified (median per n), plus the draft
    by_n = defaultdict(list)
    draft = []
    for n, _, vms, dms, _, _ in read(a.cost) + steps:
        by_n[n].append(vms)
        draft.append(dms)
    med = lambda v: sorted(v)[len(v) // 2]
    cost = {n: med(v) for n, v in by_n.items()}
    cost[0] = a.plain_ms
    d_ms = med(draft)
    print("\nverify ms by drafts verified: " + "  ".join(f"n={n}: {cost[n]:.1f}" for n in sorted(cost)) + f"; draft {d_ms:.1f}")

    def choose(conf, f):
        best, bn, run, exp = -1.0, 0, 1.0, 1.0
        for n in range(0, B + 1):
            if n > 0:
                run *= f(conf[n - 1])
                exp += run
            rate = exp / (d_ms + cost[n])
            if rate > best:
                best, bn = rate, n
        return bn

    def replay(pick):
        tok = ms = 0.0
        for n, k, _, _, conf, _ in steps:
            m = pick(conf)
            tok += 1 + min(k, m)
            ms += d_ms + cost[m]
        return tok / ms * 1000

    print("\noffline replay, t/s (all calibration steps):")
    for m in range(0, B + 1):
        print(f"  fixed n={m}: {replay(lambda c, m=m: m):.2f}")
    print(f"  rule, raw sigmoid: {replay(lambda c: choose(c, sig)):.2f}")
    print(f"  rule, refit:       {replay(lambda c: choose(c, lambda x: sig(A * x + Bb))):.2f}")
    per_log = defaultdict(list)
    for s in steps:
        per_log[s[5]].append(s)
    print("\nper log: best fixed n vs rule (refit)")
    for f, ss in per_log.items():
        def rp(pick):
            tok = sum(1 + min(k, pick(c)) for _, k, _, _, c, _ in ss)
            return tok / sum(d_ms + cost[pick(c)] for _, _, _, _, c, _ in ss) * 1000
        fixed = {m: rp(lambda c, m=m: m) for m in range(B + 1)}
        bm = max(fixed, key=fixed.get)
        print(f"  {f.split('/')[-1]:24s} best fixed n={bm} {fixed[bm]:.2f}   rule {rp(lambda c: choose(c, lambda x: sig(A * x + Bb))):.2f}")


if __name__ == "__main__":
    main()
