#!/usr/bin/env python3
"""The draft router's expert usage (README step 87), from DS_DRAFT_ROUTE_LOG files.

Per draft layer: how concentrated its choices are (the experts that cover 50 / 90 / 95 / 99% of
them, and the ones never chosen), and the test that matters for keeping only part of them: the
top k experts of the other prompts' choices, and the share of the held-out prompt's choices they
cover (each prompt held out in turn; the mean and the worst prompt). Every drafted position of a
block counts (rejected ones too: the draft computes them).

  scripts/draft_route_stats.py LOG... [--k 32,48,64,80,96]
"""
import argparse
import os
from collections import Counter


def read(path):
    """per draft call: [layer][position] -> list of chosen experts"""
    calls = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split('|')[1:]
        calls.append([[int(x) for x in p.split()] for p in parts])
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--k', default='32,48,64,80,96')
    ap.add_argument('--n-expert', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--block', type=int, default=5)
    a = ap.parse_args()
    ks = [int(x) for x in a.k.split(',')]
    per = {}  # prompt -> [layer] Counter
    for p in a.logs:
        name = os.path.basename(p).rsplit('.', 1)[0]
        cnt = [Counter() for _ in range(a.layers)]
        calls = read(p)
        for c in calls:
            for i, ch in enumerate(c):
                cnt[i // a.block].update(ch)
        per[name] = (cnt, len(calls))
    print(f"# {len(per)} prompts, {sum(n for _, n in per.values())} draft calls; {a.layers} draft layers x {a.n_expert} experts, "
          f"every block position's choices")
    print("\n## concentration over all prompts: experts covering 50 / 90 / 95 / 99% of the choices, never chosen")
    for L in range(a.layers):
        tot = Counter()
        for cnt, _ in per.values():
            tot.update(cnt[L])
        n = sum(tot.values())
        vals = sorted(tot.values(), reverse=True)
        cov = []
        for q in (0.5, 0.9, 0.95, 0.99):
            s, k = 0, 0
            while s < q * n:
                s += vals[k]
                k += 1
            cov.append(k)
        print(f"  layer {L}: {n} choices; {cov[0]} / {cov[1]} / {cov[2]} / {cov[3]} experts; never chosen {a.n_expert - len(tot)}")
    print("\n## kept: the top k of the other prompts' choices, the share of the held-out prompt's choices covered (mean, worst prompt)")
    print(f"  {'k':>4s} " + ' '.join(f"{'layer ' + str(L):>16s}" for L in range(a.layers)) + "   all layers: mean, worst")
    for k in ks:
        row, alls = [], []
        for L in range(a.layers):
            shares = []
            for name, (cnt, _) in per.items():
                rest = Counter()
                for o, (c2, _) in per.items():
                    if o != name:
                        rest.update(c2[L])
                keep = {e for e, _ in rest.most_common(k)}
                n = sum(cnt[L].values())
                shares.append(sum(v for e, v in cnt[L].items() if e in keep) / n if n else 1.0)
            row.append(shares)
        for i in range(len(per)):
            alls.append(sum(row[L][i] for L in range(a.layers)) / a.layers)
        print(f"  {k:4d} " + ' '.join(f"{sum(s) / len(s):8.1%} {min(s):6.1%} " for s in row) + f"  {sum(alls) / len(alls):6.1%} {min(alls):6.1%}")
    names = list(per)
    print(f"\n# prompts: {', '.join(names)}; a choice outside the kept set would be renormalized over the call's remaining experts")


if __name__ == '__main__':
    main()
