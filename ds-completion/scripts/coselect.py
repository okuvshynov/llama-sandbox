#!/usr/bin/env python3
"""Which experts are chosen together, and would placing them by it help? (README step 54.)

    ~/venvs/ds-ref/bin/python scripts/coselect.py data/routing/study/*.bin \
        --placement placements/study30.txt --trace results/trace/cpp-matrix.json.gz

1. Within a token, per layer: lift of a pair = how often both are chosen / how often independent
   choices with the same frequencies would pick both. Tight groups would show as many pairs with
   a large lift carrying a large share of the co-selections.
2. Across neighbouring tokens (a verify step's window of w tokens): distinct experts per layer
   vs the same tokens in shuffled order (the share the window gets for free from its neighbours).
3. Placement: the GPU-resident experts of --placement (the runtime's capacities and its
   round-robin deal per layer) vs the same experts dealt to the dies against co-selection
   (fitted on half the prompts, scored on the other half), vs a perfect balance per step (a
   bound). A layer costs max(busiest die, CPU); die costs fitted on --trace's die spans
   (a + b x distinct + c x pairs), the CPU's from README step 50 (0.1 + 0.27 ms x distinct).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from route_stats import read_log  # noqa: E402

N_EXPERT = 384
CAPACITY = [810, 1765, 1765, 1765]  # slots per die in today's runs (die 0 also holds the dense part)
CPU_FIXED_MS, CPU_PER_EXPERT_MS = 0.10, 0.27


def load(paths):
    logs = []
    for p in paths:
        d, nl, k = read_log(p)
        d = d[d["prompt"] == 0]  # generated tokens
        if len(d):
            logs.append((Path(p).stem, d["ids"].astype(np.int64)))  # [T, L, k]
    return logs


def fit_die_cost(trace):
    """least squares of die span us on (1, distinct, pairs), dies 1-3 and die 0 apart"""
    opener = gzip.open if trace.endswith(".gz") else open
    ev = [e for e in json.load(opener(trace, "rt"))["traceEvents"] if e.get("ph") == "X" and e["tid"] >= 10 and "args" in e]
    out = {}
    for name, tids in (("die 0", {10}), ("dies 1-3", {11, 12, 13})):
        rows = [(e["args"]["distinct experts"], e["args"]["pairs"], e["dur"]) for e in ev if e["tid"] in tids and e["args"]["pairs"]]
        X = np.array([[1, r[0], r[1]] for r in rows], float)
        y = np.array([r[2] for r in rows], float)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ coef
        out[name] = coef / 1000.0  # ms
        print(f"  {name}: {coef[0]:.0f} + {coef[1]:.0f} x distinct + {coef[2]:.0f} x pairs us "
              f"(n={len(y)}, residual sd {resid.std():.0f} us)")
    return out


def coselection(logs, n_layer, k):
    print("\n## 1. within a token: lift of expert pairs (both chosen vs independent choices)")
    ids = np.concatenate([d for _, d in logs])  # [T, L, k]
    T = len(ids)
    lifts, shares10, shares3, top = [], [], [], []
    for L in range(n_layer):
        x = np.zeros((T, N_EXPERT), np.float32)
        np.put_along_axis(x, ids[:, L, :], 1.0, axis=1)
        c = x.sum(0)
        C = x.T @ x  # pair counts (diagonal: counts)
        np.fill_diagonal(C, 0)
        p = c / T
        exp = np.outer(p, p) * T  # expected pair count if independent
        mask = np.triu(exp >= 5, 1)  # pairs common enough to judge
        lift = C[mask] / exp[mask]
        lifts.append(lift)
        tot = C[np.triu(np.ones_like(C, bool), 1)].sum()
        shares10.append(C[mask][lift >= 10].sum() / tot)
        shares3.append(C[mask][lift >= 3].sum() / tot)
        i, j = np.unravel_index(np.argmax(np.where(mask, C / np.maximum(exp, 1e-9), 0)), C.shape)
        top.append((L, i, j, C[i, j], exp[i, j]))
    lift = np.concatenate(lifts)
    q = np.percentile(lift, [50, 90, 99, 99.9])
    print(f"{T} tokens x {n_layer} layers; pairs with >= 5 expected co-selections: {len(lift)}")
    print(f"lift: median {q[0]:.2f}, p90 {q[1]:.2f}, p99 {q[2]:.2f}, p99.9 {q[3]:.2f}, max {lift.max():.1f}")
    print(f"share of all co-selections in pairs with lift >= 3: {np.mean(shares3):.1%}, >= 10: {np.mean(shares10):.1%} (mean over layers)")
    best = sorted(top, key=lambda r: -r[3] / r[4])[:5]
    print("strongest pairs: " + "; ".join(f"L{L} {i}+{j}: {n:.0f} vs {e:.1f} expected" for L, i, j, n, e in best))


def windows(logs, n_layer, k, rng):
    print("\n## 2. neighbouring tokens: distinct experts per layer in a window of w tokens")
    print("w   in order   shuffled   saved by neighbours")
    for w in (2, 3, 4, 6):
        a = b = n = 0
        for _, d in logs:
            T = len(d) // w * w
            if T == 0:
                continue
            for src, acc in ((d[:T], 0), (d[rng.permutation(len(d))[:T]], 1)):
                blk = src.reshape(T // w, w, n_layer, k).transpose(0, 2, 1, 3).reshape(T // w, n_layer, w * k)
                s = np.sort(blk, axis=2)
                distinct = 1 + (np.diff(s, axis=2) != 0).sum(2)
                if acc == 0:
                    a += distinct.sum()
                    n += distinct.size
                else:
                    b += distinct.sum()
        print(f"{w}   {a / n:7.2f}   {b / n:8.2f}   {1 - a / b:6.1%}")


def ranked(path):
    r = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        L, e = line.split()[:2]
        r.append((int(L), int(e)))
    return r


def deal_round_robin(rank, n_layer):
    """the runtime's place_experts: per layer the ranked pairs go to the dies in turn"""
    die = -np.ones((n_layer, N_EXPERT), int)
    used = [0] * len(CAPACITY)
    nxt = [0] * n_layer
    free = sum(CAPACITY)
    for L, e in rank:
        if free == 0:
            break
        for t in range(len(CAPACITY)):
            d = (nxt[L] + t) % len(CAPACITY)
            if used[d] < CAPACITY[d]:
                die[L, e] = d
                used[d] += 1
                free -= 1
                nxt[L] = (d + 1) % len(CAPACITY)
                break
    return die


def deal_anti(die0, train, n_layer, k, cost):
    """the same resident experts and the same number per (layer, die), dealt greedily: most used
    first, each to the die whose expected cost grows least given what is already there"""
    ids = np.concatenate([d for _, d in train])
    T = len(ids)
    die = die0.copy()
    for L in range(n_layer):
        x = np.zeros((T, N_EXPERT), np.float32)
        np.put_along_axis(x, ids[:, L, :], 1.0, axis=1)
        c = x.sum(0)
        C = x.T @ x
        res = np.where(die0[L] >= 0)[0]
        quota = np.bincount(die0[L][res], minlength=len(CAPACITY))
        die[L, res] = -1
        members = [[] for _ in CAPACITY]
        for e in res[np.argsort(-c[res])]:
            best, bd = None, -1
            for d in range(len(CAPACITY)):
                if len(members[d]) >= quota[d]:
                    continue
                b = cost["die 0" if d == 0 else "dies 1-3"][1]
                # co-selections with the die's members (they would run one after another there),
                # plus the die's volume (its share of uses) so frequent experts still spread
                score = b * (sum(C[e, m] for m in members[d]) + c[e] * sum(c[m] for m in members[d]) / T)
                if best is None or score < best:
                    best, bd = score, d
            members[bd].append(e)
            die[L, e] = bd
    return die


def layer_cost(step_ids, die, n_layer, cost, balanced=False):
    """mean ms per layer over steps: max(busiest die, CPU)"""
    tot = 0.0
    n = 0
    for blk in step_ids:  # [w, L, k]
        for L in range(n_layer):
            e = blk[:, L, :].ravel()
            where = die[L, e]
            cpu = np.unique(e[where < 0])
            t_cpu = CPU_FIXED_MS + CPU_PER_EXPERT_MS * len(cpu) if len(cpu) else 0.0
            g = where >= 0
            if balanced:  # a perfect split of this step's GPU work over the 4 dies (a bound)
                dist = len(np.unique(e[g]))
                a, b, c = cost["dies 1-3"]
                t_gpu = (a + b * dist / 4 + c * g.sum() / 4) if g.any() else 0.0
            else:
                t_gpu = 0.0
                for d in range(len(CAPACITY)):
                    m = where == d
                    if m.any():
                        a, b, c = cost["die 0" if d == 0 else "dies 1-3"]
                        t_gpu = max(t_gpu, a + b * len(np.unique(e[m])) + c * m.sum())
            tot += max(t_gpu, t_cpu)
            n += 1
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--placement", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    logs = load(a.logs)
    n_layer, k = logs[0][1].shape[1], logs[0][1].shape[2]
    print(f"# {len(logs)} logs, {sum(len(d) for _, d in logs)} generated tokens, {n_layer} routed layers, top-{k}")
    print("# die cost per layer, fitted on the trace's die spans:")
    cost = fit_die_cost(a.trace)
    coselection(logs, n_layer, k)
    windows(logs, n_layer, k, rng)

    print("\n## 3. dealing the GPU-resident experts to the dies (held out: fit on half the prompts, score on the other)")
    rank = ranked(a.placement)
    rr = deal_round_robin(rank, n_layer)
    print(f"resident: {(rr >= 0).sum()} pairs; per die {[(rr == d).sum() for d in range(4)]}")
    order = rng.permutation(len(logs))
    halves = [[logs[i] for i in order[: len(logs) // 2]], [logs[i] for i in order[len(logs) // 2:]]]
    print("w   half   round-robin   anti-affinity   perfect balance (bound)   ms per layer")
    for w in (1, 3, 6):
        for h in (0, 1):
            train, test = halves[1 - h], halves[h]
            anti = deal_anti(rr, train, n_layer, k, cost)
            steps = [d[i:i + w] for _, d in test for i in range(0, len(d) - w + 1, w)]
            r = [layer_cost(steps, x, n_layer, cost) for x in (rr, anti)] + [layer_cost(steps, rr, n_layer, cost, True)]
            print(f"{w}   {h}      {r[0]:.3f}         {r[1]:.3f} ({r[1] / r[0] - 1:+.1%})    {r[2]:.3f} ({r[2] / r[0] - 1:+.1%})")


if __name__ == "__main__":
    main()
