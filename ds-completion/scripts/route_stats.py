#!/usr/bin/env python3
"""Routing statistics from ds-completion --log-routing files: how much of the routed-expert work
a GPU-resident set of experts could serve, for a given amount of HBM.

    ~/venvs/ds-ref/bin/python scripts/route_stats.py data/routing/*.bin [--prompt] [--train a.bin,b.bin]
        [--cross prompts/routing-study.tsv]

Units: a "slot" is one (token, layer, chosen expert); a token has 40 x 6 = 240. A policy keeps a
set of (layer, expert) pairs resident; its hit rate is the fraction of slots it serves.
Capacity = experts that fit in the HBM budget after the dense part on die 0.

Policies:
  static-oracle  the top-C pairs by count over the same tokens (in-sample: an upper bound for
                 any static set)
  static-train   the top-C pairs counted on other files (--train), or on the first half of each
                 file when no --train is given (evaluated on the second half)
  window-K       at token t, the top-C pairs counted over tokens t-K..t-1 (recent usage);
                 reports the experts that enter the set per token (PCIe loads)
  lru            a least-recently-used cache of C pairs; misses are loads
  --cross        held-out prompts: for each log, the top-C pairs counted on all *other* logs
                 (leave-one-prompt-out) and on all logs of *other categories* (leave-one-
                 category-out; categories from the study's prompt table, by file name)
"""
import argparse
import sys
from collections import Counter, OrderedDict

import numpy as np

EXPERT_MB = 3 * 2304 * 5120 / 32 * 17 / 1e6  # gate + up + down in mxfp4: 18.80 MB
DENSE_GB = 9.0                               # die 0: fp8 6.37 GiB + small 1.75 GiB + scratch
BUDGETS_GB = [64, 128, 256]                  # total HBM (128 = this machine)
STEP_GPU_MS, CPU_EXPERT_MS, GPU_EXPERT_MS = 35.0, 0.27, 0.065  # measured, sclk 2 (README 8, 12)


def read_log(path):
    with open(path, "rb") as f:
        if f.read(4) != b"DSRT":
            sys.exit(f"{path}: not a routing log")
        version, n_layer, k = np.frombuffer(f.read(12), dtype="<i4")
        if version != 1:
            sys.exit(f"{path}: version {version}")
        rec = np.dtype([("pos", "<i4"), ("token", "<i4"), ("prompt", "<i4"),
                        ("ids", "<i2", (n_layer, k)), ("w", "<f4", (n_layer, k))])
        data = np.frombuffer(f.read(), dtype=rec)
    return data, int(n_layer), int(k)


def capacity(budget_gb):
    return int((budget_gb - DENSE_GB) * 1e3 / EXPERT_MB)


def pairs(ids, n_expert=384):
    """[T, L, k] expert ids -> [T, L*k] (layer, expert) pair keys L * n_expert + e"""
    T, L, k = ids.shape
    return (np.arange(L)[None, :, None] * n_expert + ids).reshape(T, L * k)


def hit_rate(keys, resident):
    """fraction of slots whose pair is in `resident` (a boolean array over all pair keys)"""
    return float(resident[keys].mean())


def top_set(keys, C, n_pairs):
    counts = np.bincount(keys.ravel(), minlength=n_pairs)
    res = np.zeros(n_pairs, bool)
    res[np.argsort(-counts, kind="stable")[:C]] = True
    return res


def window_policy(keys, C, K, n_pairs):
    """top-C over the previous K tokens; hit rate and pairs entering the set per token"""
    T = len(keys)
    counts = np.zeros(n_pairs, np.int64)
    res = np.zeros(n_pairs, bool)
    hits = loads = 0
    for t in range(T):
        if t > 0:  # resident set from tokens t-K..t-1 (recomputed every token)
            new = np.zeros(n_pairs, bool)
            nz = np.nonzero(counts)[0]
            new[nz[np.argsort(-counts[nz], kind="stable")[:C]]] = True
            loads += int((new & ~res).sum())
            res = new
        hits += int(res[keys[t]].sum())
        np.add.at(counts, keys[t], 1)
        if t >= K:
            np.add.at(counts, keys[t - K], -1)
    return hits / keys.size, loads / max(T - 1, 1)


def lru_policy(keys, C):
    cache = OrderedDict()
    hits = misses = 0
    for row in keys:
        for key in row:
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                misses += 1
                cache[key] = True
                if len(cache) > C:
                    cache.popitem(last=False)
    return hits / keys.size, misses / len(keys)


def lru_warm(keys, C, prior_counts):
    """LRU of C pairs, starting full with the top-C pairs of prior_counts (most used = most
    recent); every miss loads the pair and evicts the least recently used one"""
    order = np.argsort(prior_counts, kind="stable")[-C:]  # ascending: least used first = LRU end
    cache = OrderedDict((int(x), True) for x in order)
    hits = misses = 0
    for row in keys:
        for key in row:
            key = int(key)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                misses += 1
                cache[key] = True
                cache.popitem(last=False)
    return hits / keys.size, misses / len(keys)


def est_tps(hit, L=40, k=6):
    """decode t/s if hits run on GPUs (spread over 4 dies) concurrently with misses on the CPU"""
    per_layer = max((1 - hit) * k * CPU_EXPERT_MS, hit * k * GPU_EXPERT_MS / 4)
    return 1000.0 / (STEP_GPU_MS + L * per_layer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--prompt", action="store_true", help="include prompt tokens (default: generated only)")
    ap.add_argument("--train", default="", help="comma-separated logs to pick static-train sets from")
    ap.add_argument("--windows", default="16,64,256,1024")
    ap.add_argument("--cross", default="", help="prompt table (name, category, ...): held-out tests")
    ap.add_argument("--no-temporal", action="store_true", help="skip the (slow) window / LRU policies")
    a = ap.parse_args()

    def load(paths):
        out = []
        for p in paths:
            d, nl, k = read_log(p)
            if not a.prompt:
                d = d[d["prompt"] == 0]
            out.append((p, d, nl, k))
        return out

    logs = load(a.logs)
    train = load(a.train.split(",")) if a.train else []
    nl, k = logs[0][2], logs[0][3]
    n_pairs = nl * 384
    caps = {b: capacity(b) for b in BUDGETS_GB}
    print(f"# expert {EXPERT_MB:.2f} MB; capacity after {DENSE_GB:.0f} GB dense: " +
          ", ".join(f"{b} GB -> {c} experts ({c / n_pairs:.0%})" for b, c in caps.items()))
    print(f"# hits/misses count chosen experts (240 per token); est t/s: 35 ms GPU + per layer "
          f"max(CPU misses x {CPU_EXPERT_MS} ms, GPU hits x {GPU_EXPERT_MS} ms / 4 dies)")

    all_keys = np.concatenate([pairs(d["ids"]) for _, d, _, _ in logs])
    tokens = len(all_keys)
    print(f"# {len(logs)} log(s), {tokens} tokens ({'prompt + generated' if a.prompt else 'generated'})")
    if tokens == 0:
        return

    # 1. concentration: how many experts per layer carry a given share of the routing
    print("\n## concentration (all tokens pooled)")
    shares = [0.5, 0.8, 0.9, 0.99]
    per_layer = []
    for L in range(nl):
        c = np.sort(np.bincount(all_keys[:, L * k:(L + 1) * k].ravel() - L * 384, minlength=384))[::-1]
        cum = np.cumsum(c) / c.sum()
        per_layer.append([int(np.searchsorted(cum, s) + 1) for s in shares] + [int((c > 0).sum())])
    pl = np.array(per_layer)
    print("experts per layer covering " + ", ".join(f"{s:.0%}" for s in shares) + ", used at all: "
          f"median {np.median(pl, 0).astype(int).tolist()}, min {pl.min(0).tolist()}, max {pl.max(0).tolist()}")
    print("(uniform routing would need " + ", ".join(f"{int(s * 384)}" for s in shares) + " of 384)")

    # 2. static sets: oracle (in-sample) and train -> test
    print("\n## static resident set: hit rate (est t/s)")
    print(f"{'policy':24s}" + "".join(f"{f'{b} GB':>18s}" for b in BUDGETS_GB))
    row = [hit_rate(all_keys, top_set(all_keys, caps[b], n_pairs)) for b in BUDGETS_GB]
    print(f"{'static-oracle':24s}" + "".join(f"{h:>10.1%} ({est_tps(h):4.1f})" for h in row))
    if train:
        tr = np.concatenate([pairs(d["ids"]) for _, d, _, _ in train])
        row = [hit_rate(all_keys, top_set(tr, caps[b], n_pairs)) for b in BUDGETS_GB]
        print(f"{'static-train (--train)':24s}" + "".join(f"{h:>10.1%} ({est_tps(h):4.1f})" for h in row))
    else:
        hs = []
        for b in BUDGETS_GB:
            num = den = 0
            for _, d, _, _ in logs:
                kk = pairs(d["ids"])
                h = len(kk) // 2
                if h == 0:
                    continue
                num += hit_rate(kk[h:], top_set(kk[:h], caps[b], n_pairs)) * (len(kk) - h)
                den += len(kk) - h
            hs.append(num / den if den else float("nan"))
        print(f"{'static 1st half -> 2nd':24s}" + "".join(f"{h:>10.1%} ({est_tps(h):4.1f})" for h in hs))
    uni = [min(1.0, caps[b] / n_pairs) for b in BUDGETS_GB]
    print(f"{'uniform (no stats)':24s}" + "".join(f"{h:>10.1%} ({est_tps(h):4.1f})" for h in uni))

    # 2b. held-out prompts: is a hot set learned on other prompts / categories good for a new one?
    if a.cross:
        import os
        cat = {}
        for line in open(a.cross, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            cat[f[0]] = f[1]
        names = [os.path.basename(p).rsplit(".", 1)[0] for p, _, _, _ in logs]
        kks = [pairs(d["ids"]) for _, d, _, _ in logs]
        counts = [np.bincount(kk.ravel(), minlength=n_pairs) for kk in kks]
        total = np.sum(counts, axis=0)
        def top_from(cnt, C):
            res = np.zeros(n_pairs, bool)
            res[np.argsort(-cnt, kind="stable")[:C]] = True
            return res
        print("\n## held-out prompts: hit rate per prompt (128 GB), and pooled at each budget")
        print(f"{'prompt':18s}{'category':11s}{'tokens':>7s}{'own (bound)':>13s}{'other prompts':>15s}{'other categories':>18s}")
        pooled = {b: [0.0, 0.0, 0.0] for b in BUDGETS_GB}
        n_all = 0
        for i, (nm, kk) in enumerate(zip(names, kks)):
            c = cat.get(nm, "?")
            others = total - counts[i]
            other_cat = np.sum([counts[j] for j in range(len(kks)) if cat.get(names[j], "?") != c], axis=0)
            row = {}
            for b in BUDGETS_GB:
                C = caps[b]
                hs = (hit_rate(kk, top_from(counts[i], C)), hit_rate(kk, top_from(others, C)),
                      hit_rate(kk, top_from(other_cat, C)) if np.any(other_cat) else float("nan"))
                for j in range(3):
                    pooled[b][j] += hs[j] * len(kk)
                row[b] = hs
            n_all += len(kk)
            h = row[128]
            print(f"{nm:18s}{c:11s}{len(kk):7d}{h[0]:>13.1%}{h[1]:>15.1%}{h[2]:>18.1%}")
        print(f"{'pooled':24s}" + "".join(f"{f'{b} GB':>26s}" for b in BUDGETS_GB))
        for j, lab in enumerate(["own prompt (bound)", "other prompts", "other categories"]):
            print(f"{lab:24s}" + "".join(f"{pooled[b][j] / n_all:>16.1%} ({est_tps(pooled[b][j] / n_all):4.1f} t/s)"
                                         for b in BUDGETS_GB))
        # adapting during the prompt: start from the other prompts' hot set, then LRU
        print("\n## adaptive: LRU warm-started with the other prompts' hot set; hit rate / experts loaded per token")
        print(f"{'start set from':24s}" + "".join(f"{f'{b} GB':>26s}" for b in BUDGETS_GB))
        for lab, which in [("other prompts", "p"), ("other categories", "c")]:
            cells = []
            for b in BUDGETS_GB:
                hit = load = 0.0
                for i, (nm, kk) in enumerate(zip(names, kks)):
                    c = cat.get(nm, "?")
                    prior = (total - counts[i]) if which == "p" else np.sum(
                        [counts[j] for j in range(len(kks)) if cat.get(names[j], "?") != c], axis=0)
                    h, l = lru_warm(kk, caps[b], prior)
                    hit += h * len(kk)
                    load += l * len(kk)
                cells.append(f"{hit / n_all:>10.1%} ({est_tps(hit / n_all):4.1f}) /{load / n_all:5.1f}")
            print(f"{lab:24s}" + "".join(f"{x:>26s}" for x in cells))

    # 3. temporal: recent-usage sets and LRU, per file (time order matters), pooled
    if a.no_temporal:
        return
    print("\n## temporal: hit rate / experts loaded per token (PCIe at ~1 GB per step is ~50 experts)")
    print(f"{'policy':24s}" + "".join(f"{f'{b} GB':>18s}" for b in BUDGETS_GB))
    for K in [int(x) for x in a.windows.split(",")]:
        cells = []
        for b in BUDGETS_GB:
            hit = load_n = n = 0
            for _, d, _, _ in logs:
                kk = pairs(d["ids"])
                h, l = window_policy(kk, caps[b], K, n_pairs)
                hit += h * len(kk)
                load_n += l * len(kk)
                n += len(kk)
            cells.append(f"{hit / n:>10.1%} /{load_n / n:5.1f}")
        print(f"{f'window-{K}':24s}" + "".join(f"{c:>18s}" for c in cells))
    cells = []
    for b in BUDGETS_GB:
        hit = miss = n = 0
        for _, d, _, _ in logs:
            kk = pairs(d["ids"])
            h, m = lru_policy(kk, caps[b])
            hit += h * len(kk)
            miss += m * len(kk)
            n += len(kk)
        cells.append(f"{hit / n:>10.1%} /{miss / n:5.1f}")
    print(f"{'lru':24s}" + "".join(f"{c:>18s}" for c in cells))

    # 4. drift inside a generation: overlap of the hot sets of the first and last quarter
    print("\n## drift: per file, hit rate of the first quarter's top-C on the last quarter (128 GB)")
    for p, d, _, _ in logs:
        kk = pairs(d["ids"])
        q = len(kk) // 4
        if q < 8:
            continue
        C = caps[128]
        print(f"{p}: {len(kk)} tokens; first->last {hit_rate(kk[-q:], top_set(kk[:q], C, n_pairs)):.1%}, "
              f"last->last (oracle) {hit_rate(kk[-q:], top_set(kk[-q:], C, n_pairs)):.1%}")


if __name__ == "__main__":
    main()
