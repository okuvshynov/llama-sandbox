#!/usr/bin/env python3
"""The draft's head over part of the vocabulary, simulated (README step 92).

The draft picks the argmax of its logits (the shared head plus its Markov bias). Over a subset S
of the vocabulary the argmax is the same token whenever the full argmax is in S, so a verify
step's accepted drafts stay accepted up to the first accepted one outside S, where the chain
breaks (the subset's pick there is assumed rejected: a slight underestimate). Outputs cannot
change (the main model verifies); only tokens per step and the draft's cost do.

Input: DS_DRAFT_TOKEN_LOG files of a held-out run (per verify step: n_verified, accepted,
verify_ms, draft_ms | drafted ids | emitted ids) and that run's .err files for its t/s.
Rankings of the vocabulary (S = the first K ids of a ranking, plus every id >= 128000: special
tokens such as the end of a turn; calib+ctx also every token the run has emitted so far):
  id        token id order (contiguous rows)
  calib     frequency among the generated tokens of the 30 routing-study prompts
            (data/routing/study/*.bin, README step 13; no held-out prompt), the rest by id
  oracle    frequency among the held-out run's own emitted tokens, then calib: an upper bound
            only (it has seen the answers)
The t/s estimate: each step's measured time minus the draft's saving (its head and Markov parts,
~2.2 + 1.0 ms per call at the full vocabulary, scaled by 1 - K / V), the rest of a step's time
(the time outside verify and draft, per step) unchanged, tokens per step from the simulation.

  ~/venvs/ds-ref/bin/python scripts/draft_vocab_sim.py LOGDIR ERRDIR [--sizes 1,2,4,...] (thousands)
"""
import argparse
import collections
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from route_stats import read_log  # noqa: E402

V, SPECIAL0 = 129280, 128000
HEAD_MS, MARKOV_MS = 2.2, 1.0  # the draft's parts per call (results/trace/critpath.txt, step 89)
HELDOUT8 = ['tcp-udp', 'py-lru', 'cpp-matrix', 'sql-report', 'geometry', 'zh-travel', 'ja-email', 'summary-json']


def read_steps(path):
    steps = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        head, drafted, emitted = line.split('|')
        n, k, vms, dms = head.split()
        steps.append((int(n), int(k), float(vms), float(dms), [int(x) for x in drafted.split()], [int(x) for x in emitted.split()]))
    return steps


def gen_time(err):
    m = re.search(r'generation: (\d+) tokens in ([0-9.]+) s', open(err).read())
    return int(m.group(1)), float(m.group(2))


def ranking(counts, then=None):
    """ids by count (ties by id), then the rest in `then`'s order (default: id order)"""
    seen = sorted(counts, key=lambda t: (-counts[t], t))
    rest = then if then is not None else range(V)
    s = set(seen)
    return np.array(seen + [t for t in rest if t not in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logdir')
    ap.add_argument('errdir')
    ap.add_argument('--sizes', default='1,2,4,8,16,24,32,48,64,96')
    a = ap.parse_args()
    sizes = [int(x) * 1000 for x in a.sizes.split(',')] + [V]
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.logdir, '*.log'))):
        name = os.path.basename(p)[:-4]
        runs[name] = (read_steps(p), gen_time(os.path.join(a.errdir, f'vocab-{name}.err')))
    calib = collections.Counter()
    for p in sorted(glob.glob('data/routing/study/*.bin')):
        d, _, _ = read_log(p)
        calib.update(d['token'][d['prompt'] == 0].tolist())
    held = collections.Counter()
    for steps, _ in runs.values():
        for s in steps:
            held.update(s[5])
    r_calib = ranking(calib)
    ranks = {'id': np.arange(V), 'calib': r_calib, 'calib+ctx': r_calib, 'oracle': ranking(held, then=r_calib.tolist())}
    names = [n for n in HELDOUT8 if n in runs]
    print(f"# {len(runs)} runs ({', '.join(runs)}); the 8-prompt mean over {', '.join(names)}")
    print(f"# calibration: {sum(calib.values())} generated tokens, {len(calib)} distinct; held-out emitted: "
          f"{sum(held.values())} tokens, {len(held)} distinct")
    # the emitted tokens' coverage: share of the held-out run's emitted tokens inside each S
    print("\n## share of the held-out run's emitted tokens inside S (8 prompts pooled)")
    print(f"{'K':>7s} " + ' '.join(f"{r:>8s}" for r in ranks))
    em = np.concatenate([np.array(sum((s[5] for s in runs[n][0]), [])) for n in names])
    for K in sizes:
        row = []
        for r in ranks.values():
            inS = np.zeros(V, bool)
            inS[r[:K]] = True
            inS[SPECIAL0:] = True
            row.append(inS[em].mean())
        print(f"{K:7d} " + ' '.join(f"{x:8.1%}" for x in row))
    rows = {}  # ranking -> per K: (mean accepted, tokens per step, mean t/s, per-prompt t/s)
    for rname, r in ranks.items():
        rows[rname] = []
        for K in sizes:
            inS = np.zeros(V, bool)
            inS[r[:K]] = True
            inS[SPECIAL0:] = True
            save = (HEAD_MS + MARKOV_MS) * (1 - K / V)
            acc, per = [], {}
            for n in names:
                steps, (ntok, secs) = runs[n]
                other = (secs * 1000 - sum(s[2] + s[3] for s in steps)) / len(steps)  # commit, emit, the host
                tok = t = 0.0
                S = inS.copy()  # calib+ctx: the run's emitted tokens join S as they come (its prompt's would too)
                for (nv, k, vms, dms, dr, emt) in steps:
                    k2 = 0
                    while k2 < k and S[dr[k2]]:
                        k2 += 1
                    if rname == 'calib+ctx':
                        S[emt] = True
                    tok += k2 + 1
                    t += vms + other + (max(0.0, dms - save) if dms > 0 else 0.0)
                    acc.append(k2)
                per[n] = tok / t * 1000
            rows[rname].append((float(np.mean(acc)), float(np.mean(acc)) + 1, float(np.mean(list(per.values()))), per))
    print("\n## per K and ranking: accepted drafts per step (all steps of the 8 prompts); est. t/s change against the full "
          "vocabulary, 8-prompt mean and the worst prompt (oracle: an upper bound only)")
    print(f"{'K':>7s} " + ' '.join(f"{r:>34s}" for r in ranks))
    for i, K in enumerate(sizes):
        cells = []
        for r in ranks:
            acc, _, mean, per = rows[r][i]
            fmean, fper = rows[r][-1][2], rows[r][-1][3]
            ch = {n: per[n] / fper[n] - 1 for n in per}
            w = min(ch, key=ch.get)
            cells.append(f"{acc:6.3f} {mean / fmean - 1:+7.2%} {ch[w]:+7.2%} {w:>10s}")
        print(f"{K:7d} " + ' '.join(f"{c:>34s}" for c in cells))
    print(f"# full vocabulary: est. {rows['id'][-1][2]:.2f} t/s from the logs' step times (the runs measured "
          f"{np.mean([runs[n][1][0] / runs[n][1][1] for n in names]):.2f})")


if __name__ == '__main__':
    main()
