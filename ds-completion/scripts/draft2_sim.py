#!/usr/bin/env python3
"""A second draft block before the verify, from a shadow run's logs (README step 107).

DS_DRAFT2_LOG (ds-completion) writes, per drafted step, the anchor's index in the generated
tokens, the number of drafts the rule verified, the block's 5 drafts, and two second blocks
drafted from the block's last token without a verify in between: "gap" (at its true positions,
the window of main-model states 5 positions behind) and "shift" (as if it followed the window
directly). Its last line holds every generated id, so each block is scored against what the
main model went on to emit: m1 = the first block's matching prefix, m2 = the second block's
(counted only when m1 = 5).

Estimate per prompt: on the steps where the rule verified all 5 drafts, also verify the first n2
drafts of the second block (n2 fixed, or the longest prefix whose confidence product stays above
a threshold). Tokens gained: min(m2, n2) when m1 = 5. Time added: one more draft call on each such
step and `slope` ms per extra verified token (the run's fitted verify cost). The new speed is
(tokens + gained) / (time + added) against tokens / time: a per-step account, not a replay (with
more tokens per step the later anchors would differ).

    python3 scripts/draft2_sim.py LOGDIR ERRDIR [--draft-ms 8.0]

ERRDIR: a held-out run's .err files of the same build (generation time, the verify cost fit).
"""
import math
import re
import sys

NAMES = ['tcp-udp', 'py-lru', 'cpp-matrix', 'sql-report', 'geometry', 'zh-travel', 'ja-email', 'summary-json']


def read(path):
    steps, gen = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('G'):
            gen = [int(x) for x in line.split()[1:]]
            continue
        parts = line.split('|')
        idx, nv = (int(x) for x in parts[0].split())
        blocks = []
        for p in parts[1:]:
            ids, conf = p.split(';')
            blocks.append(([int(x) for x in ids.split()], [float(x) for x in conf.split()]))
        steps.append((idx, nv, blocks))
    return steps, gen


def prefix(draft, actual):
    k = 0
    while k < len(draft) and k < len(actual) and draft[k] == actual[k]:
        k += 1
    return k


def main():
    logdir, errdir = sys.argv[1], sys.argv[2]
    draft_ms = float(sys.argv[sys.argv.index('--draft-ms') + 1]) if '--draft-ms' in sys.argv else 8.0
    rules = [('n2 = 1', 1, None), ('n2 = 2', 2, None), ('n2 = 3', 3, None), ('n2 = 5', 5, None), ('conf > 0.5', 5, 0.5),
             ('conf > 0.7', 5, 0.7), ('oracle n2', -1, None)]
    for variant, vi in (('gap', 1), ('shift', 2)):
        print(f"\n## second block: {variant}")
        print(f"{'prompt':13s} {'steps':>5s} {'n=5':>4s} {'m1=5':>5s}  m2 = 0 1 2 3 4 5 (of the n=5, m1=5 steps)   mean m2  "
              + '  '.join(f"{r[0]:>10s}" for r in rules))
        gains = [[] for _ in rules]
        for name in NAMES:
            steps, gen = read(f"{logdir}/{name}.log")
            err = open(f"{errdir}/{name}.err").read()
            m = re.search(r'generation: (\d+) tokens in ([0-9.]+) s', err)
            ntok, ms = int(m.group(1)), float(m.group(2)) * 1000
            slope = float(re.search(r'verify cost fit [0-9.]+ \+ ([0-9.]+) n ms', err).group(1))
            hist, n5, full = [0] * 6, 0, 0
            gained = [0.0] * len(rules)
            added = [0.0] * len(rules)
            for idx, nv, blocks in steps:
                b1, b2 = blocks[0][0], blocks[vi][0]
                conf2 = blocks[vi][1]
                if nv < len(b1):
                    continue
                n5 += 1
                m1 = prefix(b1, gen[idx + 1:])
                m2 = prefix(b2, gen[idx + 1 + len(b1):]) if m1 == len(b1) else 0
                if m1 == len(b1):
                    full += 1
                    hist[m2] += 1
                for ri, (_, n2, thr) in enumerate(rules):
                    if n2 < 0:  # oracle: verify exactly the drafts that will be accepted; no second draft when none
                        k = m2
                        if k == 0:
                            continue
                    elif thr is None:
                        k = n2
                    else:
                        k, run = 0, 1.0
                        while k < n2:
                            run *= 1 / (1 + math.exp(-conf2[k]))
                            if run < thr:
                                break
                            k += 1
                    gained[ri] += min(m2, k)
                    added[ri] += draft_ms + slope * k
            cells = []
            for ri in range(len(rules)):
                g = (ntok + gained[ri]) / (ms + added[ri]) / (ntok / ms) - 1
                gains[ri].append(g)
                cells.append(f"{g:+9.1%} ")
            mean_m2 = sum(i * h for i, h in enumerate(hist)) / max(1, full)
            print(f"{name:13s} {len(steps):5d} {n5:4d} {full:5d}  {' '.join(f'{h:3d}' for h in hist):42s} {mean_m2:6.2f}   " + '  '.join(cells))
        print(f"{'mean':13s} {'':60s} {'':15s}" + '  '.join(f"{sum(g) / len(g):+9.1%} " for g in gains))


if __name__ == '__main__':
    main()
