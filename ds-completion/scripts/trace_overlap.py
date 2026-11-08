#!/usr/bin/env python3
"""Die 0 and the CPU's experts against each other per verify step, from --trace files (README step 94).

Per verify step of the chosen size (default 6 tokens), medians in ms: the step, die 0 busy (the
union of its groups, without its waits for the CPU's and the dies' experts), the CPU's experts
(the union of the host's "CPU experts" spans), both at once, the host's wait for die 0's lists
("wait for the staged block"), and the CPU calls per step. With --by-size: die 0 busy per verify
size instead (how its work grows with the tokens of a step).

  scripts/trace_overlap.py TRACE.json[.gz] ... [--size 6] [--by-size]
"""
import argparse
import bisect
import gzip
import json
import re
import statistics as st
from collections import defaultdict


def load(p):
    f = gzip.open(p, 'rt') if p.endswith('.gz') else open(p)
    return [e for e in json.load(f)['traceEvents'] if e.get('ph') == 'X']


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(iv):
    return sum(b - a for a, b in iv)


def inter(x, y):
    i = j = 0
    s = 0.0
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            s += b - a
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--size', type=int, default=6)
    ap.add_argument('--by-size', action='store_true')
    a = ap.parse_args()
    for p in a.traces:
        ev = load(p)
        g = sorted([e for e in ev if e['tid'] == 2 and re.fullmatch(r'\S+ L\d+', e['name']) and not e['name'].startswith('ffn.wait_for')],
                   key=lambda e: e['ts'])
        gts = [e['ts'] for e in g]
        host = sorted([e for e in ev if e['tid'] == 1], key=lambda e: e['ts'])
        hts = [e['ts'] for e in host]
        steps = [e for e in ev if e['tid'] == 3 and re.fullmatch(r'verify \d+ tokens', e['name'])]
        by_size = defaultdict(list)
        rows = defaultdict(list)
        for s in steps:
            n = int(s['name'].split()[1])
            a0, a1 = s['ts'], s['ts'] + s['dur']
            gs = g[bisect.bisect_left(gts, a0 - 1):bisect.bisect_right(gts, a1 + 1)]
            busy = union([(e['ts'], e['ts'] + e['dur']) for e in gs])
            by_size[n].append(total(busy) / 1000)
            if n != a.size:
                continue
            hs = host[bisect.bisect_left(hts, a0 - 1):bisect.bisect_right(hts, a1 + 1)]
            cpu = union([(e['ts'], e['ts'] + e['dur']) for e in hs if e['name'].startswith('CPU experts L')])
            stg = union([(e['ts'], e['ts'] + e['dur']) for e in hs if e['name'] == 'wait for the staged block'])
            rows['step'].append(s['dur'] / 1000)
            rows['die 0 busy'].append(total(busy) / 1000)
            rows['CPU experts'].append(total(cpu) / 1000)
            rows['both at once'].append(inter(busy, cpu) / 1000)
            rows['host waits for die 0 lists'].append(total(stg) / 1000)
            rows['CPU calls'].append(sum(1 for e in hs if e['name'].startswith('CPU experts L')))
        name = p.split('/')[-1]
        if a.by_size:
            print(f"# {name}: die 0 busy per verify size, median ms (steps): " +
                  ', '.join(f"{n}: {st.median(v):.1f} ({len(v)})" for n, v in sorted(by_size.items())))
            continue
        print(f"# {name}: {len(rows['step'])} verify steps of {a.size} tokens, medians in ms")
        for k, v in rows.items():
            print(f"  {k:28s} {st.median(v):8.2f}" if k != 'CPU calls' else f"  {k:28s} {st.median(v):8.0f}")


if __name__ == '__main__':
    main()
