#!/usr/bin/env python3
"""The table of scripts/draft-leave-out.sh's summary (README step 125): per prompt, the mean over
the repeats of each configuration.

  scripts/draft_leave_out_table.py [results/draft-pool/leave-out/summary.txt]
"""
import collections
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/draft-pool/leave-out/summary.txt'
runs = collections.defaultdict(list)
names = []
for line in open(path):
    m = re.match(r'(\S+) (\S+) \d+: (\d+) [\d ]+\| (\d+) (\d+) \| ([\d.]+)% \| (\d+) (\d+) \| ([\d.]+) \| (\w+)', line)
    if m:
        c, n = m.group(1), m.group(2)
        if n not in names:
            names.append(n)
        runs[(c, n)].append([float(x) for x in m.groups()[2:9]] + [m.group(10) == 'same'])
mean = lambda c, n, i: sum(r[i] for r in runs[(c, n)]) / len(runs[(c, n)])
print('prompt        die 0 experts   GPU share %          verify steps         verify ms per step   t/s                   t/s out vs all')
print('              all   out       all    out    res    all    out    res    all    out    res    all    out    res')
tot = []
for n in names:
    f = lambda i, fmt: ' '.join(format(mean(c, n, i), fmt) for c in ('all', 'out', 'resident'))
    vs = ' '.join(f'{mean(c, n, 5) / mean(c, n, 1):6.1f}' for c in ('all', 'out', 'resident'))
    d = 100 * (mean('out', n, 6) / mean('all', n, 6) - 1)
    tot.append(d)
    ok = all(r[7] for c in ('all', 'out', 'resident') for r in runs[(c, n)])
    print(f'{n:13s} {mean("all", n, 0):3.0f}   {mean("out", n, 0):3.0f}       {f(3, "6.2f")}   {f(1, "6.1f")}   {vs}   {f(6, "6.2f")}   {d:+5.1f}%' + ('' if ok else '  AN OUTPUT DIFFERS'))
print(f'mean of the prompts: t/s out vs all {sum(tot) / len(tot):+.1f}%  (all, out: {len(runs[("all", names[0])])} runs each; res = resident: {len(runs[("resident", names[0])])})')
