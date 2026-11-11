#!/usr/bin/env python3
"""A run's kernel launches by wave count (README step 102), from scripts/hip_launch_waves_shim.cpp.

Flags the launches the GPU dispatches slowly: FLOOR = 2017-2048 waves (~48 us whatever the kernel
does), MULT = another multiple of 2048 (a light kernel pays ~30 us there).

  python3 scripts/launch_waves.py LAUNCHLOG BINARY [--all]
"""
import collections
import subprocess
import sys

log, binary = sys.argv[1], sys.argv[2]
names = {}
for line in subprocess.run(['nm', '-C', binary], capture_output=True, text=True).stdout.splitlines():
    p = line.split(' ', 2)
    if len(p) == 3 and p[0]:
        names[int(p[0], 16)] = p[2].split('(')[0].replace('void ', '')
rows = []
for line in open(log):
    n, w, off = line.split()
    rows.append((int(n), int(w), names.get(int(off, 16), off)))
total = sum(r[0] for r in rows)
print(f"# {total} launches, {len(rows)} (kernel, waves) pairs")
flag = lambda w: 'FLOOR' if 2016 < w <= 2048 else 'MULT' if w % 2048 == 0 else ''
by = collections.Counter()
for n, w, k in rows:
    by[flag(w)] += n
print(f"# on the floor (2017-2048 waves): {by['FLOOR']}; other multiples of 2048: {by['MULT']}")
print(f"{'launches':>9s} {'waves':>7s}       kernel")
for n, w, k in sorted(rows, reverse=True):
    if flag(w) or '--all' in sys.argv:
        print(f"{n:9d} {w:7d} {flag(w):5s} {k}")
