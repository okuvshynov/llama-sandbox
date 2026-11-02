#!/usr/bin/env python3
"""Per-group GPU timeline of speculative verify steps, from ds-completion --spec --profile stderr
(README step 49): medians over the second half of the run (the cache warm), by the number of
tokens verified per step.

    python3 scripts/profile_verify.py run.err [more.err ...]
"""
import re
import statistics as st
import sys

for f in sys.argv[1:]:
    steps = []
    for line in open(f):
        m = re.match(r'# verify (\d+) tokens: ([\d.]+) ms \(cpu experts ([\d.]+); experts gpu (\d+) cpu (\d+)\); GPU timeline:(.*)', line.strip())
        if m:
            g = {k: float(v) for k, v in re.findall(r'(\S+) ([\d.]+)', m.group(6))}
            steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5)), g))
    half = steps[len(steps) // 2:]
    print(f"== {f}: {len(steps)} verify steps, the second half ({len(half)}), medians in ms")
    by = {}
    for s in half:
        by.setdefault(s[0], []).append(s)
    groups = sorted({k for s in half for k in s[5]})
    print(f"{'tokens':>6} {'n':>3} {'step':>6} {'cpu':>5} {'experts gpu/cpu':>15} " + " ".join(f"{g:>24}" for g in groups))
    for M in sorted(by):
        ss = by[M]
        med = st.median
        print(f"{M:6d} {len(ss):3d} {med([s[1] for s in ss]):6.1f} {med([s[2] for s in ss]):5.1f} "
              f"{med([s[3] for s in ss]):7.0f}/{med([s[4] for s in ss]):<7.0f} " + " ".join(f"{med([s[5].get(g, 0) for s in ss]):24.1f}" for g in groups))
