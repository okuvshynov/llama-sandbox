#!/usr/bin/env python3
"""README step 151: the cheaper draw, step 150's runs again.
  results/sampler/step151/summary.py
Reads the records and spec logs of results/sampler/step151/ and step 150's
(results/sampler/spec/, results/sampler/spec-heldout/)."""
import collections
import glob
import json
import statistics


def records(d):
    out = {}
    for f in sorted(glob.glob(d + '/*.json')):
        r = json.load(open(f))
        out[r['label']['prompt'], r['label']['config']] = r
    return out


def steps(path):
    by = collections.defaultdict(list)
    for l in open(path):
        if l[0] != '#':
            f = l.split()
            by[int(f[0])].append(float(f[2]))
    return by


print('1. outputs: against step 150\'s with the same settings')
n = bad = 0
for d in ('spec', 'spec-heldout'):
    new, old = records('results/sampler/step151/' + d), records('results/sampler/' + d)
    for k, r in sorted(new.items()):
        same = r['output']['hash'] == old[k]['output']['hash']
        n += 1
        bad += not same
        if not same:
            print(f'  DIFFERS: {d} {k[0]} {k[1]}')
    print(f'  {d}: {len(new)} runs')
print(f'  {n - bad} of {n} equal')

print('\n2. a verify step\'s time by drafts verified, median ms (count); step 150 | step 151')
for p in ['cpp-matrix', 'geometry', 'tcp-udp']:
    for c in ['greedy', 's1', 's1-n5']:
        a, b = steps(f'results/sampler/spec/{c}-{p}.log'), steps(f'results/sampler/step151/spec/{c}-{p}.log')
        print(f'  {p:11s} {c:7s} ' + '  '.join(
            f'{k}: {statistics.median(a[k]):.1f} ({len(a[k])}) | {statistics.median(b[k]):.1f} ({len(b[k])})' for k in sorted(a) if k in b))
print('  five drafts, sampled minus greedy:')
for p in ['cpp-matrix', 'geometry']:  # tcp-udp's greedy run has one such step
    for name, d in (('step 150', 'results/sampler/spec'), ('step 151', 'results/sampler/step151/spec')):
        g = statistics.median(steps(f'{d}/greedy-{p}.log')[5])
        print(f'  {p:11s} {name}: ' + ', '.join(f'{c} {statistics.median(steps(f"{d}/{c}-{p}.log")[5]) - g:+.1f} ms' for c in ('s1', 's1-n5')))

print('\n3. all held-out prompts, one run each, t/s: step 150 greedy, sampled | step 151 greedy, sampled')
old, new = records('results/sampler/spec-heldout'), records('results/sampler/step151/spec-heldout')
prompts = sorted({p for p, _ in new})
col = collections.defaultdict(list)
for p in prompts:
    v = [d[p, c]['counters']['gen_tps'] for d in (old, new) for c in ('greedy', 's1')]
    for i, x in enumerate(v):
        col[i].append(x)
    print(f'  {p:12s} {v[0]:6.2f} {v[1]:6.2f} ({100 * (v[1] / v[0] - 1):+5.1f}%) | {v[2]:6.2f} {v[3]:6.2f} ({100 * (v[3] / v[2] - 1):+5.1f}%)')
m = [statistics.mean(col[i]) for i in range(4)]
print(f'  {"mean":12s} {m[0]:6.2f} {m[1]:6.2f} ({100 * (m[1] / m[0] - 1):+5.1f}%) | {m[2]:6.2f} {m[3]:6.2f} ({100 * (m[3] / m[2] - 1):+5.1f}%)')
print('  (the sampled text of a prompt is the same in both steps, and the greedy one; the sampled and the greedy texts differ)')
