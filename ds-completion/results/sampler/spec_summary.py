#!/usr/bin/env python3
"""README step 150: the speculative loop, greedy against sampled.
  results/sampler/spec_summary.py
Reads the records of results/sampler/spec/ (and the plain loop's, results/sampler/plain/), of
results/sampler/spec-heldout/, the spec logs in results/sampler/spec/ and the stored outputs."""
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

ref = {}
for l in open('results/record/reference/heldout-output.tsv'):
    r = l.rstrip('\n').split('\t')
    if len(r) >= 5:
        ref[r[0]] = r[4]

print('1. outputs: the speculative loop against the plain loop with the same settings (and greedy against the stored output)')
spec, plain = records('results/sampler/spec'), records('results/sampler/plain')
bad = 0
for (p, c), r in sorted(spec.items()):
    base = {'greedy': 'greedy', 's1': 's1', 's1-n5': 's1', 's2': 's2', 's1-p1': 's1-p1'}[c]
    want = plain[p, base]['output']['hash']
    same = r['output']['hash'] == want and (c != 'greedy' or want == ref[p])
    bad += not same
    print(f"  {p:11s} {c:7s} {r['output']['hash']}  {'= plain ' + base if same else 'DIFFERS from plain ' + base}"
          f"  {r['counters']['gen_tps']:6.2f} t/s  {r['counters']['gen_tokens'] / r['spec']['steps']:.2f} tokens per step")
print(f'  {len(spec) - bad} of {len(spec)} equal')

print('\n2. a verify step\'s time by drafts verified, median ms (count): greedy | seed 1 | seed 1 with --spec-n 5')
for p in ['cpp-matrix', 'geometry', 'tcp-udp']:
    for c in ['greedy', 's1', 's1-n5']:
        by = collections.defaultdict(list)
        for l in open(f'results/sampler/spec/{c}-{p}.log'):
            if l[0] != '#':
                f = l.split()
                by[int(f[0])].append(float(f[2]))
        print(f'  {p:11s} {c:7s} ' + '  '.join(f'{n}: {statistics.median(v):.1f} ({len(v)})' for n, v in sorted(by.items())))

print('\n3. all held-out prompts, fastest single-machine settings, one run each: greedy | temperature 1, top-p 0.95, seed 1')
print('  prompt         t/s  tokens/step  = stored |    t/s  tokens/step  tokens')
held = records('results/sampler/spec-heldout')
tot = {'greedy': [0, 0, 0], 's1': [0, 0, 0]}
prompts = sorted({p for p, _ in held})
for p in prompts:
    cells = []
    for c in ('greedy', 's1'):
        r = held[p, c]
        cells.append((r['counters']['gen_tps'], r['counters']['gen_tokens'] / r['spec']['steps'], r['counters']['gen_tokens']))
        tot[c][0] += cells[-1][0]
        tot[c][1] += r['counters']['gen_tokens']
        tot[c][2] += r['spec']['steps']
    print(f"  {p:12s} {cells[0][0]:6.2f} {cells[0][1]:10.2f}  {'yes' if held[p, 'greedy']['output']['hash'] == ref[p] else 'NO':>8s} | {cells[1][0]:6.2f} {cells[1][1]:10.2f} {cells[1][2]:7d}")
for c in tot:
    print(f'  {c:7s} mean {tot[c][0] / len(prompts):.2f} t/s over {len(prompts)} prompts; {tot[c][1] / tot[c][2]:.2f} tokens per step over all steps')
