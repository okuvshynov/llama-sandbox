#!/usr/bin/env python3
"""A perfect-knowledge estimate for the draft-length rule (README step 146).

  results/spec-oracle/oracle.py results/spec-oracle/runs

Per prompt, from two runs of the same settings with --spec-log and --record:
  n5-PROMPT        --spec-n 5: every step verifies the whole block, so its log has k, the number
                   of drafts accepted, for every step. A step advances k + 1 tokens.
  adaptive-PROMPT  the rule as it is: the measured speed, and verify times at every width.
An oracle that verifies exactly k drafts also advances k + 1 tokens, so it visits the n5 run's
positions with the same drafts: its steps are the n5 log's, at other costs.
  full oracle     k = 0: a plain step, no draft at all; k >= 1: the draft + a verify of k
  length only     the draft always; a verify of k
Costs, per prompt: verify(n) = the mean verify ms of the steps that verified n drafts, both logs
pooled; a width with under 5 steps takes the least-squares line through the pooled steps;
plain = the adaptive run's steps without a draft, if it has 5, else verify(0); draft = the mean
draft ms; rest = the generation time outside draft and verify, per step.
Check of the model: the n5 run's mean verify predicted by the line fitted to the ADAPTIVE log alone.
"""
import glob, json, os, sys

d = sys.argv[1]
names = [l.split('|')[0] for l in open('results/spec-heldout/prompts.txt') if l.strip()]


def log(path):
    return [[float(x) for x in l.split()[:4]] for l in open(path) if l.strip() and not l.startswith('#')]


def line(pts):
    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


print('prompt        k: 0    1    2    3    4    5 | n5 verify: fit from adaptive, measured | t/s: n5  adaptive | length only     | full oracle')
tot = {'n5': 0, 'ad': 0, 'len': 0, 'full': 0}
for p in names:
    ra, r5 = (json.load(open(f'{d}/{c}-{p}-1.json')) for c in ('adaptive', 'n5'))
    la, l5 = log(f'{d}/adaptive-{p}.log'), log(f'{d}/n5-{p}.log')
    a, b = line([(r[0], r[2]) for r in la])
    v5 = sum(r[2] for r in l5) / len(l5)
    check = (a + 5 * b) / v5 - 1
    pooled = la + l5
    a2, b2 = line([(r[0], r[2]) for r in pooled])
    cost = {}
    for n in range(6):
        v = [r[2] for r in pooled if r[0] == n]
        cost[n] = sum(v) / len(v) if len(v) >= 5 else a2 + b2 * n
    sa = ra['spec']
    plain_ms = sa['verify_ms'] - sum(r[2] for r in la)
    plain = plain_ms / sa['plain_steps'] if sa['plain_steps'] >= 5 else cost[0]
    draft = sum(r[3] for r in pooled) / len(pooled)
    rest = sum((r['counters']['gen_s'] * 1000 - r['spec']['verify_ms'] - r['spec']['draft_ms']) / r['spec']['steps'] for r in (ra, r5)) / 2
    ks = [int(r[1]) for r in l5]
    tokens = len(ks) + sum(ks)
    t_len = sum(draft + cost[k] + rest for k in ks)
    t_full = sum((plain if k == 0 else draft + cost[k]) + rest for k in ks)
    tps = {'n5': r5['counters']['gen_tps'], 'ad': ra['counters']['gen_tps'], 'len': 1000 * tokens / t_len, 'full': 1000 * tokens / t_full}
    for k in tot: tot[k] += tps[k] / len(names)
    hist = ' '.join('%4d' % ks.count(k) for k in range(6))
    print(f'{p:13s} {hist} | {a + 5 * b:7.1f} {v5:7.1f} ms ({100 * check:+.1f}%)        | {tps["n5"]:6.2f} {tps["ad"]:6.2f}   | '
          f'{tps["len"]:6.2f} ({100 * (tps["len"] / tps["ad"] - 1):+5.1f}%) | {tps["full"]:6.2f} ({100 * (tps["full"] / tps["ad"] - 1):+5.1f}%)')
    print(f'              costs ms: verify ' + ' '.join('%.1f' % cost[n] for n in range(6)) + f'; plain {plain:.1f}; draft {draft:.1f}; rest {rest:.2f} per step; tokens {tokens} of {r5["counters"]["gen_tokens"]}')
print(f'mean of the prompts' + ' ' * 63 + f'| {tot["n5"]:6.2f} {tot["ad"]:6.2f}   | {tot["len"]:6.2f} ({100 * (tot["len"] / tot["ad"] - 1):+5.1f}%) | {tot["full"]:6.2f} ({100 * (tot["full"] / tot["ad"] - 1):+5.1f}%)')
