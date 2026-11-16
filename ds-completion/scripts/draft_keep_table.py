#!/usr/bin/env python3
"""The table of scripts/draft-keep-sweep.sh's summary (README step 124): per configuration the
draft experts kept (mean over the prompts), the drafts accepted per verify step for each prompt,
and over all prompts the drafts accepted, the verify steps and the tokens per step.

  scripts/draft_keep_table.py [results/draft-pool/keep/summary.txt]
"""
import collections
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/draft-pool/keep/summary.txt'
rows = collections.OrderedDict()
for line in open(path):
    m = re.match(r'(\S+) (\S+): ([\d ]+) \| (\d+) (\d+) [\d.]+ \| [\d.]+ \| (\w+)(.*)', line)
    if m:
        c, n, per, steps, acc, same, rest = m.groups()
        rows.setdefault(c, {})[n] = (int(steps), int(acc), same == 'same', sum(map(int, per.split())), 'read' in rest)
names = list(rows['all'])
print('config     kept  ' + ' '.join(f'{n[:10]:>10}' for n in names) + '   accepted  steps  tokens/step  steps vs all  outputs  cold runs')
base = sum(rows['all'][n][0] for n in names)
for c, r in rows.items():
    acc = sum(r[n][1] for n in names)
    st = sum(r[n][0] for n in names)
    print(f'{c:9s} {sum(r[n][3] for n in names) / len(names):5.0f}  ' + ' '.join(f'{r[n][1] / r[n][0]:10.3f}' for n in names)
          + f'   {acc:8d}  {st:5d}  {(acc + st) / st:11.3f}  {100 * (st / base - 1):+11.1f}%  {"same" if all(r[n][2] for n in names) else "DIFFER":>7}  {sum(r[n][4] for n in names):9d}')
