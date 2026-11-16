#!/usr/bin/env python3
"""A DS_DRAFT_KEEP file (README step 124): the draft experts its router may choose, from
DS_DRAFT_ROUTE_LOG files (step 87's are results/decode-next/draft-route/<prompt>.log).

  scripts/draft_keep.py N LOG... > keep.txt

The N most chosen (layer, expert) pairs over the given logs, all three layers ranked together
(the layers differ: over step 87's prompts 94 / 62 / 41 experts cover 99% of the choices of
layers 0 / 1 / 2), ties by layer then expert. Every layer keeps at least --min of its own most
chosen (default 3: the router chooses 3). To test what a set chosen without the prompt gives,
leave that prompt's log out; with only its own log the set is the best one for it.
"""
import argparse
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int)
ap.add_argument('logs', nargs='+')
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--block', type=int, default=5)
ap.add_argument('--min', type=int, default=3)
a = ap.parse_args()

cnt = Counter()
for path in a.logs:
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        for i, part in enumerate(line.split('|')[1:]):
            for e in part.split():
                cnt[(i // a.block, int(e))] += 1
ranked = sorted(cnt, key=lambda p: (-cnt[p], p))
keep = []
for L in range(a.layers):  # each layer's own top --min first
    keep += [p for p in ranked if p[0] == L][:a.min]
for p in ranked:
    if len(keep) >= a.n:
        break
    if p not in keep:
        keep.append(p)
total = sum(cnt.values())
per = [sum(1 for p in keep if p[0] == L) for L in range(a.layers)]
print(f"# ds-completion draft keep v1: {len(keep)} of the pairs chosen in {len(a.logs)} logs ({len(cnt)} distinct, "
      f"{total} choices); per layer {per}; they hold {100 * sum(cnt[p] for p in keep) / total:.1f}% of those logs' choices")
for L, e in sorted(keep):
    print(L, e)
