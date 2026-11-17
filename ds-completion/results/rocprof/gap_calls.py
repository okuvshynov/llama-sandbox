#!/usr/bin/env python3
"""What the host was calling during the home die's idle gaps between two given kernels, from a
rocprofv3 --runtime-trace run (docs/rocprof.md; README step 145).

  results/rocprof/gap_calls.py DIR/NAME AFTER_KERNEL BEFORE_KERNEL
  e.g. results/rocprof/gap_calls.py prof/r canon_down_kernel quant_act_kernel

Takes die 1's gaps of 100 us and more that follow AFTER_KERNEL and end at BEFORE_KERNEL, within
the generation (first to last output-head launch; a plain run), and sums the HIP calls of
NAME_hip_api_trace.csv that overlap them, by function and by thread.
"""
import bisect, collections, csv, re, sys

base, after, before = sys.argv[1:4]
rows = []
for r in csv.DictReader(open(base + '_kernel_trace.csv')):
    rows.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Agent_Id'], re.sub(r'\(.*', '', r['Kernel_Name']).replace('void ', '')))
heads = sorted(s for s, e, a, k in rows if k.startswith('gemv_partial_kernel<unsigned short'))
t0, t1 = heads[0], heads[-1]
ntok = len(heads) - 1
iv = sorted(r for r in rows if r[2] == '1' and t0 <= r[0] < t1)
gaps = []
ce, last = iv[0][1], iv[0][3]
for s, e, a, k in iv[1:]:
    if s > ce and s - ce >= 100e3 and last == after and k == before:
        gaps.append((ce, s))
    if e >= ce:
        ce, last = e, k
total = sum(b - a for a, b in gaps)
print(f'{after} -> {before}: {len(gaps) / ntok:.1f} gaps per token, mean {total / len(gaps) / 1e3:.0f} us, {total / 1e6 / ntok:.2f} ms per token')
api = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Function'], r['Thread_Id']) for r in csv.DictReader(open(base + '_hip_api_trace.csv')))
starts = [a[0] for a in api]
tot, cnt, thr = collections.Counter(), collections.Counter(), collections.Counter()
for a, b in gaps:
    i = bisect.bisect_left(starts, a - 2_000_000)
    while i < len(api) and api[i][0] < b:
        s, e, f, t = api[i]
        i += 1
        o = min(e, b) - max(s, a)
        if o > 0:
            tot[f] += o; cnt[f] += 1; thr[t] += o
print(f'HIP calls in progress cover {100 * sum(tot.values()) / total:.0f}% of the gaps\' time; threads: {len(thr)}')
for f, v in tot.most_common(8):
    print(f'  {f:24s} {cnt[f] / len(gaps):5.1f} calls per gap, {v / len(gaps) / 1e3:6.1f} us per gap')
