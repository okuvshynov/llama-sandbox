#!/usr/bin/env python3
"""Each die's busy share during generation, from a rocprofv3 kernel trace (docs/rocprof.md; README step 143).

  results/rocprof/busy.py DIR/NAME_kernel_trace.csv

The window is from the first to the last launch of the output head (one per token). Per die: launches,
the time a kernel is running (the union of the launches' intervals), the idle gaps between them by
length, and for the home die its kernels and between which kernels the gaps of 100 us and more fall.
The run was traced, so it is slower than a plain one: the gaps hold the profiler's host cost too.
"""
import csv, sys, re, collections
rows = []
for r in csv.DictReader(open(sys.argv[1])):
    rows.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Agent_Id'], r['Queue_Id'], re.sub(r'\(.*', '', r['Kernel_Name'])))
heads = sorted(s for s, e, a, q, k in rows if 'gemv_partial_kernel<unsigned short' in k)
t0, t1 = heads[0], heads[-1]
ntok = len(heads) - 1
win = (t1 - t0) / 1e6
print(f'window: output head launch 1 to {len(heads)}: {ntok} tokens, {win:.1f} ms, {win/ntok:.2f} ms per token ({1000*ntok/win:.1f} t/s)')
gen = [r for r in rows if t0 <= r[0] < t1]
print('die  launches/token  queues  sum of kernel time  busy (union)  busy share   idle gaps: count/token  median us  share of idle in gaps<20us')
for a in sorted({r[2] for r in gen}):
    iv = sorted((s, e) for s, e, aa, q, k in gen if aa == a)
    tot = sum(e - s for s, e in iv)
    busy = 0; gaps = []; cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs; gaps.append(s - ce); cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    gaps.sort(); idle = sum(gaps)
    small = sum(g for g in gaps if g < 20000)
    print(f'{a:>3} {len(iv)/ntok:15.0f} {len({r[3] for r in gen if r[2]==a}):7d} {tot/1e6/ntok:14.2f} ms/tok {busy/1e6/ntok:8.2f} ms/tok {100*busy/(t1-t0):9.1f}% {len(gaps)/ntok:22.0f} {gaps[len(gaps)//2]/1e3:10.1f} {100*small/idle:14.1f}%')
    if a == '1':
        home_gaps = gaps
        by = collections.defaultdict(lambda: [0, 0])
        for s, e, aa, q, k in gen:
            if aa == a: by[k][0] += 1; by[k][1] += e - s
        print('   home die, per token:  kernel                              launches   ms    ')
        for k, (n, d) in sorted(by.items(), key=lambda kv: -kv[1][1])[:10]:
            print(f'      {k[:40]:40s} {n/ntok:8.1f} {d/1e6/ntok:7.3f}')
g = home_gaps
for lo, hi in ((0, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e6), (1e6, 1e12)):
    x = [v for v in g if lo <= v < hi]
    print(f'   home die gaps {lo/1e3:6.0f}-{hi/1e3:<9.0f} us: {len(x)/ntok:7.1f} per token, {sum(x)/1e6/ntok:6.2f} ms per token')

iv = sorted(r for r in gen if r[2] == '1')
pairs = collections.defaultdict(lambda: [0, 0]); ce = iv[0][1]; last = iv[0][4]
for s_, e_, a, q, k in iv[1:]:
    if s_ > ce and s_ - ce >= 100e3:
        pairs[(last, k)][0] += 1; pairs[(last, k)][1] += s_ - ce
    if e_ >= ce: ce = e_; last = k
print('   home die, gaps of 100 us and more: after kernel -> before kernel: per token, ms per token, mean us')
for (a, b), (n, d) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:5]:
    print(f'      {a[:30]:30s} -> {b[:30]:30s} {n/ntok:6.1f} {d/1e6/ntok:7.2f} {d/n/1e3:8.0f}')
