import csv, collections, bisect, sys
pre = sys.argv[1]; gen_s = 3.9
K = [r for r in csv.DictReader(open(pre + '_kernel_trace.csv'))]
C = [r for r in csv.DictReader(open(pre + '_memory_copy_trace.csv'))]
for r in K: r['s'] = int(r['Start_Timestamp']); r['e'] = int(r['End_Timestamp']); r['a'] = int(r['Agent_Id']); r['q'] = int(r['Queue_Id'])
for r in C: r['s'] = int(r['Start_Timestamp']); r['e'] = int(r['End_Timestamp']); r['d'] = int(r['Destination_Agent_Id'])
main = sorted((r for r in K if r['a'] == 1 and r['q'] == 1), key=lambda r: r['s'])
t1 = main[-1]['e']; t0 = t1 - int(gen_s * 1e9); main = [r for r in main if r['s'] >= t0]
big = [r for r in C if r['Direction'].endswith('HOST_TO_DEVICE') and r['e'] - r['s'] >= 250e3 and r['e'] > t0]
# die 0's loader queue: the queue of agent 1 that runs only unpack kernels inside the window (none with the unpack skipped)
qs = collections.Counter(r['q'] for r in K if r['a'] == 1 and r['s'] >= t0 and r['Kernel_Name'].startswith('unpack_blocks'))
unp = [(r['s'], r['e']) for r in K if r['a'] == 1 and r['s'] >= t0 and r['Kernel_Name'].startswith('unpack_blocks')]
def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
U = {d: union([(r['s'], r['e']) for r in big if r['d'] == d]) for d in (1, 2, 3, 4)}
U['u'] = union(unp); U['after'] = union([(e, e + 200000) for s, e in unp])
S = {d: [x[0] for x in U[d]] for d in U}
def inside(d, t):
    i = bisect.bisect_right(S[d], t) - 1
    return i >= 0 and U[d][i][1] > t
share = lambda d: 100 * sum(min(b, t1) - max(a, t0) for a, b in U[d] if b > t0 and a < t1) / (t1 - t0)
n0 = sum(1 for r in big if r['d'] == 1); d0 = sorted(r['e'] - r['s'] for r in big if r['d'] == 1)
print('# %s: die 0 main stream %d kernels in 3.9 s; load copies to die 0: %d (median %.0f us), in flight %.0f%% of the time (die 1: %.0f%%); unpack kernels on die 0: %d'
      % (pre, len(main), n0, d0[len(d0) // 2] / 1e3 if d0 else 0, share(1), share(2), len(unp)))
def cls(t):
    if inside(1, t): return 'a copy to die 0'
    if inside(2, t): return 'b to die 1 only'
    return 'c none on this card'
st = collections.defaultdict(lambda: [0, 0, []]); byname = collections.defaultdict(lambda: collections.defaultdict(list))
st3 = collections.defaultdict(lambda: [0, 0])
for i, r in enumerate(main[:-1]):
    gap = main[i + 1]['s'] - r['e']; dur = r['e'] - r['s']
    if gap > 100e3: continue
    c = cls(r['s']); x = st[c]; x[0] += 1; x[1] += dur + max(gap, 0); x[2].append(dur)
    byname[r['Kernel_Name'].split('(')[0][-28:]][c].append(dur)
    if c[0] == 'a':
        k = 'unpack running' if inside('u', r['s']) else 'within 0.2 ms after an unpack' if inside('after', r['s']) else 'copy only, no unpack near'
        st3[k][0] += 1; st3[k][1] += dur + max(gap, 0)
for c in sorted(st):
    n, tot, d = st[c]; d.sort(); print('   %-22s %6d kernels, median run %5.1f us, mean per kernel %5.1f us' % (c[2:], n, d[n // 2] / 1e3, tot / n / 1e3))
for k, (n, tot) in sorted(st3.items()): print('      of the first row, %-30s %6d kernels, mean per kernel %5.1f us' % (k, n, tot / max(n, 1) / 1e3))
sel = ['quant_act_kernel', 'rope_kernel', 'ial_m_kernel<float, 4, 6, 1>', 'void fq4_kernel<16, true>', 'kv_finish_kernel', 'void router_kernel<6, 6>', 'hc_split_kernel', 'copy16_kernel', 'attn_partial_kernel']
print('   median run, us [copy to die 0 | die 1 only | none]: ' + '; '.join('%s %s' % (n.split('<')[0].replace('void ', '').replace('_kernel', '')[-14:], '/'.join('%.0f' % (sorted(byname[n][c])[len(byname[n][c]) // 2] / 1e3) if byname[n][c] else '-' for c in sorted(st))) for n in sel))
