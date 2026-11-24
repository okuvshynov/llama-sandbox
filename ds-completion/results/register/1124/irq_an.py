import csv, re, bisect, collections
ev = []  # (t_ns, die, client, src, ring)
cur = {}
for l in open('gen_events.txt'):
    m = re.match(r'\s*\[(\d+)\]\s+([\d.]+):\s+(\S+):\s+(.*)', l)
    if not m: continue
    cpu, t, name, rest = m.group(1), int(round(float(m.group(2)) * 1e9)), m.group(3), m.group(4)
    if name == 'irq:irq_handler_entry':
        q = re.search(r'irq=(\d+)', rest); cur[cpu] = int(q.group(1))
    elif name == 'amdgpu:amdgpu_iv':
        q = re.search(r'client_id:(\d+) src_id:(\d+) ring:(\d+)', rest); irq = cur.get(cpu, 0)
        if 291 <= irq <= 294: ev.append((t, irq - 291, int(q.group(1)), int(q.group(2)), int(q.group(3))))
ev.sort()
print('interrupt sources per die in the 3.9 s (client 20 = compute end-of-pipe, 8/9 = the two copy engines; src 224 = their trap):')
c = collections.Counter((d, cl, s, r) for t, d, cl, s, r in ev)
for k, v in sorted(c.items()): print('   die %d client %2d src %3d ring %2d: %6d' % (k + (v,)))
K = [r for r in csv.DictReader(open('prof2/home_kernel_trace.csv')) if r['Agent_Id'] == '1' and r['Queue_Id'] == '1']
C = [r for r in csv.DictReader(open('prof2/home_memory_copy_trace.csv'))]
main = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'].split('(')[0][-24:]) for r in K); t1 = main[-1][1]; t0 = t1 - 3.9e9; main = [m for m in main if m[0] >= t0]
load = sorted((int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in C if r['Direction'].endswith('HOST_TO_DEVICE') and r['Destination_Agent_Id'] == '1' and int(r['End_Timestamp']) - int(r['Start_Timestamp']) >= 250e3 and int(r['End_Timestamp']) > t0)
LS = [x[0] for x in load]
def inflight(t):
    i = bisect.bisect_right(LS, t) - 1
    return i >= 0 and load[i][1] > t
# clock check: the profiler's kernel ends against die 0's end-of-pipe interrupts
eop0 = [t for t, d, cl, s, r in ev if d == 0 and cl == 20]; sd0 = [(t, cl, s, r) for t, d, cl, s, r in ev if d == 0 and cl in (8, 9)]
def nearest_after(arr, t):
    i = bisect.bisect_left(arr, t); return arr[i] - t if i < len(arr) else None
d = sorted(x for x in (nearest_after(eop0, e) for s, e, n in main[::50]) if x is not None)
print('clock check: from a kernel\'s end (profiler) to die 0\'s next end-of-pipe interrupt, us: p10 %.0f p50 %.0f p90 %.0f' % (d[len(d)//10] / 1e3, d[len(d)//2] / 1e3, d[len(d)*9//10] / 1e3))
# die 0's interrupts while a load copy to die 0 is in flight, and not
tin = sum(min(b, t1) - max(a, t0) for a, b in load) / 1e9
for label, sel in (('a load copy to die 0 in flight (%.2f s)' % tin, True), ('not (%.2f s)' % (3.9 - tin), False)):
    T = tin if sel else 3.9 - tin
    cc = collections.Counter((cl, s, r) for t, d, cl, s, r in ev if d == 0 and inflight(t) == sel)
    print(label + ': die 0 interrupts per ms: ' + ', '.join('client %d src %d ring %d: %.2f' % (k + (v / T / 1e3,)) for k, v in sorted(cc.items())))
# cadence of each die-0 interrupt source during a load copy
for key in sorted(set((cl, s, r) for t, d, cl, s, r in ev if d == 0)):
    ts = [t for t, d, cl, s, r in ev if d == 0 and (cl, s, r) == key and inflight(t)]
    gaps = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] - ts[i] < 400e3)
    if len(gaps) > 200:
        h = collections.Counter(min(int(g // 10e3) * 10, 200) for g in gaps)
        print('   client %d src %d ring %d: %d intervals under 0.4 ms during load copies; median %.0f us; share in 80-100 us: %.0f%%; by 10 us: %s' % (key + (len(gaps), gaps[len(gaps) // 2] / 1e3, 100 * (h[80] + h[90]) / len(gaps), ' '.join('%d:%d' % (k, h[k]) for k in sorted(h)))))
# stalled kernels: is there an interrupt of some source just before their end?
stalled = [(s, e) for s, e, n in main if inflight(s) and 60e3 <= e - s < 100e3]; fast = [(s, e) for s, e, n in main if inflight(s) and e - s < 20e3]
print('%d stalled kernels (60-100 us, copy in flight), %d fast ones (< 20 us)' % (len(stalled), len(fast)))
for key in sorted(set((cl, s, r) for t, d, cl, s, r in ev if d == 0)):
    ts = [t for t, d, cl, s, r in ev if d == 0 and (cl, s, r) == key]
    if len(ts) < 500: continue
    def inside(s, e):
        i = bisect.bisect_left(ts, s); return i < len(ts) and ts[i] < e
    print('   client %d src %d ring %d: an interrupt of this source falls inside the kernel\'s interval for %.0f%% of the stalled, %.0f%% of the fast' % (key + (100 * sum(inside(s, e) for s, e in stalled) / len(stalled), 100 * sum(inside(s, e) for s, e in fast) / max(1, len(fast)))))
