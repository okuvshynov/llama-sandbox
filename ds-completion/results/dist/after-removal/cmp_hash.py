import glob, json, sys
new, old = sys.argv[1:3]
n = bad = 0; tps = {}
for f in sorted(glob.glob(new + '/*.json')):
    r = json.load(open(f)); k = (r['label']['prompt'], r['label']['config'])
    g = glob.glob(f"{old}/{k[1]}-{k[0]}-*.json")
    o = json.load(open(sorted(g)[0])) if g else None
    n += 1
    tps.setdefault(k[1], []).append((r['counters']['gen_tps'], o['counters']['gen_tps'] if o else 0))
    if not o or o['output']['hash'] != r['output']['hash']:
        bad += 1; print('  DIFFERS or missing:', k)
print(f"{new}: {n - bad} of {n} output hashes equal to {old}")
for c, v in sorted(tps.items()):
    print(f"  {c:10s} {len(v)} runs  mean t/s {sum(a for a, _ in v) / len(v):6.2f}  (stored {sum(b for _, b in v) / len(v):6.2f})")
