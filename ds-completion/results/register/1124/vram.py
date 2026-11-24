import subprocess, sys, time, glob
cards = sorted(glob.glob('/sys/class/drm/card?/device/mem_info_vram_used'))
rd = lambda: [int(open(c).read()) / 1e6 for c in cards]
idle = rd()
p = subprocess.Popen(sys.argv[1:], stdout=subprocess.DEVNULL)
rows = []; t0 = time.time()
while p.poll() is None:
    rows.append((time.time() - t0, rd())); time.sleep(0.02)
print('  idle %s' % ['%.0f' % x for x in idle])
step = max(1, len(rows) // 12)
for t, r in rows[::step] + [rows[-1]]: print('  %5.2f s  %s' % (t, ['%.0f' % x for x in r]))
