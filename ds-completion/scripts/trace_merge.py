#!/usr/bin/env python3
"""One trace of the model on two machines (README step 112).

The front (layers 0-19, --remote) and the back (layers 20-39, --serve) each write their own
--trace, on their own machine's monotonic clock. The front's trace also holds clock samples
(peerSync): for every small exchange (HELLO, SYNC, DRAFT) its own times when the request left and
the answer arrived, and the back's times when the request had arrived and the answer left. As in
NTP, a sample's offset is ((t2 - t1) + (t3 - t4)) / 2 and its delay (t4 - t1) - (t3 - t2): the
time the two messages spent between the machines, or waiting in the back's socket while it was
busy (a draft request usually waits behind the commit sent just before it; that wait is on one
side only and would bias the offset by half of it). So only the samples whose delay is within
1.5x of the least one are kept (the SYNC exchanges at the start and the end of a run, and the
drafts that found the back idle); the offset's error is then under half that delay. A line is
fitted through their offsets over time (the two clocks drift apart by some ppm) and the back's
events are moved onto the front's clock with it.

Output: one Chrome trace, process 1 = the front's tracks, process 2 = the back's; open in
ui.perfetto.dev. Prints the fit: samples, offset, drift, the residuals and the delays.

    python3 scripts/trace_merge.py FRONT.json BACK.json -o MERGED.json[.gz]
"""
import gzip
import json
import statistics
import sys


def load(path):
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        return json.load(f)


def main():
    args = [a for a in sys.argv[1:] if a != '-o']
    if len(args) != 3 or '-o' not in sys.argv:
        sys.exit(__doc__)
    front_path, back_path = args[0], args[1]
    out_path = sys.argv[sys.argv.index('-o') + 1]
    front, back = load(front_path), load(back_path)
    if front['layers'][0] != 0:
        front, back = back, front
    sync = front.get('peerSync') or []
    if len(sync) < 2:
        sys.exit('the front trace has no clock samples (peerSync): was it run with --remote and --trace?')
    off = [((t2 - t1) + (t3 - t4)) / 2 for t1, t2, t3, t4 in sync]
    delay = [(t4 - t1) - (t3 - t2) for t1, t2, t3, t4 in sync]
    med = statistics.median(delay)
    keep = [i for i in range(len(sync)) if delay[i] <= 1.5 * min(delay)]
    # least squares offset = a + b (t - t0), t = the front's clock at the sample's middle
    t0 = sync[0][0]
    xs = [(sync[i][0] + sync[i][3]) / 2 - t0 for i in keep]
    ys = [off[i] for i in keep]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 and max(xs) - min(xs) > 1e6 else 0.0
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    rms = (sum(r * r for r in res) / n) ** 0.5

    fe, be = front['clockEpochUs'], back['clockEpochUs']

    def to_front(ts):  # a back trace time -> the merged trace's time (the front's)
        t_back = be + ts
        t_front = t_back - (a + b * (t_back - a - t0))  # offset at (about) that moment
        return t_front - fe

    fl, bl = front['layers'], back['layers']
    events = [
        {'name': 'process_name', 'ph': 'M', 'pid': 1, 'args': {'name': f'front: layers {fl[0]}-{fl[1]}'}},
        {'name': 'process_name', 'ph': 'M', 'pid': 2, 'args': {'name': f'back: layers {bl[0]}-{bl[1]}, the head, the draft'}},
        {'name': 'process_sort_index', 'ph': 'M', 'pid': 1, 'args': {'sort_index': 1}},
        {'name': 'process_sort_index', 'ph': 'M', 'pid': 2, 'args': {'sort_index': 2}},
    ]
    for e in front['traceEvents']:
        events.append(dict(e, pid=1))
    for e in back['traceEvents']:
        e = dict(e, pid=2)
        if 'ts' in e:
            e['ts'] = round(to_front(e['ts']), 1)
        events.append(e)
    merged = {'traceEvents': events, 'clockEpochUs': fe,
              'clockFit': {'samples': n, 'of': len(sync), 'offset_us': a, 'drift_ppm': b * 1e6, 'residual_rms_us': rms,
                           'delay_median_us': med, 'delay_min_us': min(delay)}}
    with (gzip.open(out_path, 'wt') if out_path.endswith('.gz') else open(out_path, 'w')) as f:
        json.dump(merged, f, separators=(',', ':'))
    span_s = (sync[-1][3] - sync[0][0]) / 1e6
    print(f'# clocks: {n} of {len(sync)} samples over {span_s:.1f} s kept (delay within 1.5x of the least, {min(delay):.0f} us; median of all {med:.0f} us)')
    print(f'# back - front = {a / 1e6:.6f} s at the start, drifting {b * 1e6:+.2f} ppm; residuals rms {rms:.1f} us, '
          f'largest {max(abs(r) for r in res):.1f} us')
    print(f'# {len(front["traceEvents"])} front + {len(back["traceEvents"])} back events in {out_path}')


if __name__ == '__main__':
    main()
