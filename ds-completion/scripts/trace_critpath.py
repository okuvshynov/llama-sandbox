#!/usr/bin/env python3
"""Where a verify step's time goes on the critical path, from a --trace file (README step 84).

Per verify step (the "verify N tokens" spans) and per layer of it:
  front      die 0 from the layer's start to its wait for the CPU's experts (attention, router,
             dispatch, its own experts' launch; the whole layer when no expert went to the CPU)
  handoff in die 0's dispatch end -> the host's "CPU experts" start (the copies to the host, the
             host's wake-up)
  CPU        the host's "CPU experts" span
  handoff out the CPU's end -> the end of die 0's wait (the copy of the CPU's rows back); with
             step 84's traces split into: issuing the copy (host), the lag until the copy starts
             on die 0, and the copy itself (die 0's "ffn.cpu_rows_copy" group)
  tail       the rest of the layer after the wait (the sum, the next layer's hyper-connections)
and per step: the time outside its layers (the head, the step's own overhead) and the draft
before it. Medians over the steps of each size, ms per step; then the CPU's experts per layer.

  scripts/trace_critpath.py TRACE.json[.gz] [--sizes 2,6]
"""
import argparse, gzip, json, re, statistics as st
from collections import defaultdict

HOST, GPU0, STEPS = 1, 2, 3


def load(path):
    f = gzip.open(path, 'rt') if path.endswith('.gz') else open(path)
    d = json.load(f)
    return [e for e in (d['traceEvents'] if isinstance(d, dict) else d) if e.get('ph') == 'X']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--sizes', default='', help='verify sizes to show (default: all)')
    a = ap.parse_args()
    ev = load(a.trace)
    by_tid = defaultdict(list)
    for e in ev:
        by_tid[e['tid']].append(e)
    for v in by_tid.values():
        v.sort(key=lambda e: e['ts'])
    steps = [e for e in by_tid[STEPS] if e['name'].startswith('verify ')]
    drafts = [e for e in by_tid[STEPS] if e['name'] == 'draft']

    def within(tid, a0, a1):
        return [e for e in by_tid[tid] if e['ts'] >= a0 - 1 and e['ts'] < a1 + 1]

    rows = defaultdict(lambda: defaultdict(list))  # size -> part -> per-step totals
    cpu_layers = defaultdict(list)                  # size -> (distinct experts, pairs, ms)
    cpu_phase = defaultdict(lambda: defaultdict(list))  # size -> phase -> per-layer ms
    for s in steps:
        n = int(s['name'].split()[1])
        a0, a1 = s['ts'], s['ts'] + s['dur']
        g = within(GPU0, a0, a1)
        h = within(HOST, a0, a1)
        layers = {int(e['name'].split()[1]): e for e in g if re.fullmatch(r'layer \d+', e['name'])}
        groups = defaultdict(dict)
        for e in g:
            m = re.fullmatch(r'(\S+) L(\d+)', e['name'])
            if m:
                groups[int(m.group(2))][m.group(1)] = e
        cpu, issue, phases = {}, {}, defaultdict(dict)
        for e in h:
            m = re.fullmatch(r'CPU experts L(\d+)', e['name'])
            if m:
                cpu[int(m.group(1))] = e
            elif e['name'] == "issue the CPU rows' copy":
                issue[round(e['ts'])] = e
        for L, c in cpu.items():  # its nested phases (step 84 traces)
            for e in h:
                if e['name'].startswith('cpu.') and e['ts'] >= c['ts'] - 1 and e['ts'] + e['dur'] <= c['ts'] + c['dur'] + 1:
                    phases[L][e['name']] = e
        if not layers:
            continue
        t = defaultdict(float)
        for L, le in layers.items():
            ls, lend = le['ts'], le['ts'] + le['dur']
            w = groups[L].get('ffn.wait_for_cpu_experts')
            if w is None or L not in cpu:
                t['front'] += le['dur']
                continue
            c = cpu[L]
            t['front'] += w['ts'] - ls
            d = groups[L].get('ffn.dispatch')
            if d:
                t['handoff in'] += max(0.0, c['ts'] - (d['ts'] + d['dur']))
            t['CPU'] += c['dur']
            cp = groups[L].get('ffn.cpu_rows_copy')
            wait_end = (cp['ts'] + cp['dur']) if cp else (w['ts'] + w['dur'])
            t['handoff out'] += max(0.0, wait_end - (c['ts'] + c['dur']))
            if cp:
                iss = issue.get(round(c['ts'] + c['dur']))
                t['  issue the copy (host)'] += iss['dur'] if iss else 0.0
                t['  lag to the copy'] += max(0.0, cp['ts'] - (c['ts'] + c['dur']) - (iss['dur'] if iss else 0.0))
                t['  the copy (die 0)'] += cp['dur']
            t['die 0 waits for CPU'] += wait_end - w['ts']
            t['tail'] += lend - wait_end
            for ph in ('cpu.inputs', 'cpu.gate_up', 'cpu.act', 'cpu.down'):
                pe = phases[L].get(ph)
                if pe:
                    cpu_phase[n][ph].append(pe['dur'] / 1000)
                    sp = pe.get('args', {}).get('thread spread us')
                    if sp is not None:
                        cpu_phase[n][ph + ' spread'].append(sp / 1000)
            args = c.get('args', {})
            cpu_layers[n].append((args.get('distinct experts', 0), args.get('pairs', 0), c['dur'] / 1000))
        first = min(e['ts'] for e in layers.values())
        last = max(e['ts'] + e['dur'] for e in layers.values())
        t['outside layers'] += s['dur'] - (last - first)
        prev = [d for d in drafts if d['ts'] + d['dur'] <= a0 + 1]
        if prev:
            t['draft'] += prev[-1]['dur']
        t['step'] += s['dur']
        for k, v in t.items():
            rows[n][k].append(v / 1000)
    sizes = sorted(rows) if not a.sizes else [int(x) for x in a.sizes.split(',')]
    parts = ['front', 'handoff in', 'CPU', 'handoff out', '  issue the copy (host)', '  lag to the copy', '  the copy (die 0)',
             'die 0 waits for CPU', 'tail', 'outside layers', 'draft', 'step']
    print(f"# {a.trace}: verify steps by size, median ms per step (die 0 front + its wait for the CPU + tail + outside = step)")
    print(f"{'part':26s}" + ''.join(f"{f'{n} tok ({len(rows[n][chr(115)+chr(116)+chr(101)+chr(112)])})':>14s}" for n in sizes if n in rows))
    for p in parts:
        if p.startswith('  ') and not any(rows[n][p] for n in sizes if n in rows):
            continue
        print(f"{p:26s}" + ''.join(f"{st.median(rows[n][p]) if rows[n][p] else 0:14.2f}" for n in sizes if n in rows))
    print("\n# the CPU's experts per layer that had any (medians): distinct experts, pairs, ms, ms per distinct expert")
    for n in sizes:
        c = cpu_layers.get(n)
        if not c:
            continue
        de = st.median(x[0] for x in c)
        print(f"  {n} tokens: {len(c)} layers, {de:.0f} experts, {st.median(x[1] for x in c):.0f} pairs, "
              f"{st.median(x[2] for x in c):.3f} ms, {st.median(x[2] / max(1, x[0]) for x in c):.3f} ms per expert")
        ph = cpu_phase.get(n)
        if ph:
            print("      phases (median ms per layer): " + ", ".join(
                f"{k[4:]} {st.median(v):.3f}" for k, v in ph.items() if not k.endswith('spread')) +
                  "; thread spread: " + ", ".join(f"{k[4:-7]} {st.median(v):.3f}" for k, v in ph.items() if k.endswith('spread')))


if __name__ == '__main__':
    main()
