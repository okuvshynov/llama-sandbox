#!/usr/bin/env python3
"""Tokens per second on the held-out prompts, against a stored baseline (README step 190):
`scripts/check.py speed`.

  python3 tests/speed_check.py [--out DIR] [--store]        exit code 1 if slower, or if an output differs

The measurement the documents quote, unchanged: the eight held-out prompts, each in a fresh
process with the fastest settings (results/cleanup/heldout.matrix: placement, adaptive cache
with --cache-wait, the draft and speculation, --canon), twice. The first repeat is a warm-up
(after a boot prompts read from storage while generating, README step 178); the second is the
measurement. ~41 minutes, most of it the 16 starts (README step 189).

Verdict, on the mean of the second repeat against tests/speed/baseline.tsv:
  slower by more than 2%   FAIL
  within 2%                the same (run to run the mean moves ~0.5%; no difference is claimed)
  faster by more than 2%   ok, and it says so: --store makes it the baseline
Per prompt the difference is printed, and marked beyond 5%: a single prompt moves more than the mean.
Not a measurement, and so a failure whatever the numbers: a GPU clock level other than the
baseline's, or a second-repeat run that read more than 8 MB from storage while generating (a
cold first pass reads 17-77 MB per prompt; a warm one 0 to ~1 MB).
Every output must also be the stored bytes (results/cache-test/plain-canon-static/): a faster
wrong answer is not a result.
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX, STORED_OUT, BASELINE = 'results/cleanup/heldout.matrix', 'results/cache-test/plain-canon-static', 'tests/speed/baseline.tsv'
TOL, TOL_PROMPT, READ_MAX = 0.02, 0.05, 8 << 20


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    store = '--store' in args
    out = args[args.index('--out') + 1] if '--out' in args else f'data/check/speed-{time.strftime("%Y%m%d-%H%M%S")}/runs'
    if not glob.glob(out + '/*.json'):
        rc = subprocess.run([sys.executable, 'scripts/run_matrix.py', MATRIX, '--reps', '2', '--keep-out', '--out', out]).returncode
        if rc != 0:
            print(f'FAIL  a run failed (scripts/run_matrix.py, exit code {rc}): see {out}/')
            return 1
    recs = [json.load(open(f)) for f in sorted(glob.glob(out + '/*.json'))]
    failed = 0

    def check(ok, what):
        nonlocal failed
        failed += not ok
        print(('ok    ' if ok else 'FAIL  ') + what, flush=True)

    # the outputs: every one the stored bytes
    bad = []
    for f in sorted(glob.glob(out + '/*.out')):
        prompt = os.path.basename(f)[:-4].split('-', 1)[1].rsplit('-', 1)[0]
        if open(f, 'rb').read() != open(f'{STORED_OUT}/{prompt}.out', 'rb').read():
            bad.append(os.path.basename(f))
    n_out = len(glob.glob(out + '/*.out'))
    check(not bad and n_out == len(recs), f'{n_out - len(bad)} of {n_out} outputs are the stored bytes ({STORED_OUT}/)' + (f'; not: {", ".join(bad)}' if bad else ''))

    rep2 = {r['label']['prompt']: r for r in recs if r['label']['rep'] == '2'}
    rep1 = {r['label']['prompt']: r for r in recs if r['label']['rep'] == '1'}
    tps = {p: r['counters']['gen_tps'] for p, r in rep2.items()}
    mean = sum(tps.values()) / len(tps)
    sclk = sorted({json.dumps(r['env'].get('gpu_sclk')) for r in recs})
    read = max(r['read_bytes']['gen'] for r in rep2.values())
    base, base_sclk = {}, None
    if os.path.exists(BASELINE):
        for l in open(BASELINE):
            if l.startswith('# sclk '):
                base_sclk = l[7:].strip()
            elif l.strip() and not l.startswith('#'):
                p, v = l.split()
                base[p] = float(v)
    print(f'#   prompt          t/s   baseline   difference     (first repeat, read from storage while generating)')
    for p in tps:
        d = tps[p] / base[p] - 1 if p in base else None
        print(f'    {p:12s} {tps[p]:6.2f}   {base.get(p, float("nan")):8.2f}   {"" if d is None else f"{d:+7.1%}"}{"  <- beyond 5%" if d is not None and abs(d) > TOL_PROMPT else "":14s}'
              f' ({rep1[p]["counters"]["gen_tps"]:.2f}, {rep1[p]["read_bytes"]["gen"] / 1e6:.1f} MB)' if p in rep1 else '')
    check(len(sclk) == 1 and (base_sclk is None or sclk[0] == base_sclk), f'the GPU clock level: {", ".join(sclk)}' + (f' (the baseline\'s: {base_sclk})' if base_sclk else ''))
    check(read <= READ_MAX, f'the second repeat read at most {read / 1e6:.1f} MB from storage while generating (over 8 MB: not a measurement, run it again)')
    if store and not failed:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        with open(BASELINE, 'w') as f:
            f.write(f'# tests/speed_check.py --store: the second repeat\'s generation speed (t/s) per held-out prompt; commit {commit}; {time.strftime("%Y-%m-%d")}\n')
            f.write(f'# sclk {sclk[0]}\n')
            for p in tps:
                f.write(f'{p} {tps[p]:.3f}\n')
        print(f'#   {BASELINE} written: mean {mean:.2f} t/s' + (f' (it was {sum(base.values()) / len(base):.2f})' if base else ''))
    elif not base:
        check(False, f'mean {mean:.2f} t/s; no baseline ({BASELINE}): --store writes it')
    else:
        bm = sum(base.values()) / len(base)
        d = mean / bm - 1
        word = 'slower' if d < -TOL else 'faster: --store makes it the baseline' if d > TOL else 'the same within 2%, no difference claimed'
        check(d >= -TOL, f'mean {mean:.2f} t/s against the baseline\'s {bm:.2f}: {d:+.1%}, {word}')
    if failed:
        print(f'# the records and outputs: {out}/')
        print('# next: scripts/records.py on these records against an earlier run\'s (docs/record.md); where a step\'s time went:\n'
              '#   --trace FILE on the slower prompt and scripts/trace_critpath.py, against the same from the last good commit')
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
