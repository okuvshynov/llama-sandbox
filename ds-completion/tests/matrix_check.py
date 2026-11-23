#!/usr/bin/env python3
"""Run a matrix of runs and compare every output's hash with a stored run of the same matrix
(README step 190; for `scripts/check.py complete`).

  python3 tests/matrix_check.py MATRIX STORED_DIR [--prompts a,b] [--out DIR]     exit code 1 if a hash differs

scripts/run_matrix.py runs MATRIX (prompts x configurations) into DIR (new, under data/check/);
each record's output.hash must be that of STORED_DIR's record of the same prompt and
configuration. Where the matrix's lines name files under STORED_DIR/ (--spec-log), a copy of the
matrix with DIR-logs/ in its place is run, so that the stored files are not written over.
A hash says that an output differs, not where: the next step is the same run with --keep-out
on both commits and cmp (the hint printed on a failure).
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.chdir(ROOT)
    a = sys.argv[1:]
    matrix, stored = a[0], a[1]
    prompts = a[a.index('--prompts') + 1] if '--prompts' in a else None
    out = a[a.index('--out') + 1] if '--out' in a else f'data/check/matrix-{os.path.basename(matrix)}-{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    text = open(matrix).read()
    if stored.rstrip('/') + '/' in text:  # (the matrix writes logs beside its stored records)
        os.makedirs(out + '-logs', exist_ok=True)
        matrix = out + '.matrix'
        open(matrix, 'w').write(text.replace(stored.rstrip('/') + '/', out + '-logs/'))
    cmd = [sys.executable, 'scripts/run_matrix.py', matrix, '--out', out] + (['--prompts', prompts] if prompts else [])
    print('# ' + ' '.join(cmd[1:]), flush=True)
    rc = subprocess.run(cmd).returncode  # (its progress lines: one per run)
    old = {}
    for f in glob.glob(stored + '/*.json'):
        r = json.load(open(f))
        old.setdefault((r['label']['prompt'], r['label']['config']), r['output']['hash'])
    n = bad = 0
    for f in sorted(glob.glob(out + '/*.json')):
        r = json.load(open(f))
        k = (r['label']['prompt'], r['label']['config'])
        n += 1
        if old.get(k) != r['output']['hash']:
            bad += 1
            print(f'FAIL  {k[0]} {k[1]}: {r["output"]["hash"]}, stored {old.get(k, "nothing")}')
    ok = rc == 0 and bad == 0 and n > 0
    print(f'{"ok    " if ok else "FAIL  "}{n - bad} of {n} output hashes are those of {stored}/' + ('' if rc == 0 else f' (run_matrix.py: exit code {rc})'))
    if not ok:
        print(f'# the records: {out}/\n# next: the differing run again with --keep-out on this commit and on the last good one, and cmp of the two outputs')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
