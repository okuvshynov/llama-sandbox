#!/usr/bin/env python3
"""A golden set cut down to what an end-to-end run reads: per step the token ids and the
reference's logits (README step 187). No torch.

    python3 ref/trim_golden.py data/golden/readme1100 tests/data/e2e/readme1100

A golden set (export_golden.py) holds every layer's tensors: 0.1 to 65 GB, not in git.
`ds-test --e2e` reads only each step's `tokens` (what to run) and `logits` (to print the
reference's choice beside ours): 4 bytes per token and 517 KB per step. The cut set has the same
format (index.txt, data.bin), so `ds-test -g tests/data/e2e/<name> --e2e FILE` runs on it, and it is
small enough for git. The per-layer table needs the whole set: `ds-test` on a cut set without
--e2e stops with "golden lacks step 0 layer 0".
"""
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
data = open(os.path.join(src, 'data.bin'), 'rb')
out = open(os.path.join(dst, 'data.bin'), 'wb')
idx = open(os.path.join(dst, 'index.txt'), 'w')
SIZE = {'f32': 4, 'i32': 4}
kept = 0
for line in open(os.path.join(src, 'index.txt')):
    if line.startswith('#'):
        idx.write(line if not line.startswith('# source') else line.rstrip('\n') + f' (cut by ref/trim_golden.py from {src}: tokens and logits only)\n')
        continue
    f = line.split()
    step, layer, key, dtype, ndim = f[0], int(f[1]), f[2], f[3], int(f[4])
    if layer != -1 or key not in ('tokens', 'logits'):
        continue
    dims = [int(x) for x in f[5:5 + ndim]]
    n = SIZE[dtype]
    for d in dims:
        n *= d
    data.seek(int(f[5 + ndim]))
    blob = data.read(n)
    assert len(blob) == n, line
    idx.write(' '.join(f[:5 + ndim] + [str(out.tell())]) + '\n')
    out.write(blob)
    kept += 1
out.close()
idx.close()
print(f'{dst}: {kept} tensors, {os.path.getsize(os.path.join(dst, "data.bin"))} bytes')
