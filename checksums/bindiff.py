#!/usr/bin/env python3
"""Compact binary diff of two copies of the same file.

Reports differing byte ranges, how many bits actually flipped, and the
alignment of each run — which is the diagnosis:

    single bit, isolated        marginal memory cell or SSD bit rot; a
                                filesystem reformat will not help
    one 512B or 4K sector       storage media or transport
    aligned 128K/1M cluster     filesystem or driver bug; reformatting to a
                                journalled filesystem is the fix
    shifted/duplicated region   copy tool

    python bindiff.py good.gguf bad.gguf --out diff-report.txt

Reads in parallel with a bounded buffer, so it handles files far larger than
RAM. Only the differing bytes are retained.
"""
import argparse
import sys

CHUNK = 1 << 22  # 4 MiB


def popcount_diff(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count('1') for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('good')
    ap.add_argument('bad')
    ap.add_argument('--out', default=None, help='write the report here as well as stdout')
    ap.add_argument('--max-runs', type=int, default=200, help='stop listing after N runs')
    ap.add_argument('--context', type=int, default=16, help='bytes of hex to show per run')
    a = ap.parse_args()

    out_lines = []

    def emit(s=''):
        print(s, flush=True)
        out_lines.append(s)

    runs = []           # (offset, good_bytes, bad_bytes)
    total_bytes = 0
    total_bits = 0
    off = 0

    with open(a.good, 'rb') as fg, open(a.bad, 'rb') as fb:
        cur = None      # [start, bytearray(good), bytearray(bad)]
        while True:
            g = fg.read(CHUNK)
            b = fb.read(CHUNK)
            if not g and not b:
                break
            if len(g) != len(b):
                emit(f'!! length differs: {a.good} and {a.bad} diverge in size at {off}')
            n = min(len(g), len(b))
            if g[:n] != b[:n]:
                for i in range(n):
                    if g[i] != b[i]:
                        total_bytes += 1
                        total_bits += bin(g[i] ^ b[i]).count('1')
                        if cur is not None and off + i == cur[0] + len(cur[1]):
                            cur[1].append(g[i]); cur[2].append(b[i])
                        else:
                            if cur is not None:
                                runs.append((cur[0], bytes(cur[1]), bytes(cur[2])))
                            cur = [off + i, bytearray([g[i]]), bytearray([b[i]])]
            off += n
            if not g or not b:
                break
        if cur is not None:
            runs.append((cur[0], bytes(cur[1]), bytes(cur[2])))

    emit(f'good : {a.good}')
    emit(f'bad  : {a.bad}')
    emit(f'size : {off} bytes compared')
    emit(f'diff : {total_bytes} bytes, {total_bits} bits, in {len(runs)} run(s)')
    emit()
    for i, (start, g, b) in enumerate(runs[:a.max_runs]):
        bits = popcount_diff(g, b)
        # alignment tells us which layer of the stack to blame
        aligns = [n for n in (1 << 20, 1 << 17, 4096, 512) if start % n == 0]
        al = f'{aligns[0]}-aligned' if aligns else f'unaligned (start%512={start % 512})'
        emit(f'run {i}: offset {start} (0x{start:x})  len {len(g)}  bits {bits}  {al}')
        emit(f'   good: {g[:a.context].hex(" ")}')
        emit(f'   bad : {b[:a.context].hex(" ")}')
        if len(g) == 1:
            emit(f'   xor : 0x{g[0] ^ b[0]:02x}  (single byte, {bits} bit(s) flipped)')
    if len(runs) > a.max_runs:
        emit(f'... {len(runs) - a.max_runs} more runs not shown')

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines) + '\n')
        print(f'\nwritten to {a.out}')


if __name__ == '__main__':
    sys.exit(main())
