#!/usr/bin/env python3
"""Per-chunk SHA-256 of a file, for localizing corruption between two copies.

Run the same command on both machines and diff the output: matching chunks
agree, the differing one narrows the search. Re-run with a smaller --chunk over
that range to bisect down to the damaged bytes.

    python chunk_hash.py model.gguf                      # 1 GiB chunks
    python chunk_hash.py model.gguf --chunk 1M --start 21474836480 --len 1G

The shape of the damage is the diagnosis: a single flipped bit points at a
marginal memory cell, one 512B/4K sector at storage or transport, an aligned
cluster at a filesystem or driver bug, a shifted region at the copy tool.
"""
import argparse
import hashlib
import sys


def parse_size(s):
    mult = {'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}
    s = str(s).strip()
    if s and s[-1].lower() in mult:
        return int(float(s[:-1]) * mult[s[-1].lower()])
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--chunk', default='1G', help='chunk size (default 1G)')
    ap.add_argument('--start', default='0', help='byte offset to start at')
    ap.add_argument('--len', dest='length', default=None, help='bytes to cover (default: to EOF)')
    a = ap.parse_args()

    chunk = parse_size(a.chunk)
    start = parse_size(a.start)
    limit = parse_size(a.length) if a.length else None

    read_sz = 1 << 22  # 4 MiB reads, independent of chunk size
    with open(a.path, 'rb') as f:
        f.seek(start)
        off = start
        done = 0
        while True:
            if limit is not None and done >= limit:
                break
            want = chunk if limit is None else min(chunk, limit - done)
            h = hashlib.sha256()
            got = 0
            while got < want:
                buf = f.read(min(read_sz, want - got))
                if not buf:
                    break
                h.update(buf)
                got += len(buf)
            if got == 0:
                break
            print(f'{off:014d}  {got:12d}  {h.hexdigest()}', flush=True)
            off += got
            done += got
            if got < want:
                break


if __name__ == '__main__':
    sys.exit(main())
