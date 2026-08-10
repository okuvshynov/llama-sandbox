#!/usr/bin/env python3
"""Find descriptor-table splices in a weight file, without a reference copy.

The August 2026 corruption on this machine (see README.md, shard06/shard14
notes) was not random bit rot. A Windows writeback-DMA bug spliced an in-memory
page-descriptor list into the file: runs of ~216-232 bytes made of repeating
8-byte records shaped

    f1 04 XX XX 07 00 40 00

That is a *signature*, not noise, and it is essentially impossible in Q6_K
weight data — which means a damaged file can be diagnosed on its own, before
the good copy is available and before overwriting destroys the evidence.

    python scan_splice.py model.gguf                 # default signature
    python scan_splice.py model.gguf --tail 07004000 # a different one

Reports each run with its offset, length and alignment, because the shape is
the diagnosis: 512/4096-aligned points at storage or transport, unaligned at a
host-memory structure being handed to DMA, a regular stride at a descriptor
list rather than a one-off.

Exit code 0 if nothing matched, 1 if it found something.
"""

import argparse
import sys

CHUNK = 1 << 26          # 64 MiB
RECORD = 8               # descriptor record size the splices are made of


def scan(path, tail, min_records):
    """Yield (offset, length) for runs of >= min_records consecutive records
    whose bytes [4:8] equal `tail`."""
    runs = []
    with open(path, "rb") as f:
        base = 0
        carry = b""
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            data = carry + buf
            start_off = base - len(carry)

            # Walk 8-byte-aligned records relative to absolute file offset, so
            # a run spanning a chunk boundary is not split by the reader.
            first = (-start_off) % RECORD
            run_start = None
            run_len = 0
            i = first
            while i + RECORD <= len(data):
                if data[i + 4:i + 8] == tail:
                    if run_start is None:
                        run_start = start_off + i
                        run_len = 0
                    run_len += RECORD
                else:
                    if run_start is not None and run_len >= min_records * RECORD:
                        runs.append((run_start, run_len))
                    run_start = None
                i += RECORD
            if run_start is not None and run_len >= min_records * RECORD:
                runs.append((run_start, run_len))

            base += len(buf)
            carry = data[-RECORD:] if len(data) >= RECORD else data
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--tail", default="07004000",
                    help="hex of bytes [4:8] of the record (default: the 2026-08 signature)")
    ap.add_argument("--min-records", type=int, default=4,
                    help="consecutive records needed to call it a run (default 4)")
    args = ap.parse_args()

    tail = bytes.fromhex(args.tail)
    runs = scan(args.path, tail, args.min_records)

    if not runs:
        print("no runs matching %s found in %s" % (args.tail, args.path))
        return 0

    print("%d run(s) matching the descriptor signature in %s" % (len(runs), args.path))
    prev = None
    for off, ln in runs:
        align = ("4096-aligned" if off % 4096 == 0 else
                 "512-aligned"  if off % 512 == 0 else
                 "unaligned (start%%512=%d)" % (off % 512))
        stride = "" if prev is None else "  stride %+d (%.2f MiB)" % (off - prev, (off - prev) / 2**20)
        print("  offset %14d (0x%x)  len %5d  %s%s" % (off, off, ln, align, stride))
        prev = off
    return 1


if __name__ == "__main__":
    sys.exit(main())
