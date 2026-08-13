#!/usr/bin/env python3
"""Read a MOESERV_CAPTURE directory — what the backend was actually handed.

    python tape_inspect.py results/cap              # summary
    python tape_inspect.py results/cap --record 0   # one split, node by node

`tape` records the split generically: an ordered node list with ops, shapes,
types, op_params and source indices, plus the data of every tensor with no
producer inside the split and of the output. This reads that back, which is the
cheapest possible check that the format is self-consistent before a C++ replay
tool is written against it.

Deliberately not a replay: it does no arithmetic. Replay needs ggml to
dequantize MXFP4 and run the ops, and reimplementing either here would create a
second definition of the thing being tested — the mistake this project has
already paid for once.

Stdlib only, as everything else here is.
"""

import argparse
import os
import struct
import sys

MAGIC = b"MOETAPE2"
GGML_MAX_DIMS = 4
GGML_MAX_SRC = 10
GGML_MAX_OP_PARAMS = 64

# Op names come from the capture, written by ggml_op_name at record time. An
# earlier version of this file hardcoded the enum and decoded MUL_MAT_ID as
# "op50", because the numbering is internal and had moved. Type numbers are
# still decoded here, but only for display, and unknown ones print as t<N>.
TYPES = {0: "f32", 1: "f16", 30: "bf16", 39: "mxfp4", 14: "q6_K", 8: "q8_0",
         26: "i32", 27: "i64"}


def type_name(v):
    return TYPES.get(v, "t%d" % v)


class Reader:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def u32(self):
        v, = struct.unpack_from("<I", self.d, self.o)
        self.o += 4
        return v

    def u64(self):
        v, = struct.unpack_from("<Q", self.d, self.o)
        self.o += 8
        return v

    def i32(self):
        v, = struct.unpack_from("<i", self.d, self.o)
        self.o += 4
        return v

    def raw(self, n):
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def string(self):
        n = self.u32()
        return self.raw(n).decode("utf-8", "replace")


def read(path):
    with open(os.path.join(path, "records.bin"), "rb") as f:
        data = f.read()
    r = Reader(data)
    if r.raw(8) != MAGIC:
        raise SystemExit("%s: not a capture (bad magic)" % path)
    n_records = r.u32()

    records = []
    while r.o < len(data) and len(records) < n_records:
        rec = {"index": r.u32(), "n_nodes": r.u32()}
        n_all = r.u32()
        # Thread count is part of the arithmetic, not metadata: ggml partitions
        # a matmul by it, and llama.cpp uses a different one for prefill than
        # for decode. Replay reads this; nothing else can reproduce the bits.
        rec["n_threads"] = r.u32()
        tensors = []
        for _ in range(n_all):
            t = {"name": r.string(), "op_name": r.string(), "type": r.u32(), "op": r.u32()}
            t["ne"] = [r.u64() for _ in range(GGML_MAX_DIMS)]
            t["nb"] = [r.u64() for _ in range(GGML_MAX_DIMS)]
            t["op_params"] = r.raw(GGML_MAX_OP_PARAMS)
            t["src"] = [r.i32() for _ in range(GGML_MAX_SRC)]
            t["view_src"] = r.i32()
            t["view_offs"] = r.u64()
            if r.u32():
                t["nbytes"] = r.u64()
                t["blob"] = "%016x" % r.u64()
            tensors.append(t)
        rec["tensors"] = tensors
        records.append(rec)
    return records


def shape(t):
    return "x".join(str(d) for d in t["ne"] if d != 1) or "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--record", type=int, default=None)
    args = ap.parse_args()

    recs = read(args.dir)
    blobs = os.path.join(args.dir, "blobs")
    n_blobs = len(os.listdir(blobs)) if os.path.isdir(blobs) else 0
    blob_bytes = sum(os.path.getsize(os.path.join(blobs, b))
                     for b in os.listdir(blobs)) if n_blobs else 0

    print("%s: %d records, %d blobs, %.1f MB"
          % (args.dir, len(recs), n_blobs, blob_bytes / 1e6))

    if args.record is None:
        # Distinct split shapes: prefill and decode differ and both matter.
        shapes = {}
        for rec in recs:
            key = tuple((t["op_name"], shape(t)) for t in rec["tensors"][:rec["n_nodes"]])
            shapes.setdefault(key, []).append(rec["index"])
        print("\n%d distinct split shapes:" % len(shapes))
        for key, idxs in shapes.items():
            ops = {}
            for op, _ in key:
                ops[op] = ops.get(op, 0) + 1
            desc = " ".join("%s x%d" % (o, c) for o, c in sorted(ops.items()))
            print("  %-56s %d records (first: %d)" % (desc, len(idxs), idxs[0]))
        print("\nrun with --record N for one split's nodes")
        return 0

    rec = recs[args.record]
    print("\nrecord %d: %d nodes, %d tensors, %d threads\n"
          % (rec["index"], rec["n_nodes"], len(rec["tensors"]), rec["n_threads"]))
    print("%-4s %-12s %-7s %-22s %-14s %s"
          % ("#", "op", "type", "shape", "srcs", "data"))
    for i, t in enumerate(rec["tensors"]):
        srcs = ",".join(str(s) for s in t["src"] if s >= 0) or "-"
        data = ("%s %.1f KB" % (t["blob"][:8], t["nbytes"] / 1024.0)) if "blob" in t else ""
        mark = "N" if i < rec["n_nodes"] else "L"
        print("%-4s %-12s %-7s %-22s %-14s %s"
              % ("%s%d" % (mark, i), t["op_name"], type_name(t["type"]),
                 shape(t), srcs, data))
    print("\nN = node (computed here), L = leaf (input, data captured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
