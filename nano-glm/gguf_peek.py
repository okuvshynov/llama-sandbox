#!/usr/bin/env python3
"""Print a GGUF file's metadata and tensor shapes. Stdlib only.

    python gguf_peek.py <file.gguf>              # KV metadata + tensor summary
    python gguf_peek.py <file.gguf> --tensors    # every tensor, name/shape/type
    python gguf_peek.py <file.gguf> --grep blk.0 # only matching tensor names

Exists because bringing up a new model starts with "what is actually in this
file", and the obvious answers are worse than they look: llama.cpp's gguf-py
needs numpy, and reading the shapes out of a loader that already assumes an
architecture tells you what the loader expects rather than what the file holds.

Reads only the header, so it is instant even on a 50 GB shard.
"""

import struct
import sys

# gguf_metadata_value_type
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
        F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}

# ggml_type -> (name, block size, bytes per block); only what we need to size
# tensors. Unknown types still print, just without a byte count.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17),
}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def scalar(self, t):
        fmt = _FMT[t]
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.scalar(U64)
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et = self.scalar(U32)
            n = self.scalar(U64)
            # Token lists run to hundreds of thousands of entries; skipping the
            # payload keeps this instant and nothing here needs the contents.
            if n > 64:
                if et == STRING:
                    for _ in range(n):
                        self.raw(self.scalar(U64))
                else:
                    self.raw(n * struct.calcsize(_FMT[et]))
                return "<%d entries, type %d>" % (n, et)
            return [self.value(et) for _ in range(n)]
        return self.scalar(t)


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if r.raw(4) != b"GGUF":
            raise SystemExit("%s: not a GGUF file" % path)
        version = r.scalar(U32)
        n_tensors = r.scalar(U64)
        n_kv = r.scalar(U64)

        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.scalar(U32))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.scalar(U32)
            dims = [r.scalar(U64) for _ in range(nd)]
            ttype = r.scalar(U32)
            r.scalar(U64)  # offset within the data section
            tensors.append((name, dims, ttype))
    return version, kv, tensors


def type_name(t):
    return GGML_TYPES.get(t, ("type%d" % t, 0, 0))[0]


def tensor_bytes(dims, t):
    info = GGML_TYPES.get(t)
    if not info:
        return 0
    _, blk, nb = info
    n = 1
    for d in dims:
        n *= d
    return (n // blk) * nb if blk else 0


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    path = sys.argv[1]
    want_tensors = "--tensors" in sys.argv
    grep = None
    if "--grep" in sys.argv:
        grep = sys.argv[sys.argv.index("--grep") + 1]

    version, kv, tensors = read_header(path)
    print("GGUF v%d — %d tensors, %d KV pairs" % (version, len(tensors), len(kv)))

    print("\n=== metadata")
    for k in sorted(kv):
        v = str(kv[k])
        print("  %-44s %s" % (k, v if len(v) <= 90 else v[:87] + "..."))

    if grep or want_tensors:
        print("\n=== tensors")
        for name, dims, t in tensors:
            if grep and grep not in name:
                continue
            print("  %-44s %-22s %-8s %10.2f MiB"
                  % (name, "x".join(str(d) for d in dims), type_name(t),
                     tensor_bytes(dims, t) / 1048576.0))
    else:
        # Group by the block-index-stripped name so one layer's shape stands in
        # for all of them.
        seen, order = {}, []
        for name, dims, t in tensors:
            key = name
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "blk":
                key = "blk.N." + ".".join(parts[2:])
            if key not in seen:
                seen[key] = [dims, t, 0]
                order.append(key)
            seen[key][2] += 1
        print("\n=== tensor shapes (one row per distinct name, blk.N collapsed)")
        for key in order:
            dims, t, count = seen[key]
            print("  %-40s %-22s %-8s x%d" %
                  (key, "x".join(str(d) for d in dims), type_name(t), count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
