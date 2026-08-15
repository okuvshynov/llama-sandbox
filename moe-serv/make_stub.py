#!/usr/bin/env python3
"""Cut a real GGUF down to its first N layers, so the correctness gate runs in
seconds against a model that is still genuinely the architecture under test.

    python make_stub.py <model-00001-of-000NN.gguf> <out.gguf> [--layers 1]

Why a stub rather than a capture-and-replay harness: the test we want is
"llama.cpp with our backend produces the same logits as llama.cpp without it",
and llama.cpp can only answer that if it can load a model. DeepSeek-V4-Flash is
150 GiB and takes minutes; one layer of it is ~5.8 GiB and takes seconds. The
gate is then two runs of a stock tool, with no bespoke serialisation format in
between to get wrong.

WHAT IS AND IS NOT TOUCHED

Nothing is sliced, requantised or rewritten. Tensors are copied byte for byte
and the only metadata change is `<arch>.block_count` (plus dropping the
`split.*` keys, since the output is a single file).

That is possible because a *prefix* of layers keeps its original numbering, and
every per-layer thing llama.cpp reads is indexed by that number:

  <arch>.hash_layer_count       layers below it route by token id; layer 0 does
  <arch>.attention.compress_ratios   entry i is layer i's compressor kind
  <arch>.swiglu_clamp_exp/_shexp     entry i is layer i's clamp limit

A prefix therefore needs no edits to any of them — llama.cpp reads the first
`block_count` entries and the rest are ignored. Renumbering layers (keeping,
say, only layer 7 as blk.0) would silently give it layer 0's compressor kind and
layer 0's clamp, which is how a stub stops being the model it claims to be.

The corollary is that the layer set is a prefix and not a choice. For
DeepSeek-V4-Flash the four kinds appear at layers 0 (hash router, no
compressor), 2 (ratio-4 lightning indexer) and 3 (learned router with
`exp_probs_b`, ratio 128), so `--layers 4` covers all of them at ~16 GiB, and
`--layers 1` is enough for the expert block, which is identical in all of them
and is the only part this project owns.

For GLM-5.2 (glm-dsa, 79 layers) the map is simpler: layers 0-2 are dense
(`leading_dense_block_count = 3`), 3-77 are homogeneous MoE, and 78 is a
NextN/MTP layer the prefix drops — which is why `nextn_predict_layers` is
edited below. `--layers 5` is the minimum with MoE in it twice (~20 GiB).

Stdlib only, as everything else here is.
"""

import argparse
import os
import struct
import sys

# GGUF metadata value types.
(T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL,
 T_STR, T_ARR, T_U64, T_I64, T_F64) = range(13)

SCALAR = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
          T_I32: "<i", T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q",
          T_F64: "<d"}
SCALAR_SZ = {t: struct.calcsize(f) for t, f in SCALAR.items()}

# (block size, bytes per block) for the ggml types this model uses. Deliberately
# not a full table: an unknown type raises rather than guessing a size, and the
# layout self-check below would catch a wrong entry anyway.
GGML_TYPE = {
    0:  (1, 4),     # f32
    1:  (1, 2),     # f16
    8:  (32, 34),   # q8_0
    12: (256, 144), # q4_K
    13: (256, 176), # q5_K
    14: (256, 210), # q6_K
    24: (1, 1),     # i8
    25: (1, 2),     # i16
    26: (1, 4),     # i32
    30: (1, 2),     # bf16
    39: (32, 17),   # mxfp4
}
TYPE_NAME = {0: "f32", 1: "f16", 8: "q8_0", 12: "q4_K", 13: "q5_K", 14: "q6_K",
             24: "i8", 25: "i16", 26: "i32", 30: "bf16", 39: "mxfp4"}


def nbytes(ne, ty):
    if ty not in GGML_TYPE:
        raise SystemExit("unknown ggml type %d — add it to GGML_TYPE" % ty)
    blck, size = GGML_TYPE[ty]
    if ne[0] % blck:
        raise SystemExit("ne[0]=%d is not a multiple of block size %d" % (ne[0], blck))
    n = (ne[0] // blck) * size
    for d in ne[1:]:
        n *= d
    return n


class Reader:
    """Strings are kept as raw bytes and never decoded: they are copied through
    verbatim, and a round trip through a codec is a way to corrupt a chat
    template or a tokenizer entry for no benefit."""

    def __init__(self, f):
        self.f = f

    def u32(self): return struct.unpack("<I", self.f.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.f.read(8))[0]

    def raw_str(self):
        n = self.u64()
        return self.f.read(n)

    def value(self, t):
        if t == T_STR:
            return self.raw_str()
        if t == T_ARR:
            et = self.u32()
            n = self.u64()
            if et == T_STR:
                return (et, [self.raw_str() for _ in range(n)])
            if et == T_ARR:
                raise SystemExit("nested arrays are not supported")
            return (et, list(struct.unpack("<%d%s" % (n, SCALAR[et][1]),
                                           self.f.read(SCALAR_SZ[et] * n))))
        return struct.unpack(SCALAR[t], self.f.read(SCALAR_SZ[t]))[0]


class Writer:
    def __init__(self, f):
        self.f = f

    def u32(self, v): self.f.write(struct.pack("<I", v))
    def u64(self, v): self.f.write(struct.pack("<Q", v))

    def raw_str(self, b):
        self.u64(len(b))
        self.f.write(b)

    def value(self, t, v):
        if t == T_STR:
            self.raw_str(v)
        elif t == T_ARR:
            et, items = v
            self.u32(et)
            self.u64(len(items))
            if et == T_STR:
                for s in items:
                    self.raw_str(s)
            else:
                self.f.write(struct.pack("<%d%s" % (len(items), SCALAR[et][1]), *items))
        else:
            self.f.write(struct.pack(SCALAR[t], v))


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise SystemExit("%s: not a GGUF file" % path)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        kv = []                      # ordered: [(key_bytes, type, value)]
        for _ in range(n_kv):
            k = r.raw_str()
            t = r.u32()
            kv.append((k, t, r.value(t)))

        tensors = []                 # [(name_bytes, ne, type, offset)]
        for _ in range(n_tensors):
            name = r.raw_str()
            nd = r.u32()
            ne = [r.u64() for _ in range(nd)]
            ty = r.u32()
            tensors.append((name, ne, ty, r.u64()))

        align = 32
        for k, t, v in kv:
            if k == b"general.alignment":
                align = v
        pos = f.tell()
        if pos % align:
            pos += align - pos % align
        return version, kv, tensors, pos, align


def check_layout(path, tensors, data_start, align):
    """Each tensor's computed size must fit before the next one starts, and the
    last must fit in the file. This is what makes the GGML_TYPE table safe to
    hand-write: a wrong entry moves an offset and is caught here rather than
    producing a model that loads and computes nonsense."""
    order = sorted(tensors, key=lambda t: t[3])
    size = os.path.getsize(path)
    for i, (name, ne, ty, off) in enumerate(order):
        n = nbytes(ne, ty)
        end = data_start + off + n
        limit = data_start + order[i + 1][3] if i + 1 < len(order) else size
        if end > limit:
            raise SystemExit(
                "%s: %s (%s %s) computed %d bytes, but only %d until the next tensor"
                % (path, name.decode(), TYPE_NAME.get(ty, ty),
                   "x".join(map(str, ne)), n, limit - (data_start + off)))


def shard_paths(first):
    """A split model names its parts `-00001-of-000NN.gguf`; a single file has
    no such suffix and is its own only shard."""
    base = os.path.basename(first)
    if "-00001-of-" not in base:
        return [first]
    total = int(base.split("-00001-of-")[1].split(".gguf")[0])
    return [first.replace("-00001-of-", "-%05d-of-" % i) for i in range(1, total + 1)]


def wanted(name, n_layers):
    if not name.startswith(b"blk."):
        return True
    return int(name.split(b".")[1]) < n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="first shard of the source model")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=1)
    args = ap.parse_args()

    shards = shard_paths(args.src)
    print("source: %d shard(s)" % len(shards))

    version, kv, _, _, align = read_header(shards[0])

    arch = None
    for k, t, v in kv:
        if k == b"general.architecture":
            arch = v.decode()
    if arch is None:
        raise SystemExit("no general.architecture")

    n_layer = None
    for k, t, v in kv:
        if k == b"%s.block_count" % arch.encode():
            n_layer = v
    print("arch: %s, %d layers -> %d" % (arch, n_layer, args.layers))
    if args.layers > n_layer:
        raise SystemExit("--layers %d exceeds the model's %d" % (args.layers, n_layer))

    # Pick tensors, remembering which shard each lives in.
    picked = []       # [(name, ne, ty, src_path, src_off, nbytes)]
    for path in shards:
        _, _, tensors, data_start, a = read_header(path)
        check_layout(path, tensors, data_start, a)
        for name, ne, ty, off in tensors:
            if wanted(name, args.layers):
                picked.append((name, ne, ty, path, data_start + off, nbytes(ne, ty)))
    picked.sort(key=lambda t: t[0])
    total = sum(p[5] for p in picked)
    print("keeping %d tensors, %.2f GiB" % (len(picked), total / (1 << 30)))
    if not picked:
        raise SystemExit("no tensors selected")

    # Metadata: block_count changes, split.* goes, per-layer arrays are cut to
    # the same prefix as the layers, everything else is verbatim.
    #
    # "Per-layer" is recognised by length rather than by name: llama.cpp's
    # `get_key_or_arr` insists an array be exactly block_count long
    # (swiglu_clamp_exp is), so any array of that length has to be cut with the
    # layers. Others are left alone even when they are per-layer in spirit —
    # compress_ratios has 46 entries for 43 layers and is read with a
    # `>= block_count` check, so cutting it would be wrong twice over. Each cut
    # is printed, because a silent metadata edit is the thing that would make
    # this stub quietly not the model it claims to be.
    out_kv = []
    for k, t, v in kv:
        if k.startswith(b"split."):
            continue
        if k == b"%s.block_count" % arch.encode():
            v = args.layers
        elif k == b"%s.nextn_predict_layers" % arch.encode() and v:
            # NextN/MTP layers live at the TOP of the stack (glm-dsa's blk.78
            # carries nextn.eh_proj/enorm/hnorm/shared_head_norm), so a prefix
            # drops them. Keeping the old count would make llama.cpp treat the
            # stub's last ordinary layer as a NextN layer — missing-tensor
            # errors at best, a silently uncomputed layer at worst.
            keep = max(0, args.layers - (n_layer - v))
            if keep != v:
                print("  nextn_predict_layers %d -> %d (prefix drops the nextn tail)" % (v, keep))
                v = keep
        elif t == T_ARR and len(v[1]) == n_layer:
            print("  per-layer array cut to %d: %s" % (args.layers, k.decode()))
            v = (v[0], v[1][:args.layers])
        out_kv.append((k, t, v))

    with open(args.out, "wb") as f:
        w = Writer(f)
        f.write(b"GGUF")
        w.u32(version)
        w.u64(len(picked))
        w.u64(len(out_kv))
        for k, t, v in out_kv:
            w.raw_str(k)
            w.u32(t)
            w.value(t, v)

        # Offsets are relative to the (aligned) start of the data section, so
        # the infos have to be written before it is known where that is. Both
        # passes use the same size arithmetic, so write placeholders and come
        # back — simpler than predicting the info block's own length.
        info_pos = f.tell()
        off = 0
        offsets = []
        for name, ne, ty, _, _, n in picked:
            offsets.append(off)
            off += n
            if off % align:
                off += align - off % align
        for (name, ne, ty, _, _, n), o in zip(picked, offsets):
            w.raw_str(name)
            w.u32(len(ne))
            for d in ne:
                w.u64(d)
            w.u32(ty)
            w.u64(o)

        pos = f.tell()
        if pos % align:
            f.write(b"\0" * (align - pos % align))
        data_start = f.tell()

        CHUNK = 32 << 20
        for i, (name, ne, ty, path, src_off, n) in enumerate(picked):
            f.seek(data_start + offsets[i])
            with open(path, "rb") as src:
                src.seek(src_off)
                left = n
                while left:
                    b = src.read(min(CHUNK, left))
                    if len(b) != min(CHUNK, left):
                        raise SystemExit("short read on %s" % name.decode())
                    f.write(b)
                    left -= len(b)
            sys.stdout.write("\r  %d/%d %-44s" % (i + 1, len(picked), name.decode()))
            sys.stdout.flush()
        # The final tensor's padding, so the file length matches the layout the
        # infos describe.
        f.seek(data_start + off)
        f.truncate()

    print("\nwrote %s (%.2f GiB)" % (args.out, os.path.getsize(args.out) / (1 << 30)))
    _, _, ts, ds, a = read_header(args.out)
    check_layout(args.out, ts, ds, a)
    print("layout check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
