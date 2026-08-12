#!/usr/bin/env python3
"""Read and compare `dump` files — llama.cpp's named intermediates.

    python dump_inspect.py ref.ntd                    # list what is in it
    python dump_inspect.py ref.ntd --name attn_norm   # one tensor's values
    python dump_inspect.py ref.ntd mine.ntd           # compare two dumps

Why compare at the tensor: end-to-end logit KL saturates on a deep model, so it
cannot tell a subtly wrong kernel from a correct one — the same argument that
made `moe-server --compare` necessary (nano-glm/OPTIMIZATION.md). Porting an
architecture one tensor at a time turns a 1000-line all-or-nothing rewrite into
a sequence of checks that each fail in one place.

The comparison reports **max absolute and max relative** difference, not a
mean. A mean over 4096 values hides one wrong element, and one wrong element is
exactly what a mis-indexed view produces.

Stdlib only, as everything else here is.
"""

import argparse
import struct
import sys

MAGIC = b"NTDUMP1\0"

# ggml_type -> name, for the few that reach a dump. The data is always f32 by
# the time it is written; this only reports what it was.
TYPE_NAMES = {0: "f32", 1: "f16", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 30: "bf16"}


def read_dump(path):
    """-> list of {name, type, ne, data}. Streams; the files are small."""
    out = []
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise SystemExit("%s: not a dump file (bad magic)" % path)
        (n_records,) = struct.unpack("<I", f.read(4))
        for _ in range(n_records):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8", "replace")
            (src_type,) = struct.unpack("<i", f.read(4))
            ne = struct.unpack("<4q", f.read(32))
            (n_elem,) = struct.unpack("<Q", f.read(8))
            data = struct.unpack("<%df" % n_elem, f.read(4 * n_elem))
            out.append({"name": name, "type": src_type, "ne": ne, "data": data})
    return out


def _n_elem(ne):
    n = 1
    for d in ne:
        n *= d
    return n


def moments(v):
    n = len(v)
    if n == 0:
        return 0.0, 0.0, 0.0
    s = sum(v)
    amax = max(abs(x) for x in v)
    bad = sum(1 for x in v if x != x or abs(x) > 3.0e38)
    return s, amax, bad


def cmd_list(recs, name_filter):
    print("%-34s %-6s %-24s %14s %14s" % ("tensor", "type", "shape", "sum", "max|.|"))
    for r in recs:
        if name_filter and name_filter not in r["name"]:
            continue
        s, amax, bad = moments(r["data"])
        shape = "x".join(str(d) for d in r["ne"] if d != 1) or "1"
        print("%-34s %-6s %-24s %+14.6e %14.6e%s"
              % (r["name"], TYPE_NAMES.get(r["type"], str(r["type"])), shape, s, amax,
                 "   NaN/inf: %d" % bad if bad else ""))


def cmd_show(recs, name, limit):
    hits = [r for r in recs if name in r["name"]]
    if not hits:
        raise SystemExit("no tensor matching %r" % name)
    for r in hits:
        print("\n=== %s  %s  ne=%s  (%d elements)"
              % (r["name"], TYPE_NAMES.get(r["type"], r["type"]), list(r["ne"]), len(r["data"])))
        for i, v in enumerate(r["data"][:limit]):
            print("  [%6d] %+.8e" % (i, v))
        if len(r["data"]) > limit:
            print("  ... %d more" % (len(r["data"]) - limit))


def cmd_compare_by_order(a, b, tol):
    """Pair the i-th record of each file, ignoring names.

    For tensors llama.cpp never named: `dump --op FLASH_ATTN_EXT` gives them
    ggml's auto-generated "node_N", which cannot line up with anything a port
    calls them. Graph order can, as long as both sides are filtered down to the
    same short list — so this is only honest on a handful of tensors whose
    correspondence you have checked by shape.
    """
    n = min(len(a), len(b))
    n_shape = 0
    n_unpaired = abs(len(a) - len(b))
    if len(a) != len(b):
        print("!! %d vs %d records — comparing the first %d in order\n" % (len(a), len(b), n))
    worst = None
    print("%-20s %-20s %-18s %12s %12s" % ("A", "B", "shape", "max|a-b|", "max rel"))
    for i in range(n):
        r, o = a[i], b[i]
        if r["ne"] != o["ne"]:
            print("%-20s %-20s  SHAPE %s vs %s" % (r["name"], o["name"], list(r["ne"]), list(o["ne"])))
            n_shape += 1
            continue
        if len(r["data"]) != len(o["data"]):   # see cmd_compare: a partial check is not a pass
            print("%-20s %-20s  PARTIAL %d vs %d elements of %d"
                  % (r["name"], o["name"], len(r["data"]), len(o["data"]), _n_elem(r["ne"])))
            n_shape += 1
            continue
        amax = rmax = 0.0
        for x, y in zip(r["data"], o["data"]):
            d = abs(x - y)
            if d > amax:
                amax = d
            scale = max(abs(x), abs(y))
            if scale > 1e-30 and d / scale > rmax:
                rmax = d / scale
        shape = "x".join(str(d) for d in r["ne"] if d != 1) or "1"
        flag = "" if amax <= tol else "   <-- above tol"
        print("%-20s %-20s %-18s %12.4e %12.4e%s"
              % (r["name"], o["name"], shape, amax, rmax, flag))
        if worst is None or amax > worst[1]:
            worst = (r["name"], amax, rmax)
    print("\n%d compared, %d shape mismatches, %d unpaired" % (n - n_shape, n_shape, n_unpaired))
    if worst:
        print("worst: %s  max|a-b| %.4e  max rel %.4e" % worst)
    # A shape mismatch or an unpaired record is a failure, not a silent pass:
    # nothing was compared for it. Zero tensors compared is a failure too — a
    # check that cannot fail is worse than no check.
    if n_shape or n_unpaired or n == 0:
        return 1
    return 0 if (worst is None or worst[1] <= tol) else 1


def cmd_compare(a, b, name_filter, tol):
    """Match by name and position, so a repeated name (llama.cpp reuses `norm`)
    still lines up as long as both sides emit it in the same order."""
    seen = {}
    idx_b = {}
    for r in b:
        k = (r["name"], seen.get(r["name"], 0))
        seen[r["name"]] = seen.get(r["name"], 0) + 1
        idx_b[k] = r

    seen = {}
    worst = None
    n_cmp = n_missing = n_shape = n_trunc = 0
    print("%-34s %-20s %12s %12s" % ("tensor", "shape", "max|a-b|", "max rel"))
    for r in a:
        k = (r["name"], seen.get(r["name"], 0))
        seen[r["name"]] = seen.get(r["name"], 0) + 1
        if name_filter and name_filter not in r["name"]:
            continue
        o = idx_b.get(k)
        if o is None:
            print("%-34s  MISSING on the right" % r["name"])
            n_missing += 1
            continue
        if r["ne"] != o["ne"]:
            print("%-34s  SHAPE %s vs %s" % (r["name"], list(r["ne"]), list(o["ne"])))
            n_shape += 1
            continue
        # `dump --max-elem` truncates a record's data while leaving `ne` intact,
        # so two records can agree on shape and carry different amounts of it.
        # Comparing the overlap and calling that a pass is how a tensor gets
        # declared identical on the strength of its first third: at 384 tokens
        # the 4M default cut `attn_raw` exactly where a sliding-window mask
        # starts to differ from a causal one. A partial check is not a pass.
        if len(r["data"]) != len(o["data"]):
            print("%-34s  PARTIAL %d vs %d elements of %d — raise dump --max-elem"
                  % (r["name"], len(r["data"]), len(o["data"]), _n_elem(r["ne"])))
            n_trunc += 1
            continue

        amax = 0.0
        rmax = 0.0
        for x, y in zip(r["data"], o["data"]):
            d = abs(x - y)
            if d > amax:
                amax = d
            scale = max(abs(x), abs(y))
            if scale > 1e-30:
                rel = d / scale
                if rel > rmax:
                    rmax = rel
        n_cmp += 1
        shape = "x".join(str(d) for d in r["ne"] if d != 1) or "1"
        flag = "" if amax <= tol else "   <-- above tol"
        print("%-34s %-20s %12.4e %12.4e%s" % (r["name"], shape, amax, rmax, flag))
        if worst is None or amax > worst[1]:
            worst = (r["name"], amax, rmax)

    print("\n%d compared, %d missing, %d shape mismatches, %d truncated"
          % (n_cmp, n_missing, n_shape, n_trunc))
    if worst:
        print("worst: %s  max|a-b| %.4e  max rel %.4e" % worst)
    print("\nRead the max, not a mean: one wrong element is what a mis-indexed\n"
          "view produces, and an average over 4096 values hides it.")
    return 0 if (n_missing == 0 and n_shape == 0 and n_trunc == 0
                 and (worst is None or worst[1] <= tol)) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one dump to list, two to compare")
    ap.add_argument("--name", default=None, help="substring filter on the tensor name")
    ap.add_argument("--show", action="store_true", help="print values, not just moments")
    ap.add_argument("--limit", type=int, default=16, help="values to print with --show")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs difference treated as agreement (default 0: exact)")
    ap.add_argument("--by-order", action="store_true",
                    help="pair the i-th record of each file instead of matching names "
                         "(for tensors llama.cpp never named — see --op in dump)")
    ap.add_argument("--name-b", default=None,
                    help="with --by-order: name filter for the second file, when the "
                         "two sides call the same tensor different things")
    args = ap.parse_args()

    if len(args.files) == 1:
        recs = read_dump(args.files[0])
        print("%s: %d tensors\n" % (args.files[0], len(recs)))
        if args.show:
            cmd_show(recs, args.name or "", args.limit)
        else:
            cmd_list(recs, args.name)
        return 0

    a, b = read_dump(args.files[0]), read_dump(args.files[1])
    print("A %s: %d tensors\nB %s: %d tensors\n"
          % (args.files[0], len(a), args.files[1], len(b)))
    if args.by_order:
        keep = lambda recs, f: [r for r in recs if not f or f in r["name"]]
        return cmd_compare_by_order(keep(a, args.name),
                                    keep(b, args.name_b if args.name_b is not None else args.name),
                                    args.tol)
    return cmd_compare(a, b, args.name, args.tol)


if __name__ == "__main__":
    sys.exit(main())
