#!/usr/bin/env python3
"""Statistics over a nano-glm routing trace (--expert-log, see lib/expert_trace.h).

The trace records, for every position and every MoE layer, which experts the
router selected. This script turns that into the numbers PLAN.md needs before
committing to a GPU-resident expert subset (step 3) or to speculative prefetch:

- **skew** — how far from uniform is expert usage? A uniform router makes a
  resident subset worth exactly its size and nothing more.
- **static residency** — if the top f% of experts per layer live in VRAM, what
  fraction of selections hit? Measured *out of sample*: the ranking comes from
  the first half of the run and is scored on the second, because a ranking
  fitted and scored on the same tokens flatters itself by exactly the amount of
  noise in the counts.
- **locality** — how much does one token's expert set overlap the previous
  token's, and how many distinct experts does a layer touch in a window? This
  is what decides whether a cache (as opposed to a static placement) can pay.

Every locality number is printed next to what statistical independence would
predict, since "70% of experts touched in 32 tokens" means nothing until you
know that independent uniform routing would touch 64%.

`--null` replaces the trace's selections with i.i.d. uniform draws of the same
shape and reports on those instead. Run it once per trace: it is the honest
zero point. Notably, at 1138 positions the in-sample residency column reads
~30% for a 23% subset *with no skew whatsoever* — that inflation is finite
sample size, and it is why the out-of-sample column is the one to read.

Usage:
    python expert_stats.py run.trace [--resident 5,10,23,33,50]
                                     [--windows 1,8,32,128] [--csv counts.csv]
                                     [--null]
"""

import random
import sys
from collections import Counter


# ---------------------------------------------------------------------------
# parsing

def parse_trace(path):
    meta = {}
    tokens = []            # token id per position, in file order
    positions = []         # the pos field, for a monotonicity check
    layers = []            # layer ids, ascending, in the order they appear
    sel = {}               # layer -> list over positions of tuple(expert ids)
    layer_index = {}

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
                continue
            if not line.strip():
                continue
            kind, rest = line.split(" ", 1)
            if kind == "p":
                a, b = rest.split()
                positions.append(int(a))
                tokens.append(int(b))
            elif kind == "l":
                a, b = rest.split()
                layer = int(a)
                if layer not in layer_index:
                    layer_index[layer] = len(layers)
                    layers.append(layer)
                    sel[layer] = []
                sel[layer].append(tuple(int(x) for x in b.split(",")))
            else:
                raise SystemExit("unexpected line kind %r in %s" % (kind, path))

    n_pos = len(tokens)
    for layer in layers:
        if len(sel[layer]) != n_pos:
            raise SystemExit("layer %d has %d rows, expected %d — truncated trace?"
                             % (layer, len(sel[layer]), n_pos))
    if positions != list(range(n_pos)):
        raise SystemExit("positions are not 0..n-1 — trace from more than one sequence?")
    return meta, tokens, layers, sel


# ---------------------------------------------------------------------------
# metrics

def entropy_bits(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    from math import log2
    return -sum((c / total) * log2(c / total) for c in counts.values() if c)


def mass_to_reach(counts, frac):
    """How many experts, taken most-used first, cover `frac` of all selections."""
    total = sum(counts.values())
    acc, n = 0, 0
    for c in sorted(counts.values(), reverse=True):
        acc += c
        n += 1
        if acc >= frac * total:
            break
    return n


def resident_hit_rate(train_rows, test_rows, n_expert, frac):
    """Rank experts on train_rows, score the share of test selections that hit.

    frac is the share of the layer's experts held resident. Ties at the cut are
    broken by expert id, which is arbitrary but stable — with a long enough run
    ties only happen deep in the tail where they cost nothing.
    """
    n_res = max(1, int(round(frac * n_expert)))
    counts = Counter()
    for row in train_rows:
        counts.update(row)
    ranked = sorted(range(n_expert), key=lambda e: (-counts[e], e))
    resident = set(ranked[:n_res])
    hits = total = 0
    for row in test_rows:
        total += len(row)
        hits += sum(1 for e in row if e in resident)
    return hits / total if total else 0.0


def global_resident_hit_rate(sel, layers, n_expert, frac, split):
    """Same as resident_hit_rate, but the budget is spent across all layers at once.

    Per-layer residency gives every layer the same share whether or not it has
    a head to hold. Ranking (layer, expert) cells globally lets the budget go
    where the skew is; the difference between the two is what non-uniform
    placement is worth.
    """
    budget = max(1, int(round(frac * n_expert * len(layers))))
    counts = Counter()
    for layer in layers:
        for row in sel[layer][:split]:
            for e in row:
                counts[(layer, e)] += 1
    cells = sorted(((layer, e) for layer in layers for e in range(n_expert)),
                   key=lambda c: (-counts[c], c))
    resident = set(cells[:budget])
    hits = total = 0
    for layer in layers:
        for row in sel[layer][split:]:
            total += len(row)
            hits += sum(1 for e in row if (layer, e) in resident)
    return hits / total if total else 0.0


def consecutive_overlap(rows):
    """Mean |S_t & S_{t-1}| / k, and how often the rank-0 expert repeats."""
    if len(rows) < 2:
        return 0.0, 0.0
    ov = top1 = 0
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        ov += len(set(a) & set(b)) / len(b)
        top1 += 1 if a[0] == b[0] else 0
    n = len(rows) - 1
    return ov / n, top1 / n


def window_distinct(rows, w):
    """Mean number of distinct experts touched by a sliding window of w positions."""
    if len(rows) < w:
        return float("nan")
    live = Counter()
    total, n = 0, 0
    for i, row in enumerate(rows):
        live.update(row)
        if i >= w:
            for e in rows[i - w]:
                live[e] -= 1
                if live[e] == 0:
                    del live[e]
        if i >= w - 1:
            total += len(live)
            n += 1
    return total / n


def iid_distinct(n_expert, k, w):
    """Distinct experts a window of w would touch if every draw were independent."""
    return n_expert * (1.0 - (1.0 - k / n_expert) ** w)


def mean_min_max(xs):
    return sum(xs) / len(xs), min(xs), max(xs)


# ---------------------------------------------------------------------------
# report

def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    path = argv[1]
    resident_fracs = [0.05, 0.10, 0.23, 0.33, 0.50]
    windows = [1, 8, 32, 128]
    csv_path = None
    null = False

    i = 2
    while i < len(argv):
        if argv[i] == "--null":
            null = True
            i += 1
            continue
        if argv[i] == "--resident":
            resident_fracs = [float(x) / 100.0 for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--windows":
            windows = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--csv":
            csv_path = argv[i + 1]
            i += 2
        else:
            raise SystemExit("unknown argument %r" % argv[i])

    meta, tokens, layers, sel = parse_trace(path)
    n_pos = len(tokens)
    n_expert = int(meta.get("n_expert", 0))
    k = int(meta.get("n_expert_used", 0))
    n_prompt = int(meta.get("n_prompt", 0))
    if not n_expert or not k:
        raise SystemExit("trace is missing n_expert / n_expert_used metadata")

    if null:
        # Same shape, no structure: every metric below then reads out what this
        # many samples produce by chance, which is the only fair thing to
        # compare the real numbers against.
        random.seed(20260807)
        for layer in layers:
            sel[layer] = [tuple(random.sample(range(n_expert), k)) for _ in range(n_pos)]

    n_sel = n_pos * len(layers) * k
    split = n_pos // 2

    print("=" * 78)
    print("routing trace: %s%s" % (path, "   [--null: SELECTIONS REPLACED BY UNIFORM DRAWS]" if null else ""))
    print("  model            %s" % meta.get("model", "?"))
    print("  positions        %d (%d prompt + %d generated)" % (n_pos, n_prompt, n_pos - n_prompt))
    print("  MoE layers       %d (%s..%s)" % (len(layers), layers[0], layers[-1]))
    print("  experts          %d per layer, %d used per token" % (n_expert, k))
    print("  selections       %d total, %.1f expected per (layer, expert) cell"
          % (n_sel, n_pos * k / n_expert))

    # ---- token sanity: greedy decoding can fall into a loop, and a loop
    # would masquerade as routing concentration.
    gen = tokens[n_prompt:]
    uniq = len(set(gen)) if gen else 0
    grams = Counter(tuple(gen[i:i + 4]) for i in range(len(gen) - 3))
    rep4 = sum(c for c in grams.values() if c > 1) / max(1, sum(grams.values()))
    print("  generated tokens %d distinct of %d (%.0f%%), repeated 4-grams %.1f%%"
          % (uniq, len(gen), 100.0 * uniq / max(1, len(gen)), 100.0 * rep4))
    if rep4 > 0.30:
        print("  ** WARNING: the generation is repetitive; routing stats will inherit that **")

    # ---- per layer
    print()
    print("per-layer routing (sorted by layer)")
    print("  %-5s %6s %7s %6s %6s %7s %7s" %
          ("layer", "used", "H bits", "top1%", "n@50%", "n@90%", "ovlap%"))
    rows_hdr = "-" * 50
    print("  " + rows_hdr)

    per_layer = []
    for layer in layers:
        rows = sel[layer]
        counts = Counter()
        for r in rows:
            counts.update(r)
        H = entropy_bits(counts)
        top1 = max(counts.values()) / (n_pos * k)
        ov, sticky = consecutive_overlap(rows)
        rec = dict(layer=layer, counts=counts, H=H, used=len(counts),
                   top1=top1, n50=mass_to_reach(counts, 0.50),
                   n90=mass_to_reach(counts, 0.90), ov=ov, sticky=sticky)
        per_layer.append(rec)
        print("  %-5d %6d %7.3f %6.2f %6d %7d %7.1f" %
              (layer, rec["used"], H, 100 * top1, rec["n50"], rec["n90"], 100 * ov))

    from math import log2
    H_max = log2(n_expert)
    print()
    print("skew (uniform routing would give H = %.3f bits, top1 = %.2f%%, n50 = %d)"
          % (H_max, 100.0 / n_expert, n_expert // 2))
    for name, key, fmt in (("entropy, bits", "H", "%.3f"),
                           ("experts ever used", "used", "%.0f"),
                           ("top-1 expert share, %", "top1", "%.2f"),
                           ("experts for 50% of selections", "n50", "%.1f"),
                           ("experts for 90% of selections", "n90", "%.1f")):
        vals = [r[key] * (100 if key == "top1" else 1) for r in per_layer]
        m, lo, hi = mean_min_max(vals)
        print("  %-32s mean " % name + fmt % m + "   min " + fmt % lo + "   max " + fmt % hi)

    # ---- static residency, out of sample
    print()
    print("static residency: rank experts on positions 0..%d, score on %d..%d"
          % (split - 1, split, n_pos - 1))
    print("  %-9s %10s %10s %10s %10s %10s" %
          ("resident", "hit mean", "hit min", "hit max", "in-sample", "global"))
    print("  " + "-" * 64)
    for f in resident_fracs:
        oos, ins = [], []
        for r in per_layer:
            rows = sel[r["layer"]]
            oos.append(resident_hit_rate(rows[:split], rows[split:], n_expert, f))
            ins.append(resident_hit_rate(rows[split:], rows[split:], n_expert, f))
        m, lo, hi = mean_min_max(oos)
        g = global_resident_hit_rate(sel, layers, n_expert, f, split)
        print("  %8.0f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%% %9.1f%%" %
              (100 * f, 100 * m, 100 * lo, 100 * hi, 100 * sum(ins) / len(ins), 100 * g))
    print("  (a uniform router would put the hit rate exactly on the resident %)")
    print("  'global' spends one budget across all layers instead of the same share in each")

    # ---- locality
    print()
    print("locality: how much does a token reuse the previous token's experts?")
    ov = [r["ov"] for r in per_layer]
    st = [r["sticky"] for r in per_layer]
    m, lo, hi = mean_min_max(ov)
    print("  consecutive-set overlap   mean %.1f%%  min %.1f%%  max %.1f%%   (independent: %.1f%%)"
          % (100 * m, 100 * lo, 100 * hi, 100.0 * k / n_expert))
    m, lo, hi = mean_min_max(st)
    print("  rank-0 expert repeats     mean %.1f%%  min %.1f%%  max %.1f%%   (independent: %.1f%%)"
          % (100 * m, 100 * lo, 100 * hi, 100.0 / n_expert))

    print()
    print("working set: distinct experts touched by a sliding window of W positions")
    print("  %-6s %12s %12s %10s" % ("W", "measured", "independent", "ratio"))
    print("  " + "-" * 43)
    for w in windows:
        vals = [window_distinct(sel[r["layer"]], w) for r in per_layer]
        vals = [v for v in vals if v == v]     # drop NaN when the run is shorter than W
        if not vals:
            continue
        m = sum(vals) / len(vals)
        exp = iid_distinct(n_expert, k, w)
        print("  %-6d %12.1f %12.1f %9.2fx" % (w, m, exp, m / exp))

    # ---- prompt vs generation
    if n_prompt and n_pos > n_prompt + 1:
        print()
        print("prompt vs generation (same metrics on each segment)")
        print("  %-12s %8s %10s %10s" % ("segment", "H bits", "ovlap%", "top1 rep%"))
        print("  " + "-" * 43)
        for name, lo_i, hi_i in (("prompt", 0, n_prompt), ("generated", n_prompt, n_pos)):
            Hs, ovs, sts = [], [], []
            for r in per_layer:
                rows = sel[r["layer"]][lo_i:hi_i]
                c = Counter()
                for row in rows:
                    c.update(row)
                Hs.append(entropy_bits(c))
                a, b = consecutive_overlap(rows)
                ovs.append(a)
                sts.append(b)
            print("  %-12s %8.3f %9.1f%% %9.1f%%" %
                  (name, sum(Hs) / len(Hs), 100 * sum(ovs) / len(ovs), 100 * sum(sts) / len(sts)))

    if csv_path:
        with open(csv_path, "w") as f:
            f.write("layer,expert,count\n")
            for r in per_layer:
                for e in range(n_expert):
                    f.write("%d,%d,%d\n" % (r["layer"], e, r["counts"][e]))
        print()
        print("wrote %s (%d rows)" % (csv_path, len(per_layer) * n_expert))


if __name__ == "__main__":
    main(sys.argv)
