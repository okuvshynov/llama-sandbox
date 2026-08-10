#!/usr/bin/env python3
"""Does a static resident expert subset transfer between prompts? (PLAN.md step 6)

`ROUTING.md` measured a 23% resident subset catching 58.4% of selections, but
that was *within one continuation* — ranked on the first half of a prose run and
scored on its second half. Same topic, same register: the easy case. A static
VRAM placement is chosen once and then serves every workload, so the number that
actually decides step 3 is how well a ranking built on one prompt scores on a
different one.

    python residency_study.py results/residency/*.trace [--resident 23]

Two questions, kept separate:

  per prompt   is one prompt's routing more concentrated than another's?
  across       does a subset chosen on prompt A still work on prompt B?

**Equal ranking budget everywhere.** Every cell of the transfer matrix ranks on
the *first half* of one prompt and scores on the *second half* of another,
including the diagonal. Without that, cross-prompt cells would be compared
against a diagonal built from more data, and "prompts differ" would be
indistinguishable from "the ranking is noisier". The diagonal is the ceiling
this study is measured against, not a separate result.

Everything is also reported for a same-shape uniform null, because at these
sample sizes a resident subset scores well above its own size by chance alone.
"""

import argparse
import importlib.util
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def load_parser():
    """Reuse expert_stats.py's trace reader rather than re-implementing it."""
    path = os.path.join(HERE, "expert_stats.py")
    spec = importlib.util.spec_from_file_location("expert_stats_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Trace:
    def __init__(self, path, mod):
        self.name = os.path.splitext(os.path.basename(path))[0]
        meta, tokens, layers, sel = mod.parse_trace(path)
        self.meta, self.tokens, self.layers, self.sel = meta, tokens, layers, sel
        self.n_pos = len(tokens)
        self.n_expert = int(meta["n_expert"])
        self.k = int(meta["n_expert_used"])
        self.half = self.n_pos // 2

    def counts(self, layer, lo, hi):
        c = Counter()
        for row in self.sel[layer][lo:hi]:
            c.update(row)
        return c

    def first_half(self, layer):
        return self.counts(layer, 0, self.half)

    def score_rows(self, layer):
        return self.sel[layer][self.half:]

    def uniformise(self, seed):
        """Replace selections with i.i.d. uniform draws of the same shape."""
        rnd = random.Random(seed)
        for layer in self.layers:
            self.sel[layer] = [tuple(rnd.sample(range(self.n_expert), self.k))
                               for _ in range(self.n_pos)]


def resident_set(counts, n_expert, n_res):
    """Top n_res experts by count; ties broken by id so runs are comparable."""
    return set(sorted(range(n_expert), key=lambda e: (-counts[e], e))[:n_res])


def hit_rate(resident_by_layer, target, layers):
    hits = total = 0
    for layer in layers:
        res = resident_by_layer[layer]
        for row in target.score_rows(layer):
            total += len(row)
            hits += sum(1 for e in row if e in res)
    return hits / total if total else 0.0


def rank_from(sources, layers, n_expert, n_res):
    """Pool the first halves of `sources` into one resident set per layer."""
    out = {}
    for layer in layers:
        c = Counter()
        for t in sources:
            c.update(t.first_half(layer))
        out[layer] = resident_set(c, n_expert, n_res)
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def sweep(traces, fracs, layers, n_expert, label):
    """The headline: hit rate at each resident fraction, per prompt and pooled.

    Per-prompt columns rank and score inside one prompt (first half / second
    half). `pooled` ranks on all five and scores on all five — the placement
    you would choose knowing this corpus. `unseen` ranks on four and scores the
    fifth, averaged over which one is held out, and is the only column that
    describes a placement meeting a workload it has not seen.
    """
    names = [t.name for t in traces]
    print()
    print("=" * 78)
    print("%s: hit rate with the top f%% of experts per layer resident" % label)
    print()
    print("  %-9s %s %9s %9s" % ("resident", "".join("%10s" % n[:9] for n in names),
                                 "pooled", "unseen"))
    print("  " + "-" * (9 + 10 * len(names) + 20))
    rows = []
    for f in fracs:
        n_res = max(1, int(round(f * n_expert)))
        per = [hit_rate(rank_from([t], layers, n_expert, n_res), t, layers) for t in traces]
        pooled_rank = rank_from(traces, layers, n_expert, n_res)
        pooled = sum(hit_rate(pooled_rank, t, layers) for t in traces) / len(traces)
        loo = sum(hit_rate(rank_from([x for x in traces if x is not t], layers, n_expert, n_res),
                           t, layers) for t in traces) / len(traces)
        rows.append((f, per, pooled, loo))
        print("  %8.0f%% %s %8.1f%% %8.1f%%"
              % (100 * f, "".join("%9.1f%%" % (100 * v) for v in per), 100 * pooled, 100 * loo))
    return rows


def run(traces, frac, layers, n_expert, k, label):
    n_res = max(1, int(round(frac * n_expert)))
    names = [t.name for t in traces]
    w = max(len(n) for n in names)

    print()
    print("=" * 78)
    print("%s - resident %d of %d experts per layer (%.0f%%)"
          % (label, n_res, n_expert, 100 * frac))

    # ---- per prompt: concentration -----------------------------------------
    print()
    print("per prompt")
    print("  %-*s %7s %8s %7s %7s %8s" % (w, "prompt", "pos", "H bits", "n@50", "used", "ovlap%"))
    from math import log2
    for t in traces:
        Hs, n50s, used, ov = [], [], [], []
        for layer in layers:
            c = t.counts(layer, 0, t.n_pos)
            tot = sum(c.values())
            Hs.append(-sum((v / tot) * log2(v / tot) for v in c.values() if v))
            acc = 0
            for i, v in enumerate(sorted(c.values(), reverse=True)):
                acc += v
                if acc >= 0.5 * tot:
                    n50s.append(i + 1)
                    break
            used.append(len(c))
            rows = t.sel[layer]
            ov.append(sum(len(set(rows[i - 1]) & set(rows[i])) / len(rows[i])
                          for i in range(1, len(rows))) / max(1, len(rows) - 1))
        print("  %-*s %7d %8.3f %7.1f %7.1f %7.1f%%"
              % (w, t.name, t.n_pos, sum(Hs) / len(Hs), sum(n50s) / len(n50s),
                 sum(used) / len(used), 100 * sum(ov) / len(ov)))
    print("  (uniform routing: H = %.3f, n@50 = %d, used = %d, ovlap = %.1f%%)"
          % (log2(n_expert), n_expert // 2, n_expert, 100.0 * k / n_expert))

    # ---- transfer matrix ----------------------------------------------------
    print()
    print("transfer: rank on the FIRST half of the row prompt, score on the SECOND")
    print("half of the column prompt. Equal ranking budget in every cell.")
    print()
    print("  %-*s %s" % (w, "rank \\ score", "".join("%9s" % n[:8] for n in names)))
    single = {t.name: rank_from([t], layers, n_expert, n_res) for t in traces}
    diag, off = [], []
    for src in traces:
        row = []
        for dst in traces:
            h = hit_rate(single[src.name], dst, layers)
            row.append(h)
            (diag if src is dst else off).append(h)
        print("  %-*s %s" % (w, src.name, "".join("%8.1f%%" % (100 * v) for v in row)))
    if not off:
        raise SystemExit("this study needs at least two traces; one prompt cannot "
                         "answer whether a placement transfers between prompts")
    d = sum(diag) / len(diag)
    o = sum(off) / len(off)
    print()
    print("  within-prompt (diagonal)  %.1f%%" % (100 * d))
    print("  cross-prompt  (off-diag)  %.1f%%   %+.1f points" % (100 * o, 100 * (o - d)))
    print("  retained                  %.0f%% of the within-prompt figure" % (100 * o / d if d else 0))

    # ---- what you would actually deploy ------------------------------------
    print()
    print("deployable rankings")
    pooled = rank_from(traces, layers, n_expert, n_res)
    ph = [hit_rate(pooled, t, layers) for t in traces]
    print("  pooled (rank on all %d, score each)      mean %.1f%%  min %.1f%%  max %.1f%%"
          % (len(traces), 100 * sum(ph) / len(ph), 100 * min(ph), 100 * max(ph)))
    loo = []
    for t in traces:
        others = [x for x in traces if x is not t]
        loo.append(hit_rate(rank_from(others, layers, n_expert, n_res), t, layers))
    print("  leave-one-out (rank on %d, score held-out) mean %.1f%%  min %.1f%%  max %.1f%%"
          % (len(traces) - 1, 100 * sum(loo) / len(loo), 100 * min(loo), 100 * max(loo)))
    print("  ^ the honest estimate for a placement chosen from a corpus and met")
    print("    with a workload it has not seen.")

    # ---- how different are the prompts -------------------------------------
    print()
    print("resident-set overlap between prompts (Jaccard, mean over layers)")
    print("  %-*s %s" % (w, "", "".join("%9s" % n[:8] for n in names)))
    pair = []
    for a in traces:
        row = []
        for b in traces:
            j = sum(jaccard(single[a.name][l], single[b.name][l]) for l in layers) / len(layers)
            row.append(j)
            if a is not b:
                pair.append(j)
        print("  %-*s %s" % (w, a.name, "".join("%8.2f " % v for v in row)))
    print("  mean off-diagonal %.2f: 1.00 means every prompt picks the same experts,"
          % (sum(pair) / len(pair)))
    print("  %.2f is what independent choices of %d of %d would give."
          % (n_res / (2 * n_expert - n_res), n_res, n_expert))

    # ---- by depth ------------------------------------------------------------
    print()
    print("cross-prompt transfer by depth (off-diagonal mean)")
    lo, hi = layers[0], layers[-1]
    bands = [("early %d-%d" % (lo, lo + 4), [l for l in layers if l <= lo + 4]),
             ("middle",                     [l for l in layers if lo + 4 < l <= (lo + hi) // 2]),
             ("late",                       [l for l in layers if l > (lo + hi) // 2])]
    for name, band in bands:
        if not band:
            continue
        vals = [hit_rate({l: single[a.name][l] for l in band}, b, band)
                for a in traces for b in traces if a is not b]
        print("  %-12s %d layers   %.1f%%" % (name, len(band), 100 * sum(vals) / len(vals)))
    return d, o, sum(loo) / len(loo)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--resident", type=float, default=23.0,
                    help="percent, used by --detail (default 23)")
    ap.add_argument("--sweep", default="5,10,23,33,50",
                    help="resident percentages for the headline table")
    ap.add_argument("--detail", action="store_true",
                    help="also print the transfer matrix, Jaccard overlaps and depth bands")
    args = ap.parse_args()

    mod = load_parser()
    traces = [Trace(p, mod) for p in args.traces]
    layers, n_expert, k = traces[0].layers, traces[0].n_expert, traces[0].k
    for t in traces[1:]:
        if t.layers != layers or t.n_expert != n_expert:
            raise SystemExit("traces disagree about the model")

    print("residency study: %d prompts, %d MoE layers, %d experts, %d used"
          % (len(traces), len(layers), n_expert, k))
    print("positions per prompt: %s" % ", ".join("%s=%d" % (t.name, t.n_pos) for t in traces))
    print("expected selections per (layer, expert) cell, per half: %.1f"
          % (traces[0].half * k / n_expert))

    fracs = [float(x) / 100.0 for x in args.sweep.split(",")]
    sweep(traces, fracs, layers, n_expert, "MEASURED")
    if args.detail:
        real = run(traces, args.resident / 100.0, layers, n_expert, k, "MEASURED, detail")

    for i, t in enumerate(traces):
        t.uniformise(20260810 + i)
    sweep(traces, fracs, layers, n_expert, "NULL (uniform draws, same shapes)")
    if args.detail:
        run(traces, args.resident / 100.0, layers, n_expert, k, "NULL, detail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
