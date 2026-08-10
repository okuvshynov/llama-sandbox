#!/usr/bin/env python3
"""What if the resident expert set were a cache instead of a fixed placement?

`ROUTING.md` measures *static* placements: pick the top f% of experts per layer
once and never change them. PLAN.md then asserts "static, not LRU — PCIe is far
slower than DRAM, so cache refill never pays", which is a cost argument whose
hit-rate half was never measured. This measures it, from the traces already on
disk: a trace is exactly the access sequence a cache would see.

    python cache_sim.py results/residency/*.trace [--resident 10,23,50]

Each of the 75 MoE layers gets its own cache of round(f% x 256) experts,
starting with a random resident set — so the run begins cold and warms up, and
the warmup curve is reported rather than averaged away.

Policies: `lru`, `lfu`, `random` (eviction victim chosen uniformly) and
`static`, which never evicts and is the fixed-random-placement baseline.

**Read the cost column, not just the hit rate.** Under a static placement a
miss is a DRAM read, which is what happens today. Under a cache a miss is a
DRAM read *and* a PCIe upload to install the expert, so misses are strictly
more expensive than they are now. The simulation reports GB/token of induced
PCIe traffic next to the hit rate, because a cache that hits more while
uploading 10 GB/token has not helped.
"""

import argparse
import importlib.util
import os
import random
import sys
from collections import OrderedDict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERT_MB = 31.5          # gate+up+down for one expert, measured by nano-bench
DRAM_GBPS = 75.4          # nano-bench, the model's own access pattern
PCIE_GBPS = 13.0          # PCIe 3.0 x16 practical; the Vega II dies sit on one
ROUTED_GB = 18.90         # routed-expert bytes read per token today
TOKEN_S   = 0.5157        # measured seconds per token, whole model


def load_traces(paths):
    spec = importlib.util.spec_from_file_location(
        "expert_stats_mod", os.path.join(HERE, "expert_stats.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for p in paths:
        meta, tokens, layers, sel = mod.parse_trace(p)
        out.append((os.path.splitext(os.path.basename(p))[0], meta, layers, sel, len(tokens)))
    return out


def simulate_layer(rows, n_expert, cap, policy, rnd, buckets, per_bucket):
    """One layer's access sequence. Returns (hits, accesses, fetches) and fills
    `buckets` with per-position-band hit counts."""
    resident = OrderedDict()                       # expert -> None, ordered by recency
    for e in rnd.sample(range(n_expert), cap):     # cold start: arbitrary contents
        resident[e] = None
    freq = Counter({e: 0 for e in resident})       # for lfu

    hits = acc = fetches = 0
    for pos, row in enumerate(rows):
        b = min(len(buckets) - 1, pos // per_bucket)
        for e in row:
            acc += 1
            if e in resident:
                hits += 1
                buckets[b][0] += 1
                if policy == "lru":
                    resident.move_to_end(e)
                elif policy == "lfu":
                    freq[e] += 1
            else:
                fetches += 1
                if policy == "static":
                    pass                            # never install: fixed placement
                else:
                    if len(resident) >= cap:
                        if policy == "lru":
                            victim, _ = resident.popitem(last=False)
                        elif policy == "lfu":
                            victim = min(resident, key=lambda x: freq[x])
                            del resident[victim]
                        else:                       # random
                            victim = rnd.choice(list(resident))
                            del resident[victim]
                        freq.pop(victim, None)
                    resident[e] = None
                    freq[e] = 1
            buckets[b][1] += 1
    return hits, acc, fetches


def run(traces, fracs, policies, n_bands, seed):
    name0, meta0 = traces[0][0], traces[0][1]
    n_expert = int(meta0["n_expert"])
    k = int(meta0["n_expert_used"])

    print("cache simulation: %d prompts, %d experts/layer, %d used per token"
          % (len(traces), n_expert, k))
    print("one cache per layer, cold start with a random resident set, "
          "%.1f MB per expert" % EXPERT_MB)
    print()

    for frac in fracs:
        cap = max(1, int(round(frac * n_expert)))
        print("=" * 78)
        print("resident %d of %d experts per layer (%.0f%%)" % (cap, n_expert, 100 * frac))
        print()
        print("  %-8s %9s %9s %8s %8s %8s   %s"
              % ("policy", "hit rate", "steady", "DRAM GB", "PCIe GB", "s/token",
                 "hit rate over the run"))
        for policy in policies:
            tot_h = tot_a = tot_f = 0
            bands = [[0, 0] for _ in range(n_bands)]
            for (_, meta, layers, sel, n_pos) in traces:
                per_band = max(1, (n_pos + n_bands - 1) // n_bands)
                rnd = random.Random(seed)
                for layer in layers:
                    h, a, f = simulate_layer(sel[layer], n_expert, cap, policy, rnd,
                                             bands, per_band)
                    tot_h += h
                    tot_a += a
                    tot_f += f
            n_tokens = sum(t[4] for t in traces)
            # A miss means the expert is read from DRAM either way. A cache
            # additionally uploads it across PCIe to install it; a static
            # placement does not, which is the whole trade.
            dram = tot_f / n_tokens * EXPERT_MB / 1000.0
            pcie = 0.0 if policy == "static" else dram
            # Time for the routed half only: DRAM misses plus PCIe installs,
            # plus the resident hits which are served from VRAM and ~free.
            secs = dram / DRAM_GBPS + pcie / PCIE_GBPS
            steady = bands[-1][0] / bands[-1][1] if bands[-1][1] else 0.0
            curve = " ".join("%4.0f" % (100.0 * b[0] / b[1]) if b[1] else "   -" for b in bands)
            print("  %-8s %8.1f%% %8.1f%% %8.2f %8.2f %8.3f   %s"
                  % (policy, 100.0 * tot_h / tot_a, 100.0 * steady, dram, pcie, secs, curve))
        print()
        print("  bands are tenths of the run. s/token counts the routed experts only:")
        print("  DRAM misses at %.0f GB/s plus PCIe installs at %.0f GB/s. For scale, all"
              % (DRAM_GBPS, PCIE_GBPS))
        print("  %.2f GB of routed experts from DRAM today costs %.3f s/token, and the whole"
              % (ROUTED_GB, ROUTED_GB / DRAM_GBPS))
        print("  token including the dense trunk costs %.3f s." % TOKEN_S)
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--resident", default="10,23,50", help="percentages, default 10,23,50")
    ap.add_argument("--policies", default="static,random,lru,lfu")
    ap.add_argument("--bands", type=int, default=10, help="warmup-curve buckets")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    traces = load_traces(args.traces)
    run(traces,
        [float(x) / 100.0 for x in args.resident.split(",")],
        args.policies.split(","),
        args.bands, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
