#!/usr/bin/env python3
"""Where an RPC decode step's time goes, from `nano-glm --moe-log`.

    python moe_log_stats.py results/rpc.jsonl
    python moe_log_stats.py results/rpc.jsonl --per-layer

One record per RPC: layer, n_tokens, bytes each way, the client's round trip,
and the server's own four stages. Two things make it worth reading rather than
just summing:

**n_tokens separates the regimes.** A prompt-shaped run issues 40 calls carrying
the whole prompt and then 40 more per generated token carrying one row. Those
differ by two orders of magnitude in payload and are bound by different things,
so a mean over all of them describes neither. Records with n_tokens == 1 are
decode steps; anything wider is prefill.

**rtt minus the server's stages is the part nobody owns.** The client measures
the round trip; the server reports parse, route, compute and serialize inside
it. The difference is socket time plus scheduling — and on loopback, where both
processes want every core, "scheduling" includes each side waiting for the other
to give the CPU back. A large remainder here is the signal that the two
processes are fighting rather than that the network is slow.

Stdlib only, as everything else here is.
"""

import argparse
import json
import sys

STAGES = ("srv_parse_us", "srv_route_us", "srv_compute_us", "srv_serialize_us")


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(name, recs, n_layers_hint=None):
    if not recs:
        return
    n = len(recs)
    rtt = sorted(r["rtt_us"] for r in recs)
    tot = {k: sum(r[k] for r in recs) for k in STAGES}
    srv = sum(tot.values())
    rtt_sum = sum(rtt)
    other = rtt_sum - srv

    # calls per token: every layer that routes issues one per chunk, so the
    # number of distinct layers is the multiplier from "per call" to "per token".
    layers = n_layers_hint or len({r["layer"] for r in recs})
    steps = n / layers if layers else 1

    print("\n=== %s: %d calls over %d layers (%.0f steps)" % (name, n, layers, steps))
    print("  rtt        p50 %7.1f us   p90 %7.1f us   max %8.1f us"
          % (rtt[n // 2], rtt[min(int(n * 0.9), n - 1)], rtt[-1]))
    print("  payload    %7.1f KB out  %7.1f KB in   (per call)"
          % (sum(r["bytes_out"] for r in recs) / n / 1024.0,
             sum(r["bytes_in"] for r in recs) / n / 1024.0))
    print("  %-12s %10s %9s %9s" % ("", "per call", "per step", "share"))
    rows = [(k[4:-3], tot[k]) for k in STAGES] + [("network+sched", other)]
    for label, us in rows:
        print("  %-12s %9.1f us %8.2f ms %8.1f%%"
              % (label, us / n, us / steps / 1000.0, 100.0 * us / rtt_sum))
    print("  %-12s %9.1f us %8.2f ms %8.1f%%"
          % ("TOTAL rtt", rtt_sum / n, rtt_sum / steps / 1000.0, 100.0))


def per_layer(recs):
    by = {}
    for r in recs:
        by.setdefault(r["layer"], []).append(r["rtt_us"])
    print("\n  layer   calls   rtt p50 us")
    for layer in sorted(by):
        v = sorted(by[layer])
        print("  %5d   %5d   %9.1f" % (layer, len(v), v[len(v) // 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--per-layer", action="store_true",
                    help="decode rtt per layer — a layer that is consistently "
                         "slower is a routing or residency effect, not latency")
    args = ap.parse_args()

    recs = load(args.path)
    if not recs:
        raise SystemExit("%s: no records" % args.path)

    prefill = [r for r in recs if r["n_tokens"] > 1]
    decode = [r for r in recs if r["n_tokens"] == 1]

    summarize("prefill (n_tokens > 1)", prefill)
    summarize("decode  (n_tokens = 1)", decode)

    if args.per_layer and decode:
        per_layer(decode)

    print("\n'per step' is per prompt chunk for prefill and per generated token\n"
          "for decode: one call per routing layer, summed. 'network+sched' is\n"
          "rtt minus the server's own four stages — socket time plus whatever\n"
          "the two processes cost each other for the CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
