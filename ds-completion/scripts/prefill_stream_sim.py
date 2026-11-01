#!/usr/bin/env python3
"""Prompt processing: compute the non-resident routed experts on the CPU, or stream them to the
GPUs? (PLAN "Prompt processing", P1 -> P2). Reads --log-routing files of long prompts and models,
per chunk of B prompt tokens and per layer, three policies from measured unit costs:

    cpu     every non-resident expert on the CPU (today)
    stream  every non-resident expert loaded to a GPU and computed there
    hybrid  the non-resident experts with the most tokens streamed, the rest on the CPU, the split
            chosen to balance the CPU's time against the loads' (they run in parallel)

    ~/venvs/ds-ref/bin/python scripts/prefill_stream_sim.py data/routing/prefill/*.bin \\
        [--placement placements/study30.txt --resident 6578] [--canon]

Unit costs (README steps 30, 37, 41; sclk 2, 16 threads; --cpu-scale and --load-rate adjust them):
  CPU, one expert with t tokens of the chunk: fast 0.24 + 0.070 t ms, canonical 0.20 + 0.135 t ms
      (linear fits to bench-cpu-moe --batch: one weight read, then per token)
  stream: 0.70 ms of host-link time per expert (1,437 loads/s over the 4 dies), whatever t
  GPU: 0.018 ms per expert-token on one die (fp4 kernel at many tokens), the 4 dies in parallel
The resident set is approximated by the placement's top pairs (the runtime's count, printed at
startup: "N experts ... on GPUs"). Today's prompt path splits a chunk into two halves (the CPU/GPU
overlap), so the "cpu" policy is modelled on halves; streaming needs no halves.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_stats import read_log  # noqa: E402

LOAD_MS = 1000.0 / 1437
GPU_MS = 0.018 / 4  # per expert-token, 4 dies
CPU_SCALE = 1.0      # --cpu-scale: the runtime's CPU cost over the benchmark's (measured 1.65)


def cpu_ms(t, canon):
    return CPU_SCALE * ((0.20 + 0.135 * t) if canon else (0.24 + 0.070 * t))


def read_placement(path, n):
    pairs = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        L, e = line.split()[:2]
        pairs.append((int(L), int(e)))
    return set(pairs[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--placement", default=str(Path(__file__).resolve().parent.parent / "placements/study30.txt"))
    ap.add_argument("--resident", type=int, default=6578)
    ap.add_argument("--canon", action="store_true")
    ap.add_argument("--chunks", default="256,512,1024,2048,4096")
    ap.add_argument("--cpu-scale", type=float, default=1.0,
                    help="runtime CPU cost / benchmark cost (README step 42: 1.65 measured on readme24k)")
    ap.add_argument("--load-rate", type=float, default=1437, help="expert loads per second over the 4 dies")
    a = ap.parse_args()
    global CPU_SCALE, LOAD_MS
    CPU_SCALE, LOAD_MS = a.cpu_scale, 1000.0 / a.load_rate
    resident = read_placement(a.placement, a.resident)
    print(f"# resident: the placement's top {a.resident} pairs; CPU costs {'canonical' if a.canon else 'fast'} x {a.cpu_scale}; "
          f"{a.load_rate:.0f} loads/s")
    for path in a.logs:
        data, n_layer, _ = read_log(path)
        ids = data["ids"][data["prompt"] == 1].astype(np.int64)  # [T, L, k]
        n = len(ids)
        res = np.zeros((n_layer, 384), bool)
        for L, e in resident:
            if L < n_layer:
                res[L, e] = True
        print(f"\n## {Path(path).name}: {n} prompt tokens")
        print(f"{'B':>5} {'nonres/layer':>12} {'tok/nonres':>10} {'>=4':>5} {'>=8':>5}   ms per prompt token:"
              f" {'cpu':>6} {'stream':>7} {'hybrid':>7} (streamed) {'gpu':>5}")

        def counts(block, L):  # tokens per non-resident expert of layer L in a block of tokens
            e = block[:, L, :].ravel()
            e = e[e >= 0]
            u, c = np.unique(e, return_counts=True)
            return np.sort(c[~res[L, u]])[::-1]

        for B in [int(x) for x in a.chunks.split(",")]:
            tot = Counter()
            for c0 in range(0, n, B):
                chunk = ids[c0:c0 + B]
                if len(chunk) < B // 2 and c0 > 0:
                    continue  # a short tail chunk is not representative
                tot["tok"] += len(chunk)
                for L in range(n_layer):
                    nonres = counts(chunk, L)
                    tot["layers"] += 1
                    tot["nonres"] += len(nonres)
                    tot["pairs"] += int(nonres.sum())
                    tot["ge4"] += int((nonres >= 4).sum())
                    tot["ge8"] += int((nonres >= 8).sum())
                    h = len(chunk) // 2  # today: two halves, each with its own counts
                    for half in (chunk[:h], chunk[h:]) if h else (chunk,):
                        hc = counts(half, L)
                        tot["cpu"] += float(np.sum(cpu_ms(hc, a.canon)))
                    tot["stream"] += max(len(nonres) * LOAD_MS, nonres.sum() * GPU_MS)
                    # hybrid: stream the k most-used, the rest on the CPU; the best k
                    cpu_rest = np.concatenate([np.cumsum(cpu_ms(nonres, a.canon)[::-1])[::-1], [0.0]])
                    k = np.arange(len(nonres) + 1)
                    tm = np.maximum(k * LOAD_MS, cpu_rest)
                    tot["hybrid"] += float(tm.min())
                    tot["streamed"] += int(k[tm.argmin()])
                    tot["gpu"] += nonres.sum() * GPU_MS
            per_tok = lambda key: tot[key] / tot["tok"]  # noqa: E731
            nr = max(1, tot["nonres"])
            print(f"{B:5d} {tot['nonres'] / tot['layers']:12.0f} {tot['pairs'] / nr:10.1f} {tot['ge4'] / nr:5.0%} "
                  f"{tot['ge8'] / nr:5.0%}   {'':19s} {per_tok('cpu'):6.1f} {per_tok('stream'):7.1f} "
                  f"{per_tok('hybrid'):7.1f} ({tot['streamed'] / nr:5.0%})    {per_tok('gpu'):5.2f}")

if __name__ == "__main__":
    main()
