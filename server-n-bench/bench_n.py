#!/usr/bin/env python3
"""
One cold parallel n=N completion request against a llama.cpp server.

The server processes the prompt once and batches N generations across slots
-- this is the cost of the OpenAI-style `n` parameter (alias for
`n_cmpl`, parsed at tools/server/server-task.cpp:262). One JSONL row out
per invocation, if --jsonl is set.

Usage:
  python bench_n.py [max_tokens] [n]
  python bench_n.py --jsonl results/sweep.jsonl --n-prompt 16384 256 16
  python bench_n.py --base-url https://abc-8080.proxy.runpod.net/ 256 32

Sweeps and repetitions are driven externally, e.g.

  for n in 1 2 4 8 16 32; do
    for r in 1 2 3; do
      python bench_n.py --jsonl results/sweep.jsonl --n-prompt 16384 256 $n
    done
  done

See README.md for server setup, --base-url / --n-prompt details, and the
/metrics caveat.
"""
import sys
import time

import bench_lib

ctx, argv = bench_lib.setup(sys.argv[1:])
max_tokens = int(argv[0]) if len(argv) > 0 else 128
n          = int(argv[1]) if len(argv) > 1 else 16

if n > ctx.slots:
    sys.exit(f"n={n} exceeds total_slots={ctx.slots}: restart server with -np >= {n}")

print(f"run: max_tokens={max_tokens}  n={n}  (cold via /slots erase)")

ctx.clear_slots()
m0 = ctx.metrics(); t0 = time.time()
r = ctx.post(n, bench_lib.SEED, max_tokens)
wall = time.time() - t0
decodes = ctx.metrics()["n_decode_total"] - m0["n_decode_total"]
t = r.get("timings", {})
prompt_s = t.get("prompt_ms", 0) / 1e3
gen_s    = wall - prompt_s
gen_tok  = n * max_tokens

row = dict(
    n=n, max_tokens=max_tokens,
    wall=wall, prompt_s=prompt_s, gen_s=gen_s, gen_tok=gen_tok,
    decodes=decodes,
    prompt_n=t.get("prompt_n", 0), cache_n=t.get("cache_n", 0),
    agg_tps=gen_tok / gen_s,
    per_stream_tps=max_tokens / gen_s,
    tok_per_decode=gen_tok / decodes,
)
ctx.jdump(row)

print(f"  wall                : {wall:8.2f} s")
print(f"  prompt              : {row['prompt_n']:.0f} proc / {row['cache_n']:.0f} cached  in {prompt_s:.2f} s")
print(f"  decode calls        : {decodes:8.0f}   for {gen_tok} gen tokens")
print(f"  tokens per decode   : {row['tok_per_decode']:8.2f}   (~busy slots per step)")
print(f"  gen-only time       : {gen_s:8.2f} s  -> {row['agg_tps']:.1f} tok/s agg, {row['per_stream_tps']:.2f} tok/s per stream")
