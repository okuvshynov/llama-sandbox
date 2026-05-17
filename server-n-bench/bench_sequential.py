#!/usr/bin/env python3
"""
N sequential n=1 requests against a llama.cpp server.

Req #1 is cold (slot caches are erased before the loop); #2..N reuse the
prompt KV via prefix cache -- the property under test. One JSONL row per
request (with `req_idx`), if --jsonl is set.

Usage:
  python bench_sequential.py [max_tokens] [n_requests]
  python bench_sequential.py --jsonl results/sequential.jsonl --n-prompt 16384 256 16
  python bench_sequential.py --base-url https://abc-8080.proxy.runpod.net/ 256 32

See README.md for server setup, --base-url / --n-prompt details, and the
/metrics caveat.
"""
import sys
import time

import bench_lib

ctx, argv = bench_lib.setup(sys.argv[1:])
max_tokens = int(argv[0]) if len(argv) > 0 else 128
n_total    = int(argv[1]) if len(argv) > 1 else 16

print(f"run: max_tokens={max_tokens}  n_requests={n_total}  (#1 cold, #2..N cache-warm)")

ctx.clear_slots()
m0 = ctx.metrics(); t0 = time.time()
for i in range(n_total):
    rt0 = time.time()
    r = ctx.post(1, bench_lib.SEED + i, max_tokens)
    rwall = time.time() - rt0
    t = r.get("timings", {})
    rprompt_s = t.get("prompt_ms", 0) / 1e3
    prompt_n  = int(t.get("prompt_n", -1))
    cache_n   = int(t.get("cache_n", -1))
    ctx.jdump({
        "req_idx": i, "n_total": n_total, "max_tokens": max_tokens,
        "wall": rwall, "prompt_s": rprompt_s, "gen_s": rwall - rprompt_s,
        "gen_tok": max_tokens,
        "prompt_n": prompt_n, "cache_n": cache_n,
    })
    print(f"  req#{i:2d}  wall {rwall:6.2f}s  prompt {rprompt_s:5.2f}s  "
          f"prompt_n={prompt_n}  cache_n={cache_n}")

total_dt = time.time() - t0
total_decodes = ctx.metrics()["n_decode_total"] - m0["n_decode_total"]
total_gen_tok = n_total * max_tokens
print(f"\n  total wall          : {total_dt:8.2f} s")
print(f"  decode calls        : {total_decodes:8.0f}   for {total_gen_tok} gen tokens")
print(f"  tokens per decode   : {total_gen_tok/total_decodes:8.2f}")
