#!/usr/bin/env python3
"""
N concurrent independent n=1 requests, all sharing the same prompt.

Tests whether the llama.cpp scheduler recognizes that arrival-time
concurrent requests with an identical prefix can share prompt processing
(and batch their generations across slots) instead of processing the
prompt N times.

Distinct from the other two primitives:

  bench_n.py          one HTTP request with n=N -- server batches
                      explicitly because the client told it N completions
                      are wanted up front.
  bench_sequential.py N back-to-back HTTP requests -- req #1 processes
                      the prompt, #2..N hit the prefix cache because the
                      cache is already populated by the time they arrive.
  bench_parallel.py   N HTTP requests issued *simultaneously* from N
                      client threads, gated by a threading.Barrier so
                      they reach the server within microseconds of each
                      other. Whether prompt processing is shared is what
                      the test is measuring.

Read each row's `cache_n` vs `prompt_n` to see who processed the prompt
and who hit the cache. The headline number is `batch_wall` (the wall
between the earliest start and the latest finish across all N requests).

Usage:
  python bench_parallel.py [max_tokens] [n]
  python bench_parallel.py --jsonl results/parallel.jsonl --n-prompt 16384 256 16

See README.md for server setup, --base-url / --n-prompt details, and the
/metrics caveat.
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import bench_lib

ctx, argv = bench_lib.setup(sys.argv[1:])
max_tokens = int(argv[0]) if len(argv) > 0 else 128
n          = int(argv[1]) if len(argv) > 1 else 16

if n > ctx.slots:
    sys.exit(f"n={n} exceeds total_slots={ctx.slots}: restart server with -np >= {n}")

print(f"run: max_tokens={max_tokens}  n_concurrent={n}  (all cold via /slots erase, "
      f"barrier-synchronized issue)")

ctx.clear_slots()

barrier = threading.Barrier(n)


def one(i):
    barrier.wait()
    rt0 = time.time()
    r = ctx.post(1, bench_lib.SEED + i, max_tokens)
    rwall = time.time() - rt0
    return i, rt0, rwall, r


m0 = ctx.metrics()
with ThreadPoolExecutor(max_workers=n) as ex:
    results = sorted(ex.map(one, range(n)), key=lambda x: x[0])
total_decodes = ctx.metrics()["n_decode_total"] - m0["n_decode_total"]

start_times = [rt0 for _, rt0, _, _ in results]
end_times   = [rt0 + rwall for _, rt0, rwall, _ in results]
batch_wall  = max(end_times) - min(start_times)

for i, rt0, rwall, r in results:
    t = r.get("timings", {})
    rprompt_s = t.get("prompt_ms", 0) / 1e3
    prompt_n  = int(t.get("prompt_n", -1))
    cache_n   = int(t.get("cache_n", -1))
    ctx.jdump({
        "req_idx": i, "n_total": n, "max_tokens": max_tokens,
        "batch_wall": batch_wall,
        "wall": rwall, "prompt_s": rprompt_s, "gen_s": rwall - rprompt_s,
        "gen_tok": max_tokens,
        "prompt_n": prompt_n, "cache_n": cache_n,
    })
    print(f"  req#{i:2d}  wall {rwall:6.2f}s  prompt {rprompt_s:5.2f}s  "
          f"prompt_n={prompt_n}  cache_n={cache_n}")

total_gen_tok = n * max_tokens
print(f"\n  batch wall          : {batch_wall:8.2f} s  "
      f"(max-end minus min-start across the {n} threads)")
print(f"  decode calls        : {total_decodes:8.0f}   for {total_gen_tok} gen tokens")
print(f"  tokens per decode   : {total_gen_tok/total_decodes:8.2f}   (~busy slots per step)")
print(f"  agg tok/s           : {total_gen_tok/batch_wall:.1f}")
