# server-n-bench

Performance test for the llama.cpp server's OpenAI-style `n` parameter
(generate N completions for a single prompt; the server processes the prompt
once and batches the N generations in parallel slots).

Two modes:

- **A/B**: one parallel `n=N` request vs N sequential `n=1` requests (same total
  token count, both with prompt KV reused, so only the generation cost differs).
- **Sweep**: scaling curve over a list of `n` values, with reps + median for noise.

The OpenAI `n` is parsed at `tools/server/server-task.cpp:262` (alias for the
llama.cpp-native `n_cmpl`).

Start the server with:

```
... -np <max-n> --metrics --slot-save-path /tmp/llama-slots
```

`--slot-save-path` is required because it gates the `/slots/{id}?action=erase`
endpoint that this script uses to guarantee cold runs (see below). The path
itself is not written by `erase`, but the flag must be present.

## Cold runs via `/slots/{id}?action=erase`

Both modes need each measured run to start with an empty KV cache so we are
actually measuring prompt-processing + generation, not cache hits. The script
calls `POST /slots/{id}?action=erase` for every slot before each cold run
(setup), instead of varying the prompt with a per-run tag. There is no
`-1`-broadcast form on this endpoint — the script iterates over `0..total_slots-1`.

- **Sweep**: erase all slots before every individual run (every `(n, rep)`).
- **A/B parallel**: erase all slots, then issue one `n=N` request.
- **A/B sequential**: erase all slots once, then run N back-to-back `n=1`
  requests *without* clearing between them — that's the property under test
  (req #1 processes the prompt; #2..N reuse it via prefix cache).

## Run

```
# A/B test (default: max_tokens=128, n=16)
python3 bench_n.py [max_tokens]

# Sweep with 3 reps per n
python3 bench_n.py --sweep 128 1,2,4,8,16,32,64 3

# Same sweep, also append every individual run to a JSONL file
python3 bench_n.py --jsonl results/sweep.jsonl --sweep 128 1,2,4,8,16,32,64 3

# Sweep with a longer prompt (~2048 tokens, tokenized via /tokenize)
python3 bench_n.py --n-prompt 2048 --sweep 128 1,2,4,8,16,32,64 3
```

`--n-prompt N` builds the prompt by reading a sibling seed corpus (default
`seed_corpus.cpp` -- a few canonical algorithms in C++; override with the
`BENCH_N_SEED` env var pointing at any text file), POSTing it to `/tokenize`,
repeating the resulting token list to reach `N` tokens, truncating, and
POSTing back through `/detokenize`. Code (rather than prose) is the default
so that MoE expert routing matches what would happen on a real coding
workload — for dense models the choice of seed doesn't matter, but for MoE
the same `N` of tokens of prose vs code can fire substantially different
expert sets and produce different throughput numbers. The chat template
wraps it on the wire so the actual `prompt_n` in each jsonl row is a few
tokens larger than `N`; both numbers are preserved (`n_prompt_target` and
`seed_corpus` in row metadata vs `prompt_n` in row body). Without
`--n-prompt`, the original made-up technical-writer prompt is used.

`--jsonl PATH` accepts any position. It appends one row per individual run
(not per-`n` median), so `reps>1` preserves the full sample for variance work.
In sweep mode that's `REPS` rows per `n` value with `mode=sweep`; in A/B mode
it's one `mode=ab_parallel` row plus N `mode=ab_sequential` rows (one per
request, with `req_idx`).

Every row also carries server metadata from `/props` (`model`, `model_path`,
`build_info`, `total_slots`) plus a `ts` epoch timestamp, so concatenated
runs across builds/models stay self-describing. `build_info` is the
llama.cpp `b<number>-<commit>` string (set in `common/build-info.cpp.in`).

The `device` field records the compute device. llama.cpp's HTTP API does
not expose this (no `/devices` endpoint; `ggml_backend_dev_*` is internal
only), so the value comes from the `BENCH_N_DEVICE` env var — set it to
whatever short label disambiguates this run (`"M2 Ultra"`, `"RTX 4090"`,
`"Threadripper 7970X"`, etc.). Defaults to `"unknown"` when unset.

## Caveat — `/metrics prompt_tokens_total` is misleading for n>1

For an `n>1` request the server's `copy_state_to()` copies the parent slot's
`n_prompt_tokens_processed` counter into every child slot
(`tools/server/server-context.cpp:562`), so the Prometheus counter reports
roughly `n × actual`. Prompt is physically processed once. This script reads
real prompt cost from each response's `timings` block; `/metrics n_decode_total`
is correct and used here for the tokens-per-decode metric.
