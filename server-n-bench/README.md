# server-n-bench

One-shot perf primitives for the llama.cpp server. Each script makes a
single primitive request, optionally appends a JSONL row, and exits.
Sweeps and repetitions are driven by shell loops; medians and tables are
computed post-hoc from the JSONL.

Currently two scripts; a third (`bench_parallel.py`) is planned.

- **`bench_n.py`** — one cold parallel `n=N` request. Tests the
  OpenAI-style `n` parameter (the server processes the prompt once and
  batches N generations across slots). Parsed at
  `tools/server/server-task.cpp:262` as an alias for `n_cmpl`.
- **`bench_sequential.py`** — N sequential `n=1` requests, cold req #1
  then #2..N reuse the prompt KV via prefix cache (no slot clear between
  them). Tests the cache-hit path.
- **`bench_parallel.py`** *(future)* — N independent concurrent clients
  sharing a prefix, to test how the scheduler aggregates simultaneous
  requests at runtime.

Shared logic lives in `bench_lib.py` (prompt building, `/props`,
`/metrics`, `clear_slots`, `post`, `jdump`, common arg parsing).

## Server setup

```
... -np <max n you want to test> --metrics --slot-save-path /tmp/llama-slots
```

`--slot-save-path` is required because it gates the `/slots/{id}?action=erase`
endpoint these scripts use to guarantee cold runs. The path itself is not
written by `erase`, but the flag must be present.

## Cold runs via `/slots/{id}?action=erase`

Both scripts call `POST /slots/{id}?action=erase` for every slot before
issuing the first request, instead of varying the prompt with a per-run
tag. There is no `-1`-broadcast form on this endpoint — the scripts
iterate over `0..total_slots-1`.

- `bench_n.py` — erase all slots, then one `n=N` request.
- `bench_sequential.py` — erase all slots once, then N back-to-back `n=1`
  requests *without* clearing between them; that's the property under test.

## Run

```
# Single parallel run (max_tokens=256, n=16)
python bench_n.py 256 16

# Same, with JSONL row appended
python bench_n.py --jsonl results/sweep.jsonl 256 16

# A sweep: shell-driven, REPS rows per n value
for n in 1 2 4 8 16 32 64; do
  for r in 1 2 3; do
    python bench_n.py --jsonl results/sweep.jsonl --n-prompt 16384 256 $n
  done
done

# Sequential cache-warm test (16 requests, one row per request with req_idx)
python bench_sequential.py --jsonl results/sequential.jsonl 256 16

# Target a remote llama-server (e.g. a runpod tunnel; HTTPS works as-is)
python bench_n.py --base-url https://abc-8080.proxy.runpod.net/ 256 16
```

## Common flags

`--base-url URL` overrides the default `http://127.0.0.1:8080`. Trailing
slash is stripped. HTTPS works out of the box (urllib's system CA bundle);
no extra flag needed for valid certs. The chosen URL is recorded in each
JSONL row as `base_url` so concatenated sweeps from local + remote servers
stay disambiguated.

`--n-prompt N` builds the prompt by reading a sibling seed corpus
(default `seed_corpus.cpp` — canonical algorithms in C++; override with
the `BENCH_N_SEED` env var pointing at any text file), POSTing it to
`/tokenize`, repeating the resulting token list to reach `N` tokens,
truncating, and POSTing back through `/detokenize`. Code (rather than
prose) is the default so that MoE expert routing matches what would
happen on a real coding workload — for dense models the choice of seed
doesn't matter, but for MoE the same `N` tokens of prose vs code can fire
substantially different expert sets and produce different throughput
numbers. The chat template wraps the content on the wire, so the actual
`prompt_n` in each JSONL row is a few tokens larger than `N`; both are
preserved (`n_prompt_target` and `seed_corpus` in row metadata vs
`prompt_n` in row body). Without `--n-prompt`, a built-in
technical-writer prompt is used.

`--jsonl PATH` accepts any position. It appends one row per individual
request (no in-process median collapse), so shell-driven reps preserve
the full sample for variance work.

Every row carries server metadata from `/props` (`model`, `model_path`,
`build_info`, `total_slots`) plus a `ts` epoch timestamp, so concatenated
runs across builds/models stay self-describing. `build_info` is the
llama.cpp `b<number>-<commit>` string (set in
`common/build-info.cpp.in`).

The `device` field records the compute device. llama.cpp's HTTP API does
not expose this (no `/devices` endpoint; `ggml_backend_dev_*` is internal
only), so the value comes from the `BENCH_N_DEVICE` env var — set it to
whatever short label disambiguates this run (`"M2 Ultra"`, `"RTX 4090"`,
`"Threadripper 7970X"`, etc.). Defaults to `"unknown"` when unset.

## Caveat — `/metrics prompt_tokens_total` is misleading for n>1

For an `n>1` request the server's `copy_state_to()` copies the parent
slot's `n_prompt_tokens_processed` counter into every child slot
(`tools/server/server-context.cpp:562`), so the Prometheus counter reports
roughly `n × actual`. Prompt is physically processed once. These scripts
read real prompt cost from each response's `timings` block;
`/metrics n_decode_total` is correct and used here for the
tokens-per-decode metric.

## Existing data

`results/sweep.jsonl` was produced by an earlier monolithic `bench_n.py`
that combined a sweep loop with median computation. Those rows carry
`mode: "sweep"`, `rep`, and `reps` fields; new rows from the current
`bench_n.py` do not — script identity is implicit in the filename and
shell-loop drives the reps. The old rows still parse cleanly alongside
new ones in pandas/jq (extra keys are tolerated).
