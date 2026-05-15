# server-n-bench

Performance test for the llama.cpp server's OpenAI-style `n` parameter
(generate N completions for a single prompt; the server processes the prompt
once and batches the N generations in parallel slots).

Two modes:

- **A/B**: one parallel `n=N` request vs N sequential `n=1` requests (same total
  token count, both with prompt KV reused, so only the generation cost differs).
- **Sweep**: scaling curve over a list of `n` values, with reps + median for noise.

The OpenAI `n` is parsed at `tools/server/server-task.cpp:262` (alias for the
llama.cpp-native `n_cmpl`). Server must be started with `-np <max-n> --metrics`.

## Run

```
# A/B test (default: max_tokens=128, n=16)
python3 bench_n.py [max_tokens]

# Sweep with 3 reps per n
python3 bench_n.py --sweep 128 1,2,4,8,16,32,64 3
```

## Caveat — `/metrics prompt_tokens_total` is misleading for n>1

For an `n>1` request the server's `copy_state_to()` copies the parent slot's
`n_prompt_tokens_processed` counter into every child slot
(`tools/server/server-context.cpp:562`), so the Prometheus counter reports
roughly `n × actual`. Prompt is physically processed once. This script reads
real prompt cost from each response's `timings` block; `/metrics n_decode_total`
is correct and used here for the tokens-per-decode metric.
