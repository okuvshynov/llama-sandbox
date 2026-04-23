# validation-bench: Example configurations

Concrete invocations for the extracted per-provider scripts, with notes on why the flag values matter. All examples use `--task toml-1.0-cpp`; swap the task directory name as needed. More entries will be added here as additional providers are extracted from `validation_bench.py` into their own dedicated scripts.

## validation_bench_anthropic.py (Claude, native SDK)

### Sonnet 4.6, enabled thinking with explicit budget

```bash
python validation_bench_anthropic.py --task toml-1.0-cpp \
  --model claude-sonnet-4-6 \
  --thinking enabled --thinking-budget 30000 \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

`enabled` is the only thinking mode that enforces a hard cap on reasoning tokens. Here thinking consumes at most ~30k of the 100k `max_tokens` budget, leaving ~70k for the `submit` payload. Use this when you've seen adaptive mode run away — adaptive can burn the entire `max_tokens` on thinking before emitting a tool_use. Deprecated-but-functional on Sonnet/Opus 4.6; **returns HTTP 400 on Opus 4.7**.

### Sonnet 4.6, adaptive (model-decided depth)

```bash
python validation_bench_anthropic.py --task toml-1.0-cpp \
  --model claude-sonnet-4-6 --thinking adaptive \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

Adaptive lets the model decide how much to think. Observed to over-think on turn 0 — the full 100k can disappear into one thinking trace before any tool_use. The nudge logic recovers on turn 1 but wastes a turn. Safer for small smoke tests than large batches; consider `enabled` with a budget for production runs.

### Opus 4.7, adaptive (mandatory)

```bash
python validation_bench_anthropic.py --task toml-1.0-cpp \
  --model claude-opus-4-7 --thinking adaptive \
  --n-attempts 3 --max-turns 5 --max-tokens 100000
```

Opus 4.7 rejects `--thinking enabled` (HTTP 400) — adaptive is the only usable thinking mode. Anthropic's prose guide mentions a `thinking.adaptive.effort` field, but the server rejects it as `"Extra inputs are not permitted"` on both 4.6 and 4.7 as of 2026-04, so the flag is not plumbed.

### No thinking (baseline)

```bash
python validation_bench_anthropic.py --task toml-1.0-cpp \
  --model claude-sonnet-4-6 --thinking disabled \
  --n-attempts 5 --max-turns 5 --max-tokens 32768
```

When thinking is disabled the script uses `tool_choice={"type":"any"}` to force a tool call. Useful for measuring how much extended thinking contributes over the base model.

## validation_bench_moonshot.py (Moonshot/Kimi, native API)

### K2.6 thinking with Preserved Thinking

```bash
python validation_bench_moonshot.py --task toml-1.0-cpp \
  --model kimi-k2.6 --mode thinking \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

Default Moonshot configuration. `thinking.keep="all"` (Preserved Thinking) is K2.6-only and enabled by default; disable with `--no-preserve-thinking`. Moonshot rejects `tool_choice="required"` when thinking is on, so the script sends no `tool_choice` and relies on the nudge loop for the occasional no-tool-call turn.

### K2.6 instant (no thinking, fastest)

```bash
python validation_bench_moonshot.py --task toml-1.0-cpp \
  --model kimi-k2.6 --mode instant \
  --n-attempts 5 --max-turns 5 --max-tokens 32768
```

Fastest K2.6 path. Because `thinking.type="disabled"` is explicitly sent, `tool_choice="required"` is accepted — the model is forced to use `submit`. Default temperature is 0.6 for instant (vs 1.0 for thinking) per Moonshot's benchmarking guide.

### K2.5 thinking

```bash
python validation_bench_moonshot.py --task toml-1.0-cpp \
  --model kimi-k2.5 --mode thinking \
  --n-attempts 3 --max-turns 5 --max-tokens 100000
```

Same shape as K2.6 but `thinking.keep` is omitted (K2.6-only field; undocumented for K2.5). The no-tool-call failure mode is more frequent on K2.5 — the nudge loop is load-bearing here.

## Shared flags

- `--docker-timeout 600` — cap on `docker run -d` when starting the sandbox. Distinct from `--timeout` (API client). Both default to 600s.
- `--data-dir` / `VB_DATA_DIR` — directory for `messages.json` and per-submission files. Default `~/.vb-data/`.
- `--results-dir` — directory holding `results.jsonl`. Default `results/`.
- `--max-turns` — conversation turns per attempt. A nudge retry (when the model replies without calling `submit`) consumes one turn from this budget.
- `--slug` — override the auto-derived results slug when you want to segment data (e.g., re-running with a different system prompt).

## Failure-handling behavior

- Per-submission errors (`compile_error`, `compile_timeout`) become rows in `results.jsonl` with an `"error"` field instead of a confusion matrix.
- Mid-stream API errors that hit *after* some successful submissions produce both: the per-turn rows go to `results.jsonl` as normal, and the api_error root cause goes to `failures.jsonl`. Graded turns are not lost to a transient 500.
- When no submissions exist and an api_error or no-tool-use condition occurred, only `failures.jsonl` is written.
