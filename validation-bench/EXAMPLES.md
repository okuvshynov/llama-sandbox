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

## validation_bench_fireworks.py (Fireworks AI, OpenAI-compat endpoint)

The `fireworks-ai` Python SDK is still alpha and auto-generated; its inference path is just the OpenAI-compat endpoint anyway. This script uses the OpenAI SDK pointed at `https://api.fireworks.ai/inference/v1` and plumbs Fireworks-specific knobs (reasoning_effort, thinking budget, reasoning_history, top_k/min_p/repetition_penalty, prompt_cache_key) through `extra_body`. Bare model names are auto-prefixed with `accounts/fireworks/models/`.

### Default reasoning (server decides)

```bash
python validation_bench_fireworks.py --task toml-1.0-cpp \
  --model minimax-m2p7 \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

With both `--reasoning-effort` and `--thinking-budget` omitted, the request carries no reasoning fields and Fireworks applies the model's server-side default — reasoning-capable models typically reason at `medium`, non-reasoning models behave normally. The results row only records what was sent, so nothing about reasoning lands in `sampling_params`; pass `--reasoning-effort` explicitly when provenance matters.

### Explicit reasoning_effort

```bash
python validation_bench_fireworks.py --task toml-1.0-cpp \
  --model glm-4p6 --reasoning-effort high \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

Records `reasoning_effort=high` in `sampling_params`. The slug picks up the suffix (`fireworks-glm-4p6-high`) so runs at different effort levels stay segmented without `--slug` overrides.

### Thinking budget (Anthropic-style manual cap)

```bash
python validation_bench_fireworks.py --task toml-1.0-cpp \
  --model kimi-k2-thinking --thinking-budget 16384 --reasoning-history preserved \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

`thinking.budget_tokens=16384` hard-caps reasoning; mutually exclusive with `--reasoning-effort` (CLI enforces). `--reasoning-history preserved` echoes prior reasoning back into later turns — symmetric with Moonshot's Preserved Thinking and useful for multi-turn tool use on Kimi-K2-Thinking. Note: Kimi-K2-Thinking has reasoning force-enabled server-side, so disabling it isn't possible regardless of what you pass.

### Translating old litellm invocations

`--model fireworks_ai/accounts/fireworks/models/<name>` under `validation_bench.py` becomes `--model <name>` here. The derived slug is the same in both paths (`fireworks-<name>`) so old and new results rows remain comparable.

## validation_bench_openai.py (OpenAI, native SDK with Chat Completions + Responses routing)

Replaces the litellm-based routing in `validation_bench.py` with an explicit branch on the model ID: models with `"codex"` in the name go to the Responses API (mandatory for `gpt-5-codex`), everything else goes to Chat Completions. `reasoning_effort` placement differs between the two (`reasoning_effort="low"` flat vs `reasoning={"effort":"low"}` nested) and tool/tool-result shapes differ (nested vs flat). The script hides that behind a unified CLI while exposing the routing in the results row via a `"api"` provenance field (`"chat.completions"` or `"responses"`).

### Non-codex reasoning model (Chat Completions)

```bash
python validation_bench_openai.py --task toml-1.0-cpp \
  --model gpt-5.4 \
  --n-attempts 1 --max-turns 5
```

Drops the litellm-era `openai/` prefix from the old `validation_bench.py` invocations. `--max-tokens` is translated to `max_completion_tokens` on the wire — reasoning-era models (gpt-5.x, o-series) reject the legacy `max_tokens` with HTTP 400.

### Codex via Responses API

```bash
python validation_bench_openai.py --task toml-1.0-cpp \
  --model gpt-5.3-codex --reasoning-effort low \
  --n-attempts 1 --max-turns 5
```

`gpt-5.3-codex` contains `"codex"` so `is_codex_model()` routes to `client.responses.stream(...)`. `--reasoning-effort low` becomes `reasoning={"effort":"low"}` nested on the request. `--max-tokens` is translated to `max_output_tokens`. Assistant messages are kept in OpenAI-chat shape internally and translated into Responses `input_items` (`function_call` / `function_call_output` items) at request time, so the attempt loop stays provider-agnostic.

### Reasoning effort levels

```bash
--reasoning-effort {none,minimal,low,medium,high,xhigh}
```

Omit for the model default. `xhigh` is only accepted by some codex variants (e.g. `gpt-5.3-codex`). Reasoning traces: Chat Completions does **not** expose reasoning on stream deltas — only final message has it. Responses streams them via `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` events (visible in the heartbeat character count, but not replayed on subsequent turns since signed reasoning items aren't surfaced on the stream).

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
