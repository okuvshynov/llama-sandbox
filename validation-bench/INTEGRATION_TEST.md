# Integration test — palindrome × every provider

The `palindrome-cpp17` task is the harness's smoke test. Every provider
script is expected to reach MCC=1.000 on it within a few turns; any
deviation is a harness regression, not a model regression.

This document records the exact commands used to exercise each provider
against the palindrome task, plus the consolidation command that prints
a single table of results across all of them.

## Pre-flight

- Docker images built: `vb-sandbox-cpp17` (see `README.md` for the build
  command). The lua image isn't needed for this test since the only
  task being run is C++17.
- Setup has been run at least once so `data/specs/palindrome/tests*`
  exist: `./setup.sh`.
- For each provider, the corresponding API key env var is set
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY`,
  `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`). For llama.cpp, a local
  OpenAI-compatible server is running on `http://127.0.0.1:8080`.

Each command writes to its own `results-palindrome-<provider>/` directory
so the smoke results don't co-mingle with the main `results/` dataset.

## Commands

### OpenAI (gpt-5.4-mini)

```bash
python validation_bench_openai.py \
  --task palindrome-cpp17 \
  --model gpt-5.4-mini \
  --n-attempts 1 --max-turns 5 \
  --results-dir results-palindrome-openai
```

### Fireworks (glm-5p1)

```bash
python validation_bench_fireworks.py \
  --task palindrome-cpp17 \
  --model glm-5p1 \
  --n-attempts 1 --max-turns 5 --max-tokens 10000 \
  --results-dir results-palindrome-fireworks
```

### Anthropic (claude-sonnet-4-6, thinking)

```bash
python validation_bench_anthropic.py \
  --task palindrome-cpp17 \
  --model claude-sonnet-4-6 \
  --n-attempts 1 --max-turns 5 \
  --thinking enabled --thinking-budget 5000 --max-tokens 16000 \
  --results-dir results-palindrome-anthropic
```

### DeepSeek (v4-pro thinking)

```bash
python validation_bench_deepseek.py \
  --task palindrome-cpp17 \
  --model deepseek-v4-pro --mode thinking --tool-choice auto \
  --n-attempts 1 --max-turns 5 --max-tokens 16000 \
  --results-dir results-palindrome-deepseek
```

### Moonshot (kimi-k2.6 thinking)

```bash
python validation_bench_moonshot.py \
  --task palindrome-cpp17 \
  --model kimi-k2.6 --mode thinking \
  --n-attempts 1 --max-turns 5 --max-tokens 16000 \
  --results-dir results-palindrome-moonshot
```

### llama.cpp (local OpenAI-compatible server)

```bash
python validation_bench_llama_cpp.py \
  --task palindrome-cpp17 \
  --api-base http://127.0.0.1:8080/v1 \
  --api-key 1234 --timeout 86400 \
  --n-attempts 1 --max-turns 5 \
  --results-dir results-palindrome-llama-cpp
```

The local server's `--api-key` is a placeholder; llama.cpp accepts any
non-empty string. The `--timeout 86400` (24h) leaves the harness room
to wait on a slow local model without giving up.

## Inspecting the consolidated output

After running any subset of the above, this Python snippet collects
every `results-palindrome-*/results.jsonl` file present and prints a
provider-by-provider table plus the per-turn detail with token counts.
Run it from `validation-bench/`:

```bash
python3 << 'EOF'
"""Consolidate palindrome-cpp17 smoke results across all providers."""
import json
from pathlib import Path

PATHS = {
    "openai":     "results-palindrome-openai/results.jsonl",
    "fireworks":  "results-palindrome-fireworks/results.jsonl",
    "anthropic":  "results-palindrome-anthropic/results.jsonl",
    "deepseek":   "results-palindrome-deepseek/results.jsonl",
    "moonshot":   "results-palindrome-moonshot/results.jsonl",
    "llama.cpp":  "results-palindrome-llama-cpp/results.jsonl",
}

print(f"{'provider':12s}  {'turns':>5s}  {'final mcc':>9s}  "
      f"{'in':>7s} {'out':>7s} {'reason':>7s} {'cached':>7s}")
print("-" * 70)
for prov, path in PATHS.items():
    p = Path(path)
    if not p.exists():
        print(f"{prov:12s}  (no rows)")
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        continue
    final = rows[-1]
    final_mcc = "ERR" if "mcc" not in final else f"{final['mcc']:.3f}"
    in_t  = sum(r.get("input_tokens")     or 0 for r in rows)
    out_t = sum(r.get("output_tokens")    or 0 for r in rows)
    rea_t = sum(r.get("reasoning_tokens") or 0 for r in rows)
    cac_t = sum(r.get("cached_tokens")    or 0 for r in rows)
    print(f"{prov:12s}  {len(rows):>5d}  {final_mcc:>9s}  "
          f"{in_t:>7d} {out_t:>7d} {rea_t:>7d} {cac_t:>7d}")

print("\n--- per-turn detail ---")
for prov, path in PATHS.items():
    p = Path(path)
    if not p.exists():
        continue
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"\n{prov}:")
    for r in rows:
        verdict = f"err={r['error']}" if "error" in r else f"mcc={r['mcc']:.3f}"
        print(f"  turn{r['turn']}: {verdict:24s}  "
              f"in={r.get('input_tokens')}  out={r.get('output_tokens')}  "
              f"reason={r.get('reasoning_tokens')}  cached={r.get('cached_tokens')}")
EOF
```

## Expected outcome

Every provider should reach **MCC=1.000** within 1–2 turns. Some compile
on turn 0, some need a fix-up turn — both are normal. Token fields are
populated as follows:

- **`input_tokens` / `output_tokens`**: present on every provider.
- **`reasoning_tokens`**: populated by DeepSeek's thinking models and
  OpenAI reasoning models (gpt-5.x). Anthropic and Moonshot bundle
  thinking into `output_tokens`; the field is null there by design.
  llama.cpp doesn't track reasoning.
- **`cached_tokens`**: populated when the provider supports prompt
  caching and our prompt was large enough / repeated enough to hit it.
  Single-turn one-shots typically have no cached tokens (no prior turn
  to cache against).

If a provider fails to reach 1.000, or if a row is missing a token field
that the provider is supposed to expose, that's a harness regression and
should be investigated against the per-provider docs in this commit's
git history (look for `validation_bench_<provider>.py` and the
`normalize_*_usage` helpers in `validation_bench_lib.py`).
