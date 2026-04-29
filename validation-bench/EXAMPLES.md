# validation-bench: known-good invocations

Canonical per-script invocations, each run with `--n-attempts 5` for parity. Every command here has been run against the live API (or local server) and is known to complete without flag tweaks.

## validation_bench_anthropic.py

```bash
python validation_bench_anthropic.py --task toml-1.0-cpp \
  --model claude-sonnet-4-6 \
  --thinking enabled --thinking-budget 30000 \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

```bash
python validation_bench_anthropic.py --task toml-1.0-cpp \
  --model claude-opus-4-7 --thinking adaptive \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

## validation_bench_deepseek.py

```bash
python validation_bench_deepseek.py --task toml-1.0-cpp \
  --model deepseek-v4-pro --mode thinking --tool-choice auto \
  --n-attempts 5 --max-turns 5 --max-tokens 65536
```

## validation_bench_fireworks.py

```bash
python validation_bench_fireworks.py --task toml-1.0-cpp \
  --model glm-5p1 \
  --n-attempts 5 --max-turns 5 --max-tokens 100000
```

## validation_bench_llama_cpp.py

```bash
python validation_bench_llama_cpp.py --task toml-1.0-cpp \
  --api-base http://localhost:8080/v1 \
  --api-key 1234 --timeout 86400 \
  --n-attempts 5 --max-turns 5
```

## validation_bench_moonshot.py

```bash
python validation_bench_moonshot.py --task toml-1.0-cpp \
  --model kimi-k2.6 --mode thinking \
  --n-attempts 5 --max-turns 5 --max-tokens 65536
```

## validation_bench_openai.py

```bash
python validation_bench_openai.py --task toml-1.0-cpp \
  --model gpt-5.3-codex --reasoning-effort high \
  --n-attempts 5 --max-turns 5
```

```bash
python validation_bench_openai.py --task toml-1.0-cpp \
  --model gpt-5.4 \
  --n-attempts 5 --max-turns 5
```

```bash
# yaml-1.2-cpp17 with reasoning_effort=high on gpt-5.3-codex needs more
# output-token budget and a longer client timeout than the toml/lua
# tasks — the embedded YAML 1.2 spec is ~38K tokens of prompt and the
# model spends substantial reasoning headroom. Stream-disconnect on the
# default 600s timeout ("Didn't receive a `response.completed` event")
# is the smoke for "bump --timeout and --max-tokens".
python validation_bench_openai.py --task yaml-1.2-cpp17 \
  --model gpt-5.3-codex --reasoning-effort high \
  --max-tokens 65536 --timeout 3600 \
  --n-attempts 1 --max-turns 5
```
