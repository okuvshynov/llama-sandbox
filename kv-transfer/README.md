# kv-transfer — KV Cache Transfer Between Quantization Levels

Tests whether processing the prompt with a high-quality model (BF16) and handing off the KV cache to a lower-quality model for generation preserves output quality.

## Experiment

Three runs, same token sequence:

1. **ref**: prompt + generation with reference model (BF16). Saves full logits for all generation positions.
2. **target**: replay the same token sequence through a quantized target model (e.g. Q4). Computes per-token KL divergence vs ref logits inline. Saves compact stats file.
3. **handoff**: prompt processed with ref model, KV cache transferred to target, generation replayed through target. Computes per-token KL vs ref inline. Saves compact stats file.

If handoff produces logits closer to ref than target does, it means the KV cache from the better model carries useful information that helps the weaker model generate better.

## How KV transfer works

- llama.cpp's `llama_state_save_file` / `llama_state_load_file` saves and restores the full KV cache
- KV cache type (default F16) is a context parameter, independent of model weight quantization
- Both models must be the same architecture (same layer count, embedding size)
- Both contexts use the same default F16 KV type, so the state transfers cleanly

## Prerequisites

- [llama.cpp](https://github.com/ggml-org/llama.cpp) built from source
- CMake >= 3.14, C++17 compiler
- Python 3 with matplotlib + numpy (for static plots)

## Build

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
cmake -B build
cmake --build build
```

## Subcommands

```bash
# Run reference model — generates tokens and saves full logits
./build/kv-transfer ref -m <model> -p <prompt> -n 2048 --temp 0.6 -ngl 99 -o ref.bin

# Run target model — replays same tokens, computes per-token KL vs ref inline
./build/kv-transfer target -m <model> -i ref.bin -o target.bin -ngl 99

# Handoff — prompt with ref model, generation with target model, computes KL inline
./build/kv-transfer handoff -m-ref <ref_model> -m-tgt <tgt_model> -i ref.bin -o handoff.bin -ngl 99

# Summarize a stats file (mean/p95/p99 KL + top-1 agreement)
./build/kv-transfer compare -f target.bin

# Analyze KL decay across generation position
./build/kv-transfer decay --target target.bin --handoff handoff.bin --window 64
```

## File formats

### Trace file (ref.bin) — version 4

Stores full logits from the reference run. One file per prompt per model family, reused across all target quants.

```
Header (72 bytes): magic "qmlogits", version, n_vocab, n_prompts, sampling params
Per-prompt: path string, n_tokens, n_prompt, token IDs, float[n_gen × n_vocab] logits
```

These files are large (hundreds of MB to ~1 GB depending on vocab size and generation length).

### Stats file (target.bin, handoff.bin)

Compact per-token stats written by target and handoff commands. KL divergence is computed inline against ref logits during inference, so only the per-token results are stored.

```
Header (32 bytes): magic "qmstats", version, n_gen, n_prompt, temp
Data: float[n_gen] KL per token, uint8[n_gen] top-1 match
```

Typically ~10 KB for 2048 generation tokens (~500,000x smaller than full logit traces).

## Automated studies

`run.sh` is a generic engine that runs ref, target, handoff, compare, and decay analysis for all prompts across multiple target models. Per-model wrapper scripts set the configuration and source `run.sh`:

```bash
./run-gemma4-e2b.sh                                # Gemma4-E2B: BF16 ref vs 6 quants
./run-gemma4-26b-a4b.sh                            # Gemma4-26B-A4B (MoE): BF16 ref vs 6 quants
PROMPT_FILTER="01_*" ./run-gemma4-31b.sh           # single prompt test
THREADS=16 NGL=0 ./run-qwen3.5-2b.sh              # CPU-only, 16 threads
```

To add a new model family, create a wrapper script that sets:
- `REF_MODEL` — path to reference GGUF (ideally BF16)
- `REF_TAG` — tag for results directory name
- `TARGETS` — array of `"tag:path"` pairs
- `N_PREDICT`, `TEMP` — generation parameters

Then source `run.sh`. Example:

```bash
#!/bin/bash
REF_MODEL="$HOME/llms/MyModel-BF16.gguf"
REF_TAG="mymodel-bf16"
TARGETS=(
    "ud-q2_k_xl:$HOME/llms/MyModel-Q2.gguf"
    "ud-q4_k_xl:$HOME/llms/MyModel-Q4.gguf"
    "ud-q8_k_xl:$HOME/llms/MyModel-Q8.gguf"
)
N_PREDICT=2048
TEMP=0.6
source "$(dirname "$0")/run.sh"
```

Environment variables:
- `PROMPT_FILTER` — glob to filter prompts (default: `*`, all prompts)
- `THREADS` — inference threads (default: llama.cpp default)
- `NGL` — GPU layers (default: 99)

Resume-friendly — skips existing ref, target, and handoff .bin files. Summary CSV and decay CSV are always regenerated. A timing summary is printed at the end using llama_perf data from log files.

## Prompts

19 test prompts across three size categories:

- **Small** (8 prompts, ~300-800 tokens): short code analysis tasks (bug finding, code explanation)
- **Medium** (6 prompts, ~2000-3600 tokens): multi-file codebase reviews
- **Large** (5 prompts, ~8000-10000 tokens): full application reviews (microservices, schedulers, APIs)

## Results

Results are organized as:

```
results/
  view.html              # interactive bar charts (KL + top-1)
  decay.html             # KL ratio decay curves
  decay-compare.html     # absolute KL decay, target vs handoff per quant
  decay-by-size.html     # same, filtered by prompt size
  <ref_tag>/
    <prompt>-ref.bin     # full logits, shared across all targets
    <prompt>-ref.txt     # generated text (human-readable)
    summary.csv          # all targets × prompts
    decay.csv            # windowed decay analysis
    <target_tag>/
      <prompt>-target.bin    # per-token stats
      <prompt>-handoff.bin   # per-token stats
```

Serve and view:

```bash
cd results && python3 -m http.server 8080 --bind 0.0.0.0
```

Static plot generation:

```bash
python3 plot_results.py --results-dir results --output-dir plots    # per-model KL bar charts
python3 plot_decay.py --results-dir results --output-dir plots      # per-model decay curves
python3 plot_models.py --results-dir results --output-dir plots     # cross-model comparison
```

For detailed findings, see [TECHNICAL.md](TECHNICAL.md).
