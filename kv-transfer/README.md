# kv-transfer — KV Cache Transfer Between Quantization Levels

Tests whether processing the prompt with a high-quality model and handing off the KV cache to a lower-quality model for generation preserves output quality.

## Experiment

Three runs, same token sequence:

1. **ref**: prompt + generation with reference model (e.g. Q8)
2. **target**: prompt + generation with target model (e.g. Q2) — baseline for "just use the smaller model"
3. **handoff**: prompt processed with ref model, KV cache transferred to target, generation continues with target

If handoff produces logits closer to ref than target does, it means the KV cache from the better model carries useful information that helps the weaker model generate better.

## How KV transfer works

- llama.cpp's `llama_state_save_file` / `llama_state_load_file` saves and restores the full KV cache
- KV cache type (default F16) is a context parameter, independent of model weight quantization
- Both models must be the same architecture (same layer count, embedding size)
- Both contexts use the same default F16 KV type, so the state transfers cleanly

## Prerequisites

- [llama.cpp](https://github.com/ggml-org/llama.cpp) built from source
- CMake >= 3.14, C++17 compiler
- Python 3 (for viewing results via `python3 -m http.server`)

## Build

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
cmake -B build
cmake --build build
```

## Subcommands

```bash
# Run reference model — generates tokens and saves logits
./build/kv-transfer ref -m <model> -p <prompt> -n 256 --temp 0.6 -ngl 99 -o ref.bin

# Run target model — replays same tokens, saves logits
./build/kv-transfer target -m <model> -i ref.bin -o target.bin -ngl 99

# Handoff — prompt with ref model, generation with target model
./build/kv-transfer handoff -m-ref <ref_model> -m-tgt <tgt_model> -i ref.bin -o handoff.bin -ngl 99

# Compare two .bin files
./build/kv-transfer compare -a ref.bin -b target.bin

# Analyze KL decay across generation position
./build/kv-transfer decay --ref ref.bin --target target.bin --handoff handoff.bin --window 64
```

## Automated studies

`run.sh` is a generic engine that runs ref, target, handoff, compare, and decay analysis for all prompts across multiple target models. Per-model wrapper scripts set the configuration and source `run.sh`:

```bash
./run-qwen3.5-2b.sh                    # Qwen3.5-2B: Q8 ref vs 6 quants
./run-qwen3.5-35b-a3b.sh               # Qwen3.5-35B-A3B (MoE): Q8 ref vs 6 quants
THREADS=16 NGL=0 ./run-qwen3.5-2b.sh   # CPU-only, 16 threads
```

To add a new model family, create a wrapper script that sets:
- `REF_MODEL` — path to reference GGUF
- `REF_TAG` — tag for results directory name
- `TARGETS` — array of `"tag:path"` pairs
- `N_PREDICT`, `TEMP` — generation parameters

Then source `run.sh`. Example:

```bash
#!/bin/bash
REF_MODEL="$HOME/llms/MyModel-Q8.gguf"
REF_TAG="mymodel-q8"
TARGETS=(
    "q2:$HOME/llms/MyModel-Q2.gguf"
    "q4:$HOME/llms/MyModel-Q4.gguf"
)
N_PREDICT=512
TEMP=0.6
source "$(dirname "$0")/run.sh"
```

Resume-friendly — skips existing ref, target, and handoff .bin files. Summary CSV and decay CSV are always regenerated.

**Disk space note:** once all .bin files for a target quant are generated, that target model file is no longer needed. The reference model must be kept — it's required for handoff runs whenever a new target quant is added.

## Results

Results are organized as:

```
results/
  view.html              # interactive bar charts (KL + top-1)
  decay.html             # KL decay curves across generation position
  <ref_tag>/
    <prompt>-ref.bin     # shared across all targets
    summary.csv          # all targets x prompts
    decay.csv            # windowed decay analysis
    <target_tag>/
      <prompt>-target.bin
      <prompt>-handoff.bin
```

Serve and view:

```bash
cd results && python3 -m http.server 8080 --bind 0.0.0.0
# open view.html for bar charts, decay.html for decay curves
```

For detailed findings, see [TECHNICAL.md](TECHNICAL.md).
