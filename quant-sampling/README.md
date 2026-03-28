# quant-sampling — Sampling Parameter Matcher

When quantizing LLMs, the output distribution shifts. The default sampling parameters (temp, top\_p, top\_k) recommended for the full-precision model may no longer be optimal. **quant-sampling** compares logit distributions between a reference model and a target model (e.g. a quantized variant), computes KL divergence, and searches for the temperature that minimizes divergence — helping the target model behave more like the original.

## Core workflow

```
quant-sampling ref     -m ref_model.gguf (-p "prompt" | -P prompts/) -o ref.bin
quant-sampling target  -m target_model.gguf -i ref.bin -o target.bin
quant-sampling compare -a ref.bin -b target.bin [--optimize]
```

A single .bin file can contain one or many prompts. The workflow is always the same regardless of prompt count.

## Build

Requires a local checkout of [llama.cpp](https://github.com/ggml-org/llama.cpp).

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
cmake -B build
cmake --build build
```

For CUDA:
```bash
cmake -B build -DGGML_CUDA=ON
```

## Usage

### Step 1: Run reference model

Single prompt:
```bash
./build/quant-sampling ref \
  -m model_fp16.gguf \
  -p "Your prompt here" \
  -n 256 --temp 0.6 --top-p 0.95 --top-k 40 --seed 42 \
  -ngl 99 -c 2048 \
  -o ref.bin
```

Multiple prompts (directory of .txt files, one prompt per file):
```bash
./build/quant-sampling ref \
  -m model_fp16.gguf \
  -P prompts/ \
  -n 256 --temp 0.6 \
  -ngl 99 \
  -o ref.bin
```

Files in the directory are sorted alphabetically and each .txt file becomes one prompt (multiline supported).

### Step 2: Run target model

```bash
./build/quant-sampling target \
  -m model_q4.gguf \
  -i ref.bin \
  -o target.bin \
  -ngl 99
```

Run this once per target model variant — the ref.bin is reused:
```bash
./build/quant-sampling target -m model_q2.gguf -i ref.bin -o target_q2.bin -ngl 99
./build/quant-sampling target -m model_q4.gguf -i ref.bin -o target_q4.bin -ngl 99
```

### Step 3: Compare and optimize

```bash
./build/quant-sampling compare -a ref.bin -b target.bin --optimize
```

For single-prompt files, output includes:
- Mean/std KL divergence
- Top-1 token agreement percentage
- Per-position KL for identifying high-divergence regions
- Temperature scan with recommended value

For multi-prompt files, output includes:
- Per-prompt summary table (KL, top-1 agreement)
- Aggregate statistics across prompts
- Per-prompt optimal temperature + global recommendation

Temperature scan range is configurable:
```bash
./build/quant-sampling compare -a ref.bin -b target.bin --optimize \
  --temp-min 0.5 --temp-max 1.2 --temp-step 0.01
```

### Diff (optional)

Show detailed token-level disagreements:
```bash
./build/quant-sampling diff -a ref.bin -b target.bin -m model.gguf
```

## Prompt directory format

Create a directory with `.txt` files:
```
prompts/
  01_coding.txt
  02_math.txt
  03_creative.txt
```

Each file contains one prompt (can be multiline). Files are processed in alphabetical order.
