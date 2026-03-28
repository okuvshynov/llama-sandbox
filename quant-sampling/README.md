# quant-sampling — Quantized Model Sampling Parameter Matcher

When quantizing LLMs, the output distribution shifts. The default sampling parameters (temp, top\_p, top\_k) recommended for the full-precision model may no longer be optimal. **quant-sampling** compares logit distributions between a reference model and a target model (e.g. a quantized variant), computes KL divergence, and searches for the temperature that minimizes divergence — helping the target model behave more like the original.

## Core workflow

```
quant-sampling ref     →  ref.qmlog              (run reference model, sample continuation, save tokens + logits)
quant-sampling target  →  target.qmlog           (run target model on same tokens, save logits)
quant-sampling compare   -a ref.qmlog -b target.qmlog [--optimize]   (KL divergence + temperature search)
```

## Build

Requires a local checkout of [llama.cpp](https://github.com/ggml-org/llama.cpp).

```bash
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build
```

If you want to build llama.cpp with your preferred accelerator, make sure to use corresponding options, for example:
```bash
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp -DGGML_CUDA=ON
```

for CUDA build.

## Single-pair usage

### Step 1: Run reference model

```bash
./build/quant-sampling ref \
  -m model_fp16.gguf \
  -p "Your prompt here" \
  -n 256 --temp 1.0 --top-p 0.95 --top-k 40 --seed 42 \
  -ngl 99 -c 2048 \
  -o ref.qmlog
```

### Step 2: Run target model

```bash
./build/quant-sampling target \
  -m model_q4.gguf \
  -i ref.qmlog \
  -o target.qmlog \
  -ngl 99
```

### Step 3: Compare and optimize

```bash
./build/quant-sampling compare -a ref.qmlog -b target.qmlog --optimize
```

Output includes:
- Mean/std KL divergence
- Top-1 token agreement percentage
- Per-position KL for identifying high-divergence regions
- Temperature scan with recommended value for the target model

Temperature scan range is configurable (default: 0.05 to 2.0, step 0.05):
```bash
./build/quant-sampling compare -a ref.qmlog -b target.qmlog --optimize \
  --temp-min 1.0 --temp-max 1.5 --temp-step 0.01
```

## Batch workflow (multiple prompts, multiple target variants)

The three batch scripts separate reference generation from target model
runs, so the expensive reference pass runs only once even when comparing
multiple quant levels (Q2, Q4, Q5, …).

### Step 1: Run reference model for all prompts (once)

```bash
./ref_batch.sh <ref_model.gguf> [outdir] [n_predict] [prompts_file]

# Example:
./ref_batch.sh ~/models/model-Q8_K.gguf
```

Writes `batch_run/ref_01.qmlog`, `ref_02.qmlog`, … (one per line in `prompts.txt`).

### Step 2: Run target model for each variant

```bash
./target_batch.sh <target_model.gguf> <tag> [outdir]

# Example — run once per quant level:
./target_batch.sh ~/models/model-Q2_K.gguf q2
./target_batch.sh ~/models/model-Q4_K.gguf q4
./target_batch.sh ~/models/model-Q5_K.gguf q5
```

Writes `batch_run/<tag>_01.qmlog`, … and `batch_run/manifest_<tag>.txt`.

### Step 3: Optimize temperature for each variant

```bash
./compare_batch.sh <tag> [outdir] [extra compare-batch flags]

# Example:
./compare_batch.sh q2
./compare_batch.sh q4 batch_run --temp-min 1.0 --temp-max 1.5 --temp-step 0.01
```

Reads `batch_run/manifest_<tag>.txt`, reports per-prompt optimal temperature
and aggregate mean ± std across all prompts.

## prompts.txt format

One prompt per line, blank lines ignored:
```
Implement quicksort in C++
Write a Python function to parse JSON from a URL
...
```
