# quant-sampling — Quantized Model Sampling Parameter Matcher

When quantizing LLMs, the output distribution shifts. The default sampling parameters (temp, top\_p, top\_k) recommended for the full-precision model may no longer be optimal. **quant-sampling** compares logit distributions between a reference model and its quantized variant, computes KL divergence, and searches for the temperature that minimizes divergence — helping the quantized model behave more like the original.

## Core workflow

```
quant-sampling generate  →  reference.qmlog     (run reference model, sample continuation, save tokens + logits)
quant-sampling collect   →  quantized.qmlog     (run quantized model on same tokens, save logits)
quant-sampling compare   -a reference.qmlog -b quantized.qmlog [--optimize]   (KL divergence + temperature search)
```

## Build

Requires a local checkout of [llama.cpp](https://github.com/ggml-org/llama.cpp).

```bash
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build build
```

## Single-pair usage

### Step 1: Generate reference logits

```bash
./build/quant-sampling generate \
  -m model_fp16.gguf \
  -p "Your prompt here" \
  -n 256 --temp 1.0 --top-p 0.95 --top-k 40 --seed 42 \
  -ngl 99 -c 2048 \
  -o ref.qmlog
```

### Step 2: Collect quantized model logits

```bash
./build/quant-sampling collect \
  -m model_q4.gguf \
  -i ref.qmlog \
  -o quant.qmlog \
  -ngl 99
```

### Step 3: Compare and optimize

```bash
./build/quant-sampling compare -a ref.qmlog -b quant.qmlog --optimize
```

Output includes:
- Mean/std KL divergence
- Top-1 token agreement percentage
- Per-position KL for identifying high-divergence regions
- Temperature scan with recommended value for the quantized model

Temperature scan range is configurable (default: 0.05 to 2.0, step 0.05):
```bash
./build/quant-sampling compare -a ref.qmlog -b quant.qmlog --optimize \
  --temp-min 1.0 --temp-max 1.5 --temp-step 0.01
```

## Batch workflow (multiple prompts, multiple quant variants)

The three batch scripts separate reference generation from quantized model
collection, so the expensive reference pass runs only once even when comparing
multiple quant levels (Q2, Q4, Q5, …).

### Step 1: Generate reference logits for all prompts (once)

```bash
./generate_batch.sh <ref_model.gguf> [outdir] [n_predict] [prompts_file]

# Example:
./generate_batch.sh ~/models/model-Q8_K.gguf
```

Writes `batch_run/ref_01.qmlog`, `ref_02.qmlog`, … (one per line in `prompts.txt`).

### Step 2: Collect logits for each quantized variant

```bash
./collect_batch.sh <quant_model.gguf> <tag> [outdir]

# Example — run once per quant level:
./collect_batch.sh ~/models/model-Q2_K.gguf q2
./collect_batch.sh ~/models/model-Q4_K.gguf q4
./collect_batch.sh ~/models/model-Q5_K.gguf q5
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
