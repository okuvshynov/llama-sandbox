# quant-sampling — Sampling Parameter Matcher

When quantizing LLMs, the output distribution shifts. The default sampling parameters (temp, top\_p, top\_k) recommended for the full-precision model may no longer be optimal. **quant-sampling** compares logit distributions between a reference model and a target model (e.g. a quantized variant), computes KL divergence, and searches for the temperature that minimizes divergence — helping the target model behave more like the original.

## Core workflow

```
quant-sampling ref     -m ref_model.gguf (-p "prompt" | -P prompts/) -o ref.bin
quant-sampling target  -m target_model.gguf -i ref.bin -o target.bin
quant-sampling compare -a ref.bin -b target.bin --optimize [--csv scan.csv]
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

Multiple prompts from a directory:
```bash
./build/quant-sampling ref \
  -m model_fp16.gguf \
  -P prompts/ \
  -n 256 --temp 0.6 \
  -ngl 99 \
  -o ref.bin
```

The directory is scanned recursively — subdirectories are used to organize prompts by topic. Paths relative to the prompt directory (e.g. `math/01.txt`) are stored in the .bin file and appear in compare output.

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

Output includes:
- Per-prompt summary table (path, KL divergence, top-1 agreement)
- Aggregate statistics across prompts
- Temperature optimization: per-prompt best temperature and KL at that temperature

Temperature scan defaults are derived from the reference temperature used during generation (e.g. for ref\_temp=0.6: scans 0.30–0.90 at 0.01 steps). Override with:
```bash
./build/quant-sampling compare -a ref.bin -b target.bin --optimize \
  --temp-min 0.5 --temp-max 1.2 --temp-step 0.01
```

### CSV export

Export the full prompt x temperature KL matrix for analysis in Python/JS:
```bash
./build/quant-sampling compare -a ref.bin -b target.bin --optimize --csv scan.csv
```

Output format:
```csv
prompt,path,temp,kl
1,knowledge/01.txt,0.31,0.045123
1,knowledge/01.txt,0.32,0.043456
...
30,swe/10.txt,0.90,0.072001
```

The `path` column enables grouping by topic (e.g. aggregate math vs knowledge vs swe prompts).

### Diff (optional)

Show detailed token-level disagreements:
```bash
./build/quant-sampling diff -a ref.bin -b target.bin -m model.gguf
```

## Prompt directory

Prompts are organized by topic in subdirectories:
```
prompts/
  knowledge/
    01.txt    — Paris in the 16th century
    02.txt    — Byzantine knowledge transmission
    ...
  math/
    01.txt    — sqrt(2) irrationality proof
    02.txt    — eigenvalues and eigenvectors
    ...
  swe/
    01.txt    — merge sort in Python
    02.txt    — HTTP server in Go
    ...
```

Each `.txt` file is one prompt (multiline supported). Files are discovered recursively and sorted by full path, so ordering is deterministic: knowledge/, then math/, then swe/.
