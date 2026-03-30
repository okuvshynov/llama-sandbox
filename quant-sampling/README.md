# quant-sampling — Sampling Parameter Matcher

When quantizing LLMs, the output distribution shifts. The default sampling parameters (temp, top\_p, top\_k) recommended for the full-precision model may no longer be optimal. **quant-sampling** compares logit distributions between a reference model and a target model (e.g. a quantized variant), computes KL divergence, and searches for the temperature that minimizes divergence — helping the target model behave more like the original.

## Core workflow

```
quant-sampling ref     -m ref_model.gguf (-p "prompt" | -P prompts/) -o ref.bin
quant-sampling target  -m target_model.gguf -i ref.bin -o target.bin
quant-sampling compare -a ref.bin -b target.bin --optimize [--csv scan.csv]
```

A single .bin file can contain one or many prompts. The workflow is always the same regardless of prompt count.

## Prerequisites

- **C++17 compiler** (GCC 8+, Clang 7+, MSVC 2017+)
- **CMake** >= 3.14
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — built from source. Only the source tree is needed; quant-sampling builds llama.cpp as a subdirectory via CMake
- **[huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli)** — for downloading models (`pip install huggingface-hub`). Only needed if using `study.sh`
- **Python 3** — for `python3 -m http.server` when viewing results. Not needed for building or running the tool itself

## Build

Set `LLAMA_CPP_DIR` to your llama.cpp source checkout (or pass via `-D`):

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

## Viewing results

The `results/` directory contains CSV output files and an interactive HTML viewer (`results/view.html`). To use it, serve the directory over HTTP:

```bash
cd results
python3 -m http.server 8080 --bind 0.0.0.0
```

Then open `http://<your-ip>:8080/view.html`. The viewer auto-loads any CSV in the same directory, or you can use the file picker. It shows:

- Per-category recommended temperature (knowledge, math, swe, overall)
- Heatmap of KL ratio vs baseline across all prompt x temperature pairs
- Best temperature per prompt highlighted

The viewer discovers CSV files from the server's directory listing — just drop new CSVs in and they appear in the dropdown.

## Automated study

`study.sh` automates the full pipeline: download models, run ref, run all targets, compare. Edit the configuration section at the top for each model family.

```bash
./study.sh                    # run everything
./study.sh --skip-download    # skip model downloads (models already present)
./study.sh --skip-ref         # skip ref generation (reuse existing ref.bin)
```

The script is resume-friendly: it skips downloads and target runs for files that already exist, so you can re-run after a crash.

Supports both single-file and split GGUF models:
```bash
# single file
"iq2_m:Qwen3.5-2B-UD-IQ2_M.gguf"

# split model (pattern for download, first shard for inference)
"iq1_m:Qwen3.5-397B*UD-IQ1_M*:Qwen3.5-397B-A17B-UD-IQ1_M-00001-of-00004.gguf"
```
