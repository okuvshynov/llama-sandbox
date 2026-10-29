# ds-completion quickstart

DeepSeek-V4.1-Flash-1017 text completion on the Mac Pro (4x Vega II, 755 GB RAM).

- Dense weights (attention, shared experts, head) run on one GPU through our own HIP kernels,
  in exact fp8.
- Routed experts run on the CPU, through our own exact AVX-512 kernel.
- Greedy decoding, one sequence, text only.

## What it needs

- **Model:** `~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000{1..9}-of-00009.gguf`
  (474 GB; it runs from the page cache, so the RAM must hold it).
- **llama.cpp fork:** `~/forks/llama.cpp` with its CPU build in `build/` (static libraries;
  used for ggml, GGUF reading and the tokenizer).
- **ROCm 6.3.4** in `/opt/rocm-6.3.4` for the GPU runtime.

## Build and prepare (once)

```bash
cd ~/projects/llama-sandbox/ds-completion
make && make HIP=1                     # -> bin/ (CPU) and bin-hip/ (GPU)
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf   # lossless fp8 repack, 6.4 GB, ~20 s
```

## Run

Fastest (dense part on GPU 0, the most-used routed experts on all four GPUs, the rest on the
CPU; run from this directory, since the placement path is relative):

```bash
cd ~/projects/llama-sandbox/ds-completion
bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt \
    -p "Write a haiku about memory bandwidth." --chat -n 64
```

Without `--placement` every routed expert runs on the CPU (faster start, ~12 t/s).

Faster still, with speculative decoding (same output, byte for byte; +22% on average, more on
code and math, least on Chinese/Japanese):

```bash
bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt \
    --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf --spec \
    -p "Write a haiku about memory bandwidth." --chat -n 64
```

Options:

- `-p "text"` or `-f prompt.txt`: the prompt, inline or from a file (as is).
- `--chat`: one user turn, no thinking.
- `--think`: with thinking. `--effort N`: its reasoning effort, 1-100 (default 75), or `low` /
  `high` / `max` (50 / 75 / 100), as the official encoding; the model card's benchmarks use 100.
- neither: the prompt is raw text after BOS.
- `-n`: max new tokens.
- `-t`: CPU threads; the default 16 (one per physical core) is best with `--rt`. 32 leaves no
  CPU for the host thread and the HIP runtime and is much slower (README step 18).
- `--profile`: a per-step time breakdown.
- `--gpu D`: the die for the runtime (default 0).
- `--placement FILE`: routed experts on GPUs, the file's top-ranked ones that fit (10-30 s upload).
- `--moe-dies 0,1,2,3`: the dies that hold experts (default all four).
- `--dspark FILE`: the DSpark draft GGUF (8.5 GB on GPU 0). Alone: shadow mode, drafts are
  measured but not used.
- `--spec`: speculative decoding (needs `--dspark`); the number of guesses verified per step
  is chosen from the draft's confidence. `--spec-n N` fixes it; `--spec-log FILE` writes
  per-step data.
- `--canon`: placement-invariant expert numerics: the output no longer depends on where the
  experts run (any placement, any `--moe-dies`, with or without the draft, with or without the
  cache give the same bytes); ~2% slower.
- `--cache-loads R`: the adaptive expert cache (README step 32): the GPUs keep the experts used
  most in the last 256 tokens, loading up to R experts per second in the background (100 is a
  good default; 0 = the static placement). Use with `--canon`, or the output depends on load
  timing. It starts loading ~30-40 s after startup (registering the model's expert memory with
  the GPUs); `--cache-wait` waits for that before generating. `--cache-period N` (16) and
  `--cache-window N` (256) set how often it rebalances and over how many tokens it counts.
- `--load-test R [--load-pinned]`: a measurement aid: loads random experts into spare GPU slots
  at R loads/s in the background, as the cache would (README step 30); output unchanged.

Without `--rt` (or with `bin/ds-completion`) everything runs on the CPU.

With `--placement placements/study30.txt` the four GPUs hold the routed experts used most in
the 30-prompt routing study (43% of all, 10-30 s upload); 17.7-21.7 t/s on the test prompts (README step 18).

## Current results (2026-10-29)

| mode | command | decode | KL vs reference*: median / mean without the near-tie step |
|---|---|---|---|
| **+ adaptive expert cache** | `... --placement placements/study30.txt --dspark ... --spec --canon --cache-loads 200` | **19.9-30.6 t/s, mean 25.6** (300 tokens, held-out prompts; 29.1 at sclk 8) | output byte-identical to the plain `--canon` run |
| + speculative decoding | `... --placement placements/study30.txt --dspark ... --spec` | 16-24 t/s (300 tokens, held-out prompts; +22% geometric mean) | output byte-identical to the row below |
| own runtime + experts placed on GPUs | `bin-hip/ds-completion --rt ... --placement placements/study30.txt` | 17.7-21.7 t/s (128 tokens) | same numerics: per-layer gate identical to below |
| own runtime, dense on GPU 0 | `bin-hip/ds-completion --rt ...` | 12.0-12.5 t/s | 5.9e-6 / 4.8e-3 (earlier run 1.7e-5 / 6.3e-4) |
| CPU only | `bin/ds-completion` | 3.5 t/s | 1.3e-5 / 3.9e-3 |
| llama.cpp fork (llama-cli, CPU) | - | 3.5 t/s | 2.1e-5 / 1.3e-3 |

\* vs the official reference implementation run on the CPU (README "Step 2"): chat prompt
"Hello, how are you?", 17 steps. Every mode agrees with the reference on the top token at 16
of 17 steps. The miss is one near-tie (step 12), which dominates any plain mean, so the table
leaves it out.
Means over all 17 steps: 1.8e-2 to 2.5e-2 for every mode.

- **The runtime is also gated per layer:** attention and FFN output within ~1e-7 of the fp32 +
  fake-quant reference. It has no inexact op left, and repeated runs are bit-identical.
- **Prompt processing:** with `--rt` the prompt is decoded token by token, ~10 t/s; ~7.7 t/s
  on the CPU (one chunk). The runtime's mean KL varies run to run with fp8 rounding flips (README
  step 10); every exact mode scatters the same way.
- **Where a runtime step (~100 ms) goes:** ~67 ms CPU routed experts, ~29-34 ms GPU and copies
  (depends on the GPU clock level; README step 9).
- **GPU memory with `--rt`:** 6.37 GiB fp8 + 1.68 GiB small weights on die 0.

## Limits

- Context with `--rt`: 65,536 tokens by default (prompt + generation), up to the model's
  1,048,576 with `--max-ctx N` (~6.4 KB of GPU 0 memory per position). The prompt is processed
  one token at a time (~16-17 t/s: a 30k-token prompt takes ~30 min). Without `--rt` (the CPU
  path): 4,096 by default, at most 16k (no candidate-block selection there).
- The dense part uses one GPU; the others only hold experts with `--placement`.
- Sampling is greedy only.

Details: [OVERVIEW.md](OVERVIEW.md) (a 20-minute summary), [README.md](README.md) (all measurements and checks), [PLAN.md](PLAN.md) (what's next).
