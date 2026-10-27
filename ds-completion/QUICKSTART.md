# ds-completion quickstart

DeepSeek-V4.1-Flash-1017 text completion on the Mac Pro (4x Vega II, 755 GB RAM).

- Dense weights (attention, shared experts, head) run on one GPU through our own HIP kernels,
  in exact fp8.
- Routed experts run on the CPU.
- Greedy decoding, one sequence, text only.

## What it needs

- **Model:** `~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000{1..9}-of-00009.gguf`
  (474 GB; it runs from the page cache, so the RAM must hold it).
- **llama.cpp fork:** `~/forks/llama.cpp`, built twice:
  - `build/`: CPU only, as it is now;
  - `build-hip/`: ROCm 6.3.4, gfx906, shared libraries, with
    `patches/fork-hip-ballot-sync.patch` applied. The cmake line is in README "Step 4".

## Build and prepare (once)

```bash
cd ~/projects/llama-sandbox/ds-completion
make && make HIP=1                     # -> bin/ (CPU) and bin-hip/ (GPU)
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf   # lossless fp8 repack, 6.4 GB, ~20 s
```

## Run

```bash
bin-hip/ds-completion -m $M --gpu 0 --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    -p "Write a haiku about memory bandwidth." --chat -n 64
```

Options:

- `--chat`: one user turn, no thinking.
- `--think`: with thinking.
- neither: the prompt is raw text after BOS.
- `-n`: max new tokens.
- `-t`: CPU threads; 32 is best here.
- `--profile`: a per-step time breakdown.

Without `--rt` the GPU path is ggml's (slower); without `--gpu`, `bin/ds-completion` is CPU only.

## Current results (2026-10-27)

| mode | command | decode | mean KL vs reference* |
|---|---|---|---|
| **own runtime, dense on GPU 0** | `bin-hip/ds-completion --gpu 0 --rt ...` | **9.4 t/s** | **1.4e-2** |
| ggml, dense on GPU 0 | `bin-hip/ds-completion --gpu 0` | 6.0 t/s | 2.0e-2 |
| CPU only | `bin/ds-completion` | 3.5 t/s | 1.8e-2 |
| llama.cpp fork (llama-cli, CPU) | - | 3.5 t/s | 2.5e-2 |

\* vs the official reference implementation run on the CPU (README "Step 2"): chat prompt
"Hello, how are you?", 17 steps. Every mode agrees with the reference on the top token at 16
of 17 steps; the miss is the same near-tie for all of them.

- **The runtime is also gated per layer:** attention output within ~1e-7 of the fp32 +
  fake-quant reference, the same as the CPU exact path. Repeated runs are bit-identical.
- **Prompt processing** is ~6-8 t/s in every mode (ggml).
- **Where a runtime step (~109 ms) goes:** ~72 ms CPU routed experts, ~37 ms GPU.

## Limits

- Context: at most 4096 tokens with `--rt` (the top-k kernel), 16k otherwise (candidate-block
  selection is not implemented).
- Only one GPU is used.
- Sampling is greedy only.

Details: [README.md](README.md) (all measurements and checks), [PLAN.md](PLAN.md) (what's next).
