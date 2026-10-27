# ds-completion quickstart

DeepSeek-V4.1-Flash-1017 text completion on the Mac Pro (4x Vega II, 755 GB RAM).

- Dense weights (attention, shared experts, head) run on one GPU through our own HIP kernels,
  in exact fp8.
- Routed experts run on the CPU, through our own exact AVX-512 kernel.
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

Experiment: `--gpu-moe 0:25-27,1:36-39,2:32-35,3:28-31` also puts all experts of 15 layers on
the four GPUs (~22 s upload), for 12.0 t/s.

## Current results (2026-10-27)

| mode | command | decode | KL vs reference*: median / mean without the near-tie step |
|---|---|---|---|
| **own runtime, dense on GPU 0** | `bin-hip/ds-completion --gpu 0 --rt ...` | **9.7-10.0 t/s** | 5.9e-6 / 4.8e-3 (earlier run 1.7e-5 / 6.3e-4) |
| ggml, dense on GPU 0 | `bin-hip/ds-completion --gpu 0` | 6.0 t/s | 1.2e-5 / 6.0e-3 |
| CPU only | `bin/ds-completion` | 3.5 t/s | 1.3e-5 / 3.9e-3 |
| llama.cpp fork (llama-cli, CPU) | - | 3.5 t/s | 2.1e-5 / 1.3e-3 |

\* vs the official reference implementation run on the CPU (README "Step 2"): chat prompt
"Hello, how are you?", 17 steps. Every mode agrees with the reference on the top token at 16
of 17 steps. The miss is one near-tie (step 12), which dominates any plain mean, so the table
leaves it out.
Means over all 17 steps: 1.8e-2 to 2.5e-2 for every mode.

- **The runtime is also gated per layer:** attention and FFN output within ~1e-7 of the fp32 +
  fake-quant reference. It has no inexact op left, and repeated runs are bit-identical.
- **Prompt processing:** with `--rt` the prompt is decoded token by token, ~10 t/s; ~6-8 t/s
  in the ggml modes. The runtime's mean KL varies run to run with fp8 rounding flips (README
  step 10); every exact mode scatters the same way.
- **Where a runtime step (~100 ms) goes:** ~67 ms CPU routed experts, ~29-34 ms GPU and copies
  (depends on the GPU clock level; README step 9).
- **GPU memory with `--rt`:** 6.37 GiB fp8 + 1.68 GiB small weights on die 0.

## Limits

- Context: at most 4096 tokens with `--rt` (the top-k kernel), 16k otherwise (candidate-block
  selection is not implemented).
- Only one GPU is used.
- Sampling is greedy only.

Details: [README.md](README.md) (all measurements and checks), [PLAN.md](PLAN.md) (what's next).
