# moe-inspect — Build & Usage

Standalone utilities for inspecting GGUF model files, using llama.cpp's ggml library.

## Prerequisites

- A built llama.cpp tree (only `libggml-base` is needed)
- CMake >= 3.14
- C++17 compiler

## Build

```bash
# Set LLAMA_CPP_DIR once (e.g. in your shell profile), or pass it to cmake:
export LLAMA_CPP_DIR=$HOME/projects/forks/llama.cpp

# From this directory:
cmake -B build
cmake --build build
```

You can also pass the path explicitly: `cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp`.
`LLAMA_CPP_BUILD_DIR` defaults to `$LLAMA_CPP_DIR/build` but can be overridden the same way.

## Tools

### moe-eval-size

Estimate how many bytes are read per forward pass for a GGUF model.
Particularly useful for MoE models where only a subset of experts is active per token.

Only reads GGUF metadata (tensor names, types, shapes) — no tensor data is loaded,
so it runs instantly even on very large models.

Supports split GGUF files — pass any shard and it auto-discovers the rest.

```bash
# Single-file model
./build/moe-eval-size ~/projects/llms/Qwen3.5-122B-A10B-UD-IQ1_M.gguf

# Split model (pass any shard)
./build/moe-eval-size ~/projects/llms/qwen-3.5/Qwen3.5-397B-A17B-UD-IQ1_M/Qwen3.5-397B-A17B-UD-IQ1_M-00001-of-00004.gguf

# Show every tensor
./build/moe-eval-size <model.gguf> --verbose
```

#### Output sections

1. **Model info** — architecture, layer count, expert counts
2. **Quantization type summary** — how many tensors of each type, total bytes
3. **Category breakdown** — bytes grouped by role (attention, expert, shared expert, etc.)
4. **Forward pass estimate** — bytes read per single-token evaluation, accounting for MoE sparsity

## How it works

For MoE models, expert weight tensors (named `*_exps`) contain all experts stacked together.
During inference, only `n_expert_used` out of `n_expert` are activated per token.
The tool computes:

```
bytes_per_forward = always_read + (expert_total × n_expert_used / n_expert)
```

Where `always_read` includes: embeddings, attention projections, SSM/Mamba layers, norms,
shared experts, MoE router gates, and the output head.

## Tensor categories

The tool classifies tensors by name pattern:

| Category | Pattern | Always read? |
|---|---|---|
| embedding | `token_embd*` | yes |
| attention | `attn_*` | yes |
| ssm | `ssm_*` (Mamba layers) | yes |
| ffn_dense | `ffn_*` (non-MoE) | yes |
| ffn_expert | `*_exps` | partial (n_used/n_total) |
| ffn_shared_expert | `*_shexp` | yes |
| moe_gate | `ffn_gate_inp*` | yes |
| norm | `*_norm` | yes |
| output_head | `output.*` | yes |

## Linking

Only links against `libggml-base` from llama.cpp's build tree — no llama.cpp runtime
or backend libraries needed. The GGUF API provides all tensor metadata access.
