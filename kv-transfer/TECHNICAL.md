# KV Transfer: Technical Report

## Motivation

When running quantized LLMs, the model processes both the prompt and generates the response using the same low-precision weights. But prompt processing and generation have different characteristics:

- **Prompt processing** is a one-time cost per request. It builds the KV cache that the model attends to during generation.
- **Generation** is the iterative, latency-sensitive phase where tokens are produced one at a time.

This experiment tests whether **processing the prompt with a higher-quality model and transferring the KV cache** to a lower-quality model for generation improves output quality compared to using the lower-quality model end-to-end.

The practical scenario: a deployment where prompt processing can use more compute (e.g., a larger GPU, batched prefill server, or simply a higher-precision model) while generation runs on cheaper/smaller hardware.

## Methodology

### Three-way comparison

For each prompt, we run three configurations against a reference model (BF16 where available, Q8 otherwise):

1. **ref** — prompt + generation with the reference model. This produces the token sequence and full logits we treat as ground truth. Logits are saved to a trace file for reuse across all target quants.
2. **target** — replay the same token sequence through the target model (e.g., Q4\_K\_XL) in batches. KL divergence and top-1 agreement are computed per token inline against the ref logits. Only per-token stats are saved (~10 KB per file).
3. **handoff** — process the prompt with the ref model, save the KV cache via `llama_state_save_file`, load it into a context created from the target model, then replay generation tokens through the target model in batches. KL is computed inline, same as target.

### KV cache transfer mechanism

- llama.cpp's `llama_state_save_file` / `llama_state_load_file` serializes the full KV cache state.
- KV cache precision (default F16) is a **context parameter**, not tied to model weight quantization.
- Both models use the same default F16 KV type, so the state transfers cleanly.
- The models must share the same architecture (layer count, embedding dimensions).
- The reference model is freed before loading the target model, so peak memory is max(ref, target), not ref+target.

### Metrics

- **KL divergence** — KL(ref || other) computed at each generation token position using temperature-scaled log-softmax. Lower is better.
- **Top-1 agreement** — percentage of positions where the most likely token matches the reference. Higher is better.
- **KL ratio** — KL(handoff) / KL(target). Values below 1.0 indicate handoff improves over plain target. This is the primary metric for measuring the KV transfer benefit.

### Prompts

19 test prompts across three size categories:

- **Small** (8 prompts, ~300-800 tokens): code analysis tasks in Python, Rust, Go, C++, JS, C, Java. Each asks the model to find bugs or explain code.
- **Medium** (6 prompts, ~2000-3600 tokens): multi-file codebase reviews with threading, caching, and database patterns.
- **Large** (5 prompts, ~8000-10000 tokens): full application reviews — microservices, async schedulers, document management APIs, lock-free data structures.

All prompts are formatted with the model's chat template (jinja-based) before tokenization.

### Models tested

| Model | Type | Total params | Active params | Reference | Target quants |
|-------|------|-------------|---------------|-----------|---------------|
| Gemma4-E2B | Dense | 2B | 2B | BF16 | Q2-Q6, Q8 |
| Gemma4-E4B | Dense | 4B | 4B | BF16 | Q2-Q6, Q8 |
| Gemma4-31B | Dense | 31B | 31B | BF16 | Q2-Q6, Q8 |
| Gemma4-26B-A4B | MoE | 26B | 4B | BF16 | Q2-Q6, Q8 |
| Qwen3.5-2B | Dense | 2B | 2B | Q8 | IQ2, Q2-Q6 |
| Qwen3.5-27B | Dense | 27B | 27B | Q8 | Q2-Q6 |
| Qwen3.5-35B-A3B | MoE | 35B | 3B | Q8 | Q2-Q6 |

Generation: 2048 tokens per prompt. Sampling: temperature 0.6 (Qwen) or 1.0 (Gemma4), top-k 40/64, top-p 0.95.

## Key Findings

1. **KV transfer consistently improves generation quality.** Across all quant levels and all model families, the handoff KL is typically 0.3-0.5x the target KL at the start of generation. The improvement is roughly a constant factor across quant levels.

2. **Larger prompts benefit more.** With large prompts (~8000+ tokens), the KL ratio is lower because the reference model's KV cache covers a larger fraction of total context. The target model's own errors are diluted by the high-quality prompt KV cache.

3. **The benefit decays during generation** as the target model adds its own lower-quality KV entries. Across all models, the KL ratio starts at 0.5-0.7 and approaches 0.8-0.9 by token 2000. The decay rate varies by prompt — some maintain strong benefit throughout, others show rapid convergence.

4. **The decay pattern is consistent across architectures.** Dense and MoE models, Qwen and Gemma families, 2B to 31B parameters — all show similar decay curves when aggregated across quants. The benefit is not architecture-specific.

5. **KL divergence itself decreases with generation length** for both target and handoff. This is because the growing shared context (the replayed token sequence) increasingly constrains predictions, making both models converge regardless of weight differences.

## KV Cache Decay Analysis

We split generation tokens into 64-token windows and compute the KL ratio (handoff/target) per window, averaged across all prompts.

The decay follows a consistent pattern:
- **Tokens 0-256**: Strong benefit (ratio 0.5-0.7). The ref model's KV cache dominates.
- **Tokens 256-1024**: Gradual decline (ratio 0.7-0.85). Target model's own KV entries accumulate.
- **Tokens 1024-2048**: Approaching parity (ratio 0.8-0.95). But handoff remains measurably better even at 2048 tokens.

The decay rate depends on the prompt-to-generation ratio. With a 8000-token prompt and 2048-token generation, the ref KV cache covers ~80% of total context, so the benefit persists longer.

## Open Questions

### Practical deployment

- **Latency tradeoff**: processing the prompt with a larger model is slower. Is the quality improvement worth the latency cost? This depends on the prompt:generation ratio — long prompts with short generations benefit most.
- **State file size**: the KV state file can be large (hundreds of MB for long prompts). In a distributed setup, this transfer time matters.
- **Mixed-precision KV**: instead of full F16 KV cache, could we quantize the KV cache itself during transfer? llama.cpp supports Q8\_0 and Q4\_0 KV types.

### Further experiments

- **BF16 reference for Qwen family**: the Qwen models currently use Q8 as reference. BF16 would show the full quantization gap.
- **Cross-model transfer**: what happens if the reference and target are different model sizes within the same family (e.g., 35B prompt → 2B generation)?
- **Longer generation**: characterize the full decay curve beyond 2048 tokens.

## Reproduction

```bash
# build
export LLAMA_CPP_DIR=/path/to/llama.cpp
cd kv-transfer
cmake -B build && cmake --build build -j 8

# run a single model family
./run-gemma4-e2b.sh

# run a single prompt for quick testing
PROMPT_FILTER="01_python_small" ./run-gemma4-31b.sh

# view interactive results
cd results && python3 -m http.server 8080 --bind 0.0.0.0

# generate static plots
python3 plot_models.py --results-dir results --output-dir plots
```

## Implementation notes

### Inline KL computation

Target and handoff runs load the ref trace file (with full logits), compute KL divergence per token during inference, and write only compact stats files. This reduces output from ~1 GB (full logits) to ~10 KB (per-token KL + top-1 match) per run, a ~500,000x reduction.

### Batched evaluation

Both target and handoff process tokens in batches (not one-by-one). The handoff command was originally autoregressive (token-by-token decode) but since we replay a fixed sequence, batched decode is equivalent and ~4x faster. The batch size is determined by the llama.cpp context configuration.

### Timing

Each command prints `llama_perf_context_print` data to stderr (captured in log files), showing model load time, prompt eval time, and total time. The `run.sh` script prints a timing summary at the end parsing these logs, which helps identify model loading overhead.

For MoE models (e.g. Qwen3.5-35B-A3B), model loading can be ~24% of total time due to the large model file relative to per-token compute. The `PROMPT_FILTER` env var enables single-prompt test runs to verify setup before committing to a full 19-prompt run.
