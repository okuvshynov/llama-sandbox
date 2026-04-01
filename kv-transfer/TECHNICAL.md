# KV Transfer: Technical Report

## Motivation

When running quantized LLMs, the model processes both the prompt and generates the response using the same low-precision weights. But prompt processing and generation have different characteristics:

- **Prompt processing** is a one-time cost per request. It builds the KV cache that the model attends to during generation.
- **Generation** is the iterative, latency-sensitive phase where tokens are produced one at a time.

This experiment tests whether **processing the prompt with a higher-quality model and transferring the KV cache** to a lower-quality model for generation improves output quality compared to using the lower-quality model end-to-end.

The practical scenario: a deployment where prompt processing can use more compute (e.g., a larger GPU, batched prefill server, or simply a higher-precision model) while generation runs on cheaper/smaller hardware.

## Methodology

### Three-way comparison

For each prompt, we run three configurations against a reference model (Q8\_K\_XL):

1. **ref** — prompt + generation with Q8 (reference). This produces the token sequence and logits we treat as ground truth.
2. **target** — replay the same token sequence through the target model (e.g., IQ2\_XXS). Collect logits at every position. This is the "just use the smaller model" baseline.
3. **handoff** — process the prompt with Q8, save the KV cache via `llama_state_save_file`, load it into a context created from the target model, then replay generation tokens through the target model. Prompt logits come from Q8, generation logits from the target with Q8's KV cache.

### KV cache transfer mechanism

- llama.cpp's `llama_state_save_file` / `llama_state_load_file` serializes the full KV cache state.
- KV cache precision (default F16) is a **context parameter**, not tied to model weight quantization.
- Both models use the same default F16 KV type, so the state transfers cleanly.
- The models must share the same architecture (layer count, embedding dimensions).
- The reference model is freed before loading the target model, so peak memory is max(ref, target), not ref+target.

### Metrics

- **KL divergence** — KL(ref || other) computed at each token position, averaged across all positions. Measures how different the output distribution is from the reference. Lower is better.
- **Top-1 agreement** — percentage of positions where the most likely token matches the reference. Higher is better.

### Prompts

12 test prompts across two categories:

**Short prompts** (200-750 tokens): code analysis tasks in Python, Rust, Go, C++, JS, C, Java. Each asks the model to find bugs or explain an algorithm.

**Large prompts** (1800-2200 tokens): full codebase reviews with ~200 lines of code. Python cache store, Go worker pool, TypeScript task pipeline, Rust metrics library.

## Results

### Qwen3.5-2B (dense, 2.8B parameters)

Reference: UD-Q8\_K\_XL (2704 MB)

#### Overall KL divergence (averaged across 12 prompts)

| Target | Size (MB) | KL (target) | KL (handoff) | KL ratio | Top-1% (tgt) | Top-1% (hoff) |
|--------|----------|-------------|-------------|---------|--------------|---------------|
| ud-iq2\_xxs | 733 | 0.884 | 0.309 | 0.35 | 72.4% | 87.7% |
| ud-q2\_k\_xl | 922 | 0.289 | 0.102 | 0.35 | 85.5% | 93.0% |
| ud-q3\_k\_xl | 1106 | 0.071 | 0.024 | 0.33 | 93.3% | 96.7% |
| ud-q4\_k\_xl | 1278 | 0.022 | 0.007 | 0.32 | 96.5% | 98.2% |
| ud-q5\_k\_xl | 1399 | 0.008 | 0.003 | 0.34 | 97.7% | 98.8% |
| ud-q6\_k\_xl | 1778 | 0.002 | 0.001 | 0.40 | 98.8% | 99.3% |

#### Prompt length effect

The handoff benefit scales dramatically with prompt length:

| Prompt size | Avg prompt tokens | KL ratio (iq2\_xxs) | Top-1% (hoff, iq2\_xxs) |
|------------|------------------|--------------------|-----------------------|
| Short (01-08) | ~443 | 0.42 | 85.2% |
| Large (09-12) | ~2058 | 0.15 | 94.9% |

With large prompts, ~80% of the total context is covered by the reference model's KV cache, so the target model's own errors are diluted.

### Qwen3.5-35B-A3B (MoE, 35B total / 3B active)

Reference: UD-Q8\_K\_XL (46.4 GB)

The 35B MoE model shows even stronger results because the Q8 reference is much closer to "true" full precision for this architecture.

| Target | Size (GB) | KL (target) | KL (handoff) | KL ratio | Top-1% (tgt) | Top-1% (hoff) |
|--------|----------|-------------|-------------|---------|--------------|---------------|
| ud-iq2\_xxs | 10.2 | 0.076 | 0.024 | 0.31 | 93.5% | 97.4% |
| ud-q2\_k\_xl | 12.6 | 0.025 | 0.008 | 0.30 | 96.8% | 98.6% |
| ud-q3\_k\_xl | 14.4 | 0.010 | 0.003 | 0.31 | 97.8% | 99.0% |
| ud-q4\_k\_xl | 16.4 | 0.003 | 0.001 | 0.32 | 98.9% | 99.5% |
| ud-q5\_k\_xl | 18.5 | 0.001 | 0.000 | 0.29 | 99.3% | 99.7% |
| ud-q6\_k\_xl | 22.2 | 0.000 | 0.000 | 0.28 | 99.7% | 99.9% |

## KV Cache Decay Analysis

The benefit of the reference model's KV cache is expected to decay during generation, as the target model adds its own (lower-quality) KV entries.

We split generation tokens into 64-token windows and compute the KL ratio (handoff/target) per window:

### Qwen3.5-2B, IQ2\_XXS, averaged across all 12 prompts

| Window | KL ratio | Interpretation |
|--------|---------|----------------|
| 0-64 | 0.50 | Strong benefit |
| 64-128 | 0.55 | Strong benefit |
| 128-192 | 0.68 | Moderate benefit |
| 192-256 | 0.80 | Declining |
| 256-320 | 0.85 | Declining |
| 320-384 | 0.88 | Marginal |
| 384-448 | 0.90 | Marginal |
| 448-512 | 0.95 | Near parity |

The benefit starts strong (~50% KL reduction in the first 64 tokens) and fades toward parity by the end of 512 tokens. This is consistent with the KV cache being "diluted" as new entries from the weaker model accumulate.

**Note:** The decay rate varies significantly by prompt. Some prompts maintain strong benefit throughout (ratio stays ~0.6-0.7), while others show rapid decay. This may depend on how much the generation depends on long-range context from the prompt vs. recent local context.

## Key Findings

1. **KV transfer consistently improves generation quality.** Across all quant levels and both model families, the handoff KL is 0.28-0.40x the target KL. The improvement is not diminishing — it's roughly a constant factor across quant levels.

2. **Larger prompts benefit more.** With 2000-token prompts, the KL ratio drops to ~0.15 for aggressive quants. The reference model's KV cache covers a larger fraction of total context.

3. **The benefit decays during generation** but remains positive for the entire 512-token window we tested. For longer generation, the benefit will eventually reach parity.

4. **MoE models show the same pattern** as dense models, with even stronger absolute quality at each quant level.

## Open Questions

### Practical deployment

- **Latency tradeoff**: processing the prompt with a larger model is slower. Is the quality improvement worth the latency cost? This depends on the prompt:generation ratio — long prompts with short generations benefit most.
- **State file size**: the KV state file can be large (hundreds of MB for long prompts). In a distributed setup, this transfer time matters.
- **Mixed-precision KV**: instead of full F16 KV cache, could we quantize the KV cache itself during transfer? llama.cpp supports Q8\_0 and Q4\_0 KV types.

### Further experiments

- **BF16 reference**: our reference is Q8, not true BF16. Using BF16 as reference would show the full gap and potentially larger handoff benefits.
- **Longer generation**: test with 1024-2048 generation tokens to characterize the full decay curve.
- **Different model families**: test non-Qwen architectures (Llama, Mistral) to verify the pattern is architecture-independent.
- **Cross-model transfer**: what happens if the reference and target are different model sizes within the same family (e.g., 35B prompt → 2B generation)?

## Reproduction

```bash
# build
export LLAMA_CPP_DIR=/path/to/llama.cpp
cd kv-transfer
cmake -B build && cmake --build build -j 8

# run Qwen3.5-2B study
./run-qwen3.5-2b.sh

# run Qwen3.5-35B-A3B study
./run-qwen3.5-35b-a3b.sh

# view results
cd results && python3 -m http.server 8080 --bind 0.0.0.0
# open view.html for bar charts, decay.html for decay curves
```

## File Format

The `.bin` files use a custom trace format (v3) storing:
- Global header: magic, version, n\_vocab, n\_prompts, sampling params
- Per-prompt sections: path string, n\_tokens, n\_prompt, token IDs, full logit vectors

This format is compatible with the quant-sampling subproject's compare tool.
