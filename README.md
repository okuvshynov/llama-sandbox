# llama sandbox

A collection of experiments related to LLM inference.

## Projects

* [llama-duo](llama-duo/) - asynchronous/distributed speculative decoding for llama3
* [moe-inspect](moe-inspect/) - estimate per-token bytes read for GGUF models (MoE-aware)
* [quant-sampling](quant-sampling/) - find optimal sampling temperature for quantized models via KL divergence
* [kv-transfer](kv-transfer/) - test KV cache transfer between quantization levels
* [gemma4-2b-bench](gemma4-2b-bench/) - prefill and token generation throughput benchmark for Gemma4-E2B across quant levels
* [validation-bench](validation-bench/) - AI coding benchmark harness evaluating models on code generation tasks via tool calling
