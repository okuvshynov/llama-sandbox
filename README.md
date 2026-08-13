# llama sandbox

A collection of experiments related to LLM inference.

## Projects

* [llama-duo](llama-duo/) - asynchronous/distributed speculative decoding for llama3
* [moe-inspect](moe-inspect/) - estimate per-token bytes read for GGUF models (MoE-aware)
* [quant-sampling](quant-sampling/) - find optimal sampling temperature for quantized models via KL divergence
* [kv-transfer](kv-transfer/) - test KV cache transfer between quantization levels
* [gemma4-2b-bench](gemma4-2b-bench/) - prefill and token generation throughput benchmark for Gemma4-E2B across quant levels
* [validation-bench](validation-bench/) - AI coding benchmark harness evaluating models on code generation tasks via tool calling
* [server-n-bench](server-n-bench/) - performance testing for llama.cpp server for multiple completions of the same prompt
* [llama-variance](llama-variance/) - single-shot variance study: n=N completions per request, scored independently against a fixed task, to study how much of a local model's score is sample noise
* [mini-sql-bench](mini-sql-bench/) - one-task smoke harness on mini-swe-agent: drives a Docker-isolated SQLite query task to verify the agent loop works against any model provider
* [logit-kld](logit-kld/) - logit collection for cross-model KL divergence: greedy continuation with per-position top-K logits + log-sum-exp normalizer, raw token ids as the interface to framework-agnostic rescoring
* [nano-glm](nano-glm/) - minimal CPU-only inference on bare ggml for two architectures (GLM-5.2 and DeepSeek-V4-Flash), with the routed experts optionally served by a separate backend over TCP: loader, KV cache, trunk graph and greedy loop with no llama.cpp framework layer, gated bit-identical against llama.cpp over a 6-prompt corpus
* [checksums](checksums/) - SHA-256 manifests for the models everything here measures against, plus the tools to localize corruption when a copy goes wrong
* [moe-offload](moe-offload/) - cross-platform (Vulkan) measurement of what a GPU charges to run MoE routed experts while the trunk stays on CPU: per-phase dispatch/transfer/fence costs, macOS MoltenVK vs Windows AMD driver
* [moe-serv](moe-serv/) - a ggml backend, loaded into unmodified llama.cpp via `GGML_BACKEND_PATH`, that takes ownership of the routed expert weights and computes the whole MoE block — so the experts can be tuned or moved elsewhere without a trunk of our own
