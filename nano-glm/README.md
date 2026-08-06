# nano-glm — minimal CPU-only GLM-5.2 inference on bare ggml

An experiment in going back to basics: llama.cpp grew from a single-file
inference engine into a framework (142 model architectures, a KV-cache zoo,
15+ accelerator backends, batch/memory abstractions). nano-glm asks how small
a *single-model* engine gets if the compute layer is kept and the framework
is dropped: it links only `ggml` (kernels, GGUF reader, backend scheduler)
and reimplements the thin slice of llama.cpp that GLM-5.2 on CPU actually
needs — shard loader, single-sequence KV cache, forward graph, greedy loop —
in one source file.

Correctness bar: **bit-identical logits** vs the llama.cpp-based `collect`
baseline from `../logit-kld`, verified over the same prompts with
`compare.py` / `cmp`. That is why the forward graph is an op-for-op port of
llama.cpp's glm-dsa trunk graph (`src/models/glm-dsa.cpp` plus the
`llm_graph_context` helpers inlined), with the same op order, the same
seemingly-redundant views, and the same runtime configuration the baselines
ran under:

- flash attention ON (f16 masks, F32 FA precision) — what `auto` resolved to;
- fused lightning indexer (`ggml_lightning_indexer`) ON;
- F16 K caches (MLA: 576/row; indexer: 128/row with the arch-forced 128×128
  orthonormal Walsh-Hadamard rotation), zero-initialized, n_kv padded to 256;
- BLAS (Accelerate) offload for batches ≥ 32 via `ggml_backend_sched` with
  the same [BLAS, CPU] backend priority as llama.cpp;
- llamafile sgemm ON, Metal OFF (CPU-only by construction — see repo
  CLAUDE.md for the AMD-Metal NaN story).

What is deliberately **not** here: tokenizer (raw token ids are the
interface, same policy as logit-kld — take them from an existing lkldtopk
file or pass a comma list), chat templates, samplers (greedy = stored top-1),
batching across sequences, KV shifting/defrag, the NextN/MTP draft head, and
every non-glm-dsa architecture.

## Build

```bash
cmake -B build -DLLAMA_CPP_DIR=$HOME/projects/llama.cpp
cmake --build build -j
```

Only `${LLAMA_CPP_DIR}/ggml` is used; ggml stays unmodified, so upstream
kernel improvements keep flowing.

## Run

```bash
# prompt token ids from an existing logit-kld collect file (its n_prompt tokens):
./build/nano-glm -m .../GLM-5.2-UD-Q6_K-00001-of-00014.gguf \
    -i ../logit-kld/results/glm-5.2-ud-q6_k.n256.bin -n 256 -o nano.bin

# or raw ids:
./build/nano-glm -m .../GLM-5.2-UD-Q6_K-00001-of-00014.gguf -T "151331,98328,3837" -n 32
```

Flags: `-o` output lkldtopk file, `-n` tokens to generate, `-k` top-K stored
(128), `-c` context (4096, auto-raised), `-b` prompt chunk size (512), `-t`
threads (all cores). Generated token ids stream to stdout (no tokenizer —
no text); summary to stderr.

Output is an `lkldtopk` v1 file (writer shared with logit-kld), so the whole
logit-kld toolchain applies: `inspect.py` for sanity, `compare.py` against
any collect/rescore file over the same token sequence.

## Verification

```bash
# A/A vs the llama.cpp baseline: same prompt ids, same batch shapes
./build/nano-glm -m <model> -i <baseline.bin> -n 256 -o nano.bin
python ../logit-kld/compare.py <baseline.bin> nano.bin   # expect KL ~ 0
cmp <baseline.bin> nano.bin                              # bit-identity modulo model_desc
```

The graph replicates llama.cpp's execution shape for the collect workload
(prompt prefilled in `-b` chunks, generation one token per batch), which the
logit-kld A/A study showed is the regime where CPU llama.cpp is bit-exact
across runs. Divergence beyond the `model_desc` header bytes means a port
bug, and per-position KL points at where.

## Scope notes

- hparams are read from GGUF metadata but structural assumptions are
  asserted loudly (arch == glm-dsa, sigmoid expert gating, the degenerate
  freq_scale==1 rope case). Anything else aborts rather than mis-computes.
- The indexer-types layout (which layers run a full lightning indexer vs
  reuse the previous top-k) comes from GGUF metadata, with the GLM-5.2
  default pattern as fallback — same BC rule as llama.cpp.
- Sparse DSA attention is exercised end-to-end even when n_kv < indexer
  top_k (2048): the top-k mask path always runs; it just selects everything
  for short sequences.
