# ds-completion — DeepSeek-V4.1-Flash inference for the Mac Pro (755 GB RAM, 4x Vega II)

Goal: `./ds-completion -m <model> -p "prompt"`. One model, one machine, one
sequence. Priorities: correctness, clarity, then performance. Non-goals: other
models, other quants, other hardware, serving features.

Status: step 1 (loader + weight provenance) done; see "Step 1" below.

## Tools

```bash
make                                      # links the fork's CPU ggml (static)
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/ds-inspect -m $M [--layers 0,1,2-5] [--touch]
~/venvs/ds-ref/bin/python ref/check_weights.py [--layers 0,1,2] [--experts 8|all] [--hf remote|local]
```

## What the model is (measured from the files, 2026-10-26)

- 40 blocks, hidden 5120, 64 heads x 512 head_dim, a single shared KV latent
  (MQA-like). q is low-rank (5120 -> 1280 -> 64x512), and the output projection
  is grouped low-rank (8 groups, 4096 -> 1024 each, then 8192 -> 5120).
- **Hyper-connections**: the residual is 4 parallel copies of 5120. Each
  sublayer mixes them with coefficients from a 20480 -> 24 projection plus 20
  Sinkhorn iterations.
- **Attention**:
  - Every layer has a 128-token sliding window.
  - Layers 2-19 also see KV compressed 2:1, and layers 20-39 see it 1:1.
  - Only layers 2, 8, 14 and 20 build the compressed KV; the others reuse it.
  - An indexer (32 heads x 128) picks the top 512 compressed positions. It
    runs in 8 layers (2, 8, 14, 20, 24, 28, 32, 36), the others reuse the
    result.
  - Layer 20 also pre-selects 2048 candidate blocks of 8 for the later
    indexers.
  - Each head has an attention sink.
- **MoE**: 384 routed experts, top-6, plus 1 shared expert, inter 2304.
  - Scores are sqrt(softplus); the bias is used only for selection.
  - Routing weights are normalized, x1.5.
  - SwiGLU is clamped at 10.
- **Engram** (layers 1 and 14):
  - The last 2-4 tokens, mapped to a 99k "normalized" vocab, are hashed into
    24 rows of 256.
  - Those rows go through a 6144 -> 25600 projection, which gives a key per
    hc copy plus one value.
  - A gated add puts the value into the residual.
- MTP / DSpark draft layers exist in the safetensors but **not** in the GGUF.

| part | GGUF type | size | bytes read per decode token |
|---|---|---|---|
| routed experts | mxfp4 (= original fp4) | 269 GiB | 4.5 GB (6 x 40 x 18.8 MB) |
| engram table | mxfp8 (= original fp8) | 189 GiB | ~0 (48 rows) |
| attention + indexer | bf16 (original **fp8**, 32x32 e8m0 scales) | 9.6 GiB | 10.1 GB |
| shared experts | bf16 (original **fp8**) | 2.6 GiB | 2.8 GB |
| head / embed | bf16 (original bf16) | 1.2 + 1.2 GiB | 1.3 GB |
| engram proj, router, hc, norms | bf16 / f32 | ~1 GiB | ~1 GB |

- **The GGUF is lossless against the original** (fp4 -> mxfp4, fp8 -> mxfp8,
  fp8 -> bf16).
- **About half of each decode token's bytes are bf16 copies of fp8 weights.**
  Packing them back to fp8 is lossless and cuts the dense reads from ~15 GB
  to ~8 GB per token.

Machine:

- RAM: 755 GB. The whole GGUF is currently in the page cache.
- CPU read bandwidth: 112 GB/s (16 threads; measured today).
- GPUs: 4 x 32 GB HBM (~880 GB/s/die), XGMI within each card (0-1, 2-3).
  Cross-card traffic goes through host memory (see `macpro-2019-vega-if`).

Ceilings this implies:

- **CPU only:** ~20 GB/token at 112 GB/s gives ~5.5 t/s. The fork measures
  3.5 t/s.
- **Dense on a GPU, experts on the CPU:** the 4.5 GB of experts alone is
  ~40 ms of CPU reads, so ~20 t/s.
- **Faster than that** requires experts on the dies. After the dense weights,
  the 128 GB of HBM fits ~40% of them.

Prior work worth re-reading, with caution (it covers the previous V4-Flash,
150 GiB, which fit mostly in HBM, and ran without XGMI):

- `hip-moe/README.md`:
  - Stock llama.cpp with `-ncmoe 13` reached 10 t/s.
  - The fork's DSV4 ops (HC, lightning indexer) compile and run under HIP.
  - CPU mxfp4 experts reach ~95 GB/s with CPU_REPACK and 32 threads.
  - The ROCm 5.7 runtime has a lower dispatch overhead than 6.3.

## Correctness ladder

1. **Reference, `ref/`:** the official `inference/model.py` math, fed with weights decoded from
   the GGUF by `ref/weights.py` (see "Which checkpoint" below). tilelang does not
   run on gfx906, so a pure-torch CPU `kernel_ref.py` replaces `kernel.py`
   (act_quant, fp4/fp8 gemm, sparse_attn, sinkhorn).
   - It runs **one layer at a time**: it decodes the layer's dense weights and
     the 6 routed experts actually used, runs, then frees them.
   - That makes the whole model feasible on CPU, slowly. It produces golden
     per-layer activations and final logits for fixed prompts.
2. **Second opinion:** the fork, through libllama: logits, plus per-tensor
   dumps via the eval callback. This tells "my bug" apart from "the fork's
   bug".
3. **Gates:**
   - Each layer is checked against the reference, given the same input
     (relative error, per sublayer).
   - End to end: top-k logits and KL against the reference on a handful of
     prompts, plus greedy token agreement.
   - Every performance step re-runs the gates against the previous step:
     bit-identical where only placement changed, a tolerance where the
     kernels changed.
4. **Discrete decisions are checked separately:** expert ids (router top-6),
   indexer top-512 and the candidate blocks. A tiny numerical change can flip
   them, so we log the mismatch rate rather than only the output error.

Precision choice to make once the gates exist:

- The reference fake-quantizes activations to fp8 before every GEMM, stores
  KV as fp8/fp4, and indexer q/k as fp4.
- Plan: emulate the KV, indexer and compressor quantization exactly, since it
  changes which positions get selected, and let GEMMs use ggml's native
  q8_1/f32 path. Then measure what that costs in KL.

## Steps

0. **Setup.**
   - Python venv for the reference (needs the user: torch CPU, numpy,
     safetensors, sympy, tokenizers).
   - Link the fork's ggml + libllama. The CPU build exists; a `build-hip` of
     the fork for gfx906 comes at step 4.
1. **Loader + layer selection.** mmap the 9 GGUF shards (already in the page
   cache) and load only `--layers`. That's the "one layer" instrument without
   copying files. Extraction to a separate file is used only for layout
   experiments (step 5).
   - Test layers that cover every type:
     - 0: window only
     - 1: + engram
     - 2: ratio-2 KV source + indexer
     - 3: ratio-2 consumer
     - 20: ratio-1 source + candidates
     - 21: plain consumer
     - 24: second-level indexer
2. **Reference dumps** (`ref/`): golden activations for those layers and
   prompts, then full-model logits for 3-4 short prompts.
3. **CPU v0** (`ds-completion`, ggml CPU kernels, one graph per layer, own
   KV/state code, greedy):
   - Build order: tokenizer (llama_vocab), embed, HC, attention window path,
     MoE, engram hash + lookup, compressor + compressed KV, indexer,
     candidates, head.
   - Each piece is gated at layer level before moving on.
   - Expected speed: about the fork's 3.5 t/s.
4. **One GPU, no parallelism.**
   - All dense weights on die 0: ~15 GiB bf16, so they fit. Routed experts
     and the engram table stay on the CPU.
   - The shared expert (GPU) overlaps the routed experts (CPU).
   - Per layer and token, the host link carries only ~20 KB in and ~20 KB
     out, so the cost is latency, not bandwidth.
   - Target: >10 t/s.
5. **Layout.**
   - Pack the fp8-derived bf16 weights back to fp8. ggml's MXFP8 is
     gather-only, so this needs our own fp8 GEMV, HIP first. Verify it's
     lossless byte-wise against the safetensors.
   - Record expert-routing histograms on real prompts.
6. **Experts on the dies.**
   - Hot experts resident across the 4 dies (EP), the cold remainder on the
     CPU as a fifth worker.
   - Hit rate from the step-5 histograms.
7. **Parallelism for the dense part.**
   - TP of attention and shared experts within an XGMI pair, then across all
     4 dies.
   - Prefill path: stream expert weights to the dies for large batches.
8. **Maybe:** a DSpark drafter (needs converting the MTP weights from the
   safetensors).

## Which checkpoint (found in step 1)

`~/llms/ds-flash` is the **base DeepSeek-V4.1-Flash** release (HF commit `314113f8`). The GGUF
was converted from **DeepSeek-V4.1-Flash-1017** (HF commit `1bae0e10`, 2026-10-17).

- **Same everywhere else:** config, inference code, tokenizer, encoding and all 96085 tensor
  names are byte-identical between the two repos.
- **The weights differ:** close but not equal (correlation 0.9999997 on the attention sinks).
  Only a handful of tensors are bit-equal across the two, e.g. the router bias.

Consequences:

- The reference *code* in `ds-flash/inference` is the right code.
- Its weights must be 1017's. We take them from the GGUF (`ref/weights.py` decodes it
  independently of ggml), after proving the GGUF equals 1017 exactly.
- Proving it doesn't need a 510 GB download. `HFRemote` reads only the needed tensors from
  huggingface.co using HTTP range requests.
- `ds-flash` safetensors are no longer needed. Their files that matter (inference code,
  tokenizer) are identical to 1017's.

## Step 1: loader + weight provenance

`src/model.{h,cpp}` memory-maps the 9 shards with zero copies. Each shard is wrapped as a ggml CPU
buffer.

- **Strict schema:** every tensor of every selected layer is checked for its exact type and
  shape. Every tensor in the file must be claimed, so unknown structure fails at load.
- **Layer wiring comes from the hparams:** the KV source, index source, candidates and engram
  slot are derived there, and the tensors present must agree with it.
- **Results:**
  - The full model loads in 0.07 s: 1043 tensors, all claimed.
  - Reading through the mappings runs at 91 GB/s (32 threads) from the page cache.

Layer wiring, as `ds-inspect` prints it:

| layers | attention | compressed KV from | top-k from |
|---|---|---|---|
| 0-1 | window only (1: + engram) | - | - |
| 2-7, 8-13, 14-19 | window + 2:1 compressed | 2 / 8 / 14 | same layer as the KV |
| 20-23 | window + 1:1 compressed | 20 | 20 (also picks the candidate blocks) |
| 24-27, 28-31, 32-35, 36-39 | window + 1:1 compressed | 20 | 24 / 28 / 32 / 36, scoring inside 20's candidates |

Also: 14 is an engram layer too, and dense bytes per decode token come to 13.2 GiB of layers
plus 1.2 GiB of head. Routed experts add 4.2 GiB (6 of 384 per layer).

`ref/check_weights.py` (results in `results/check-weights-1017.txt`):

- **Hparams:** all GGUF hparams match config.json.
- **Engram constants:** the GGUF's engram hash constants match what the reference `engram.py`
  computes from scratch: primes, offsets, multipliers and the 129280 -> 99092 compressed token
  map.
- **Every tensor is exact against 1017:** each tensor of layers 0, 1, 2, 3, 14, 20, 21, 24 and 39,
  plus the globals, is bit-exact after decoding both sides. Decoding uses ggml's mxfp4/mxfp8
  block semantics for the GGUF and the reference's fp8/fp4 semantics for the safetensors.
- **Coverage:** routed experts are sampled (first, last and 8 random per tensor). The engram
  table is checked on 3 runs of 1024 rows plus 512 random rows per layer.
- **Nothing dropped:** every HF tensor of those layers is accounted for.
- **Result: ALL OK.** 256 exact comparisons, 6.25 GB fetched, 522 s.
