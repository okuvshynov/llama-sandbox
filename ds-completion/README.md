# ds-completion — DeepSeek-V4.1-Flash inference for the Mac Pro (755 GB RAM, 4x Vega II)

Goal: `./ds-completion -m <model> -p "prompt"`. One model, one machine, one
sequence. Priorities: correctness, clarity, then performance. Non-goals: other
models, other quants, other hardware, serving features.

Status: steps 1-4 done, 2026-10-26.

- CPU only: 3.5 t/s decode, matching the reference (fork: 3.5 t/s).
- Dense part on one GPU, routed experts on the CPU: **6.0 t/s**.

Performance work from here on is planned in [PLAN.md](PLAN.md).

## Tools

```bash
make                                      # links the fork's CPU ggml (static)
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/ds-completion -m $M -p "Hello, how are you?" --chat -n 64    # or --think, or raw text
make HIP=1   # -> bin-hip/, needs the fork's build-hip (see Step 4)
bin-hip/ds-completion -m $M -p "..." --chat --gpu 0 [--profile]   # dense part on GPU 0
bin/ds-inspect -m $M [--layers 0,1,2-5] [--touch]
bin/ds-test -m $M -g data/golden/hello [--layers 0-39] [--no-quant]  # teacher-forced, per layer
bin/ds-test -m $M -g data/golden/hello --e2e data/ds/hello.f32       # end-to-end logits
bin/test-fakequant                                                   # needs ref/make_fq_test.py first
~/venvs/ds-ref/bin/python ref/check_weights.py [--layers 0,1,2] [--experts 8|all] [--hf remote|local]
~/venvs/ds-ref/bin/python ref/ds_ref.py --prompt "..." --chat --n-decode 16 [--fp32] [--no-quant] --out data/ref/x.pt
~/venvs/ds-ref/bin/python ref/export_golden.py data/ref/x.pt data/golden/x
~/venvs/ds-ref/bin/python ref/compare_logits.py data/ref/x.pt data/ds/x.f32
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
   run on gfx906, so a pure-torch CPU `ref/kernel_cpu.py` replaces `kernel.py`
   (act_quant, fp4/fp8 gemm, sparse_attn, sinkhorn).
   - It runs **one layer at a time**: it decodes the layer's dense weights and
     the 6 routed experts actually used, runs, then frees them.
   - That makes the whole model feasible on CPU, slowly. It produces golden
     per-layer activations and final logits for fixed prompts.
2. **Second opinion:** the fork, through libllama: per-step logits on the reference's
   own token sequence (`bin/llama-logits`). This tells "my bug" apart from "the fork's
   bug". (Per-tensor dumps from the fork were planned but not needed: the per-layer
   gates against the reference were tight enough.)
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

Precision (decided in step 3):

- The reference fake-quantizes activations to fp8 before every GEMM, stores KV
  as fp8/fp4, and indexer q/k as fp4.
- **CPU mode emulates all of it exactly** (`--no-quant` turns it off). Mean KL
  vs the reference: 1.8e-2 with it, 3.0e-2 without.
- **GPU mode runs without it** for now (CPU-only custom ops): mean KL 2.0e-2.
- The remaining inexact op is ggml's mxfp4 expert kernel, which re-quantizes
  activations to int8.

## Steps

The original plan, with status. Steps 5-8 are superseded by [PLAN.md](PLAN.md).

0. **Setup.** Done.
   - Python venv for the reference (needs the user: torch CPU, numpy,
     safetensors, sympy, tokenizers).
   - Link the fork's ggml + libllama. The CPU build exists; a `build-hip` of
     the fork for gfx906 comes at step 4.
1. **Loader + layer selection.** Done. mmap the 9 GGUF shards (already in the page
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
2. **Reference dumps** (`ref/`). Done: golden activations for those layers and
   prompts, then full-model logits for 3-4 short prompts.
3. **CPU v0** (`ds-completion`, ggml CPU kernels, own KV/state code, greedy). Done. The
   forward is one graph over all layers; tests build one graph per layer.
   - Build order: tokenizer (llama_vocab), embed, HC, attention window path,
     MoE, engram hash + lookup, compressor + compressed KV, indexer,
     candidates, head.
   - Each piece is gated at layer level before moving on.
   - Expected speed: about the fork's 3.5 t/s. Measured: 3.47 t/s.
4. **One GPU, no parallelism.** Done, at 6.0 t/s. The overlap and the 10 t/s target below
   were not reached; see Step 4 and PLAN.md.
   - All dense weights on die 0: ~15 GiB bf16, so they fit. Routed experts
     and the engram table stay on the CPU.
   - The shared expert (GPU) overlaps the routed experts (CPU).
   - Per layer and token, the host link carries only ~20 KB in and ~20 KB
     out, so the cost is latency, not bandwidth.
   - Target: >10 t/s.
   - Measured instead: GPU and CPU take turns (no overlap). The GPU dense part runs at
     ~260 GB/s and the CPU experts at ~50 GB/s.
5. **Layout.** (Now PLAN.md B and A0.)
   - Pack the fp8-derived bf16 weights back to fp8. ggml's MXFP8 is
     gather-only, so this needs our own fp8 GEMV, HIP first. Verify it's
     lossless byte-wise against the safetensors.
   - Record expert-routing histograms on real prompts.
6. **Experts on the dies.** (Now PLAN.md A.)
   - Hot experts resident across the 4 dies (EP), the cold remainder on the
     CPU as a fifth worker.
   - Hit rate from the step-5 histograms.
7. **Parallelism for the dense part.** (Now PLAN.md C.)
   - TP of attention and shared experts within an XGMI pair, then across all
     4 dies.
   - Prefill path: stream expert weights to the dies for large batches.
8. **Maybe:** (now PLAN.md E) a DSpark drafter (needs converting the MTP weights from the
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

## Step 2: the CPU reference and golden runs

`ref/ds_ref.py` runs the official `inference/model.py` **unmodified** on the CPU:

- **Kernels:** `ref/kernel_cpu.py` replaces the tilelang kernels with plain torch, keeping the
  same semantics. That covers fp8/fp4 fake-quant (e8m0 and e4m3 scales, round-half-to-even fp4),
  the fp8 x fp8 and fp8 x fp4 GEMMs, sparse attention with sinks, and the Sinkhorn split.
- **Weights from the GGUF, all exact (asserted):**
  - fp8 linears are re-blocked into the reference's own e4m3 + 32x32 e8m0 format.
  - Experts are repacked from mxfp4 into its fp4 layout.
  - Engram rows are read straight from the table.
- **Memory:** routed experts are materialized only when routed to, then dropped again.
- **What each golden file records, per step:**
  - per layer: block input and output (hc stream and pre-mix), the attention and MoE inputs and
    outputs, router weights and expert ids, the indexer's top-k, and the engram input;
  - per step: the final logits.
- **Speed:** a 10-token prefill takes 84 s and a decode step 10-18 s (16 threads).

`ref/llama_logits.cpp` (`bin/llama-logits`) is the second opinion: the fork's libllama (CPU),
fed the reference's exact token sequence, recording the logits per step.
`ref/compare_logits.py` compares any two of these.

Golden runs (`data/ref/`, git-ignored; logs in `results/ref-*.txt`):

| file | prompt | decode | exercises |
|---|---|---|---|
| hello.pt | chat "Hello, how are you?" (10 tok) | 16 | short context, decode path |
| readme200.pt | `prompts/ds-v41-readme.md`, 200 tok | 8 | past the 128-token window (ring buffer) |
| readme1100.pt | same, 1100 tok, `--keep-last 32` | 4 | indexer top-512 actually selects (ratio-1: 1100, ratio-2: 550 positions) |

**Reference vs fork, hello** (`results/compare-hello-ref-vs-fork.txt`):

- Argmax agrees on 16 of 17 steps; KL is typically 1e-7 to 1e-3.
- The miss (step 12) is a near tie: 😊 vs " I" are top-2 for both, in swapped order.
- Max logit difference is 0.4-0.8 x the logit std.
- Two independent implementations agreeing this closely rules out a structural bug in either.
  The remaining gap fits the reference's fp8 activation quantization before every GEMM,
  which llama.cpp does not do.

Reference completion: "Hello! I'm doing well, thank you for asking. 😊 I'm ready". The fork
in llama-cli: "Hello! I'm doing well, thanks for asking. How can I help you today?".

## Step 3: CPU v0

`src/forward.{h,cpp}` is the forward pass as ggml graphs on the CPU backend, written module by
module after `model.py`. It uses the fork's ggml kernels:

- matmuls and `mul_mat_id`;
- the DeepSeek-specific ops `dsv4_hc_comb` / `hc_pre` / `hc_post` and `lightning_indexer`;
- `swiglu_clamp`, `soft_max_ext` with sinks, and `top_k`.

Design:

- **Positions are handled on the host.** Ring slots, which compression groups complete in a chunk,
  and what each query may see are all worked out in `make_plan`. The graph only does arithmetic,
  gathers and `set_rows` into persistent state tensors.
- **Any chunking works.** The reference only prefills at position 0; here tokens can be fed at any
  position in chunks of any size.
- **Reads before writes.** Every in-graph write to state takes its rows from the tensor that read
  that state, so a write cannot run before the read.
- **Fake quantization** (`src/fakequant.{h,cpp}`) copies the reference's fp8/fp4 rounding as ggml
  custom ops. It is bit-exact against `ref/kernel_cpu.py` on 45k values, ties included
  (`bin/test-fakequant`).
- **Not implemented yet:** candidate-block selection, which only changes anything past 16k tokens.
  `make_plan` refuses such contexts.

Testing has to work around a property of fp8: one bf16-vs-f32 difference on the input of an fp8
rounding can move a value by a whole fp8 step (~6%). That makes "ours vs the real reference"
noisy at the 2-10% level per layer, which is too loose to catch a subtle bug. So the reference
also runs in variants, and each of our modes is compared against the matching one (teacher
forced, per layer, all 40 layers x all steps):

| ours | vs reference variant | attn_in | attn_out | ffn_out | notes |
|---|---|---|---|---|---|
| exact quant | fp32 + fake quant | ~1e-7 | 6e-8 at decode; 1e-4..3e-2 where an fp8 rounding flips | 1e-3..1e-2 | the routed experts are the one inexact op |
| no quant | fp32, no quant | ~1e-7 | 1.3e-3..5.5e-3 on all 200 rows, no outliers | 2e-3..1e-2 | noise from ggml rounding inputs to bf16 |
| exact quant | the real reference (bf16 + quant) | ~2e-3 | 2-10% | 2-10% | rounding chaos, as expected |

- **Exact in decode:** window attention, rope (with YaRN), sinks, the grouped output projection,
  the compressor, the compressed KV, the engram and the hyper-connections all agree with the fp32
  reference to 1e-7 at decode.
- **The routed experts are the one inexact part.** ggml's mxfp4 kernels re-quantize activations to
  int8 (q8_0), which the reference does not do. That puts ~1e-2 on ffn_out and occasionally flips
  a near-tie expert.
- **f32 compressor and index-key weights.** These small weights are kept in f32 (`Engine::f32w`).
  ggml would otherwise round their input to bf16, and the compressed KV's fp4 rounding then
  magnifies that to ~2e-2 in attention. The reference runs the compressor in fp32.

End to end (`--e2e`, hello, 17 steps, vs the real reference):

| implementation | argmax agrees | mean KL | median KL |
|---|---|---|---|
| ds-completion, exact quant (default) | 16/17 | 1.8e-2 | 1.3e-5 |
| ds-completion, `--no-quant` | 16/17 | 3.0e-2 | 2.7e-5 |
| the fork (llama.cpp CPU) | 16/17 | 2.5e-2 | 2.1e-5 |

- **The one miss is a near tie:** the reference picks 😊 at step 12, while we and the fork pick
  " I". Our completion: "Hello! I'm doing well, thank you for asking. I'm ready to help you with
  whatever you need. How".

Long context (teacher forced):

- **readme200 vs the fp32/no-quant reference.** This prompt is past the 128-token window, so the
  ring buffer wraps. attn_out stays at 1.3e-3..5e-3 on every layer, at prefill and at every
  decode step: the same noise floor as at 10 tokens, with no outliers.
- **readme1100 vs the real reference.** Here the indexer's top-512 really selects (1100 ratio-1
  and 550 ratio-2 positions). Our selected sets overlap the reference's with Jaccard 0.95-0.996
  per layer. The differences are near-ties at the cutoff; the reference ranks bf16 scores.
  Random picks would give about 0.3.

Speed (CPU only, nothing optimized yet; "Write a haiku about memory bandwidth.", --chat):

| threads | prompt (12 tok) | decode |
|---|---|---|
| 16 | 7.1 t/s | 3.23 t/s |
| 32 | 8.3 t/s | 3.47 t/s |

That is the fork's speed (3.5 t/s decode in llama-cli). Prefill of 1100 tokens took ~4-6 s per
layer through `ds-test`'s per-layer graphs.

## Step 4: dense part on one GPU

Building the fork for gfx906 (ROCm 6.3.4) needed a one-line fix:

- `patches/fork-hip-ballot-sync.patch`, applied to `~/forks/llama.cpp`.
- ggml's HIP shims define `__shfl*_sync`, `__all_sync` and `__any_sync`, but not `__ballot_sync`.
  That breaks `topk-moe.cu` (grouped-expert top-k, which this model does not use).
- Build: `cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DBUILD_SHARED_LIBS=ON ...`
  (the recipe in `hip-moe/README.md`).
- `test-backend-ops` passes on ROCm0 for every op the forward pass uses.

`--gpu D` (Options::gpu):

- **On the GPU:** the dense weights (attention, indexer, compressor, shared experts, router,
  hyper-connections, engram projection, head; 921 tensors, 14.45 GiB, uploaded in ~3.5 s) and
  the sequence state.
- **On the CPU:** routed experts (mxfp4), the token embedding and the engram table, all in the
  page-cache mmap.
- **Splitting the graph:** `ggml_backend_sched` splits it, with `op_offload` off: 86 splits per
  step, i.e. GPU -> CPU (routed experts) -> GPU in every layer.
- **Weights flag:** the mmap buffers must be flagged `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`.
  Otherwise the scheduler treats experts and tables as inputs and tries to copy 470 GB to the
  GPU.
- **No fake quantization on the GPU.** The fake-quant ops are CPU custom ops; on the GPU each
  would be a round trip. A HIP version means adding a real ggml op, which is left for later.

Correctness (teacher forced, all layers, vs the fp32/no-quant reference):

- **readme200:** attn_out median 1.2e-6 (max 3e-6). The GPU's bf16 matmul keeps f32 activations,
  so this is tighter than the CPU's 3e-3.
- **hello:** 3.5e-3 at prefill. ggml picks hipBLAS for bf16 weights at batch 9-63, and hipBLAS
  rounds activations to bf16 (`bin-hip/test-matmul-prec`: ~1e-7 at batch 1-8 and >= 64, 1.7e-3
  in between). Decode inherits 5e-4 through the caches.
- **End to end vs the real reference (hello):** argmax 16/17 (the same near tie), mean KL 2.0e-2
  (CPU exact-quant 1.8e-2, fork 2.5e-2).

Speed ("Write a haiku about memory bandwidth.", --chat, 32 threads):

| | prompt (12 tok) | decode |
|---|---|---|
| CPU only (step 3) | 8.3 t/s | 3.47 t/s |
| dense on GPU 0 | 7.0 t/s | **5.9-6.0 t/s** |
| dense on GPU 0, routed experts skipped (timing only) | 17.3 t/s | 18 t/s |

`--profile` breaks a decode step down by backend, with a sync after each run of same-backend
nodes:

| part | time per token | effective rate |
|---|---|---|
| GPU | ~53 ms with experts skipped (70-87 ms with them) | ~260 GB/s on ~14 GB of dense weights |
| CPU routed experts, incl. the round trips | ~75-95 ms | ~50 GB/s on 4.2 GB |

Occasional ~700 ms steps are first touches of expert pages in the mmap (minor faults).

Where the time goes, and what would buy it back:

1. **CPU experts** (~half of each step): 50 GB/s against a 112 GB/s ceiling. This could move to
   ggml's CPU_REPACK layout or a custom kernel, but the bigger win is step 6 (hot experts in HBM).
2. **GPU dense:** ~260 GB/s against ~880.
   - bf16 MMVF efficiency, plus several thousand small kernel launches per token.
   - Packing the fp8-derived bf16 weights back to fp8 (lossless, step 5) halves the bytes.
   - Fusing and HIP graphs would cut the launches.
3. **No overlap:** every layer is GPU then CPU then GPU, so two of the three parts idle at any
   time.
