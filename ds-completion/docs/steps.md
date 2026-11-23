# ds-completion: the steps

The lab notebook: every step in order, with what was done, what was measured and what was
decided on the numbers. Everywhere in this repository (code comments, results files' headers,
the other documents) "README step N" and "step N" mean the section "Step N" of this file: it
was the README until step 156.

The front page: [../README.md](../README.md). How to run: [../QUICKSTART.md](../QUICKSTART.md).

Before the steps: what the model is, the correctness ladder every step is checked on, and
which checkpoint the files are.

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

(As of step 3. The ggml GPU path went in step 11; the expert kernels are exact since step 7 on
the CPU and step 17 on the GPUs, and the own runtime has no inexact op.)

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

## Step 5 (PLAN.md B0 + B1): fp8 dense weights and an exact fp8 GEMV

**B0, `bin/fp8pack`.** It writes the 330 dense linears that were fp8 in the checkpoint back as
fp8:

- tensors: q_a, q_b, kv, wo_a, wo_b, indexer q_b, the shared experts and the engram projection;
- format: e4m3 codes plus one e8m0 exponent per 32x32 block, in a plain GGUF (codes and scales
  as I8 tensors).
- **Lossless:** every value round-trips exactly, checked for every weight, and the e4m3 codec is
  checked against the fake-quant rounding on all 256 codes.
- **Size:** 12.73 GiB bf16 -> **6.37 GiB** fp8, in 20 s. Output in `~/llms/ds-1017-fp8/dense-fp8.gguf`.
- A decode token's GPU reads drop from 14.45 to ~8.1 GiB, with the head (bf16) and the small f32
  tensors unchanged.

**B1, `hip/fp8_gemv.{h,hip}`.** y = W x for M <= 8 columns, exactly the reference's fp8 x fp8
GEMM:

- **Decode is exact.** gfx906 has no fp8, but an e4m3 code shifted left by 7 bits is an fp16
  bit pattern equal to the value x 2^-8 (subnormals too). `v_perm` plus two masks does 4 codes.
- **Products are exact.** `v_dot2_f32_f16` multiplies the fp16 pairs exactly into fp32. Each
  32-block partial is scaled by weight scale x activation scale x 2^16.
- **Activations** are fake-quantized to fp8 on the device (`fp8_quant_act`, bit-identical to the
  host rule) and stored as codes plus one scale per 32.
- **Work split:** one row per wave, with enough lanes per row for <= 10 16-byte steps per lane
  (64 lanes once K >= 4096). Lanes double while fewer than 1024 waves are in flight. The rule
  comes from a sweep over every lanes-per-row x rows-per-wave split (`--sweep`,
  `results/bench-fp8-sweep.txt`) and is within 1% of the best at M=1.

Results (`results/bench-fp8.txt`, die 0, layer 2):

- **Every tensor and batch size:** the device quantizer is bit-exact, and y matches a
  double-precision host computation of the same fp8 math to 3e-8..6e-8.

| tensor (K -> N) | ggml bf16, M=1 (us) | fp8, M=1 (us) | fp8 GB/s (of ~880) | fp8, M=4 (us) |
|---|---|---|---|---|
| q_a 5120 -> 1280 | 51.5 | 14.4 | 456 | 32.3 |
| q_b 1280 -> 32768 | 171.8 | 62.7 | 670 | 86.8 |
| kv 5120 -> 512 | 21.4 | 8.7 | 300 | 17.1 |
| wo_a 8 x (4096 -> 1024) | 153.2 | 65.7 | 511 | 121.4 |
| wo_b 8192 -> 5120 | 224.3 | 90.7 | 463 | 193.1 |
| shared gate/up 5120 -> 2304 | 54.3 | 23.5 | 502 | 43.3 |
| shared down 2304 -> 5120 | 101.8 | 21.7 | 545 | 47.0 |
| engram kv 6144 -> 25600 (layers 1, 14) | 565.1 | 227.1 | 693 | 336.6 |
| **per layer** (q_a..down) | **~832** | **~311** | | |

(ggml bf16: `bin-hip/bench-ggml-bf16`, 16 matmuls per graph to amortize the launch as a forward
does.)

What this says:

- **The matmuls of a decode token drop from ~33 ms to ~12.4 ms** (40 layers), and become exact to
  the reference again.
- **Step 4's GPU time (~53 ms) splits into** ~33 ms of these matmuls and ~20 ms of everything
  else: ~2000 small ops (norms, rope, hc, attention core, softmax, copies).
- **Small tensors are latency-bound.** kv is 2.6 MB in 8.7 us, of which ~3 us is the transfer
  itself. Fusing them with their neighbours is part of the own-runtime step.
- **M = 2-8 costs 1.4-4x M = 1, not ~1x.** The kernel is ALU-heavy once several columns share a
  weight load (decode + 8 dot2 per 16 codes per column). This matters for speculative verify
  batches (PLAN.md E1) and will need a different inner loop (LDS-staged activations, or int8
  sdot4 on scaled codes).

## Step 6: the own decode runtime on one die (PLAN.md section 3, option 2)

`hip/rt.{h,cpp}` with `hip/kernels.{h,hip}` and `hip/fp8_gemv.*`: one token per step through our
own HIP kernels.

- **Shared state with the ggml Engine.** The runtime uses the Engine's state (window rings,
  compressor groups, compressed KV, index keys: the same GPU tensors) and its GPU copies of the
  small f32/bf16 weights (norms, hc, router, compressor, indexer projections, head).
- **Prefill** still runs through `Engine::forward`; decode runs here. (Step 10: the prompt
  runs through the runtime too.)
- **fp8 weights** come from `dense-fp8.gguf` (6.37 GiB on the die), through the exact fp8 GEMV.
- **Routed experts on the CPU:** one cached ggml CPU graph per layer (fq8 -> mul_mat_id ->
  swiglu_clamp -> weights -> fq8 -> mul_mat_id -> sum), built once and reused every step. It
  runs while the GPU does the shared expert.
- **Numerics are the reference's:** fp32 throughout, with fake quantization at the same points
  (activations before every fp8 GEMV, window KV, compressed KV, indexer q/k), and rotary
  frequencies computed on the host in precompute_freqs_cis's fp32 op order.

Gate (`ds-test --runtime`: prefill through the Engine with fake quantization, then every
decode step teacher forced through the runtime, all 40 layers, vs the fp32 + fake-quant
reference):

- attn_out median **7.5e-8** over 160 decode rows, the same as the CPU exact path (7.2e-8).
  The tail (p90 2.5e-3) is the same fp8 rounding flips at the same rows.
- No expert-set or top-k mismatch on any decode row. ffn_out ~1e-2 is the CPU experts' q8_0
  activations, as before.

**Determinism** (found while testing the runtime):

- **rocBLAS atomics.** The ggml GPU path was not repeatable: logits differed by up to 13 between
  identical runs. The first differing tensor (`ds-test --determinism`: two engines, the same
  inputs, every tap compared bitwise) was an f32 matmul at batch 10, which rocBLAS computes
  with split-K atomics. Every later fp8 rounding then amplified the last-bit differences.
  `ROCBLAS_DEFAULT_ATOMICS_MODE=0`, set in the Engine before the GPU backend starts, makes it
  bitwise repeatable. The runtime gives the same completion every run.
- **Known issue, not investigated: `--gpu-exact` decode through the ggml path** (fake-quant
  custom ops on the CPU inside a GPU graph, at one token per step) is still not repeatable, and
  its KL against the reference is 6.2e-2 (plain `--gpu`: 1.4e-2). Its prefill (multi-token) is
  repeatable, and that is the only part the runtime uses. ggml decode is superseded by the
  runtime.

End to end vs the real reference (hello, 17 steps, `ds-test --runtime --e2e`): argmax 16/17
(the same near tie), **mean KL 1.43e-2**. That is the closest of all modes (CPU exact 1.8e-2,
ggml GPU 2.0e-2, fork 2.5e-2), as expected for exact fp8 GEMVs.

Speed ("Write a haiku about memory bandwidth.", --chat, 32 threads):

| | decode |
|---|---|
| ggml, dense on GPU 0 (step 4) | 6.0 t/s |
| runtime v1 (plain kernels) | 8.1 t/s |
| runtime v2 (kernels fixed, below) | **9.4 t/s** |

The completion is "Data floods the lanes, / A momentary river— / Then the path is clear."; the
CPU exact path gives the same first line, then a different near-tie.

What it took to go from v1 to v2 (`bin-hip/bench-kernels` times each kernel alone; the floors
are 1.9 us per empty launch and ~10 us per 20 KB pinned copy):

| kernel | v1 | v2 | what was wrong |
|---|---|---|---|
| fp8 activation quantizer | ~30 us | 3.5 us | one thread per 32-block, serial; now one thread per value, amax by shuffles; inputs quantized once, not once per GEMV |
| attention (64 heads, 128 + 512 keys) | 213 us | 122 us | one block per head: every key re-read 64x, and each wave's key loads waited one by one; now 8 heads x 32 keys per block, keys through LDS tiles, no cross-lane sums for the scores |
| router top-6 | 61 us | 29 us | small arrays indexed by a runtime counter lived in scratch memory (1 KB/lane); now compile-time and unrolled |
| hc Sinkhorn split | 53 us | 20 us | 4x4 Sinkhorn on one thread; now 16 lanes with row/column shuffles |
| indexer scores (1100 keys) | 92 us | 37 us | each lane walked a 128-dim dot from global memory; now key tiles in LDS, q transposed |
| CPU-expert input copies | 3 per layer | 1 | x, ids and weights staged in one buffer |

Per decode token now (`--profile`, GPU events between groups of launches):

| group | ms |
|---|---|
| waiting for the CPU experts | **72** |
| attention out (wo_a + wo_b) | 6.8 |
| attention q (q_a, q_b) | 5.0 |
| hc (mixes, Sinkhorn, pre, post) | 5.1 |
| router (+ the staging copy) | 4.0 |
| shared expert | 3.6 |
| attention core | 2.8 |
| head | 2.5 |
| engram, kv, indexer, compressor, add | 4.9 |

The GPU part is ~37 ms; the CPU experts (4.2 GB at ~58 GB/s) are 2/3 of the step. Next by
size: the CPU expert kernel (PLAN A3), then TP (C), then experts on the dies (A1).

## Step 7 (PLAN.md A3): an exact CPU kernel for the routed experts

`src/cpu_moe.{h,cpp}`: one layer's 6 routed experts for one token, AVX-512 + OpenMP, with the
weights read in place from the GGUF mmap. It replaces the runtime's ggml graph.

- **Exact.** Every fp4 x fp8 product is exact in fp32, so the 16 fp4 codes are decoded with a
  16-entry float table (`vpermps`) and multiplied in fp32 against the fp8-rounded activations:
  the reference's math up to the order of the fp32 additions. ggml's kernel instead
  re-quantizes the activations to int8.
- **Decode without a shuffle per lane.** The 16 code bytes of a block are broadcast to the four
  128-bit lanes (a load), shifted per lane, and vpermps reads only the low 4 bits of each index.
  The lanes end up holding codes in a fixed permuted order, and the activations are stored in
  the same order once per layer.
- **Work split:** gate+up rows of all 6 experts (32-row tasks, static schedule), then swiglu +
  routing weight + fq8, then the down rows, then the sum in expert order.

`bin/bench-cpu-moe` (accuracy for 3 layers vs a double-precision host computation of the same
math; speed for a "token" of 40 layers x 6 fresh random experts, 4.5 GB):

| | relerr vs exact | ms per token (10 tokens) | GB/s |
|---|---|---|---|
| ggml mul_mat_id (step 6) | 2.2e-2..2.4e-2 | ~72-78 | ~60 |
| ours | **5.8e-8..6.3e-8** | **~66** (32 threads) | ~68 |
| read-only control (same tasks, only loads) | - | ~47 | ~95 |

Why not faster: the inner loop alone, in cache, runs at 168 GB/s on 16 cores, and the weights
stream at ~95 GB/s. The kernel gets ~68, close to compute + memory taken one after the other
(1 / (1/168 + 1/95) = 61) rather than the larger of the two. L2 software prefetch 16 KB ahead
(the thread's next task) recovered part of it (79 -> 67 ms at 16 threads).

Tried and measured no better (`results/bench-cpu-moe-*.txt`): 1, 2 or 8 rows at a time; L1
prefetch; other distances; and SMT helper threads (the core's sibling reading the next rows
into the shared L2), which was slower (~98 ms) because its loads compete for the same ports.

In the runtime (all 40 layers, teacher forced, vs the fp32 + fake-quant reference):

- **ffn_out decode median 8.0e-8** (was ~1e-2): the runtime has no inexact op left. The p90
  (1.9e-2) is fp8 rounding flips.
- **End to end vs the real reference (hello, 17 steps):** argmax 16/17. The mean KL over 17
  steps is dominated by the one near tie at step 12, so it is shown both ways:

| | median KL | mean KL | mean KL without step 12 |
|---|---|---|---|
| runtime, exact CPU experts | 1.7e-5 | 2.5e-2 | **6.3e-4** |
| runtime, ggml CPU experts (step 6) | 1.7e-5 | 1.4e-2 | 5.7e-3 |
| CPU exact (step 3) | 1.3e-5 | 1.8e-2 | 3.9e-3 |
| the fork | 2.1e-5 | 2.5e-2 | 1.3e-3 |

- **Speed:** 9.4 -> **9.7-9.9 t/s**. A step is ~101 ms: CPU experts 67 ms, waiting for the GPU
  18 ms, the rest GPU work and copies.

## Step 8 (experiment): whole layers of experts in HBM - the ceiling without routing statistics

The question: how fast would decode be if the experts were on the GPUs, without depending on
which experts are hot? The test puts *all 384 experts* of chosen layers into HBM (6.72 GiB per
layer) and measures.

`hip/fp4_moe.{h,hip}`: the routed experts of one layer for one token on a GPU, exact like the
CPU kernel.

- **Exact decode:** an e2m1 code shifted into fp16 bits is the value x 2^-14, subnormal 0.5
  included, and `v_dot2_f32_f16` multiplies exactly.
- **Upload layout:** the weights are split into aligned code bytes plus separate scale bytes,
  repacked on the GPU.
- **Pairing:** activations are fake-quantized in "paired" order (x_j, x_{j+16}), which meets
  ggml's nibble order.
- **No host round trip:** expert ids come straight from the GPU router.

`bin-hip/bench-fp4-moe` (layer 20, fresh random experts per repetition):

- matches the exact CPU kernel to **2.8e-8..3.7e-8**;
- **393 us per layer** (6 experts, 113 MB, 287 GB/s; not tuned yet), against ~1650 us for the
  CPU kernel.

`--gpu-moe dev:layers,...` places layers' experts on dies:

- **On the runtime's own die**, the experts run right after the router.
- **On another die**, the router's outputs go through pinned host memory (avoiding the ROCm
  6.3.4 cross-card copy crash), with stream/event ordering and no host waits. That die computes
  while die 0 does the shared expert, and the result comes back the same way.
- **Capacity:** dies 1-3 hold 4 layers each and die 0 one more (next to the 14.45 GiB of ggml
  dense copies and 6.37 GiB of fp8): **13 layers, a third of the model**.
- **Gate:** every placement (local die, the same card, the other card) is exact at the layer
  level (ffn_out ~1e-7, `ds-test --runtime --gpu-moe ...`).

End to end (`results/gpu-moe-e2e.txt`, haiku prompt, 32 threads):

| experts fully on GPUs | decode | step | CPU experts | rest (GPU + copies) |
|---|---|---|---|---|
| none | 9.5 t/s | ~103 ms | ~68 ms (40 layers) | ~35 ms |
| 4 layers (die 1) | 10.3 t/s | ~95 ms | ~58 ms (36) | ~37 ms |
| 12 layers (dies 1-3) | 11.2 t/s | 88 ms | 50 ms (28) | 38 ms |
| 13 layers (dies 0-3) | 11.4 t/s | 85 ms | 47 ms (27) | 38 ms |

What it says:

- **Each layer moved to a GPU** saves its ~1.78 ms of CPU and costs ~0.4 ms of GPU (the fp4
  kernel). The copies to the other card hide almost entirely.
- **Ceiling with every layer's experts on a GPU:** ~35 + 40 x 0.4 = **~50 ms per step,
  ~20 t/s**, with today's kernels. That needs 269 GB of experts, so in practice it means hot
  experts (A1) with a high enough hit rate. The table is the cost of each miss and each hit.
- **What remains then:**
  - the ~35 ms of dense GPU work and copies on one die (the TP target, PLAN C);
  - the fp4 kernel at 287 GB/s (a 2x kernel would take the 40 x 0.4 to ~8 ms, so ~43 ms per
    step, ~23 t/s);
  - the remote layers wait serially; EP could run several dies' experts at once.

## Step 9 (experiment): GPU core clock (sclk) levels

All earlier numbers were taken at sclk level 2 (1204 MHz); mclk is fixed at 1000 MHz. The level
is set by hand with sudo (`power_dpm_force_performance_level=manual`, `pp_dpm_sclk=<level>` on
every card). `results/sclk/sclk-sweep.sh` runs the GPU-side benchmarks and both decode
configurations at the current level, sampling hwmon power, into `results/sclk/sclk<level>.txt`;
`python3 results/sclk/sclk-summary.py` makes the table.

| | L0 999 | L2 1204 | L4 1445 | L6 1608 | L8 1730 MHz |
|---|---|---|---|---|---|
| fp8 GEMV per layer, 1 token | 355 us | 311 | 286 | 281 | 276 |
| fp8 GEMV per layer, 4 tokens | 673 us | 584 | 514 | 484 | 466 |
| attention, 640 keys | 147 us | 121 | 103 | 93 | 88 |
| top-k 512 of 1100 | 65 us | 54 | 45 | 41 | 38 |
| GPU experts, one layer | 469 us | 393 | 334 | 305 | 299 |
| GPU + copies per step, experts on CPU | 37.8 ms | 33.6 | 31.2 | 29.2 | 28.9 |
| GPU + copies per step, + 13 expert layers on GPUs | 43.5 ms | 38.2 | 34.7 | 35.9 | 31.4 |
| peak W, die 0, fp8 benchmark | 149 | 181 | 198 | 243 | 273 |
| peak W per die, decode + 13 layers | ~85 | ~100 | ~120 | ~150 | ~140-180 |

- **The memory-bound fp8 GEMV barely depends on clock above level 2** (311 -> 276 us). Its
  4-token version and every small, latency-bound kernel scale closer to the clock (~1.4x from
  level 0 to 8, against 1.73x of clock).
- **Decode with the experts on the CPU hardly moves.** The CPU's ~67 ms dominates; the GPU part
  goes 38 -> 29 ms from level 0 to 8.
- **Gains flatten above level 4-6, while peak power keeps rising** (+40% from level 4 to 8 for
  ~2 ms per step). Level 4 is a good default today.
- **Clocks matter more as experts move to the GPUs:** the GPU expert kernel speeds up 1.57x
  from level 0 to 8, so the per-layer split (PLAN section 6) wants level 6-8.
- **Noise:** the step times are medians of 2-3 profiled steps and the CPU part varies by
  several ms. Trust the kernel rows and the "GPU + copies" rows over the totals. Power is
  1-sample-per-100 ms hwmon peaks, not energy.

## Step 10: the prompt through the runtime; ggml's dense GPU copies dropped

`--rt` used to keep two copies of the dense weights on die 0: ggml's bf16 copies (14.45 GiB,
used only to prefill the prompt through `Engine::forward`) and the runtime's fp8 ones
(6.37 GiB). Now every token, prompt included, runs through the runtime, one token at a time:

- **`Options::runtime`:** the Engine copies only the small weights the runtime reads (norms, hc,
  router, compressor norms, indexer projection, head) plus the f32 compressor/indexer-key
  copies, and holds the sequence state. **1.68 GiB instead of 14.45 GiB** on die 0.
  `Engine::forward` / `run_layer` throw in this mode.
- **`Runtime::prefill(tokens)`:** decode() per token, skipping the 1.3 GB head GEMV for all
  but the last.
- **`ds-test --runtime`:** the prompt step is teacher-forced token by token, and each row
  pools the layer's errors over the prompt's tokens (`--per-token` prints every token).
  Tokens a `--keep-last` golden did not keep run free.

Gate (hello-fp32, all 40 layers, `results/rt-prefill-hello-fp32.txt`):

- **Decode steps:** unchanged, attn_out / ffn_out median 7.5e-8 / 8e-8.
- **Prompt step:** per token, most layers are at ~1e-7. Some token-layers are at 1e-4..1e-3:
  fp8 rounding flips, since the reference's batched fp32 matmuls sum in another order. Pooled
  over 12 tokens: attn_out median 9e-4. The CPU Engine's chunked prefill shows the same on
  layers 0-2 (6.6e-4 / 1.1e-4 / 3.6e-4 against the runtime's 6.1e-4 / 2.4e-4 / 3.8e-4), and
  the old ggml GPU prefill had 3.5e-3. One token in one layer picked a different expert set
  (a near tie after a flip).
- **End to end vs the real reference (hello, 17 steps):** argmax 16/17 (the same near tie),
  KL median 5.9e-6 (earlier runtime 1.7e-5). The mean without step 12 is 4.8e-3 against 6.3e-4.
  It is driven by steps 7, 14 and 16; per-step KL scatters over orders of magnitude in every
  exact mode (CPU exact: 1e-2..2e-2 at steps 7, 11, 14), so one 17-step prompt cannot rank
  these.

Speed (haiku prompt, 12 tokens, sclk level 2, 32 threads; `results/rt-prefill-speed.txt`):

| expert layers on GPUs | prompt | decode |
|---|---|---|
| none | 9.9 t/s (was ~6) | 10.0 t/s |
| 13 (0:27, 1:36-39, 2:32-35, 3:28-31) | 11.4 t/s | 11.8 t/s |
| **15 (die 0 now holds 25-27)** | 13.0 t/s | 12.0 t/s |
| 16 (die 0: 24-27) | out of memory | |

- **Die 0 has room for 3 expert layers instead of 1.** The prompt goes at decode speed because
  it is decoded. Batching prompt tokens (each weight read once for several tokens) is PLAN F.
- **The ggml GPU path** (`--gpu` without `--rt`) still exists, but nothing in the runtime
  depends on it any more.

## Step 11: cleanup - ggml's GPU path removed

Since step 10 nothing used ggml's GPU backend, so it is gone:

- **Engine (`forward.cpp`) is the CPU path only:** no `--gpu` mode, scheduler, GPU weight
  copies, `--gpu-exact`, `--profile` or `--debug-skip-routed`.
- **Runtime (`hip/rt.cpp`) is self-contained:** it takes the Model (`RtOptions{gpu, n_threads,
  max_ctx}`), uploads the small weights itself (1.75 GiB; bf16 compressor and indexer-key
  weights converted to f32), and keeps the sequence state in its own device buffers.
- **Build:** `make HIP=1` links the fork's static CPU libraries like `make`; the fork's
  `build-hip` and `patches/fork-hip-ballot-sync.patch` are no longer needed.
  `bench-ggml-bf16` and `test-matmul-prec` (measurements of ggml's GPU matmul, results in steps
  4 and 6) are removed.
- **CLI:** `--rt <dense-fp8.gguf>` selects the GPU runtime, `--gpu D` its die (default 0);
  without `--rt` everything runs on the CPU.

Re-run at sclk level 2:

- **Gate:** every per-layer error identical to step 10.
- **End-to-end logits:** bit-identical to step 10.
- **Speed** (`results/rt-cleanup-speed.txt`): decode 9.5 t/s, 12.2 t/s with 15 expert layers on
  GPUs (step 10: 10.0 / 12.0, run-to-run noise). CPU-only path 3.55 t/s, as before.

Steps 4-6 above describe the removed ggml GPU path as it was; their numbers stay as history.

## Step 12: engram rows per step, experts mapped at load, launch overhead measured

- **Engram:** both engram layers' rows are hashed, read and converted at the start of the step and
  go to the GPU in one copy (`Runtime::fetch_engram_rows`); the layers no longer wait for the
  host. Per-layer gate identical (layers 0-15). Engram time 1.4 -> 1.2 ms per step.
- **Experts mapped at load:** the CPU experts are mapped into the process up front, in parallel
  (`Model::map_experts`, `MADV_POPULATE_READ`, 180 GB in 1.2 s; it also reads anything not yet
  cached). Before, every expert's first use in a run paid ~4k minor page faults inside the
  decode step: a 40-token run took 29M minor faults.
- **After a reboot:** the model is mmap'd, so the first run reads experts from NVMe. Warm the
  cache with `cat` of shards 1-7 and 9 (47 s here); shard 8 (the engram table) needs only a
  few rows per token.
- **A/B** (128 generated tokens, 15 expert layers on GPUs, sclk 2, 5 interleaved runs each):
  before 13.02 t/s median, + engram 12.97, + mapping 13.11. Both gains are within ~1% noise.
  Runs of 128 tokens reach ~13 t/s against ~11-12 in 15-token runs: compare long runs.
- **Launch overhead** (`bench-kernels`, chain section; launch counts from an LD_PRELOAD
  counter): a decode step issues ~1,500 kernel launches (~37 per layer) and ~136 small copies.
  The dispatch floor is ~2 us per launch. HIP graph replay saves only 0.1-2 us per launch on
  this ROCm, so only fusion (fewer launches) helps: an estimated ~2-3 ms per step (3-4%).

## Step 13: routing statistics (30 prompts) - an adaptive expert cache is the way

`--log-routing` records every token's 6 experts per layer; `results/routing-study.sh` ran the
30 prompts of `prompts/routing-study.tsv` (8 categories: code, math with thinking, story, fact,
chat, Chinese, Ukrainian, translation, plus 2 long-document prompts), up to 1,024 generated
tokens each: 20,919 generated tokens (`data/routing/study/`, not in git).
`scripts/route_stats.py` analyzes them (`results/routing-study.txt`; with prompt tokens:
`results/routing-study-with-prompts.txt`).

Capacity for experts after the dense part (~9 GB) on die 0, at 18.8 MB per expert:
64 GB -> 2,925 (19%), 128 GB (this machine) -> 6,329 (41%), 256 GB (two machines) -> 13,137 (86%).

Hit rate (share of chosen experts served from HBM) on a held-out prompt; estimated t/s in brackets
(35 ms GPU + per layer max(CPU misses x 0.27 ms, GPU hits x 0.065 ms / 4 dies)):

| how the GPU set is chosen | 64 GB | 128 GB | 256 GB |
|---|---|---|---|
| uniform, no statistics | 19% (11.4) | 41% (13.7) | 86% (22.5) |
| fixed: top experts of the other prompts | 44% (14.0) | 71% (18.7) | 96% (25.8) |
| fixed: top experts of the other categories | 33% (12.7) | 59% (16.2) | 89% (23.9) |
| **adaptive: LRU warm-started with the other prompts' set** | **85% (22.2)** | **95% (25.8)** | 99.6% (25.7) |
| experts loaded per token (adaptive) | 37 | 11 | 1 |
| bound: the prompt's own top experts | 87% | 98% | 100% |

- **Within a prompt routing is concentrated** (~86 of 384 experts per layer cover 90%), **across
  prompts it is broad** (212 for 90%). Categories have their own experts: code, math and Chinese
  transfer worst from other prompts (31-62% at 128 GB), stories and chat best (77-86%).
- **Temporal:** recent usage beats any fixed set from other prompts. A cold LRU reaches 94% at
  128 GB (15 loads per token); the top of the last 256-1,024 tokens 94% (12-15 loads); a
  64-token window is too short (88%, 30 loads). Drift within a prompt is small: the first
  quarter's set catches 74-95% of the last quarter (code drifts most).
- **Loads are affordable at 128 GB:** 11 per token = ~210 MB, ~4 GB/s at 50 ms per step, spread
  over 4 GPU links. At 64 GB, 37 per token (~13 GB/s) is borderline.
- **At 256 GB a fixed set is enough** (96%): the two-machine pipeline split would not need
  adaptation.
- Prompt tokens included (long documents weigh in): the same picture, a few points lower
  (adaptive 94.8% at 128 GB).

Also fixed: prompts longer than `--max-ctx` ran past the runtime's caches and crashed. The
runtime now refuses positions past `--max-ctx`, and the CLI caps generation to fit.

## Step 14: routed experts split per layer between GPUs and the CPU (placement steps 1-5)

`--placement placements/study30.txt` replaces step 8's whole-layer `--gpu-moe` (removed): the
most-used (layer, expert) pairs of the routing study fill the GPUs' free memory one expert at
a time, and each layer's 6 experts run where they live, at the same time.

- **Placement** (`route_stats.py --emit-placement`): all pairs ranked by use. The runtime keeps
  the top ones that fit (6,627 = 43% here: die 0 1,305 after the dense part, dies 1-3 1,774
  each), rotating each layer's experts over the dies. Loading takes ~27 s (4.6 GB/s, pageable).
- **Pools** (`hip/expert_pool`): slots of one expert each; `fp4_moe` addresses experts by slot
  and takes a device-side count (0-6). Exact against the CPU kernel (<= 4e-8); 1/2/3/6
  experts take 130/176/231/401 us (`bench-expert-pool`): ~75 us fixed + ~55 us per expert.
- **Per layer:** a dispatch kernel splits the router's experts into one list per die and one
  for the CPU; x + lists go to the host in one copy; the shared expert runs; the host reads the
  counts and starts only the dies with work (other dies host-staged, as before) and the CPU
  misses; die 0 sums the parts in a fixed order (dies, then CPU).
- **Gate** (hello-fp32, all 40 layers): no placement = bit-identical to step 13 (all 200 rows,
  end-to-end logits); real and random placements: the same error levels (decode medians
  ~8e-8); all experts on GPUs (layers 0-5): identical to all-CPU row by row except ~1e-8
  (sum order).
- **Speed** (held-out prompt "Explain the difference between concurrency and parallelism",
  128 tokens, sclk 2): **15.5-15.7 t/s** (15 whole layers on GPUs: ~13.0; no GPU experts: ~10).
  80.7% of routed experts ran on GPUs.
- **Step (~65 ms):** dense on die 0 ~35, CPU misses ~20 (~46 per token: ~0.47 ms each with the
  per-layer overhead, slower than the 0.27 ms of a 6-expert call: the CPU kernel on 1-2 experts
  is the next thing to measure), die 0's own expert lists 5.5 (mostly overlapped), waiting for
  the other dies 2.4, dispatch + copy 1.6.

## Step 15: upper bound with perfect knowledge of the prompt (oracle placement)

Each prompt's own routing (logged on the same run) as its placement: 97.3% of routed experts
on GPUs (`placements/oracle-*.txt`, `results/oracle-placement.txt`; 256 tokens, sclk 2):

| prompt | no placement | study30 placement | oracle placement |
|---|---|---|---|
| concurrency (held out) | 9.4 t/s | 15.2 t/s (81%) | 15.9 t/s (97.3%) |
| code-rust | 9.6 t/s | 12.6 t/s (66%) | 16.0 t/s (97.3%) |

With today's expert path, perfect caching gives ~16 t/s: above ~80% hits the GPU expert path's
serialization is the limit, not the CPU. Oracle step (~54 ms profiled): dense ~26 ms; routed
experts on GPUs ~20 ms where the kernels need ~6-10 (die 0's lists run after the shared expert;
the other dies start only after the host reads the lists; router + dispatch + sum ~5 ms). Rare
CPU misses cost ~0.9 ms each (thread wake-up; `OMP_WAIT_POLICY=active`: CPU 7.8 -> 2.7 ms per
step); frequent ones ~0.5 ms each either way (the CPU kernel on 1-2 experts).

## Step 16: no die waits for the host (expert path fix 1)

`Runtime::moe` reordered: the home die runs its own expert list on a second stream next to the
shared expert; every other die starts as soon as the staged block is in host memory and reads
its count on the device (0: exits early); the host only runs the CPU's misses. Gate: no
placement, study30 and all-GPU placements identical to step 14 row for row; end-to-end logits
bit-identical.

| prompt (256 tokens, sclk 2) | placement | step 15 | now |
|---|---|---|---|
| concurrency | study30 | 15.2 t/s | 16.0 t/s (82% on GPUs) |
| concurrency | oracle | 15.9 t/s | **18.8 t/s** (97.5%) |
| code-rust | study30 | 12.6 t/s | 13.5 t/s (61%) |
| code-rust | oracle | 16.0 t/s | 17.5 t/s (97.2%) |

Oracle step (concurrency, ~47 ms profiled, was ~54): the expert-related groups (router,
dispatch, shared expert with the home die's list next to it, waits for GPU and CPU experts,
add) take ~20 ms, down from ~25. Next: the GPU expert kernel for 1-3 experts (~75 us fixed per
call) and the CPU threads' wake-up for rare misses.

## Step 17: faster GPU expert kernel for few experts (fix 2)

`fp4_moe` is now 4 launches instead of 7: input quantization; gate and up of every expert in
one grid; swiglu x routing weight + act_quant in one pass; down of every expert summed in the
kernel in a fixed order (no separate sum). The e2m1 weights are decoded 4 bytes at a time with
masks and byte permutes (~36 instructions per 16 bytes instead of ~128); activations stay fp8
codes converted per row (storing them as fp16 doubled the cache traffic and was slower). Lanes
per row from a sweep: 32 for gate/up, 16 for down.

| experts per call (sclk 2, `bench-expert-pool`) | 1 | 2 | 3 | 6 |
|---|---|---|---|---|
| before | 129 us | 173 | 228 | 397 |
| now | **72** | **110** | **151** | **281** |

Still exact against the CPU kernel (<= 4e-8). A whole layer (6 experts, `bench-fp4-moe`):
278 us = 406 GB/s (was 393 us). Gate: no placement unchanged; with placements a few rows move
in the last printed digit (largest change 3e-10).

End to end (256 tokens, sclk 2): concurrency study30 16.3 t/s, oracle 18.6; code-rust study30
13.4, oracle 17.9. Little change: since fix 1 most GPU expert work overlaps other work (its
groups shrank ~2.6 ms per step), and the oracle runs are now limited by rare CPU misses
(~1.4 ms each, thread wake-up: 9.4 ms per step in the oracle profile). Next: fix 3.

## Step 18: 16 CPU threads, not 32 (fix 3)

Every runtime measurement so far used `-t 32` (from the CPU-only tests of step 3). With the
GPU runtime that is wrong: 32 threads take all 32 logical CPUs, leaving none for the host
thread and the HIP runtime's threads, and OpenMP, seeing a full machine, puts its threads to
sleep between calls (each rare CPU miss then paid ~1 ms of wake-ups). Thread-count sweep
(128 tokens, sclk 2, `results/threads-sweep.txt`):

| prompt | placement | -t 32 | -t 12 | **-t 16** | -t 20 | -t 24 | -t 28 |
|---|---|---|---|---|---|---|---|
| concurrency | none | 9.7 | 11.6 | **12.1** | 10.4 | 11.2 | 11.8 |
| concurrency | study30 | 16.3 | 20.5 | **21.7** | 20.3 | 20.6 | 20.9 |
| concurrency | oracle | 16.3-17.5 | 25.8 | **25.3** | 25.1 | 24.7 | 24.7 |
| code-rust | none | 9.6 | 11.9 | **12.5** | 10.7 | 11.4 | 11.9 |
| code-rust | study30 | 13.6 | 16.6 | **17.7** | 16.0 | 16.2 | 16.5 |
| code-rust | oracle | 14.1 | 24.9 | **24.9** | 24.2 | 23.8 | 23.8 |

16 (one per physical core, the CLI default) is best or within 2% everywhere, even with no
placement where the CPU does all routed experts. A thread pool whose workers spin between calls
was tried and dropped: at 16 threads it matched OpenMP (within 1%), so OpenMP stays. Scripts now
use `-t 16`.

Where this leaves the placement work (16 threads): no GPU experts 12.0-12.5 t/s; the study30
placement 17.7-21.7; perfect knowledge (oracle) ~25 t/s.

## Step 19: DSpark draft head, shadow mode

The 1017 DSpark draft (`~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf`, arch `dflash`,
8.5 GB): 3 layers (window-only attention, 128 experts top-3), `fc` from the main model's stream
at the inputs of layers 37-39, a rank-256 Markov head and a confidence head; it shares the main
model's embedding and output head. One pass drafts a block of 5 tokens: [last token, 4 noise
tokens] attending to a 128-slot window of main positions and to the block itself (both ways).
The official `model.py` has its forward pass (`forward_spec`) but no speculative loop.

- **Weights:** every dense draft linear (and `fc`) is an exact fp8 expansion (100% of 32x32
  blocks), so it runs through our exact fp8 GEMV after re-blocking at load
  (`fp8_repack_bf16`, now shared with `fp8pack`, whose output is unchanged). The Markov tables are
  true bf16.
- **Runtime** (`hip/dspark.{h,cpp}`, `--dspark FILE`): the main forward hands over its stream at
  layers 37-39 (mean over the hc copies); every token seeds the draft's windows; after each
  generated token `draft()` returns 5 guesses. Draft experts (7.2 GB) live on GPU 0.
- **Gate** (`ds_ref.py --dspark` runs the official draft; `ref/compare_dspark.py`,
  `ref/dspark_check.py`): fed our main model's hidden states, the official draft and ours agree
  to ~8e-8 per position except where an fp8 rounding step flips at a near tie (one position in
  the first draft), which then spreads through the block attention. Against the reference run,
  the main model's own end-to-end differences at layers 37-39 add more; drafts agree at 5 of 12
  steps on the hello prompt.
- **Shadow mode** (drafts measured against what the main model then generates, nothing skipped;
  256 tokens, `results/dspark-shadow.txt`):

| prompt | accepted prefix (of 5) | tokens per main step if verified | per position correct | fork (llama.cpp) |
|---|---|---|---|---|
| concurrency (chat) | 1.95 | **2.95** | 77 / 55 / 39 / 27 / 24% | 2.58 |
| code-rust | 4.10 | **5.10** (all 5 accepted at 171 of 251 steps) | 96 / 90 / 86 / 79 / 75% | 3.60 |

- **Cost:** 26 ms per draft as written (one token at a time: 5 reads of the 1.3 GB output head,
  the Markov head with a host round trip per position). Reading the head once for all 5
  positions and doing the Markov step on the GPU should bring it to ~5-7 ms.

## Step 20: speculative decoding loop (batched verification phase 1)

`--spec` (with `--rt --dspark`): greedy speculative decoding. Each step drafts 5 tokens from
the anchor (the last emitted token, not yet run), verifies [anchor, drafts] with the main
model, keeps the matching prefix plus the main model's own next token, and commits.

- `Runtime::verify(chunk)` runs the chunk through the single-token path (phase 1: no batching)
  and returns the greedy next token after each position; `Runtime::commit(n_keep)` keeps the
  first n_keep positions:
  - **window rings:** the slots the chunk overwrote are saved before and restored for the
    discarded positions;
  - **compressor partial groups (ratio 2):** not self-healing as first assumed. A rejected
    position can overwrite the slot of a kept one with the same pos % ratio, which the real
    token completing that group would then pool. Each chunk position's row is kept, and commit
    rebuilds every slot from the last kept position writing it (or the state before the chunk).
    The compressed cache and index keys need nothing: they are keyed by group and rewritten by
    the real token completing it before anything reads them;
  - **sequence:** n_past and the token history (engram) truncated; the draft seeded with the
    kept positions only.
- **Gate: speculative output = plain greedy output, byte for byte**
  (`results/spec-phase1.txt`, study30 placement, draft loaded in both runs so the expert
  placement is the same):

| prompt | tokens | output | tokens per verify step |
|---|---|---|---|
| concurrency (chat) | 300 | identical | 2.78 |
| code-rust | 300 | identical | 4.92 |
| math-proof (thinking) | 502 | identical | 3.80 |
| zh-essay | 378 | identical | 2.17 |

  The first version (without the compressor fix) diverged after ~50 and ~250 tokens.
- No speedup yet by design: the chunk runs token by token, so a step costs the accepted tokens
  plus the rejected ones plus the 26 ms draft. (The timings in the results file are not a fair
  comparison: the plain runs had the draft loaded in shadow mode, drafting after every token.)
  Phase 2 batches the chunk.

## Step 21: batched verification phase 2 - chunk forward, batched expert block

- **Chunk forward** (`Runtime::forward_chunk`, used by `verify`): per-token buffers hold up to 8
  tokens. Attention is split into batched projections (one fp8 weight read for all tokens), a
  per-token core in position order (window K/V, compressor, indexer, attention: each token sees
  exactly the state it would alone) and batched output projections. The head runs through a new
  multi-token bf16 GEMV. `decode()` is the same code with one token.
- **Batched expert block:** router (multi-token f32 GEMV), one dispatch for every token's
  experts, one staging copy, one GPU launch per die over (token, expert) pairs
  (`fp4_moe_chunk`), the shared expert as multi-token fp8 GEMVs, one host wait per layer for the
  CPU misses (still one CPU call per token).
- **Bit-identity, tested per kernel:** `bench-kernels --bitwise` (multi-token f32/bf16 GEMVs:
  head, hc_fn, router, idx_proj, M = 2..8), `bench-expert-pool` (a 6-token chunk with 0..6
  experts per token vs one call per token). Single-token path unchanged (per-layer gate with and
  without placement, end-to-end logits bit-identical). `--spec` output byte-identical to plain
  greedy on the 4 prompts and in every draft-length run.
- **Speed** (`results/spec-draft-length.txt`, 300 tokens, study30, sclk 2; `--spec-n` drafts
  verified per step):

| | plain (no draft loaded) | n=1 | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|---|
| chat (concurrency) | **19.9** | 16.4 | 16.8 | 15.3 | 13.8 | 12.1 |
| code (code-rust) | 17.7 | 15.7 | 17.7 | **18.0** | 17.5 | 17.5 |

  Still not a win: a 6-token verify step costs 207 ms (was 267 before the expert block was
  batched): CPU misses 70 ms (~45 per token at study30), hyper-connection kernels 25 (still per
  token), attention core 20 (per token by design), attention output 20 and shared expert 16
  (fp8 at 6 tokens costs ~3x one token: activation decoding per row dominates), router 11,
  head 10, the rest ~26; plus the 26 ms draft.
- Also fixed on the way: an intermittent GPU memory fault at startup came from uploading experts
  straight from the mmap; uploads now go through pinned buffers (4x faster, step 20's commit).

## Step 22: faster draft (phase 3)

- The draft's 5 block positions go through each layer together: multi-token fp8 projections,
  mixes, router, `fp4_moe_chunk` for the routed experts, shared expert and head (one weight read
  per block instead of 5). Attention stays per position after every position's K/V is written.
- The Markov head runs on the GPU, left to right: row gather of the previous token's rank-256
  embedding, bias GEMV, add, argmax (`k_gather_bf16_row`, `k_argmax`: lowest index on ties, as
  the host's `max_element`); one host sync per draft.
- Multi-token GEMV for long matrices (the head, N >= 16384): a wave takes 4 rows (up to 3
  inputs) or 2 rows (more) and loads each input's activations once for all of them. Same
  expression per element (`load_x` + `dotw`), so still bit-identical to the single-input GEMV
  (`bench-kernels --bitwise`); the head for 5 positions 9.8 -> 4.3 ms, for a 6-token verify
  ~4.8 ms.
- Draft 26 -> **13.1 ms**; its outputs (ids, confidences, logits) byte-identical to the previous
  version on the reference sequence. `DS_DRAFT_PROFILE=1` prints its GPU time per part.
- The draft now takes ~70 MB more on die 0 (4 fewer expert slots), which moves a few experts to
  the CPU and so changes the plain output bits, like any placement change; the gate compares
  `--spec` with plain greedy from the same binary and configuration (identical on 4 prompts x 5
  draft lengths).
- Speed (`results/spec-draft-length.txt`; plain = no draft loaded):

| | plain | n=1 | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|---|
| chat (concurrency) | **20.8** | 19.2 | 18.8 | 18.2 | 15.7 | 14.2 |
| code (code-rust) | 17.6 | 17.4 | 18.7 | 18.4 | **18.7** | 17.9 |
| math (think) | 20.4 | 19.9 | **20.5** | 19.7 | 18.2 | 16.8 |
| zh essay | **16.6** | 14.3 | 12.6 | 10.6 | 9.1 | 7.7 |

  Roughly break-even: code +6%, math +1%, chat -8%, zh -14%. The verify step (~200 ms for 6
  tokens vs ~48 ms per plain token) still costs too much per extra token; next are the verify
  side's per-token kernels (hyper-connections, fp8 activation decoding) and CPU misses.

## Step 23: batched hyper-connection kernels

- `k_hc_split`, `k_hc_pre_norm`, `k_hc_pre`, `k_hc_post` take a token count: one block (row) per
  token doing exactly the single-token computation, so the bits cannot change; the mixes use the
  multi-token f32 GEMV (already bitwise-tested). Used by the chunk forward, decode (M = 1) and the
  draft.
- Gates: per-layer rows identical (with and without placement), end-to-end logits identical,
  drafts byte-identical, `--spec` identical to plain greedy on 4 prompts x 5 draft lengths.
- Verify step of 6 tokens 207 -> **181 ms** (hc 25 -> 6 ms; the head 9.8 -> 4.9 ms from step
  22's GEMV); draft 12.3 ms.
- Speed (`results/spec-draft-length.txt`; plain = no draft loaded, same build):

| | plain | n=1 | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|---|
| chat (concurrency) | **20.5** | 20.2 | 20.4 | 19.8 | 17.3 | 15.8 |
| code (code-rust) | 17.2 | 17.8 | 19.7 | 19.6 | **20.2** | 19.4 |
| math (think) | 21.0 | 21.1 | **22.0** | 21.5 | 19.9 | 18.4 |
| zh essay | **17.3** | 15.1 | 13.5 | 11.4 | 9.8 | 8.3 |

  First clear wins: code +17%, math +5%; chat even; zh -13% (acceptance there is ~1.2 drafts per
  step). What's left in the verify step: CPU misses 68 ms (they grow with the tokens verified),
  fp8 at 6 tokens (attn.out 20 + shared 16 + q 10: activation decoding per row), attention core
  20 (per token by design), router 10.5.

## Step 24: multi-token fp8 GEMV

- At 6 inputs the fp8 GEMV took ~3x a one-input call. Counting instructions: with one row per
  lane group, every row decoded all M inputs again (~240 of ~350 instructions per step).
- Tried and dropped: storing the activations pre-decoded as fp16 (twice the bytes per load;
  slower even at M = 1); bigger blocks for the tile kernel (no gain over 64-lane rows).
- Kept (`fp8_gemv_cfg`, M >= 2): rows split over 8 or 16 lanes -> a new kernel decodes each
  1024-wide tile of the inputs once per block into shared memory as fp16, and every lane reuses
  it for its rows; rows over 64 lanes -> the old kernel. Two rows per lane group (four at M = 8
  over 64 lanes). Same lanes per row and the same per-lane order of k steps and per-element
  expression as one input, so still bit-identical (`bench-fp8 --bitwise`: 63 of 63).
- fp8 time per layer (no engram; `results/bench-fp8-multi.txt`): M=2 427 -> 360 us, M=5
  745 -> 572, M=6 877 -> 623, M=8 1002 -> 903; M=1 unchanged.
- Gates: e2e logits and drafts byte-identical; `--spec` identical to plain greedy on 4 prompts
  x 5 draft lengths. Verify step of 6 tokens 181 -> 171 ms; draft 11.7 ms.

| | plain | n=1 | n=2 | n=3 | n=4 | n=5 |
|---|---|---|---|---|---|---|
| chat (concurrency) | 20.5 | **20.9** | 20.7 | 19.8 | 18.1 | 16.6 |
| code (code-rust) | 17.2 | 18.5 | 19.9 | 19.5 | **20.8** | 20.1 |
| math (think) | 21.0 | 21.8 | **22.2** | 21.3 | 20.7 | 19.4 |
| zh essay | **17.3** | 15.6 | 13.7 | 11.4 | 10.2 | 8.7 |

  Code +21%, math +6%, chat +2%, zh -10%. Over 64-lane rows (attn.out, shared expert, q_a)
  the multi-token fp8 is still ~2x one input; the CPU misses (68 ms of 171) are now the
  largest part of a verify step.

## Step 25: draft length from the confidence head (phase 4)

- **What the head says:** the draft's confidence head (one linear layer per position, unused in
  the official code) is well calibrated as is: among drafts the verifier reached, sigmoid(conf)
  predicts acceptance (draft 0: 0.71 predicted, 0.71 observed; 0.9-1.0 bin: 99% accepted,
  0.1-0.2 bin: 23%); AUC 0.875; refitting the sigmoid gains almost nothing (log-loss 0.390 ->
  0.385). Measured on 6 calibration prompts that are not among the 4 test prompts
  (`results/spec-conf/`, `results/spec-conf-calibration.txt`, `results/spec-conf/spec_conf.py`).
- **Rule** (`--spec` without `--spec-n`): with p_i = sigmoid(conf_i), verify the n in 0..5 with
  the most expected tokens per ms, (1 + p0 + p0 p1 + ..) / (draft + verify(n)). Costs are
  running averages from the run itself, starting at 52 + 28 n ms. No other parameters. n = 0
  is a plain step. Offline replay of the calibration logs: 22.4 t/s vs 18.7 for the best
  fixed n, and above each prompt's own best fixed n on all 6.
- **Backoff:** after a step that chose n = 0, the next 1, 2, 4.. (at most 8) steps are plain and
  skip the draft; a step that verifies a draft resets it.
- `--spec-log FILE` writes per step: drafts verified, accepted, verify and draft ms,
  confidences.
- Gates: output identical to plain greedy on the 4 test prompts (and the calibration ones).

| t/s | plain | best fixed n (step 24) | auto | auto + backoff |
|---|---|---|---|---|
| chat (concurrency) | 20.5 | 20.9 | 22.4 | **22.4** (+9%) |
| code (code-rust) | 17.2 | 20.8 | 21.6 | **21.6** (+25%) |
| math (think) | 21.0 | 22.2 | 23.9 | **24.0** (+14%) |
| zh essay | **17.3** | 15.6 | 16.2 | 16.6 (-4%) |

  The rule picks per prompt what a fixed n cannot: chat verifies 1 draft on most steps, code
  spreads over 1-5, zh chooses n = 0 on most steps (117 of them plain, without a draft). zh
  stays ~4% under plain: steps that verify 1 draft at ~40% acceptance break even, and a plain
  step through verify() may cost more than decode(). Plain runs vary by up to ~7% between
  runs, so differences under ~5% are noise.

## Step 26: measurement on held-out prompts (phase 5)

- **Cost model fix found here:** under the oracle placement the rule verified 1 draft on 158 of
  162 steps (concurrency): the per-n running averages only updated the n that ran, and the
  others kept the study30 prior. Now verify cost is a line a + b n, fitted by weighted least
  squares to the run's steps (decay 0.97 per step) plus a weak prior (52 + 28 n, weight 3
  steps), so every step updates every n. Fitted lines: study30 ~45-68 + 26-36 n ms, oracle
  37.5 + 21.4 n (concurrency), 53.6 + 12.9 n (code). Oracle concurrency 27.0 -> 29.3 t/s; other
  prompts unchanged within noise. All numbers below use it.
- **Held-out prompts** (`results/spec-heldout/`: 8 prompts used by neither calibration nor the
  earlier tests; study30, sclk 2; plain = no draft loaded). Output identical to plain greedy
  (with the draft loaded, same placement) on all 8.

| prompt | plain t/s | `--spec` t/s | | tokens per step |
|---|---|---|---|---|
| tcp-udp (explain) | 17.6 | 20.5 | +17% | 2.1 |
| py-lru (code) | 17.2 | 21.2 | +23% | 2.9 |
| cpp-matrix (code) | 16.8 | 22.4 | +33% | 4.0 |
| sql-report (code) | 16.3 | 20.7 | +27% | 2.8 |
| geometry (math, think) | 17.3 | 22.8 | +32% | 3.4 |
| zh-travel (Chinese) | 15.0 | 16.1 | +7% | 2.0 |
| ja-email (Japanese) | 14.9 | 16.2 | +9% | 2.1 |
| summary-json | 17.8 | 23.6 | +33% | 3.9 |

  Geometric mean +22%; slowest gain on Chinese/Japanese text (the draft guesses those worst).
- **Test prompts** with the final rule: chat 22.7 (+11%), code 21.7 (+26%), math 24.4 (+16%),
  zh essay 16.4 (-5%; plain runs vary by up to ~7%).
- **With a good expert cache** (oracle placement: experts from the prompt's own routing):

| prompt | plain t/s | `--spec` t/s | |
|---|---|---|---|
| concurrency | 24.6 | 29.3 | +19% |
| code-rust | 23.9 | **40.8** | +71% |

  Speculation and the expert cache multiply: misses grow with the tokens verified, so a better
  cache makes longer drafts pay (code-rust verifies 5 drafts on 36 of 67 steps).
- **Verify step of 6 tokens** (`results/spec-profiles.txt`): 171 ms with study30 (CPU misses
  68, attention core 20, attention output 15, shared expert 14, router 10, q 7, hc 6,
  indexer + compressor 11, head 5); 121 ms with the oracle placement (CPU misses 9, GPU
  experts 9).

## Step 27: speculation and the GPU clock (sclk levels 0, 2, 8)

`results/spec-sclk/spec-sclk.sh` at the current sclk level (set by hand): the 8 held-out prompts, plain and
`--spec` back to back, study30, with the 4 dies' power sampled every 100 ms; then the median
6-token verify step. Outputs byte-identical across all three levels (`results/spec-sclk/`).
Means over the 8 prompts (`results/spec-sclk/summary.txt`):

| sclk level (MHz) | plain t/s | `--spec` t/s | spec gain | 4-die power plain / spec | J/token plain / spec | verify step (6 tokens) |
|---|---|---|---|---|---|---|
| 0 (999) | 16.8 | 19.1 | +14% | 168 / 158 W | 10.0 / 8.4 | 184 ms |
| 2 (1204) | 17.8 | 20.3 | +14% | 176 / 165 W | 9.9 / 8.3 | 170 ms |
| 8 (1730) | 19.2 | 22.3 | +16% | 236 / 217 W | 12.4 / 9.9 | 152 ms |

- The clock helps speculation only a little more than plain (+9% vs +8% from level 2 to 8). In
  the verify step the GPU parts scale with the clock (~108 / 98 / 74 ms at levels 0 / 2 / 8:
  attention core 24 / 20 / 15, fp8 projections and shared expert 41 / 35 / 27, router 12 / 10
  / 8), but the CPU expert misses do not (66 / 67 / 73 ms) and are half the step at level 8.
- Level 0 saves ~8 W against level 2 and is ~6% slower: the same energy per token, so level 2
  dominates it. Level 8 costs ~30% more power and ~17-25% more energy per token for ~8-9%
  more speed.
- `--spec` uses less energy per token than plain at every level (the dies draw less and finish
  sooner).
- Expected after the adaptive expert cache: a verify step that is mostly GPU work (the oracle
  placement's 121 ms at level 2), where the clock should be worth ~20-25%.
- Run-to-run noise: plain at level 2 measured 4-10% faster here than in step 26's runs a few
  hours earlier; within one level, plain and `--spec` ran back to back.

## Step 28: placement-invariant expert numerics (cache plan, phase 0): option A is feasible

The question (PLAN.md "Adaptive expert cache"): can an expert give the same bits on the CPU and
on a GPU, so that a cache whose contents depend on timing cannot change the output?

- **The specification** (`src/canon_dot.h`): per 32-block the exact integer sum of (2 w)(2^9 x)
  (weights as integers -12..12, activations as integers up to 2^18; |S| < 2^27), converted to
  float once (round to nearest even) and scaled by a power of two; 16 accumulators per row
  (accumulator l takes blocks l, l + 16, ...), then a fixed xor tree (8, 4, 2, 1). Exact block
  sums are order-free, so each side uses its fastest integer instructions; only the conversion,
  the accumulators and the tree are fixed. The block sum is correctly rounded: at least as
  accurate as either kernel's fp32 chains.
- **CPU** (`canon_dot_rows`): AVX-512 VNNI. Weights decoded to bytes 2 w + 12 by a table,
  activations as three signed 7-bit digits, 3 vpdpbusd per pair of blocks, an integer
  transpose-reduce over 16 blocks, then the 16 accumulators in one register; 4 rows at a time,
  L2 prefetch. A scalar version of the specification (`canon_dot_ref`) is the reference.
- **GPU** (`fp4_moe_canon`): v_dot4_i32_i8 on signed bytes (weights by two byte-permute tables),
  16 lanes per row with the same tree, 2 rows per lane group; `__fmul_rn` / `__fadd_rn` where
  hipcc would otherwise fuse a multiply-add the CPU does not.
- **The activation** (`src/canon_math.h`): a shared `canon_expf` (Cephes, explicit fmaf) and
  correctly rounded division on the GPU; the libraries' expf differ in the last bit.

Results (`results/canon-phase0.txt`, sclk 2):

| check | result |
|---|---|
| CPU vector dot vs scalar specification (gate, up, down rows; 3 layers) | 0 of 2,304 / 2,304 / 5,120 rows differ |
| whole expert block, GPU vs CPU (quantization, gate/up, swiglu, quantization, down, sum; 3 cases) | **0 of 5,120 outputs differ** |
| accuracy vs the double-precision reference | 5.5-6.6e-8, as the current kernel (5.6-6.6e-8) |
| CPU speed (40 layers x 6 experts, 16 threads) | 71.0 vs 67.5 ms (+5%) |
| GPU speed (6 experts, one layer) | 319 vs 283 us (+13%); 370 us (+31%) with 1 row per lane group |

Decision: **A**, within the 15% bound on both sides. Still to do for the runtime (phase 3 of
the plan, alongside the cache): per-expert outputs summed in router-rank order (today each die
and the CPU pre-sum their own experts), the multi-token chunk version, the small-count GPU case
(1-3 experts per die), the gather in `canon_dot_rows` that reads 3 bytes past a row's last
block (harmless inside the file, to be bounded at the tensor's end), and further speed (shared
memory for the activation digits on the GPU).

## Step 29: cache simulation with realistic constraints (cache plan, phase 1)

The question: how much of the oracle placement's speed can an adaptive cache reach when loads
are limited, take time and compete with the CPU, and what policy should decide the loads?

- **Data**: routing logs now cover verify chunks too (log version 2: every forwarded token,
  rejected drafts included, with its forward's number; the last record of a position is the
  committed one). `results/spec-routing.sh` logged the 8 held-out prompts with `--spec` and
  study30, up to 1,024 tokens (5,480 generated tokens in 2,018 steps); every output starts with
  the earlier 300/500-token output (logging changes nothing). Plain decoding of the same texts
  is replayed from the committed tokens; the 30 study logs (plain) are a second set, warm
  started leave-one-out.
- **Simulator** (`results/cache_sim.py`): 6,201 slots (the runtime's, with the draft loaded),
  warm start from study30; a step's slots hit if their expert is resident when it starts; loads
  start at up to R per second of modelled time, evict a victim at once and become resident 1
  step later (sensitivity: 2-8).
- **Time model**, fitted on the 16 measured study30 runs (plain and `--spec`):
  step = 39 + 15 (M - 1) + 10.5 draft (M > 1) + max(0, 0.27 x missed slots - 6.5) ms. The GPU
  part (39 + 15 per extra token) is the oracle placement's verify step by chunk size; the miss
  cost matches step 12's 0.27 ms per expert-token, with 6.5 ms per step hidden under GPU work.
  Error 4.8% rms, within 10% on every run, no bias between plain and spec. (Fitting all four
  numbers on the study30 runs alone confounds the chunk size with the misses and gave either
  `--spec` slower than plain at full hits or plain at full hits far above the oracle's 25 t/s.)
  Loads are shown free (fully overlapped) and charged like one missed slot each (an 18.8 MB host
  read competing for memory bandwidth); phase 2 will measure where between the two they land.

Results, held-out prompts (`results/cache-sim.txt`; est t/s; "charged" in brackets):

| policy | loads/s | hits | load GB/s | `--spec` t/s | plain t/s |
|---|---|---|---|---|---|
| static study30 | 0 | 60.5% | 0 | 18.8 | 17.2 |
| LRU (load on every miss) | 50 | 69.4% | 0.9 | 21.2 (21.0) | 19.5 |
| | 100 | 77.6% | 1.8 | 24.2 (23.5) | 21.8 |
| | 200 | 86.2% | 3.7 | 28.1 (26.7) | 23.9 |
| | unlimited | 94.7% | 8.0 | 33.5 (30.0) | 25.5 (23.5) |
| admit after 3 misses in 256 tokens | 50 | 78.6% | 0.9 | 24.6 (24.3) | 21.4 |
| | 100 | 85.7% | 1.8 | 27.9 (27.2) | 23.7 |
| | 200 | 90.9% | 2.7 | 30.9 (29.7) | 24.9 |
| rebalance every 16 tokens, last 256 | 50 | 83.3% | 0.9 | **26.7 (26.4)** | 23.1 |
| | 100 | 87.5% | 1.9 | **28.9 (28.1)** | 24.1 |
| | 200 | 90.6% | 3.3 | 30.7 (29.3) | 24.6 |

- **The load rate is the knob, and returns diminish fast**: 50 loads/s (0.9 GB/s) already
  gives `--spec` +42% over static, 100 loads/s +54%; unlimited LRU needs ~8 GB/s for +60-78%.
- **Frequency beats recency when loads are limited**: at 50-100 loads/s, LRU spends its budget
  on one-off experts; counting use over the last 256 tokens (rebalance every 16 tokens: the
  most-used non-resident experts replace the least-used residents) gives +2.5 to +5.5 t/s more
  at the same rate. LRU wins only when loads are unlimited.
- **Speculation gains more from the cache than plain decoding**: `--spec` over plain goes from
  +9% (static) to +20% (100 loads/s) and +25% (200), since the CPU misses grow with the chunk
  (more tokens, more distinct experts) while the GPU part grows by only 15 ms per token.
- **Rejected drafts do not pollute the cache**: driving it from committed tokens only changes
  hits by under 0.5 points; the forwarded tokens' hit rate equals the committed ones'. So the
  cache can act on all forwarded tokens and need not wait for the verify result.
- **Latency is cheap at moderate rates**: 1 -> 8 steps costs 0.4 t/s at 100 loads/s (2.2 at
  unlimited).
- On the 30 study logs (plain, leave-one-out) the order is the same, from a higher static start
  (70.6% hits, 19.4 t/s): rebalance at 50-100 loads/s 24.1-24.7 t/s, unlimited LRU 25.5.

Not modelled: the split over dies (die 0 has half the others' slots; a layer's hot experts
could crowd one die), the canonical numerics' cost (+5% CPU, +13% GPU expert kernels, step 28),
and loads' PCIe traffic next to the per-layer staging copies. For phase 2 the target is
100-200 loads/s (1.9-3.7 GB/s); for phase 3 the policy is the frequency rebalance, with the
load rate as the CLI knob (0 = today's static placement).

## Step 30: the load path, measured (cache plan, phase 2)

A load copies one expert (gate, up, down: 3 x 6.27 MB of mxfp4 blocks) from the model's mmap
to a die and unpacks it into a pool slot, on a low-priority stream (`hip/expert_loader.{h,cpp}`:
the staging, `load_expert`, and `ExpertLoader`, one thread per die fed by a queue). Two paths:

- **pinned**: a host memcpy into pinned staging, then the copy (the startup upload's path);
- **direct**: the expert tensors registered with HIP (`hipHostRegister`, read-only; 289 GB in
  6 merged ranges, ~40 s, so on a background thread with pinned loads until it is done), and the
  copy engine reads the page cache itself.

Two details mattered: at most 2 matrices in flight per loader (with an unbounded queue a die's
other copies wait behind the loads, whatever the stream priority: a decode-like step went from
64 ms to 3.2 s), and blocking-sync events (a spinning wait in 4 loader threads cost the CPU
expert kernel ~5% at any rate).

Standalone (`bench-expert-load`, `results/bench-expert-load.txt`, sclk 2):

| | pinned | direct |
|---|---|---|
| throughput, 1 die / 4 dies (1 thread each) | 468 / 1,437 loads/s | 491 / 1,437 loads/s (27 GB/s) |
| loader CPU per load | 2.1-2.8 ms | 1.0-1.1 ms (API calls) |
| CPU expert kernel at 50 / 100 / 200 / 400 loads/s | +14 / +17 / +33 / +102% | 0 / -1 / +6 / +10% |
| decode-like GPU step at 50 / 100 / 200 loads/s | +0.7 / +1.4 / +2.7% | the same |

Throughput is 7-14x the 100-200 loads/s the simulation asks for. The cost is on the CPU: a
pinned load reads and writes 18.8 MB through the same memory the CPU expert kernel streams.
Copying in 1 MB pieces instead of 6.27 MB was no better for the GPU step.

Live (`results/load-test.sh`, `results/load-test/summary.txt`): the runtime with
`--load-test R` keeps 8 spare slots per die and loads random experts into them at R loads/s in
all, as the cache would, while decoding the 8 held-out prompts (study30):

| mode | loads/s | path | mean t/s | vs no loads | output |
|---|---|---|---|---|---|
| `--spec` | 0 | | 20.10 | | |
| | 50 | direct | 20.29 | +0.9% (noise) | identical |
| | 100 | direct | 19.87 | -1.1% | identical |
| | 200 | direct | 19.05 | -5.2% | identical |
| | 100 | pinned | 18.29 | **-9.0%** | identical |
| plain | 0 | | 17.38 | | |
| | 100 | direct | 16.99 | -2.2% | identical |

- **Direct is the path**: at 100 loads/s it costs ~1-2%, pinned ~9%.
- **The cost is at or below the simulation's "charged" column** (step 29: one missed slot,
  0.27 ms, per load: 2.7% at 100 loads/s), so the simulated gains hold: at 100 loads/s the
  frequency policy's +54% (`--spec`) against ~1-2% for the loads.
- 200 loads/s costs 5%; the simulation's gain from 100 to 200 is +6% free / +4% charged, so
  ~100 loads/s is the sensible default and 200 the useful maximum.

(The spec outputs differ from the plain ones here: without the draft, die 0 holds 1,296 experts
instead of 871, a different placement; the gate compares runs with the draft loaded, step 19.
The canonical numerics of step 28 remove this.)

## Step 31: placement-invariant numerics in the runtime (cache plan, phase 3a)

`--canon` makes the runtime's output independent of where the routed experts run, so an
adaptive cache (whose contents depend on timing) cannot change it.

- **Rows instead of partial sums.** Today each die and the CPU pre-sum their own experts, so the
  result depends on the split. With `--canon` every expert writes its own output row: on a die
  `fp4_moe_canon_rows` (the phase-0 kernels, now for chunks of tokens: grid over token and
  expert; rows compact per die), on the CPU `CpuMoeKernel::forward_canon_rows`. Both give the
  same bits (`bench-fp4-moe`: 0 of 5,120 differ).
- **One sum in a fixed order.** The dispatch also records, for each token's router rank, which
  place runs it and its index there; `k_moe_combine` adds shared + the 6 rows in rank order
  (`__fadd_rn`), finding each row from the dispatch counts.
- **Copies sized exactly.** The host already waits for the dispatch lists (for the CPU's work),
  so it enqueues each die's copy back with the exact number of rows, about the same bytes as
  the fast path's one partial sum per token.
- **Cost of the GPU kernel:** batching over experts (each expert's down projection in parallel
  instead of a loop) brought the canonical expert block from +13% to +2% over the fast kernel
  (287 vs 283 us per layer, 6 experts). The end-to-end cost on the held-out prompts: step 32.
- **The fast path is unchanged** (the default; same output as before, checked on a 300-token
  run against `results/spec-sclk/out2`).

Gate (`results/canon-gate/canon-gate.sh`, `results/canon-gate.txt`, 128 tokens, 3 prompts): with `--canon`,
every configuration gives the same output bytes:

| configuration | tcp-udp | py-lru | zh-travel |
|---|---|---|---|
| no placement (all experts on the CPU) | reference | reference | reference |
| study30 | same | same | same |
| oracle placement (code-rust's) | same | same | same |
| study30 on dies 1-3 only | same | same | same |
| study30 + draft, `--spec` (die 0 holds fewer experts) | same | same | same |
| oracle + draft, `--spec --spec-n 5` | same | same | same |

So with `--canon`, speculative decoding also equals plain decoding *without* the draft loaded,
which was impossible before (the draft's memory changes the placement, step 19).

## Step 32: the adaptive expert cache (cache plan, phase 3b)

`--canon --cache-loads R` (`hip/expert_cache.{h,cpp}`): the GPU pools follow the experts in use.

- **Policy** (phase 1's best at limited rates): count each (layer, expert)'s use over the last
  256 forwarded tokens (rejected drafts included, from the dispatch lists the host already
  has); every 16 tokens the most-used non-resident experts replace the least-used residents (a
  swap only if the newcomer was used more); loads start at up to R per second.
- **Protocol** (between forwards, when no GPU work reads a pool): finished loads are published;
  a swap first removes its victim from the routing table (it runs on the CPU from the next
  forward), the table is uploaded, and only then the loader may overwrite the slot. The loader
  (`expert_loader.h`) reports a load once its copy and unpack have completed.
- **Loads only direct:** until the expert tensors are registered (~30-40 s after startup) the
  cache counts and plans but does not load (pinned loads cost ~9%, step 30); `--cache-wait`
  waits for registration at startup.
- **Stress test:** 1,000 loads/s, rebalance every token over 16 tokens: 1,255 swaps in 128
  tokens, output byte-identical to the all-CPU canonical run.

**The canonical cost, found and fixed.** The first live run cost 9-11% (plain 15.69 vs 17.58
t/s), against the kernel benchmarks' +5% (CPU) / +2% (GPU). A profile (median step 56.7 -> 63.2
ms) split it into: the combine kernel, one block per token (+2.6 ms: a single block of 256
threads per layer); and the CPU path (+2.6 ms of compute), whose input digits and per-expert
activation ran on one thread: the runtime's CPU usually has 1-2 misses per layer, where the
benchmark used 6 (`bench-cpu-moe -k`: +8.3% at 1 expert, +6.6% at 2, +5.0% at 6). Both now run
per 32-value block over all threads, and the combine kernel over (elements / 256, tokens):
canonical costs 1-2%, and its CPU kernel is now faster than the fast one at 1-2 experts (-22%,
-14%; the fast path has the same one-thread preparation, a later fix). Exactness unchanged
(vector = scalar specification on every row; GPU vs CPU 0 of 5,120 differ).

Results (`results/cache-test/cache-test.sh`, `results/cache-test/`, 8 held-out prompts, 300 tokens,
study30 warm start, sclk 2; "same" = every output byte-identical to the plain static `--canon`
run, `--spec` included):

| configuration | plain t/s | on GPUs | `--spec` t/s | on GPUs |
|---|---|---|---|---|
| fast kernels, static | 17.78 | 63.5% | 19.97 | 60.8% |
| `--canon`, static | 17.42 (-2.0%) | 63.5% | 19.73 (-1.2%) | 60.8% |
| `--canon --cache-loads 50` | | | 23.73 (+18.8%) | 76.3% |
| `--canon --cache-loads 100` | 20.44 (+15.0%) | 82.3% | 24.51 (+22.7%) | 79.9% |
| `--canon --cache-loads 200` | | | **25.55 (+27.9%)** | 83.8% |

At sclk 8 (`results/cache-test-sclk8/`): fast static 22.19, `--canon --cache-loads 200`
**29.09 t/s (+31%)**, 83.1% on GPUs, outputs identical to the level-2 runs. The clock now
helps `--spec` more (+14% with the cache vs +11% without; step 27 found +9% without, as CPU
misses do not scale with it): the cache turned CPU misses into GPU work.

- **Every prompt gains**, the ones the static placement serves worst most: ja-email 15.7 ->
  22.0 (`--spec`, 100 loads/s), zh-travel 15.9 -> 20.4.
- **Against the simulation** (step 29: 87% hits, ~28.9 t/s at 100 loads/s): live 80% and 24.5
  t/s, in 300-token runs (the simulation used 1,024: the cache adapts for a larger share of a
  short run) and including the canonical cost the simulation did not model.
- **It does not settle in long runs:** each rebalance plans 600-1,700 swaps, more than the rate
  allows before the next one; near the cut-off (~6,200 slots of ~10,000 experts in use) counts
  are 1-3 and noisy. Hysteresis (a newcomer must beat its victim by a margin) is the tuning
  item (PLAN phase 4, deferred until after long context).
- The full placement-invariance gate (`results/canon-gate/canon-gate.sh`) passes again on this code.

## Step 33: where a verify step goes with the cache

`results/verify-breakdown.txt`: 6-token verify steps (`--spec-n 5 --profile`), `--canon`,
sclk 8, the concurrency prompt (chat), medians of the second half of a 256-token run:

| ms per verify step (6 tokens) | static study30 | + cache 200/s |
|---|---|---|
| **CPU experts** (the GPU waiting for them) | **77.6** | **24.8** |
| (host CPU expert compute, partly overlapped) | 85.4 | 32.3 |
| (expert uses per step on GPU / CPU) | 1,156 / 284 | 1,348 / 92 |
| **GPU experts** (visible part; the rest overlaps the shared expert) | 2.9 | 4.2 |
| attention: core (per token, in order) | 15.8 | 15.9 |
| attention: output projection | 11.1 | 11.2 |
| attention: q, indexer, compressor, kv | 15.0 | 15.2 |
| shared expert | 13.4 | 14.4 |
| router + dispatch + sum (per layer) | 10.5 | 10.6 |
| hyper-connections | 4.7 | 4.7 |
| output head | 3.7 | 3.7 |
| engram | 14.6 (cold table pages; ~3 warm) | 2.8 |
| **step** | **172.4** (~160 warm) | **110.8** |
| draft (separate) | 8.8 | 11.6 |

With the cache, a verify step plus its draft is ~122 ms: **the dense part ~78 ms (~64%)**, CPU
experts ~25 ms (~20%), the draft ~12 ms (~10%), GPU experts ~4 ms visible (~3%). Before the
cache the CPU experts were nearly half. So the next levers are the dense part of a verify step
(attention ~42 ms, of which the core runs token by token; the shared expert's fp8 GEMVs ~2x one
token at 6; the per-layer routing with 6 top-k launches), then the remaining CPU misses, then the
draft. The ideas are collected in PLAN.md "Further optimization ideas".

## Step 34: long context in the runtime (PLAN.md "Long context", items 1, 2, 4)

Checked against the official `inference/model.py`, everything the runtime implemented already held
at any position (YaRN only on the compressed layers, rope angles with full-precision cosf/sinf, the
window, compressor groups, engram, the draft's window). What stopped it at 4,096 tokens:

- **Top-k for any number of positions** (`k_topk`): the indexer keeps the best 512 compressed
  positions (up to n = max_ctx on the ratio-1 layers). The old kernel was a bitonic sort of all
  scores in shared memory (at most 4,096). Now one block does a radix select on the scores' order-
  preserving bit patterns (four 8-bit passes give the 512th value and how many of its ties to take),
  then compacts in position order: the output is sorted by position, as the reference sorts its
  `topk` indices, and ties go to the lower index. (Attention now receives the positions in sorted
  order, so outputs with 512-4,096 compressed positions differ in the last bits from before; below
  512 every position is attended and nothing changed: a 300-token run is byte-identical.)
- **Candidate blocks beyond 16,384 positions** (`k_candidate_blocks`, `k_mask_candidates`): layer
  20's indexer scores blocks of 8 positions by their best score (the newest, partly filled block
  pinned in), keeps the top 2,048 with the same top-k kernel, and layers 24, 28, 32, 36 mask their
  scores outside those blocks before their own top-k. Per token of a verify chunk, as the top-k.
- **`--max-ctx`**: the runtime defaults to 65,536 (the original training context; YaRN extends it
  16x) and accepts up to 1,048,576; ~6.4 KB per position on GPU 0 (fp32 compressed caches and
  index keys): ~420 MB at 64k (13 fewer expert slots on die 0), ~6.7 GB at 1M (until the caches move
  to fp8/fp4, open). The CPU path keeps 4,096 and still stops at 16k.

**Gate against the reference's own code** (`ref/check_topk.py` with `bin-hip/test-topk`,
`results/check-topk.txt`): random and indexer-like scores (sums of rectified values), n = 513 to
1,048,576: `k_topk` gives exactly `torch.topk(...).indices.sort()`; `k_candidate_blocks` exactly
`select_candidate_blocks` (copied verbatim); candidates + masked top-k exactly the reference's
two-level selection. Where the 512th value is tied (exact zeros of rectified sums), torch's pick is
unspecified: the selected values match and our tie rule holds. 46 of 46 cases.

**End to end** (`results/long-context/`, fast kernels, study30, sclk 2):

| prompt | tokens | prompt t/s | generation t/s (at position) | output |
|---|---|---|---|---|
| first 24 KB of README.md + a summary request | 7,394 | 16.4 | 17.5 (~7.4k) | accurate summary |
| all of README.md + a summary request | 29,553 | 17.0 | 17.2 (~29.6k) | accurate summary of the whole document |
| short (step 32) | ~20 | | ~18.5 | |

The indexer's cost grows with the context but is still small at 30k (-7%). The prompt is processed
one token at a time: 29.5k tokens took 29 minutes (batched prompt processing, PLAN item 20, is
what fixes that). With `--canon`, the 65,536-token default leaves die 0 with 858 expert slots
instead of 871, and the output is byte-identical to the canonical gate reference.

Not done yet: compressed caches in fp8/fp4; an end-to-end numerical gate at long positions
(step 35); speed against position beyond 30k.

## Step 35: the long-context gate against the reference (20,000 tokens)

The reference run that was expected to take all night (22 minutes for 1,100 tokens) took 42
minutes for 20,000: the 1,100-token run was dominated by fixed costs (materializing the experts
of every layer), not by the tokens. Prompt: the first 20,000 tokens of this file at commit
ddf9b43 (`prompts/ds-completion-readme-ddf9b43.md`, raw, no chat template), then 8 greedy tokens.
That covers both new paths: layers 2-19 pick 512 of 10,000 positions (the old kernel held 4,096),
layers 20-39 512 of 20,000, after layer 20 kept 2,048 of 2,500 candidate blocks. Peak memory of
the reference: ~500 GB, mostly freed-but-kept heap (the block inputs it keeps are 34 GB).

**End to end** (`ds-test --runtime --e2e`, `results/compare-readme20k-ref-vs-rt.txt`): argmax
agrees on 9 of 9 steps (": batched hyper-connections, batched"), mean KL 2.2e-2 (hello, short
context: 1.8e-2). The large KLs (4.6e-2, 6.0e-2, 7.8e-2) are the steps where the reference itself
is unsure (top-1 probability 0.14-0.36); where it is sure, KL is 1e-6 to 5e-4.

**Per layer, teacher-forced** (`results/rt-readme20k.txt`): attn_out errors 3-10e-2 at the
prompt's last 32 positions and at every decode step, ffn_out 3-13e-2, the same levels as readme1100
(3.6-7.6e-2; both against the real bf16 reference, whose own rounding sets the floor). Expert sets
agree on 93% of decode layer-tokens (readme1100: 94%).

**Indexer picks** overlap the reference's with Jaccard 0.85-0.94 (readme1100: 0.97-0.99). Not
comparable: at 1,100 tokens layers 2-19 picked 512 of 550 positions, almost all of them. Whether
the differences are near-ties was checked against the reference's own scores: `ds_ref.py
--scores` records what each indexer hands to its top-k, `ds-test --dump-topk` our picks, and
`ref/check_topk_ties.py` ranks every pick we make that the reference did not
(`results/topk-ties-readme20k.txt`):

| layer | Jaccard | extra picks per query | their rank in the reference's order: median / p99 / max | gap below its 512th score (fraction of its top-512 range): median / p99 / max | outside its candidate blocks |
|---|---|---|---|---|---|
| 2 | 0.907 | 24.9 | 547 / 738 / 890 | 0.1% / 0.9% / 1.3% | - |
| 8 | 0.936 | 16.9 | 536 / 655 / 972 | 0.3% / 1.9% / 3.8% | - |
| 14 | 0.941 | 15.6 | 533 / 672 / 776 | 0.5% / 3.1% / 5.6% | - |
| 20 (source) | 0.923 | 20.4 | 540 / 692 / 996 | 0.5% / 2.5% / 4.2% | - |
| 24 | 0.853 | 40.8 | 566 / 946 / 1428 | 0.3% / 1.8% / 3.2% | 92 of 1,632 |
| 28 | 0.919 | 21.6 | 543 / 704 / 822 | 0.4% / 2.2% / 3.6% | 8 of 864 |
| 32 | 0.914 | 23.1 | 544 / 731 / 997 | 0.6% / 3.5% / 6.0% | 10 of 924 |
| 36 | 0.925 | 20.1 | 538 / 754 / 902 | 0.7% / 4.8% / 7.1% | 23 of 804 |

(40 queries per layer: the prompt's last 32 positions and the 8 decode steps.) Every pick we make
that the reference did not lies just past its cutoff (median rank 533-566 of 10,000-20,000), less
than 1% of the top-512 range below its 512th score in the median: near-ties, which the reference's
bf16 scores (32 heads summed in bf16) and our fp32 ones break differently. Before the consumers no
pick falls where the reference could not pick at all (no visibility or masking bug); on layers
24-36 a few do, the blocks layer 20 itself ranked differently near its 2,048-block cutoff.

Tooling fixed on the way: `ds-test --runtime` compared the prompt step only for goldens that keep
every position (with `--keep-last` it skipped all but h_in: each key now uses its own kept rows);
`ds-test --max-ctx` (default 65,536 with `--runtime`, as `ds-completion`).

## Step 36: prompts from a file, and the reasoning effort

`-f prompt.txt` reads the prompt from a file (as is). `--effort N` sets `--think`'s reasoning
effort, 1-100 or `low` / `high` / `max` (50 / 75 / 100); until now it was fixed at 75, the
official default. The prompt string is the one `encoding/encoding.py` builds
(`encode_messages(..., thinking_mode="thinking", reasoning_effort=...)`): for a one-line question
at `--effort max`, 39 tokens on both sides. First long-reasoning run (the person's, not logged;
`--spec --canon --cache-loads 200`, sclk 6, effort max): 9,813 tokens of reasoning and answer at
31.7 t/s. Its prompt took 5 minutes for 192 tokens because the model's pages had been evicted by
the step 35 reference runs (~500 GB) and were read from disk on first use; warm, the prompt runs
at its usual ~15 t/s.

## Step 37: the prompt in chunks (PLAN item 20, stage 1 and the first part of stage 2)

The prompt went through the runtime one token at a time (~19-25 t/s): a 30k-token document took
half an hour. Every change below keeps the output byte-identical to token by token
(`results/prefill/prefill-gate.sh`, `results/prefill/gate.txt`).

- **Chunks through the batched forward.** `Runtime::prefill` runs all but the last prompt token
  through `forward_chunk` (the path speculation verifies with) without the output head, then
  `decode()` for the last. The dense projections are bit-identical per token at any batch size
  (step 20), the attention part still runs token by token inside the chunk, so the window ring
  and compressor state end up exactly as after per-token decoding. First version (chunks of 8
  through verify + commit): 7,384-token prompt 19.4 -> 24.5 t/s plain, 17.8 -> 27.6 with `--spec`.
- **Chunks of up to 512** (`--prefill-chunk N`, default 512 = `Runtime::kChunk`). The
  fp8 and f32/bf16 matrix-vector kernels take at most 8 inputs, so larger chunks call them in
  slices of 8 (each row's result does not depend on M). The per-token buffers grew (~1 GB on die
  0: 1,254 expert slots instead of ~1,300). That changes the fast path's bits through the
  placement, so the gate compares with `--canon` across builds, and fast chunk 1 against 512
  within one build.
- **Profile** (`--profile` now also sums the prompt's chunks): CPU experts were 53% of the time,
  0.25 ms per expert-token, one token at a time even inside a chunk.
- **CPU experts batched** (`CpuMoeKernel::forward_batch`): a chunk's (token, expert) pairs grouped
  by expert, every expert's rows read once for all its tokens, per pair exactly the per-token
  arithmetic. The fast kernel tiles 4 rows x 4 tokens (an input block loaded once for 4 rows, a
  weight block decoded once for 4 tokens); the canonical one decodes a row once for all tokens.
  `bench-cpu-moe --batch T` (`results/prefill/bench-cpu-moe-batch.txt`, 6 experts per layer,
  outputs compared bit for bit, none differ):

| tokens per expert | fast, one at a time | fast, batched | canonical, one at a time | canonical, batched |
|---|---|---|---|---|
| 1 | 0.31 ms | 0.32 ms | 0.27 ms | 0.28 ms |
| 8 | 0.23 | 0.097 | 0.22 | 0.173 |
| 16 | 0.21 | 0.086 | 0.21 | 0.185 |

  (ms per expert-token.) One token is bound by memory (18.8 MB per expert at ~70 GB/s); batched,
  the fast kernel is bound by its own arithmetic (three fp32 operations per 32 products, fixed by
  its numerics), the canonical one more so (three integer digit passes per block).

**Where a prompt token goes now** (`results/prefill/profile.txt`, 1,989 tokens, chunks of 512;
the clock level of that sitting is uncertain, 2 or 6): 33.6 t/s (from ~20 token by token;
at sclk 2 in the gate: 35.0 vs 16.8 t/s fast, 26.3 vs 17.4 with `--canon`, whose CPU kernel
gains little from batching): CPU experts 15.7 ms per token, while the GPUs wait;
the GPU's own work ~15 ms (attention core 5.0, fp8 projections ~7.5: 8-input kernels, router,
indexer, compressor). Next (PLAN item 20): overlap the CPU experts with the GPU's work on the
other half of a chunk, then an fp8 matrix-multiply kernel for many tokens and a batched
attention core.

Found on the way: with `--cache-loads`, the first ~90 s of a run stall every GPU call, while the
loader registers the 289 GB of expert tensors (`hipHostRegister` blocks the other HIP calls);
after a memory-heavy job it also pages the model back in. A short prompt then "takes" 90-300 s
(step 36's slow prompt was mostly this, not only the page cache). Open (PLAN).

## Step 38: CPU experts overlapped with the GPU's work on the prompt

After step 37 the GPU and the host took turns: the GPU waited for the CPU experts (~16 ms per
prompt token), then the host waited for the GPU (~15 ms). A prompt chunk now runs as two halves
A and B, each layer split at the point the routed experts start (`block_pre`: everything up to
the dies' experts and the shared expert; `block_post`: the CPU's experts onwards):

    pre(L, A)  pre(L, B)  post(L, A)  pre(L+1, A)  post(L, B)  pre(L+1, B)  ...

The host computes A's CPU experts while the GPU runs B's attention, and B's while the GPU runs A's
next layer. The causal order holds (B's attention at L follows A's). `set_half` points every
per-token buffer at the half's first token; each half has its own residual-stream pair (h / h2
swap per sublayer) and its own events; per-die outputs are per token, so B's experts never
overwrite A's results before they are copied back. On the way: the staging copy to the host
moved the expert lists of all 512 chunk tokens every layer, also when decoding one token; now
only the forward's own tokens. `--no-overlap` turns it off.

`results/prefill/overlap.txt` (sclk 2; outputs identical with and without overlap, and with
`--canon` identical to the token-by-token build):

| prompt | fast, no overlap | fast, overlap | `--canon`, overlap (no overlap: step 37) |
|---|---|---|---|
| 364 tokens | 37.2 t/s | 54.2 t/s | 46.2 (30.7) |
| 1,989 tokens | 34.8 | 51.6 | 40.0 (26.3) |

~3x over token by token (16.8 t/s). Profile (1,989 tokens, 40.6 s): the host computes CPU experts
for 33 s, the GPU is busy ~34 s and waits for the host only 6 s, so both sides are now the limit:
the GPU's attention core 10.1 s, shared expert 6.5 s, attention output 5.5 s (8-input fp8 kernels),
router, indexer, q and compressor 1.7-2.9 s each; the CPU's ~256 tokens per half give each expert
~2-3 tokens. Next: an fp8 matrix-multiply kernel for many tokens and a batched attention core on
the GPU; larger chunks for the CPU.

## Step 39: an fp8 matrix-multiply kernel for prompt chunks

The fp8 projections ran as 8-input matrix-vector calls (64 per 512-token chunk). `fp8_gemm` takes
any number of inputs with the same arithmetic per output as `fp8_gemv`: the same lanes-per-row
split (`fp8_lpr(N, K)`, a function of the shape), each lane's 16-code steps in the same order, the
same shuffle tree, so every result is bit-identical to a one-input call. Free to choose: rows per
lane group (r), inputs per tile (mt), the K tile. A block decodes a tile of mt inputs into shared
memory once (the 64-lane GEMV decoded every input again per row group); the K tile is at least one
lane stride so every lane has a step in every tile (a 512-code tile left half of the 64-lane rows
idle). Built-in: r = 2, mt = 8 (sweep in `results/prefill/bench-fp8-gemm.txt`); 8-lane rows (q_b)
gain nothing and stay with the GEMV. Not yet tuned further: 96 registers per lane (2 waves per
SIMD), ~4-6 TFLOPS of ~26; software pipelining made it slower (more registers).

| tensor (512 inputs) | 8-input slices | fp8_gemm |
|---|---|---|
| attn_q_a, attn_kv | 5.3, 2.1 ms | 1.9, 0.8 ms (2.7-2.8x) |
| attn_output_a / _b | 16.6 / 16.7 | 10.2 / 11.3 (1.5x) |
| ffn_gate_shexp / down | 6.4 / 2.9 | 3.3 / 2.3 (1.3-1.9x) |
| engram_kv | 56.7 | 42.8 (1.3x) |

All bit-identical (bench, and end to end: `results/prefill/gemm-e2e.txt`: `--canon` against the
token-by-token build, fast chunk 512 against chunk 1). In the prompt (1,989 tokens): shared
expert 6.5 -> 1.9 s, attention output 5.5 -> 3.5 s, the GPU's busy time ~34 -> ~27 s; prompt 51.6
-> 57.1 t/s fast, 40.0 -> 40.5 with `--canon`: the CPU's experts (30 s of 36) are now the limit.

The options for prompt processing from here (and an estimate of when streaming the CPU's experts
to the GPUs beats computing them on the CPU: from ~512-token chunks, clearly above ~1,000) are in
PLAN.md, "Prompt processing: where it stands and the options" (P1-P9).

## Step 40: tuning the fp8 matrix-multiply kernel (PLAN P8)

Three sweeps (`results/prefill/bench-fp8-gemm-tuning.txt`, 256 inputs, every configuration
bit-identical to one-input calls) of knobs that leave each output's arithmetic alone:

- **A minimum of blocks per CU** (`__launch_bounds__`, caps the registers): helps the 8- and
  16-lane shapes at 2 rows x 16 inputs and 3 blocks per CU (q_b 1.0 -> 1.3x, indexer q_b 1.4 ->
  1.8x, shared down 1.3 -> 1.6x over 8-input slices). Kept as their built-in choice.
- **Capping the unroll of the input loop**: much slower everywhere (the accumulators then need
  dynamic register indexing). Dropped.
- **Inputs decoded once per call into global memory** instead of once per block into shared
  memory (each input tile was decoded again by every row block, 160 times for q_a): not faster.
  Decoding the inputs was not the limit. Dropped.
- **Scales from the cache instead of shared memory** (lets 16-input tiles of 64-lane rows fit two
  blocks per CU): 64-lane rows at 16 inputs 1.35 -> 1.97x, still below 8 inputs (2.8x); 16-lane
  rows slower. Dropped.

What is left: at sclk 2 (1,204 MHz) the best configurations reach ~20% of `v_dot2` peak; the
instruction mix (weight decode, scale multiply, shared-memory reads around each 8-`dot2` chain)
caps it near ~40%. End to end nothing changes (52.8-57.1 t/s over three runs, the spread is the
CPU's expert time, 30-32 s): the CPU's experts set the pace (PLAN P2/P4/P5).

## Step 41: the canonical CPU kernel tiled for prompt chunks (PLAN P5)

Batched, the canonical kernel (exact integer block sums, `canon_dot.h`) cost 0.17-0.19 ms per
expert-token, twice the fast one: like the fast kernel before step 37's tiling, it took one row
at a time and re-read every token's activation digits (15 KB each) per row, mostly from L2.
`canon_dot_rows_multi` now takes 2 rows per pass: per 16-block group both rows' decoded weights
go to a 1 KB buffer in L1 (they do not fit in registers next to the 2 x 8 partial sums), each
token's digits are loaded once for the two rows, the tokens go in blocks of 8 (their digits for
a group then stay in L1; at 16+ tokens it was slower than at 8 without). Per (row, token) the
same steps, so bit-identical. 3 rows per pass: no better.

`results/prefill/bench-cpu-moe-batch.txt` (ms per expert-token, batched; no layer differs):

| tokens per expert | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| canonical, step 37 | 0.28 | 0.173 | 0.185 | |
| canonical, now | 0.30 | 0.139 | 0.145 | 0.142 |
| fast (unchanged) | 0.32 | 0.088 | 0.088 | 0.077 |

End to end with `--canon` (`results/prefill/canon-e2e.txt`, output identical to the token-by-token
build): 364 tokens 46.5 -> 53.6 t/s, 1,989 tokens 40.4 -> 48.8 t/s. The canonical kernel is still
~1.6x the fast one: 3 integer digit passes and 2 shifts per block pair on one port, against 3
fp32 operations on two.

## Step 42: stream the CPU's experts to the GPUs for the prompt? Measured routing, modelled cost (PLAN P1)

Step 41 left the prompt limited by the CPU's experts. Whether loading them to the GPUs instead
pays depends on how many tokens each non-resident expert gets per chunk: a load costs a fixed
~0.7 ms of host-link time per expert (1,437 loads/s over the four dies, step 30), the CPU a weight
read plus a cost per token. Routing logs of two long prompts (README prose, 7,369 tokens; C++ code,
11,286) and `results/prefill/prefill_stream_sim.py` model three policies per chunk and layer: all on the
CPU (today, on the overlap's half-chunks), all streamed, and a hybrid (the most-used non-resident
experts streamed, the rest on the CPU, split so the two take equally long; they run in
parallel).

Calibration: on the same README prompt (`--profile`, 58.5 t/s) the runtime's CPU experts took
14.4 ms per token where the benchmark-based model says 8.7: the benchmark routes every token to
the same 6 experts, the runtime pays for uneven counts, grouping and per-layer synchronization.
The model's CPU costs are scaled by that 1.65. `results/prefill/stream-sim.txt`:

| chunk | non-resident experts per layer | tokens each | CPU, today | all streamed | hybrid (share streamed) |
|---|---|---|---|---|---|
| 256 | 82-89 | 6-7 | 16.6-19.5 ms per token | 9.0-9.7 | 4.1-4.4 (45%) |
| 512 | 109-116 | 9-11 | 14.4-16.9 | 6.0-6.3 | 2.8-3.0 (47%) |
| 1,024 | 136-141 | 15-18 | 12.7-15.0 | 3.7-3.8 | 1.8-1.9 (49%) |
| 2,048 | 156-161 | 24-29 | 11.5-13.7 | 2.4 | 1.2 (51%) |

(fast numerics; canonical: CPU 22-27 ms at 512, hybrid 3.0-3.2; at 1,000 loads/s the hybrid at 512
is 3.4-3.6.) One long document uses fewer experts than the per-token average suggests: ~110 of the
~220 non-resident ones per layer at 512, with ~10 tokens each (estimated before: ~6).

So at today's chunk size the hybrid brings the expert side from ~15 to ~3 ms per prompt token in
both numerics, and the GPU's own work (11.3 ms per token measured here) becomes the limit: ~58 ->
~80 t/s fast, ~49 -> ~80 canonical. A design constraint: layer L + 1's routing depends on layer L's
output, so loads cannot run a layer ahead; they start when a half-chunk's routing is known and
overlap the other half's attention, as the CPU's experts do today. The expert side needs no
tensor parallelism (each die takes whole experts; all four dies count because each card's x16
link carries its share of the loads). Splitting the dense part (TP2 within an XGMI pair, or a
layer pipeline) is the lever after that. Caveats: the load rate was measured alone; the CPU
calibration is from one prompt; direct loads need the registration stall fixed (P3).

## Step 43: the registration stall, found and fixed (PLAN item 30, P3)

With `--cache-loads`, a run's first ~90 s did nothing: the loader registers the 289 GB of expert
tensors with HIP (`hipHostRegister`, for direct loads), and a 13-token prompt took 92 s. The
profile put the time in groups that are microseconds of GPU work (`attn.q` 27 s, `ffn.router`
26 s, `engram.fetch` 22 s): the GPU idled, waiting for the host thread to submit.

`bin-hip/bench-register` (`results/prefill/bench-register.txt`) tested the hypotheses in turn,
registering 8 GB of a model shard per run (a fresh region each: pages pinned once pin again
faster) while probes run:

- **Does registration block other HIP calls?** No: a kernel + two small copies + an event
  synchronize kept their ~0.04 ms round trip (worst 3.5 ms) even with the whole range in one call.
- **Does it block page faults on the model?** No: a reader of another, not yet mapped shard kept
  13-16 GB/s (kernel 6.10 handles faults under per-area locks).
- **Does it block `mmap` / `munmap`?** Yes: one 1 MB map + unmap waited up to **1.7 s** while
  8 GB registered in one call. A registration holds the process's memory-map lock for its whole
  duration (~2.7 GB/s), and anything that maps or unmaps memory (malloc of a large block, the HIP
  runtime's own allocations) waits for it. The runtime registered ranges of up to ~48 GB, so its
  host thread stalled for many seconds at a time.

| piece | longest mmap + munmap during the registration |
|---|---|
| whole 8 GB | 1,696 ms |
| 1 GB | 328 ms |
| 256 MB | 93 ms |
| 64 MB | 49 ms |

The loader now registers 64 MB pieces with a short pause between them. A copy may not span two
registrations ("invalid argument"), so `load_expert` splits a matrix's copy at piece boundaries
(the split copies run at full speed: 13.4 vs 13.6 GB/s). `results/prefill/register-stall.txt`:
the same short prompt with the cache now takes 0.86 s (was 92 s), and generation during the
registration runs at full speed (17.3 t/s, `--canon` plain). With `--cache-wait` the cache loads
as before (2,429 loads in a 300-token `--spec` run, GPU share of the experts 65.6% -> 81.8%),
output identical to the run without the cache. Load throughput is unchanged (1,440 loads/s over
the four dies, `results/prefill/bench-expert-load-pieces.txt`). The registration itself still
takes ~95-135 s, now in the background; the cache starts loading only after it.

## Step 44: streaming the CPU's experts during the prompt, tried and dropped (PLAN P2)

Built as step 42 proposed, behind a flag: per half-chunk and layer, the CPU's most-used experts
loaded into spare GPU slots and computed there, the rest on the CPU, the split by the calibrated
cost model; canonical rows scattered into the places the CPU's rows would have had, so the
combine and the output stay the same. It was correct (with `--canon` every run byte-identical to
the token-by-token build) and reached 65.3 t/s on the 1,989-token prompt, against 48.8 without
it (`results/prefill/stream-attempt.txt`). But the model's assumption failed: loads ran at 1.9-2.4
ms each instead of the benchmark's 0.70, and finished mostly after the CPU's share, so the
host waited for them (16-22 s of a 30-36 s prompt):

| stage | prompt t/s (`--canon`) | per load |
|---|---|---|
| without streaming (step 41, sclk 2; today at sclk 6: 48.7) | 48.8 | - |
| streaming, direct loads | 55.5 | 2.27 ms |
| the home die left out | 48.8 | 2.36 |
| loader pipelined (no wait per request) | 55.9 | 2.25 |
| CPU kernel on 12 threads | 56.0 | 2.25 |
| the loads' streams at high priority | 65.3 | 1.87 |

(Steps 42-44 ran at sclk 6, set before that session; the prompt is CPU-bound, so the clock
level hardly changes these numbers.) What limits it is contention inside the dies: a load is a copy and an unpack kernel, and the
unpack queued behind the die's own compute kernels (raising the loads' priority was the one
change that helped). Separate copy and unpack streams with more staging buffers was the next
step; the attempt was stopped there. Dropped (the person's decision) as too complicated for its
gain: it needs spare slots, a second loader, a per-layer schedule and per-half buffers, and its
benefit depends on load timing under contention that a standalone benchmark does not show. The
code was removed; registration in pieces (step 43) stays.

## Step 45: the CPU expert kernel ran 1.6x slower in the runtime than in its benchmark: -O2

Step 42's calibration found the runtime's CPU experts 1.65x slower than the benchmark-based
model. Measured in place (`forward_batch` now times its phases; `--profile` prints them):
gate/up and down were 29 of 31 s, the grouping, inputs, activations and sums ~1.7 s, so the
time was in the kernels. Ruled out one at a time (`results/prefill/o3.txt`):

- **The workload** (32% of a half-chunk's experts have one token, 26% two or three; the
  benchmark had 6 equal groups): `bench-cpu-moe --replay LOG` runs exactly the runtime's CPU
  pairs of a logged prompt: 0.111 ms per pair, against the runtime's 0.18 on the same pairs.
- **The overlap with the GPU's work:** `--no-overlap` the same per pair (0.165).
- **Thread placement:** one OpenMP thread pinned per core, the same (0.157).
- **The build:** the benchmark is compiled with `-O3`, the runtime binaries with the common
  `CXXFLAGS`, `-O2`. At `-O2` GCC does not fully unroll the tiled kernels' small loops over their
  accumulator arrays, so the accumulators live in memory.

`CXXFLAGS` is now `-O3` for every binary. In place (sclk 6, set before this session):

| 1,989-token prompt | -O2 | -O3 |
|---|---|---|
| fast | 55.5 t/s (0.161 ms per pair) | **76.0 t/s** (0.107) |
| `--canon` | 48.7 t/s (0.187) | **56.7 t/s** (0.156) |
| `--canon`, 364 tokens | 53.6 (step 41, sclk 2) | 61.5 |

No output changed: the canonical prompts are byte-identical to the token-by-token build, the
fast prompt to the `-O2` build, generation in both modes too (its one-token kernel was not
affected: 19-20 t/s either way), and the canonical placement gate passes with all 18 outputs
identical to the committed ones (`results/canon-gate.txt`). At sclk 2 the fast prompt may be
somewhat lower than 76 t/s: with the CPU this much faster, the GPU's own work (~13.6 ms per
prompt token at sclk 2, step 39) gets close to setting the pace.

## Step 46: chunks of 1,024: slower, reverted (PLAN P4)

The idea (step 37's reasoning): twice the tokens per chunk gives each CPU expert twice the tokens
per half-chunk, so fewer weight reads per pair. Measured on the 7,369-token README prompt
(`results/prefill/chunk1024.txt`, sclk 6): fast 78.6 -> 73.8 t/s, `--canon` 62.8 -> 54.9. The CPU's
cost per pair barely moved (0.114 -> 0.107 ms fast): since step 45 the kernel is limited by its
arithmetic per pair, not by reading the weights, so sharing a read across more tokens saves
little. Meanwhile the overlap gets coarser (the GPU waited 16.5 s for the CPU instead of 8.8), and
die 0 holds 12 fewer experts. `kChunk` stays 512.

## Step 47: no momentum needed in the draft-length rule; a one-page timing summary

**Momentum?** Should the draft-length rule (step 25) trust speculation more after recent
successes? Only if past success predicts acceptance beyond the draft's own confidence head. On the
step-25 calibration logs (every draft verified; 1,443 reached drafts, 6 prompts;
`results/spec_momentum.py`, `results/spec-momentum.txt`), adding the previous step's accepted
fraction, its surprise, or a running average to a fit on the confidence improves the fit on its own
data by a hair and makes predictions for left-out prompts slightly worse (log-loss 0.3872 without,
0.3879-0.3896 with). Within confidence bins, acceptance after a poor and after a good step differs
little and in both directions. The head reads the main model's last layers at the current
position, so an easy stretch of text already shows in its confidence. Not added. (Short runs
without the cache; long reasoning runs are not checked yet.)

**[TIMING.md](timing.md):** where the time goes in generation and in the prompt, on one page, with
what a millisecond is worth (~0.25 t/s per ms of a speculative step) and what-ifs (a 1 ms draft:
+11%; both prompt sides 2x faster: 76 -> ~150 t/s).

## Step 48: speculation with the cache, measured per step; a regression since step 37 fixed

`scripts/spec-measure.sh` runs the 8 held-out prompts and one long reasoning prompt with
`--spec --canon --cache-loads 200 --cache-wait --spec-log`; `scripts/spec_analysis.py` reads the
per-step logs (`results/spec-measure-sclk6/`, sclk 6). All 8 held-out outputs are byte-identical
to step 32's plain static `--canon` runs.

| prompt | t/s | tokens per step |
|---|---|---|
| tcp-udp | 26.2 | 2.16 |
| py-lru | 29.3 | 3.12 |
| cpp-matrix | 30.6 | 4.35 |
| sql-report | 28.5 | 3.66 |
| geometry | 33.5 | 4.11 |
| zh-travel | 23.7 | 1.70 |
| ja-email | 25.0 | 2.03 |
| summary-json | 26.9 | 3.50 |
| **mean** | **28.0** | |

(Step 32: 25.6 at sclk 2, 29.1 at sclk 8.)

- **The confidence head stays calibrated** in live runs with the cache: predicted vs observed
  acceptance 0.53/0.51, 0.71/0.70, 0.85/0.87, 0.98/0.99 by bin, within 0.04 per run.
- **The rule is near the best choice per step:** 94% of a hindsight lower bound (29.3 vs 31.2
  tokens per second of step time, pooled; 91-96% per run). The bound takes, per step, the number
  of drafts that turned out to be accepted, at the run's own fitted cost.
- **Momentum again adds nothing** (step 47's test: 0.2649 without, 0.2655 with).
- **The verify cost follows the cache:** in the long prompt the verify step falls from ~94 to
  ~57 ms over the first ~150 steps as the cache warms. The fitted lines with the cache are
  38-77 + 11-21 n ms.

**The long reasoning prompt loops.** Its greedy output repeats "What about (45,0). What about
(0,45)." from ~800 tokens on to the 3,000-token limit, with correct reasoning before it.
Checked as a possible bug (`results/spec-loop-check.txt`):

- plain decoding gives the same bytes, so speculation and the cache are not the cause;
- the build from before the prompt-processing work (73b6891) gives the same bytes, so nothing
  since has changed it;
- the independent CPU path (ggml, fast numerics) diverges early on a near tie, then does not
  loop;
- given the GPU's own text up to the loop as a prompt, both the CPU path and the runtime
  continue with the same loop.

It is the model's greedy behaviour on this prompt; the official evaluations sample at
temperature 1.0, and sampling is a non-goal here. One trap on the way: the prompt prefix must end
where the generated token did. With a trailing space (a separate token), both implementations
continued without the loop.

**A regression since step 37, found by this measurement.** The first run of these prompts was
~10% slower than step 32 at the same acceptance. Builds compared on the same prompt and clock
(`results/engram-copy-fix.txt`): with the cache, the build from before the prompt work did
26.5-26.9 t/s, the current one 25.6-25.7. The cause: `fetch_engram_rows` copied all `kChunk`
token rows of every engram slot to the GPU each step, 512 rows since step 37 for a step that
needs 1-6. Copying only the forward's rows: 26.1-26.4 t/s, output identical (the rest, ~0.5 ms
per step, is die 0's ~70 fewer expert slots). The byte gates could not catch this: the output
never changed, only the time.

## Step 49: a decode profile with speculation and the cache, and what launches cost

`results/decode-profile-sclk6.txt` (sclk 6, `--spec --canon --cache-loads 200 --cache-wait
--profile`, medians of the second half of a 300-token run, by tokens verified per step;
`scripts/profile_verify.py`):

| ms | chat, 2 tokens | chat, 4 tokens | code, 6 tokens |
|---|---|---|---|
| **step** | **61.8** | **91.2** | **133.6** |
| the GPU waiting for the CPU's experts | 15.7 | 19.6 | **50.2** |
| attention core (per token) | 5.5 | 11.3 | 17.6 |
| attention output projection | 7.2 | 13.2 | 11.7 |
| shared expert | 5.5 | 9.6 | 10.2 |
| router | 4.0 | 6.2 | 8.2 |
| q, compressor, indexer, kv | 9.0 | 12.6 | 15.8 |
| hyper-connections, head, engram, dispatch, sum | ~11 | ~12 | ~13 |
| waiting for the other dies' experts | 3.7 | 4.2 | 3.9 |

Even with the cache warm (87% of expert uses on GPUs), the CPU's experts are again the largest
item: a 6-token code step has ~185 of them, ~0.3 ms each. Most serve one token, so the kernel is
bound by reading 18.8 MB per expert: ~63 GB/s, about half this machine's DDR4 peak (~140 GB/s).

**Launches:** `results/hip_count_shim.c` (an LD_PRELOAD counter) counts ~2,450 kernel launches
per plain decode step and ~3,000-3,900 per verify step (draft included); `bin-hip/bench-launch`
measures ~2.2-2.5 us per launch back to back (~11.5 us for a launch plus a synchronize). So
launches cost at most ~7-12% of a step, and part of that overlaps work.

Levers by size (code / chat steps): the CPU's expert misses (38% / 25%: fewer misses, cheaper
misses); a batched attention core for verify chunks (~10% / ~4%); a batched router (~4%); kernel
fusion or graphs (at most 7-12%); the small-batch fp8 kernels (~5%). The ROCm 5.7 profiler
installed cannot trace the 6.3.4 runtime ("no ROCm-capable device"), hence the shim.

## Step 50: the CPU's misses: a cheaper kernel, a tuned cache, and how they cost per layer

Step 49 put the CPU's expert misses first. Three changes and one measurement, all at sclk 6,
every output byte-identical to step 32's plain `--canon` run:

**The kernel's preparation, in parallel** (`src/cpu_moe.cpp`): the input's quantization and the
activation (SwiGLU, quantize, permute) ran on one thread between the two matrix passes; they now
run in blocks of 8x32 values over all threads, in both `forward` and `forward_batch` (bit-identical:
each block's math is unchanged). `results/cpu-moe-parallel-act.txt`: one token, one expert 17.6 ->
11.6-13.7 ms per 40 layers (2 experts 26.7 -> 21.0-24.5); batched output unchanged (0 layers
differ). End to end ~+2.5%.

**The cache's settings** (`results/cache-tuning-sclk6.txt`): two prompts at 200, 400, 800 loads
per second, then the rebalance period and the window at 800. Faster loads let the cache follow
the routing (the actual rate stays ~300/s: the ranking, not the limit, decides how many it
swaps); rebalancing every 8 tokens adds a little more; a 128-token window does not. On the 8
held-out prompts (`CACHE="..." scripts/spec-measure.sh`):

| t/s, sclk 6 | step 48 (200 loads/s, period 16) | + parallel activation | + 800 loads/s, period 8 |
|---|---|---|---|
| tcp-udp | 26.22 | 26.93 | 27.75 |
| py-lru | 29.25 | 29.66 | 31.82 |
| cpp-matrix | 30.60 | 31.78 | 34.96 |
| sql-report | 28.52 | 29.13 | 30.19 |
| geometry | 33.53 | 34.26 | 36.32 |
| zh-travel | 23.67 | 24.27 | 25.62 |
| ja-email | 24.98 | 25.56 | 27.46 |
| summary-json | 26.89 | 27.89 | 28.18 |
| **mean** | **27.96** | **28.69 (+2.6%)** | **30.29 (+5.6%; +8.3% total)** |

Recommended now: `--canon --cache-loads 800 --cache-period 8` (the defaults are unchanged, so
older commands reproduce).

**Misses per layer** (`results/miss-histogram-sclk6.txt`; `--profile` now prints, per token
count of the forward, how many distinct experts each layer computed on the CPU, the CPU's time
and how long die 0 waited). Does a single miss hide behind the GPU's work? Barely:

| distinct CPU experts in the layer | code, 4-6 tokens: share of layers | CPU ms | die 0 waits ms |
|---|---|---|---|
| 0 | 7.9% | | |
| 1 | 15.8% | 0.475 | 0.328 |
| 2 | 16.4% | 0.750 | 0.614 |
| 3 | 14.6% | 1.039 | 0.907 |
| 4 | 11.1% | 1.343 | 1.245 |
| 5-7 | 16.0% | 1.879 | 1.807 |
| 8+ | 4.7% | 2.860 | 2.791 |

A layer costs ~0.1 ms for having misses at all plus ~0.27 ms per distinct expert: linear, no
penalty for clustering. The overlap is 0.05-0.15 ms because the work alongside is short: the
shared expert on die 0's main stream (~0.1-0.15 ms) and each die's routed experts (~0.1-0.3 ms,
step 51's trace) all finish while the CPU is still on its first expert. So the total number of
misses is the target (the hit rate), not their spread over layers.

## Step 51: a trace to look at: `--trace FILE`

`--trace FILE` (implies `--profile`) writes a Chrome trace of every step; open it in
ui.perfetto.dev. Tracks: steps (drafts, verify steps), the host thread (per layer the wait for
the staged block and the CPU's experts, with distinct experts / pairs / tokens as arguments), die
0's main stream (a "layer N" span per layer with the profile groups nested under it; engram
before layer 0 and the head after the last stay outside), die 0's expert stream and dies 1-3 (each
die's experts per layer, copies included; in generation with distinct experts / pairs / tokens as
arguments: per layer the dies' and the CPU's pairs add up to 6 x tokens in both traces). GPU events are placed on the host clock where the two are known to meet: die 0 (both streams)
at each layer's host wait for the staged routing lists (the smallest bound of the step), dies 1-3
by waiting for a reference event on their idle streams at the step's start. Checked on both
traces: die 0's wait for the CPU ends 31-69 us after the CPU's experts (the upload), dies 1-3
start 9-28 us after the host has the lists, and no expert span ends inside another layer. (The
first version used the time the step's first event was issued: die 0's track came out 0.3-0.4 ms
early, putting layer 13's experts inside layer 14. The prompt's chunks still use it: their two
halves make the per-layer bound ambiguous.) `results/trace/` holds the two runs gzipped
(chat and code, 300 tokens; Perfetto opens `.json.gz`; download them raw, not GitHub's file page).
A layer on die 0 takes a median 1.6 ms on chat (p90 2.7) and 2.5 ms on code (p90 4.0). Cost:
~7% of speed (first written as 1-3%, against the wrong baseline; README step 58); outputs
identical. (Right after a reboot runs are ~15% slower until the model's
files are back in the page cache: the CPU's experts are read from them.)

A 4-token code step, layers 0-2: the CPU 1,380 / 804 / 489 us, die 0 waiting 1,313 / 614 / 404
us, the four dies' experts 152-318 / 184-348 / 178-343 us, all starting together after routing.

## Step 52: prefetching the next layer's CPU expert? Not predictable enough

While die 0 runs the next layer's attention the CPU is idle for ~1-2 ms; it could read the next
layer's most likely CPU expert into its caches (18.8 MB vs a 22 MB L3; a correct guess would save
~0.15 ms of the ~0.3). Measured first (`--profile`, `results/prefetch-prediction-sclk6.txt`): how
often a guess made before the layer is among its actual CPU experts:

| tokens in the forward | layers with CPU experts | cache's top-1 non-resident | any of its top 3 | previous forward's top CPU expert |
|---|---|---|---|---|
| chat 1 | 38% | 17% | 29% | 20% |
| chat 2-3 | 66% | 20% | 34% | 16% |
| chat 4-6 | 82% | 28% | 42% | 22% |
| code 4-6 | 90% | 23% | 41% | 20% |

The CPU's experts are the long tail the cache did not keep, so they are the hardest to guess: at
~25% a prefetch would save ~1% of a step while evicting useful data on every wrong guess. Not
built (the bar set beforehand was ~60%). The measurement stays in `--profile`
(`ExpertCache::top_nonresident`).

## Step 53: the router's top-k in one launch

The code trace showed the router (`ffn.router` on die 0) at 72 us per layer for one token and
+26 us per extra token: 202 us at 6, more than die 0's own experts. Its matrix is small (384 x
5,120 f32, 7.9 MB, one pass for all tokens: 18 us); the time was the top-6 selection, one launch
per token in a row, each a single 64-lane wave on a 64-CU die, and each lane looping over 16
candidates (sized for 1,024 experts) where this model has 6. Now one launch for all of a forward's
tokens (one wave each) and the per-lane count sized to the expert count; the same comparisons in
the same order, so the same bits. Main model and draft both (`k_router(..., M, id_stride)`).

`results/router-batched.txt` (sclk 6): top-6 for one token 22.2 -> 12.2 us; 6 tokens 75.2 -> 12.4.

Held-out generation (`results/spec-measure-sclk6-cacheloads800cacheperiod8-router/`, the page
cache warm): mean **30.29 -> 31.69 t/s (+4.6%)**, every prompt +3-7%, every output identical to
step 32's. The prompt (`results/prefill/router-batched-ab.txt`, against a build of the previous
commit, alternating): +1% at 364 tokens, +2-3% at 1,989, +4.3% fast / +2.6% `--canon` at 6,106
(87 / 68 t/s); outputs identical, and the prompt gate's `--canon` chunks still equal the saved
token-by-token output.

**Cold page cache:** the first measurement after a reboot came out 5-15% slow on the prompts not
run since: only 18 of shard 8's 189 GB were cached, and the CPU's experts and the engram rows are
read from the mapped files. `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` (508 GB, ~3 min) before
measuring; `fincore` shows what is resident.

## Step 54: experts chosen together: some affinity, nothing to place by

Could co-selection guide the split over the CPU and the dies (`results/coselect.py`,
`results/coselect.txt`; the 30-prompt study's logs, generated tokens)?

- **Within a token**, per layer, pairs common enough to judge: the median pair is chosen together
  0.33x as often as independent choices would, p99 7.3x, the strongest 51x; pairs at >= 3x carry
  27.5% of all co-selections, >= 10x 5.2%. Affinities, but no tight clusters.
- **Neighbouring tokens:** 6 consecutive tokens need 21.1 distinct experts per layer, shuffled
  25.5 (17% fewer; 2 tokens 9%): verify steps already get this.
- **Dealing the GPU-resident experts to the dies** against co-selection (fitted on half the
  prompts, scored on the other; same experts, same count per layer and die) vs today's
  round-robin: -0.0 to -0.3% per layer; even a perfect balance of every step would gain only
  0.8-3.1%, since a layer with CPU experts waits for the CPU. On the CPU, grouping would save at
  most the fixed ~0.1 ms of a layer with misses (step 50).

**A finding on the way:** a die's expert time per layer (fitted on the code trace's spans) is
~90-100 us + **~30 us per (token, expert) pair**, almost nothing per distinct expert (1-7 us):
an expert used by 3 tokens costs ~3x one. (I took this for the weights being read once per
pair; step 55 tried sharing them and found the per-pair cost is mostly elsewhere.)

## Step 55: an fp4 expert kernel that reads each expert once: tried, reverted

Step 54's fit said a die pays ~30 us per (token, expert) pair and almost nothing per distinct
expert, and I read that as the weights being read once per pair (~20 us of HBM each). The
attempt (`results/fp4-grouped/`: numbers and `attempt.patch`): a one-block kernel sorts a
chunk's (slot, pair) keys on the device (die 0 launches before the host sees the lists), and
grouped gate/up and down kernels load each weight block once and apply it to the group's tokens,
4 at a time, with every (row, token) computed by the one-pair kernel's operations in its order.
The rows were bit-identical in every case (`bench-fp4-moe`'s new chunk test against the old
kernels).

Speed, 8 tokens: -21% when they all share 6 experts, **+23%** when they hardly share; real
steps (~1.5-2.5 pairs per distinct expert on a die, the code trace) sit near break-even. Two
probes of the hypothesis: more rows per 16-lane group (each activation load shared by more rows)
is slower (244 -> 281 -> 322 us for one token's 6 experts at 2 / 4 / 8 rows), and decoding each
weight block once for the group (more registers) is slower still (+50% without sharing). So the
weight reads are not what a pair costs; the per-pair work is the canonical kernel's own
arithmetic and latency, and without a working profiler (step 49) the next guess would be a
guess. Reverted: the gain does not pay for a sort kernel and two kernel variants. The die
expert time is off the critical path in most layers anyway (step 50).

## Step 56: the CPU's verify-step experts: schedule, pages and CPU settings, measured and dropped

A verify step's CPU experts run in `forward_batch`: pairs grouped by expert (each expert read
once for all its tokens; 83% serve one token), then all experts' gate + up rows in one parallel
phase (32-row tasks handed out dynamically), the activation, then the down rows. In the code
trace that was 0.34 ms per expert (gate + up at 62 GB/s, down at 55) against a read-only control
of 82-93 GB/s (~0.21 ms): ~35-40% headroom, ~7% of a code step if half of it were closed. Three
candidates, measured with a new `bench-cpu-moe --decode D` (D tokens per call, each its own
fresh expert, as a verify step's misses; `results/cpu-decode/`):

- **Static schedule** (each thread a contiguous range of rows, so its 16 KB-ahead prefetch runs
  in its own rows; the dynamic schedule hands the next rows to another core): 15-22% faster per
  expert at D = 2-6, the same bits. But 2.2-2.6x slower on prompt chunks (`--replay`: groups of
  1 to hundreds of tokens), and slower than dynamic after a 1.5 ms idle gap between calls (the
  runtime's wait for the GPU): 0.46 vs 0.33 ms at D = 6.
- **2 MB pages** (a copy of 4 layers' experts with `MADV_HUGEPAGE`; the model's mapping uses 4 KB
  pages, where the hardware prefetchers stop): no change for the kernel or the control.
- **CPU settings** with the gap: one worker per core (`OMP_PLACES=cores OMP_PROC_BIND=spread`) and
  spinning workers (`OMP_WAIT_POLICY=active`) did not help; the performance governor brought
  static level with dynamic, and with C6 also disabled static kept its 16-18% lead (0.254 vs
  0.304 ms at D = 6).

End to end with the performance governor and C6 off (`runtime-ab.txt`, outputs identical): static
28.5 / 36.1-36.4 t/s vs dynamic 28.7-28.8 / 36.3-36.4 on tcp-udp / cpp-matrix, and the settings
themselves gave nothing (28.94 / 36.46 with the defaults, step 53). The runtime has company the
benchmark lacks (the cache's loader threads, ~300 loads/s, and the host thread driving the GPUs);
a static split waits for its slowest thread, so a guess is that a worker sharing a core with them
cancels the gain. Not pursued: the runtime keeps the dynamic schedule and no flag; the benchmark
keeps the switch (`--static`) and the decode mode (`--gap-us`, the 2 MB copy, the control). The
CPU settings were reverted.

Still open: a call with one expert costs ~0.6 ms in the benchmark under every setting, about
0.25 ms more than per-expert streaming explains: fixed per-call work (wake-up, four barriers, the
grouping's allocations), worth finding before anything else on the CPU side.

## Step 57: a verify step's one-token expert groups on the one-token routine

Step 56 left a puzzle: a call with one expert cost ~0.6 ms, ~0.25 ms more than streaming 18.8 MB
explains. `bench-cpu-moe --decode` now splits a call into its phases and sets the one-token
kernel and an empty parallel region beside it (`results/cpu-decode/per-call.txt`). There is no
fixed overhead (the team's fork and join: 3 us; grouping, inputs, activation: ~0.03 ms): the
**down phase** took 0.32 ms for 6.3 MB, longer than gate + up's 12.5 MB, and the one-token kernel
did the whole expert in 0.34 ms. The batch path ran every group through `canon_dot_rows_multi`,
written for many tokens per row: 2 rows per step (the one-token routine: 4) and each block's
decoded weights stored to an L1 buffer and read back per token, a round trip that buys nothing
for one token, which is 83% of a verify step's CPU groups.

One-token groups now take `canon_dot_rows` (the multi routine's header promises the same bits;
`--batch 1` and `--batch 3` against one token at a time: 0 of 40 layers differ). Per call: 1
expert 0.622 -> 0.405 ms, 2 experts 0.776 -> 0.616, 6 experts 1.786 -> 1.656. Groups of 2-4 tokens
on the one-token routine once per token: within noise of the multi routine (+3, -7, +4%), not
kept. Plain one-token steps already used the one-token kernel.

Held-out generation (`results/spec-measure-sclk6-cacheloads800cacheperiod8-nx1/`): mean
**31.69 -> 32.03 t/s (+1.1%)**, code and structured prompts +0.9 to +1.5%, tcp-udp flat; outputs
identical. Smaller than the per-call gain because many layers have no CPU experts and part of the
CPU's time hides behind the GPU's. Only the canonical path changed (the fast path's
`dot_range_multi` is the analogue, unmeasured).

## Step 58: traces of today's build, and what tracing costs

`results/trace/` regenerated with the build of step 57 (same commands, sclk 6; outputs identical;
alignment as in step 51: die 0's wait for the CPU ends 31-69 us after the CPU's experts). What
measuring costs, the same two prompts (step 57's untraced run, then `--profile`, then `--trace`):

| t/s | untraced | `--profile` | `--trace` |
|---|---|---|---|
| tcp-udp | 28.86 | 27.33 (-5.3%) | 26.79 (-7.2%) |
| cpp-matrix | 36.90 | 34.66 (-6.1%) | 34.46 (-6.6%) |

Most of it is `--profile` itself (an event per group on die 0, ~70 per layer, read back after
every step, and the per-layer miss and prediction bookkeeping); the trace's own work (the dies'
events, the per-step clock anchors, the spans) adds 0.5-2%. Step 51 had put the trace at 1-3%,
comparing its traced runs (26.2 / 32.9 t/s) with nothing measured alike; the step-50 untraced
runs (27.8 / 35.0) show the same ~6% then. A trace's times are therefore a few percent long, and
its proportions hold. TIMING.md now points at the traces.

TIMING.md's generation and prompt sections were still the step 25-46 numbers (sclk 2 and 8, the
cache at 200 loads/s, before the router and CPU-routine fixes); they are now from this build at
sclk 6 (`results/timing-sclk6/`: the verify steps' and the 1,989-token prompt's `--profile`
breakdowns). A code step: 12 ms draft + ~109 ms verify for 4.5 tokens (36.9 t/s), a chat step
10 + ~65 ms for 2.2 (28.9 t/s); a 6-token verify step is attention 39%, the other dense parts
24%, the CPU's experts 22%, the draft 10%, the dies' experts 5%. On the prompt the router went
from 1.1 to 0.2 ms per token, leaving the GPU side (9.6 ms) just under the CPU's (10.4).


## Step 59: the cache's counting: decay, hysteresis and the threshold, simulated and checked live

The cache counts each expert's uses over the last 256 forwarded tokens and, every 8 tokens, swaps
the most used non-resident experts (at least 2 uses) for the least used residents. Would decayed
counts (each use +1, every token multiplying all scores by 2^(-1/half-life)) or hysteresis (a
newcomer must beat its victim by a margin) do better? `results/cache_sim.py --decay` (step 29's
simulator with `half_life`, `margin` and `min_count`; its time model is the sclk-2 fit, so
policies are compared, not absolute t/s): `results/cache-decay/sweep.txt`, `threshold.txt`.

At 800 loads/s, rebalance every 8 (est. t/s with loads free / charged like a missed expert):

| policy | held-out `--spec` | study prompts, plain |
|---|---|---|
| window 256, min 2 (today) | 92.9% hits, 6.9 loads/token, 32.2 / 30.4 | 93.7%, 5.6, 25.3 / 24.4 |
| window 128 or 512 | 92.7-92.8%, 32.1 / 30.4 | 93.6%, 25.3 / 24.3-24.5 |
| decay, half-life 128-512, min 1.5 | 92.8-93.0%, 32.1-32.3 / 30.3-30.5 | 93.6-93.7%, 25.3 / 24.3-24.4 |
| decay, half-life 128-512, min 1 | 93.3-93.5%, 8.2-9.7 loads, 32.5-32.6 / 30.1-30.3 | 93.8-94.1%, 25.3-25.4 / 24.1-24.3 |
| window 256, min 1 | 93.6%, 10.6 loads, 32.7 / 29.9 | 94.2%, 9.8 loads, 25.4 / 23.8 |
| margin 1 / 2 / 4 | 92.8 / 91.6 / 89.0%, 32.2 / 31.4 / 29.8 free | 93.7 / 92.7 / 90.6%, 25.3 / 25.2 / 24.8 free |

- **Decay is the window again.** At first it looked worse (91.1-91.9%), but "at least 2" is a
  stricter bar for decayed scores; with the bar matched, every half-life lands within 0.4 est.
  t/s of the window. Window length (128-512) does not matter either.
- **Hysteresis only costs.** Margins cut loads by up to 55%, but every load they skip was a
  useful one: even with loads charged, no margin is best. These runs are up to ~1,000 tokens;
  very long ones are not covered by the logs.
- **The threshold moves hits**, +0.5-0.7 points at min 1 for 50-75% more loads, and the
  simulator cannot say whether that pays. Checked live (`--cache-min N`, new, default 2;
  `results/cache-decay/live-min.txt`, outputs identical): tcp-udp 28.84 / 29.20 t/s at min 2,
  28.65 / 28.67 at min 1 (on GPUs 87.7 -> 88.7%, loads 3,200 -> 5,200); cpp-matrix 37.01 / 37.03
  vs 36.72 / 36.79 (86.1 -> 87.7%, 3,700 -> 5,470). Loads are not free: min 2 stays.

The cache serves ~93% of expert uses in simulation (86-88% live); the rest are experts no
counting predicts. More GPU slots, not a different policy, is the lever left on this side.

## Step 60: where GPU memory goes

The runtime now prints a memory ledger at startup: each GPU's used memory (`hipMemGetInfo`, the
driver's view, so allocator rounding and contexts are counted) after each phase of the
constructor. `results/gpu-memory/` holds it at `--max-ctx` 4096, 16384 and 65536 (the
recommended command, `-n 1`); a chart of it: a private dashboard ("GPU memory ledger"). In GB:

| phase | GPU 0 | GPU 1-3, each |
|---|---|---|
| context | 0.22 | 0.07 |
| dense fp8 weights | 6.85 | |
| small weights (embeddings, output head 1.3, norms) | 1.90 | |
| activations, scratch (512-token chunks) | 0.80 | |
| sequence state at 4k / 16k / 64k | 0.04 / 0.12 / 0.43 | |
| draft (DSpark: 7.2 its 384 mxfp4 experts, ~0.5 fp8 dense, ~0.3 the rest) | **8.02** | |
| expert path buffers | 0.13 | 0.32 |
| expert slots | 15.6 (831 slots at 4k) | 33.2 (1,765) |
| free (the 0.75 GB reserve) | 0.75 | 0.76 |
| total | 34.34 | 34.34 |

- **Experts take 84% of the four GPUs' memory** (115 of 137 GB, 6,126 slots at 4k).
- **Long context is cheap**: 64k instead of 4k costs 0.39 GB, 21 slots (6,126 -> 6,105). The
  compressed caches are sized for 1/4 or 1/128 of the positions; fp8 for them (step 59's
  candidate lever) would win back almost nothing.
- **The draft is the largest item besides the main model's experts**: 8.0 GB on GPU 0. (First
  written here as bf16 weights that fp8 could halve; the loader already repacks its dense
  matrices to fp8 exactly and its experts are mxfp4, README step 61.) 7.2 GB of it are the draft's
  own 3 x 128 routed experts, the size of 384 expert slots. Next come the dense fp8 weights
  (6.85) and the bf16 output head (1.3).

## Step 61: the draft's memory, corrected; nothing cheap to free

Step 60 proposed the draft in fp8. Reading its loader (`hip/dspark.cpp`): its attention,
shared-expert and `fc` matrices are already repacked from bf16 to fp8 at load (exact in 32 x 32
blocks), and its routed experts are mxfp4, as the main model's. Of its 8.0 GB, 7.2 are those 384
experts, all resident. Freeing them would mean computing them elsewhere: on the CPU a draft step
(5 tokens x 3 experts x 3 layers) would add up to ~12 ms to its ~10-12, and an adaptive cache
shared with the main model's experts would keep them anyway (they are used every step). Left as
is. So GPU memory is experts (the main model's and the draft's) plus ~10 GB of dense weights and
buffers on GPU 0.

## Step 62: the output head through LDS

The draft's profile (`DS_DRAFT_PROFILE=1`, cpp-matrix) put its output head first: 3.5 of ~13 ms
per call, the main model's bf16 head (129,280 x 5,120, 1.32 GB) read for the draft's 5 tokens at
~375 GB/s. The same kernel computes a verify step's logits. Expert parallelism for the draft was
weighed and set aside: its experts are 2.1 ms of the 13, a cross-die exchange per layer would
take back half, and moving them off GPU 0 frees no memory in total.

`bench-kernels --bitwise` (`results/head-lds.txt`) showed why the head was slow from 4 inputs:
each wave of the multi-input GEMV re-read every input's activations from L2 for its 2 rows.
`gemv_head_m_kernel` (bf16, >= 16,384 rows, one K chunk): a block of 8 waves (16 at 8 inputs)
loads each 512-element step of all inputs' activations into LDS once; each lane then applies the
same `dotw` to the same k in the same order, so the logits are bit-identical (28 of 28 cases
against the single-input kernel). Head time at 2-8 inputs: 1838 / 1964 / 2882 / **3340** / **4035**
/ 5589 / 7701 -> 1732 / 1836 / 1969 / **2231** / **2467** / 2718 / 3119 us (5 = the draft, 6 = a code
verify step).

Held-out generation (`results/spec-measure-sclk6-cacheloads800cacheperiod8-head/`): mean
**32.03 -> 32.48 t/s (+1.4%)**, code and structured prompts +1.7 to +2.1%, outputs identical. The
draft's Markov head stays token by token: each token's bias needs the previous argmax.

TIMING.md has a section on the cache's background loads: what one costs (~2 ms of one GPU's copy
engine, 1,437 loads/s over the four), how many the cache starts (310-465 loads/s, ~11-12 per
token in this step's runs, from their `expert cache:` lines), and what they cost generation
(steps 30, 59).


## Step 63: the next layers' experts from this layer's router input

Each layer changes the hidden state a little, so layer L's router input passed through layer
L+j's router may already say which experts L+j will choose. `DS_ROUTE_AHEAD=K` (a study mode,
generation only) runs, at each layer L, the routers of layers L..L+K on L's router input (the
same top-6 rule with each layer's bias) and at the step's end compares the guesses with what
each layer chose, including the experts it had to compute on the CPU (residency as it was when
the layer routed). `results/route-ahead/route-ahead.sh`, the 8 held-out prompts, `results/route-ahead/`
(`pooled.txt`; outputs identical):

| j layers ahead | chosen experts in the guess's top 6 / 12 / 24 | CPU experts among the guess's top 1 / 3 / 6 not on a GPU | the cache's top 1 / 3 / 6 by use (step 52's guess) |
|---|---|---|---|
| 0 (check) | 100 / 100 / 100% | 69.3 / 98.3 / 100% | 9.9 / 19.8 / 27.6% |
| 1 | 71.6 / 86.0 / 92.1% | **45.0 / 71.3 / 81.6%** | 10.0 / 20.0 / 27.8% |
| 2 | 64.7 / 79.4 / 87.5% | **38.4 / 62.0 / 73.2%** | 10.1 / 20.2 / 28.1% |
| 3 | 59.5 / 74.0 / 83.4% | 33.4 / 54.6 / 66.0% | 10.2 / 20.3 / 28.2% |
| 4 | 56.3 / 70.4 / 80.4% | 30.9 / 50.4 / 61.3% | 10.2 / 20.4 / 28.3% |

(604,320 chosen experts, 64,441 of them on the CPU, over 816 steps.) The approximation holds for
this model and fades slowly: one layer ahead 72% of the chosen experts are in the guess's top 6,
four layers ahead still 56%. For the CPU's experts, the ones a prefetch would need, three
guesses one layer ahead catch 71% and two layers ahead 62%, against 20% for the cache's counts.
(A first run on two prompts read the residency at the step's end, after the cache had already
published that step's loads; the numbers above take it at routing time.)

What it could serve: a prefetch to the CPU's caches one layer ahead (top-1 right 45% of the
time; an expert nearly fills the L3, step 52) or loads to a GPU two or more layers ahead (a load
takes ~2 ms, a layer 1.6-2.5 ms) within the links' budget: step 64 simulates the second.

## Step 64: prefetching the CPU's experts to the GPUs from the lookahead, simulated

A load takes ~2 ms of one GPU's link (~2.9 with the cache's own loads), a layer 1.6-2.5 ms: a
guess must be acted on two or more layers ahead. `results/route-ahead/prefetch_sim.py` replays the 8
prompts' logs (`DS_ROUTE_LOG`, step 63): per step and target layer, a policy picks up to B
experts from the guess made j layers earlier, a load goes to the least busy of 4 links and is
issued only if it can land before the layer routes (the logged routing times, scaled to the
runs without the study), a landed expert takes itself off that layer's CPU time (0.10 + 0.27 ms
per expert, at least 0.25 ms of GPU work), and each load costs 0.036 ms of the step (measured,
step 59). `results/route-ahead/prefetch-sim.txt`; savings against the untraced generation time:

| policy | best setting | loads per step | CPU experts covered | time saved |
|---|---|---|---|---|
| perfect knowledge (the bound) | 3 / 4 layers ahead | 42 / 49 | 62 / 72% | **9.0 / 10.0%** |
| votes: tokens whose top 6 hold it, not on a GPU | 3 ahead, 4 per layer | 50 | 24% | **2.5%** |
| score + use counts (lambda 0.05-0.2), not on a GPU | 3 ahead, 2 per layer | 48 | 23% | 2.4% |
| score, not on a GPU | 3 ahead, 2 per layer | 48 | 22% | 2.2% |
| score, a resident pick wasted | 4 ahead, 8 per layer | 38 | 22% | 2.7% |

One layer ahead is too late (a first run that queued every load landed 1-6% of them); two
layers ahead the links fit ~24 timely loads per step and the bound is 5.4%; three to four give
the time, but those guesses are weaker (half of the top-3 non-resident ones wrong), and wrong
loads take link time from right ones, so 4-8 picks per layer do worse than 2. The policies land
within 0.3% of each other; use counts barely reorder the guesses (the experts not on a GPU are
the rarely used ones, with similar low counts). The simulation is optimistic (free transient
slots; the logged layer times are those without prefetching, and faster layers shorten the
windows), so a real gain would likely be 1-2%, for a scheduler sharing the links with the
cache, transient slots and per-layer timing. Not built.

## Step 65: two quick checks for decode: the fp8 projections and the host

After step 62 a 6-token code step was attention 39% (the core 15%), the other dense parts 24%,
the CPU's experts 22%, the draft ~9%. Two cheap checks before the larger items
(`results/decode-next/`):

**The fp8 projections at several inputs** (`fp8-multi.txt`): from 3 inputs the 64-lane-per-row
tensors jump (attention output A 58 -> 101 -> 122 us at 1 / 3 / 6, B 85 -> 150 -> 155, the
shared gate 20 -> 36 -> 45, engram 206 -> 315 -> 462): each lane group decodes every input's fp8
codes again, as the shared-memory variant (used for 8-16 lanes) avoids, but a 4-wave block of
64-lane rows covers too few rows to pay for it. Bit-identical options: rows per lane group (R)
by M (3-4 inputs: -10 to -25% on the large tensors; 6: today's R = 2 is best), the shared-memory
kernel in 8-wave blocks (better at 3-4, worse at 6: `fp8-lds64.patch`, not kept), the prompt's
many-input kernel (slower at 2-6). Other lane splits are faster but fix a different order of
additions, changing every output. So ~1% on chat-sized steps, nothing at 6; not done.

**Does the GPU wait for the host?** (`host-lag.txt`; `--trace` now prints, for generation, the
lag between the host issuing each group and GPU 0 starting it). Most groups start 100 us to
1 ms after their issue: the host is ahead. Starts within 30 us of the issue (the GPU had run
dry) are almost all the step after each layer's wait for the CPU's experts, which the CPU time
already counts. The rest, an upper bound: 4.7% on chat, 2.9% on code, mostly before `engram`
(the host computes the engram rows per layer, ~2%) and `ffn.add` (1-2% on chat). Launch overhead
is not the limit (HIP graphs would not help); the host's own work is, and little of it.

The batched attention core for verify chunks is the large item left (~5-8% on code steps).

## Step 66: the batched attention core, stage 1: a ring of window + chunk

A verify chunk runs each layer's attention token by token (`attn_core`: rope, the KV into the
window ring, the compressor and indexer, the attention, the inverse rope), so a 6-token step
pays 6 launches of an attention kernel whose ~160 blocks hardly fill the GPU (39-94 us each,
17.7 ms per step in `attn.core`). The plan (PLAN, "batched attention core"): (1) a larger ring,
(2) the per-token work in phases with the small kernels batched, (3) one attention launch per
chunk, (4) the indexer batched.

The obstacle to batching is the ring: 128 slots, so a chunk's later tokens overwrite positions
its earlier tokens still need. It is now window + `kChunk` = 640 slots per layer (0.07 GB more),
position p at slot p mod 640. The attention kernel still takes window key j as the position
pos - ((pos - j) mod 128), so every token sees the same keys in the same order, now at slot
ps mod 640. Verify's ring backup and restore are gone: a rejected draft's slot is written again
by the token that takes its position before any window reads it, and what it overwrites
(positions 640 back) no window reads.

`results/attn-batch/stage1.txt`: the prompt gate passes (`--canon` chunks of 512 against the
token-by-token baseline, fast 1 vs 512, 1,200 and 6,000 bytes); a 6,106-token prompt with
`--canon` gives the same bytes as a build of the previous commit (fast mode differs: GPU 0
holds 4 fewer expert slots, and fast numerics follow the placement, step 31); the 8 held-out
prompts are identical to step 32 at a mean 32.58 t/s (32.48 before: the backup copies, ~0.3%).

## Step 67: stage 2, the attention in phases

`attention()` now runs a chunk's attention in phases: (A) every token's rope of q and its window
KV into the ring, one launch each for the chunk (`k_rope_m`, `k_kv_finish_m`, the same
per-element arithmetic with the token as the grid's second dimension); (B) token by token the
compressor and the indexer (`attn_prep`: a compressor group completed mid-chunk makes a
compressed position only the later tokens count; each token's key count and top-k are its own);
(C) each token's attention; (D) the inverse rope, one launch. The stage-1 ring makes (A) safe.
`results/attn-batch/stage2.txt`: the prompt gate passes; a 6,106-token prompt is identical to the
stage-1 build in both numerics; the 8 held-out prompts are identical to step 32 at a mean **33.25
t/s** (32.58).

## Step 68: stage 3, one attention launch per chunk

The attention kernel takes the token as its grid's third dimension: up to 8 tokens per launch
(`k_attn_decode_m`; a struct by value carries each token's compressed-key count and top-k
list), each token's blocks exactly a one-token launch's (the same 32-key chunks in the same
order; blocks past a token's last chunk exit), the combine likewise per token; a prompt chunk
runs in groups of 8 (the scratch: 8 x 2.6 MB). The one-token call is the same kernel with one
token. `DS_ATTN_CHECK=1` also runs every token alone and compares: 0 of 14,400 tokens differ in a
`--spec` run (2,880 launches) and 0 of 245,480 in a 6,106-token `--canon` prompt
(`results/attn-batch/stage3.txt`).

| | before stage 1 | stage 3 |
|---|---|---|
| held-out mean, sclk 6 | 32.48 t/s | **34.33 t/s (+5.7%)** (cpp-matrix 37.6 -> 41.4) |
| a 6-token verify step (`--profile`) | 108.2 ms | 96.3 ms |
| its attention core | 17.7 ms | 7.8 ms |
| 1,989-token prompt, fast / `--canon` | ~83.5 / 59.5 t/s | 93.3 / 63.0 t/s |
| 6,106-token prompt, fast / `--canon` | 87.2 / 68.2 t/s | 96.7 / 73.2 t/s |

All outputs identical (prompt gate, the 8 held-out prompts; the long prompt identical with
`--canon`, fast mode differing by one expert slot fewer on GPU 0). Stage 4 (the indexer batched)
matters only past ~2k context and stays on the list. The CPU's experts (25.9 ms of the 96.3) are
now the largest item of a code step.

## Step 69: traces of the batched attention

`results/trace/` regenerated with the build of step 68 (same commands, sclk 6; outputs identical;
28.84 / 38.83 t/s traced against 30.71 / 41.37 untraced, the same ~6% cost as in step 58).
`scripts/trace_compare.py` lists two traces' spans in generation per verify step
(`results/trace/compare-step58.txt`):

| ms per verify step | cpp-matrix, step 58 | step 69 | tcp-udp, step 58 | step 69 |
|---|---|---|---|---|
| all layers (die 0) | 105.8 | 100.1 | 64.9 | 63.0 |
| attention core | 14.8 | **6.0** | 7.0 | **3.8** |
| CPU experts (host) | 40.3 | 41.5 | 18.9 | 19.4 |
| die 0 waiting for the CPU | 36.6 | 38.0 | 17.4 | 18.0 |
| draft | 12.4 | 11.3 | 9.9 | 9.0 |
| head | 3.3 | 2.2 | 2.1 | 1.8 |

The attention core is the only span that moved by more than ~1 ms (the head is step 62's); the
CPU's experts and die 0's wait for them are now ~40% of a code step's layer time. In Perfetto a
verify layer's attention is one short `attn.core` block instead of one per token.

## Step 70: decode ideas revisited; verify steps in two halves (tried, slower)

Every decode idea so far (PLAN's lists and the dropped ones) re-scored against step 69's traces.
A code verify layer (6 tokens, medians per layer) is now die 0's attention, router and dispatch
(0.95 ms), then the CPU's experts (0.95 ms; the shared expert and the dies' experts finish inside
it), with handoffs between: 0.08 ms from the dispatch to the CPU's start and 0.13 ms from the
CPU's end to the sum (9.0 ms per code step, 4.7 per chat step). The GPU idles during the CPU's
part, the CPU during attention, dies 1-3 most of the layer. What moved up: those handoffs (new;
local), attention split over the XGMI pair 0-1 (~36 ms of a code step; exact if split by heads
and output rows with all-gathers), and a CPU-cache prefetch during attention from step 63's
lookahead guesses (step 52 dropped it at 20-28% right; the lookahead finds 71% of the CPU's
experts in 3 guesses; the caches hold ~2 experts: ~2-3%). The rest is unchanged: the CPU side
and the dies' experts did not change since they were dropped.

One idea not tried before: the prompt's CPU/GPU overlap (step 38: the chunk in halves A and B, the
host computing A's CPU experts while the GPU runs B's attention) was kept out of the verify
forward only by its condition. Let in (`results/decode-next/verify-halves.patch`, an env flag):

| t/s | halves | without (step 68) |
|---|---|---|
| tcp-udp | 26.73 | 30.71 (-13%) |
| cpp-matrix | 37.18 | 41.37 (-10%) |

Outputs identical, slower on both. The likely reason (not measured): one host thread both
computes the CPU's experts and issues all GPU work, and the halves double both the launches and
the CPU's per-call fixed costs, which a 512-token prompt chunk amortizes and a 6-token verify
does not. It would need a separate launching thread. Not kept.

## Step 71: the handoffs around the CPU's experts

Step 70's trace showed ~0.2 ms per layer between the GPU's dispatch and the CPU's experts and
between their end and the sum (9.0 ms per code step). Moving the dies' result copies to a second
stream that starts them as each die finishes shortened the tail (126 -> 50 us per layer) and
gained nothing: the HIP calls issuing them, ~10-15 us each across devices, now ran before the
CPU's experts (a temporary host timer: 189 us from the staging event to the CPU's start).

The fix: in a short forward (up to 8 tokens, not a prompt half), `moe_launch` issues each die's
copy back to the host and on into the home die (a second stream, one event the sum waits for)
while the host is still ahead of the GPU, copying a bound of rows (tokens x 6 in canon) instead
of the counts the host learns only after staging; the rows past the count are never read. The
prompt's chunks keep the old path. Now the CPU's experts start ~5 us after staging, and the sum
follows them by ~60 us instead of ~125 (`results/decode-next/handoff.txt`):

| | step 68 | step 71 |
|---|---|---|
| held-out mean, sclk 6 | 34.33 t/s | **34.98 t/s (+1.9%)**, every prompt faster (chat +2.3-3.6%, code +0.4-1.3%) |
| a code layer with CPU experts (trace median) | 2,326 us | 2,214 us |
| generation after the 6,106-token prompt, fast / `--canon` | 19.4 / 20.0 t/s | 21.5 / 21.5 t/s |

Outputs identical (prompt gate, the 8 held-out prompts, the long prompt in both modes against
step 68). Chat gains most since its steps are shortest; on code the draft-length rule took
slightly shorter drafts on some steps (sql-report: 3.90 tokens per step vs 4.00), which hides
part of a ~2-3% faster step.

## Step 72: the held-out history: every decode change on the same 8 prompts

One point per decode change: commit, a one-line summary, and generation speed on the 8 held-out
prompts (300 tokens, sclk 6), mean and per prompt (`results/heldout-history/`: `points.tsv` the
plan, `history.tsv` the numbers, `history.html` the chart, all from `results/heldout-history/heldout_history.py`).
Steps 48-71 were measured then at sclk 6 (27.96 -> 34.98 t/s); steps 6-45 were measured at other
clocks or settings, so `results/heldout-history/heldout-rerun.sh` rebuilds their commits in worktrees (all 16
build against today's tree of dependencies) and reruns them at sclk 6 with each commit's own
flags: 13 points in tier 1 (~6-7 h), 3 optional, and steps 48-71 again so every point shares one
night's conditions (~12 h in all). The chart's per-prompt grid shows each step's
change from the previous point: a column of one color is a change every prompt feels; a column
split by row is speculation-specific (the top rows accept the most drafts per step).

**The run (2026-11-03, 14:10-19:00, ~4 h 50 min instead of the ~12 h estimated):** all 26 points
complete (25 run, step 67 from its source), no failures. Mean t/s on the 8 prompts:

| step | 6 | 7 | 14 | 16 | 17 | 18 | 20 | 21a | 21b | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t/s | 10.06 | 12.32 | 18.08 | 19.30 | 19.43 | 19.36 | 11.44 | 12.31 | 15.23 | 16.55 | 17.81 | 18.61 | 22.92 |

| step | 31 | 32 | 45 | 48 | 50 | 53 | 57 | 62 | 66 | 67 | 68 | 71 | 73 | 74 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t/s | 20.70 | 29.02 | 27.55 | 28.07 | 30.42 | 31.73 | 32.03 | 32.61 | 32.80 | 33.25 | 34.46 | 34.80 | 35.08 | 35.94 |

- **Three dips, each recovered:** step 20 (speculation verified token by token, by design: a
  step paid for rejected tokens and a 26 ms draft; the chat prompts, which accept fewest drafts,
  fell most, ja-email 16.7 -> 6.9) recovered through 21a-25 to 22.9, above step 18; step 31
  (`--canon`, the GPU's experts computing the CPU's bits) cost 10% and bought exact outputs
  whatever the placement, repaid by the cache in step 32; step 45 (-O3) sits on top of the
  extra engram copy per token since step 37, found and fixed in step 48.
- **Repeatable:** steps 48-74 rerun agree with their measurements of the time within 0.2 t/s
  (71: 34.98 then, 34.80 now; 74: 35.94 both).
- **Outputs:** every point from 31 on gives the reference output on all 8 prompts; points 6-25
  differ from it by design (the fast numerics of their time).
- `results/heldout-history/heldout-status.sh` (added after the start): points complete, the current point and
  its prompts, the age of the last write (`STALLED?` after 15 min), the means so far; run it
  under `watch -n 60`.

## Step 73: the engram fetch off the critical path; traces regenerated

At the start of every forward the host fetched the engram rows of layers 1 and 14 (hash the
last 2-4 tokens, read 24 rows per layer and token from the 189 GB table, convert, copy) while
die 0 waited with nothing queued: ~2.0 ms per code verify step in the step 69 trace. Three
changes (`results/decode-next/engram.txt`):

- **Fetched after layer 0 is queued** (`block_post` of layer 0, while die 0 runs its front);
  the first engram layer is 1. Alone this gained ~0.3 ms: the fetch (~1.4-1.7 ms) outran
  layer 0's front and delayed its CPU experts.
- **Read with pread** (`Model::file_of`): through the mapping nearly every row was a minor page
  fault (~200 per 5-token fetch, ~7 us each; no major faults, hashing 3 us). pread copies from
  the page cache without mapping: 1428 -> 1036 us per 5-token fetch, 733 -> 491 us at 2.5.
- **Copied on its own stream** (`s_engram`; layer 1's engram waits for its event): on die 0's
  stream the copy sat between layer 0's shared expert and its sum.

A 6-token verify step now reaches layer 1 after 3.5 ms instead of 4.5. Held-out
(`spec-measure-sclk6-cacheloads800cacheperiod8-engram/`): mean **34.98 -> 35.09 t/s**; per step
0.4-2.3% faster on 8 of 9 prompts (summary-json +0.3%), the t/s gain partly hidden by the
draft-length rule's step counts. Outputs identical (prompt gate, the 8 prompts, the 6,106-token
prompt in both modes against step 71). Prompt speed unchanged within run-to-run noise.

Huge pages for the table (asked): the reads scatter over ~49M 4 KB pages (~94k 2 MB pages), a
300-token run touches ~14k rows, so most reads would still be first touches; pread avoids the
mapping entirely. Batched reads (io_uring) would matter only for long prompts (~24k rows per
512-token chunk).

The traces in `results/trace/` are regenerated with this build (29.33 / 38.20 t/s traced, outputs
identical; `compare-step69.txt` against step 69's): on a code verify step die 0's
`engram.fetch` (2.00 ms) is gone, the host's `engram fetch` (1.10 ms) runs beside layer 0, and
die 0's wait for the other dies' results is 4.72 -> 2.15 ms (step 71).


## Step 74: the indexer skipped while it picks every compressed position

The indexer (layers 2, 8, 14, 20, 24, 28, 32, 36) scores every compressed key for each token and
keeps the best 512 for that token's attention. With 512 keys or fewer it keeps all of them, and
the runtime then used `0 .. n-1` (iota) anyway: the projections, scores and top-k were computed
and dropped. That is every held-out prompt (at most ~400 positions: ~200 keys on layers 2-19,
~400 on 20-39), at ~76 us per token and source layer, one token at a time (~3.4 ms of a 6-token
code verify step in step 73's trace). Now a source layer with every key picked and no candidate
blocks (those start beyond 16,384 keys) sets iota without running the indexer: the same picks by
construction. Past 512 keys (~512 tokens on layers 20-39, ~1,024 on 2-19) it runs as before.

Held-out (`spec-measure-sclk6-cacheloads800cacheperiod8-indexer/`): mean **35.09 -> 35.94 t/s
(+2.4%)**, every prompt faster (`results/decode-next/indexer.txt`). Outputs identical (prompt
gate, the 8 prompts, the 6,106-token prompt in both modes against step 73, where the indexer
does pick). Prompt and long-context speed unchanged. Batching the indexer over a step's tokens,
which is what would help long context, is in PLAN (Further ideas, F).

## Step 75: traces after step 74; the fp8 GEMV for 2-8 inputs (tried, not kept)

New traces (`results/trace/`, `compare-step73.txt`) put a 6-token code verify step at ~110 ms with
the draft: die 0's dense part before the CPU starts ~41 ms, the CPU's experts ~42 ms (3.5 per
layer, ~314 us each), the draft ~10 ms, handoffs and the head ~13 ms; a 2-token chat step ~62 ms
in the same shape. The die and the CPU take turns; dies 1-3 are busy ~20% of a step.

The largest dense piece is attn.out (wo_a + wo_b, ~0.30 ms per layer at 6 tokens, 12.1 ms per
step): at M = 6 the kernel reads weights at ~270 GB/s against ~580 at M = 1. Step 65 had measured
this kernel already (rows per lane group, the shared-memory kernel in 8- and 16-wave blocks: gains
at 3-4 inputs only), which was missed before starting. Three variants it had not tried, all
bit-identical to the one-input call (`results/decode-next/fp8-multi-step75.txt`, with step 65's
shared-memory variant rerun in 4-wave blocks): inputs pre-decoded to fp16 by the quantizer
(slower: decoding is not the limit), the weight rows first and one input at a time (half the
registers, 3 waves per SIMD instead of 2, yet ~2x slower at M = 6: each input's load is waited for
just before its use), and that with the loads first and a register cap (spills). Reverted. A gain
needs a different work split (more rows per wave without holding M decoded inputs), a
restructured kernel rather than a local change; the CPU side is next.

## Step 76: attention's output over the XGMI pair 0-1 (tried, not kept)

Step 70's next lever after the handoffs: the attention work on die 0 (attn.out ~12 ms, core ~6,
q ~6 per code step) split over dies 0 and 1, which share an XGMI link. First the hand-off itself
(`results/decode-next/attn-split.txt`, a `bench-xdie` ping-pong on 2-6 tokens of activations): an
event between the dies costs the host ~15-20 us per hand-off; a kernel spinning on a flag in
fine-grained memory, raised by the other die's kernel, costs the GPU ~6-8 us and the host ~3.

Built first: the output projections only, which need no state on die 1. With `--attn-split`
(M <= 8) die 1 holds wo_a's groups 4-7 and wo_b's rows 2560-5119 (1.41 GiB); die 0 sends o's codes
for heads 32-63, each die runs its half of wo_a, they swap their halves of oa's codes, each runs
its half of wo_b, and die 1's rows come back. Hand-offs are put/get kernels on mailboxes and flags,
no events and no host. The GEMV took row strides and the whole matrix's lanes per row, so every
row keeps the one-die bits: outputs identical on every run, and bit-identical at M = 1-8 in the
bench.

- **The split's fixed cost is ~55 us per layer** (wo_a + wo_b over 40 layers on the real shapes:
  M = 6 268 -> 190 us where halving would give ~134; M = 2 173 -> 146): four more kernels on die 0
  and two waits for die 1. The host is not the limit (24 us of submits per layer).
- **The weight halves came out of die 1's expert cache** (die 0 keeps the whole matrices for the
  prompt): 1.41 GiB fewer slots, 4.4% more CPU expert pairs, CPU experts +1.5 ms per code step,
  against attn.out -2.1 ms.
- **End to end +0.5-0.7%** (cpp-matrix 43.30 -> 43.57 t/s, tcp-udp 32.14 -> 32.35, two
  alternating runs each).

Not kept; the change is `results/decode-next/attn-split.patch` (applies to 4b1a52c). It would pay
with the halves dropped from die 0 (the prompt's fp8_gemm split too; est. +1.5-2%) or with the
attention core and q_b split by heads as well, so die 1 makes its own heads' o and the largest
transfer goes (needs the KV caches, the indexer's picks and speculation's rollback on die 1; est.
+4-6%).

## Step 77: would a prefetch to the CPU's caches pay? The kernel's ceiling from cache

A larger model will keep experts on the CPU however many GPUs it gets, so the question was
whether a CPU with a large last-level cache (an EPYC with 768 MB) could prefetch the guessed
experts a few layers ahead (step 63: the top 6 guesses two layers ahead catch 73% of the CPU's
experts) while the CPU idles during attention. The ceiling of any such prefetch is the kernel's
speed on an expert that is already in cache. `bench-cpu-moe --warm` runs the same call on fresh
experts every time (from DRAM) and on the same experts every time (from L2 + L3: one expert,
18.8 MB, fits in the 16 + 22 MB here), with the read-only control
(`results/cpu-decode/warm.txt`, 16 threads, ms per expert, one token per call):

| D = 1 | from DRAM | from cache | |
|---|---|---|---|
| canonical kernel | 0.39-0.43 | 0.34-0.35 | x1.2 |
| fast kernel | 0.41-0.47 | 0.30-0.40 | x1.2-1.4 |
| read only | 0.19-0.24 | 0.03-0.07 | x3-6 |

(The ranges: no gap and a 1.5 ms gap between calls. Two or six experts per call do not fit, and
their warm runs gain only 1.04-1.2x.) **The call is not bound by DRAM:** from cache it still
processes only ~55 GB/s of expert bytes, and the canonical (exact integer) kernel and the fast
one behave alike. Step 56 read the gap between the kernel and a plain read (~35-40%) as lost
memory efficiency; it is not that either. So a large cache or HBM pays only once the call itself
is faster; step 78 takes the call apart.

## Step 78: where a one-token CPU call goes; per-thread activation copies (tried, not kept)

Step 77 left a warm call (one expert in cache) at 0.34 ms. Taken apart with two new benchmarks,
on 16 threads, ms per expert:

- **The kernel's compute alone** (`bin/bench-fp4-decode`: each thread its own 64 rows in its L2,
  repeated; `results/cpu-decode/decode-bench.txt`): canonical ~0.15-0.17, fast ~0.11; of the
  canonical one, the fp4 decode ~0.045-0.065 and the integer arithmetic (3 vpdpbusd per 64
  weights) ~0.07-0.09. A CPU layout (a row's codes contiguous, 128 per 64-byte load) decodes in
  ~0.026-0.03. Gate rows (K 5120) and down rows (K 2304) cost the same per weight.
- **The call's two phases as a bare loop** (`bin/bench-fp4-decode call`, and the same loop on a
  real expert in `bench-cpu-moe --warm`): 32-row tasks of `canon_dot_rows`, gate + up then down,
  activations written once: warm 0.19-0.20, cold 0.29-0.30 (`call.txt`, `warm.txt`). 4 KB or
  2 MB pages: no difference. A static schedule: warm 0.17-0.22, cold 0.22-0.25.
- **`forward_batch` on the same expert: 0.34.** The difference is reproduced by the bare loop
  rewriting its activation digits in parallel every call, as `forward_batch` does: 0.19 -> 0.33.
  Down, which excludes the rewrite, 0.066 -> 0.12. Reading digits the whole team has just written
  (~15 KB for gate + up, ~7 KB for down) costs ~45% of a warm call. The writers flushing them:
  no change; the readers prefetching them: no change; each thread copying them into its own
  buffer and reading that: 0.33 -> 0.245.

The copies went into `forward_batch` and `forward_canon_rows` for calls of up to 8 tokens (the
same bits: `bench-cpu-moe --batch 3`, 0 of 40 layers differ). Back to back the call went 0.334 ->
0.256 warm, 0.395 -> 0.359 cold; but after a 1.5 ms idle gap, the runtime's case (the GPU's part
of a layer), the copy cost what it saved: cold 0.433 -> 0.431 with one expert, 0.321 -> 0.358
with two. End to end, alternating, outputs identical (`private-copies-ab.txt`): cpp-matrix 43.13
-> 43.05 t/s, tcp-udp 32.19 -> 32.21. Not kept (`results/cpu-decode/private-copies.patch`).

**After an idle gap** (`bench-fp4-decode call 16 2m 1500`, `call-gap.txt`), the runtime's case:
libgomp's workers spin through 1.5 ms by default, so the last one enters a call 2-4 us after it
starts; with `OMP_WAIT_POLICY=passive` they sleep and arrive 0.13 ms late (a call 0.64 ms). With
the dynamic schedule a gap costs ~0.01 ms warm and ~0.02-0.03 cold (0.30 -> 0.32-0.33), about the
same whether the master sleeps or spins through it; the static schedule, which waits for its
slowest thread, loses 0.1-0.2 ms. So the gap is worth 5-10% of a call, not the 2x.

Every thread computing all of the input's digits itself (no shared, freshly written lines for
gate + up) costs what it saves: its ~0.09 ms of single-thread work replaces the penalty. Why
reading digits the team has just written is slow, when a thread's copy of them is fast, is
open; hardware counters (snoops, L2 misses) would answer it rather than more variants.

## Step 79: why freshly written activations read slowly (hardware counters); aligned chunks (tried, not kept)

Step 78 left open why the canonical call's activation digits, written by the 16 threads, are slow
for the same threads to read. `bench-fp4-decode call 16 2m 0 <variant> N` runs only the warm call
(dynamic schedule, no gap) with the activations written once, rewritten every call by the team
(as `forward_batch`), rewritten then copied per thread, rewritten by one thread, or rewritten by
the team in 16-block chunks of 64-byte aligned arrays (each thread whole cache lines of its own).
Counted with perf (`results/cpu-decode/counters.txt`, user space, 2000 calls minus 0, all threads),
per call:

| | once | rewrite | copy | one writer | aligned 16-block |
|---|---|---|---|---|---|
| ms per call | 0.179 | 0.323 | 0.238 | 0.321 (its writing alone ~0.14) | 0.276 |
| down phase (excludes the rewrite), ms | 0.064 | 0.113 | 0.061 | **0.063** | 0.083 |
| cycles | 8.6M | 16.1M | 11.3M | 15.7M | 13.7M |
| loads missing L2 | 3.8K | 17.8K | 7.5K | **4.4K** | 13.3K |
| of them from another core's modified line (HITM) | 163 | 2,916 | 1,176 | 286 | 1,693 |
| memory-ordering machine clears | 38 | 6,177 | 1,964 | **136** | 3,976 |
| cycles stalled on an L2 miss | 0.31M | 4.53M | 2.15M | **0.46M** | 2.77M |

4K aliasing, DRAM misses and prefetches do not change. **The penalty comes from the digits being
written by many threads, not from their being freshly written:** written by one thread they read
as if never rewritten. Whole lines per thread remove ~40% of it (false sharing at the 8-block
chunk edges: `corr` and `c` are 32 bytes per chunk, the digit arrays only 16-byte aligned); the
rest goes with lines spread over 16 writers' caches. One writer is not a fix as it stands:
`canon_act_blocks` is scalar, ~0.14 ms for the 5,120 input values on one thread.

The aligned 16-block chunks went into `forward_batch` and `forward_canon_rows` (the same bits:
`bench-cpu-moe --batch 3`, 0 of 40 layers differ): per call after the runtime's gap 0.433 ->
0.413 ms with one expert, 0.321 -> 0.306 with two, none with six. End to end, alternating
(`aligned16-ab.txt`): cpp-matrix 43.32 -> 43.43, tcp-udp 32.17 -> 32.47 (every run above every
base run); the 8 held-out prompts (`spec-measure-sclk6-cacheloads800cacheperiod8-aligned16/`):
mean **35.94 -> 36.00 t/s (+0.2%)**, chat-like prompts +0.5-1.1%, code flat; outputs identical.
Below the +0.5% kept for a change; not kept (`results/cpu-decode/aligned16.patch`). Next: a fast
(AVX-512, bit-exact) `canon_act_blocks`, so that one thread, or a few, can write the digits.

## Step 80: the canonical activation digits on one thread each (AVX-512)

Step 79 showed the penalty goes away when one thread writes the digits, but the scalar
`canon_act_blocks` took ~0.14 ms for one token's input on one thread. It is now AVX-512, with the
scalar version kept as the specification (`canon_act_blocks_ref`): per block of 32, the absolute
maximum, the scale's next power of two from its exponent bits, `x * (1 / s)` (the same rounding as
`x / s` for a power of two), the e4m3 rounding as a step 2^(binade - 3) from the exponent bits and
`roundscale` to nearest even, and the digits by masked integer ops
(`results/cpu-decode/act-single.txt`):

- **The same bytes:** `bench-fp4-decode act`, 2,097,152 blocks of mixed inputs (zeros,
  subnormals, amax below 1e-4 and above 448, ties, exact powers of two): 0 differing bytes in
  d0, d1, d2, x9, corr and c.
- **~12.5x faster on one thread:** gate's input (5,120 values) 87 -> 7.0 us, down's (2,304) 38.5
  -> 3.0 us.

`forward_batch` and `forward_canon_rows` now give each token's input digits and each pair's
activation digits one writer (a task per token / pair); the SwiGLU stays parallel, in its
8-block tasks, writing floats that the one writer reads once. The same bits (`bench-cpu-moe
--batch 1 / 3 / 8`: 0 of 40 layers differ; the canonical rows against the scalar specification:
0 differ). Per call (`bench-cpu-moe --warm`, against `warm.txt` part 1), cold after the runtime's
1.5 ms gap: one expert 0.405 -> 0.383 ms (-5%), two 0.341 -> 0.302 (-11%), six 0.289 -> 0.265
(-8%); warm -6 to -14%; back to back up to -33%. In the bench the warm call is 0.355 -> 0.201 ms
with no gap and 0.341 -> 0.239 after one (written once: 0.189 / 0.217).

End to end, alternating with step 79's base (`act-single-ab.txt`, outputs identical): cpp-matrix
42.92 -> 43.65 t/s, tcp-udp 32.14 -> 32.53 (every run above every base run). Held-out
(`spec-measure-sclk6-cacheloads800cacheperiod8-actsingle/`): mean **35.94 -> 36.31 t/s (+1.0%)**,
29.6-44.0; outputs identical.

| prompt | step 74 | step 80 |
|---|---|---|
| tcp-udp | 32.26 | 32.25 |
| py-lru | 37.21 | 38.20 |
| cpp-matrix | 43.14 | 43.41 |
| sql-report | 36.88 | 37.60 |
| geometry | 43.69 | 43.95 |
| zh-travel | 29.44 | 29.58 |
| ja-email | 31.61 | 32.31 |
| summary-json | 33.26 | 33.14 |
| long-proof | 50.62 | 50.68 |

(tcp-udp and summary-json took fewer tokens per step in this run, 2.14 / 3.63 against 2.19 /
3.77: the draft-length rule, not the CPU.) Left open: after an idle gap the single-threaded
input phase takes ~0.05 ms instead of the bench's 7 us (perhaps the core's first AVX-512 after
idling); the next CPU items are a task scheduler that claims ahead (memory and compute still add
up in a cold call) and a weight layout made for the CPU.

## Step 81: the CPU call after the runtime's idle gap (tried, not kept)

Step 80 left the input phase at ~0.05 ms after the runtime's ~1.5 ms wait for the GPU, against
7 us in the bench. It is not that phase: whichever phase comes first after a gap loses ~0.03-0.04
ms (`results/cpu-decode/after-gap.txt`; in `bench-fp4-decode call` gate + up goes 0.137 -> 0.17
ms with the digits written once or by one thread), while every thread still joins within 2-4 us.

- **The master's core idles.** The host thread is the team's master, and it sleeps through the
  gap: in 1,000 bench calls its core entered C6 1,483 times, the workers' cores (spinning in
  libgomp) almost never. With the master spinning instead, the call after a gap is 0.243 ->
  0.221 ms (no gap: 0.206), ~60% of the loss back.
- **C6 is not the cause:** disabled on every CPU (root), the bench is unchanged (0.242-0.256 ms
  after a gap). Nor is the writer: the digits by a worker whose core spun through the gap gain
  ~0.01 ms (0.233). What a spinning master keeps is its core's state in the shallow idle states
  as well (clock, caches), not C6 alone.
- **The runtime's wait** for the staged block (`hipEventSynchronize`, before each layer's CPU
  experts) made to poll: by `HSA_ENABLE_INTERRUPT=0` it changed nothing (cpp-matrix 43.42 ->
  43.44, tcp-udp 32.40 -> 32.45; C6 entries of a run about the same); as a loop of
  `hipEventQuery` (`DS_SPIN_WAIT=1`, `spin-wait.patch`) cpp-matrix 43.15 -> 43.79 and tcp-udp
  32.50 -> 32.54 in two alternating rounds, but the 8 held-out prompts
  (`spec-measure-sclk6-cacheloads800cacheperiod8-spinwait/`) **36.31 -> 36.14 t/s (-0.5%)**, the
  prompts with the same steps flat (sql-report 37.60 -> 37.51, geometry 43.95 -> 43.93).

Not kept. The loss after the gap (~0.03-0.04 ms per call with CPU experts, ~1-1.5 ms per code
step) stays; a spinning host thread takes a core the CPU's experts share, and the held-out set
does not show a gain. Next on the CPU: the task scheduler that claims ahead (step 80).

## Step 82: the CPU's 32-row tasks handed out in pairs

The next CPU item (step 80): a cold call reads ~0.2 ms of DRAM and computes ~0.15 ms, and the two
nearly add up. The plan was a scheduler that claims its next task ahead and prefetches its
first rows (each 32-row task starts on rows nobody prefetched; the kernel's prefetch looks 16 KB
ahead within its own rows). `bench-fp4-decode sched [gap]`: one expert per call, the digits
written once, cold (24 experts in turn) and warm, ms per call (`results/cpu-decode/sched.txt`):

| schedule | cold, no gap | cold, 1.5 ms gap | warm, 1.5 ms gap |
|---|---|---|---|
| OpenMP dynamic, 1 task (the runtime) | 0.29-0.32 | 0.30-0.32 | 0.21 |
| claim-ahead, no prefetch / 16 KB / 32 KB / the whole next task | 0.28 / 0.30 / 0.33 / 0.35 | | |
| static (contiguous rows per thread) | 0.21-0.22 | 0.36-0.39 | 0.31-0.36 |
| guided | 0.22-0.23 | 0.29-0.30 | 0.26 |
| **dynamic, 2 tasks** | 0.25 | **0.27** | 0.23 |
| dynamic, 3 / 5 tasks | 0.24 / 0.22 | 0.28-0.31 / 0.28-0.29 | 0.25-0.26 |

**Prefetching the next task only costs** (the more, the slower). **Contiguity is what helps:**
a thread walking a long run of rows keeps its prefetchers streaming, and static gets a cold call
to ~90 GB/s, near a plain read. But static waits for its slowest thread, and after the
runtime's idle gap that is the master's core (step 81): the worst of all there. Chunks of two
tasks (64 rows) are the balance; three and more wait on the slow core again.

In the real kernel (`bench-cpu-moe --warm`, chunk 1 vs 2 in `forward_batch`) after the gap: two
experts 0.328 -> 0.29 ms cold (-12%), six 0.28 -> 0.24 (-13%), one 0.35 -> 0.32-0.36; warm -9 to
-17%. The one-token path (`forward_canon_rows`, verify steps of one token) was static: after
the gap one expert 0.35-0.44 -> 0.32, two 0.65-0.84 -> 0.51-0.56, six 1.78-2.20 -> 1.39 (with no
gap static is ~10% better there; the runtime never calls without one). Both now hand out their
weight tasks two at a time (`kTaskPair`); the same bits (`--batch 1 / 3 / 8`: 0 of 40 layers
differ).

Held-out (`spec-measure-sclk6-cacheloads800cacheperiod8-taskpair/`): mean **36.31 -> 37.16 t/s
(+2.3%)**, 30.0-45.1; outputs identical.

| prompt | step 80 | step 82 |
|---|---|---|
| tcp-udp | 32.25 | 33.38 |
| py-lru | 38.20 | 38.71 |
| cpp-matrix | 43.41 | 44.95 |
| sql-report | 37.60 | 38.16 |
| geometry | 43.95 | 45.14 |
| zh-travel | 29.58 | 30.00 |
| ja-email | 32.31 | 32.38 |
| summary-json | 33.14 | 34.58 |
| long-proof | 50.68 | 50.71 |

(Steps per token moved both ways in this run: tcp-udp and summary-json took more tokens per step
than in step 80's, py-lru, cpp-matrix and sql-report fewer; geometry and zh-travel, the same,
gained 2.7% and 1.4%.) Code-like prompts gain most, as the per-call numbers predict: their
layers send 2-6 experts to the CPU. Next on the CPU: a cold call is still ~0.29 ms against
~0.2 ms of DRAM time; the CPU weight layout (step 78's decode at ~half the cost) is the next
candidate.

## Step 83: a CPU weight layout for the experts, benchmarked (not deployed)

The canonical kernel decodes the experts' mxfp4 rows as stored in the GGUF: 17-byte blocks (a
scale byte, 16 code bytes), two blocks per register by two 16-byte loads, broadcasts and an
insert, and the 16 scales of a group by a gather. Step 78 measured that decode at ~0.045-0.065 ms
of ~0.15 ms of compute per expert. `bench-fp4-decode layout` repacks an expert into a CPU layout:
each row's codes contiguous (byte j of a 64-byte group holds element j in its low nibble and j +
64 in its high one, so one load gives the two registers of two blocks each that the kernel
builds today, in the same order) and the scales in their own [N][nb] array; `canon_dot_rows64`
is `canon_dot_rows` with those loads, the same arithmetic and reduction
(`results/cpu-decode/layout.txt`):

- **The same bits:** every row of an expert's gate, up and down: identical.
- **Compute (one core, 64 rows in L2) x1.7-1.85:** 17.7 -> 31-33 G weights/s.
- **A whole one-token call** (16 threads, dynamic chunks of 2, after a 1.5 ms gap): cold 0.282
  -> 0.256 ms (-9%), warm 0.22 -> 0.18 (-18%). A cold call is mostly DRAM time now, so it gains
  less than the compute. The new kernel needs its own prefetch distance: 8 KB ahead (none:
  cold 0.36; 4 KB 0.28; 16 KB 0.275; 32 KB 0.35).

Estimated +1.5-2.5% end to end (CPU experts are nearly always cold). Not deployed: the CPU
reads its experts from the GGUF mmap, and RAM (755 GB, ~488 GB of model files cached) cannot hold
a second copy, so the layout would have to become the only source of expert weights: a one-time
repacked file (~269 GB; 555 GB free), CPU-layout versions of the multi-token canonical kernel,
the fast path's two kernels and the read-only control, a variant of the GPU loader's
`fp4_unpack` that reads it (producing today's GPU layout, checked buffer for buffer), and the
model's experts pointed at the new file. Recorded for later.

## Step 84: traces that split the CPU's handoff; the CPU's rows copied by a kernel

The traces were regenerated with step 82's build and read with a new script,
`scripts/trace_critpath.py`: per verify size, die 0's front, the handoff to the CPU, the CPU's
experts, the handoff back, the tail, the time outside the layers and the draft (medians per
step). It showed a handoff back of ~6.6 ms per 6-token code step (~160 us per layer for a few
hundred KB), not visible as such before. Two additions to `--trace` (no change untraced):

- **The handoff back in parts.** The copy of the CPU's rows to die 0 is its own group on die 0
  (`ffn.cpu_rows_copy`); on the host, issuing it and the profile's bookkeeping are spans. The
  bookkeeping (which ran before the copy was issued, in traced and profiled runs only, and
  counted as CPU time) now runs after it.
- **The CPU call's phases** nested in "CPU experts": `cpu.inputs`, `cpu.gate_up`, `cpu.act`,
  `cpu.down`, the weight phases with the spread between the first and the last thread to finish
  (`CpuMoeKernel::last_call`). Per layer: digits ~17 us, activation ~12-20 us, threads within
  ~15-30 us of each other, the rest the weight phases (~0.28-0.30 ms per expert).

**The copy waited behind the cache's background loads** (`results/decode-next/copy-engine.txt`):
on die 0's copy engine the copies of a code run had a median of 44 us but p75 222 us, p90 507,
p99 1,455: 367 of their 433 ms in copies over 100 us, waiting for an 18.8 MB expert load (~1.4
ms) to pass. Without loads (`--cache-loads 0`) the tail is gone (p90 55 us). **The CPU's rows of
a short forward (M <= 8) are now copied by a kernel** that reads the pinned buffer (`k_copy16`;
die 0 addresses it by its host pointer, checked at startup): median 23 us, p99 123, 71 ms in all;
the handoff back per 6-token code step 7.6 -> 1.8 ms. Prompt chunks (megabytes) keep the copy
engine.

About half of that comes back as a wait for the other dies: die 0 reaching its wait for their
results sooner, they arrive last in ~10% of layers (their own copies wait behind their loads;
`ffn.wait_for_gpu_experts` 2.0 -> 5.3 ms per code step). Tried and not kept: the gathers of
their results by kernel (no change), their input and result copies by kernels on dies 1-3 (40.1
t/s traced against 42.3: kernels reading and writing host memory across the cards cost more),
and the background loads in 512 KB or 128 KB copies instead of whole matrices (41.9, 40.7).

Held-out (`spec-measure-sclk6-cacheloads800cacheperiod8-kcopy/`): mean **37.16 -> 37.57 t/s
(+1.1%)**, 30.0-45.6; outputs identical.

| prompt | step 82 | step 84 |
|---|---|---|
| tcp-udp | 33.38 | 33.41 |
| py-lru | 38.71 | 39.55 |
| cpp-matrix | 44.95 | 45.60 |
| sql-report | 38.16 | 38.69 |
| geometry | 45.14 | 45.45 |
| zh-travel | 30.00 | 29.96 |
| ja-email | 32.38 | 32.81 |
| summary-json | 34.58 | 35.12 |
| long-proof | 50.71 | 51.31 |

`results/trace/` holds the traces of this build (31.40 / 42.67 t/s traced, outputs identical),
`compare-step74.txt` against step 74's and `critpath.txt`. Left: the dies' small copies behind
their loads (~3 ms per code step), e.g. loads paused while a layer's copies are in flight.

## Step 85: the background loads paced by the runtime's layers (tried, not kept)

Step 84 left the dies' own small copies waiting behind the cache's background loads (die 0's
wait for their results ~5 ms per code step). A load already on a copy engine cannot be
interrupted, so the loads were timed instead (`results/decode-next/load-gate.txt`,
`load-gate.patch`; traces against step 84's, outputs identical in every run):

| | cpp-matrix t/s | layer, ms per step | wait for the dies | tcp-udp t/s |
|---|---|---|---|---|
| no gate (step 84) | 42.67 | | | 31.40 |
| one matrix per die per layer, right after it | 35.34 | +18.7 | -2.6 | 30.87 |
| two per layer | 41.24 | +7.3 | +0.4 | 30.82 |
| three per layer | 41.53 | +4.7 | +0.4 | 31.14 |
| none starting during a layer's expert phase | 42.47 | +0.6 | -0.2 | 31.26 |

One matrix per layer does halve the wait for the dies, but the cache falls behind (3,281 loads
published instead of 3,614 in the run; the CPU's experts +14 ms per step). Two or three keep up
with the loads, but bunched right after a layer they slow the next layer's attention (their
unpack kernels and copies meet die 0's work) and the wait does not shrink. Holding new loads
back during the expert phase changes nothing: the ones started during attention are still on
the copy engine when the phase begins. Not kept; the dies' copies stay as they are.

## Step 86: the draft in the traces; the home die keeps its startup experts

The draft was one host span in the traces. Four additions to `--trace` (no change untraced):

- **The draft's parts on die 0's track** (`draft.*`, per draft layer, and "draft (GPU)"), placed
  on the host clock from its end (it synchronizes there), and "draft: issue": the host issuing it
  up to its copies out (they block: pageable memory).
- **The gap after a verify step:** `commit` and its parts (the compressor rows, seeding the
  draft's window, the synchronize), the loop's work after verify and after commit.
- **The cache's background loads**, each on a track per die, from the loader taking it to its
  copies and unpack completing (`ExpertLoader::take_trace`).
- `scripts/trace_critpath.py` reports the cycle, the draft's parts and the drafts against the
  loads.

What they show (`results/decode-next/draft.txt`):

- **A draft is ~8 ms on die 0** (5 positions; head 2.2 ms, routed experts 1.7, the Markov head
  1.0, attention 2.1, shared expert 0.5, the rest 0.7), and GPU-bound: the host issues it in
  ~0.6 ms.
- **The gap after a verify step is `commit`'s synchronize** (~1.2 ms on a code step, 0.6 on
  chat): the draft's window seeded one kept token at a time. The host's own work there is ~20 us.
  (Inside the traced verify step the host returns 2-3 ms after the head: the trace's per-step
  event processing, traced runs only.)
- **Half the drafts took 14-18 ms instead of 8,** in turns: exactly those during which the
  cache was loading experts into die 0 (36 of 36 slow drafts overlap such a load, none of the
  fast ones; without loads every draft is ~8 ms). The parts made of many small kernels doubled,
  the head (the largest read) did not; loads to dies 1-3 alone leave the draft at 8.4 ms.
- **The loader's unpack kernel** took 153 us per matrix with every CU busy (a thread per 17-byte
  block, byte by byte); coalesced through shared memory it takes 21 us, the same bytes. It was not
  the cause (the slow drafts stay) but is kept.

**The home die now keeps its startup experts:** the cache never picks a victim there
(`ExpertCache::Options::fixed_die`), so its ~830 slots hold the placement's experts for the whole
run and every load goes to dies 1-3. Traced: cpp-matrix 42.75 -> 44.99 t/s, tcp-udp 31.18 ->
32.39; the cache still publishes as many loads (3,654 / 3,175). No draft overlaps a load on the
home die any more, and commit's wait is shorter too (1.4 -> 1.0 ms per code step).

Held-out (`spec-measure-sclk6-cacheloads800cacheperiod8-homefixed/`): mean **37.57 -> 40.82 t/s
(+8.7%)**, 31.5-50.1; outputs identical.

| prompt | step 84 | step 86 |
|---|---|---|
| tcp-udp | 33.41 | 35.28 |
| py-lru | 39.55 | 42.62 |
| cpp-matrix | 45.60 | 50.08 |
| sql-report | 38.69 | 43.14 |
| geometry | 45.45 | 49.73 |
| zh-travel | 29.96 | 31.52 |
| ja-email | 32.81 | 33.34 |
| summary-json | 35.12 | 40.87 |
| long-proof | 51.31 | 51.62 |

(Every prompt took fewer tokens per step than in step 84's run, 1-10%: the draft-length rule
chooses from the measured step costs, which changed.) The gain is larger than the traces' (+4-5%):
die 0's loads had also slowed its own dense work. The startup placement now matters more: die
0's set is fixed for the run. Ideas (PLAN): a measure of how out of date die 0's set is, and its
loads in the CPU phase if it is; die 0 seeded with the most stable experts; the draft's own
experts pruned to free die 0's space; the draft's window seeded for all kept tokens at once.
`results/trace/` holds the traces of this build (32.28 / 45.02 t/s traced, outputs identical),
`compare-step84.txt` and `critpath.txt`.

## Step 87: the draft router's usage; the verify step after the head

**Could the draft's experts be pruned?** Its 3 x 128 routed experts are 7.2 GB on die 0, the
room of ~384 of the main model's experts. A smaller draft cannot change the output (the main
model verifies every drafted token), only how much of it is accepted. `DS_DRAFT_ROUTE_LOG=path`
writes each draft call's choices (layer, block position, 3 experts);
`results/decode-next/draft_route_stats.py` reads them (`results/decode-next/draft-route.txt`, the 8 held-out
prompts, 713 draft calls):

| draft layer | experts covering 50 / 90 / 95 / 99% of its choices | never chosen |
|---|---|---|
| 0 | 17 / 57 / 73 / 94 | 19 |
| 1 | 10 / 39 / 48 / 62 | 47 |
| 2 | 8 / 25 / 32 / 41 | 72 |

Concentrated within a prompt (tcp-udp alone: 68 / 31 / 23 experts for 99%), but not stable across
prompts. Kept per layer the top k of seven prompts' choices, the eighth prompt's choices covered:
k = 96 (96 experts freed, ~1.8 GB) 94% on average, 83% for the worst prompt; k = 64 (~3.6 GB) 88%,
70%; layer 2 alone stays at 70% for the worst prompt from k = 64 on (experts only one prompt
uses). A fixed pruning would cost the draft most on unfamiliar prompts, for ~3% more expert
capacity: not done. A draft-expert cache following the prompt (one uses ~120 of the 384) could
free ~4-5 GB; recorded as an idea.

**The verify step after the GPU's head** (`results/decode-next/verify-tail.txt`, host spans
`tail: ...`): the host returns 2.8 ms (code) / 1.9 ms (chat) after the head ends. Of that, the
profile and trace bookkeeping (1.6 / 1.2 ms) runs in traced runs only; in every run the logits of
all the step's tokens (~3 MB at 6) are copied to pageable host memory (~0.2-0.3 ms) and scanned
for the argmax on one thread (0.80 / 0.32 ms), and the cache's step takes ~0.1 ms (~0.7 on a
rebalance). The argmax on the GPU is step 88.

## Step 88: the verify step's argmax on the GPU

Step 87 found the host copying all of a verify step's logits (~3 MB at 6 tokens, into pageable
memory) and scanning them for the argmax on one thread. `k_argmax_rows` now does it on the GPU,
one workgroup per token, and only the M ids come back. The first maximum wins, as
`std::max_element`'s, so the tokens cannot change (`results/decode-next/gpu-argmax.txt`):

| GPU head end -> the host's verify end, median ms | step 87 | step 88 |
|---|---|---|
| cpp-matrix, without the trace's own bookkeeping | 1.20 | 0.41 |
| tcp-udp, the same | 0.67 | 0.24 |

Held-out (`spec-measure-sclk6-cacheloads800cacheperiod8-gpuargmax/`): mean **40.82 -> 41.19 t/s
(+0.9%)**, 31.7-50.8; outputs identical, every prompt with the same tokens per step as step 86's.

| prompt | step 86 | step 88 |
|---|---|---|
| tcp-udp | 35.28 | 35.74 |
| py-lru | 42.62 | 43.07 |
| cpp-matrix | 50.08 | 50.80 |
| sql-report | 43.14 | 43.78 |
| geometry | 49.73 | 49.73 |
| zh-travel | 31.52 | 31.71 |
| ja-email | 33.34 | 33.53 |
| summary-json | 40.87 | 41.12 |
| long-proof | 51.62 | 52.08 |

## Step 89: traces at step 88; the time outside the layers

`results/trace/` regenerated at step 88 (b3d3b24; the command in `trace/README.txt`, sclk 6):
cpp-matrix 45.97 t/s and tcp-udp 32.46 t/s traced, outputs identical to step 32's.
`compare-step86.txt` against step 86's traces: no span within the layers moved by more than ~0.3 ms
per step; the head +0.07 ms (it now includes the argmax), and the new `tail: ...` spans.

`scripts/trace_critpath.py` now splits a verify step's time outside its layers (from step 87's
`tail: ...` spans; older traces print the total only). Median ms per verify step, a 6-token code
step and a 2-token chat step:

| outside the layers | cpp-matrix, 6 tokens | tcp-udp, 2 tokens |
|---|---|---|
| before layer 0 (embedding, its copies, the issue) | 0.28 | 0.21 |
| the head on die 0 (norm, logits, argmax) | 2.47 | 1.81 |
| the ids to the host after it | 0.06 | 0.10 |
| the cache's step | 0.45 | 0.07 |
| profile and trace bookkeeping (traced runs only) | 1.74 | 1.20 |
| the rest | 0.00 | 0.00 |
| total | 4.98 | 3.47 |

So untraced, the time outside the layers is ~2-3 ms, almost all the head: what is left there is the
head's kernel (bf16, 1.32 GB: ~730 GB/s for 1-2 tokens, ~600 for 5-6, against ~880 GB/s per die),
the same kernel as the draft's head (2.2 ms of its 8.3 ms per call, 5 tokens). A faster kernel could
save ~0.3-0.5 ms per head; reading fewer rows is the larger lever. `critpath.txt` (step 88's and
step 86's traces): a 6-token code step ~89 ms with the draft 8.4, die 0's front 54, its wait for the
CPU 25, the tail 6.3.

PLAN's sections B (the dense part) and D (the draft) rewritten: what is done (steps 53, 61-62,
66-68, 73-74, 86, 88), tried (65, 70, 75, 76), left, with the current per-part times.

## Step 90: the GPU clock level, at step 89 (sclk 2 / 6 / 8)

After a reboot (the page cache warmed by reading the model files), the held-out set
(`CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`) at three sclk levels set by
hand on all four cards, commit 2509593 (`results/decode-next/sclk-levels.txt`). Every output
identical to step 32's at every level.

| sclk level (MHz) | mean t/s (8 prompts) | range | ms per token |
|---|---|---|---|
| 2 (1204) | 36.77 | 28.8-44.2 | 27.2 |
| 6 (1608) | 41.10 | 31.6-50.6 | 24.3 |
| 8 (1730) | 42.10 | 32.3-51.9 | 23.8 |

From 2 to 6 the clock rises 33.6% and the speed 11.8%; from 6 to 8, 7.6% and 2.4%. A token's time
as GPU work that scales with the clock plus time that does not (fitted on levels 2 and 6): at
sclk 6 ~8.6 of its 24.3 ms scale with the clock (35%), ~15.8 do not (the CPU's experts, the
handoffs, the host, and GPU work bound by HBM bandwidth, which runs on its own clock: the fp8
GEMVs; corrected in step 94); the fit predicts level 8 within 0.1%. Level 6 after the reboot: 41.10
against 41.19 before it (step 88), the same within noise. The draft-length rule sees the verify
cost, so tokens per step move a little with the clock (ja-email 2.01 at sclk 2, 1.72 at 6 and 8).
Level 6 stays the measuring level (the baseline of every step since 32).

## Step 91: the CPU layout in the runtime, as a copy in RAM (tried, not kept)

Step 83 benchmarked a CPU layout for the experts' mxfp4 rows (each row's codes contiguous, the
scales apart: the same bits, compute x1.8, a cold call -9%) and left it undeployed: the GGUF mmap
was the CPU's only source and RAM cannot hold a second copy next to the page cache. Tested here
without touching the GPU side (`results/cpu-decode/layout-live.patch`): `--cpu-layout
even|odd|all|a,b-c` repacked those layers' experts at startup into RAM (`canon_repack64`, 16
threads, 2 MB pages) and the canonical CPU kernels read that copy; the GPUs kept loading from the
GGUF. `--cpu-layout-control` copied the same layers in the GGUF's own layout instead (a fresh copy
on 2 MB pages, the layout unchanged).

- **Kernels:** `canon_dot_rows64` (step 83's, moved from the bench into `canon_dot.cpp`) and a
  new `canon_dot_rows64_multi` (`canon_dot_rows_multi`'s tile on the layout). Bit-identical to the
  GGUF-layout kernels (`bench-fp4-decode layout`: every row of an expert's three matrices; 1-8
  activations for the multi kernel).
- **Within one traced run** (`results/cpu-decode/layout-live.txt`; the patch's
  `scripts/trace_cpu_copy.py` over a "cpu copy" argument on the "CPU experts" spans), total ms
  per CPU expert:

| run | GGUF layers | copied layers | change |
|---|---|---|---|
| even layers in the CPU layout, cpp-matrix | 0.287 | 0.268 | -7% |
| even, tcp-udp | 0.296 | 0.275 | -7% |
| odd layers, cpp-matrix | 0.292 | 0.257 | -12% |
| odd, tcp-udp | 0.300 | 0.268 | -11% |
| control (even, the GGUF's layout copied), cpp-matrix | 0.289 | 0.291 | 0% |
| control, tcp-udp | 0.295 | 0.296 | 0% |

  The layout, not the fresh copy or its 2 MB pages, made the difference (as step 56 found for
  2 MB pages). Down ~18% faster (0.095 -> 0.077-0.079 ms per expert); gate + up only 2-3%: it
  follows the idle gap and already reads at ~73 GB/s.
- **Held-out with all 40 layers** (`spec-measure-sclk6-cacheloads800cacheperiod8cpulayoutall/`):
  mean **41.10 -> 41.76 t/s (+1.6%)** against step 90's sclk 6 run (+1.4% against step 88's
  41.19), every output identical; a second run, stopped after 6 prompts, gave the same within
  ~1%. The copy is 269 GiB and the repack 44-51 s per startup; free memory reached 0 and 14 GB of
  the page cache were evicted (no disk reads during generation).
- **One abort:** py-lru's process aborted at exit, after its results (`malloc_consolidate():
  invalid chunk size`), once in 27 runs with a copy; none in any earlier run's log, not
  repeated in 6 py-lru runs with `MALLOC_CHECK_=3`; no core was kept and no cause found (the new
  code's writes stay inside their buffers). The patch also has a fatal-signal backtrace handler
  in `ds_completion.cpp` for a next look. Step 92 then saw the same abort without this code
  (`double free or corruption`, at the same point of the exit): an older teardown bug.

Not kept: +1.6% for a 269 GiB copy that fills RAM next to the page cache, ~45 s more per start,
and an unexplained abort. A deployment would need a repacked expert file as the only source
(step 83's list: the GPU loader's unpack reading it), not a second copy.

## Step 92: the draft's head over part of the vocabulary, simulated (not built)

The draft reads the full head (129,280 rows, 2.2 ms per call) and adds its Markov bias over the
full vocabulary (1.0 ms), then takes the argmax. Over a subset S the argmax is the same token
whenever the full one is in S, so this can be simulated exactly for accepted drafts: a step
keeps its accepted drafts up to the first one outside S. `DS_DRAFT_TOKEN_LOG` (new, a study log
like step 87's) writes per verify step the drafted and emitted ids; one held-out run with it
(`results/decode-next/draft-vocab/`, every output identical), then `results/decode-next/draft_vocab_sim.py`
(`results/decode-next/draft-vocab.txt`) for K from 1k to 96k and four orders of the vocabulary
(every subset also holds the 1,280 ids from 128,000: special tokens):

- **id:** token id order (contiguous rows, no copy needed). The ids are not close to frequency
  order: 10% of generated tokens have ids above 39k.
- **calib:** frequency among the 20,919 generated tokens of the 30 routing-study prompts (step 13,
  8 categories; no held-out prompt), the rest by id.
- **calib+ctx:** calib plus every token the run has emitted so far.
- **oracle:** frequency in the held-out run's own tokens: an upper bound only.

| K | share of emitted tokens in S (calib) | est. t/s: id | calib | calib+ctx | oracle |
|---|---|---|---|---|---|
| 1k | 44.0% | -49.6% | -46.3% | -27.1% | -3.0% |
| 4k | 57.9% | -38.7% | -35.7% | -21.1% | +4.0% |
| 16k | 79.4% | -19.5% | -17.8% | -11.2% | +3.6% |
| 32k | 87.7% | -11.9% | -8.8% | -4.8% | +3.1% |
| 64k | 94.5% | -5.6% | -4.8% | -2.9% | +2.0% |
| 96k | 98.2% | -0.6% | -0.5% | -0.1% | +1.0% |

(8-prompt means against the full vocabulary; the estimate takes each step's measured time minus
the draft's saved head and Markov time, scaled by 1 - K / V, and the simulated tokens per step;
at the full vocabulary it gives 41.07 t/s against 41.04 measured.) Every deployable subset loses:
a missed token cuts the chain of accepted drafts, which costs more than the at most ~3.2 ms per
step the smaller head saves; even the oracle gains at most ~4%. Not built; the triage's item 2 is
closed. In the same run ja-email's process aborted at exit after its results (`double free or
corruption (fasttop)`): step 91's abort, without step 91's code.

## Step 93: a rare abort at exit, investigated (no cause found yet)

Steps 91 and 92 each saw one process abort at exit, after all its results: py-lru with
`malloc_consolidate(): invalid chunk size` (step 91, with its copy) and ja-email with `double
free or corruption (fasttop)` (step 92, without it). Both came right after the expert cache's
stats, which the `Runtime` destructor prints just before it stops the cache's loader threads
(corrected in step 96: not after they are joined); so the error was detected in the loader's
shutdown, the rest of the teardown (the draft, the CPU kernel, the GPU buffers and streams),
`main`'s locals, static destructors or the HIP / HSA runtime's exit. Both are glibc's
checks on small freed blocks (<= 128 bytes): one freed twice, or its header written after the
free. `results/decode-next/exit-abort.txt`:

- **Not repeated in 56 runs** of the same command: 14 plain (ja-email and py-lru in turn), 30
  under gdb (stopping only on SIGABRT / SIGSEGV / SIGBUS, with every thread's stack at one) and
  10 under AddressSanitizer (`bin-asan/`: the g++ objects instrumented, malloc and free
  intercepted for the whole process, HIP runtime included; ja-email, then all 9 held-out
  prompts): no signal, no report, every output identical.
- **So not a double free on every run:** ASan checks every free and would have caught one in its
  first run, while glibc catches a fastbin double free only when the block is the last one freed.
  It is rare and timing-dependent: most likely a race at shutdown, between two threads or between
  our teardown and the HIP runtime's own.
- **Not the cause:** step 91's layout code (seen without it), loads in flight (ja-email had every
  started load published; loads are synchronous in their threads, which are joined first), a
  device buffer freed twice in the destructor's list (no two names hold one allocation).
- **No measurement affected:** outputs and timings are written before the teardown (the trace
  file too).

Kept for the next occurrence: a fatal-signal handler in `ds_completion.cpp` prints the stack
(`backtrace` is called once at startup so the handler does not load libgcc after a corruption;
addresses resolve with `addr2line -f -C -e bin-hip/ds-completion <+offset>`), and
`scripts/spec-measure.sh` notes a nonzero exit code on the prompt's summary line. No workaround
(an `_exit` after the results would hide where it is).

## Step 94: step 70's verify halves, traced: why they lose (closed)

Step 70 let a verify step run as the prompt does in two halves (A's CPU experts while die 0 runs
B's attention, and back) and found it 10-13% slower without measuring why; die 0 and the CPU
still take turns (~25 ms of CPU experts per code step against die 0's ~54 ms), so it was the
triage's largest open idea. Its patch (`results/decode-next/verify-halves.patch`,
`DS_VERIFY_HALVES=1`) still applies; traced against step 88's traces
(`results/decode-next/verify-halves-trace.txt`, new `results/decode-next/trace_overlap.py`):

| a 6-token code step, median ms | without halves | halves (traced) |
|---|---|---|
| the step | 89.4 | 128.2 |
| die 0 busy | 54.5 | 82.6 |
| the CPU's experts | 30.4 | 32.6 |
| both at once | 7.6 | 4.2 |
| the host waiting for die 0's lists | 23.5 | 0.1 |

Untraced: cpp-matrix 50.62 -> 43.85 t/s (-13%), tcp-udp 35.53 -> 29.37 (-17%), outputs identical.
The halves overlap less, not more, and the host becomes the bottleneck; but the cause is die 0's
own work. Its dense part reads weights: ~29 ms for one token, +4-5 ms per further token (die 0
busy by verify size without halves: 1 token 28.9 ms, 2: 34, 3: 42, 4: 50, 5: 50, 6: 54.5). Two
halves of 3 tokens read every weight twice, ~2 x 42 = ~84 ms against 54.5: ~30 ms more on die 0,
more than the CPU's whole 25-30 ms could hide even with a perfect launch thread. A verify step's
tokens must share one weight read on this machine (the prompt's 512-token chunks amortize it, so
there the halves pay, step 38). Closed for good; the patch stays as it was.

Correction to step 90: the ~65% of a token's time that does not scale with the GPU clock is not
only the CPU side: GPU kernels bound by HBM bandwidth (the fp8 GEMVs) do not scale with the shader
clock either.

## Step 95: llama.cpp upstream cannot read our GGUF files; the fork stays

Setting up a second machine raised whether the llama.cpp fork (branch `deepseek41`, one commit,
`b72050d03`, on upstream `797de63b2`) could give way to upstream, which has since merged
DeepSeek-V4.1 support. It cannot, for these files: upstream `a4d4f2fd4` loads Unsloth's 1017
UD-Q8_K_XL without an error and generates garbage (`results/dist/llama-cpp-fork.txt`). Exactly one
tensor is the cause: the Engram table `per_layer_token_embd.weight` has ggml type 43, which the PR
(and so the files) defines as MXFP8 (33 bytes per 32 values: 202.76 GB) and merged ggml as plain
`F8_E4M3` (1 byte per value: 196.61 GB, misaligned rows, scale bytes read as weights). Upstream's
converter writes the table as BF16 instead; the MXFP8 quants (UD-Q3/Q4/Q8_K_XL) are known broken
upstream (llama.cpp issue #31385, open). Details, the options and the evidence:
`docs/llama-cpp-upstream.md`.

Kept: the fork, until a newer DeepSeek model is quantized with the merged code. A fresh machine gets
it pinned by hash (`git fetch origin b72050d03ae1...`, the head of PR #30817). Checked here from a
fresh clone: `fp8pack` regenerates `dense-fp8.gguf` byte for byte, two held-out prompts give the
reference outputs byte for byte (49.60 and 35.47 t/s), and its `llama-completion` matches the fork's.

## Step 96: the exit abort revisited: a wider window, three teardown fixes, more in the handler

A second reading of the teardown against step 93 (`results/decode-next/exit-abort-step96.txt`):

- **Step 93 placed the abort too late.** The last line before both aborts (`# expert cache: ...`)
  is printed before the cache's loader is stopped, not after. So the loader's shutdown is inside
  the window: on stop the four loader threads woke together and each freed its stream, its 2 pinned
  and 2 device buffers and its 3 events at the same moment.
- **The expert tensors stayed registered at exit:** ~269 GB of the model's mapping registered with
  HIP for direct loads (step 43), never unregistered; the mapping was then removed and HIP's own
  exit ran after it. On every run, so not by itself the rare abort.
- **A possible hang:** `~ExpertLoader` set `stop` without the mutex its threads wait under, so a
  wakeup could be lost and the join never return (not seen).

The first tested alone: `bin-hip/bench-loader-exit` (new) does only that shutdown, one thread per
die, ~37 ms per iteration: 7,800 concurrent shutdowns in all, no abort. So concurrent HIP frees
alone do not show it (threads that had served thousands of loads from registered memory are not
covered).

Changed, each right on its own: `stop` set under the mutex; the loader threads free their GPU
objects one at a time; the expert tensors unregistered once the threads are joined, before the
model's mapping goes (4,308 pieces, no measurable time). ja-email and py-lru after it: outputs
identical, 33.76 and 42.80 t/s, clean exits. If the abort never returns we will not know which
change it was; if it does, the handler now says more.

**The handler** (`ds_completion.cpp`, state in `src/fatal.h`) prints, besides the stack: the
shutdown phase (set at each step: the loader's stop and unregistering, the draft, the CPU kernel,
the dies' buffers, the home die's, `main`'s locals, after `main`), the thread's name and whether
it is the main thread (the loader's are named `load-die<N>`, `load-register`), and the loads
started / finished per die. Checked with `kill -ABRT` during a run:

    # fatal signal 6; phase: run; thread: ds-completion (the main thread); loads started / finished per die: 0/0 658/658 664/664 660/660

`main` now destroys the runtime explicitly before its other locals, so the phases around it are
exact. Not done (needs sudo): `kernel.core_pattern` to a file instead of apport, for a core.

## Step 97: the dies' results in smaller copies, the staged block in one

A review of step 88's traces for local misses (`results/decode-next/local-review.txt`): each of
die 0's spans by verify-step size, against the kernel bench and the code. Two things stood out;
this step is the first, step 98 the second.

**The tail** (the triage's item 4, "measure only"): after the CPU's rows, die 0 waits for dies 1-3
in about a third of a 6-token step's layers and 60% of a 1-token step's (the CPU often has
nothing there). Their path is all copies through the host: the lists and x out (2 copies), into
each die (2), the kernel, the rows back to the host and into die 0. A short forward issues the
copies back before the host knows the counts (step 71), so each took the bound: 6 rows per token,
720 KB per die and direction at 6 tokens, while a die holds 1.5 rows per token on average. From
the last die's end to die 0 having the rows: 53 us at 1 token, 142 at 6.

Changed (short forwards only, `hip/rt.cpp`):
- **The first copy takes 3 rows per token** (`early_rows_per_tok`); a die with more (3.5% of the
  (die, layer) results) gets a second copy for the rest, issued by `moe_finish` once the counts
  are known. `DS_EARLY_ROWS=1` forces second copies everywhere: the same output.
- **The staged block is one copy:** a short forward's lists sit in the last 8 tokens' list slots,
  next to x, so the lists and x leave die 0 in one copy and enter each die in one.

Pure data movement: every output identical. Held-out mean **41.10 -> 41.96 t/s (+2.1%)**, every
prompt +1.6% to +3.6% (the prompts with short steps most).

**The exit abort, a third time** (ja-email in this step's run, `malloc_consolidate(): invalid
chunk size`; `results/decode-next/exit-abort-step96.txt`), now with step 96's handler: phase
"after main", the main thread, no load in flight. The stack is libamdhip64's exit handler
(`__cxa_finalize`) into libhsa-runtime64's shutdown, which calls `free` and fails glibc's heap
check. So all of our teardown had finished, step 96's three fixes did not remove it, and it is
either the HSA runtime's own shutdown or a heap damage from earlier that is noticed there (10
ASan runs were clean, step 93). Both libraries are stripped. Outputs and timings are unaffected
(they are complete before it). Ending the process with `_exit` after our teardown would avoid it
and would also hide an earlier corruption; not done, documented only.

## Step 98: the compressor's projections once per chunk

The second find of step 97's review: `attn_prep` ran the compressor's two projections (`cmp_kv`,
`cmp_gate`: 512 x 5120 f32, 10 MB each) once per token, on the four layers that build compressed
keys (2, 8, 14, 20): ~139 us per token and layer, 4.5 ms of a 6-token step, growing 0.74 ms per
token while everything around it was batched (steps 24, 66-68). PLAN listed its time but had an
item only for the indexer.

Now `attention()` computes both for the whole chunk with `k_gemv_f32_m` (each weight read once
per 8 tokens; every row is the one-token kernel's, bit for bit), and `attn_prep` takes its
token's rows: the group state, the pooling and the latent stay token by token, in position order.
Prompt chunks take the same path.

Every output identical. Held-out mean **41.96 -> 42.52 t/s (+1.3%)**: code and structured prompts
+1.4% to +2.2%, the chat prompts with short steps flat (-0.1% to +0.6%). With step 97: **+3.45%**
since step 90.

Left from the review (PLAN): the compressor's per-token copies in one kernel; the hc projection
and split in one kernel (~1 ms per step, the reduction order must stay); the draft's head as a
lossy fp8 copy (~1 ms per step, 0.66 GB of die 0, acceptance to simulate); the dies' three copies
into die 0 as one.

## Step 99: the review's leftovers (three tried and not kept, one estimated)

The four smaller items left by step 97's review (`results/decode-next/review-leftovers.txt`;
every variant's outputs identical to the references):

- **The hyper-connection kernels fused** (`hc-mix.patch`): the hc_fn GEMV's split sum, `hc_split`
  and `hc_pre_norm` in one kernel, statement for statement (2 launches per sublayer instead of 4).
  35.2 us against 35.7 for the separate calls, and no change in a run. The time is the kernels'
  work (one block per token: its reads of 20,480 values, two block sums, 20 Sinkhorn iterations
  ~5 us), not their launches: step 65's conclusion again, now with the kernel written.
- **The compressor's group state, pooling and latents per chunk** (`compressor-chunk.patch`): one
  kernel walks the chunk's tokens per element (slot writes, a pooled latent per completed group,
  the commit rows), then the latent chain runs once for all groups. Held-out mean 42.52 -> 42.52
  t/s; the prompt gate passes (chunks 1, 8, 13, 512) and a 1,989-token prompt gains 1.1% (67.0 ->
  67.8 t/s). Under the bar for decode; step 98 had taken the part that mattered (the two 10 MB
  reads per token).
- **One gather stream per die** (`gather-streams.patch`), so a die's copy into die 0 waits for its
  own die only instead of queueing behind the others in die order: ~0.5% slower on three prompts
  (52.04 / 36.54 / 34.29 t/s against 52.2-52.4 / 36.7-36.9 / 34.4-34.5). The copies share one
  engine; more streams add waits and no overlap.
- **The draft's head as a lossy fp8 copy** (not built): the bf16 head through LDS already reads at
  ~600 GB/s (1.32 GB in 2.2 ms at 5 tokens); the fp8 GEMV at 5 inputs reads at ~240 GB/s (step
  75), so half the bytes would take ~2.7 ms. Only with the restructured multi-input fp8 kernel
  (PLAN, "big"), and it would cost 0.66 GB of die 0.

So the review's yield is steps 97 and 98 (+3.45%); what is left in die 0's front is kernel work
at the memory limit or the known multi-input fp8 problem.

## Step 100: traces at step 98

The traces in `results/trace/` were step 88's; regenerated at step 98's code (`e23d8fe`: steps 97
and 98 changed the verify step), with `compare-step88.txt` and `critpath.txt`. Outputs identical;
47.60 and 33.29 t/s traced (52.41 and 36.85 untraced).

A 6-token code step (cpp-matrix, medians, ms): 89.4 -> 87.1. Die 0's front 53.9 -> 52.3, the tail
(its wait for dies 1-3) 6.3 -> 4.4, its wait for the CPU 24.5 -> 26.5 (the CPU's own time, which
neither step touched: 30.4 -> 32.5, run to run), outside the layers 5.0, the draft 8.3. Over all
of cpp-matrix's steps: attn.compressor 4.0 -> 2.5 ms per step, the wait for the dies 5.0 -> 3.6,
dispatch 2.5 -> 2.1. PLAN's section B has the spans.

## Step 101: weight rows a multiple of 4096 bytes apart read slowly; the fp8 matrices' rows padded

Started as the question behind PLAN's "big" item: what limits the fp8 GEMV at several inputs?

**The ceiling** (`results/decode-next/fp8-probe.txt`, bench-only variants of the kernel, not
exact; `fp8-probe.patch`): with the activations loaded once and nothing decoded, all 6 inputs' dot
products cost about what one input costs today (attention output A 66 us against 57.8 at one input
and 121.6 at six; B 79.5 / 85.3 / 155.4; engram 238 / 203 / 461). So the multiplies are not the
limit; bringing each lane its activations every step is. And more rows per wave, which would
spread that cost, slowed even the bare weight loads (output A's shape: 64 us at 1 row per wave,
250 at 4).

**Why** (`bin-hip/bench-row-stride`, new: the kernel's load pattern alone over a random matrix;
`results/decode-next/row-stride.txt`): it depends on the row length. 5120 rows of 8192 bytes read
at 279 GB/s, and worse with more rows per wave (122 at 8); the same bytes with the rows 64 bytes
further apart read at 715 GB/s at any rows per wave. Likewise 8192 x 4096: 79 GB/s at 8 rows per
wave, 679 padded. Rows a multiple of 4096 bytes apart put every wave's loads at the same offset
within a page, and those collide in the GPU's caches (which cache and how was not established;
the effect is measured). Rows of 1280 or 5120 bytes show little or none of it.

**The change:** `fp8_row_stride(K)` (`hip/fp8_gemv.h`): a matrix whose K is a multiple of 4096
gets its rows K + 64 bytes apart in its device copy (the runtime uploads the fp8 file tensor by
tensor, those row by row; the draft's likewise), and the three fp8 kernels take the stride. That
is the two attention output matrices of every layer (80 matrices with the draft's, +37 MB on die
0: 3 startup experts fewer there). The bytes and every sum's order are unchanged: `bench-fp8
--bitwise` 36 of 36, every held-out output identical. `DS_FP8_PAD=0` gives the old layout.

| us per call (`bench-fp8`) | 1 input | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| attention output A (8192 x 4096) | 58.0 -> 45.4 | 65.4 -> 51.0 | 100.8 -> 77.2 | 116.6 -> 97.3 | 123.2 -> 113.3 |
| attention output B (5120 x 8192) | 86.1 -> 57.3 | 101.5 -> 62.3 | 148.3 -> 93.3 | 151.8 -> 113.1 | 158.4 -> 131.8 |

Held-out mean **42.52 -> 43.55 t/s (+2.4%)**: the prompts with short steps gain most (ja-email
+5.6%, tcp-udp +3.1%, zh-travel +2.9%; cpp-matrix +0.5%), as the table says: at one or two inputs
the kernel was bound by these reads, at six by the activations. Prompt speed is unchanged (the CPU
bounds it).

Open after this: the same check on the other kernels' matrices (the router's and the compressor's
rows are 20,480 bytes apart, the hyper-connection projection's 81,920, the head's 10,240); and the
six-input case, where more rows per wave is still not faster with padded rows (the sweep's best
stays 2), so the activations' cost needs the kernel change itself. One lead from the sweep: the
8- and 16-lane splits read at 450-520 GB/s at six inputs (they share decoded activations across
a block's 64 rows) but fix another order of additions; the 64-lane order could be kept on such
a split by letting each lane carry several of the 64 partial sums.

## Step 102: step 101's check on the other kernels; launches that fill the GPU's wave slots exactly

Step 101's question on the matrices of the other kernels, and what turned up on the way
(`results/decode-next/wave-slots.txt`).

**Row padding** (`bin-hip/bench-kernels --stride`, new: the f32 / bf16 GEMVs with rows K apart or
padded, outputs compared bit for bit). Only the router gains (rows 20,480 bytes apart: 20.4 -> 17.1
us at one input, 28.7 -> 21.8 at three, 36.8 -> 32.2 at six). The head (10,240), the
hyper-connection projection (81,920, 24 rows), the indexer's projection and the Markov head do
not change. The compressor's projections have the router's row length, so they are padded too
(~1 us).

**A launch of 2017-2048 waves takes ~48 us whatever it computes.** The compressor's 512 x 5120
GEMV read 51.5 us at one input and at six, while 448 and 576 rows took 19 and 23. Its grid is
exactly 2048 waves (128 x 4 blocks of 4), the GPU's wave slots (64 CUs x 32). `bin-hip/bench-wave-slots`
(new: a kernel whose waves read N KB each, by wave count):

| waves per launch | empty kernel | 1 KB per wave | 5 KB | 20 KB |
|---|---|---|---|---|
| 1792 | 3.3 us | 4.5 | 14.8 | 71.4 |
| 2016 | 3.4 | 4.8 | 16.2 | 79.7 |
| 2020, 2032 | 8.4 | 9.6 | 19.8 | 82 |
| 2040, 2048 | **48.2** | **48.2** | **48.4** | 93.8 |
| 2052 | 3.5 | 5.3 | 55.2 | 110.0 |
| 2112 | 3.5 | 5.0 | 25.6 | 90.9 |
| 2304 | 3.6 | 5.0 | 19.1 | 114.3 |
| 4096 | 4.8 | 6.9 | 62.5 | 139.2 |
| 2048 as two launches (768 + 1280) | 5.7 | 7.4 | 19.2 | 78.7 |

The same with 64-, 128-, 512- and 1024-thread blocks: it is the wave count, not the grid's shape.
Later multiples of 2048 cost a light kernel ~30 us more than their neighbours (4096, 6144, 8192).
What the dispatcher does there was not established; the effect is measured. Two launches avoid it
for a light kernel and cost a heavy one 10-15% (4096 waves x 20 KB: 139 -> 173 us).

**Which of the runtime's launches are there** (`results/decode-next/hip_launch_waves_shim.cpp`, an LD_PRELOAD
count of every launch by kernel and waves, and `results/decode-next/launch_waves.py`; tcp-udp, 382 k launches):
7,920 on the floor, about 55 per verify step. Most are fp8 GEMVs that take longer than 48 us
anyway: attention query B (2048 waves at 2+ inputs, 4096 at one) and output A (8192 / 4096). As two
launches those were slower in every case (query B 61.8 -> 66-67 us at one input, output A 46.1 ->
51), so they stay. The light ones: the compressor's two projections (~7 launches per step) and
the activation quantize of 4 tokens of the attention output (131,072 values = 512 blocks; 5.5
per step on average, 48 us against ~19).

**The change** (three small pieces, the same bits; `bench-kernels --bitwise` 28 of 28, `bench-fp8
--bitwise` 36 of 36, every held-out output identical):
- `stall_split()` (`hip/fp8_gemv.h`): a grid of 2017-2048 waves goes out as two launches, blocks
  [0, 3/8) and the rest; used by `fp8_quant_act` and, up to 3 inputs, by the f32 / bf16 GEMV
  (the compressor, with its rows padded: 51.4 -> 29.2 us at one input, 51.5 -> 39.4 at three; at
  six the two launches took 61 us against 51.5, so from 4 inputs it stays one launch). `DS_STALL_SPLIT=0` turns it off.
- `f32_row_stride(K)` (`hip/kernels.h`): the router's and the compressor's matrices sit with rows
  K + 16 floats apart on the GPU (`DS_F32_PAD=0`: K); the GEMVs take a row stride.

Held-out mean **43.55 -> 44.00 t/s (+1.0%)**, every prompt up (py-lru +2.1%, tcp-udp +1.6%,
cpp-matrix +1.1%, zh-travel +1.0%, sql-report +0.9%, summary-json +0.7%, geometry and ja-email
+0.5%); long-proof 56.04 t/s. The compressor's split alone, measured first on two prompts, showed
nothing (38.04 / 38.18 -> 38.02 / 38.30 and 36.45 / 36.40 -> 36.38 / 36.39 t/s), so the parts were
not separated further: each is a few tenths of a percent at most by its launch counts.

**Tried, not kept: the expert slots padded** (`results/decode-next/pool-pad.patch`). A slot's
six parts (gate, up, down: codes and scales) all start a multiple of 4096 bytes from the pool's
base, and slots are a multiple of 4096 apart. With 256 bytes after every part: py-lru 45.77 /
45.62 -> 46.16 / 45.94 t/s, cpp-matrix 53.34 / 53.08 -> 53.15 / 53.05; 1280 bytes: 45.92 and
52.75. About +0.25% over the two, inside the runs' spread. (An expert's rows are 2560 or 1152
bytes: 8 and 32 offsets within a page, not one.) One of those runs ended with step 96's exit
abort, again after main in the HSA runtime's shutdown.

Step 101's check is closed with this: the padding gains where rows are long and few offsets
repeat (the attention outputs, a little the router), nowhere else. Open: the 6-input fp8 kernel
(step 101's lead), and whether query B's and output A's wave counts can leave the multiples
without two launches (their rows per wave must divide 32; a 3-wave block does not change the
count).

## Step 103: the fp8 GEMVs by wave count: no penalty on multiples of 2048; rounds of ~512 waves at several inputs

Step 102 left one question: attention query B and output A launch on multiples of 2048 waves
(~50 launches per verify step); as two launches they were slower, which does not say whether
they pay for the multiple. The test: the same kernel and split over a matrix cut by a few waves'
rows (`bench-fp8 --waves`, new; `results/decode-next/wave-rounds.txt`).

**They do not pay.** On the multiple they are a little faster than just below it:

| us per call, one input | on the multiple | 4 waves fewer | 64 fewer | 512 fewer |
|---|---|---|---|---|
| attention query B (4096 waves) | 60.5 | 64.7 | 64.7 | 58.1 |
| attention output A (8192 waves) | 45.3 | 48.7 | 50.4 | 48.0 |

and the same at 2, 3 and 6 inputs (query B at 2048 waves: 63.6 / 67.5, 67.7 / 70.9, 78.4 / 82.0;
output A at 4096: 50.9 / 54.0, 77.2 / 76.9, 113.6 / 115.5). So step 102's floor concerns light
launches only, and those are handled. Closed.

**What the sweep shows instead:** the several-input kernels take time in steps of about 512
waves, and a launch a little over a multiple of 512 pays for a whole further round:

| us per call | waves today | today | 128 waves fewer | 256 fewer |
|---|---|---|---|---|
| attention query A, 3 inputs | 640 | 25.7 | 20.8 (512) | 19.5 |
| attention query A, 6 inputs | 640 | 30.0 | 21.6 (512) | 21.2 |
| shared expert gate (and up), 3 inputs | 1152 | 36.1 | 27.4 (1024) | 27.1 |
| shared expert gate (and up), 6 inputs | 1152 | 45.3 | 37.4 (1024) | 36.9 |
| shared expert down, 3 inputs | 640 | 25.6 | 22.0 (512) | 21.0 |
| shared expert down, 6 inputs | 640 | 31.7 | 26.4 (512) | 25.6 |

The last 128 waves (10-20% of the rows) cost 4-9 us, the next 128 removed almost nothing. The
one-input kernels fall smoothly (query A, 1280 waves: 11.7, 11.3, 9.9). The likely reason, not
verified: with one accumulator per input and row the kernel needs more registers, so fewer of its
waves fit on the GPU at once (the compiler's occupancy warnings for the larger instances point
the same way), and the waves beyond a full set run as a short extra round.

**Worth, if the extra round went away for free:** per layer at 3 inputs ~5 + 2 x 9 + 4 = 27 us, at
6 inputs ~8 + 2 x 8 + 5 = 29 us; over the layers about 1.2 ms of a 65-97 ms step, 1-1.5%.
Nothing is changed here: the wave count is rows / rows per wave, the rows per wave must divide 32
(1, 2, 4, 8: the sweeps chose 2), so removing the tail means some waves taking a third row, or
fewer registers per wave. Both belong to the several-input kernel's design (PLAN's "big" item),
where this is now one of its measurements.

## Step 104: the fp8 GEMV at several inputs: 16 or 32 physical lanes carrying the 64 sums

PLAN's "big" item. What fixes a 64-lane row's bits: lane l adds its steps (k = 16 l, 16 l + 1024,
...) in order into one partial sum, and the 64 sums meet in a fixed tree (lanes 32 apart, then 16,
8, 4, 2, 1). Any kernel that forms the same 64 sums and the same tree gives the same bits,
whichever lane does the work. And what costs at several inputs (step 101's probe): with 64 lanes
per row a wave holds one or two rows and loads and decodes every input's activations for itself.
The 8- and 16-lane kernels share one decoded tile per block of 16-64 rows through shared memory
and read the large matrices at 450-620 GB/s at 3-6 inputs (64 lanes: 290-450), but add in
another order.

**The kernel** (`fp8_gemv_virt_kernel`, `hip/fp8_gemv.hip`): the shared-memory kernel's work
split over P = 8, 16 or 32 physical lanes per row, each lane holding 64 / P accumulators per row
and input. In a 1024-code tile a P-lane kernel's lane l already takes the steps k0 + 16 (l + s P),
s = 0 .. 64 / P - 1: exactly the one step each of spec lanes l, l + P, l + 2 P, ... there. Step s
adds into accumulator s instead of a shared one; at the end the tree's first levels are sums
inside the lane, the rest the shuffles over P lanes. Loads, decodes and dot products are the
shared-memory kernel's. The one-input kernel is untouched and stays the specification. Needs K a
multiple of 1024 and a block's rows in one activation group (true for the 64-lane matrices).

**Measured** (`bench-fp8 --virt`, new: every layout against one-input calls bit for bit, and
timed; `results/decode-next/fp8-virt.txt`): all layouts identical at 2-8 inputs; `--bitwise` 63 of
63. The extra accumulators cost: the attention outputs reach about two thirds of the 8- / 16-lane
kernels' speed-up (output A at 6 inputs: 113 us, 68 for the best plain 8-lane layout, 79 here).
The layout per matrix and input count is a measured table (`virt_choice`); 2 inputs and matrices
under 1024 rows stay on the 64-lane kernel. `DS_FP8_VIRT=0` turns it off.

| us per call (64-lane -> the built-in choice) | 3 inputs | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| attention output A (8192 x 4096) | 77.2 -> 60.4 | 97.4 -> 63.5 | 101.8 -> 71.5 | 113.4 -> 79.4 | 184.0 -> 89.9 |
| attention output B (5120 x 8192) | 93.2 -> 72.4 | 113.1 -> 82.5 | 112.4 -> 98.3 | 131.6 -> 105.6 | 170.1 -> 100.0 |
| shared expert gate, up (2304 x 5120) | 36.2 -> 28.0 | 42.0 -> 29.9 | 40.6 -> 34.3 | 45.3 -> 42.3 | 75.7 -> 42.4 |
| engram projection (25600 x 6144) | 312.9 -> 266.3 | 394.4 -> 279.6 | 417.0 -> 305.0 | 466.9 -> 340.8 | 675.4 -> 360.2 |
| attention query A (1280 x 5120) | 25.6 -> 23.1 | 27.0 -> 24.3 | 27.1 (kept) | 30.0 -> 28.2 | 59.7 -> 30.4 |

The shared expert's gain is partly step 103's extra round going away (1152 waves become 144-288).

Held-out mean **44.00 -> 44.57 t/s (+1.3%)**, every output identical: geometry +2.2%, py-lru and
summary-json +1.7%, cpp-matrix +1.5%, sql-report +1.0%, ja-email +0.9%, tcp-udp +0.7%, zh-travel
-0.1% (its steps are mostly 1-2 tokens, which do not use the kernel); long-proof 56.04 -> 57.11.
Less than the table suggests (~60 us per layer at 3-6 inputs would be ~3% of a step): not
explained here; the next traces will show what the dense front takes now.

Open: the prompt's many-input kernel is now slower than slices of 8 through this one on the large
matrices (output A at 64 inputs 1008 us against 720, B 1118 / 801, engram 4282 / 2878; same
bits), though the CPU bounds prompt speed; 2-input calls; fewer registers per lane.

## Step 105: traces at step 104, and why its gain was under the bench's

Step 104's bench table added up to ~3% of a 3-6-token step; the held-out mean rose 1.3%. The
traces in `results/trace/` were step 98's, so: new ones at step 104 (`compare-step98.txt`,
`critpath.txt`), and the same build traced with the new kernel off (`DS_FP8_VIRT=0`) and on
(`results/decode-next/virt-onoff.txt`).

**The kernel delivers in the runtime what the bench measured.** cpp-matrix, ms per verify step
over all its steps, off -> on: attn.out 9.49 -> 7.65 (-1.84), ffn.shared 9.20 -> 8.30 (-0.90),
engram 1.28 -> 1.08, the draft 8.23 -> 7.98: 3.2 ms of GPU work, 3.6% of the ~88 ms cycle. (The
bench predicts ~2.3 ms for attn.out at this prompt's mix of step sizes: 80% arrives.)

**Three reasons the mean shows 1.3%:**
1. The shared expert's 0.9 ms is not on the critical path: it runs while the CPU computes its
   experts, so die 0 only waits longer (ffn.wait_for_cpu_experts 29.53 -> 30.66). What is on the
   path is attn.out, the engram and the draft: 2.3 ms, 2.6% on a code prompt.
2. The CPU's experts were 0.4 ms slower in the "on" trace (36.07 -> 36.45, cpu.gate_up and
   cpu.down both ~1%), which is their run-to-run spread (step 100 saw 30.4 -> 32.5 at 6 tokens).
   The layers' total fell 1.50 ms, the cycle 1.75.
3. Untraced and back to back, the kernel is worth **+2.0% on cpp-matrix** (53.56, 53.56 -> 54.64,
   54.67 t/s) and **+2.1% on geometry** (54.21, 54.16 -> 55.37, 55.25), which matches 1.75 of 88 ms.
   The held-out mean is lower because three of its eight prompts take mostly 1-2-token steps
   (tcp-udp 2.1 tokens per step, ja-email 1.8, zh-travel 1.5), which do not use the kernel:
   +0.7%, +0.9%, -0.1%.

So the +1.3% is the kernel's whole effect on this prompt set, and nothing is lost in between;
the ~3% was my arithmetic counting the shared expert as critical and every prompt as code.

**Where a step goes now** (cpp-matrix, 6-token steps, medians): 84.8 ms, plus the draft's 8.1
before it. Die 0's dense front 48.2 (52.3 at step 98), its wait for the CPU's experts 26.7, its
wait for dies 1-3 4.7, outside the layers 5.0. Since step 98 over all steps: attn.out 11.52 ->
7.65 ms (steps 101 and 104), the layers 82.8 -> 78.6; tcp-udp: attn.out 8.74 -> 5.83, the layers
53.3 -> 50.1. Traced 49.26 and 34.88 t/s (54.03 and 38.89 untraced).

What this says about the next step: on code, die 0's wait for the CPU (~27-31 ms per step) is
now the largest single piece, and any saving in the shared expert stays invisible until that
wait shrinks. The dense front's largest spans: ffn.shared 8.3 (hidden), attn.out 7.7, attn.core
5.7, attn.q 5.1, hc 5.0.

## Step 106: drafting ahead, estimated from the draft's logs (not built)

The draft runs serially on die 0 (8 ms per step: three layers 4.3, of them the routed experts 1.6,
the head 2.15, the Markov bias 1.0), and during it dies 1-3 and the CPU idle. Spreading its
experts over the dies would save little (1.6 ms to split, ~100 us of handoff per layer, the
memory zero-sum). The larger idea is PLAN's "drafting ahead": start the next draft on another die
while the verify runs, from a guessed anchor. Estimated before building anything
(`results/decode-next/draft_ahead_sim.py` over step 92's draft token logs; `results/decode-next/draft-ahead.txt`).

**When the guess holds:** every verified draft is accepted and the main model's own next token
equals the draft's next guess (drafted token n + 1, when n < 5 were verified).

| prompt | steps | the guess holds | share | est. gain |
|---|---|---|---|---|
| tcp-udp | 142 | 40 | 28% | +2.6% |
| py-lru | 99 | 36 | 36% | +2.8% |
| cpp-matrix | 65 | 13 | 20% | +1.2% |
| sql-report | 77 | 10 | 13% | +0.8% |
| geometry | 85 | 24 | 28% | +1.9% |
| zh-travel | 194 | 44 | 23% | +2.5% |
| ja-email | 174 | 43 | 25% | +2.7% |
| summary-json | 28 | 8 | 29% | +1.8% |
| mean | | | 25% | +2.0% |

**Two limits:**
- **The draft cannot start until the verify is nearly done.** It reads the main model's hidden
  states at layers 37-39 (of 40) for the tokens just verified (`DSpark::seed`), so only the last
  layer, the head and the commit can overlap it: ~5.5 ms of its 8 (the table's figure; with the
  whole 8 ms the mean would be +3.0%).
- **Code steps have no guess.** There the rule mostly verifies all 5 drafts (34 of 65 steps on
  cpp-matrix, 34 of 77 on sql-report) and the draft's block is 5: no sixth token to start from.
  Counting every fully accepted 5-draft step as a hit (a sixth guess always right; it would take
  a second draft call) bounds the mean at +3.3%.

**Not in the table:** issuing a draft takes ~0.65 ms of the host thread, which is driving the
verify's last layers then: without a second host thread that lands on every step and about
halves the gain. The draft would have to move to die 1 whole (a second copy costs ~410 expert
slots), with the hidden states copied over each step.

So ~+1% without a helper thread, up to +2% with one, least on code, for a change to the
speculative loop, the draft's placement and the host's threading. Not built.

A correction found on the way: the model has 40 layers; step 103's and 104's per-step estimates
multiplied per-layer savings by 43 (~7% high). No measurement is affected.

## Step 107: a second draft block before the verify, in shadow (not built)

On code more than half the steps verify all 5 drafts and accept them, and an extra verified token
costs ~3 ms against ~75 ms for a new step. The draft's block is 5, so more drafts per step means a
second draft call from the first block's last token, without the main model's states for those 5
tokens (the draft reads them from layers 37-39 of the verify that has not run yet). Whether such
a block is any good was unknown, so it was measured without using it.

**How:** `DS_DRAFT2_LOG` (a study log like step 92's) runs two extra blocks after every draft and
writes them with their confidences: "gap", the block at its true positions with the window of
main-model states 5 positions behind (`DSpark::draft`'s new `gap` argument), and "shift", as if it
followed the window directly. The log ends with every generated id, so each block is scored
against what the main model went on to emit (`results/decode-next/draft2_sim.py`,
`results/decode-next/draft2.txt`; the logs themselves, `results/decode-next/draft2/*.log`, are not in git,
as step 92's: REPRODUCE has the command; every output identical).

**The second block is weak.** Of the steps that verified 5 drafts and accepted all of them:

| prompt | such steps | second block ("gap"): first draft right | mean accepted of its 5 |
|---|---|---|---|
| cpp-matrix | 29 | 13 | 1.24 |
| geometry | 31 | 17 | 1.42 |
| sql-report | 29 | 12 | 0.97 |
| py-lru | 19 | 9 | 1.42 |

Its first token is right under half the time, where the first block's drafts on these steps were
all right. "shift" is worse (0.4-1.1 accepted). The chat and Chinese / Japanese prompts hardly
have such steps (0-4 each).

**Every rule loses.** With 8 ms for the second draft call on each 5-draft step and the run's
fitted cost per extra verified token: always verifying its first 1, 2, 3 or 5 drafts changes the
8-prompt mean by -1.7%, -1.8%, -2.3%, -4.3%; using its own confidence (the product above 0.5 or
0.7) -0.8% and -1.0%. An oracle that calls the second draft only when it will be accepted, and
verifies exactly that far, gains +3.2% (cpp-matrix +8.0%, geometry +7.7%): the bound, not
reachable with the confidences this block gives.

Not built. The shadow log and the `gap` argument stay (off by default; no change without the
variable). What it says about the draft: its accuracy comes from the main model's states right
behind the block; five tokens behind, it drafts like a much weaker model.

## Step 108: an interactive chat session (`-i`)

A first step towards serving the model: `ds-completion --chat -i` keeps the model loaded and
reads one user line per turn (`scripts/chat.sh` is the launcher with the fast configuration; an
empty line or the end of input ends the session).

**No reset, no recomputation.** Every turn's tokens are appended to the same context, in the
format `encoding.py` renders a chat: the reply's end of sequence, then
`<｜User｜>line<｜Assistant｜></think>`. The runtime's prompt pass already starts at the current
position, so a turn costs its new tokens only, and a turn's reply is what a fresh run on the whole
conversation prints.

**What the context must hold: exactly what was printed.** Two details:
- A reply cut by `-n` ends with a token that was printed but never run through the model; it
  is carried into the next turn's tokens (`pending`).
- With `--spec`, the last step of a reply could commit accepted drafts beyond what was printed
  (after an end of sequence among them, or past `-n`). The step now commits only up to there.
  Output unchanged; it matters only when a session continues.

**Checked** (`results/chat/`): a three-turn session's second and third replies are byte-identical
to fresh runs on the conversation before them as one raw prompt, at `-n 200` and at `-n 12`
(every reply cut); a session without speculation prints the same bytes; cpp-matrix, tcp-udp and
summary-json single-shot are unchanged.

**Limits:** `--chat` only. In thinking mode the reference drops earlier turns' reasoning from
the prompt (`drop_thinking`), which one growing context cannot do, so `-i` refuses `--think`
rather than differ from it; that needs a rewind or reset of the context. One line per turn, no
system prompt; each turn prints the usual statistics lines to stderr.

## Step 109: thinking mode across turns: a mark and a rewind of the context

Step 108's session was `--chat` only. In thinking mode the reference renders a finished turn
without its reasoning (`encoding.py`, `drop_thinking`): `<｜Assistant｜></think>answer`, where the
turn was generated after `<｜Assistant｜><think>reasoning</think>`. One growing context cannot do
that: the reasoning has to leave it.

**`Runtime::mark()` / `rewind()`:** one saved point of the sequence. What a position's forward
leaves behind is small: each layer's window ring (640 slots), the compressor's partial group,
the draft's rings; the compressed caches are keyed by position and read only up to the current
one, so the positions run again rewrite them. mark() copies those buffers on the device (~55 MB
on GPU 0, allocated at the first mark); rewind() copies them back and truncates the sequence,
however far it went.

**The session** marks each turn after its `<｜Assistant｜>`. The next turn rewinds there and runs
`</think>`, the answer's tokens, the end of sequence and its own question: everything before the
mark is kept, only the answer runs again (at prompt speed; its states depended on the reasoning).
A reply cut by `-n` inside its reasoning leaves an empty answer, as the reference renders it.
`scripts/chat.sh --think [--effort max]`; its `-n` default is now 30000 (a reply's reasoning
counts).

**Checked against the reference's own rendering** (`results/chat/chat_ref_prompts.py` calls
`encoding.py` for the conversation before each reply; `results/chat/think/`): the second and
third replies of four three-turn sessions are byte-identical to fresh runs on those prompts:
`-n 600`; `-n 20` and `-n 45` (replies cut inside the reasoning and inside the answer); and
effort max with replies up to 2,500 tokens, where the rewind is longer than the window ring. A
session without speculation prints the same bytes; step 108's chat session and three held-out
prompts are unchanged.

**A bug on the way, worth knowing:** the first version's replies after a rewind were wrong and
differed from run to run. A mark followed at once by a rewind already changed the output, and
only when the ring was among the copies. The cause: `hipMemset` (the zero fill of a new device
buffer) is not ordered with the stream's copies, so it could land after the first save into the
buffer. `hipDeviceSynchronize()` after allocating fixes it. Every other buffer of the runtime is
allocated at startup, long before its first use; not audited.

## Step 110: the second machine, and the runtime on a range of layers

**The second machine** (`shower`, the same Mac Pro: W-3245, 755 GB, four Vega II dies, the same
kernel, ROCm 6.3.4 and compiler; 10 GbE to this one, ping 0.20 ms; its Duo cards have no XGMI
links, which the runtime never uses: every hand-off between dies goes through pinned host memory).
Set up: the branch, the llama.cpp fork's CPU build, the fp8 file copied, the model paths linked.
Every model file has the same sha256 on both (`results/split/sha256.txt`). The held-out set run
on both at the same time: every output identical to the references, long-proof identical between
them; **44.60 t/s here, 44.33 there** (-0.6%; `results/spec-measure-sclk6-cacheloads800cacheperiod8-shower/`,
`-tomb-baseline/`).

**How to use it: capacity before parallelism** (the person's suggestion). The simplest split is
the model in two halves, one per machine: each machine's GPUs then hold experts for 20 layers
instead of 40. One hand-off per step each way instead of one per layer (40 x 0.2 ms would be 8 ms
of latency alone).

**Layer 20 is a clean cut.** From the layer table (`ds-inspect`): layers 0-19 read compressed
caches and index picks from layers 2, 8 and 14; layers 20-39 from layer 20 (index picks from 20,
24, 28, 32, 36; the candidate blocks from 20). The engram layers are 1 and 14; the draft reads
layers 37-39 and uses the head. So the only thing that crosses is the residual stream after
layer 19: per token `h` (4 x 4096 floats) and the next sublayer's mix `pre` (4 floats), 64 KB.

**The runtime on a range** (this step; no network yet): `--layers a-b` loads a contiguous range
(the loader and the expert placement already took a layer list). A range that does not start at
layer 0 takes each token's stream from the host instead of the embedding (`Runtime::stream_in`);
one that does not end at the last layer has no head and copies the stream out (`stream_out`). A
range uploads only its own dense weights, the head only with the last layer, and fetches engram
rows only if it has an engram layer. The draft is refused without the last layers.

**Checked** (`results/split/teacher.txt`): `--teacher` runs a token sequence (a held-out prompt
and its reference output) through verify + commit in chunks of 5 and prints the greedy next id
after every position. The whole model; then layers 0-19 writing the stream to a file
(`--stream-out`); then layers 20-39 reading it (`--stream-in`): the ids are identical at every
position, on cpp-matrix (325), geometry (375) and zh-travel (317), with and without the expert
cache. Single-shot outputs and step 108's chat session are unchanged.

| | whole | layers 0-19 | layers 20-39 |
|---|---|---|---|
| dense fp8 weights on GPU 0 | 6.41 GiB | 3.34 | 3.06 |
| small weights (head, norms, ...) | 1.75 GiB | 0.29 | 1.47 |
| expert slots (no draft) | 6,523 | 6,794 | 6,747 |
| share of its expert pairs on GPUs | 42.5% | 88.5% | 87.9% |

**A first signal, not a measurement:** the same test's time for all positions, whole against
front + back: cpp-matrix with the static placement 7.82 s against 1.83 + 1.87; zh-travel with
the cache 4.91 s against 1.88 + 1.97 (-22%). Teacher-forced, fixed chunks, no draft, no network.

Next: two processes over TCP (first both here, front on dies 0-1 and back on dies 2-3), the
speculative loop driving the front, the back holding the head and the draft.

## Step 111: the two halves as two processes, then on two machines: 44.6 -> 50.9 t/s

**The link** (`hip/peer.h`): the front (layers 0-19) drives and is used as the whole model would
be: `--layers 0-19 --remote host:port`. The back (`--layers 20-39 --dspark .. --serve port`)
holds the head and the draft and serves one front. Every message mirrors one `Runtime` call, so
both sides run the code one process runs, in the same order, and the floats cross as they are in
memory:

| message | to the back | answer |
|---|---|---|
| CHUNK | prompt tokens and their stream | none: the front goes on with its next chunk |
| DECODE | one token, its stream | the logits, if wanted |
| VERIFY | a chunk and its stream | the next ids |
| COMMIT, MARK, REWIND | a number | none |
| DRAFT | the anchor | the draft's ids and confidences |

Plain blocking TCP with `TCP_NODELAY`. In `Runtime`, prefill / decode / verify / commit / draft /
mark / rewind do their part and hand the rest to the back (`peer`), so `ds_completion.cpp`'s
loops (speculation, `-i` sessions) are unchanged. A speculative step is two round trips: the
draft, then the verify (64 KB per token up, a few ids down).

**First on one machine** (front on dies 0-1, back on dies 2-3, localhost): cpp-matrix
byte-identical at 48.7 t/s (no capacity gained there, by construction). It found an old bug: with
the expert cache, the runtime reset the current device to the home die's *index* instead of its
device (a local `home` shadowing the outer one): right only with `--gpu 0`, which is all that had
ever run.

**Then on two** (`scripts/split-measure.sh`, `results/split/two-machines/`; front here, back on
`shower`, sclk 6 on both, against the one-machine run of the same day):

| prompt | one machine | two | | front's expert uses on GPUs: one -> two |
|---|---|---|---|---|
| tcp-udp | 39.04 | 42.10 | +7.8% | 87.4% -> 99.0% |
| py-lru | 46.45 | 53.67 | +15.5% | 85.3% -> 98.9% |
| cpp-matrix | 54.30 | 62.89 | +15.8% | 85.7% -> 98.8% |
| sql-report | 46.81 | 57.67 | +23.2% | 81.9% -> 97.9% |
| geometry | 54.55 | 61.01 | +11.8% | 86.1% -> 98.3% |
| zh-travel | 34.17 | 35.00 | +2.4% | 87.8% -> 98.5% |
| ja-email | 37.05 | 38.01 | +2.6% | 87.0% -> 98.2% |
| summary-json | 44.43 | 56.43 | +27.0% | 75.4% -> 96.9% |
| **mean** | **44.60** | **50.85** | **+14.0%** | |
| long-proof | 56.99 | 57.00 | 0% | 96.9% -> 99.7% |

Every output is byte-identical to the references (long-proof to the one-machine run).

**What it shows:**
- The gain is what the CPU's experts cost. Where a third of them ran on the CPU and the steps
  are long (summary-json, sql-report), +23-27%; the long proof, whose cache already had 97% on
  GPUs after 3,000 tokens, gains nothing.
- Steps of 1-2 tokens gain little (zh-travel, ja-email +2.5%): their fitted verify cost is the
  same on one machine and two (28 + 19 n ms), so they were not waiting for the CPU; the dense
  work, now on two machines in turn, is what they take.
- The network is not visible: the front's waits add up to the back's own work (cpp-matrix: 152
  answers, 2.85 s waited, 2.85 s worked there; 29 MB sent in the whole run).
- No parallelism yet: the two machines take turns, and each one's dense part runs on one die.

A session leaves the back when it ends (one front per back process); `scripts/chat.sh` is still
one machine.

## Step 112: one trace of both machines

Each half already wrote its own `--trace`; two files on two clocks do not show a step.

**The clocks:** every answer from the back carries its clock when the request had arrived and
when the answer left; with the front's own times around it that is one sample of the offset, as
NTP takes them: offset = ((t2 - t1) + (t3 - t4)) / 2, delay = (t4 - t1) - (t3 - t2). Samples come
from the small exchanges only (HELLO, a new SYNC message sent 8 times at the start and the end of
a traced run, DRAFT): a verify's 64 KB per token upward would bias it. The front's trace keeps
them (`peerSync`), each trace its zero on its machine's monotonic clock (`clockEpochUs`).

**The merge** (`scripts/trace_merge.py FRONT BACK -o MERGED`): a line through the offsets over
time (the clocks drift), the back's events moved onto the front's clock, the two as processes 1
and 2 of one Chrome trace. The first fit used the samples with a delay up to the median and came
out with residuals of ~90 us: most samples are drafts, and a draft request waits in the back's
socket behind the commit sent just before it, a wait on one side only. Keeping only the samples
within 1.5x of the least delay (the SYNCs, and drafts that found the back idle): 15 samples,
least delay 68-70 us, **residuals ~1 us rms**, drift under 1 ppm (the offset itself: the back's
clock 2.09 s behind, as both machines booted together).

`TRACE=1 scripts/split-measure.sh` runs both halves traced, fetches the back's file and merges:
`results/trace/two-machines/{cpp-matrix,tcp-udp}.json.gz` (ui.perfetto.dev). New spans: the
front's "send to the back: ..." and "waiting for the back"; the back's "receive the stream: ..."
and a "messages from the front" track with one "serve ..." span per message. Every answered
message's work on the back lies inside one of the front's waits (134 and 257 of them), and an
answer reaches the front ~155 us after the back finished it.

**A 6-token code step on two machines** (cpp-matrix, medians of 37 steps, traced):

| part | ms |
|---|---|
| the front's layers 0-19 | 36.4 |
| sending 6 tokens' stream (384 KB) / the back receiving it | 0.4 |
| the back's layers 20-39 and the head | 36.1 |
| the ids back | 0.14 |
| the draft (the back's 8.1, behind its commit of ~1 ms) | 9.35 |
| **the cycle** | **82.7** |

Against 92.9 on one machine (step 105: 84.8 + 8.1). Neither half waits for the CPU's experts in
these steps (0 ms on both; it was 26.7). A 2-token chat step (tcp-udp): 21.4 + 22.4 + the draft
8.8 = 53.7 ms. What the trace shows next: the back's commit seeds the draft before the draft
call can start (~1 ms per step, visible as the front's 9.1 ms wait for an 8.1 ms draft), and at
any moment one machine is idle.

Traced runs: cpp-matrix 59.0 t/s (62.9 untraced), tcp-udp 38.2 (42.1); outputs identical.

**Two guards added while reviewing the documents for both setups:**
- A range must contain what its layers read (their compressed-KV source, index source and the
  candidate source): `--layers 10-39` is now refused ("layer 10 reads layer 8's state") instead
  of reading memory that range never allocated. For this model that leaves layer 20 as the only
  cut inside the stack.
- `--spec` without `--placement` aborted since step 97 (the early copy of the dies' results used
  a stream that only exists with a placement); never run in a measurement. Fixed: same output as
  plain decoding there too.

The documents now describe both setups: this file's status, QUICKSTART (what the second machine
needs, the two commands, what to expect, limits), OVERVIEW (results, what is next, the code
map), TIMING (a two-machine section), PLAN (today's numbers; "Two machines": what the split is
and is not, the ideas after it), REPRODUCE (the second machine).

## Step 113: review round 1 (the code of steps 108-112, the documents for both setups)

No performance work: two independent reads, one of the code added in steps 108-112, one of the
documents by a newcomer setting up two machines. Neither found a wrong output; outputs are
unchanged (checks below).

**Code:**
- A range of layers alone is refused: `--layers 0-19` without `--remote` ran, got no logits and
  printed token 0 `-n` times; `--layers 20-39` without `--serve` aborted.
- Errors after startup end the run with `error: ...` and status 1 (`main` is a try around the
  old body): a peer that left or a refused connection aborted the process, and its trace was lost.
- The halves check each other: HELLO carries a version of `hip/peer.h`; every request must have
  its type's size (token messages at most 512 tokens, the others no payload) and every answer the
  size asked for. Two different builds now give an error, not a read past a vector.
- The back takes the front's `in_prefill` with CHUNK and DECODE. It had prompt chunks as
  generation and plain decode steps as prompt: its profile, trace spans and routing log only.
- `stream_out` holds the last forward's tokens (it appended, with four clears before forwards).
- `scripts/split-measure.sh`: the back's PID is kept: a back that does not start is reported
  and stopped, the script waits for the back's process to end (it waited for a log line, so the
  next back could start while the last still held GPU memory); an unknown prompt name no longer
  starts a back that nobody connects to; a failed merge keeps the two traces.
- Smaller: a session's `--spec-log` kept only the last turn; `draft_shadow` on a front; socket
  errors say why, EINTR is retried, keepalive is on; `trace_merge.py` takes `-o` anywhere and
  does not divide by zero when the least delay is ~0 (one machine).

**Documents:** QUICKSTART has a "Two machines" section a newcomer can follow (commands with their
`cd` and `$M`, what must agree between the halves and what is checked, the port, what failures
look like, sessions), the commands to set and read the GPU clock, a short glossary, and no XGMI
requirement; the headline numbers say they were measured with `--cache-wait`; rows at sclk 2 are
marked. REPRODUCE's open items about the llama.cpp fork are settled (the PR head; the `hip.h`
line is not needed: the second machine was built without it). TIMING, OVERVIEW and PLAN: stale
"today" numbers, the experts' share on GPUs as measured (88% / 42%), the gain per kind of prompt
with chat in it (+8%).

**Checks:** a 3-turn thinking session (`results/chat/think/th45.*`) on one machine and on two:
the saved bytes; cpp-matrix and tcp-udp on one machine 54.31 and 39.13 t/s, sql-report on two
57.44, outputs the same. The merged traces of step 112 were collected again with the back's
flag right (`results/trace/two-machines/`: 60.3 and 38.1 t/s traced; the clock fit 14 samples,
residuals 0.7 and 1.4 us, drift 16 and 14 ppm this time against under 1 in step 112's runs).

Left open by the review: one order for the rows of REPRODUCE section 6 and PROCESS's log; one
owning document per number; the plain and the speculative loop in `main` keep their end-of-reply
state differently; step 107's shadow log sits in the speculative loop.

## Step 114: review round 2 (the runtime, the kernels and loaders, the log and the repository)

Three independent reads: `hip/rt.cpp` and `rt.h`; the kernels, the CPU expert code, the draft and
the model loading; the documents' long tail and the repository. None found an input on this
model that gives wrong or batch-dependent bits. Two things the code relies on were checked
through and hold: speculation's rollback (the window ring, the compressors' partial groups, the
compressed caches, the token history, the draft's rings), and the cache's rule that a kernel
never reads a slot being loaded (a table entry names a slot only after its load has finished,
and a slot is reused only after its entry is gone and the table uploaded at a step boundary).

**Fixed in the runtime:**
- A decode step without logits (every step of a front, and prompt tokens run one by one)
  skipped the step's bookkeeping: the profile's misses piled up across steps and its spans never
  reached the trace. One `step_stats()` on both paths.
- `dalloc` waits for its zero fill. `verify()`'s backup buffers were allocated on first use and
  written at once, the pattern behind step 109's race; no run had shown it here.
- An exception in an expert-loader thread ended the process without a word: now with its message.
- The model's shapes are checked against what the kernels are written for (hc_mult 4, at most
  6 experts per token, the widths' multiples): another model stops at startup, not in a kernel.
- The fp8 file's open and map are checked. Removed: `hist` (a copy of the whole token history
  every step, never read), `Runtime::moe`. Three comments that described older code.

**Fixed in the kernels and loaders:**
- `gemv_m` (the f32 matrix-vector product with a split launch) issued three launches for two:
  the third computed the last blocks again. The same bits; one launch less per call (the
  compressor's projections at 1-3 tokens).
- The draft's loader checks the file, each tensor's end against the file's size (a truncated
  draft crashed in the upload) and the noise token's id; `draft()`'s input block lives in members
  (its asynchronous copies outlived the locals they read).
- `Model::load`: o_groups and the shard count are checked before use.
- `fp8_gemv` with an input count outside 1..8 throws (it launched nothing); the top-k's
  histogram walk is bounded. Removed: two unused `dot_row_multi`, `bf16f`.

**Documents and repository:** REPRODUCE section 6 and PROCESS section 4 in one order
(ascending); REPRODUCE says which files are not in a clone (`*.log` is ignored: 339 files, two
directories empty) and lists the `DS_*` switches; QUICKSTART's options complete (`-m`,
`--max-ctx`, `--log-routing`, `--no-quant`); NUMERICS has `--canon`; PLAN's items that later
steps closed are marked; the "Steps" list at the top of this file is named the original plan.

**Tried, not kept:** stating `-ffp-contract` in the Makefile. The fast kernels' bits depend on
which a*b+c the compilers fuse, and no flag says so. With the modes I took for the defaults
(g++ fast, hipcc fast-honor-pragmas) every object's hash changed, and hipcc's objects differ in
hash for every value of the flag at the same size, so the hash cannot tell which mode is the
default. Not changed without knowing.

**Checks** (sclk 6): the 8 held-out prompts, outputs the same, 39.20 46.51 54.86 47.29 55.19
34.37 37.10 45.15 t/s (mean 44.96; 44.60 this morning, not a `spec-measure.sh` run); cpp-matrix
with `--profile` the same output; the 3-turn thinking session the saved bytes; the fast path
(no `--canon`) with and without `--spec` the same 1,105 bytes; a 2,905-token prompt in chunks of
512 and of 7 the same continuation; cpp-matrix and summary-json on two machines 62.55 and
55.60 t/s, outputs the same.

**Left open** (found, not changed):
- `--cache-loads` without `--canon` is accepted and makes the output depend on load timing.
- Three places shift member pointers during a forward (`early_gather`, `set_half`, the h / h2
  swap): right today, and a throw in between leaves them shifted.
- `decode()` and `forward_chunk()` repeat their prologue and epilogue; `mark` names both the
  session's save point and the profiler's events; the constructor, `moe_launch` and `moe_finish`
  are long, with natural seams.
- About 200 lines of finished studies run next to the hot path (the lookahead and prefetch
  predictions of steps 52 and 63, the host-bound estimate of step 65) and six experiment
  switches in the kernels.
- The fake-quant rule exists in several copies that must stay bit-identical (three files' helpers,
  six kernels' absmax-scale-round block); four fp8 GEMV kernels repeat one prologue.
- Loading checks structure, not content: a flipped byte inside a tensor passes (the checksum
  manifest is the only guard). NaN logits would become an out-of-range expert id.
- A trace is held in memory until exit; the destructor leaves a few events and buffers unfreed.
- The compilers' contraction modes (above). The `*.log` files outside git.

## Step 115: testing from here on: a run as one record (`--record`, a first version)

The project's checks are ~60 scripts and programs, each written for one step; the gate scripts
repeat one loop (prompts x configurations, numbers read off stderr with `grep`, outputs compared
with `cmp`). For the changes expected next (small performance work, refactoring, other draft
modules, parameter tuning, a new checkpoint) the plan is one primitive instead: a run writes what
went in (arguments, environment) and what came out (values that must repeat, counters that
vary) as one record, and a test is a run plus a check of its record. The idea, its scope and the
next steps: [docs/record.md](record.md).

**This step is only the start of it:** `ds-completion --record FILE` writes one JSON object with
ten of the options as parsed, the prompt's token count and a hash of its token ids (FNV-1a, 64
bits), and the load time and prompt / generation speeds of the `# ...` lines. One prompt only
(refused with `-i`, `--serve`, `--teacher`). The file is opened before the load and written after
the generation, so a run that does not finish leaves it empty. No hash of the output yet, no
script changed, no test uses it.

**Checked** (`results/record/first.txt`; sclk auto, not the 1 this step first said: step 116): the same prompt twice. Both records parse, carry
the same prompt hash and repeat the stderr lines' numbers; the outputs are the same bytes;
`--record` with `-i` exits with 2 and an unwritable path with 1, both before the load. The CPU
build compiles with it (not run).

**The fastest single-machine settings, recorded once** (`results/record/fast.txt`; cpp-matrix, 300
tokens): 47.7 t/s (54.86 at sclk 6 in step 114; here the cards were on auto, and step 116's repeat gave 55.05), the output the stored one's bytes. The
record holds neither the clock setting nor the speculation and cache numbers stderr prints (68
verify steps, 4.41 tokens per step, 85.7% of expert uses on GPUs): the next fields.

**Seen on the way:** the page cache was empty before the first run. Load 92.7 s against 4.0 s,
and generation 13.72 against 14.52 t/s after it (one pair). The record shows a cold run only
through a load time one has to recognise: bytes read from storage per phase are among the next
fields.

**Not measured:** the option's cost. It adds one hash of the prompt's ids before the load and one
file write after the generation, nothing inside a step.

## Step 116: the GPU clock setting in the record; step 115's "sclk 1" was auto

**The field.** `--record` now writes `"env": {"gpu_sclk": [...]}`: one value per AMD card, in
card order. A card whose level is fixed (`power_dpm_force_performance_level` = `manual`) gives
its level, 0-8; any other mode gives the mode's name, so a card on auto reads `"auto"`. On auto
the driver moves the level with the load, and the level at any one moment says nothing about
the run. Read once, as the run starts, before the load. It is not read again at the end: a
setting that changed and changed back would look unchanged, so "the same at both ends" would
promise more than it checks. A file that cannot be read gives `"unknown"`. The record's helpers
moved to `apps/record.h`.

**A correction to step 115.** Its results files and text said "sclk 1". The cards were on auto:
the machine had rebooted two hours earlier, which resets the setting, and 1 was the starred
level of `pp_dpm_sclk` read while the GPUs idled. The headers and the step's text are corrected
in place, each with a note. The scripts' headers read the level the same way
(`grep '\*' .../pp_dpm_sclk`, nine scripts): after a reboot they too would name an idle level.
Not changed here.

**Checked** (`results/record/sclk.txt`):
- The reader on a copy of the sysfs files with six made-up cards (fixed at 6, auto, another
  vendor, no mode file, fixed with no starred line, fixed at 0): `[6, "auto", "unknown",
  "unknown", 0]`. The fixed-level case was not run on the real cards (setting it needs root).
- On this machine a short run records `["auto", "auto", "auto", "auto"]`; its output and prompt
  hash are step 115's.
- Does auto move during a run? The fastest single-machine settings again, each card's level
  sampled every 0.1 s: every card was seen at levels 1, 5 and 8. So one number could not have
  described the run.

**Not explained:** that run gave 55.05 t/s, step 115's 47.70, with the same command and the
same clock mode; the output is the stored one's bytes both times. Two runs. Step 115 put its
47.7 down to "sclk 1 against 6": the clock was not at 1, and the clock mode does not separate
the two runs. A candidate, not checked: storage reads during generation. The page cache held
291 GB before the first run and 300 GB after the second, of ~488 GB of model files, so part of
the model was uncached in both (the engram table is read as the tokens need it). Bytes read
from storage per phase, the next field planned, would say what each run read.

## Step 117: bytes read from storage per phase in the record; a new prompt reads the engram table from disk

**The field.** `--record` now writes `"read_bytes": {"load": .., "prompt": .., "gen": ..}`: what
the process had fetched from storage during the load, the prompt and the generation
(`/proc/self/io`, `read_bytes`, read at the four boundaries; `null` if the file is not there).
It counts this process only, all its threads, and only reads that reached the disk: a read
served from the page cache adds nothing. So zero means "this phase did not wait for storage",
which is the question for a timing. It does not say how much of the model is cached: a run that
does not touch the uncached part reads nothing.

**Why not major page faults:** tried first on a 64 MiB file on the model's filesystem. Evicted
and read through a mapping, `read_bytes` grew by 64.0 MiB and the major faults by 1 (readahead
brings the rest); through `pread` 64.0 MiB and 0 faults; cached, 0 either way; read by another
thread, the process's counter still has it.

**Checked** (`results/record/read-bytes.txt`; sclk 6, every record says `[6, 6, 6, 6]`):

| run (32 tokens, experts on the CPU) | load | prompt | generation | |
|---|---|---|---|---|
| the usual prompt, twice | 0 | 0 | 0 | |
| after the fp8 file (6,843,437,120 bytes) was evicted | 6,843,441,152 | 0 | 0 | load 6.6 s against 4.0; the file plus one page |
| a prompt not run before | 0 | 5.1 MB | 7.3 MB | prompt 18.6 t/s, generation 13.9 |
| that prompt again | 0 | 0 | 0 | 20.3 and 14.5 t/s |

**What it showed: a new prompt reads from the disk while it generates.** The page cache held
~300 GB of the ~488 GB of model files (a reboot that afternoon; the model was not read through
afterwards): the CPU's experts, which the load maps, and the dense weights. Most of the 189 GB
engram table was not in it, and its rows are read as the tokens need them. With the fastest
settings (300 tokens):

| prompt | read during generation | t/s |
|---|---|---|
| cpp-matrix (run three times since the reboot) | 0 | 54.62 |
| sql-report, not run since the reboot | 72 MB | 41.17 |
| sql-report again | 0.8 MB | 46.77 (46.63 in step 104's held-out run) |

12% for 72 MB, one pair. The second run still read 0.8 MB: its rejected drafts were not the
first run's (3.90 tokens per step against 3.80), and a draft's tokens fetch rows too.

**Step 116's 47.70 t/s:** that was the first run of cpp-matrix after the reboot, 13% under the
three since (55.05, 54.55, 54.62). It was made before this field existed, so its reads were not
recorded; cold engram rows fit its size and its place, and nothing else found does.

**Step 116's fixed level, on the real cards:** set to 6 by hand before these runs. The record
says `[6, 6, 6, 6]`, and sampled every 0.1 s through one run every card was at 6 in all 1,029
samples.

**For measurements:** QUICKSTART's `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` after a reboot
matters for more than the first run: until the engram table is cached, every new text is slow
once. A check on records can now drop a run whose `read_bytes.gen` is not 0 by itself.

## Step 118: the speculative loop's totals in the record

`--record` with `--spec` now writes a `"spec"` object; without it, `"spec": null`.

| field | what |
|---|---|
| `steps` | verify steps (each emits the accepted drafts and the main model's own next token) |
| `plain_steps` | those of them that made no draft (the backoff after a step that verified none) |
| `accepted` | drafts the main model agreed with, all steps together |
| `verified` | the steps by the number of drafts verified, from 0: `[0, 3, 7, 12, 10, 34]` is 34 steps with 5 |
| `draft_ms`, `verify_ms` | the time in `draft()` and in `verify()`, all steps together |

Totals, not ratios: drafts accepted per step, tokens per step and ms per draft, which stderr
prints, are divisions a reader of the record does. `verify_ms` is new: the loop had the time of
each verify step (it fits the draft-length rule's cost line with it) but kept no sum.

**Checked** (`results/record/spec.txt`; sclk 6):
- No `--spec`: `null`, the output unchanged.
- The fastest single-machine settings, cpp-matrix, 300 tokens: 66 steps, 233 accepted,
  `[0, 3, 7, 12, 10, 34]`, draft 514.5 ms, verify 4,904.9 ms; 54.73 t/s, the output the stored
  one's bytes. From them 3.53 accepted and 4.55 tokens per step and 7.8 ms per draft, and the
  histogram: the stderr lines' numbers.
- `--spec-n 5`, 40 tokens: 7 steps, all in the histogram's last place, 33 accepted; a prefix of
  the 300-token output.

**What the two sums show:** draft + verify are 5,419 ms of the generation's 5,482. The rest of
the speculative loop (the rule, the commit, printing) is 63 ms, 1.1%.

**The object also proves the feature ran:** `"spec": true` among the arguments with `steps` 0
would be a run that asked for speculation and did none.

## Step 119: the routed experts' uses, placement and cache totals in the record

`--record` with `--rt` now writes an `"experts"` object (`null` on the CPU path):

| field | what |
|---|---|
| `gpu_uses`, `cpu_uses` | routed-expert uses by where they ran, the whole run: the prompt, the generation, and the drafts a verify step rejected (they were computed) |
| `dies`, `placed` | the dies that hold experts and how many each held after startup; both empty without `--placement` |
| `cache` | with `--cache-loads`: `rebalances`, `swaps_planned`, `loads_started`, `loads_published` so far; else `null` |

The runtime gained one accessor for it (`Runtime::expert_stats()`): the placement and the cache
are private there. Totals again: the share of experts on GPUs is `gpu_uses / (gpu_uses + cpu_uses)`.

**Checked** (`results/record/experts.txt`; sclk 6, nothing read from storage in any run), each
against the stderr lines of the same run:

| run | uses on GPUs / CPU | placed | cache |
|---|---|---|---|
| no placement, 32 tokens | 0 / 12,240 | `[]` | `null` |
| study30 on dies 1-3, `--canon`, plain, 40 tokens | 9,140 / 6,220 (59.5%) | `[1765, 1765, 1765]` | `null` |
| the fastest settings, 300 tokens | 72,804 / 12,156 (85.7%) | `[801, 1765, 1765, 1765]` | 32 rebalances, 3,857 swaps planned, 3,686 loads started, 3,621 published |

The cache's numbers at the record's time equal the ones the runtime prints when it shuts down:
loads are started and published only between forwards, and the record is written after the last.

**Outputs:** the first as before; the third the stored one's bytes (54.56 t/s). The second, on
three dies without speculation, printed the same 40 tokens as step 118's run on four dies with
`--spec --spec-n 5`: `--canon`'s promise, seen in passing.

**One whole record is kept as an example:** `results/record/example.json`, the third run's file
as the program wrote it (its command, commit and clock level are in `results/record/README.txt`,
since JSON holds no comment).

**Proof that the features ran** is now in the record for all three: `spec.steps`, `placed`, and
`cache.loads_published` (a cache that never got to load would show 0 next to `"cache_loads": 800`).

## Step 120: the tuning knobs among the record's arguments

`args` gains the four settings a tuning sweep varies and the record did not hold:
`cache_window`, `cache_min`, `cache_wait` and `spec_n` (next to `cache_loads` and
`cache_period`, there since step 115). Two runs of a sweep over them had identical `args`.

**Arguments are written as they took effect:** an option that was not given has its default in
the record (`"cache_window": 256`, `"cache_min": 2`, `"cache_wait": false`), not a missing key.
Every record then has the same keys, and a default that changes in the code shows as a changed
value. `"spec_n": 0` is the default too, and means the number of drafts verified is chosen per
step; 1-5 fixes it. The values are there whether or not the feature is on: with
`"cache_loads": 0` the cache settings were not used.

`cache_wait` matters for reading a record's `load_s`: with it the load waits for the experts'
memory to be registered (~117 s in the runs of steps 118-119, against 10-14 s for their runs
with a placement and no cache).

**Checked** (`results/record/knobs.txt`): a run without any of them records the defaults; a run
with `--cache-period 8 --cache-window 128 --cache-min 3 --cache-wait --spec-n 4` records those;
both print the same bytes as before.

## Step 121: the options and switches that perturb timing, in the record

A record of a run with `--profile`, or with an experiment switch set in the environment, is not
a clean timing, and nothing in it said so. Now:

- `args` has `profile`, `trace`, `spec_log` and `log_routing`: on or off (not their files' names).
- `env` has `ds`: every environment variable whose name starts with `DS_`, with its value, by
  name; `{}` when none is set. The program reads twelve of them (REPRODUCE.md lists them:
  `DS_FP8_PAD`, `DS_STALL_SPLIT`, the logs of the studies, ...). All of that prefix are written,
  read or not, so a misspelt switch is in the record too; the record cannot say it was ignored.

**Checked** (`results/record/timing-options.txt`; sclk 6): a short run with nothing on records
four `false` and `{}`; with everything on, four `true` and `{"DS_FP8_PAD": "0", "DS_MISSPELT":
"a \"b\"", "DS_STALL_SPLIT": "0"}`. The same output both times.

**What two of them cost,** the fastest single-machine settings on cpp-matrix, every output the
stored one's bytes:

| run | t/s | read during generation |
|---|---|---|
| clean | 55.11 | 0 |
| `DS_FP8_PAD=0` | 50.69 | 741 KB |
| `--profile` | 50.33 | 0 |
| clean | 54.81 | 0 |
| `DS_FP8_PAD=0` | 50.60 | 0 |

- **Clean runs repeat within ~1%:** six of them today at sclk 6, 54.55-55.11 t/s (mean 54.73).
- **`--profile`: -8.0%** (one run). A profiled run is for its breakdown, not for its speed.
- **`DS_FP8_PAD=0`: -7.5%** (two runs). Step 101 measured the padding at +2.4%, at 42.5 t/s;
  why it is worth three times that now was not looked into (candidates: step 104's kernel reads
  the same matrices in fewer, wider launches; the draft's time rose too, 514 -> 547 ms).
- The first `DS_FP8_PAD=0` run read 741 KB from storage: it took 65 verify steps where the clean
  runs take 66, so it verified other drafts and needed engram rows no run had fetched. Its
  repeat read nothing and ran at the same speed: reads of that size do not show here (step
  117's 72 MB did).

This is the first of the faults put in on purpose (docs/record.md): a slowdown of known origin,
and the record shows both that the run was slower and why.

## Step 122: the cache's loads per die in the record; step 86's rule switched off on purpose

**A review of step 86 first** (the person's question: what exactly was happening there). The
cache loaded experts into every die. Die 0 also runs the dense part and the draft, and a draft
that overlapped a load into die 0 took 14-18 ms instead of 8 (36 of 36 slow drafts, none of the
fast ones). Since step 86 the cache picks no victim on the home die, so nothing is loaded there.
What the review found:

- The cause is known by correlation and by removing it, not as a mechanism: why a load on a
  low-priority stream doubles the die's small kernels was not looked into (two guesses, the
  host and the unpack kernel, had been ruled out).
- "The loads had also slowed die 0's own dense work" was an inference from +8.7% held-out
  against +4-5% in the traces.
- The price is a home die whose ~800 slots (13%) never adapt. In step 86's own held-out runs the
  share of expert uses on GPUs fell on 7 of 9 prompts, by up to 0.6 points (cpp-matrix 86.0 ->
  85.5%, geometry 86.8 -> 86.2%).
- Nothing outside a `--trace` showed that the rule holds.

**The field.** `experts.cache` gains `loads_started_by_die`: the loads started per die, in the
order of `experts.dies` (the cache counts them in its statistics). The home die's place holds 0.

**The rule switched off on purpose** (`results/record/loads-by-die.txt`; a second binary with
one line changed so that no die is fixed, not kept; sclk 6; the fastest settings on cpp-matrix;
every output the stored one's bytes; nothing read from storage):

| | this build | this build | fault | fault |
|---|---|---|---|---|
| `loads_started_by_die` | `[0, 1197, 1219, 1270]` | `[0, 1200, 1213, 1273]` | `[486, 1045, 1024, 1095]` | `[492, 1046, 1038, 1093]` |
| t/s | 54.48 | 54.64 | 49.85 | 50.47 |
| `draft_ms` | 515.5 | 514.8 | 779.2 | 762.3 |
| `verify_ms` | 4,928.6 | 4,914.0 | 5,119.3 | 5,063.1 |
| uses on GPUs | 85.7% | 85.7% | 85.8% | 85.8% |

- **The record shows the fault three ways:** 13% of the loads on the home die, -8.1% in speed,
  and where the time went.
- **Step 86's rule is worth 8% today** (it measured +8.7% then, at 37.6 t/s).
- **Both halves of step 86's explanation are now measured.** The drafts: +50%, 7.8 -> 11.8 ms
  each. And die 0's own work: the verify steps take 3.5% longer (~2.6 ms each) with no more
  expert uses on the CPU, which is the part that was an inference. Of the ~480 ms the faulty
  runs lose, ~255 are the drafts and ~170 the verify steps.
- **The price of the rule is small here:** with every die adapting, 0.1 points more of the
  uses ran on GPUs.

## Step 123: the draft's routed experts in the home die's expert pool (nothing else changes)

**Why.** A draft expert and a main-model expert are the same thing: three mxfp4 matrices of the
same shapes (5120 x 2304 twice, 2304 x 5120), 18.8 MB, the same layout on the GPU and the same
kernels. They differ in number and in how they are kept: the main model has 15,360 (40 layers
of 384, 6 per token), ~6,100 of them in the dies' slot pools and under the cache; the draft has
384 (3 layers of 128, 3 per token), all in arrays of their own on the home die, 7.2 GB that the
main model's ~800 slots there cannot use. A prompt uses ~120 of the 384 (step 87). To give the
unused ones' memory to the main model, the draft's experts first have to be slots.

**This step only moves them.** With an expert pool on the home die, the draft no longer uploads
its experts: the runtime sizes that pool for the main model's slots, then the draft's 384, then
any spare ones, and the draft loads its experts into its range (`DSpark::load_experts`). A draft
layer's experts are the pool's view moved to the layer's first slot, so the kernels still take
expert ids and no kernel changed. All 384 stay resident; nothing is evicted, masked or cached.
Without a pool on the home die (no `--placement`, or `--moe-dies` without it) the draft keeps
its own arrays, as before.

**Checked** (`results/draft-pool/in-pool.txt`; sclk 6), the old build against the new with a
fixed draft length (`--spec-n 5`, so the drafts do not depend on timing), cpp-matrix, 120 tokens:

| placement | the draft's experts (new) | output | drafted and emitted ids, 26 steps |
|---|---|---|---|
| all four dies | in the pool | same | same |
| dies 1-3 | its own arrays | same | same |
| none | its own arrays | same | same |

The ids are the stricter test: the output only says the main model is unchanged, the drafted
ids that the draft is. The three configurations print the same bytes as each other.

**Memory** on GPU 0 moves from one line of the startup table to another: "draft" 8.02 -> 0.79
GB, "expert slots" 15.07 -> 22.31 GB, free 0.77 -> 0.76. The main model got one slot more on
the home die (802, was 801): one allocation where the draft made eighteen.

**Speed** with the fastest settings, twice: 54.90 and 54.81 t/s (the old build that evening:
54.48-55.11 in eight runs), the draft 513 and 516 ms (512.9-515.5), the stored output's bytes.

**Next, not built:** leaving draft experts out. Two things to decide first. What a draft expert
that is not resident means: the draft has no CPU path; masking such experts in its router would
let it choose among the resident ones, which cannot change the output, only how often drafts
are accepted. And how the resident set follows the prompt: loads into the home die during a
run are what step 86 removed, 8% in step 122's measurement.

## Step 124: fewer draft experts, measured before any is left out

Step 123 made the draft's experts slots. Leaving some out would give their memory to the main
model; two ways were on the table: keep fewer of them, or requantize all of them to ~2 bits.
Keeping fewer was tried first, as a measurement of what the draft loses.

**The switch** (a study, like the other `DS_*` ones): `DS_DRAFT_KEEP=FILE` lists the (layer,
expert) pairs the draft's router may choose; every other expert gets a bias of -1e30 at load.
`k_router` picks the top 3 of scores + bias and takes the weights from the scores alone, so the
kept experts act as before. Every expert stays loaded: this measures the loss only.
`results/draft-pool/draft_keep.py N LOG...` writes such a file from step 87's routing logs: the N most
chosen pairs, the three layers ranked together.

**The sweep** (`results/draft-pool/draft-keep-sweep.sh`, `results/draft-pool/keep/`; sclk 6): the 8 held-out
prompts, study30, `--canon`, no cache, `--spec-n 5`. With every draft verified, the drafts
accepted per step measure the draft alone and do not depend on timing. Sets chosen without the
prompt (`loo`: from the other seven prompts' logs) and sets chosen for it (`own`: from its own
log, the best of their size). All 64 outputs are the stored ones' bytes.

| set | kept of 384 | drafts accepted | verify steps | tokens per step | steps |
|---|---|---|---|---|---|
| all | 384 | 1,592 | 653 | 3.44 | |
| loo-seen: every pair the others chose | 242 | 1,582 | 663 | 3.39 | +1.5% |
| loo-192 | 192 | 1,516 | 729 | 3.08 | +11.6% |
| loo-144 | 144 | 1,487 | 758 | 2.96 | +16.1% |
| loo-96 | 96 | 1,453 | 796 | 2.83 | +21.9% |
| own-144 | 136 | 1,591 | 654 | 3.43 | +0.2% |
| own-96 | 96 | 1,583 | 662 | 3.39 | +1.4% |
| own-48 | 48 | 1,567 | 683 | 3.29 | +4.6% |

(Steps and accepted drafts are one number: a step emits its accepted drafts and one token, and
the tokens are the same in every run. Per prompt: `results/draft-pool/keep/table.txt`.)

- **A third of the draft's experts can go at once:** the 142 pairs that seven other prompts
  never chose cost 1.5% more steps, nothing on 5 of the 8 prompts (geometry loses most, 3.61 ->
  3.27 accepted per step).
- **Below that a set chosen without the prompt fails on the prompt it does not know.** At 192,
  ja-email's accepted drafts halve (1.53 -> 0.73) and geometry falls to 2.89: the set holds 77%
  and 67% of the choices their own logs made, 95% of cpp-matrix's. Seven prompts of 300 tokens
  are a small sample to choose from; ja-email is the only Japanese one.
- **A set chosen for the prompt can be very small:** 96 of 384 cost 1.4% more steps, 48 cost
  4.6%. The draft's experts are concentrated per prompt, as step 87 said; this is what that is
  worth in accepted drafts.
- The `own` sets are an upper bound, not a method: they come from the same text's earlier run.
- Small effects go both ways (summary-json accepts more with 96 than with all; it has 24 steps).

**Not measured here:** speed. Most runs read from storage during generation (verifying 5
drafts at every step fetches engram rows no earlier run had; the records say which), so their
t/s are not comparable, and the experts were not left out, so no slot was gained.

**What it says about the two ways:** keeping fewer is cheap where the set fits the prompt (a
quarter of the memory for 1.4% more steps) and expensive where it does not. So the question
moves from "how many" to "how is the set chosen": the 142 never chosen are free now; beyond
that the set has to follow the prompt, without loads into the home die during a run (step 86).
Requantizing was not tried.

**Next:** leave the unchosen ones out for real (the draft's slots compacted, an id -> slot table
after its router) and measure end to end what ~140 more main-model slots on the home die give.

## Step 125: the unchosen draft experts left out for real: the slots give almost nothing

Step 124 measured what the draft loses when a third of its experts are masked (+1.5% verify
steps). This step frees their memory and measures what the main model gains from it.

**The change.** With `DS_DRAFT_KEEP` and the draft's experts in the home die's pool (step 123),
only the kept experts get slots, per layer in id order; the pool's remaining space goes to the
main model through the same capacity rule as before. The draft's router still chooses among
expert ids; one small kernel after it (`k_remap_ids`) turns the ids into slots with a table per
layer. `DS_DRAFT_KEEP_RESIDENT=1` keeps step 124's behaviour (masked, all loaded), which
separates the two effects. Without `DS_DRAFT_KEEP` nothing changes. Still a study switch.

**Left out equals masked** (geometry and ja-email, `--spec-n 5`, the other seven prompts' pairs
kept): the same drafted ids at every step (81 and 118 steps), the same routing log, the stored
outputs' bytes. The home die holds 944 and 950 main-model experts, was 802.

**End to end** (`results/draft-pool/draft-leave-out.sh`, `results/draft-pool/leave-out/`; sclk 6): the 8
held-out prompts with the fastest settings, `all` and `out` twice each, `resident` once. All 40
outputs are the stored ones' bytes.

| prompt | die 0 experts: all, out | GPU share of expert uses: all, out | verify steps: all, out | t/s: all, out | |
|---|---|---|---|---|---|
| tcp-udp | 802, 941 | 87.53%, 88.06% | 144, 144 | 38.97, 39.38 | +1.1% |
| py-lru | 802, 943 | 85.16%, 85.59% | 100, 100.5 | 46.49, 46.67 | +0.4% |
| cpp-matrix | 802, 940 | 85.68%, 86.09% | 66, 71 | 54.70, 53.58 | -2.0% |
| sql-report | 802, 943 | 81.95%, 82.44% | 77, 76 | 47.10, 46.69 | -0.9% |
| geometry | 802, 944 | 86.29%, 86.47% | 85, 93 | 54.72, 53.12 | -2.9% |
| zh-travel | 802, 947 | 87.78%, 87.91% | 199, 205 | 34.26, 33.56 | -2.0% |
| ja-email | 802, 950 | 86.99%, 87.19% | 170, 168.5 | 37.04, 37.19 | +0.4% |
| summary-json | 802, 942 | 75.34%, 75.98% | 28, 28 | 44.76, 44.82 | +0.1% |

Mean of the prompts: **-0.7%**. (cpp-matrix's -2.0% is one run of the two: 76 steps and 52.20
t/s, with 0.4 MB read from storage; the other was 66 steps and 54.96. The draft-length rule
follows measured times.)

- **The slots are worth little:** ~140 more experts on the home die (+17% of its slots, 2.7 GB)
  move 0.1-0.6 points of the expert uses from the CPU to GPUs. Where the steps are the same
  (tcp-udp, summary-json) that is 0-1% in speed. The cache already holds what a prompt uses
  most on dies 1-3; the home die's slots are static (step 86) and hold the placement study's
  next-ranked experts, which a given prompt seldom needs.
- **The draft's loss is the same size and decides the sign:** on geometry and zh-travel the
  masked-but-loaded runs (`resident`) are as slow as `out`: the steps the draft loses (8 of 85,
  6 of 199) cost more than the slots return.
- The run-to-run spread of one configuration is ~1%; no difference here is far outside it
  except geometry's and zh-travel's.

**Decision.** Not made the default: no gain, and a prompt outside the sample can lose much more
(step 124: ja-email at 192). The switch and the pool layout stay; they cost nothing when off.
What this rules out is the motive: memory on the home die is not what limits speed now, so
neither a draft-expert cache nor a 2-bit draft would pay through slots. Requantizing could
still pay through a faster draft (its experts are 1.7 of ~8 ms), which is a different question.

## Step 126: the record's first value: a hash of the printed token ids

Steps 123-125 were used as a test of `--record` (docs/record.md, "A case study"). The record
carried the measurements, but every "the output is the same" in them, 104 runs, was a `cmp`
against a stored file, outside the record. So the first of the record's *values*:

    "output": {"tokens": 300, "hash": "fnv1a64:..."},

The prompt's hash again (FNV-1a, 64 bits, each id as four bytes), over the ids of the tokens
the run printed, in order. Not the end-of-sequence id that ends a run: the plain loop keeps
that id and the speculative loop does not, and the hash must not depend on the loop.

**Checked** (`results/record/output-hash.txt`; sclk 6), one prompt:

| run | tokens | hash |
|---|---|---|
| GPU runtime, `--canon`, plain, `-n 32` | 32 | `29b8836ab745322e` |
| the same with the placement and `--spec` | 32 | `29b8836ab745322e` |
| plain, `-n 33` | 33 | `ca037a04a081e54a` |
| the CPU path (not `--canon`), `-n 32` | 32 | `23d4401c77fe3496` |
| `--spec`, `-n 2000`: ends by itself | 579 | `8bc3d2716bdfcad8` |

Plain and speculative print the same bytes and have one hash; a token more, another. The CPU
path prints other text from byte 35 and has another hash: without `--canon` the output depends
on where an expert runs, which is why `--canon` is among the inputs that may change the output.
And the hash was recomputed outside the program, from `DS_DRAFT_TOKEN_LOG`'s emitted ids: it is
the record's for both speculative runs, the end-of-sequence id left out.

It is a hash of tokens, not of logits: two runs with it equal chose the same tokens, their
logits may differ (the argmax absorbs small differences). A logits checksum is still to come.
No check uses the field yet.

## Step 127: a label in the record, and what the scripts are

**The scripts, sorted.** `scripts/` has 43 files. By what they are for: 2 launchers (`chat.sh`,
`split-measure.sh`), 13 sweeps (prompts x configurations, numbers scraped from stderr, a
table), 2 gates (a sweep that ends in pass or fail), 5 table makers, 6 log analysers, 6
simulators, 4 trace tools, 5 helpers and shims. 31 of the 43 have one commit: written for a
step, never changed. Five are named in more than two rows of REPRODUCE.md (`spec-measure.sh`
in 28). So the directory is a few tools among ~35 pieces of evidence, and no name says which.

**The plan agreed** (not done here): sort by lifetime, not by topic. What is still used stays
in `scripts/`; what belongs to one step is filed with that step's results. The sweeps, gates
and table makers are one program written twenty times; going forward two small tools on top of
`--record` replace the pattern: one that runs prompts x configurations x repeats, each run
writing a record, and one that reads records, groups them, and prints means with the spread of
the repeats (a gate: the same tool with a condition, e.g. one `output.hash` per prompt). Old
sweeps are not rewritten. The order: the label (this step), the reader, the runner, then the
filing.

**The label.** The reader must know which run a record is. Steps 124-125 put that in file
names. Now:

    --record run.json --record-label config=out,prompt=geometry --record-label rep=2
    "label": {"config": "out", "prompt": "geometry", "rep": "2"},

`key=value` pairs, comma-separated, the option repeatable; written back as strings in the
order given, `{}` without the option. The program never reads them: a prompt's name or a
configuration's belongs to whoever starts the run. Refused before the load (exit code 2): a
part without `=`, a key given twice, a label without `--record`. A value cannot hold a comma.

**Checked** (`results/record/label.txt`): the label as given, quotes escaped; the same output
and `output.hash` with and without it; the three refusals.

## Step 128: a reader of records (`scripts/records.py`)

The first of the two tools of step 127. It reads `--record` files and nothing else:

    scripts/records.py FILE.json... --by prompt,config --show counters.gen_tps,spec.steps \
        --vs config=all --same output.hash --within prompt

- **Groups** by label keys (`--by`); the runs of a group are repeats of each other.
- **Fields** (`--show`): a path into the record (`experts.placed.0`), or a ratio of two
  (`spec.accepted/spec.steps`): the record holds totals. Per group the mean and, with repeats,
  the spread, (max - min) / mean.
- **A comparison** (`--vs config=all`): each group against the one that differs only in that
  key. A difference is printed only if it is larger than the spread of both groups' repeats;
  inside it the tool prints `~`, and `?` beside a difference when a side has one run and so no
  spread. This is the rule from step 125's table, which printed every difference, now in the tool.
- **A condition** (`--same output.hash --within prompt`): one value among the runs that agree on
  those keys; exit code 1 if not, or if a run has no such field. A gate is this.
- An empty record (a run that did not finish) is reported and the exit code is 1.
- `--name-label REGEX`: the label from the file's path, for records older than step 127.

**Tried on step 125's 40 records** (`results/record/reader.txt`): one command gives that step's
table (the same means: 54.70 against 53.58 t/s on cpp-matrix, 802 against 940 experts) and
says more than the table did. Of the eight speed differences it prints four (tcp-udp +1.1%,
ja-email +0.4%, geometry -2.9%, zh-travel -2.0%) and calls four inside the spread, cpp-matrix's
-2.0% among them (its two `out` runs are 5.1% apart: the odd run). Every difference of the
`resident` runs carries `?`: one run each. **The condition** fails as it should on step 126's
three records (the CPU path prints other text), holds without the CPU one, and fails on
records that have no `output`.

Two repeats give a low estimate of the spread (tcp-udp's `out` runs are 0.1% apart, and +1.1%
is printed): the tool says so, and a difference just outside is not yet a result. Not built
yet: the runner, and a spread taken over all the groups instead of each pair.

## Step 129: a runner of records (`scripts/run_matrix.py`)

The second tool of step 127. It starts `ds-completion` over prompts x configurations x repeats,
each run writing a record with its label, and does nothing else: no number is read from
stderr, and `scripts/records.py` (step 128) does the reading. A sweep is a text file:

    set M  ~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
    set F8 ~/llms/ds-1017-fp8/dense-fp8.gguf
    set D  ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
    prompts results/spec-heldout/prompts.txt
    out results/record/runner/leave-out
    reps 2
    base --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon
    base --cache-loads 800 --cache-period 8 --cache-wait
    config all
    config out DS_DRAFT_KEEP=results/draft-pool/keep/loo-seen/{prompt}.keep

(`results/record/runner/leave-out.matrix`: step 125's comparison, which took a 63-line script
and a 32-line table maker.) A configuration is a name, environment variables, arguments;
`{prompt}` is the prompt's name. The runs go repeat by repeat, prompt by prompt, the
configurations in turn, so a drift of the machine falls on all alike. A run's record is
`CONFIG-PROMPT-REP.json` with the same three names as its label; its stdout is dropped (the
record has the output's hash), its stderr kept only if it failed or left no record. The
directory gets `matrix.txt`: the `#` header and a copy of the matrix. `--dry-run` prints the
commands; a directory that has records is refused, or `--resume` makes only the missing runs.

**Tried** (`results/record/runner/`; sclk 6): two of step 125's prompts, 8 runs, then

    scripts/records.py results/record/runner/leave-out/*.json --by prompt,config \
        --show counters.gen_tps,spec.steps,experts.placed.0 --vs config=all --same output.hash --within prompt

| prompt | all | out | reader | step 125 |
|---|---|---|---|---|
| tcp-udp | 39.17 t/s, 144 steps | 39.59, 144 | +1.1% | +1.1% |
| geometry | 55.38 t/s, 85 steps | 53.47, 93 | -3.4% | -2.9% |

The same result as step 125 by another route, and the condition holds: one `output.hash` per
prompt over its four runs, which step 125 checked with `cmp` against stored files. A
configuration with an unknown option is reported as failed with its stderr kept (exit code 1);
`--resume` then skips the finished run.

**With this the pattern is replaced going forward:** a sweep is a matrix file, its table and
its gate one `records.py` command, and both files of evidence are small. Left from step 127's
plan: file the old one-step scripts with their results.

## Step 130: the one-step scripts filed with their results

The last part of step 127's plan. `scripts/` had 44 files; 33 were written for one step. They
moved to that step's results: `results/<topic>/` where the step has a directory (the draft
studies to `results/decode-next/`, the cache test to `results/cache-test/`, ...), `results/`
itself where its result is a single file (`results/cache_sim.py` beside `cache-sim.txt`).
`scripts/` keeps 11, listed in `scripts/README.md`: the runner and the reader, two launchers,
`spec-measure.sh` with its analysis (28 rows of REPRODUCE.md use it), the trace tools,
`profile_verify.py` and `route_stats.py`.

What moving needed, and nothing else changed in them:

- a moved script finds the project's root one directory further up, and the simulators that
  import `route_stats` look for it in `scripts/`;
- every mention of a moved script in the documents, the code's comments and the scripts
  themselves has the new path (153 places; REPRODUCE.md's rows first of all);
- the results' own text files were **not** rewritten: a `#` header names the command at its
  commit, and that was `scripts/<name>` then. `scripts/README.md` says where to look.

**Checked without the GPUs:** every shell script parses and its `cd` reaches the project's
root; every Python file compiles; the ones with `--help` start (the numpy ones under the
reference's Python); `draft_keep_table.py`, `draft_leave_out_table.py` and `draft_keep.py`
write their stored files again byte for byte from the new place, and `heldout_history.py`
rewrote `history.tsv` with only its own name changed in the header. **Not checked:** the moved
sweeps were not run again (hours of GPU time each); what they need beyond the path is unchanged.

## Step 131: `run()` split, part 1: the options and the record out of it

A cleanup round begins: the same speed and the same observability, code that is easier to
read. A survey first. The main program was one function of 664 lines (`run()` in
`apps/ds_completion.cpp`): options, load, prompt, the speculative loop, the plain loop, five
logs and the record, interleaved. `hip/rt.cpp` is 2,297 lines. About 200 lines belong to
studies that were not adopted (`DS_DRAFT_KEEP`, the router lookahead, `--load-test`, the second
draft block, `DS_EARLY_ROWS`, `DS_ATTN_CHECK`), spread through the hot paths as conditions. The
order agreed: split `run()`; remove the unadopted studies one per commit (each keeps its README
step and its commit); `rt.cpp` last. The kernels and their A/B switches stay as they are.

**This step moves code and changes none:**

- `struct Args`: the 37 option variables that were `run()`'s first locals;
- `parse_args()`: the option loop and the usage checks;
- `write_record()`, with a small `RecordRun` of what a turn measured: the JSON.

`run()` is 505 lines; the two loops are next. The renaming was done by a script that leaves
strings and comments alone (`"# prompt: ..."` stays, the variable `prompt` becomes `args.prompt`).

**Checked** old build against new (`results/cleanup/run-split-1.txt`; sclk 6): eight usage
errors give the same message and exit code; seven kinds of run (the CPU path, plain, shadow
drafting, speculative with `--profile`, the fastest settings, a two-turn `--think` session,
`--teacher`) print the same bytes, write the same record but for its times, and the same kinds
of lines on stderr; speeds agree within 1% (54.56 and 54.52 t/s on the fastest settings). The
record's expert counters differ in the fastest run as they do between any two runs: the cache
follows time.

## Step 132: `run()` split, part 2: the two generation loops

The second half of step 131's first item. Out of `run()`, again moved and not changed:

- `struct Turn`: what one turn's generation did: the tokens printed, the generated ids, the
  shadow drafts, the speculative totals;
- `generate_spec()`: the speculative loop with its draft-length rule and its logs;
- `generate_plain()`: a token per step, on the runtime (in shadow mode with a draft loaded) or
  on the CPU engine;
- `report_shadow()`: shadow mode's statistics.

`run()` is 268 lines (664 before step 131). Its generation part now reads: start the clock,
one of the two loops, stop the clock, the summaries, the record. One unused local went
(`std::string out`).

**Checked** old build against new (`results/cleanup/run-split-2.txt`; sclk 6), eleven kinds of
run: the seven of step 131, two that end by themselves (CPU and speculative), a plain run with
`--profile`, a plain two-turn session. Every stdout is the same bytes; every record the same
but for its times and the cache's counters; the same kinds of stderr lines; the speculative
loop's three logs the same but for their time columns; shadow mode's report the same numbers.
The fastest settings twice: 54.57 and 54.60 t/s before, 54.62 and 54.48 after.

`apps/ds_completion.cpp`, top to bottom, is now: the tokenizer helpers, the crash handler,
`Args` and `parse_args`, `write_record`, `Turn` and the two loops, `run` (load, the session's
turns, the summaries), `main`.

## Step 133: an unadopted study out: the second draft block (step 107)

The second item of step 131's order: studies that were measured and not built leave the code,
one per commit. Each keeps its README step, its results and the commit that last had it.

Step 107 asked whether a second draft block, drafted from the first one's last token before any
verify, would pay; it would not. Its instrument stayed: `DS_DRAFT2_LOG` (two extra blocks after
every draft, logged), `Runtime::draft_shadow`, and a `gap` parameter through `DSpark::draft`
into the block's positions. All three are gone (33 lines, four files). The study's results and
its simulator (`results/decode-next/draft2.txt`, `draft2_sim.py`) stay; to run it again, build
commit `012bf44`, the last one with the switch.

**Checked** old build against new on the draft's paths (`results/cleanup/study-draft2.txt`;
sclk 6): shadow mode's report the same numbers; with a fixed draft length the same drafted and
emitted ids at all 26 steps; the fastest settings twice the same output hash, steps and
accepted drafts, draft time 514.0 and 515.8 ms before, 515.3 and 513.9 after, 54.5 t/s both; a
two-turn `--think` session the same bytes.

## Step 134: an unadopted study out: `--load-test` (step 30)

Step 30 measured what background expert loads cost a running generation, before the cache
existed: `--load-test R` kept 8 spare slots per die and loaded random experts into them at R
loads/s, `--load-pinned` chose the staging path. The cache (step 32) has done real loads since,
through the same loader's queue. Removed (59 lines, six files): the two options, the spare
slots, the runtime's test loader, and the loader's test mode, so its worker loop only serves
its queue. `results/load-test.sh` and the step's results stay; the last commit with the
options is `def1a2a`.

**Checked** old build against new (`results/cleanup/study-load-test.txt`; sclk 6): a plain run
and a speculative one without the cache, the fastest settings three times, the cache without
`--cache-wait`, and the fastest settings traced: the same output hash everywhere, the same
experts placed per die, 3,621-3,627 loads published in both builds, 55.05 / 54.99 / 55.02 t/s
before and 54.99 / 54.95 / 54.59 after, and in the trace one cache-load span per load published
(3,679 and 3,675). `--load-test` is now an unknown argument.

## Step 135: an unadopted study out: the router lookahead (step 63)

Step 63 asked how well a later layer's router, run on an earlier layer's input, predicts the
experts that layer will choose, as the basis of a prefetch; the prefetch simulated to a few
percent and was not built. Its instrument was the largest leftover: `DS_ROUTE_AHEAD` and
`DS_ROUTE_LOG`, a branch in every layer's routing that ran up to 8 further routers, a
comparison at each step's end and a report: 174 lines of `hip/rt.cpp` and `hip/rt.h`, and an
accessor on the cache only it used. All removed. `results/route-ahead/` keeps the results, the
script and the simulator; the last commit with the switches is `8798740`.

**Checked** old build against new (`results/cleanup/study-route-ahead.txt`; sclk 6): a plain
run's `--log-routing` file the same bytes; a speculative run with the cache and `--profile`
the same output, counts and kinds of stderr lines (step 52's prefetch-prediction line, which
stays, among them); the fastest settings three times the same hash and counts, 54.71 / 54.91 /
54.32 t/s before and 54.85 / 54.80 / 54.57 after.

**Not removed, after a second look:** the draft's hidden-state dump (`draft_log_hidden`). It
is not a leftover: `ds-test` writes it and `ref/dspark_check.py` compares it layer by layer
with the reference, the check a new draft or checkpoint would need. A search of the engine's
files alone had found no caller; the search now covers `src/`, `hip/`, `ref/`, `scripts/` and
`results/`. Of the switches removed in steps 133-135, only their own filed study scripts name
them.

## Step 136: `hip/rt.cpp` split, part 1: observability into `hip/rt_trace.cpp`

The study removals end with step 135: the draft's hidden-state dump and `DS_DRAFT_KEEP` stay
for now (the person's decision). The last item of step 131's order is `hip/rt.cpp`, 2,123
lines holding four things: construction and teardown (~600 lines; the constructor alone ~360),
the forward (~800: attention, the experts' launch and finish, the blocks, the entry points),
observability (~250) and the two-machine code (~150).

**This step moves the third out, whole functions and nothing inside them changed:**
`hip/rt_trace.cpp` has the startup memory ledger, `mark()` with `--profile`'s bookkeeping, and
all of `--trace`; `hip/rt_internal.h` has the two helpers both files use (the error check and
the clock). `rt.cpp` is 1,872 lines.

The one risk is speed: `mark()` runs at every profile group of every layer and the compiler
can no longer inline it from another file.

**Checked** old build against new (`results/cleanup/rt-split-trace.txt`; sclk 6): with
`--profile`, plain and speculative, every stderr line the same but for its numbers; a traced
run the same tracks and the same 52 kinds of spans; the memory ledger the same text; the
fastest settings three times 54.52 / 54.90 / 54.61 t/s before and 54.53 / 54.61 / 54.57 after
(means 54.68 and 54.57, inside the ~1% spread of repeats).

Next: the two-machine code the same way, then the constructor in parts.

## Step 137: `hip/rt.cpp` split, part 2: the two-machine code into `hip/rt_peer.cpp`

The same move as step 136 for the model in two halves (step 111): the front's connection and
its messages, the back's `serve`, and the stream's two copies to and from the host go to
`hip/rt_peer.cpp` (142 lines), unchanged. `rt.cpp` is 1,735 lines: construction and the forward.

**Checked on one machine** (`results/cleanup/rt-split-peer.txt`; sclk 6): the two halves as two
processes over localhost, every expert on the CPU so that both fit, in every pairing of the old
and the new build (old-old, new-new, new front with old back, old front with new back), and
new-new traced. All five print the whole model's bytes (one output hash in the six records)
and end with exit code 0; old-old and new-new write the same kinds of stderr lines on both
sides. The mixed pairings say the new build speaks the old protocol in both roles.

**Not checked:** the speed on two machines. The second machine was not touched (it needs this
commit built; `scripts/split-measure.sh` then measures). The moved functions run a few times
per step.

## Step 138: the held-out set after the cleanup round, and the first reference values

Steps 131-137 were each checked against the build before, on one or two prompts. This step
checks the end state on all eight held-out prompts, with the tools of steps 127-129 only: a
matrix file (`results/cleanup/heldout.matrix`, the fastest settings, two repeats), the runner,
the reader. The runner got one option, `--keep-out` (a run's stdout kept as a file): the stored
references are output files, not hashes.

**Outputs** (`results/cleanup/heldout.txt`; sclk 6): all 16 are the same bytes as
`results/cache-test/plain-canon-static/<prompt>.out`, and each prompt has one output hash over
its two runs.

**Speed** against the build before the round (step 125's `all` runs: the same command, their
records in git), read by one `records.py` command:

| prompt | before | after | reader |
|---|---|---|---|
| tcp-udp | 38.96 | 38.96 | inside the spread |
| py-lru | 46.49 | 46.51 | inside the spread |
| cpp-matrix | 54.70 | 54.54 | -0.3% |
| sql-report | 47.10 | 46.93 | inside the spread |
| geometry | 54.72 | 54.94 | inside the spread |
| zh-travel | 34.26 | 34.11 | inside the spread |
| ja-email | 37.04 | 36.84 | inside the spread |
| summary-json | 44.76 | 44.74 | inside the spread |

Mean of the prompts 44.75 and 44.70 t/s (-0.1%). Seven differences are inside the spread of
the repeats; cpp-matrix's -0.3% is printed because both of its pairs happen to be 0.2% apart
(two repeats underestimate the spread, step 128). The round changed no speed that this can see.

**The first reference values** (`results/record/reference/heldout-output.tsv`): per held-out
prompt its token count and hash and the output's, taken from these records right after their
outputs were compared with the stored files. A later check needs no output file: `--same
output.hash` over new records, and the hash against this table. With `--canon` the values do
not depend on placement, cache, speculation or the machine; a new checkpoint or tokenizer
changes them, and then this file is made again the same way.

A mistake on the way, kept for the record: the first start of this run never ran. An edit to
the runner had a wrong indent, the command ended with `echo` and so reported success, and it
was announced as running without its output being read. Since then: the dry run first, chained
with `&&` to the real run, and the exit code that is reported is the runner's.

## Step 139: libraries and programs apart: `apps/`, `tests/`, `bench/`

`src/` and `hip/` each mixed the code the engine is made of with programs that have a `main`:
the CLI, two tools, three tests, twelve benchmarks. Now:

| directory | holds |
|---|---|
| `src/` | the CPU-side library: the model's loading, the CPU forward, the CPU's experts, the canonical dot product, the fake quantization |
| `hip/` | the GPU runtime library: the runtime, the draft, the expert pool, loader and cache, the kernels |
| `apps/` | the programs one runs: `ds_completion.cpp` (with `record.h`, which only it uses), `ds_inspect.cpp`, `fp8pack.cpp` |
| `tests/` | `ds_test.cpp`, `test_fakequant.cpp`, `test_topk.cpp` |
| `bench/` | the twelve `bench_*` programs, CPU and GPU together (the Makefile knows which need `HIP=1`) |

No file of `src/` or `hip/` has a `main` now, and the three new directories have nothing else.
No `#include` changed (the compile flags already named both library directories); the
Makefile's paths did, and the mentions in the documents (results files' headers and the
prompt fixtures not, as in step 130). The binaries keep their names and places.

**Checked** (`results/cleanup/layout.txt`; sclk 6): all 25 programs of both builds rebuilt (the
first `make` rebuilt none: a moved file keeps its time; the moved sources were touched and the
binaries' times checked); `test-fakequant` passes and a program from each new directory runs;
the eight held-out prompts once with the fastest settings: **all eight match the reference
values of step 138** (prompt and output hashes, the table's first use), speeds within 0.8% of
step 138's.

## Step 140: a vendor profiler for the GPU side: `rocprofv3`

`perf` covers the CPU side; for HIP the machine had only ROCm 5.7.1's profilers, which do not
fit the 6.3.4 runtime the engine loads. The person installed `rocprofiler-sdk6.3.4`; its
`rocprofv3` attaches to our binary as built (the runtime already links the hook).

**Tried** on a 64-token plain run (`results/rocprof/first-run.txt`; sclk 6, one run each):
the output's bytes do not change; generation 24.25 t/s without it, 19.28 with `--kernel-trace`
(-21%), 17.96 with `--runtime-trace` (-26%). So it ranks and counts; it does not measure speed.

What it adds to `--profile` and `--trace` is the time each kernel runs on the device. Over the
whole process: 39 kernels, 167,441 launches, 2.88 s on the devices; `fp8_gemv_kernel<64,1,1>`
16.5%, `canon_gate_up_kernel` 16.4%, `unpack_blocks_kernel` 14.7%, `canon_down_kernel` 8.2%.
From the launches' timestamps: **2,197-2,242 launches per generated token** (step 49's count
was ~2,450), 27,562 before the first token; the home die takes 119,876 of them and each other
die 15,855. Host-side times include the profiler (`hipLaunchKernel` 8.5 us against 2.2-2.5 us
in step 49): of the HIP calls only the counts are usable.

`docs/rocprof.md`: how to run it, the files, the timeline, how to read the numbers. Not tried:
hardware counters (`--pmc`), and the timeline in a viewer.

## Step 141: `rocprofv3`'s hardware counters

Step 140 left `--pmc` untried. It needs one more package (`hsa-amd-aqlprofile6.3.4`, installed
by the person; `rocprofiler-sdk` does not pull it in). Then the card reports 86 counters per die.

**What works** (`results/rocprof/counters.txt`; sclk 6): each of twelve counters alone and
some small sets; a set that needs more hardware counters than the card has is refused, e.g.
five at once, or `L2CacheHit` with `FetchSize`. A refusal aborted the model at once and left
a small test program hanging, so every counter run goes under `timeout`. The cost is of
another order than tracing: a 1.6 us launch takes 57-146 us, the model generates at 4.0-7.4
t/s against 26.6; the output's bytes do not change.

**A first table**, four 16-token runs with one set each, generation only: the fp8 kernels
read 18.8 MB (`<64>`) and 41.0 MB (`<8>`) per launch with the memory unit busy 49% and 79% of
the time and stalled 1.5-2.3%, cache hits 13% and 3%; the two expert kernels read 13.8 and
6.8 MB per launch, memory unit busy 32-36%, cache hits 41-44%. None of the values is checked
against a kernel of known behaviour yet: relative for now. `docs/rocprof.md` has the how-to.

## Step 142: the counters against kernels whose reads are known

Step 141's values were unchecked. `bench-fp8` gives the check: fp8 weights are one byte each,
so a call on a K x N tensor must read K x N bytes once. Eight tensors of one layer, 2.5 to 150
MB, each under four counter sets (`results/rocprof/calibration.txt`; sclk 6).

**`FetchSize` is exact:** every tensor reads its weights plus 0.10-0.38% (attn_q_b: 41,023.9
KB against 40,960; engram_kv: 153,759.6 against 153,600). A cache miss is one 64-byte line:
misses x 64 bytes equals `FetchSize` to 0.3%, and `L2CacheHit` is hits / (hits + misses).
This settles step 141's doubt: the model's `fp8_gemv_kernel<8,1,1>` at 41,024 KB per launch
is attn_q_b read once.

**`MemUnitBusy` is not comparable between kernels of different length:** on the same kernel
and the same kind of data it reads 11.7% at 7 us per call and 91% at 209 us, growing with the
call's time. The share is of an interval that has a fixed part around the kernel. So step
141's column of busy shares ranks kernels by length more than by behaviour; the bytes and the
hit rate are the columns to use. The instruction counters are still unchecked.

## Step 143: how busy each die is in generation, and what fusing launches could give

The question behind fusing kernels (steps 49, 99): is a die idle between its launches? Step
140's kernel trace has every launch's start and end on the device, so the answer needs no new
run: `results/rocprof/busy.py` takes the window from the first to the last output-head launch
and, per die, the time a kernel is running (`results/rocprof/busy.txt`; a plain run, traced:
52.7 ms per token where the untraced run takes 41.2, so the gaps hold the profiler's cost too).

| die | launches per token | a kernel running | share of the traced step |
|---|---|---|---|
| home (dense part, its experts) | 1,740 | 26.2 ms | 49.8% |
| each other die (experts) | 160 | 2.7-2.8 ms | 5.1-5.3% |

**The home die's idle time is not between back-to-back launches.** Of its 1,740 launches per
token only 194 follow an idle gap, and the 111 gaps under 20 us sum to 0.92 ms per token: what
removing every launch boundary could give at most, about 2% of a step. Step 99's fusion
gaining nothing agrees with this.

**It is two waits per layer:** 35 gaps per token of 506 us before `copy16_kernel`, the copy of
the CPU's expert rows (step 84), 17.7 ms per token: the die waits for the CPU's experts, as
`--trace` shows from the host as `ffn.wait_for_cpu_experts`; and 37 gaps of 176 us between
the die's own expert kernel and the next `quant_act_kernel`, 6.5 ms per token, not yet matched
to a span of our trace. The other dies run 5% of the time: on this placement their experts
are a small part of the step.

So on a plain step the GPU-side lever is not launches; it is the same one as before, the wait
for the CPU's experts, now seen from the device. Not done: the same on the fastest settings
(speculation and the cache change both the rows per launch and the CPU's share).

## Step 144: the dies' busy share on the fastest settings

Step 143 on the configuration that matters: the draft, speculation, the cache
(`results/rocprof/busy-fast.txt`; cpp-matrix, 300 tokens, sclk 6). `busy.py` takes the token
count as an argument now: with speculation the head runs per step and for the draft, so its
launches no longer count tokens. Tracing costs less here, 55.07 -> 49.35 t/s (-10%): a verify
step has fewer launches per token (475 on the home die against 1,740).

| per token | plain step (143) | fastest settings |
|---|---|---|
| the traced step | 52.7 ms | 20.0 ms |
| home die: a kernel running | 26.2 ms (49.8%) | 10.8 ms (53.8%) |
| home die: waiting for the CPU's experts | 17.7 ms | 7.2 ms |
| home die: the 176 us gap after its own experts | 6.5 ms | 1.0 ms |
| home die: gaps under 20 us | 0.92 ms | 0.17 ms |
| each other die: a kernel running | 2.7 ms (5%) | 2.8-3.0 ms (14-15%) |

The picture holds. The home die still runs a kernel about half the time; launch boundaries
are under 1% of a step; and **the wait for the CPU's experts is 36% of the traced step** (7.2
of 20.0 ms per token; 33 ms of a 92 ms verify step, 926 us per layer that has a CPU expert).
The 176 us gap is the same length and far rarer per token, since it is per layer and step.
The other dies run 14-15% of the time, of which the cache's loads (`unpack_blocks`) are 0.24
ms per token: the dies have room, the limit on what they can take is memory, not time.

So from the device's side too, the largest single item of a fast step is the CPU's experts:
what the cache, the placement and the second machine all work on. Nothing new to build here;
the profile agrees with `--trace`.

## Step 145: the gap after the home die's own experts, named: the host issuing the other dies' work

Steps 143-144 left one idle gap of the home die unexplained: ~176 us per layer between its own
expert kernel (`canon_down_kernel`) and the next `quant_act_kernel`. Step 140's
`--runtime-trace` run has every HIP call with its times on the same clock as the kernels, so
`results/rocprof/gap_calls.py` sums the calls in progress during those gaps
(`results/rocprof/gap-calls.txt`; under this heavier trace the gap is 244 us).

**91% of the gap is the main thread inside HIP calls**, per gap: 8.7 `hipLaunchKernel`, 5.2
`hipMemcpyAsync`, 3.2 `hipStreamWaitEvent`, 2.9 `hipEventRecord`, 3.0 `hipSetDevice`. Three
`hipSetDevice` per gap is the signature: it is `moe_launch`'s loop over the three other dies
(`hip/rt.cpp`: set the device, wait for the staged block, copy it in, the expert kernels, the
copy back). The shared expert, whose first kernel is that `quant_act`, is issued after the
loop; the home die has finished its own experts by then and waits for the host.

The control is the other long gap, the wait for the CPU's experts: HIP calls cover 3% of it
(the host is in the CPU's kernel, not in HIP).

**What it is worth.** The die's idle time here is not the cost: after the shared expert the
die waits for the CPU's experts anyway (35 of 40 layers), so issuing the shared expert first
would only move the idle time. What is on the critical path is the host's time in that loop,
since the CPU's experts start when `moe_launch` returns; the code says so already ("HIP calls
there would delay the CPU's experts"), and steps 71 and 97 worked on it. New here is the
size per call under tracing: a copy 16 us, a wait or a record 12-13 us, a launch 7 us. These
include the profiler (a launch is 2.2-2.5 us without it, step 49), so the untraced loop is
shorter; it is not measured.

Not tried: the shared expert issued before the loop (expected: nothing, by the argument
above, except in the layers without a CPU expert); the loop's untraced time from `--trace`.

## Step 146: what a perfect draft-length rule would give (an estimate)

The person's question: the same draft, but a rule with perfect knowledge. It verifies exactly
as many drafts as will be accepted, and when none will be, it does not draft at all and takes
a plain step.

**It needs no oracle to be built.** A `--spec-n 5` run verifies the whole block every step, so
its log has k, the drafts accepted, for every step, and such a step advances k + 1 tokens. An
oracle verifying exactly k also advances k + 1: it visits the same positions with the same
drafts. Its run is the n5 run's steps at other costs (`results/spec-oracle/oracle.py`):
verify(k) from the measured steps of that width on that prompt, the draft 7.8 ms, a plain
step for k = 0, and the time outside both (0.4-1.0 ms per step).

**Result** (`results/spec-oracle/oracle.txt`; the eight held-out prompts, the fastest settings,
one run each, sclk 6; t/s):

| prompt | always 5 | the rule today | perfect length, draft always | perfect, no draft at k = 0 |
|---|---|---|---|---|
| tcp-udp | 36.08 | 39.03 | 50.30 (+28.9%) | 52.07 (+33.4%) |
| py-lru | 45.47 | 46.51 | 55.97 (+20.3%) | 56.98 (+22.5%) |
| cpp-matrix | 53.32 | 54.62 | 58.48 (+7.1%) | 58.93 (+7.9%) |
| sql-report | 44.12 | 47.23 | 52.66 (+11.5%) | 53.40 (+13.1%) |
| geometry | 53.75 | 55.18 | 61.27 (+11.0%) | 61.61 (+11.7%) |
| zh-travel | 28.97 | 34.05 | 43.74 (+28.4%) | 46.30 (+36.0%) |
| ja-email | 31.70 | 37.08 | 46.41 (+25.2%) | 48.57 (+31.0%) |
| summary-json | 39.80 | 44.85 | 50.24 (+12.0%) | 50.85 (+13.4%) |
| mean | 41.65 | 44.82 | 52.39 (+16.9%) | 53.59 (+19.6%) |

- **The rule today is 7.6% above always verifying 5; perfect knowledge would be another
  ~20%.** Most of it is the length (+16.9%); skipping the draft when nothing will be accepted
  adds under 3 points, since a draft is 7.8 ms next to a 34-49 ms plain step.
- **The room is where acceptance is poor:** +29-36% on tcp-udp, zh-travel, ja-email, where
  half the steps accept 0 or 1 draft; +8-12% on cpp-matrix and geometry, where 5 of 5 is the
  usual step.
- It is a bound for rules, not for speculation: the draft and the block of 5 stay.

**How far to trust it.** The step sequence is exact; the costs are a model. Verify time is not
a line in the width (a line fitted to the adaptive log alone misses the n5 run's mean verify
by -4.5% to +11%), which is why the table uses the measured mean per width; steps that accept
little may also be costlier than the mean of their width (harder text, more of the CPU's
experts), which would make the oracle look better than it is. A few percent either way; the
size of the gap and its concentration on three prompts do not depend on that.

No rule can reach it: the outcome is not known before the verify. What it says is that the
confidence head's information, not the step costs, is what limits speculation on the poor
prompts.

## Step 147: usability: the reasoning as one moving line (`--think-show`)

The person's observation: `scripts/chat.sh --think --effort max -n 50000` prints thousands of
reasoning tokens before the answer, and the answer is then hard to find. Seeing the reasoning
move is useful; having it on the screen afterwards is not.

`--think-show` says where a reply's reasoning (everything up to the model's `</think>` token)
goes:

- `all`, the default: on stdout like the answer, as before. Every stored output stays valid.
- `line`: one line on stderr, redrawn at every token: a count and the reasoning's last
  characters, cut to the terminal's width (never inside a character; line breaks as spaces).
  When the reasoning ends the line is erased, a note stays (`# reasoning: N tokens in T s`),
  and the answer goes to stdout alone. If stderr is not a terminal it behaves as `none`.
- `none`: only the note.

The tokens, the context, the record and the output hash do not depend on it; a reply that ends
inside its reasoning (`-n`) says so in the note. `apps/think_view.h` has it, and both
generation loops print through it. `scripts/chat.sh` passes `--think-show line` (give
`--think-show all` to it for the old behaviour).

**Checked** (`results/usability/think-show.txt`): the default's stdout is the previous build's
bytes; `none` and `line` print exactly the end of that output, with one output hash in the
records; on a terminal of 100 columns the frames are drawn and erased before the answer; the
cut on a terminal of 60 columns with Japanese and Greek text stays inside the width; a
two-turn session; a reply cut off inside its reasoning. On a real terminal with a long
reasoning: the person's own try.

A mistake on the way: the first check ran in the background while the person started a chat
session, and theirs failed for lack of GPU memory. Since then a test asks whether an engine
process is running before it starts one, and says so before it starts.

## Step 148: a sampler (temperature, top-p), as a library with its test

The person's plan: a benchmark of theirs sends single requests to an OpenAI- or
Anthropic-compatible server, so the engine is to get a minimal server. The engine has the chat
format and the reasoning effort; it lacks tools in the prompt, the parsing of a reply, a process
that stays loaded, and a sampler: it is greedy everywhere, and the model card evaluates at
temperature 1.0 with top-p 0.95 or 1.0. The sampler comes first, as every route needs it. This
step is the sampler alone; no program uses it yet.

`src/sampler.h`: `sample_token(logits, n, options, position)` with a temperature, top-p and a
seed.

- **Temperature 0 is greedy** (the first of equal maxima, as the loops do today).
- **The random number is a function of (seed, position)**, a hash, not the next value of a
  running generator. The choice at a position then does not depend on how the earlier positions
  were computed. This is for speculative decoding: with a draft that proposes one token, the
  exact rule is "sample the main model's token at the position, accept the draft if it is that
  token", and with position-keyed numbers a seed gives the same output with and without drafts.
- **top-p** keeps the most probable tokens (equal ones by rising id) until their sum reaches
  top-p. The whole vocabulary is not sorted: tokens under a floor cannot be in the kept set
  (together they hold under half of what top-p leaves out), and of the rest only the first 256,
  then 8 times more, are sorted while the sum is short.
- Probabilities are computed in double in a fixed order, so equal logits give an equal token.

The reference samples with torch's generator (`inference/model.py: sample`), so a sampled
output cannot be compared with it byte for byte. The checks are of the distribution instead
(`bin/test-sampler`, `results/sampler/test-sampler.txt`; no model, no GPU, and the same numbers
on every run):

| check | result |
|---|---|
| the uniform numbers, 1M positions of one seed and 1M seeds of one position | means 0.49996 and 0.49990; chi-square of 16 bins 15.4 and 12.8 (limit 37.7 at p = 0.001) |
| 8 logits, 400k draws, temperatures 0.5 / 1 / 2, three seeds each | chi-square against softmax 4.4-14.6 (limit 24.3) |
| top-p 0.9 of {0.5, 0.3, 0.15, 0.04, 0.01} | the two smallest never drawn, the rest renormalized (chi-square 1.3, limit 13.8) |
| 129,280 logits, top-p 0.5-0.999, three temperatures | the same token as with the whole vocabulary sorted in 4,800 of 4,800 draws |
| one seed twice; two seeds | equal at 1,000 of 1,000 positions; different at 780 |

What a draw costs on one thread, for a vocabulary of the model's size: 0.9 ms, nearly all of it
the 129,280 exponentials; top-p adds nothing when a few tokens hold the probability. On "flat"
test logits whose tail holds most of it (top-p then keeps tens of thousands of tokens, which a
model does not give) it is 6.3 ms. A plain step takes ~38 ms, so 0.9 ms is ~2% and is left as it
is until measured in the engine.

Next: `--temp`, `--top-p`, `--seed` in the plain loop; then the logits of every verified
position out of `verify()` and sampling in the speculative loop, where fewer drafts will be
accepted than with greedy decoding.

## Step 149: the sampler in the plain loop (`--temp`, `--top-p`, `--seed`)

`ds-completion` takes `--temp T [--top-p P] [--seed S]`, and the plain loop (one token per step,
on the runtime or the CPU engine) chooses every token with `sample_token`. The position that
keys the random number is the token's index in the context. The default, `--temp 0`, is the
greedy choice as before. `--top-p` without `--temp` is refused, and so is `--temp` with `--spec`:
the speculative loop is still greedy (next step). The record's `args` has `temp`, `top_p`, `seed`.

The check (`results/sampler/plain.matrix`, 18 runs: three held-out prompts, plain generation with
`--canon`, sclk 6; `results/sampler/plain.txt`, the records in `results/sampler/plain/`):

| question | runs | result |
|---|---|---|
| is greedy unchanged? | `greedy` | the three outputs have the stored hashes |
| one seed, one output, whatever the placement? | seed 1 with `study30`, with another fixed placement, with the expert cache moving experts; and once more with `study30` | one hash per prompt in all of them |
| another seed? | seed 2 | another output for each prompt |
| top-p 1 (the sampler's path without a sort) | seed 1, `--temp 1` | runs; another output than with top-p 0.95 |

So with `--canon` a sampled run is as checkable as a greedy one: the logits are the same bits on
every placement, and the choice is a function of the logits, the seed and the position. The
sampled text reads as an answer of the same kind (the start of one is in the results file).
The reference's sampled outputs cannot be reproduced (step 148), so nothing here compares a
sampled output with the reference.

Speed, one run each on `study30`: greedy 25.4 / 22.9 / 24.5 t/s, seed 1 at 23.3 / 22.3 / 23.5 (-8%,
-3%, -4%). The sampler's own cost predicts ~-2% (0.9 ms on a ~40 ms step); the texts differ, so
the experts used and their share on the GPUs differ too, and one run per cell does not separate
the two. Not pursued: the speculative loop is the one that matters for speed.

A mistake in the matrix: `reps=2` on one configuration did nothing (in `run_matrix.py` a
configuration's `reps` can only lower the matrix's), so the repeat on the same placement was
run by hand. The matrix file is corrected; the copy among the records shows what ran.

## Step 150: the sampler in the speculative loop

`--temp` now works with `--spec`. The rule is the one step 148 prepared: the draft proposes one
token per position; the main model's token there is the sampler's, keyed by the position; the
draft is accepted if it is that token. This samples exactly from the main model's distribution
(for a draft that proposes one token with certainty, the general accept rule "with probability
p(x), else resample from the rest" is the same thing), and the output is the plain loop's.

- `Runtime::verify()` still returns the greedy ids from the device. With
  `keep_verify_logits` set it also copies the chunk's logits to the host (up to 6 rows of
  129,280), and `verify_logits(i)` gives a row.
- `generate_spec` samples every row (in parallel, one thread per row) and compares the drafts
  with the sampled tokens. The first token of a reply is sampled from the prompt's logits.
  The time is counted in the verify step, so the draft-length rule's cost line includes it.
- Not with `--remote` yet: on two machines the head's logits are on the back. Refused.
- The draft-length rule is unchanged. Its confidence is calibrated for greedy acceptance.

Outputs (`results/sampler/spec.matrix`, three prompts, fastest single-machine settings;
`results/sampler/spec.txt` section 1): 15 of 15 equal to the plain loop's of step 149 with the
same settings: greedy (the stored outputs), seed 1, seed 1 with `--spec-n 5` (every step
verifies five drafts: another sequence of steps), seed 2, and top-p 1.

Speed and acceptance on the 8 held-out prompts, one run each
(`results/sampler/spec-heldout.matrix`, section 3). The sampled texts are other texts, so this
compares two workloads:

| | mean t/s | tokens per step, all steps |
|---|---|---|
| greedy | 44.76 | 2.58 |
| temperature 1, top-p 0.95, seed 1 | 41.79 (-6.6%) | 2.80 |

The prediction (fewer drafts accepted at temperature 1) was wrong on these prompts: as many or
more tokens per step. The model is sure of most tokens here, and where it is not, the greedy
draft was often wrong under greedy decoding too. Per prompt 6 of 8 are slower by 4-12%, two
are level or faster (`tcp-udp` +9%, another text with more accepted).

The loss is the step's cost, not acceptance. A verify step of the same draft count takes
~4-6 ms longer when sampling (section 2: five drafts 73.7 -> 78.9 ms and 70.0 -> 73.6 ms
medians). Expected were ~1 ms for six draws in parallel plus the copy of 3 MB. Where the rest
goes (the copy into pageable memory, the threads' start, the 1 MB allocation per draw) is not
measured. It is the open item of this step; recovering it would give most of the 6.6% back.

A mistake in the announcement: "all 16 held-out prompts, 32 runs". The set has 8 prompts (16 was
the number of runs in step 138, two repeats); 16 runs were made.

## Step 151: where the sampled step's milliseconds go, and a cheaper draw

Step 150 left ~4-6 ms per verify step unexplained. Measured (`results/sampler/step151/where.txt`,
one prompt, five drafts every step):

| part | ms per step |
|---|---|
| the copy of the logits to the host (6 x 129,280 floats) | 0.3 |
| the six draws, one thread each | 2.8 without tracing, 4.0 with |

So the draws, not the copy. A draw was ~1.0 ms of arithmetic alone and 1.45 ms with six
running at once, and starting the threads took ~1.2 ms more. The guesses of step 150 were
mostly wrong: the 1 MB allocation per draw is ~0.04 ms (reused buffers changed the step by
0.15 ms), and `OMP_WAIT_POLICY=active` did not help. Inside one draw (`phases.cpp`):
`std::max_element` 0.22 ms (it is not vectorized), 129,280 scalar `exp` 0.63 ms, the sum 0.13 ms.

Changed:

- **`exp` 8 at a time** (glibc's libmvec, AVX-512; 0.63 -> 0.20 ms) and **the maximum with
  AVX-512** (the first index of the largest value, as before). A draw: 0.9 -> 0.5 ms in
  `test-sampler`, the same token on its 4,800 draws against a full sort with the scalar `exp`.
  The vector `exp` can differ from the scalar one in the last bits, so a build without AVX-512
  could in principle take another token at a boundary; none seen.
- **Buffers reused** between draws (`SamplerScratch`, optional).
- **The speculative loop draws on its own thread, row by row, and stops at the first draft
  that is not the sampled token.** The rows after it were never used.

Result (`results/sampler/step151/summary.txt`; step 150's 31 runs again):

- 31 of 31 outputs equal to step 150's.
- The sampling span: 3.8 -> 2.4 ms per step (means, traced runs, `where.txt` section 4).
- Held-out mean, one run each: greedy 44.94 t/s, sampled 43.06 (-4.2%; step 150: 44.76 and
  41.79, -6.6%). Greedy moved by +0.4% between the two sets, which is the size of what one run
  each can show.

What is left is ~0.5 ms per draw times one draw per accepted draft plus one: 2.4 ms of a
~65 ms step, which is most of the remaining 4%. Not done: the sum in 8 lanes and `exp` in
float (~0.3 ms per draw, other last bits again).

One run of the 31 (`s2` on cpp-matrix) aborted at exit after writing its record: the abort in
the library's teardown of step 93 (`malloc_consolidate(): invalid chunk size`). Four reruns of
the same command exited cleanly with the same output.

## Step 152: a request mode: `--requests`

The engine's side of a server (PLAN item 29: the person's benchmark sends single-turn HTTP
requests): one process, loaded once, that reads prompts from stdin and writes replies to
stdout, each request on an empty context. The HTTP front comes next and will be a separate
program; the protocol between them is `docs/requests.md`.

- **A request** is one header line, `gen n=N [temp=T] [top_p=P] [seed=S] bytes=B`, and `B`
  bytes of prompt: the whole text with its special tokens written out, as the model's
  `encoding.py` renders it. The engine adds nothing (no BOS, no template), so tools and system
  blocks are the front's business.
- **A reply** is `prompt tokens=.. n=..`, then one line per token as it is generated, `tok <id>
  <bytes in hex>`, then `end stop=eos|length ...` with the counts and the times. The text
  includes special tokens (`</think>`, the tool-call tags), which the front needs to parse a
  reply; hex because a token can be a part of a UTF-8 character.
- **The empty context:** `Runtime::mark()` right after the load, at position 0, and
  `rewind()` before every request (step 109's saved point). Nothing new in the runtime. The
  expert cache is not reset: it carries what the earlier requests used.
- **The sampler's settings are per request**, greedy by default.
- A mistake in a request gives an `error` line and the next request is read; a header that
  cannot be followed ends the engine.

In the code: the request is read at the top of the turn loop in `run` (where `-i` reads its
line), tokens are printed by a new mode of `ThinkView` (`Ids`), and `piece()` can render
special tokens.

**Test** (`results/requests/requests_test.py`, `requests_test.txt` with `--spec`,
`requests_test-plain.txt` without): one engine, 14 generations and 5 bad requests in a row.

- 9 replies (five held-out prompts, greedy and temperature 1 / top-p 0.95 / seed 1, two of
  them twice, in mixed order, one a `--think` prompt) hash to the stored outputs of single
  runs: 9 of 9 with the speculative loop and 9 of 9 with the plain one.
- `n=5` gives the first 5 ids of the full reply and `stop=length`; a one-word reply ends with
  `stop=eos`; a think reply carries `</think>` as text.
- Four requests with a bad value give an error each; the prompt after them gives the single
  run's reply. An unknown request line closes the engine, exit code 0.
- Speeds are close to single runs' (with speculation 38.9 / 47.9 / 51.2 t/s on tcp-udp,
  cpp-matrix sampled, geometry sampled; step 151's single runs: 38.9 / 50.8 / 52.0).

Not here: a common prefix kept between requests (the benchmark's prompts share their tools
block and rules), stop strings, more than one request at a time.

## Step 153: an HTTP front: the Anthropic Messages API

`server/ds_server.py` (Python, standard library only; `docs/server.md`) starts the engine with
`--requests` as a child and serves `POST /v1/messages`. With it the person's benchmark
(`endgame_anthropic.py`: one user message, one tool, adaptive thinking with an effort,
streamed) runs against the engine unchanged.

- **Nothing of the model's format is written again.** The server loads the model's own
  `encoding.py` and calls `encode_messages` for the prompt (system block, tool schemas,
  reasoning effort, `<think>`) and `parse_message_from_completion_text` for the reply
  (reasoning, text, the tool-call block). The server converts between the API's JSON and that
  module's messages.
- **Effort:** `low` 50, `high` 75, `max` 100 are the model's names; `medium` 62 and `xhigh` 88
  are placed between them.
- **Sampling:** temperature 1 and top-p 0.95 unless the request says otherwise, and one seed
  (`--seed`) for every request, so a repeated request gives the same reply.
- **A stream** carries pings while the engine works and all the events at the end: a tool call
  can only be parsed from the complete reply.
- **One request at a time;** the others wait.
- **Refused (400):** forced tool use (`tool_choice` `any` / `tool`), tool results and other
  non-text blocks, a prefilled reply.
- A cut or malformed reply is returned as far as it can be (reasoning, text), never dropped.

**Test without the model** (`server/test_server.py` with `server/fake_engine.py`, a few
seconds; `results/server/test_server.txt`): 26 checks. The prompt the engine receives equals
`encoding.py`'s rendering; a canned reply with reasoning and a tool call becomes `thinking` and
`tool_use` blocks with the input as sent; the stream's events in order; replies cut inside the
reasoning and inside the answer; a broken tool call; an engine error; seven kinds of refused
requests, none of which reach the engine.

**With the model** (`results/server/endgame.txt`): the benchmark's script, unchanged, with
two synthetic samples (`results/server/make_samples.py`; the person's samples are not here) at
effort `low`:

| | |
|---|---|
| prompt | 1,305 tokens in 23.9 s |
| reply | 42,960 tokens in 1,117 s (38.5 t/s; 2.97 tokens per verify step) |
| result | `stop_reason` `tool_use`, one `make_guesses` call, parsed; 1 of 2 secrets guessed |

The reasoning is 107,000 characters and not a loop (693 distinct lines of 985). So one
request of this benchmark takes ~19 minutes at effort `low` on one machine. The SDK's stream
held through it on pings.

Two things found on the way:

- **A long request was silent.** For ten minutes neither the server nor the engine printed
  anything (the mistake CLAUDE.md lists first). The server now logs a line every 30 s with the
  tokens so far.
- **The benchmark's script does not start with the current Anthropic SDK** (1.17, 2.9): they
  use `httpx2` and refuse the script's `httpx.Timeout`. It ran with SDK 0.125.0.

Not done: the OpenAI Responses endpoint (the benchmark's other script), tool results in a
conversation, a reply streamed as it is generated, cancelling a request whose client left.

## Step 154: a request whose client left is cancelled

A request of the benchmark runs for ~19 minutes (step 153), and the engine answers one at a
time: a client that gave up (a timeout, Ctrl-C) blocked every later request until its reply,
which nobody would read, was finished.

- **Engine:** in `--requests` mode the signal `SIGUSR1` stops the request being generated.
  The handler sets a flag; both generation loops read it between steps, and `Runtime::prefill`
  asks before every prompt chunk (`prefill_stop`). The reply ends with `end stop=cancel` after
  the tokens made so far. The flag is cleared as the next request is read, so a signal that
  arrives between requests does nothing. `docs/requests.md`.
- **Server:** while a request runs or waits, its handler looks once a second whether the
  client closed the connection (the socket is readable with no bytes), and a failed ping of
  a stream says the same. Then `Engine.cancel` sends the signal if the engine is working on
  that request, or marks it so it is dropped when its turn comes. `docs/server.md`.

Why a signal and not a `cancel` line on stdin: the engine's main thread is in the generation
loop, so reading stdin meanwhile needs a second thread and a queue between them; a flag set by
a handler needs neither. The price is the rule in `docs/requests.md`: send it only while
waiting for a reply's `end`.

**Engine test** (`results/requests/requests_test.py`, both loops, 25 checks each, 6 of them
new):

| | speculative loop | plain loop |
|---|---|---|
| the signal 2 s into a reply of up to 4,000 tokens | `stop=cancel` after 56 tokens, 2.0 s | 46 tokens, 2.0 s |
| those tokens | the first of the full reply | the same |
| the signal 1 s into a 3,008-token prompt | `stop=cancel`, 0 tokens, after 9.8 s (the first chunk) | 8.5 s |
| two prompts after the cancels, greedy and sampled | the single runs' replies | the same |
| a signal between requests, then a prompt | the single run's reply | the same |

The first version checked the flag only in the generation loops: the signal during the prompt
then waited 40.4 s for the whole prompt. Hence `prefill_stop`.

**Server test** (`server/test_server.py`, fake engine, now 28 checks): two clients leave 60 s
replies, one plain and one a stream; the request after them is answered 0.5 s later, and the
log has both cancels.

**With the model** (`results/server/cancel.txt`): `curl --max-time 8` on a reasoning request:
cancelled after 236 tokens, 8.1 s after it arrived; the same as a stream with `--max-time 12`:
after 438 tokens; the next request answered in 0.4 s. The progress lines of step 153 are in
that log too, now from the real engine.

Left: a first request cancelled during its prompt leaves `--spec-log` to be appended, not
opened anew, by the next one.

## Step 155: the traces taken again

`results/trace/` held the traces of step 104's code (2026-11-11). Fifty steps later the same
two runs (`results/trace/README.txt`: tcp-udp and cpp-matrix, fastest single-machine settings,
`--trace`) on the current code, greedy:

| | step 104's | now |
|---|---|---|
| cpp-matrix, traced | 49.26 t/s | 49.28 |
| tcp-udp, traced | 34.88 t/s | 35.05 |

Outputs identical to the stored ones. `compare-step104.txt` (span totals per verify step, old
against new): nothing moved by 1 ms but the host's wait for the staged block on cpp-matrix
(13.79 -> 12.52 ms per step); on tcp-udp the largest are under 0.8 ms. That is as expected:
the steps since were tools, records, the sampler and the server, and the greedy path was not
changed. `critpath.txt` is from the new traces.

**The first attempt was 14% and 10% slow** (42.57 and 31.41 t/s traced). The machine had been
restarted three hours before; the page cache held 288 of the model's 508 GB. The comparison
named it at once: `engram fetch` 1.18 -> 13.13 ms per verify step (its rows read from the disk),
and the same 12 ms as die 0's wait in `ffn.shared`; everything else level. After
`cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` (2 min 42 s) the runs were as above. QUICKSTART and
step 53 say to do this after a reboot; I had not checked the uptime before measuring.

The two-machine traces (`results/trace/two-machines/`, step 113) were not taken again: the
second machine did not answer (`ssh: No route to host`).

## Step 156: the documents reorganized

Eight documents stood in the project's directory, and the README was the lab notebook: 5,680
lines, a front page only in its first 40. Reviewed with the person; decided: only README and
QUICKSTART stay there, the README becomes a front page, the rest moves to `docs/`, one
document per commit, and what is out of date is updated or removed.

- **The notebook is now this file, `docs/steps.md`** (moved with `git mv`, so `git log
  --follow` has its history). "README step N" stays as it is written in 166 code comments and
  204 results files; this file's header and the README say what it means.
- **The README is a front page:** what the project is, the status as of today (it had
  stopped at 2026-11-12: no sampling, no server), the list of documents, the docs-and-commits
  rule.
- **Removed from the old README:** the "Tools" list of commands (October's; QUICKSTART and
  REPRODUCE have the current ones) and "The original plan (items 0-8)" (superseded since
  step 8; `git show 2c8e6fe:ds-completion/README.md`).
- "What the model is", "Correctness ladder" and "Which checkpoint" stay in front of the
  steps.
- **QUICKSTART** (second commit): it said "Greedy" in its first lines and "Sampling is greedy
  only" under Limits, both wrong since step 149. Now: sampling named, a section "A server"
  (the command, what a request may carry, the limits), the results table at today's date with
  the fastest rows, the sampled row and the CPU rows (the sclk 2 milestones between them are
  in the overview), the server's limits.
- **PLAN** (third commit): 1,221 -> 501 lines. Removed: the plans that were carried out (the
  cost model of 2026-10-27, batched verification, expert placement, the adaptive cache, long
  context, prompt processing) and the first brainstorm with its sections 1-6, which started
  from the ggml GPU path removed in step 11. Kept: the items with their status, "Two
  machines", "Further optimization ideas". Steps cite the removed sections by name; PLAN's
  header says where their text is (`git show 64f5e05:ds-completion/PLAN.md`).
- **Moved to `docs/`, one commit each, every link and mention in the documents following**
  (code comments and results files keep the old names; the README lists them): PLAN.md ->
  `docs/plan.md`, OVERVIEW.md -> `docs/overview.md`, TIMING.md -> `docs/timing.md`,
  NUMERICS.md -> `docs/numerics.md`, REPRODUCE.md -> `docs/reproduce.md`, PROCESS.md ->
  `docs/process.md`. A link check over all the documents after each move: no broken link.

- **The overview and the timing page brought up to date** (a later commit). Overview: the
  scope names sampling and the three ways in; results and mechanism gain sampling and the
  server; section 6, three tables of a token's time at sclk 2 before the cache, is now today's
  step in one table with two sentences on where it started; "What is next" says that
  optimization is paused and what the server work leaves open; the code map has the ten files
  it lacked. Timing: sections "Sampling" and "A request through the server", the traces of
  step 155, and what a reboot costs. No new measurement: every number is a step's.

## Step 157: tool results in a conversation (for the pi harness)

The person wants the server to work with the pi coding harness (`~/projects/pi`, cloned, not
installed), reusing the Anthropic endpoint: first working in principle, then efficient. This
step is the reading of pi's client, the one thing it showed missing, and pi run against the
server (Node 26.13.1, installed by the person after the change).

**What pi sends** (`packages/ai/src/api/anthropic-messages.ts`), against the server:

| pi | the server before this step |
|---|---|
| `POST /v1/messages?beta=true`, always `stream: true`, any key | read |
| `system` as text blocks; tools with `cache_control`, `eager_input_streaming` | read, the extra fields ignored |
| reads `message_start`, the block events, `message_delta`; skips events it does not know (`ping`) | sent |
| aborts by closing the connection | cancelled (step 154) |
| earlier replies as `thinking`, `text`, `tool_use` blocks | **refused** |
| tool results as `tool_result` blocks in a user turn | **refused** |
| a thinking block with an empty signature goes back as a *text* block (unless `allowEmptySignature`) | **the signature was empty** |
| images, if the model is declared to take them | refused: declare it text only |

**The change** (`server/ds_server.py`): a user turn may hold `text` and `tool_result` blocks, an
assistant turn `text`, `thinking` and `tool_use`; they become the messages `encoding.py`
already renders (`content_blocks`, `reasoning_content`, `tool_calls`). `is_error` is not passed
on. A thinking block's signature is the fixed word `ds-completion`, so that any client sends
reasoning back as reasoning.

**What `encoding.py` does with a conversation** matters for the next step: with tools in the
request it keeps the reasoning of every earlier reply, so a loop's next prompt is the last
prompt, the reply as the model wrote it, and the new turn. The test checks it to the
character, for a reply with reasoning and a tool call sent back as blocks: the tool call
rendered from JSON is the text the model wrote. (For other argument values the two may
differ, so a kept prefix should be compared by tokens.)

**Test** (`server/test_server.py`, fake engine, 28 -> 35 checks; `results/server/test_server.txt`):
the signature; a tool loop's second request (the prefix, the rendered result, answered); the
same without thinking (the call in the prompt, the reasoning not); refused: an image in a tool
result, a `tool_use` block from the user, `redacted_thinking`.

**pi against the server** (`results/server/pi.txt`; pi at f0f9478a2 run from its sources,
`npm install --ignore-scripts`, `npm run build`). pi's side is one file, `server/pi-models.json`,
copied to `<agent dir>/models.json`: `api: "anthropic-messages"`, the server's address without
`/v1`, `compat.forceAdaptiveThinking` so that pi sends an effort and not a token budget, its
thinking levels mapped onto the five efforts, text only, `contextWindow` 65,536 as `--max-ctx`.
First with the fake engine (pi ran the canned `read` call and sent the result), then the model,
fastest single-machine settings, pi's print mode at `--thinking medium`:

| task | requests | wall | of it prompts | result |
|---|---|---|---|---|
| "What does notes.txt say?" | 2 | 47 s | 43.6 s | one `bash` call, the file's line quoted |
| write `primes.py`, run it, change it to print the count, run it again | 3 | 71 s | 57.5 s | `write` + `bash`, `edit` + `bash` (two calls in a reply, both times), "Count: 17", the file correct |

- pi sends 4 tools (`read`, `bash`, `edit`, `write`) and a system prompt: ~1,500 tokens before
  the first word of the task.
- Every prompt started with the last request's prompt and the reply as the model wrote it
  (3 of 3 pairs, from the dumps), tool calls rendered back from JSON included.
- The model wrote no reasoning in any of the 5 replies at effort medium (`</think>` at once);
  pi then sends no thinking block, and the prompt is the same.
- **101 of the 118 seconds were prompts** (8,952 tokens, 54-110 t/s; replies: 721 tokens in
  14 s, 40-60 t/s), because every request is processed from its first token. Of the last
  request's 2,312 prompt tokens, 2,186 were the request before it and its reply.
- `pi -p` waits for its standard input to end: `< /dev/null`, or it hangs before any request.

**Next:** a context kept between the requests of a conversation; then a reply streamed as it
is generated (pi shows nothing until a reply is complete).

## Step 158: a context kept between requests

Step 157's two pi tasks spent 101 of 118 seconds on prompts, and each request's prompt was the
last one, its reply, and a new turn. The engine ran every request from an empty context.

**The change** (`apps/ds_completion.cpp`, the request loop; `Runtime::tokens()`): after a
request the context is left as it is: its prompt and the reply's tokens that were run. If the
next prompt's tokens start with all of them and have more, only the others are run; any other
prompt starts on the empty context as before (`rewind()` to the mark at 0). The comparison is
of token ids, so a request's reply is the one it gets on an empty context: the state at a
position does not depend on how the tokens before it were split into forwards (prompt chunks,
verify steps, single tokens), which the gates of prompt processing and of speculative decoding
already rest on. The protocol (`docs/requests.md`): `kept=N` in the `prompt` and `end` lines;
`keep=0` in a request forces the empty context, for the test. No new saved point in the
runtime: this is all or nothing.

**The gate** (`results/requests/requests_test.py`, 25 -> 37 checks, both loops;
`requests_test.txt`, `requests_test-plain.txt`). The earlier 25 pass unchanged (replies equal
single runs' stored hashes). New: five conversations, each a request, then its prompt + reply +
a new turn, then the same again with `keep=0`. The treatment is asserted apart from the result:
`kept` must be above 0 in the one and 0 in the other, and equal to what the context held.

| conversation | prompt tokens | kept | prompt time | with `keep=0` | replies |
|---|---|---|---|---|---|
| a short chat, greedy | 27 | 18 | 0.20 s | 0.27 s | equal |
| the same, sampled (temp 1, top-p 0.95, seed 1) | 27 | 18 | 0.13 s | 0.27 s | equal |
| a reply with reasoning, cut by `n` | 212 | 204 | 0.13 s | 1.63 s | equal |
| a 3,000-token prompt, 1,000 more | 4,046 | 3,037 | 10.08 s | 39.92 s | equal |
| a 300-token reply, sampled | 335 | 324 | 0.29 s | 2.60 s | equal |

(Times of the speculative run; the plain loop's: 9.99 and 39.41 s for the long one.) Also: the same request again
(kept 0: the context holds its reply; the same reply), and a prompt cancelled between its
chunks and sent again (kept 512 of 3,008: it goes on from the chunks that ran; the reply of
the empty context). In all five conversations the reply's text, tokenized again as a part of
the next prompt, gave the tokens the model had written.

**The server** reports `kept` as the usage's `cache_read_input_tokens` and the other prompt
tokens as `input_tokens`, as the API counts a cached prefix; its log line says `(N kept)`
(`server/test_server.py`, 36 checks).

**pi again** (`results/server/pi-kept.txt`; step 157's two tasks, the same commands):

| | step 157 | step 158 |
|---|---|---|
| task 1 (2 requests) | 47 s | 33.1 s |
| task 2 (3 requests) | 71 s | 30.1 s |
| prompts, all 5 requests | 101.1 s | 46.3 s |
| replies | 14.3 s | 14.4 s |

The three follow-on requests kept 1,565, 1,818 and 2,186 tokens and ran 96, 116 and 126 in
1.1-1.3 s each (step 157: 15-25 s). The five raw replies are step 157's, character for
character.

**What is left:** 42.7 of the 46.3 prompt seconds are the two first requests: pi's system
prompt and tools, ~1,500 tokens, the same in both tasks up to character 6,216 of ~6,300. A new
conversation shares that but not the whole context, so it runs whole. Keeping it needs a
second saved point in the runtime (`mark()` has one, used for the empty context) and a rule
for where to put it. Also seen: the engine's first request after its load took 28.8 s for
1,505 tokens against 13.9 s for the next first request of 1,540.

**Next:** a saved point at the end of what two conversations share; a reply streamed as it is
generated.

## Step 159: the pi setup as a document

The person wants to repeat the pi setup by hand on the second machine. `docs/pi.md`: the
engine and server (with what QUICKSTART does not say for this use: the model's `encoding.py`,
`--host 0.0.0.0` for a client on another machine), Node and pi from its sources at the tested
commit, `models.json` field by field, the commands, what to expect (no text until a reply is
complete, ~15 s before a conversation's first reply, a second for the later ones, compaction
at ~49,000 tokens), a table of failures seen in steps 157-158 with their causes, and what was
not tested (only print mode on two small tasks was run). No code changed, nothing measured;
the statements about pi (the settings file, compaction's threshold, Escape) are from pi's own
documents at f0f9478a2.

## Step 160: a saved point for what conversations share

After step 158, 43 of the 46 prompt seconds of pi's two tasks were the first request of each:
the system prompt and tools, ~1,500 tokens, the same in both. A new conversation does not go
on from the context, so it ran whole.

**The runtime** (`hip/rt.h`): `mark(slot)` / `rewind(slot)`, two saved points where there was
one, each with its own copies of the window rings, the compressors' partial groups and the
draft's rings. Slot 0 is what it was (the empty context in `--requests`, the turn's start in
`-i --think`). A point is only the state *at* a position: the compressed caches before it are
not copied, so a point is good only while the positions before it hold its tokens. The
two-machine messages MARK and REWIND carry the slot (the protocol's version 2 -> 3: both
machines need this build).

**The rule** (`apps/ds_completion.cpp`, the request loop), in the order tried for a prompt:

1. it starts with all of the context and has more: it goes on from there (step 158);
2. it starts with the point's tokens: back to the point, the rest runs;
3. else the empty context. If its first 64 tokens or more are the context's, the prompt runs
   up to where the two part, the point is saved there, and the rest runs.

A prompt that does not start with all of the point's tokens ends the point before anything
else: it will run over the positions before it. The point never moves otherwise. So it is set
by the second of two conversations and used from the third on. `saved=N` in the `end` line
says where it is.

**The gate** (`results/requests/requests_test.py`, 37 -> 46 checks, both loops). Conversations
with one system prompt of ~1,500 tokens, each compared with the same request on an empty
context (`keep=0`), `kept` and `saved` asserted apart from the replies:

| | kept | prompt time | with `keep=0` | replies |
|---|---|---|---|---|
| the first conversation | 0 | 14.3 s | | |
| the second (sets the point at 1,518) | 0 | 14.5 s | | |
| a third | 1,518 of 1,525 | 0.14 s | 14.44 s | equal |
| a fourth, a 45-token reply | 1,518 of 1,540 | 0.31 s | 14.60 s | equal |
| the same, sampled | 1,518 of 1,540 | 0.29 s | 14.61 s | equal |
| the second one again | 1,518 of 1,526 | 0.15 s | 14.44 s | equal |

Also: a conversation's next turn goes on from the context, not from the point (kept 1,527); an
unrelated prompt gives its single run's stored reply and ends the point; the next three
conversations keep 0, 0 and 1,518 (the point set again by the second); a system prompt that
shares half with the point's ends it and gets a point at the shared part (755), which the next
two conversations use. The 37 earlier checks pass unchanged.

- **The first version of the test failed 5 checks, all of them the test's.** Its users' turns
  were "Name a color." / "Name a fruit.", so the two prompts parted two tokens into the user's
  turn and the point was at 1,520; the differently worded fourth conversation ended it and
  set one at 1,518. Every reply was equal to its control throughout. The rule was left as it
  is (it corrects itself at the cost of one prompt run whole); the test's turns now start with
  different words, and `docs/requests.md` says it.

**pi** (`results/server/pi-saved.txt`): step 157's two tasks and a third, each a new
conversation.

| | step 157 | step 158 | step 160 |
|---|---|---|---|
| task 1 (2 requests; the server's first) | 47 s | 33.1 s | 32.9 s |
| task 2 (3 requests) | 71 s | 30.1 s | 30.5 s |
| task 3 (3 requests) | | | 7.6 s |

Task 2's first request ran whole and set the point at 1,497 tokens; task 3's first request
kept them and ran 18 tokens in 0.3 s (task 2's: 14.1 s). So the first two conversations after
the server's start cost what they did, and those after them start at once.

**Not tested:** two machines (the slot in MARK / REWIND; the second machine was not
reachable). The second point's device memory was not measured.

**Next:** a reply streamed as it is generated.

## Step 161: a reply streamed as it is generated

A stream got pings and then the whole reply at its end (step 153): a tool call can only be
parsed from a complete reply, and the server parsed everything at once. With pi a person
watches the reply, and a long reasoning was minutes of nothing.

**The change** (`server/ds_server.py`, class `Live`). The model writes reasoning, `</think>`,
text, and perhaps a tool block that starts with two line ends and `<｜DSML｜ calls`. `Live` is
fed the engine's bytes token by token and returns events: the reasoning as `thinking_delta`,
the text as `text_delta`. What could still become `</think>` or the tool block's start is held
back until the next bytes decide (in practice the two line ends). From the tool block on
nothing is sent; at the end the whole reply is parsed as before (`parse_reply`, the model's
`encoding.py`) and the tool calls go out as blocks. If the block did not parse, or the reply
was cut inside it, its characters are sent as text, which is what the plain message has.

- `message_start` is sent when the engine takes the request (the `prompt` line), with the
  prompt's token counts; pings fill the silences as before.
- The engine's thread puts what it reads on a queue and the request's thread sends it, so
  one thread writes to a connection.
- The invariant: the thinking and text a stream carried are the message's blocks. `Live`
  checks it at the end of every reply and the log says if it ever fails.

**Tests.** `server/test_server.py` (38 checks): the stream of a slow canned reply
(`message_start`, pings, three blocks, the thinking and text in pieces, the tool call whole),
and `Live` alone over 180 cases: every canned reply and five odd ones (text ending in line
ends, `\n\n<b>`, a start of `</think>` inside reasoning, an empty reply, a reply cut inside a
tool block), in both modes, in pieces of 1, 2, 3 and 7 bytes and whole. The blocks rebuilt
from the events equal `parse_reply`'s in all of them.

**With the model** (`results/server/stream_check.py` -> `stream.txt`): three requests, each as
a stream and then plain (one seed: one reply).

| | tokens | the first piece | the last | `message_stop` | rebuilt = plain |
|---|---|---|---|---|---|
| reasoning and text | 205 | 1.25 s (thinking), 3.23 s (text) | 5.55 s | 5.67 s | yes |
| text and a tool call | 50 | 3.26 s (text) | 3.71 s; the call at 4.35 s | 4.35 s | yes |
| no reasoning | 38 | 0.21 s | 1.81 s | 1.86 s | yes |

- **The first version sent `message_stop` a second late** (6.50 s where the last piece came at
  5.48): the sending loop waited out its one-second timeout after the engine's thread had
  ended. The thread now puts an end mark on the queue. Seen only because the check prints
  the events' times.
- The stream of a pair took longer than the plain request (5.67 against 4.16 s). The engine's
  own generation was slower in the first of each pair (46.4 / 43.5 / 23.2 t/s) than in the
  second (55.5 / 49.4 / 29.1), and the order was not swapped: this is not a measure of what
  streaming costs, and none was taken.

**pi** (`results/server/pi-stream.txt`): step 160's three tasks with `--mode json`, which
prints pi's own events: task 2 had 113 `text_delta` updates and 4 tool calls, each whole. The
tasks' results are right, the replies those of step 160 where the prompts are (the dates of an
`ls -la` differ in three). Only print mode again: the interactive interface was not driven.

**Left open:** a tool call in pieces (the model writes its arguments as tagged text, not JSON:
it would need a second, incremental parser); the OpenAI endpoint; forced tool use.

## Step 162: the server's observability, as a plan

pi works well against the server (the person, 2026-11-19). Before
it runs for long: what would we need to see? Two things were asked for: a metrics page over
HTTP in the Prometheus format, and logs detailed enough to debug the kept context and the
saved point. `server/ds_server.py` and the `--requests` loop of `apps/ds_completion.cpp` were
read for it. No code changed, nothing measured.

**Found:**

- The server already has most of a metrics page's numbers: per request the prompt's tokens,
  the kept ones, both times, the wait for the engine, the stop reason, a reply that did not
  parse, a stream that differs from its message.
- The engine computes the whole choice of context (how much the prompt shares with the context
  and with the saved point, and which of three ways it takes) and reports `kept=N`. Where a
  prompt parts from the context is printed nowhere.
- The server does not read the `saved` field of the engine's `end` line.
- `GET /health` answers ok whether the engine's process lives or not.
- `--dump` leaves out the cancelled, failed and refused requests.
- The time to a reply's first token is not taken.
- `--record` is refused with `--requests`: the record's counters are not there for a request.
- Both tests read the protocol's lines as key=value pairs, so new fields in `prompt` and `end`
  do not break them.

**Planned** (`docs/plan.md`, item 33, "Observability of the server"), four steps, one commit
each: (1) the engine's choice of context in the `prompt` line and on stderr, with the tokens
around the place where a prompt parts from the context; (2) `GET /metrics`, counters and
gauges written by hand, and `/health` looking at the engine; (3) a log of requests as JSON
lines, the time of day on the log's lines, dumps for requests that did not complete; (4) the
speculative loop, the routed experts, bytes read from storage and the GPU clock in the `end`
line, for a second set of metrics.

No histograms: nothing collects the metrics yet (the person will set a collector up), and one
request runs at a time. The plan is to be reviewed by the person before any of it is built.

## Step 163: a trial build against upstream llama.cpp

The setup asks for llama.cpp at a PR's commit, fetched by its hash (step 95,
`docs/llama-cpp-upstream.md`). The person asked what would make the setup shorter, for pi on
one machine above all, and whether upstream would do now.

**Upstream is as it was for these files.** At master `e626db7ca` (2026-11-20) ggml's type 43
is still plain fp8 (`F8_E4M3`) and the PR's MXFP8 is not there: upstream's own programs still
misread the Engram table.

**But we need little from the fork.** Every ggml function our code calls is in upstream. Three
places depend on type 43 being MXFP8: the type's name in `src/model.cpp`, the GPU runtime's
fetch of the Engram rows (ggml's `to_float` and row size), and the CPU-only path's
`ggml_get_rows` on the table.

**The trial** (`results/dist/trunk-trial.patch`; a copy of the sources outside the repository,
whose code is unchanged): the type checked as the number 43, and the rows decoded by 20 lines
of our own in the GPU runtime (33 bytes per 32 values: one e8m0 scale, 32 e4m3 codes). Built
against upstream `e626db7ca`:

| check | result |
|---|---|
| `make`, `make HIP=1` (`ds-completion`, `ds-inspect`, `fp8pack`) | no errors |
| the 8 held-out prompts, the held-out command | every output the stored one's bytes (`results/dist/trunk-trial.txt`) |
| `fp8pack` | `dense-fp8.gguf` byte-identical, 18.8 s |

The machine had been restarted (the page cache held 1 GB): the model files were read through
first, 3 min 13 s. The speeds in the results file are single runs and are not a measurement.

**Not covered:** the CPU-only path (it would read the rows as plain fp8 on upstream, without
an error, until it uses the same decoder); the gates against the reference; the other programs;
a check at load of the tensor's size in the file; two machines.

**Decided:** documented only (the person). The pin stays. What a move would take is in
`docs/llama-cpp-upstream.md`, "A trial against upstream": the decoder on both paths with the
size check (the code then builds against the pin and upstream alike), then the gates against
upstream and the documents naming a tested upstream commit.

Also looked at for a shorter pi setup, not done: a `scripts/serve.sh` like `scripts/chat.sh`
(the server's eight-line command as one); a command that fetches only the model's `encoding/`
directory (the default path is inside the 510 GB base download, which pi does not need); pi's
defaults in its `settings.json`.

## Step 164: the engine says how it chose the context

Plan item 33, its first step. The engine computed its whole choice of where a request starts
(step 158: on from the context; step 160: from the saved point; else empty) and reported one
number, `kept=N`. A prompt that did not go on from the context left no trace of why.

**In the protocol.** The `prompt` line has three more fields: `ctx` (the tokens the context
held), `common` (how many of them the prompt starts with), `how` (`continue`, `saved`,
`empty`). The server and both tests read the line's fields by name, so nothing else changes.

**On stderr**, one line per accepted request, and a second one when the prompt and the
context part before either ends. From the test, the second and third conversations with one
system prompt:

    # context: held 1527 tokens, the prompt (1526 tokens) starts with 1518 of them: it starts on the empty context; a point is saved at 1518
    # context: they part at token 1518: the context has 10232 "Color", the prompt 40 "F"; before it " jumps over the lazy dog. <｜User｜>", then the context "Color: name one.<｜Assistant｜></think>Brown", the prompt "Fruit: name one.<｜Assistant｜></think>"
    # context: held 1528 tokens, the prompt (1525 tokens) starts with 1518 of them: it returns to the saved point (1518 kept)
    # context: they part at token 1518: the context has 40 "F", the prompt 31060 "City"; before it " jumps over the lazy dog. <｜User｜>", then the context "Fruit: name one.<｜Assistant｜></think>", the prompt "City: name one.<｜Assistant｜></think>"

The second line has the index of the first token that differs, its id and text on each side,
the 8 tokens before it and 8 from it on each side. The first line also says when a saved point
ends, and why a prompt starts empty where there is a reason to give (`keep=0`; the prompt is
the context itself). `docs/requests.md`, "What the engine says about its choice".

The choice itself is the code of step 160 in another order: it is made first, reported, and
then the runtime is rewound (the report reads the runtime's own token list).

**Checked** (`results/requests/requests_test.py`, 46 -> 59 checks, with `--spec` and without):
the new fields on the cases the test already had (a turn that goes on: `how=continue`,
`ctx` = `common` = the tokens the context held; the same with `keep=0`: `how=empty` and the
whole prompt inside the context; the second conversation: `common` is where the point is
saved; the later ones: `how=saved`), and the stderr lines: 61 first lines for 61 accepted
requests (7 `continue`, 8 `saved`, 46 `empty`), each with its prompt line's choice in words,
and 37 second lines, the token index of each its request's `common`. Every reply is still the
single run's or the empty context's, as before. `server/test_server.py`: 38 checks, unchanged.

**Not done:** the server does not read the new fields yet (the metrics page and the log of
requests, the plan's steps 2 and 3). Not run: pi, two machines. No conversation was seen
whose next turn failed to go on, so the line has not yet explained a real miss.

## Step 165: a metrics page

Plan item 33, its second step. The server's numbers were in its log, a line per request;
nothing could be read from outside while it ran, and `GET /health` said ok whether the engine
lived or not.

**`GET /metrics`**: 17 metrics (25 series) in the Prometheus text format, written by hand: the
server still uses only the standard library. Counters: requests by outcome (`ok`, `cancelled`,
`bad_request`, `engine_error`), completed requests by stop reason, prompt tokens run and kept,
generated tokens, the engine's seconds on prompts and on replies, the wait for the engine, the
time to the first token, the engine's choice of context (step 164's `how`), replies that did
not parse, streams that differed. Gauges: the engine up, busy, requests waiting, the saved
point, the load's time, the start's time. The table and the ratios to take: `docs/server.md`,
"Metrics".

- **No histograms** (the person, 2026-11-19): nothing collects the page yet, and one request
  runs at a time. A speed is a ratio of two counters.
- **Tokens and seconds are counted for completed requests only.** A cancelled prompt's
  `prompt_tokens` are not the tokens that ran, so counting them would spoil every ratio; a
  cancelled or failed request adds to `ds_requests_total` (and, if the engine took it, to
  `ds_context_total`).
- **The page does not take the engine's lock**: it answers while a request runs.
- Every series is there from the start, at 0.

**`GET /health`** is 200 while the engine's process lives and 503 with its exit code when it
is gone.

**Not there:** `ds_context_tokens`, which the plan listed. After a cancelled request the
server cannot tell the context's size from what the engine says, and an estimate that is
sometimes wrong is worse than none; it comes with step 4, when the `end` line gets more
fields. `ds_engine_load_seconds` was added.

**Checked without the model** (`server/test_server.py`, 38 -> 48 checks; `fake_engine.py` now
writes `how` and `saved` and can exit in place of a reply):

- the page before any request: the format, every counter at 0, the engine up;
- while one request runs and another waits: busy 1, waiting 1, answered at once;
- after a request that kept 300 tokens and left a saved point at 280: both on the page;
- the totals against the test's own count of what it sent: 11 answered, 11 refused, 2 clients
  that left; stop reasons 6 / 3 / 2; one reply that did not parse; the fake engine's seconds
  (0.5 and 0.1 a request) times the answered ones;
- the engine gone: the request is a 500, `ds_engine_up` 0, one `engine_error`, `/health` 503
  with exit code 3.

My first expected count of tool calls in that test was wrong (4; the canned replies have 3,
the fourth is the broken one): the test failed on its own expectation, not on the server.

**With the model** (`results/server/metrics_check.py` -> `metrics.txt`): four requests with one
1,500-token system prompt and one refused.

| | on the page | from the usages |
|---|---|---|
| requests | `ok` 4, `bad_request` 1 | 4 answered, 1 refused |
| prompt tokens run | 3065 | 1525 + 1526 + 7 + 7 |
| prompt tokens kept | 3045 | 0 + 0 + 1518 + 1527 |
| the choice of context | `empty` 2, `saved` 1, `continue` 1 | the engine's four `# context:` lines |
| the saved point | 1518 | the second request's line: "a point is saved at 1518" |
| prompt seconds | 39.6 | the log's 25.1 + 14.2 + 0.2 + 0.1 |

**The documents' commands corrected:** `server/ds_server.py`, `server/test_server.py` and
`results/requests/requests_test.py` are not executable in the repository (mode 644), and
QUICKSTART, `docs/pi.md`, `docs/server.md` and `docs/reproduce.md` wrote them as commands: as
written they failed with "Permission denied" (found when this step's check started the server
that way). The documents and the three files' own usage lines now say `python3 <file> ...`
(the person: the documents rather than the files' modes). Not looked at: the other scripts
with mode 644 that a document may write as a command (five in `scripts/`, the ones in `ref/`
and under `results/`).

**Not done:** no collector is set up (the person's). The numbers are per server process: they
start at 0 with every start. Two machines: the back has no page.

## Step 166: a log of requests

Plan item 33, its third step. The server said what it did in lines for a person to read, on
stderr, without the time; `--dump` kept only the requests that completed, which are the ones
least in need of a second look.

**`--log FILE`**: one JSON object per request to `POST /v1/messages`, on one line, appended
when the request ends, however it ends. 30 keys, the same in every line, `null` where the
request did not get that far: the arrival's time, the request's number and message id, the
outcome and the error's text, what was asked (stream, mode, messages, tools, the sampler),
where the engine started (step 164's `how`, `ctx`, `common`), the tokens (prompt, kept,
saved point, generated), the seconds (prompt, generation, the wait for the engine, to the
first token, the total), the stop reason, the blocks, the note of a reply that did not parse,
whether the client left, whether a stream differed. The table: `docs/server.md`, "The log".

**`--dump DIR`** now writes a file for every request, with its outcome: a cancelled one has
the prompt and the reply so far, a refused one the request and the error, a body that is not
JSON its text.

**The server's stderr lines start with the date and time.** The engine's own lines do not.

**In the code** the handler has one way out, `done(outcome, error)`, which counts the request
for `/metrics`, writes the log line and the dump; step 165's four scattered counts are gone.
A request gets its number when it arrives, not after it is found valid, so the refused ones
have a number and a message id too. The line of a plain request is written before its reply
is sent: a client that has its reply finds the line there.

**Checked without the model** (`server/test_server.py`, 48 -> 58 checks): 25 lines for the 23
requests the test sent and the 2 clients that left, numbered 1 to 25, every line with the same
keys; by outcome what `/metrics` counted; the answered ones' tokens add up to `/metrics`'
totals; the request that kept 300 tokens (`how=continue`, `saved` 280); the slow stream's
first token after 2.5 s; the two cancelled ones; the refused ones' errors; the times in order;
25 dump files, one per line; the stderr lines' dates. Two of the test's own expectations were
wrong when first run (I took "no prompt tokens" for "never reached the engine", but a request
the engine refuses has none either): the test was corrected, the server was right.

**With the model** (`results/server/log_check.py` -> `log.txt`): step 165's four requests, one
refused, a stream with reasoning whose client leaves after 3 s, and one more.

| request | outcome | how | ctx | common | kept | saved | prompt s | wait s |
|---|---|---|---|---|---|---|---|---|
| 1 a conversation | ok | empty | 0 | 0 | 0 | 0 | 24.9 | 0.01 |
| 2 another, the same system prompt | ok | empty | 1527 | 1518 | 0 | 1518 | 14.2 | 0.01 |
| 3 a third | ok | saved | 1528 | 1518 | 1518 | 1518 | 0.15 | 0.01 |
| 4 its next turn | ok | continue | 1527 | 1527 | 1527 | 1518 | 0.15 | 0.01 |
| 5 no messages | bad_request | | | | | | | |
| 6 a stream with reasoning, the client leaves | cancelled | empty | 1536 | 2 | 0 | 0 | 4.1 | 0.01 |
| 7 a conversation as the first | ok | empty | 512 | 2 | 0 | 0 | 14.3 | 1.1 |

**What the first run showed, unasked.** Request 6 has the same system prompt as the others
and shares 2 tokens with them. The engine's line says why:

    # context: they part at token 2: the context has 3476 "You", the prompt 77666 "Reason"; before it "<｜begin▁of▁sentence｜><｜System｜>", then the context "You answer in one short line. Background", the prompt "Reasoning Effort: 75 ("

With reasoning, `encoding.py` writes the effort line at the start of the system turn, before
the system prompt. So a conversation with reasoning and one without part at token 2: the saved
point ended (`saved` 0), and request 7, a conversation like the first three, ran its 1,525
tokens again (14.3 s where the point gave 0.15). For pi this means: a change of `--thinking`
between off and a level costs one whole prompt, twice (once to end the point, once more before
a new one is set). Two efforts would part a few tokens later by the same line; that was not
run. `docs/pi.md` says it now. Nothing is changed: the format is the model's.

Also in the table: request 7 waited 1.1 s for the engine (the server looks for a client that
left once a second, then the engine stops at its next prompt chunk), and after the cancelled
prompt the context held 512 tokens, one chunk.

**Not done:** the engine's lines have no time and no request number (the number would be a
new field of the protocol; the plan's "Not decided"). The log is one file that grows: no
rotation. Not run: pi, two machines.

## Step 167: a script that starts the server with every log

The person installed pi system-wide (`~/.local/bin/pi`, 1.5.2) and will use it against the
server for real work; the logs of that use are to show what needs changing. Until now the
server was started by an eight-line command, and nothing was kept unless `--log` and `--dump`
were added by hand.

**`scripts/serve.sh`** starts `server/ds_server.py` with the fastest single-machine settings
(those of `scripts/chat.sh`, with `--cache-wait`) and writes a new directory per start,
`~/ds-serve-logs/<date>-<time>/` (`DS_LOGS` for another place):

| file | what |
|---|---|
| `info.txt` | the commit, the whole command, each GPU's clock, the page cache's size, pi's version and `models.json` |
| `server.log` | the server's and the engine's lines, also shown on the terminal |
| `requests.jsonl` | `--log` (step 166) |
| `dump/` | `--dump`: each request's body, prompt, raw reply, message |
| `metrics.prom` | `GET /metrics`, taken again every 60 s: what is left after a stop is at most a minute old |

The directory is outside the repository: its files hold the prompts and the replies.

- **`--max-ctx` is 262,144** (`MAX_CTX` for another). pi compacts a conversation by the
  `contextWindow` of its own `models.json`, and a prompt past the engine's limit is refused: so
  the script reads that number (of the provider whose `baseUrl` ends with the port) and warns
  when it is larger than the engine's.
- It warns when the page cache holds less than the model (after a reboot: step 155) and when
  a GPU's clock is on auto (step 116).
- More options of the server after the script's name (`--seed 3`, `--host 0.0.0.0`); `PORT`.

**Tried** (`results/server/serve.txt`): started, `listening 8080` after about a minute, one
request answered, the five files there, `metrics.prom` after a minute, and on SIGINT to its
process group (what Ctrl-C sends) the server and the engine stopped with exit code 0 and
nothing was left running. Twice: with `--max-ctx 1000000`, then with 262,144. The warning: a
start with `MAX_CTX=65536` against a `contextWindow` of 262,144, stopped after 2 s.

**The context's size, and why 262,144.** The script's first version took `--max-ctx` from
pi's `models.json`, where the person had `contextWindow` 1,000,000 (the repository's
`server/pi-models.json`: 65,536). The trial showed what that takes from the home die (step 60:
6.4 KB per position):

| `--max-ctx` | sequence state on GPU 0 | expert slots on GPU 0 | experts on the four GPUs |
|---|---|---|---|
| 65,536 (the engine's default) | 0.50 GB | 802 | 6,097 |
| 262,144 | 1.76 GB | 734 | 6,029 |
| 1,000,000 | 6.50 GB | 480 | 5,775 |

The person chose 262,144 as the script's default and set the same in pi's `models.json`. What
any of the three costs in speed was not measured.

**Not done:** nothing rotates or removes the directories. pi itself was not run here. The
documents' section on installing pi still describes a build from its sources (step 159); how
the system-wide one was installed is not recorded.

## Step 168: the Engram table decoded by our own code, and upstream llama.cpp

**The question.** Step 163 found that one tensor ties us to the llama.cpp PR's commit: the
Engram table, type 43, MXFP8 in the PR and in our files, plain fp8 upstream. It left two steps:
our own decoder on both paths with a check at load, then the gates against upstream. Predicted:
the GPU runtime gives the same bytes; the CPU path may move, since it runs ggml's CPU kernels
and upstream has a month of changes to them.

**First, the build before the change** (9a57d22, the pin; the machine had been restarted, the
model files read through first; sclk 6): the held-out matrix of step 138 again. 16 of 16
outputs the stored ones, the mean of the 8 prompts 44.78 t/s (step 138: 44.70).

**The change** (five files):

- `src/model.h`, `src/model.cpp`: `mxfp8_to_float` (a block is 33 bytes: one e8m0 scale, 32 e4m3
  codes; value = e4m3(code) x 2^(scale - 127), a table of the 256 codes) and `mxfp8_row_bytes`.
  The tensor's type is asked for as the number 43 (`kEngramGgmlType`), not as a name only the
  PR has.
- A check at load: the tensor's place in its shard, up to the next tensor or the file's end,
  must hold 33 bytes per 32 values. The number 43 does not say which format it is; the size does.
- `hip/rt.cpp`, `fetch_engram_rows`: our decoder and row size in place of the type's `to_float`
  and `nb[1]`.
- `src/forward.cpp`, `Engine::engram`: the rows are gathered and decoded on the host into the
  chunk's plan and enter the graph as an input tensor, in place of `ggml_get_rows` on the table.
- `apps/ds_inspect.cpp`: the table's size from the layout.

**Measured** (`results/dist/trunk.txt`). Three builds: before the change on the pin ("old"),
the change on the pin, the change on upstream `e626db7ca` (every program of `make` and
`make HIP=1`; the only new warnings are two deprecations in `ref/llama_logits.cpp`).

| check | the change, pin | the change, upstream |
|---|---|---|
| per-layer gate, `hello-fp32`, CPU path | 201 rows, 0 differ from old | 0 differ |
| per-layer gate, GPU runtime | 0 differ | 0 differ |
| end-to-end logits (5 steps), both paths | the same bytes as old | the same bytes |
| held-out, 8 prompts x 2 | 16 of 16 the stored outputs | 16 of 16 |
| `fp8pack` | | the same bytes as `dense-fp8.gguf`, 19.1 s |

The prediction was half wrong: the CPU path did not move either. On this gate, at least: it is
5 steps of a 10-token prompt, and the longer golden sets were not run against upstream.

**Speed** is as step 138's on both builds, within about 2%, with one exception that was looked
at: cpp-matrix on the pin build gave 52.60 and 52.78 t/s (-3.4%), while the upstream build,
with the same change, gave 55.28. Three more runs of the pin binary right after: 54.87, 55.03,
54.99. The slow pair belongs to that session, not to the change; what caused it was not found.

**Decision.** QUICKSTART names upstream `e626db7ca` for a new setup. The PR's commit stays in
the documents for what still needs it: llama.cpp's own reading of these files (`llama-cli` as
the CPU baseline, `bin/llama-logits`), which upstream gets wrong with no error. This machine's
build stays on the pin, and every earlier result was made with it.

**Not done:** the longer golden sets against upstream; a file that fails the size check (it
was only seen to pass on the real one); two machines and `--requests` with the upstream build;
`llama-logits` does not refuse to run when built against upstream. A serving-only build would
need only the tokenizer and the GGUF reader of llama.cpp (the ggml graphs are the CPU path's);
not tried.

## Step 169: which code is the product

**The question.** The project began as an experiment and now serves requests. The code that
does so stands among what was built to get there: a second implementation on the CPU, tests
against the reference, benchmarks, analysis scripts. All of it is kept. Can a line be drawn
between the product and the rest, in the documents first, without moving a file?

**How it was read:** the Makefile (what each program is built from), the includes, and which
llama.cpp functions each source file calls. Not from a link map.

**The line.**

- **The product** is what runs under `scripts/serve.sh`: `bin-hip/ds-completion --rt`, the
  server, and `fp8pack`. About 10,000 lines: `hip/` (6,700), the program (1,300), the loader,
  the CPU expert kernels, the sampler. On this path no ggml graph is built or computed. From
  llama.cpp it calls the tokenizer, the `gguf_*` reader, tensor metadata (`ggml_nbytes` and
  the like) and two conversions at load.
- **What checks it:** `src/forward.cpp` (the model as ggml graphs on the CPU), `tests/`,
  `ref/`, the server's test with its fake engine.
- **Tools:** `bench/`, most of `scripts/`, `ds-inspect`, `results/`.

**Where the line does not hold yet** (four places):

1. `hip/rt.h` includes `src/forward.h`, for `engram_rows` (the Engram hash) and `Taps`. A tap
   is a layer's named intermediate copied out for `ds-test`; in serving the pointer is null and
   every `tap()` returns on its first line. The hook belongs in the layer code; only the
   type's definition stands in the wrong header.
2. `src/fakequant.cpp` holds the runtime's host functions and the reference path's graph ops.
3. `ds-completion` without `--rt` is the CPU reference path, so the product's binary links
   `forward.cpp` and ggml's CPU backend.
4. `make` builds all 9 programs (20 with `HIP=1`) as one target.

**Written:** a section in `README.md`; `docs/overview.md`'s code map in three parts, the same
rows as before, with the rule and the four places; item 34 in `docs/plan.md` with three small
steps (this map; Makefile targets; the two names out of `forward.h`) and what is deferred.

**Nothing measured, no code changed.** Deferred by the person until the three steps are done:
whether the CPU-only mode is part of the product, and where `ds-test` belongs.

## Step 170: `make product`, `make tests`, `make tools`

**The question.** Step 169's map says which programs are the product, which check it and which
are tools; `make` built all of them as one target. Can the build say the same, with `make`
itself unchanged?

**The change** (the Makefile only): three lists and a target for each.

| target | programs | more with `HIP=1` |
|---|---|---|
| `make product` | `ds-completion`, `fp8pack` | |
| `make tests` | `ds-test`, `test-fakequant`, `test-sampler`, `llama-logits` | `test-topk` |
| `make tools` | `ds-inspect`, `bench-cpu-moe`, `bench-fp4-decode` | the ten GPU benchmarks |

`all` is the three together.

**Checked** (`results/dist/make-groups.txt`):

- `make -n -B` and `make -n -B HIP=1` name the same outputs as before the change (9 and 30,
  the GPU objects included), and on the existing tree both say there is nothing to do.
- From nothing, one job (`-j1`), into a new directory:

| | CPU | `HIP=1` |
|---|---|---|
| `make product` | 19 s | 82 s |
| `make` | 58 s | 240 s |

- The product's programs built alone and built with everything: the same bytes for both CPU
  programs and for the GPU build's `fp8pack`.

**Not expected:** the GPU build's `ds-completion` differed between the two directories. Every
object `hipcc` writes differs with the path it is written to: `expert_cache.o` built twice to
one path gave the same bytes, and to another path different ones (nothing of the path is
readable in the file; presumably an id the compiler derives from it). So two GPU builds can be
compared with `cmp` only if made at the same path. It does not touch what the program computes
(step 168's upstream build, in another directory, gave the stored outputs).

**Not done:** `make -j` was not tried with the new targets. The GPU build's `product` still
compiles the whole runtime, tracing and two-machine code included: the groups are of programs,
not of the runtime's parts.

## Step 171: the runtime without the reference path's header

**The question.** `hip/rt.h` included `src/forward.h`, the header of the CPU reference path,
for two names (step 169): `engram_rows` and `Taps`. Can they get a place of their own, with
nothing changed in what any program computes?

**The change.**

- `src/engram.h`, `src/engram.cpp`: `engram_rows`, the n-gram hash that says which rows of the
  Engram table a position reads. The function's text is moved from `forward.cpp` as it was.
- `src/taps.h`: `Tap` and `Taps`, with a comment on who fills them (either path's `run_layer`,
  for `ds-test`) and that in serving the runtime's pointer is null.
- `src/forward.h` includes the two, so `ds-test` and `ds-completion` include what they did.
- `hip/rt.h` includes `engram.h` and `taps.h` in place of `forward.h`.
- The Makefile: `src/engram.cpp` and the two headers in the core lists.

What `hip/` includes from `src/` now: `model.h`, `engram.h`, `taps.h`, `fakequant.h`,
`cpu_moe.h`, `canon_math.h`, `fatal.h`.

**Checked** (`results/dist/forward-split.txt`; the pin, sclk 6, the page cache full):

| check | result |
|---|---|
| `make`, `make HIP=1` | no errors, no new warnings |
| per-layer gate (`hello-fp32`), CPU path and GPU runtime | 201 rows each, 0 differ from step 168's |
| end-to-end logits, both paths | the same bytes as step 168's |
| held-out cpp-matrix, ja-email, summary-json (one run each) | the stored outputs; 54.91, 37.32, 44.98 t/s |
| `python3 server/test_server.py` | all checks passed |

**Item 34's three small steps are done.** What is left of step 169's list: `fakequant.cpp`
holds both kinds of code, and `ds-completion` without `--rt` is the reference path, so the
product's binary still links `forward.cpp` and ggml's CPU backend. Both wait for the person's
answers (is the CPU-only mode part of the product; where `ds-test` belongs).

**Not done:** the full held-out set (3 of 8 prompts, once); two machines; `--requests` with
the model (the hash and the taps are not on a path of their own there).

## Step 172: the rounding functions without ggml

**The decision before it** (the person, 2026-11-21): `ds-completion` without `--rt`, the model on
the CPU through ggml graphs, is a reference tool and not part of the product. So the product
should in the end be buildable without it. Step 169 listed what stands in the way; the smaller
of the two is this step.

**The question.** `src/fakequant.cpp` held two kinds of code: the scalar rounding rules and
host functions, which the GPU runtime, `fp8pack` and the kernel benchmarks call, and the same
roundings as ggml custom ops, which only the reference path's graphs use. Can they part with
the same bytes everywhere?

**The change.**

- `src/fakequant.h`, `src/fakequant.cpp`: the scalar rules (`fq_e4m3_round`, `fq_e2m1_round`),
  the host functions (`fq_fp8_host`, `fq_fp8_split`) and the fp8 repack. No ggml include.
- `src/fakequant_graph.h`, `src/fakequant_graph.cpp`: `fq_fp8`, `fq_fp4`, `fq_fp4_e4`, the ops,
  with the comment on where the reference applies each.
- The scale rule both halves use (`pow2_ceil`, a static function before) is `fq_pow2_ceil`,
  inline in `fakequant.h`.
- `forward.cpp`, `bench_cpu_moe.cpp` and `test_fakequant.cpp` include the new header; the
  Makefile's core lists and `test-fakequant` have the new file.

**Checked** (`results/dist/fakequant-split.txt`; the pin, sclk 6):

| check | result |
|---|---|
| `make`, `make HIP=1` | no errors, no new warnings |
| `bin/test-fakequant data/fq` | fp8, fp4, fp4e4: 45,312 values each, 0 differ |
| per-layer gate, CPU path and GPU runtime | 201 rows each, 0 differ from step 168's |
| end-to-end logits, both paths | the same bytes |
| `bin/fp8pack` | `dense-fp8.gguf` byte for byte, 18.9 s |
| held-out cpp-matrix, ja-email, summary-json (one run each) | the stored outputs; 55.04, 37.23, 44.83 t/s |

**What is left between the product and the reference path:** one thing. `ds-completion` without
`--rt` runs `Engine` (`src/forward.cpp`), so the binary links it, `fakequant_graph.cpp` and
ggml's CPU backend. Taking it out changes the command line of the product or adds a program,
so it is its own step, not part of this one.

**Not done:** the full held-out set; two machines.

## Step 173: the CPU reference path as a program of its own

**The decision before it** (the person): the model on the CPU alone is a reference tool, and it
should be a program of its own, not an option of the product's build.

**The question.** `ds-completion` without `--rt` ran the CPU `Engine`, so the product's binary
held the reference path. Can the two part with one source file for the command line, and
with neither path's output changed?

**The change.**

- `apps/ds_completion.cpp` builds two programs. With `-DDS_RUNTIME` (`make HIP=1`):
  `bin-hip/ds-completion`, the product, the GPU runtime only. `--rt` is required; without it the
  program says so and names `bin/ds-ref`. Without the macro (`make`): `bin/ds-ref`, the `Engine`
  only; it refuses `--rt`. Before, the macro added the runtime to a program that always had the
  engine; now it chooses between them.
- `src/options.h`: `Options` (threads, context size, the rounding switch) out of `forward.h`,
  so the product's source does not include the reference path's header.
- The Makefile: `CORE` without the reference path; `REF` = `forward.cpp`, `fakequant_graph.cpp`,
  linked by `ds-ref`, `ds-test` and `bench-cpu-moe`. `ds-completion` is in `product` with `HIP=1`
  only, `ds-ref` in `tests` without it; `make bin/ds-completion` stops with a message.
- `--no-quant` is `ds-ref`'s.

**Checked** (`results/dist/ds-ref.txt`; the pin, sclk 6):

| check | result |
|---|---|
| `make`, `make HIP=1` | no errors, no new warnings |
| `bin/ds-ref` against the CPU `ds-completion` of the commit before, one prompt, 24 tokens | the same bytes, the same `output.hash`; 3.53 t/s both |
| per-layer gate and end-to-end logits, CPU path and GPU runtime (`ds-test`) | 201 rows each, 0 differ; the same bytes |
| held-out cpp-matrix, ja-email, summary-json with `--spec` | the stored outputs; 55.10, 37.25, 45.10 t/s |
| held-out cpp-matrix, summary-json without `--spec` (the plain loop, which was edited) | the stored outputs; 29.20, 28.17 t/s |
| `results/requests/requests_test.py` (the model, ~10 min) | all 59 checks |
| `server/test_server.py` | all checks |

**What the product's binary holds now.** No `Engine` symbol (53 before) and none of the graph
rounding ops; 16.7 MB against 18.5.

**The prediction's wrong half:** ggml is as much in the binary as before, 967 functions,
`ggml_graph_compute` among them. Our code no longer calls the graph functions, but the
tokenizer comes from `libllama.a`, whose model code refers to all of ggml. So "the product
takes only the tokenizer and the GGUF reader from llama.cpp" is true of what it calls and not
of what it links.

**Not done:** two machines (`--serve`, `--remote`: untouched code, not run); the full held-out
set; `-i`. The stale `bin/ds-completion` of earlier builds was removed by hand on this machine:
on another checkout it stays until `make clean`, and it is the old program. `scripts/` and the
server start `bin-hip/ds-completion --rt` and are unchanged.

## Step 174: the longer golden sets again

**The question.** Steps 168 to 173 moved code between files and programs and were each gated on
the shortest golden set (`hello-fp32`: a 10-token prompt, 5 steps) and on stored outputs. The
longer sets were last run weeks ago: the 20,000-token one on 2026-10-29 (step 35), with an
early runtime and the GPU clock at level 2. Do they still give their numbers? Predicted: the
CPU path yes; the runtime's errors against the reference might differ slightly from then.

**Run** (`results/gates-long.txt`; commit c68cd43, the pin, sclk 6): `ds-test` per layer on four
sets with the CPU reference path, and on `readme20k-s` with the GPU runtime, per layer and end
to end.

| set | path | rows | differ from the stored table |
|---|---|---|---|
| `hello-fp32-nq` | CPU | 200 | 0 |
| `readme200` | CPU | 360 | 0 |
| `readme200-fp32-nq` | CPU | 200 | 0 |
| `readme1100` | CPU | 200 | 0 |
| `readme20k-s`, 20,000 tokens | runtime | 360 | 0 |

(Every column but the time.) At 20,000 tokens also: the 9 steps' tokens are the reference's;
the logits file is byte for byte `data/ds/readme20k-rt.f32` of 2026-10-29, and the top-k dump
`data/ds/readme20k-topk.bin`; `ref/compare_logits.py` against the reference prints the stored
table (argmax 9 of 9, mean KL 2.15e-02, max 7.75e-02).

**So the prediction was too careful:** the runtime computes at 20,000 tokens exactly what it
did three weeks and some 140 steps ago, and at clock level 6 what it did at level 2.

**Seen on the way:** the 20,000-token prompt took 465 s in the end-to-end run, 1,684 s then
(prompt processing in chunks came after step 35).

**Not done:** the same with the upstream llama.cpp build (step 168 gated it on the short set
only). `data/golden/hello` (the reference as released) ran without an error, 680 rows, but no
stored table of it was found to compare with.

## Step 175: upstream llama.cpp as the only llama.cpp

**The question.** Since step 168 the code builds against upstream llama.cpp and against the
DeepSeek-V4.1 PR's commit, and the documents named both. Upstream was gated on the shortest
golden set and the held-out outputs only (step 174's "not done"). Does everything else hold
with upstream, so that the PR's commit can leave the instructions? Predicted: yes; from
llama.cpp the code takes the tokenizer, the GGUF reader and ggml's CPU operations, and none of
those differ where we use them.

**Run** (`results/dist/upstream.txt`; commit 2acdcaa, sclk 6). `~/forks/llama.cpp` moved to
upstream `master` at `e626db7ca` and was built with QUICKSTART's two `cmake` lines; here
`make clean && make && make HIP=1`, so `bin/` and `bin-hip/` are the upstream build. Then
every check there is, 90 minutes:

| check | result |
|---|---|
| unit tests (`test-fakequant`, `test-sampler`, `test-topk`) | pass |
| `fp8pack` | the bytes of `dense-fp8.gguf` |
| per layer, CPU path: `hello-fp32`, `hello-fp32-nq`, `readme200`, `readme200-fp32-nq`, `readme1100` | 0 rows differ from the stored tables |
| per layer, runtime: `hello-fp32`, `readme20k-s` (20,000 tokens) | 0 rows differ; the top-k dump the stored bytes |
| end to end: `hello-fp32` on both paths, `readme20k-s` on the runtime | logits files the stored bytes |
| `bin/ds-ref`, one prompt | step 173's output hash |
| held-out prompts, twice each | 16 of 16 outputs the stored bytes; 44.9 t/s mean |
| request test with the model, server test | 59 checks and all checks pass |

**So upstream is the one llama.cpp now.** QUICKSTART and `docs/reproduce.md` name
`e626db7ca`, a commit of upstream's main branch, and no fork or pull request. The PR's commit
stays in `docs/llama-cpp-upstream.md` as history and in the older steps, whose results were
made with it.

**`bin/llama-logits` left `make tests`.** It runs llama.cpp's own implementation of the model
(the "second opinion" of steps 2-3). Built with upstream it compiles, runs and gives wrong
logits without an error, because upstream reads the Engram table's type as another one (step
168). A check that is silently wrong does not belong in the default build:
`make bin/llama-logits FORK=<a checkout of the PR's commit>` still builds it, and its source
says so. With it gone, `make` and `make HIP=1` have no warnings.

**Seen on the way:** one held-out run (`sql-report`, the second) aborted at the process's exit
with `double free or corruption`, after its output and its numbers were written, in
`libhsa-runtime64`'s exit handler. That is step 93's abort (not repeated in 56 runs then):
a second sighting, once in 16 runs here, again after the work. `run_matrix.py` counts such a run
as failed, which is right for a gate and worth knowing when one line of a matrix says FAILED.

**Not done:** two machines (the second Mac Pro still has the PR's checkout and build; it needs
the same switch and a rebuild before the next two-machine run). The Makefile's variable is
still called `FORK` and the directory `~/forks/llama.cpp`: names only.

## Step 176: the checkout's place, the variable's name, and the Python reference

**The question.** After step 175 two names were left from the time llama.cpp was a pull
request's commit: the Makefile's variable `FORK` and the directory `~/forks/llama.cpp`. Rename
both. Predicted: only a path changes, so the same binaries.

**Done.**
- The checkout is `~/projects/llama.cpp`. A clean upstream checkout was already there (at
  `ce8caa6e6`, 2026-09-20); its `master` was moved forward to `e626db7ca` and `build/` built
  with QUICKSTART's two `cmake` lines.
- The Makefile's variable is `LLAMA_CPP`. `make FORK=<dir>` still sets it, so the commands of
  older steps run as written.

**The prediction was wrong, harmlessly.** llama.cpp's libraries built in the new directory are
not the bytes of the old ones (they hold their source's path), so the programs differ in bytes
from step 175's. Gated by the checks instead (`results/dist/llama-cpp-dir.txt`; sclk 6):

| check | result |
|---|---|
| unit tests, `fp8pack` | pass; the bytes of `dense-fp8.gguf` |
| per layer: `hello-fp32` on both paths, four CPU sets to 1,100 tokens | 0 rows differ |
| end to end, `hello-fp32` on both paths | logits the stored bytes |
| `bin/ds-ref`, one prompt | step 173's output hash |
| held-out prompts, twice each | 16 of 16 outputs the stored bytes; 44.8 t/s mean; no run failed |
| request test with the model, server test | pass |

The 20,000-token set was not run again: a path changed, not the source.

**Found on the way: the Python reference depended on the PR's `gguf-py`.** `ref/weights.py`,
which `ref/ds_ref.py` (the golden data's generator) and `ref/check_weights.py` read the GGUF
with, imported `gguf-py` from `~/forks/llama.cpp` and decoded the Engram table under the PR's
type `MXFP8`. With upstream's `gguf-py` that failed: `AttributeError: MXFP8`, and the table's
bytes came 3% short (upstream counts one byte per value). Step 175 did not see it, because its
"every check" was every check of the programs; nothing in it ran the Python reference. A grep
for the old path found it.

Fixed the way step 168 fixed the C++: type 43 is read by its layout (33 bytes per 32 values,
the byte count our own, checked against the shard's size), whichever meaning `gguf-py` gives
the number. The checkout comes from `LLAMA_CPP` in the environment, with the Makefile's
default. Checked with `ref/check_weights.py --layers 0 --experts 0`: 7,168 rows of the table
and layer 0 exact against the Hugging Face checkpoint, 80 lines ok.

**Seen on the way:** `cpp-matrix`'s second run reads 52.3 t/s after 54.9, here and in step 175
(52.27 after 54.97). Twice the same is not chance; step 168 saw this prompt 3.4% slow once
and called it session variation. The cause was not looked for.

**Also corrected:** step 175's files named the held-out script `scripts/matrix.py`; it is
`scripts/run_matrix.py --keep-out`.

**Not done:** `ref/ds_ref.py` itself was not run (golden data not regenerated); it reads the
table through the same `GGUF.raw` that the check covers. The second Mac Pro: its checkout,
its build and its place are still the old ones.

## Step 177: the documents against the state after steps 168-176

**The question.** Nine steps changed what the programs are called (`ds-ref`), what
`ds-completion` needs (`--rt`, always), what `make` builds, and where llama.cpp comes from and
lies. Do the front page, QUICKSTART, the code map and the programs' own help say the same?
Predicted: a few leftovers.

**First:** `~/forks/llama.cpp` was deleted (the PR's build and branch with it; the commit is
on GitHub, `docs/llama-cpp-upstream.md` has the commands). The build did not notice.

**How it was reviewed** (`results/dist/review-177.txt`): the options the program parses
against the ones QUICKSTART names (the same set); every path named in the current documents
against the tree (all there); a search for the old names (`fork`, `FORK`, the pin, "without
`--rt`", `bin/ds-completion`, `forward.h`); and a read of README, QUICKSTART, the code map and
the setup sections of `docs/reproduce.md`.

**Found and corrected:**

| where | what it said | now |
|---|---|---|
| `README.md`, the first command | `./ds-completion -m <model> -p "prompt"`: runs nothing since step 173 | with `--rt <dense-fp8.gguf>`, from `bin-hip/` |
| `README.md`, the map | the product is `src/` except `forward.*` | except `forward.*` and `fakequant_graph.*` (step 172) |
| `README.md`, status | nothing on what it depends on | one line: upstream llama.cpp at one commit, ROCm, Python's standard library |
| `QUICKSTART.md`, options and limits | eight statements qualified by "with `--rt`" (`-i`, `--requests`, `--max-ctx`, `--layers`, `-t`, prompt processing, GPU memory, context) | unqualified: the program has no other mode. "Without `--rt` (the CPU path)" is `bin/ds-ref` |
| `QUICKSTART.md`, results | a row "own runtime, ... `--rt ...`" as one mode among others | "every routed expert on the CPU: without `--placement`" |
| `QUICKSTART.md`, limits | the exit abort: step 96 | steps 93 and 96, and step 175's sighting |
| `apps/ds_completion.cpp` | one usage text and one header for both programs: `ds-completion` was offered `--no-quant`, `ds-ref` was offered `--rt` and the cache options | a usage text per program |
| `docs/reproduce.md`, open items | nothing on the second machine | it is as at step 167 (the PR's checkout in the old place, an older build) and what it needs |

**Looked at and left:** `docs/overview.md`'s code map and plan item 34 are current. The
Makefile's comments were rewritten in steps 170-176. "Fork", "the pin" and `FORK=` inside old
steps, old plan lines and results headers describe their own time and stay.

**Checked** after the one code change (the usage text): both programs print their own; the
short set's logits on both paths are the stored bytes; `ds-ref`'s output hash is step 173's;
two held-out prompts give the stored bytes at 46.8 and 45.1 t/s; the server test passes.

**Not looked at in this review:** `docs/timing.md`, `docs/numerics.md`, `docs/record.md`,
`docs/pi.md` and `docs/server.md` beyond the searches (none of them names what changed); the
scripts kept with old results.


## Step 178: the two-machine link out of the code

**The decision** (the person, 2026-11-22, after a review of how two machines were done): for
this model and this hardware the aim is one solid, well-tested and clear single-machine
implementation. The model in two halves does not earn its place, and goes: both of its layers,
the link between two processes (this step) and the range of layers in the runtime (the next).

**What the review found.** The work of steps 110-112 is sound and its measurements stand
(held-out mean 44.60 -> 50.85 t/s, outputs identical). Against keeping it:

- The gain is GPU room, not parallelism: the machines take turns, one always idle. +14% on
  300-token runs, 0 on the 3,000-token run whose cache had adapted, +2.5% on 1-2-token steps.
- It was a second branch in every public call of the runtime (prefill, decode, verify, commit,
  draft, mark, rewind, the destructor), and "has this process a head, a draft" was a question
  asked at run time.
- It was the least tested path. Every step since 112 lists two machines under "not done"; step
  137 checked it as two processes on one machine only. Sampling with `--spec` was refused on
  it, `--record` refused a back, the server, the kept context and the saved point never ran on it.
- It could not run today: the second machine has an older build and the PR's llama.cpp.

**Removed in this step** (30 lines added, 376 removed; the last commit with all of it: `efa4e03`):

| where | what |
|---|---|
| `hip/peer.h` | the file: the messages, the TCP link, the clock samples |
| `hip/rt_peer.cpp` | the front's sends and waits, `connect_peer`, the back's `serve` loop. Left in it for one more step: `stream_from_host` and `stream_to_host`, which `--teacher` uses |
| `hip/rt.cpp` | the `if (peer)` branches in `verify`, `decode`, `prefill_chunk`, `commit`, `mark_state`, `draft` and the destructor |
| `hip/rt.h` | `connect_peer`, `serve`, `peer`, `peer_block` and three helpers; `has_draft()` and `draft_block()` now speak of this process's draft only |
| `hip/rt_trace.cpp` | the track "messages from the front"; the keys `clockEpochUs`, `layers` and `peerSync` after the events (only the merge script read them) |
| `apps/ds_completion.cpp` | `--remote`, `--serve`, the back's branch of `run()`, and the rules that named them (`--temp` with `--spec`, `--requests`, `--record`, `--spec` without `--dspark`). A part of the model now runs only with `--teacher`, and says so |
| `scripts/` | `split-measure.sh` and `trace_merge.py` are filed with their results (`results/split/two-machines/`, `results/trace/two-machines/`), each with a line saying it needs a build of `efa4e03` or earlier |

`--layers a-b` with `--teacher` and the stream files still works (checked below): it is the next
step's.

**Checked** (`results/dist/two-machines-link.txt`; against the programs of `efa4e03` built in a
second checkout):

- `--remote` and `--serve` are unknown arguments; a range without `--teacher` is refused.
- Per layer on the runtime (hello-fp32): 200 rows, none differ; the end-to-end logits are the
  same bytes.
- `bin/ds-ref` (the same source file): step 173's output hash.
- The whole model against layers 0-19 then 20-39 through a stream file: identical ids at 325
  positions.
- The held-out prompts twice: 16 of 16 outputs are the stored bytes. The second runs' mean is
  44.80 t/s (step 176: 44.8).
- A trace is valid JSON with the one key `traceEvents`; `scripts/trace_critpath.py` reads it.
- The request test with the model and the server test pass.

**Two things seen on the way, neither from this change:**

- *The first runs after a boot are slower.* The first repeat's mean is 41.14 t/s: six of the
  eight prompts read 17-80 MB from storage while generating and ran 7-14% slower than their
  second run; the two that read under 4 MB ran at their usual speed. Step 117 saw this once
  (the engram table out of the page cache). A measurement taken right after a boot needs a
  first pass that is thrown away, or the record's `read_bytes.gen` checked.
- *The starred level of `pp_dpm_sclk` is not the setting in this boot.* After the usual two
  writes (`manual`, then 6) every card reads `1: 1000Mhz *` when idle and `6: 1608Mhz *` while
  it works (sampled four times a second during a run). The speeds are level 6's. In earlier
  boots the star read 6 at idle too; the kernel, its command line and the driver's boot
  messages are the same as the boot before, and the driver lists `GFXCLK_DS` as off. The one
  difference found: this time a GPU program had run before the clock was set. Not tested. The
  consequence is for the records: `--record`'s `gpu_sclk` and `run_matrix.py`'s header read the
  star as a run starts, so these runs are labelled `[1, 1, 1, 1]` (step 116 corrected the same
  kind of reading for auto). Not changed here.

**Documents.** The current ones no longer offer two machines: README (the status), QUICKSTART
(the section, the options, the results row, the limits), `docs/overview.md` (the goal, the code
map), `docs/timing.md` and `docs/plan.md` (their two-machine sections kept as a record, marked
removed, with why), `docs/reproduce.md` (rows 111-112 name the commit and the scripts' new
place), `scripts/README.md`, and single lines in `docs/numerics.md`, `docs/record.md` and
`docs/requests.md`. Steps 110-114 and the results stay as written.

**Not done:** the longer golden sets; `-i` sessions and sampling with the new build. The second
machine is untouched.

## Step 179: the layer range out of the runtime and the program

**The second half of step 178's decision.** With the link gone, the only user of a runtime that
starts above layer 0, or hands its stream to another, was `--teacher`: the test of two halves
against the whole model that step 110 wrote for the two machines. It goes with them.

**Removed** (44 lines added, 166 removed; the last commit with it: `519ffd4`):

| where | what |
|---|---|
| `apps/ds_completion.cpp` | `--layers`, `--teacher`, `--stream-in`, `--stream-out`, the teacher's branch of `run()`; `Model::load` is called without a list of layers. `ds-ref` loses `--layers` too (the same parser; it never ran a part of the model) |
| `hip/rt.h` | `Stream`, `stream_in`, `stream_in_tok`, `stream_out`, `layer0()`, `layer1()`; the paragraph on ranges |
| `hip/rt.cpp` | the first layer `L0` and its branches (the stream in place of the embedding, in `forward_chunk` and `decode`), the copy of the stream to the host after the last layer, the check that a range holds the layers its own read from |
| `hip/rt_peer.cpp` | the file (the two stream copies were all it still held) |

**Kept, and why.** `ds-test --runtime --layers 3,20` loads layers 0-20 and checks them one by
one, so a runtime must still stand without the last layers. That is now the whole of it: the
runtime runs layers 0 .. `n_run` - 1, which is every layer except under `ds-test --layers`; a
partial one has no head (`has_head()`), uploads no weights of the missing layers, fetches engram
rows only if an engram layer is loaded (`has_engram`, was `range_engram`), and its `decode()`
returns no logits. The constructor refuses loaded layers that are not layers 0 .. k.
`--layers` of `ds-test`, `fp8pack` and `ds-inspect` is theirs and unchanged.

**Checked** (`results/dist/layer-range.txt`; sclk 6; against the programs of `519ffd4`):

- The four options are unknown arguments, in `ds-ref` too.
- Per layer on the runtime: all layers (200 rows), `--layers 0-5` (30 rows, the runtime without
  a head) and `--layers 3,20` (10 rows): no row differs. End-to-end logits: the same bytes.
- `bin/ds-ref`: step 173's output hash.
- A 2,731-token prompt through the prompt chunks (their two-halves loop was edited): the same
  output from both programs; prompt speed 84.7 / 86.1 t/s before, 85.9 / 85.1 after.
- The held-out prompts twice: 16 of 16 outputs are the stored bytes; the second runs' mean is
  44.91 t/s (step 178: 44.80).
- A trace, the request test with the model, the server test: as before.

**The two observations of step 178, a second time.**

- *The first runs after a boot:* again. The first repeat's mean is 40.17 t/s, all eight prompts
  reading 17-77 MB from storage while generating; the second repeat is at the usual speed. And
  the long prompt's first run on the old program read 68.8 t/s against ~85 on every later run of
  either program: taken alone it would have read as a 25% gain from this change. After a boot,
  the first pass of anything is not a measurement.
- *The starred clock level:* the machine was rebooted between the two steps, and this time the
  clock was set before any GPU program ran. The star reads 6 at idle and under load, and the
  records say `[6, 6, 6, 6]`. So: clock set first, star right (this boot and the two before step
  178's); a GPU program first, star at 1 when idle though level 6 was in effect (step 178's
  boot). One case of the second kind; it fits the order being the cause and does not prove it.
  The rule that follows either way: fans and clock first after a boot, before anything else.

**Documents.** QUICKSTART (the option's entry, `--record`'s limits), the overview's code map
(`hip/rt_peer.cpp` out), `docs/requests.md` and `docs/record.md` (a word each), `docs/plan.md`
(item 31 and its section: both steps done), `docs/reproduce.md` (row 110: its teacher commands
need a build of `519ffd4` or earlier).

**What is left of two machines:** steps 110-114 here, their results and the two filed scripts,
the sections marked removed in `docs/plan.md` and `docs/timing.md`, and the second machine
itself, as a place for A/B runs of one-machine builds. In the code: nothing.

**Not done:** the longer golden sets; `-i` sessions and sampling with the new build; a last read
of the documents for what the two steps left (the third step of the plan).

## Step 180: a last read after the two-machine code left

The third step of the plan of step 178: the current documents and the code searched for what
steps 178 and 179 missed (`results/dist/two-machines-last-read.txt` has the search). Little was
left, as the documents had moved with the two code commits.

- **Code:** `Runtime::verify()` had a parameter `head` (false: no logits). Only the back half of
  step 111 could want a verify step without logits; no caller passed it. Removed; the comment
  that described it is now on `forward_chunk()`, where prompt chunks do use it.
- **Documents:** two passages still spoke of two machines in the present tense: the overview's
  paragraph under the step's time table, and the two-machine section of `docs/timing.md` (its
  heading already said "measured, then removed"). Both are in the past tense now.
- **Left as they are:** the notes "measured, then removed" in the README, the overview, the
  plan, `timing.md`, `numerics.md`, `record.md` and `reproduce.md`; the plan's history lines;
  `--layers` of `ds-test`, `ds-inspect`, `fp8pack` and the benchmarks, which are those programs'
  own options; "two halves" of a prompt chunk, which is step 45's overlap on one machine.

**Checked** (sclk 6, the same boot as step 179): both builds without warnings; the held-out
prompts once, 8 of 8 outputs the stored bytes, mean 44.88 t/s (step 179: 44.91), nothing read
from storage. Not run: `ds-test`, the request and server tests (one parameter of one function,
and all its calls are in those runs).

**Not done**, carried from steps 178-179: the longer golden sets, `-i` sessions and sampling
with the build after the removal. Open: why the starred clock level read 1 in step 178's boot.

## Step 181: sessions, sampling and the longer golden sets on the build after the removal

Steps 178-180 each listed these as not run. No code changed here: the programs are those of
step 180's commit (`results/dist/after-removal-gates.txt`; the script, its helpers and its log
in `results/dist/after-removal/`; sclk 6; 3 h 04 min).

| | what ran | result |
|---|---|---|
| sessions | `scripts/chat.sh` with the questions of steps 108-109: chat at `-n 200` and `-n 12`, thinking at `-n 600`, 20 and 45, and the long one (`--effort max -n 2500`, replies of 1,344, 2,500 and 764 tokens) | 6 of 6 print the stored bytes |
| a session against fresh runs | `ds-completion -f` on the whole conversation before reply 2 and reply 3 | 10 of 10 print the session's reply |
| sampling, the plain loop | step 149's matrix: 6 configurations on 3 prompts | 18 of 18 output hashes are step 149's |
| sampling with `--spec` | step 151's two matrices: 5 configurations on 3 prompts, greedy and sampled on the 8 held-out prompts | 15 of 15 and 16 of 16 are step 151's |
| the CPU reference path | `bin/ds-test` on `hello-fp32-nq`, `readme200`, `readme200-fp32-nq`, `readme1100` | 960 rows, 0 differ from the stored tables |
| the runtime at 20,000 tokens | `bin-hip/ds-test --runtime` on `readme20k-s`, per layer with the top-k dump, and end to end | 360 rows, 0 differ; the dump and the logits are the stored bytes; against the reference 9 of 9 argmax, mean KL 2.15e-02 (step 174's line) |

Speeds are those of the stored runs to ~2% in every configuration (held-out greedy with
`--spec` 45.11 t/s against 44.94 in step 151's records; sampled 42.50 against 43.06).

So the build after the removal prints, on everything the project has stored outputs for, what
the build before it printed.

**Two things seen on the way.**

- *One false alarm, mine:* the helper that compared a fresh run with a session's reply split
  the stored file by line, and the chat replies have newlines. It said DIFFERS on bytes that
  were equal; corrected during the run. The comparisons of steps 178, 179 and this one were each
  written by hand for the step.
- *Most of the three hours is loading the model.* A held-out run is ~140 s of load and ~8 s of
  generation; 52 loads are about two of the three hours. And the load itself reads 138.6 s in
  these records against 115.7 s in step 151's of the same matrix (20% longer; when it changed
  and why was not looked for).

**Next** (the person, 2026-11-22: this is what to focus on): the testing process itself. The
same guarantees in less time: PLAN "The testing process".

## Step 182: the documents, comments and names against the tree

After the clean-ups of steps 168-181 the person asked for a check of everything written: the
documents, the comments, the names (`results/dist/review-182.txt` has every finding).

**How.** Two checks by program: every path the current documents name exists, and every
option they name is parsed by some program (both clean; all 38 options of `ds-completion` are
in QUICKSTART). Then four independent read-only passes, each over one part (the runtime; the
programs, `src/`, the tests and the Makefile; the server, requests and records with their
scripts; the front documents), and every finding checked against the code before a change.

**Found.** Nothing of the two-machine code, anywhere. What was stale is older, and of a few
kinds:

- *Numbers that drifted apart:* the model's size (474 "GB" in three documents is GiB; 508 GB
  in a fourth); the draft on GPU 0 (8.5 GB is its file, 8.0 its memory); the experts placed at
  startup (the overview said 6,626 = 43%; a record says 6,097 = 40%); `--record`'s arguments
  ("fourteen": 21); the load time.
- *Text a later step overtook:* `--requests` "each on an empty context" in three comments
  (steps 158 and 160 changed that); the server's first sentence; the overview's "What is next";
  a plan sentence saying step 173 was still to come; dates over sections edited since.
- *Comments that moved away from their code:* one stood over the wrong function since step
  136, two on the wrong member, one named a study removed in step 135, one described a sort
  the kernel no longer uses, one line carried four comments of which three were other options'.
- *One command the program refuses* (the overview's, without `--rt`), and two scripts in
  `scripts/` that no document listed.
- *One line of code:* the Makefile's `FORK` alias assigned `LLAMA_CPP` to itself since step
  176, while two documents said `FORK=<dir>` still worked. Fixed.

About 40 corrections in 13 documents and 30 in comments and usage texts of 17 source files.

**Checked.** The programs are the same code: the disassembly of `bin-hip/ds-completion` and
`bin-hip/ds-test` equals the previous build's, and so do the GPU code and the constants. So
no GPU run. The server test passes.

**Found and left for steps of their own** (they change code or behaviour; section 3 of the
file): a `--requests` header whose bad field comes after `bytes=` leaves the prompt unread and
the engine out of step with its input; `ds-ref` accepts and ignores every runtime option;
unused members, parameters and includes; Makefile prerequisites (the product is rebuilt when
the reference path's headers change); usage texts shorter than their parsers; "GB" that is
sometimes GiB.

**Not read:** `docs/steps.md` and `docs/process.md` (history), the bodies of the CPU kernels
and the benchmarks, numbers that stand in one document only.

## Step 183: a request header that was not understood closes the engine, wherever the mistake stands

Found in step 182's reading, the first of its code findings to be fixed. In `--requests` mode a
header whose bad field came **after** `bytes=` (`gen n=3 bytes=5 foo=1`) was answered with an
error line and the engine went on, but without reading the 5 bytes of the prompt: it then read
the prompt as the next header. With the bad field before `bytes=`, the engine closed, as the
document said. And `bytes=five` set the length to 0 the same way. The server always sends a
well-formed header with `bytes=` last, so it never met this; another front could.

**The fix** (`apps/ds_completion.cpp`): any header that was not understood (an unknown first
word, a field that is unknown or not a number, no `bytes=`) is answered with `error ...;
closing`, and the engine exits without reading further. A value out of range (`n=0`,
`top_p=7`) in a well-formed header is as before: the prompt is read, an error line, the next
request follows. This is what `docs/requests.md` already said; the two places that decided
are now one.

**Tested** (`results/requests/requests_test.py`, a new part; `--headers-only` runs it alone in
~80 s): eight headers, each on an engine of its own without experts on GPUs (it loads in ~9 s),
each followed by a good request. Six must give one error line ending in `; closing` and
nothing after it; two must give an error line and then the next request's tokens.

| | the program before | after |
|---|---|---|
| an unknown field after `bytes=` | went on, the prompt unread | closes |
| a value that is not a number, after `bytes=` | went on, the prompt unread | closes |
| `bytes=five` | went on with a length of 0 | closes |
| an unknown field or a bad number before `bytes=`; no `bytes=` | closes | closes |
| `n=0`, `top_p=7` | error, then the next request | the same |

(`results/requests/headers-before.txt`: 3 of 8 fail on step 182's program; all 8 pass now.)

**Checked** (sclk 6): the whole request test with the model, 67 checks (59 before, and the 8),
all passed: every reply's hash is its single run's. `server/test_server.py` passes. Not run
again: the test with `--no-spec` (`requests_test-plain.txt` is step 173's), the held-out
matrix (the change is in the reading of a request's header only).

Still open from step 182's list: `ds-ref` taking the runtime's options, the unused code, the
Makefile's prerequisites, the usage texts, the units. Next, as decided in step 181: the
testing process.

## Step 184: the testing as it is, written down

Before the testing process changes (PLAN item 35), the person asked for a description of its
current state: `docs/testing.md`. No code changed.

What the page has: what "correct" means here as a chain of seven links, each with the test
that checks it; the terms (the reference against our CPU reference path, a golden set, a
variant, teacher-forced, a table, stored bytes, `--canon`, held-out, a record, a gate); every
test with its target, what it is compared with, what it needs, its time and whether it says
pass or fail itself; what each needed thing is and whether it is in git; which sets of tests
the last ten steps ran; and what is thin, double or by hand, as facts.

**Measured for it** (`results/dist/testing-times.txt`): the short tests, once each. The unit
tests and the server test: 0 to 8 s, except `bin/test-sampler`, 80 s (its timing loops; the
documents said "a few seconds"). `ds-test` on a 12-token golden set: 3 to 9 s on either path,
per layer or end to end, since it maps the model and uploads only the dense weights.

**What writing it showed** (none of it new code, all of it of use for item 35):

- Of 27 tests, 9 say pass or fail themselves; 18 print something a person compares.
- `ds-test` has no verdict at all: its gate is "the table equals the stored one".
- The golden sets (73 GB), the stored logits and the rounding test's data are not in git: a
  clean checkout cannot run the tests that compare with the reference.
- The whole comparison with the reference on the small set, both paths, per layer and end to
  end, is under a minute. The four hours are the product's outputs against stored outputs: 52
  loads of ~140 s for a few minutes of generation.
- The draft has not been compared with the reference since step 19, as far as the steps say.

Next: item 35, starting from this page.

## Step 185: verdicts and localizers

The person's observation on step 184's page: some tests give a binary outcome, and some give
debugging information (which layer exactly was wrong). `docs/testing.md` has it now as a term
and as section 3.6: for every test, whether it gives a verdict, and what it says about where;
then the tests as a ladder from the cheapest verdict down to the cause (an output's bytes ->
the logits on a golden set -> the per-layer table -> per token). No code changed, nothing run.

What the split shows about the practice:

- Localizers are run as gates: the per-layer tables at every step, though the logits' bytes
  are the sharper verdict on the same data. At 20,000 tokens, 32 minutes of localizer for what
  a 9-minute verdict answers.
- Some verdicts have no localizer behind them: a sampled run's hash that differs says nothing
  more, since a record keeps the hash and not the ids.
- Some localizers have no verdict in front of them: the draft against the reference, a trace.
- The ladder was in people's heads, not in a tool.

For item 35 this gives a rule the first draft of the plan lacked: a tier runs verdicts; a
localizer runs when its verdict fails.

## Step 186: the runs decided, and the first of them: `scripts/check.py unit`

**The decision** (the person, 2026-11-23, on `docs/testing.md`): aim for simplicity and speed
of iteration, and avoid deciding per change which tests to pick, because mistakes happen there.
So a few fixed runs:

1. **unit**: no model files, fast; after every change.
2. **default**: with the model, avoiding reloads; after most changes. 5-10 minutes is fine.
3. **speed**: the held-out prompts' tokens per second, its own run: small performance gains
   were the common case and will be again.
4. **complete**: whatever it takes, overnight.
5. **localizers**: only when a verdict failed.

PLAN's section "The testing process" has the contents of each and the order of work; the tiers
"by what changed" of step 181's draft are gone.

**Built here: `unit`.** `scripts/check.py unit` builds both ways (a stale program is a wrong
verdict), runs five checks and prints a line for each, then one verdict; exit code 1 on any
failure; every check's output kept under `data/check/`. A failed check prints where its
output is and what to look at next. A check whose needed file is missing fails, it is not
skipped.

| check | time |
|---|---|
| the rounding ops against the reference's (`bin/test-fakequant`) | 0.0 s |
| the sampler (`bin/test-sampler`) | 19.0 s |
| the selection kernels against torch (`ref/check_topk.py`) | 2.1 s |
| the batched kernels against one token per launch (`bin-hip/bench-kernels --bitwise`) | 0.8 s |
| the server against a fake engine (`server/test_server.py`) | 8.1 s |
| all, with nothing to build | 30 s |

Two changes to make them fit:

- **`test-sampler`: 80 s -> 19 s.** One check compared 4,800 draws with "the whole vocabulary
  sorted", and sorted 129,280 values for each draw. The order depends on the logits and the
  temperature only: sorted 48 times now. The same draws, the same lines of output. The 17 s
  left are the sampler's own 4,800 draws.
- **`test-fakequant`'s data in git** (`tests/fq/`, 725 KB; it was under `data/`, which git
  ignores): the test now runs from a clean checkout.

**A negative control** (`results/dist/check-unit.txt`): one value of the reference's expected
output changed; the run says FAIL for that check, passes the other four, exit code 1.

**What `unit` still needs from outside the repository:** the GPUs with ROCm (two checks run
kernels), the Python environment with torch (the reference's top-k), and the model's
`encoding.py` for the server test. No weights.

**Seen while doing it:** `test-fakequant` checks the rounding ops of the reference path's
graphs only. The runtime's host functions (`src/fakequant.cpp`: the fp8 codec and repack) have
a self-test at the draft's load and no unit test.

Next: `default`. First the expected logits and token ids of the golden sets into git, and the
end-to-end runs of the runtime on the longer sets timed.

## Step 187: `check.py default`, its first half: logits against stored hashes

The run for after every change (step 186's plan), in two parts. This step has the first: the
arithmetic. The second, the product's outputs in one load, is next.

`scripts/check.py default` runs `unit` and then `ds-test --e2e` five times: the runtime on
golden token sequences of 12, 202 and 1,102 tokens, the CPU reference path on the first two.
Each run writes the logits of every step; the file's sha256 must be the one in
`tests/e2e/expected.txt`. `ds-test` maps the model and uploads only the dense weights, so
these start in seconds.

| check | time |
|---|---|
| unit (five checks) | 30 s |
| the runtime: `hello-fp32`, `readme200`, `readme1100` | 4.5, 11.7, 33.7 s |
| the CPU reference path: `hello-fp32`, `readme200` | 3.3, 21.6 s |
| all, with nothing to build | 105 s |

**What had to exist for it:**

- *The golden sets in git, cut down.* An end-to-end run reads of a golden set only each step's
  token ids and the reference's logits. `ref/trim_golden.py` writes those in the same format:
  `tests/e2e/<name>/`, 2.6 to 4.7 MB each, 9.5 MB together, against 137 MB to 3.8 GB. `ds-test
  --e2e` writes the same bytes on a cut set as on the whole one. The whole sets stay under
  `data/` for the per-layer table.
- *Expected hashes* (`tests/e2e/expected.txt`), written by `check.py default --store`, which
  prints old and new and is meant for a change that should move the logits, after a look at
  the difference against the reference. What was stored here was first compared with the
  reference (argmax 5/5, 9/9, 5/5; KL as at earlier steps) and made three times in separate
  processes with the same hash.

**A negative control** (`results/dist/check-default-logits.txt`): the eps of the query's norm
in `hip/rt.cpp` plus 1e-3. The three runtime checks fail, the two of the CPU path pass. And
following the failed check's hint: against the reference that build still agrees on every
argmax (mean KL 5e-4 where it was 7e-5). **A comparison of output bytes would have passed
it;** the logits' bytes did not. This is the reason for this half.

**Fixed on the way:** `ds-test` without `--e2e` on a cut set ended in a segmentation fault;
it now says the set serves `--e2e` only.

**Limits, as planned:** no prompt past 1,102 tokens here (the 20,000-token set is `complete`'s);
the CPU path on `readme1100` (114 s) is left to `complete` too. The stored hashes are of this
toolchain and of 16 threads.

Next: the second half, one engine with `--requests` and the full configuration.

## Step 188: `check.py default`, its second half: the product's outputs in one load

The logits (step 187) cover the arithmetic. This half covers what stands around it in the
product, and with it `default` is whole: **13 checks, about 5.5 minutes, one command, one
verdict.**

`tests/outputs_check.py` starts one engine with the fastest settings (placement, adaptive
cache, the draft and speculation, `--canon`) in `--requests` mode and sends it in turn: the
eight held-out prompts greedy; two of them sampled (temperature 1, top-p 0.95, seed 1); one
with `n=5`; a chat turn and a second turn that goes on from the kept context, and the same
request with `keep=0`. Then one fresh process: an interactive session with thinking, three
turns. Every reply's token ids must be those in `tests/outputs/expected.json`.

- **Ids are stored, not hashes** (17 KB for 12 replies), so a failed check says which token
  differs first: step 185 found the hash to be a verdict with nothing behind it.
- **Tied to the reference outputs:** the eight greedy replies must have the hashes of
  `results/record/reference/heldout-output.tsv` (plain greedy decoding on a fixed placement),
  and `--store` refuses a greedy reply that has not. The two sampled ones are step 150's.

| part | time |
|---|---|
| unit | 30 s |
| logits, five runs | 76 s |
| the engine: its start 119 s, then 12 requests | 204 s |
| the session in a fresh process | 21 s |
| `scripts/check.py default`, nothing to build | ~5 min 30 s |

**The control of step 187 again** (one eps in the runtime): the three runtime logits checks
fail, and in the outputs check 7 of the 8 greedy replies, both sampled ones and the session.
One greedy reply of 300 tokens is unchanged by it. Outputs see such a change often, logits
always.

**Found while timing the session: where the "load" goes** (`results/dist/check-default-outputs.txt`).
The same session takes 15 s with no experts on GPUs, 18 s with the placement, 21 s with the
draft and speculation, and 126 s with `--cache-loads`. The model loads in 13 s. The other 105 s
are the registration of the 289 GB of expert tensors with HIP for the cache's direct loads:
`--cache-wait` waits for it before the prompt, and a run without `--cache-wait` waits for it
at its exit, even if it started no load (that session: "0 loads started"). So:

- the "~140 s load" of steps 181-187 is ~13 s of load and the registration (139 s in step
  181's records against 116 s in step 151's: the registration is what grew; why is open);
- every `scripts/chat.sh` session, however short, takes two minutes to end;
- for `speed` (eight fresh processes, twice) the registration is ~28 of ~41 minutes.

The session in `default` therefore runs without the cache (the same bytes under `--canon`);
the cache is exercised by the engine's 12 requests.

**What `default` does not cover** (all of it in `complete`'s list): prompts past 1,102 tokens;
`-i` together with the cache; `-p` and `-f` as the way in (the engine part enters through
`--requests`; the generation loops are the same code); cancels, the saved point, the server
with the real engine; fresh-process speed.

Next: `speed` as its own run, and with it the registration: whether a process must wait for
it at exit, and what it costs at the start.

## Step 189: the registration of the expert tensors, looked at

Step 188 found that the "load" of a run is ~13 s of load and ~105 s of registering the 289 GB
of expert tensors with HIP, which the adaptive cache's loads need. Before building the `speed`
run, whose time is mostly that: what does the registration cost, what changes it, and who
has to wait for it? (`results/register/registration.txt`.)

**Measured.**

- *It does not vary within a boot, and it does between boots.* 130 stored runs of 16-17
  November: 115.4 to 116.3 s to the first token. Step 181's boot: 138.4 to 138.8. Today: 119.
  So "the load grew from 116 to 139 s" (steps 181-188) was a difference between two boots,
  not a change in the code. Why boots differ is open.
- *Threads do not help.* 6 GiB registered by 1, 4 and 16 threads: 2.17, 2.14, 2.10 GB/s. The
  calls are serialized below us.
- *The flags do not matter* (read-only, portable, both, none: 2.87-2.90 GB/s).
- *It is all in the kernel:* no user time; 57% of the wall time the calling thread is in the
  kernel, the rest it waits.
- *Two things help a little:* mapping the pages first (2.2 -> 2.9 GB/s) and larger pieces (2.8;
  but step 43 chose 64 MB pieces so that one call does not hold the memory-map lock for long).
  Together perhaps 100 s where it is 105-130.

So from our side the rate is fixed. What can change is how much is registered, and when.

**Changed here, the one thing that needs no decision:** a process that ends before the
registration is over now stops it, where it used to wait for it. A `scripts/chat.sh` session
of three short turns: **126 s -> 21 s**, the same bytes. (It registered 289 GB, loaded nothing
from them, and unregistered them.) `scripts/check.py default` passes.

**What remains, and needs a decision** (not done):

1. *Loads before the registration is complete.* A 300-token run starts ~1,850 loads, at most
   35 GB of the 289: 12% of what is registered before the first load may start. The cache
   could use each piece as soon as it is registered, with the pieces in the order of the
   placement's ranking (the experts just below the cut are the likeliest to be loaded). Then a
   server or a session has a working cache seconds after its start, and `--cache-wait` keeps
   its meaning for measurements.
2. *Registering only what is loaded,* expert by expert at its first load (6.5 ms each): no
   start-up cost at all, but the first loads are slower, which changes the short runs' speed.
3. *The `speed` run.* With `--cache-wait` each of its 16 processes waits ~105-125 s: 28-33 of
   its ~41 minutes. Either accept it (the numbers stay comparable with every earlier step), or
   measure in one process with the cache put back to its start between prompts (new numbers,
   a new baseline).

## Step 190: `check.py speed` and `check.py complete`: the four runs exist

The person, on step 189's open questions: leave the registration and the way speed is
measured as they are; build the reorganized testing first; once unit, default, speed and
complete are separate, improve each. So both runs are built from what is measured today.

**`scripts/check.py speed`** (`tests/speed_check.py`; 36 min): the measurement the documents
quote, unchanged: the eight held-out prompts, each in a fresh process with the fastest
settings, twice (`results/cleanup/heldout.matrix`). What is new is the verdict:

- the second repeat's mean against `tests/speed/baseline.tsv`: more than 2% slower fails;
  within 2% is "the same within 2%, no difference claimed" (the mean has moved ~0.5% between
  runs of the same code: 44.80, 44.91, 44.88, 44.88); more than 2% faster says so, and
  `--store` makes it the baseline. Per prompt the difference is printed, marked past 5%;
- every output must be the stored bytes: a faster wrong answer is not a result;
- not a measurement, so a failure whatever the numbers: a clock level other than the
  baseline's, or a second repeat that read from storage while generating (over 8 MB).

The baseline is this step's run: mean **44.88 t/s**, 16 of 16 outputs the stored bytes.

**`scripts/check.py complete`** (~4 h): `default`, then what `default` leaves out, then `speed`:

| check | by |
|---|---|
| the runtime's logits at 20,000 tokens; the CPU path's at 1,102 | `ds-test --e2e` on cut sets, hashes |
| the request protocol on one engine, with and without speculation | `results/requests/requests_test.py` |
| six sessions against their stored replies; ten single runs on the conversations | `tests/sessions_check.py` |
| sampling: the plain loop; with speculation; on the held-out prompts | `tests/matrix_check.py` on the three matrices, hashes against the stored records |
| `ds-ref`'s output; a 2,731-token prompt from a file | `tests/record_check.py`, one hash each |
| speed | as above |

**Not run as a whole.** `complete` was started and stopped by the person in its 12th check,
to be run overnight; until then 11 checks had passed (`default`'s). Its other parts were tried
one by one (`results/dist/check-speed-complete.txt`): the session comparisons on step 181's
kept outputs (16 of 16, and three wrong pairings refused), the matrix check on one prompt,
the two single-run checks with a control. Two things its first whole run will meet: the CPU
path's logits at 1,102 tokens have no expected hash yet (that check fails until `--store`),
and `tests/sessions_check.py` has not run as a script.

**Not in any run,** by decision: what gives no verdict (the per-layer tables and the other
localizers; the draft against the reference), `ref/check_weights.py` (it reads the checkpoint
over the network), pi on the real server.

**Progress** (the person, during the speed run: its output showed nothing for 36 minutes; one
had to look into the run's directory). `check.py` now prints a line when a check starts, and
for the checks that take minutes it passes their own lines through, indented.

**A mistake of mine in the first speed run:** the limit for "read from storage in the second
repeat" was 1 MB, and one prompt reads 1.1 MB in every second repeat. The run measured for 36
minutes and then said FAIL. The records were kept, so the verdict was computed again from
them with the limit at 8 MB; nothing was measured twice. A check that can refuse a
measurement should be tried on stored measurements before it is trusted with a new one.

**Where the testing stands** (PLAN item 35): four runs, each one command and one verdict, and
nothing to choose per change: `unit` 30 s, `default` 5.5 min, `speed` 36 min, `complete` ~4 h.
What to improve in each is listed in the plan.

## Step 191: the last expected hash for `complete`, and what its logits say against the reference

Step 190 left one check of `complete` without an expectation: the CPU reference path's logits
on the 1,102-token set. Made twice in separate processes (114 s each), the same sha256; it is
in `tests/e2e/expected.txt` now, so the overnight run needs no `--store`.

Before storing it, the file against the reference (`ref/compare_logits.py`):

| | argmax | mean KL | max KL |
|---|---|---|---|
| the CPU reference path, `readme1100` | **4 of 5** | 1.27e-02 | 2.48e-02 |
| the runtime, `readme1100` (step 187) | 5 of 5 | 1.30e-02 | 2.42e-02 |

The one that differs is step 0, the token after the 1,100-token prompt: the reference's two
best logits there are 23.815 (token 769) and 23.734 (token 313), 0.08 apart; our CPU path has
them at 23.475 and 23.717 and so takes 313. The KL of that step (2.5e-02) is of the size of
the runtime's on the same set. `readme1100` is a golden set of the real reference (bf16 with
rounding), against which a correct implementation is 2-10% off per layer by construction
(the reason the fp32 variants exist), and the per-layer table of this path on this set has not
changed since it was stored. So this reads as a near-tie decided by that noise, on a path that
has computed the same since step 174 at least. It is not proven: there is no fp32 variant of
this set to compare with, which is what would decide it.

The hash stored is what the path computes today. It says "unchanged", not "right": that is
all a stored hash can say, and the comparison above is the kind of look `--store` asks for.

**Open:** an fp32 golden set at ~1,100 tokens (the reference, ~10 minutes), to tell the
near-tie from a difference; until then the 4 of 5 stands as an observation.

## Step 192: removing the CPU reference path, 1: the rounding test tests what the runtime calls

The person asked whether the CPU reference path (`ds-ref`, `src/forward.cpp`: the model as ggml
graphs, the first implementation) is needed at all. It is not, for checking the product: the
runtime is compared with the reference's golden sets directly, nothing in `ds-completion`
links it since step 173, and on the 1,102-token set it is the less faithful of the two (step
191). Decision: remove it. Three steps, `scripts/check.py default` after each: this one; then
its checks and `ds-ref`; then its sources.

**First, what would be left testing nothing.** `test-fakequant` compared only the reference
path's graph ops (`src/fakequant_graph.cpp`) with the reference. The functions the runtime
calls on the host (`fq_fp8_host` in the CPU experts, `fq_fp8_split` in the canonical sums) were
not called by any test (a finding of step 182's review). Now the test calls those, and the
scalar fp4 rules put together as the reference does; it links no ggml.

| | before | after |
|---|---|---|
| fp8 | graph op | `fq_fp8_host`; and `fq_fp8_split` (codes times scale) |
| fp4, fp4e4 | graph ops | `fq_e2m1_round`, `fq_e4m3_round`, `fq_pow2_ceil` composed in the test |
| result | 3 x 45,312 values, 0 differ | 4 x 45,440 values, 0 differ |

**A control found a hole in the data.** Two wrong versions of `src/fakequant.cpp`, built
beside the tree:

| the change | before the new rows | after |
|---|---|---|
| ties rounded away from zero (`roundf` for `nearbyintf`) | fp8: **0 differ**; fp4 96 | fp8 64, fp4 96, fp4e4 104 |
| the fp8 scale's floor 1e-3 for 1e-4 | fp8 1,312 | the same |

The fp8 comparison could not see a wrong tie rule: `ref/make_fq_test.py` promised "exact ties
of the e4m3 / e2m1 grids", and its tie rows were e2m1 ties, which under the fp8 scale land on
e4m3 values exactly. Two rows of e4m3 ties were added (blocks whose largest value fixes the
scale, then odd multiples of half a step in several binades), made by the reference; the
45,312 earlier values of each file are unchanged, byte for byte. The reference rounds those
ties to even, as we do.

The runtime's fp4 rounding itself is in GPU kernels and stays covered by the logits checks.
`scripts/check.py unit`: all checks passed (31 s).

## Step 193: removing the CPU reference path, 2: `ds-ref` and its checks

**The checks.** `scripts/check.py default` loses the CPU path's logits at 12 and 202 tokens,
`complete` the ones at 1,102 tokens (stored two steps ago) and `ds-ref`'s output hash.
`tests/e2e/expected.txt` keeps the runtime's four hashes. `default` is 11 checks in 301 s
(13 in 326 s).

**The program.** `apps/ds_completion.cpp` built two programs, chosen by `-DDS_RUNTIME` at 15
places. It builds one now: the sections of the other are gone, with what only allowed for
them: the `Engine*` that `generate_plain` took and never used, the tests for "no runtime"
(`if (rt)`), `--no-quant` (an unknown argument now), the `"experts": null` of a record, the
usage text and the messages that pointed to `bin/ds-ref`. 1,004 lines -> 947. The Makefile
has no `ds-ref` target.

One line of output is kept as it was though half of it is constant now: `# loaded in N s;
prompt N tokens; fake-quant on; N threads` (stored logs and scripts read it).

**Checked.** `scripts/check.py default`: all checks passed. The long prompt from a file with
`--record`: the stored hash; a record still has its ten fields. `--no-quant`: "unknown
argument"; without `--rt`: the message, without the pointer to `ds-ref`.

A mistake on the way: resolving the conditional sections by script left an `else` whose `if`
I had then removed by hand; the build caught it, `check.py` said "the build: no check was
run", which is what that line is for.

What was lost with the program: a way to run the model without GPUs (3.5 t/s). The last commit
that builds it is `bb42f51`; README and QUICKSTART say so where they quoted its speed.

Left for step 194: `src/forward.*`, `src/fakequant_graph.*`, `bin/ds-test` without `--runtime`
and its `--no-quant` golden sets, `bench-cpu-moe`'s use of the graph ops, `src/options.h`.

