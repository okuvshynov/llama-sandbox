# ds-completion — DeepSeek-V4.1-Flash inference for the Mac Pro (755 GB RAM, 4x Vega II)

Goal: `./ds-completion -m <model> -p "prompt"`. One model, one machine, one
sequence. Priorities: correctness, clarity, then performance. Non-goals: other
models, other quants, other hardware, serving features.

Status, 2026-10-27:

- CPU only: 3.5 t/s decode, matching the reference (fork: 3.5 t/s).
- Dense part on one GPU through ggml, routed experts on the CPU: 6.0 t/s.
- Own decode runtime on one GPU (exact fp8, deterministic), routed experts on the CPU with
  our own exact kernel: **9.7-9.9 t/s**. The CPU experts are 2/3 of every step.

Performance work from here on is planned in [PLAN.md](PLAN.md). The short version of how to run
it and what it does now: [QUICKSTART.md](QUICKSTART.md).
Operand types and accumulation: [NUMERICS.md](NUMERICS.md).

## Tools

```bash
make                                      # links the fork's CPU ggml (static)
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/ds-completion -m $M -p "Hello, how are you?" --chat -n 64    # or --think, or raw text
make HIP=1   # -> bin-hip/: also the own GPU runtime (ROCm 6.3.4; same CPU fork libraries)
bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf            # fp8 repack (PLAN.md B0), 20 s
bin-hip/bench-fp8 ~/llms/ds-1017-fp8/dense-fp8.gguf [--ms 1,4] [--sweep]  # fp8 GEMV (B1)
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
bin-hip/ds-completion -m $M -p "..." --chat --rt $F8 [--gpu 0] [--profile]   # own GPU runtime
bin-hip/ds-test -m $M -g data/golden/hello-fp32 --runtime $F8 [--per-token]  # its per-layer gate
bin-hip/ds-test -m $M -g data/golden/hello --runtime $F8 --e2e data/ds/x.f32  # its end-to-end logits
bin/ds-test -m $M -g data/golden/hello-fp32-nq --no-quant --determinism  # bitwise repeatability (CPU)
bin-hip/bench-kernels                                                    # each runtime kernel alone
bin/bench-cpu-moe -m $M [-t 32] [--tokens 5]                             # the CPU expert kernel vs ggml
bin-hip/bench-fp4-moe -m $M [--layer 20] [--dev 0]                       # one layer's experts in HBM
bin-hip/ds-completion ... --rt $F8 --placement placements/study30.txt   # most-used experts on GPUs
bin-hip/bench-expert-pool -m $M -p placements/study30.txt                # expert pools: load, exactness, time
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
every card). `scripts/sclk-sweep.sh` runs the GPU-side benchmarks and both decode
configurations at the current level, sampling hwmon power, into `results/sclk/sclk<level>.txt`;
`python3 scripts/sclk-summary.py` makes the table.

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

`--log-routing` records every token's 6 experts per layer; `scripts/routing-study.sh` ran the
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
