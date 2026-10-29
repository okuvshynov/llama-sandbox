# ds-completion — DeepSeek-V4.1-Flash inference for the Mac Pro (755 GB RAM, 4x Vega II)

Goal: `./ds-completion -m <model> -p "prompt"`. One model, one machine, one
sequence. Priorities: correctness, clarity, then performance. Non-goals: other
models, other quants, other hardware, serving features.

Status, 2026-10-29 (sclk 2, 16 CPU threads; details in [OVERVIEW.md](OVERVIEW.md)):

- CPU only: 3.5 t/s decode, matching the reference (the llama.cpp fork: 3.5 t/s).
- Own GPU runtime (exact fp8, deterministic), routed experts on the CPU: ~12 t/s.
- The most-used experts placed on the 4 GPUs (`--placement placements/study30.txt`): 17.7-21.7 t/s.
- Speculative decoding with the DSpark draft (`--dspark ... --spec`): +7% to +33% on held-out
  prompts, output byte-identical to plain greedy decoding.
- In progress: an adaptive expert cache (PLAN.md "Adaptive expert cache"); phases 0-2 done
  (steps 28-30), placement-invariant numerics in the runtime (`--canon`, step 31).
- Long context: 65,536 tokens by default, up to 1M (step 34); gated against the reference at
  20,000 tokens: same greedy tokens, KL as at short context, indexer differences are near-ties
  (step 35).

How to run it: [QUICKSTART.md](QUICKSTART.md). Plans and cost models: [PLAN.md](PLAN.md).
Operand types and accumulation: [NUMERICS.md](NUMERICS.md). A 20-minute overview:
[OVERVIEW.md](OVERVIEW.md). How the work was approached, decision by decision:
[PROCESS.md](PROCESS.md). How to rebuild the setup and rerun every result:
[REPRODUCE.md](REPRODUCE.md).

## Docs and commits

Every commit that changes code, scripts or results updates, in the same commit: this file (a
step: what was done and measured), PROCESS.md (the question, prediction, result and decision),
REPRODUCE.md (new scripts and results and what produces them), and QUICKSTART.md / OVERVIEW.md /
PLAN.md when what they say changed. New result files start with a `#` header naming the command
or script, the commit, the clock level and the date. (The rule for the assistant is in the
sandbox's CLAUDE.md.)

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
  (`results/spec-conf/`, `results/spec-conf-calibration.txt`, `scripts/spec_conf.py`).
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

`scripts/spec-sclk.sh` at the current sclk level (set by hand): the 8 held-out prompts, plain and
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
  committed one). `scripts/spec-routing.sh` logged the 8 held-out prompts with `--spec` and
  study30, up to 1,024 tokens (5,480 generated tokens in 2,018 steps); every output starts with
  the earlier 300/500-token output (logging changes nothing). Plain decoding of the same texts
  is replayed from the committed tokens; the 30 study logs (plain) are a second set, warm
  started leave-one-out.
- **Simulator** (`scripts/cache_sim.py`): 6,201 slots (the runtime's, with the draft loaded),
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

Live (`scripts/load-test.sh`, `results/load-test/summary.txt`): the runtime with
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

Gate (`scripts/canon-gate.sh`, `results/canon-gate.txt`, 128 tokens, 3 prompts): with `--canon`,
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

Results (`scripts/cache-test.sh`, `results/cache-test/`, 8 held-out prompts, 300 tokens,
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
- The full placement-invariance gate (`scripts/canon-gate.sh`) passes again on this code.

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
