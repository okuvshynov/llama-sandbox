# ds-completion — DeepSeek-V4.1-Flash inference for the Mac Pro (755 GB RAM, 4x Vega II)

Goal: `./ds-completion -m <model> -p "prompt"`. One model, one machine, one
sequence. Priorities: correctness, clarity, then performance. Non-goals: other
models, other quants, other hardware, serving features.

Status, 2026-11-07 (the milestones in order; details in [OVERVIEW.md](OVERVIEW.md), today's numbers in [TIMING.md](TIMING.md)):

- CPU only: 3.5 t/s decode, matching the reference (the llama.cpp fork: 3.5 t/s).
- Own GPU runtime (exact fp8, deterministic), routed experts on the CPU: ~12 t/s.
- The most-used experts placed on the 4 GPUs (`--placement placements/study30.txt`): 17.7-21.7 t/s.
- Speculative decoding with the DSpark draft (`--dspark ... --spec`): +7% to +33% on held-out
  prompts, output byte-identical to plain greedy decoding.
- Placement-invariant numerics (`--canon`, step 31) and the adaptive expert cache (step 32):
  with `--spec`, 25.6 t/s mean on held-out prompts at sclk 2 (step 32), every output
  byte-identical. Since then, at sclk 6 with the cache tuned (800 loads/s every 8 tokens) and
  the decode work of steps 50-88: **41.2 t/s mean (31.7-50.8)** on the 8 held-out prompts, every
  output byte-identical (step 88; the whole history: `results/heldout-history/`, step 72). The
  fastest settings per job: QUICKSTART.md.
- Prompt processing in chunks of 512 (steps 37-45, 68): ~17 -> 93 t/s on a 1,989-token prompt
  (63 with `--canon`; sclk 6), output byte-identical to token by token.
- Long context: 65,536 tokens by default, up to 1M (step 34); gated against the reference at
  20,000 tokens: same greedy tokens, KL as at short context, indexer differences are near-ties
  (step 35).

How to run it: [QUICKSTART.md](QUICKSTART.md). Where the time goes, on one page:
[TIMING.md](TIMING.md). Plans and cost models: [PLAN.md](PLAN.md).
Operand types and accumulation: [NUMERICS.md](NUMERICS.md). A 20-minute overview:
[OVERVIEW.md](OVERVIEW.md). How the work was approached, decision by decision:
[PROCESS.md](PROCESS.md). How to rebuild the setup and rerun every result:
[REPRODUCE.md](REPRODUCE.md).

## Docs and commits

Every commit that changes code, scripts or results updates, in the same commit: this file (a
step: what was done and measured), PROCESS.md (the question, prediction, result and decision),
REPRODUCE.md (new scripts and results and what produces them), and QUICKSTART.md / OVERVIEW.md /
PLAN.md / TIMING.md when what they say changed. New result files start with a `#` header naming the command
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

## Step 37: the prompt in chunks (PLAN item 20, stage 1 and the first part of stage 2)

The prompt went through the runtime one token at a time (~19-25 t/s): a 30k-token document took
half an hour. Every change below keeps the output byte-identical to token by token
(`scripts/prefill-gate.sh`, `results/prefill/gate.txt`).

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
11,286) and `scripts/prefill_stream_sim.py` model three policies per chunk and layer: all on the
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
`scripts/spec_momentum.py`, `results/spec-momentum.txt`), adding the previous step's accepted
fraction, its surprise, or a running average to a fit on the confidence improves the fit on its own
data by a hair and makes predictions for left-out prompts slightly worse (log-loss 0.3872 without,
0.3879-0.3896 with). Within confidence bins, acceptance after a poor and after a good step differs
little and in both directions. The head reads the main model's last layers at the current
position, so an easy stretch of text already shows in its confidence. Not added. (Short runs
without the cache; long reasoning runs are not checked yet.)

**[TIMING.md](TIMING.md):** where the time goes in generation and in the prompt, on one page, with
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

**Launches:** `scripts/hip_count_shim.c` (an LD_PRELOAD counter) counts ~2,450 kernel launches
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

Could co-selection guide the split over the CPU and the dies (`scripts/coselect.py`,
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
newcomer must beat its victim by a margin) do better? `scripts/cache_sim.py --decay` (step 29's
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
the layer routed). `scripts/route-ahead.sh`, the 8 held-out prompts, `results/route-ahead/`
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
guess must be acted on two or more layers ahead. `scripts/prefetch_sim.py` replays the 8
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
plan, `history.tsv` the numbers, `history.html` the chart, all from `scripts/heldout_history.py`).
Steps 48-71 were measured then at sclk 6 (27.96 -> 34.98 t/s); steps 6-45 were measured at other
clocks or settings, so `scripts/heldout-rerun.sh` rebuilds their commits in worktrees (all 16
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
- `scripts/heldout-status.sh` (added after the start): points complete, the current point and
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
`scripts/draft_route_stats.py` reads them (`results/decode-next/draft-route.txt`, the 8 held-out
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

