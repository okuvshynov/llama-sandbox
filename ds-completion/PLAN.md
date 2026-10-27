# ds-completion plan

Status: **done**, **next** (agreed, in order), **planned** (agreed, later), **idea** (not
decided). Today, 15 expert layers on GPUs, sclk 2: decode ~12.5 t/s, a step ~80 ms = CPU
experts 40 + die-0 GPU ~35 (of which ~7 overhead) + other dies' experts 4.3.

## Overview

Correctness and the base path

| | item | status | one line |
|---|---|---|---|
| 1 | CPU reference forward (ggml Engine) | done | every module of the reference, gated per layer against the official implementation (steps 1-3) |
| 2 | Exact fake quantization | done | fp8/fp4 rounding at the reference's points, so outputs match to ~1e-7 per layer |
| 3 | ggml GPU path | done, removed | dense part on one die through ggml (6 t/s); superseded by the own runtime (steps 4, 11) |
| 4 | fp8 repack of dense weights (B0) | done | lossless 6.4 GB fp8 file instead of 14 GB bf16 |
| 5 | Exact fp8 GEMV (B1) | done | 311 us per layer vs ggml's 832, bit-exact products |
| 6 | Own HIP decode runtime (D3) | done | every op of a token in our kernels on one die: 9.4 -> ~10 t/s |
| 7 | Exact CPU expert kernel (A3) | done | AVX-512 fp4 x fp8, ~68 GB/s, faster than ggml and exact |
| 8 | Exact GPU expert kernel | done | one layer's 6 experts in 0.39 ms on a die (287 GB/s) |
| 9 | Whole expert layers on GPUs | done, replaced | `--gpu-moe`: 15 layers resident on 4 dies -> 12.2 t/s; superseded by item 15 |
| 10 | Prompt through the runtime | done | frees 14 GB on die 0; prompt 6 -> 10 t/s |
| 11 | Remove ggml's GPU path | done | the runtime is self-contained; no fork HIP build needed |
| 12 | GPU clock sweep | done | dense GPU part 38 -> 29 ms from sclk 0 to 8; level 4-6 is the sweet spot |

Speed

| | item | status | one line |
|---|---|---|---|
| 13 | Routing statistics (A0) | done | 30 prompts: fixed hot sets transfer poorly (71% at 128 GB), an adaptive LRU cache reaches 95% with ~11 loads per token (README step 13) |
| 14 | HIP graphs + kernel fusion (D) | started | engram per step + experts mapped at load done (~1%); graphs save little on this ROCm, fusion ~37 -> ~22 launches per layer for ~2-3 ms (README step 12) |
| 15 | Per-layer expert split, CPU and all dies concurrently (A1/EP) | done (steps 1-5) | `--placement`: fixed set from the study, 81% on GPUs, 15.5 t/s on a held-out prompt (README step 14) |
| 16 | Adaptive expert cache (A1 + A2) | next (fixes 1-3 of the expert path done: oracle bound ~25 t/s, study30 17.7-21.7) | start from a prior hot set, load missed experts into HBM in the background (LRU): ~95% hits at 128 GB, est. ~25 t/s |
| 17 | Tensor parallelism, 2-way then 4-way (C1/C2) | planned | split the dense GEMVs over dies: ~-6..10 ms per step, matters once experts are mostly on GPUs |
| 18 | Shared expert by TP, routed by EP (C4) | planned | the dense shared expert splits like any other GEMV under TP |
| 19 | DSpark speculative decoding (E1) | planned | several tokens per weight read; needs the full 1017 checkpoint's MTP weights |
| 20 | Batched prompt processing (F) | planned | read each weight once for many prompt tokens; matters for long prompts |
| 21 | Dynamic expert cache (A2) | merged into 16 | item 13 showed recent usage beats fixed sets |
| 22 | Faster GPU fp4 expert kernel | idea | 287 -> ~600 GB/s; matters once many experts run on GPUs |
| 23 | Weight interleaving for TP (C3) | idea | lay out weights so each die's slice is contiguous |
| 24 | fp8 output head | idea | halves the 2.3 ms head, but is no longer exact vs the reference |
| 25 | Clock level by phase | idea | sclk 4-6 for decode with CPU experts, 8 once GPU experts dominate |

Features and limits

| | item | status | one line |
|---|---|---|---|
| 26 | Context above 4096 with the runtime | idea | a top-k kernel for larger candidate sets |
| 27 | Candidate-block selection (>16k tokens) | idea | the reference's layer-20 path, not implemented anywhere yet |
| 28 | Full 1017 checkpoint | waiting | download pending (user); needed for item 19 and exact safetensors refs |
| 29 | Sampling beyond greedy | idea | temperature/top-p; a non-goal so far |

## Expert placement and cache: implementation plan (2026-10-27)

Goal: each layer's chosen experts split into GPU hits (on whichever die holds them) and CPU
misses, run concurrently. Estimates (README step 13 + unit costs): fixed set ~18 t/s, adaptive
~22 t/s on one machine.

Phase 1: fixed set, no swapping

1. **Placement file** (`route_stats.py --emit-placement`): top (layer, expert) pairs from the
   routing logs, sized to each die's free HBM; each layer's hot experts round-robin across dies;
   die 0 gets fewer. Check: simulated hit rate of that exact placement.
2. **Expert slots per die:** a pool of expert-sized slots (gate + up + down, 18.8 MB) per die,
   filled at startup from the mmap; a device table [layer][384] -> (die, slot) or CPU.
   Check: upload time and memory per die.
3. **Slot-addressed fp4 kernel:** `fp4_moe` takes (slot, weight) lists whose count (0-6) lives
   in device memory. Check: exact against the CPU kernel for 0-6 experts.
4. **Dispatch kernel after the router (die 0):** splits the 6 experts into per-die lists and a
   CPU list, written into the staging block that already goes to the host in one copy.
5. **Concurrent execution + fixed-order sum:** die 0 runs its list; dies 1-3 take the block
   host-staged, run theirs, send results back; the CPU runs only the misses (skipped when 0);
   die 0 sums die 0 + dies 1-3 + CPU in a fixed order (deterministic).
6. **Gate:** `ds-test --runtime --placement ...` on all-CPU (must equal today), all-GPU,
   random, and the real placement.
7. **Measure:** held-out prompts, live hit rate vs simulated, t/s vs ~18, per-layer CPU vs GPU.

Phase 2: adaptive swapping

8. **Cache manager on the host:** the host already knows each layer's misses; LRU per die
   picks victim slots.
9. **Background loads:** a low-priority copy stream per die (mmap -> pinned -> GPU -> repack),
   with a per-step load budget (e.g. <= 16 experts).
10. **Safe table updates:** only at step boundaries, stream-ordered; a victim leaves the table
    before its slot is overwritten, a loaded expert joins only after its copy completes.
11. **Measure:** live hit rate vs 95%, loads per token vs ~11, the loads' effect on the CPU
    miss kernel, t/s vs ~22.

Later refinements (each measured): peer copies for die 1 (XGMI pair) instead of host staging;
a host flag the CPU spins on instead of an event wait (cuts the ~0.07 ms fixed miss cost);
fewer dies per layer to cut staging; warm start from all 30 prompts.

Risks: the host still syncs once per layer to learn the misses (~40 waits per step); the CPU
miss kernel on 1-2 experts and the background loads both change the CPU's effective speed.

Details below: the original brainstorm (sections 1-5) and the re-prioritization (section 6).

## 1. Where a decode token goes today (step 4, `--gpu 0`, 32 threads)

| part | time | bytes | effective | ceiling | count |
|---|---|---|---|---|---|
| GPU: dense (attention, indexer, shared expert, router, hc, head) | ~53 ms | 14.45 GiB bf16/f32 | ~260 GB/s | ~880 GB/s/die (`mlp-matmul` ceiling probe) | ~2700 kernels |
| CPU: routed experts + 80 GPU<->CPU round trips | ~75-95 ms | 4.2 GiB mxfp4 | ~50 GB/s | ~112 GB/s (measured) | 40 x 3 mul_mat_id |
| total | ~150-170 ms | | **6.0 t/s** | | 86 graph splits |

- **The two halves run one after the other.** In every layer the GPU waits for the CPU and the CPU
  for the GPU; nothing overlaps.
- **The GPU half is about half bandwidth and half launches.** 2700 small kernels at a few us each
  is ~15-25 ms; the rest is bf16 GEMV at ~45-60% of HBM (as measured for quantized MMVQ in
  `mlp-matmul`).
- **What each half would cost at its ceiling:**
  - GPU: 14.45 GiB at ~700 GB/s is ~22 ms, or ~12 ms at fp8 bytes.
  - CPU: 4.2 GiB at ~100 GB/s is ~42 ms. That is the floor for experts in host memory.

The CPU experts are the biggest single item: ~55% of the step now, and still >70% once the GPU
side is fast. **Any plan that leaves all routed experts in host RAM tops out near ~20 t/s.**

## 2. Ideas

### A. Routed experts (the biggest lever; prompt-dependent)

**A0. Routing statistics** (cheap, needed by A1/A2)

- Log (layer, expert) per token over a varied prompt set: code, prose, chat, math, other
  languages, long documents.
- Questions to answer:
  - How skewed is the routing?
  - How much does the hot set move between prompts and within one generation?
- The numbers to get: hit rate vs number of resident experts, per layer.
- ds-completion only needs a `--dump-routing` flag.

**A1. Static hot set in HBM (EP over the dies)**

- **Capacity:** after the dense part (see C), each die has ~25-28 GB free, ~100-110 GB in all,
  which holds ~5500-5800 of 15360 experts (~40%).
- **Placement:** the most-used experts, from A0, are spread so each layer's experts are balanced
  across dies.
- **Miss path:** a token's 6 experts per layer resolve to some on dies and some on the CPU. Both
  run concurrently, and the result is the sum of partials.
- **Hit rate:** if routing is skewed (common in MoE: a few hundred experts per layer take most
  of the traffic), 40% capacity may give 60-80%.
- **What it buys:** CPU time drops by (1 - hit rate). At 70%: ~13 ms CPU, overlapped with the
  GPU work.
- **Cost:** per-layer EP dispatch, i.e. the router's ids go to every die and the CPU each layer.
  `hip-moe` measured the "CPU as a fifth EP target" pattern on the previous model: misses hide
  under the GPU wave when there are few of them.

**A2. Dynamic expert cache**

- **What:** keep the hot set fresh by loading new experts over PCIe in the background:
  18.8 MB per expert at ~13 GB/s per card, so ~1.4 ms each.
- **When it pays:** only if the hot set drifts slowly within a generation, which A0 will show.
- **Cost:** more machinery (residency tables, async copies, eviction). Do it after A1, and only
  if A0 shows prompt-to-prompt drift that a static set cannot cover.

**A-overlap. Shared expert in parallel with the routed experts**

- The shared expert depends only on the FFN input x, like the routed ones. So within a layer the
  GPU can run the shared expert (and any routed experts resident on dies) while the CPU runs the
  routed misses. Today they run one after the other: shared on the GPU, then routed on the CPU.
- **What it buys is small while the CPU dominates:** the shared expert is ~70 MB bf16 per layer
  (~35 MB at fp8), ~0.1 ms of GPU time, against ~1.9-2.4 ms of CPU experts per layer. Hiding it
  saves at most ~4 ms per token.
- It matters more later. Once A1 puts most routed work on the dies, the per-layer step becomes
  "GPU experts + shared || CPU misses", and the overlap is what keeps rare misses hidden.
  `hip-moe` measured exactly this umbrella on the previous model.
- It comes for free with the own runtime (section 3): launch the GPU work, then the CPU work,
  then join.

**A3. Faster CPU expert kernel** (helps every miss)

- **Gap:** ggml's plain mxfp4 path reaches ~50 GB/s here. `hip-moe` measured ~95 GB/s with
  CPU_REPACK and 32 threads.
- **Repack:** CPU_REPACK copies the weights (269 GB of extra RAM, next to a 474 GB page cache in
  755 GB), so it does not fit. A repacked copy *on disk* would compete for the space reserved
  for the 1017 safetensors.
- **Custom kernel instead:** AVX512-VNNI over mxfp4 in place, with a single parallel pass over
  all 6 experts x 3 matrices of a layer and better thread use at M=1. The target is ~90 GB/s.
- **Bonus:** it can use the reference's exact fp8 activations (e2m1 x e4m3 products are exact
  in int16/fp32), which removes today's one inexact op (ggml's q8_0 re-quantization).

### B. Dense weights back to fp8 (lossless)

**B0. Format and proof**

- Every dense linear except the head, embedding and a few small ones was e4m3 with 32x32 e8m0
  block scales in the checkpoint. The GGUF's bf16 is an exact expansion.
- Re-blocking is exact. `ds_ref.py` already does it and asserts the round trip.
- Store the original **32x32-block** layout: 1 scale byte per 1024 weights, so 1.001 bytes per
  weight, versus mxfp8's 1.03.
- Bytes per token: 14.45 GiB down to ~8 GiB. Head (1.2 GiB) and hc (f32, 0.16 GiB) stay; the
  router can drop to bf16 (exact, 0.16 GiB saved).

**B1. fp8 GEMV/GEMM kernel for gfx906 (HIP)**

gfx906 has no fp8 hardware, but:

- **Decode:** e4m3 -> fp16 is exact with a shift and mask; the bias difference folds into the
  block scale.
- **Dot product:** `v_dot2_f32_f16` takes two fp16 products with fp32 accumulation, and the
  products are exact.
- **Exact numerics:** with activations fake-quantized to fp8 (e4m3 values are exact in fp16), the
  kernel computes **exactly the reference's fp8 x fp8 GEMM**. Exact numerics come back on the
  GPU for free.
- **Target:** ~70-80% of HBM at M=1 (simple layout, one pass, no int8 re-quantization).

**B2. Same trick for the experts on the dies (A1).** e2m1 -> fp16 is exact, and so is
mxfp4 x fp8.

**B3. The same idea for the CPU** (A3).

**Risk:** writing and tuning GEMV kernels. `hip-moe` has done this for mxfp4 on these dies
(253-330 GB/s, at the latency-bound plateau for that shape). fp8 at 8 bits per weight is easier:
fewer decode ops per byte.

### C. Tensor parallelism over the dies (dense part) and expert parallelism

**C1. Per-layer TP split, 4 ways**

Everything is replicated except where the table says otherwise. Weights are fp8 per B.

| part | split | per-die bytes | reduction |
|---|---|---|---|
| q_a (5120->1280), kv (5120->512), compressor, indexer k | replicated (small) | ~10 MB | - |
| q_b (1280->64x512) | by head, 16 heads/die | 10 MB | - |
| attention core, sinks | by head | - | - |
| wo_a (grouped, 8 groups of 8 heads) | by group, 2 groups/die (aligned with heads) | 8 MB | - |
| wo_b (8192->5120) | by input K (each die's 2 groups) | 10 MB | **all-reduce [5120]** |
| indexer q_b (32 heads) | 8 heads/die | 1.3 MB | **all-reduce [n_comp] scores** (8 layers) |
| shared expert (2304) | by inter dim, 576/die | 9 MB | joins the MoE all-reduce |
| routed experts on dies (A1) | by expert (EP) | - | joins the MoE all-reduce |
| router, hc mixes, norms, engram proj | replicated | ~13 MB | - |
| head (129280 x 5120) | by vocab, 4 x 32320 | 0.33 GB | gather argmax / top-k |

- **Per die per token:** ~8 GiB / 4 + replicated parts, so ~2.3 GiB. That's ~4-5 ms at 600 GB/s.
- **Communication:**
  - 2 all-reduces of 20 KB per layer (attention output, MoE output), plus 8 small indexer
    reductions: ~90 per token.
  - Within an XGMI pair this is a peer copy, a few us each.
  - Across cards it goes through pinned host memory; `macpro-2019-vega-if` measured ~13 GB/s
    pipelined, and latency is what matters at 20 KB.
- **Hierarchical reduce:** reduce within each pair over XGMI, swap one 20 KB partial between the
  cards via the host, broadcast within the pair.
- **Budget:** ~10-30 us per all-reduce, so 1-3 ms per token. That is fine against a ~5 ms GPU
  part.
- **Capacity:** TP frees HBM for A1. With replication, the dense part costs ~2.5 GB/die instead
  of 14.45 GB on one die.

**C2. 2-way first** (one XGMI pair, peer copies only), to get the structure right before
cross-card.

**C3. Weight interleaving for TP**

- Write per-die shards so each die's slice is contiguous: q_b rows by head, wo_b columns by
  group, shared-expert gate/up interleaved for a fused SwiGLU.
- This is a one-time conversion into our own file format, needing ~9 GB at fp8, or slicing at
  load time from the bf16 mmap.
- The per-die shard file is also where B's fp8 re-blocking happens.

**C4. Expert parallelism for the shared expert?**

- The shared expert is dense, one per layer, so TP (C1) splits it directly.
- "EP for shared experts" in the sense of placing whole shared experts on different dies would
  need a reduction anyway and does not balance better.
- **Proposal:** TP for the shared expert, EP for routed experts (A1).

**Risk:** ROCm 6.3.4 crashes on runtime copies between non-peer GPUs (found in
`macpro-2019-vega-if`). Cross-card traffic must use explicit host staging, as the mixtest recipe
does.

### D. Fewer, fatter GPU kernels

- **D0. Measure first:** whether ggml's HIP graphs engage (GGML_HIP_GRAPHS=ON is compiled in)
  under the scheduler and graph rebuilds; what one launch costs here; and a per-kernel time
  breakdown. No rocprofv3 is installed; the in-process `--profile` can be extended, or
  rocprof (v1) can be tried.
- **D1. Stable graphs:** keep the decode graph's shape fixed across steps (inputs in fixed
  buffers, the same topology for every position), so HIP graph capture can be reused.
- **D2. Fusion:**
  - The hc chain (rms + 24-row GEMV + sigmoids + Sinkhorn + hc_pre) is ~15 ops per sublayer.
  - The attention decode chain (q_a, norm, q_b, rope, kv, norm, rope, window, compressed gather,
    softmax with sinks, PV, inverse rope, wo_a, wo_b) is ~30 ops.
  - A few custom kernels per layer instead of ~67 ops.
- **D3. Own decode runtime:** see section 3.

### E. More tokens per weight read (decode batching)

- **E1. DSpark speculative decoding.**
  - The checkpoint's MTP/DSpark heads (3 stages) are in the safetensors, not in the GGUF, and
    the local safetensors are the *base* release. So the 1017 `mtp.*` tensors must be fetched
    (HTTP range reads, a few GB) or taken from a full 1017 download.
  - `hip-moe` measured 1.5x on the previous model at n=3.
  - Verify batches make experts and dense reads shared across 2-4 tokens. The union of routed
    experts grows, though: 6n distinct at most.
- **E2. Batch of independent sequences:** this is a serving feature, a non-goal.

### F. Prefill

- **Today:** 12 tokens take 1.7 s, CPU-expert bound: every token in the batch touches different
  experts.
- **For long prompts:**
  - stream each layer's expert weights to the dies (6.7 GB per layer over 4 PCIe links, ~0.13 s
    per layer, ~5 s per prompt);
  - compute experts there at GEMM rates;
  - overlap the next layer's copy with the current layer's compute.
- **Break-even** around a few hundred tokens.
- **Separate from decode work.** Do it after decode, unless long prompts are the main use.

### G. Correctness work that rides along

- **GPU fake quantization:** comes for free with B1/B2/A3-style kernels (exact fp8 x fp8 / fp4 x
  fp8). It would bring GPU mode back to the 1.8e-2 KL of CPU exact mode.
- **Candidate-block selection (>16k tokens)** for long contexts.
- **Every kernel gets a teacher-forced gate:**
  - against the fp32+quant reference (exact kernels must hit ~1e-7);
  - against the previous implementation for placement-only changes (bit-identical).

## 3. A strategic choice: stay inside ggml graphs, or own the decode path

Most of B, C and D needs things ggml does not give us:

- a new weight type (fp8 with 32x32 scales) with a GEMV;
- custom fused kernels;
- control over multi-die execution and the host-staged cross-card reductions;
- exact fake quantization on the GPU.

Three ways to get them:

1. **Patch ggml** (the fork) with new ops and a new type, CPU and HIP implementations each.
   - Pros: keeps graphs, the scheduler and tests.
   - Cons: every op touches 4-6 files of ggml internals; maintaining a growing patch against a
     moving WIP fork; the scheduler's multi-device model does not express "reduce within the
     XGMI pair, then via host".
2. **Own decode runtime in HIP.**
   - What: per-layer kernels, our buffers, our streams and events, our TP/EP orchestration.
     ggml stays as the loader helper, the CPU expert kernel (until A3), the test reference and
     the prefill path.
   - Pros: full control over fusion, launches, placement and comms. Clear code (one file per
     kernel family). Numerics we define, exact to the reference.
   - Cons: more code to write; we lose ggml's breadth (fine: one model, one machine is the
     stated scope).
3. **Hybrid.** ggml graphs per die for the big pieces, our kernels and comms between graphs.
   Every boundary is a sync, though: ~80-160 per token, which costs what D is trying to save.

**Decided (2026-10-26): option 2 for decode.** Keep the current ggml path as the reference
implementation that every new kernel is gated against. It also stays the prefill path until F.
That matches "correctness, clarity, performance" and the non-goals: no other models or
hardware to stay general for.

## 4. Proposed order (first version, superseded by section 6)

Each item has a gate (correctness) and a number (speed). Rough per-token targets are cumulative.

| # | item | why now | expected decode |
|---|---|---|---|
| 0 | **A0 routing stats + D0 launch/graph measurements** | cheap; both decide later designs | 6.0 t/s (no change) |
| 1 | **B0+B1: fp8 dense + exact fp8 GEMV on one die** (done as standalone: README "Step 5"; per-layer matmuls 832 -> 311 us, exact) | halves GPU bytes; first custom kernel, sets up the runtime; brings exact numerics to the GPU | ~7-8 t/s (CPU-bound) |
| 2 | **D2 / own decode runtime on one die** (fused attention + hc, few launches) (done: README "Step 6", 9.4 t/s, exact and deterministic; CPU experts now 72 of 109 ms) | removes the launch half of the GPU time | ~8-9 t/s (still CPU-bound) |
| 3 | **C2 -> C1: TP 2-way (XGMI pair), then 4-way** | 4x dense bandwidth, frees HBM for experts | ~10-12 t/s (CPU experts now ~75% of the step) |
| 4 | **A1: static hot experts on the dies (EP), CPU for misses, overlapped** | the biggest lever | ~20-30 t/s, depending on A0's hit rate |
| 5 | **A3: CPU expert kernel** (done early: README "Step 7"; exact, ~68 GB/s vs ggml ~60; 9.4 -> 9.8 t/s. It is compute + memory serialized: in cache 168 GB/s, streaming 95) | the misses get cheaper | +10-30% on top of 4 |
| 6 | E1 DSpark, F prefill, A2 dynamic cache | each a separate project | E1 ~1.3-1.5x on top |

Why not experts first? A1 is the largest single win, but:

- it needs HBM that only TP frees: today die 0 holds 14.45 GB of dense weights, and dies 1-3
  are idle but reachable only through cross-device EP;
- its payoff depends on A0;
- running EP over dies 1-3 *while* the dense part stays on die 0 is possible as an early
  experiment. It is cheap to try once A0 says the hit rate is good, and it does not depend on
  B/C. If A0 shows strong skew, it can move up to right after #1.

## 5. Decisions and open questions

Decided (2026-10-26):

1. **Own the decode path in HIP** (section 3, option 2). The ggml path stays as the gate and
   the prefill path.
2. **Per-die fp8 weight files (~9-10 GB) on disk are fine.** Repacked expert copies (269 GB)
   are still not planned, because they would compete with the 1017 download.

Open:

- **Which prompts represent the intended use?** This matters for A0 (the hot-expert set) and for
  whether prefill (F) should move up. Until the user says otherwise, A0 uses a default mix, and
  the routing stats are reported per category so a skew toward one use shows up:
  - short chat Q&A (`--chat`);
  - code: write and explain Python and C++;
  - reasoning (`--think`): math word problems;
  - long-document reading: `prompts/ds-v41-readme.md` at 1-4k tokens, then a question about it;
  - non-English chat: e.g. Ukrainian, Chinese.
- **Is the overlap in A-overlap worth doing before A1?** Proposal: no. It comes with the own
  runtime anyway.

## 6. Re-prioritized after steps 5-8 (2026-10-27)

What steps 5-8 measured, per decode token, sclk level 2:

| piece | cost |
|---|---|
| dense part on die 0 (runtime) | ~35 ms: ~13 fp8 GEMV, the rest small kernels, launches and copies |
| one routed expert on the CPU | ~0.28 ms (memory-bound: 4.5 GB per token at ~68 GB/s) |
| one routed expert on a GPU (resident) | ~0.065 ms (fp4 kernel at 287 GB/s, untuned) |
| moving one expert to a GPU over PCIe | ~1.4 ms (18.8 MB at ~13 GB/s) |
| HBM for experts | ~118 GB after the dense part = ~17.5 layers' worth = ~44% of all experts |

Conclusions:

- **The GPUs can only help with experts already resident.** Copying a cold expert costs 5x
  more than the CPU computing it.
- **The dense part cannot overlap the experts** (each layer needs the previous one's output),
  except for the shared expert.
- **CPU and GPUs should work on the same layer at once.** Whole layers on GPUs (step 8) still
  run one after the other; splitting every layer's experts between the CPU and the dies, and
  running both at once, makes a layer cost the slower part.
- **Tensor parallelism saves ~8-10 ms of the dense 35.** That is <10% of today's step, but
  20-40% once most experts are resident, so it comes after the expert work.

Projected step times (from the unit costs above):

| placement (same HBM) | CPU experts per layer | step | decode |
|---|---|---|---|
| 13 whole layers (measured, step 8) | 1.7 ms on 27 layers | 85 ms | 11.4 t/s |
| per-layer split, 44% resident, uniform routing | ~0.94 ms | ~73 ms | ~13.7 t/s |
| per-layer split, hot set catching 70% of routing | ~0.5 ms | ~55 ms | ~18 t/s |
| per-layer split, hot set catching 90% | ~0.17 ms | ~42 ms | ~24 t/s |
| ... + single-die overhead 35 -> 25 ms, TP 4-way | | ~25 ms | ~40 t/s (optimistic) |

New order:

1. **Free die 0's HBM.** Done (README step 10). The prompt runs through the runtime one
   token at a time, and die 0 now has room for +2 expert layers (15 in all with dies 1-3).
2. **Per-layer expert split (EP over 4 dies + the CPU, concurrently).**
   - A static placement, uniform to start, so no routing statistics are needed.
   - The router's ids are dispatched to the owners (die or CPU) and the partials summed.
   - Per layer: one fan-out and one gather (host-staged to the other card).
   - Expected ~13-14 t/s.
3. **Routing statistics -> hot-set placement.** The biggest lever (~18-24 t/s). It also
   measures how well a hot set built from one prompt category serves another, which is the
   "prompt-specific" question answered with numbers.
4. **Single-die GPU overhead:** HIP graphs for the runtime's fixed launch sequence (positions
   moved into device memory), and fusing the small kernels. 35 -> ~25 ms.
5. **Tensor parallelism:** 2-way over the XGMI pair, then 4-way.
6. **Separately:**
   - DSpark speculative decoding, which amortizes every weight read over several tokens (needs
     the 1017 MTP weights);
   - prefill;
   - a faster GPU fp4 kernel (it matters less once its work runs alongside the CPU);
   - GPU clock levels: measured in README step 9. The dense GPU part gains ~2 ms from level 4
     to 8; the GPU expert kernel gains ~10%. Revisit once more experts live on the GPUs.
