# ds-completion plan

Status: **done**, **next** (agreed, in order), **planned** (agreed, later), **idea** (not
decided). Today (sclk 6, 16 CPU threads): `--spec` with the adaptive cache (`--canon
--cache-loads 800 --cache-period 8`) 41.19 t/s mean on the 8 held-out prompts (31.7-50.8; README
step 88; every decode change on the same ruler: `results/heldout-history/`). Prompt processing
93 t/s (63 `--canon`; 1,989 tokens, README step 68). Earlier milestones (sclk 2): no GPU experts
~12 t/s; the study30 placement 17.7-21.7; + `--spec` and the cache at 200 loads/s 25.6. The cost
model below (2026-10-27) says where a token's time went before the cache; today: TIMING.md.

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
| 7 | Exact CPU expert kernel (A3) | done | AVX-512 fp4 x fp8, exact; ~85 GB/s at 16 threads in its benchmark, back to back (0.22 ms per expert in a 6-expert call); in the runtime, after its wait for the GPU, a cold call ~0.24-0.32 ms per expert (README steps 77-82) |
| 8 | Exact GPU expert kernel | done | 4 launches, 4-byte weight decode: 1/2/3/6 experts in 72/110/151/281 us (406 GB/s for 6) (README step 17) |
| 9 | Whole expert layers on GPUs | done, replaced | `--gpu-moe`: 15 layers resident on 4 dies -> 12.2 t/s; superseded by item 15 |
| 10 | Prompt through the runtime | done | frees 14 GB on die 0; prompt 6 -> 10 t/s |
| 11 | Remove ggml's GPU path | done | the runtime is self-contained; no fork HIP build needed |
| 12 | GPU clock sweep | done | dense GPU part 38 -> 29 ms from sclk 0 to 8; level 4-6 is the sweet spot |
| 12a | 16 CPU threads, not 32 | done | 32 starved the host thread and the HIP runtime: every configuration +25-50% (README step 18) |

Speed

| | item | status | one line |
|---|---|---|---|
| 13 | Routing statistics (A0) | done | 30 prompts: fixed hot sets transfer poorly (71% at 128 GB), an adaptive LRU cache reaches 95% with ~11 loads per token (README step 13) |
| 14 | HIP graphs + kernel fusion (D) | started | engram per step + experts mapped at load done (~1%); graphs save little on this ROCm, fusion ~37 -> ~22 launches per layer for ~2-3 ms (README step 12) |
| 15 | Per-layer expert split, CPU and all dies concurrently (A1/EP) | done | `--placement`: fixed set from the study; 17.7-21.7 t/s (README steps 14-18) |
| 15a | Expert path fixes: no die waits for the host; faster small-count kernel; 16 threads | done | oracle bound 16 -> ~25 t/s (README steps 15-18) |
| 16 | Adaptive expert cache (A1 + A2) | done; rate and period tuned (step 50) | `--canon --cache-loads R`: frequency rebalance, held-out `--spec` +28% at sclk 2 (25.6 t/s), 29.1 at sclk 8, outputs identical (README steps 28-33; with steps 50-82 37.16 at sclk 6); tuning: "Further optimization ideas" A |
| 17 | Tensor parallelism, 2-way then 4-way (C1/C2) | planned; step 76 tried wo_a/wo_b over 0-1 (+0.6%, not kept) | split the dense GEMVs over dies: ~-6..10 ms per step, matters once experts are mostly on GPUs |
| 18 | Shared expert by TP, routed by EP (C4) | planned | the dense shared expert splits like any other GEMV under TP |
| 19 | DSpark speculative decoding (E1) | done | `--dspark ... --spec`, output identical to plain greedy; held-out prompts +7% to +33% (geometric mean +22%), oracle placement +19% / +71% (README steps 19-26) |
| 20 | Batched prompt processing (F) | done (steps 37-46, 68: 93 / 63 t/s fast / `--canon`, 1,989 tokens) | README steps 37-38: chunks of up to 512, CPU experts batched, CPU/GPU overlap; byte-identical output: 16.8 -> 51.6 t/s (fast), 17.4 -> 40.0 (canon), 1,989 tokens. CPU and GPU now balanced (~33 s each): step 39 (fp8_gemm, bit-identical): 57.1 t/s fast; now the CPU experts are the limit (30 of 36 s). Options and the expert-streaming estimate: "Prompt processing: where it stands and the options" (P1-P9) |
| 30 | Expert registration stalls the GPU | done | README step 43: not HIP calls but the memory-map lock: one registration call holds it, any mmap/munmap meanwhile waits (up to seconds for the ~48 GB ranges). 64 MB pieces (copies split at piece boundaries): no stall; registration (~95-135 s) still delays the cache's first loads |
| 21 | Dynamic expert cache (A2) | merged into 16 | item 13 showed recent usage beats fixed sets |
| 22 | Faster GPU fp4 expert kernel | partly done | 287 -> 406 GB/s (item 8); the per-row fp8 activation decode is the next cost; matters little while it overlaps |
| 22a | Router + dispatch fusion, copy only what dies need | idea | the per-layer expert overhead (dispatch, copies, waits) is ~7 ms per token at high hit rates |
| 23 | Weight interleaving for TP (C3) | idea | lay out weights so each die's slice is contiguous |
| 24 | fp8 output head | idea | halves the 2.3 ms head, but is no longer exact vs the reference |
| 25 | Clock level by phase | idea | sclk 4-6 for decode with CPU experts, 8 once GPU experts dominate |
| 30 | Further optimization ideas (2026-10-29) | ideas | cache tuning, the dense part of a verify step and TP, speculation tuning, a faster draft: see the section below |

Features and limits

| | item | status | one line |
|---|---|---|---|
| 26 | Context above 4096 with the runtime | done | 65,536 by default, 1M with `--max-ctx`; see "Long context" below (README step 34) |
| 27 | Candidate-block selection (>16k tokens) | done (runtime) | the reference's layer-20 path, checked against its own function; the CPU path still stops at 16k |
| 28 | Full 1017 checkpoint | waiting | download pending (user); needed for item 19 and exact safetensors refs |
| 29 | Sampling beyond greedy | idea | temperature/top-p; a non-goal so far. Greedy loops on some long reasoning (README step 48) |

## Current cost model (2026-10-27, sclk 2, 16 CPU threads)

ms per token. Measured: profiled medians of 253 steps (256-token runs); estimated: from the
measured unit costs below. Two machines = the layers split in half (pipeline), so each half's
experts get twice the HBM; hit rates from the routing study (README step 13).

| ms per token | 1 machine, no GPU experts | 1 machine, fixed (study30) | 1 machine, adaptive | 2 machines, fixed | 2 machines, adaptive |
|---|---|---|---|---|---|
| | *measured* | *measured* (concurrency / code-rust) | *estimated* | *estimated* | *estimated* |
| experts on GPUs | 0% | 81% / 61% | ~95% | ~96% | ~99.6% |
| dense: attention, hyper-connections, norms, engram, output head | 26.1 | 26.1 | 26.1 | 26.1 | 26.1 |
| shared expert | 3.5 | 3.2 | 3.2 | 3.2 | 3.2 |
| router (matrix-vector + top-6) | 3.1 | 3.3 | 3.3 | 3.3 | 3.3 |
| dispatch + copy to host + final add | 1.9 | 2.2 | 2.2 | 2.2 | 2.2 |
| routed experts on GPUs (critical-path part) | - | 4.2 / 3.0 | ~5.2 | ~5.2 | ~5.3 |
| routed experts on CPU (critical-path part) | 51.6 | 11.0 / 23.7 | ~3.5 | ~3.0 | ~0.5 |
| network between machines | - | - | - | 0.25 | 0.25 |
| **step** | **86.6** | **50.6 / 62.2** | **~43.5** | **~43.3** | **~41** |
| **t/s** | **11.5** | **19.8 / 16.1** | **~23** | **~23** | **~24** |
| background expert loads (off the critical path) | - | - | ~210 MB/token (~4.8 GB/s) | - | ~20 MB/token |

(Unprofiled 128-token runs of the measured columns: 12.1 / 21.7 / 17.7 t/s.)

Unit costs: CPU 0.22 ms per expert in a 6-expert call, ~0.3 ms per miss when scattered (1-2 per
layer); GPU 72/110/151 us for 1/2/3 experts on a die, mostly overlapped with the shared expert
and the CPU; per layer a 20 KB copy to the host, the staging to the other dies and events.

What follows:

1. **A floor of ~35 ms (~28 t/s at sclk 2)** whatever the caching: dense + shared + router +
   dispatch. Beyond it: clock (sclk 6-8: dense ~34 -> ~29 ms, GPU experts ~25% faster),
   fusion (~3 ms), tensor parallelism on the ~13 ms of fp8 GEMVs.
2. **A second machine adds ~1-2 ms once the cache adapts.** Without adaptation, two machines
   with a fixed set get what one machine needs an adaptive cache for.
3. **The adaptive cache is worth it** (~20 -> ~23 t/s on chat, ~16 -> ~23 on code), but its
   background loads (~4.8 GB/s) equal today's whole pageable upload speed: pinned staging
   comes first.
4. **At high hit rates the remaining expert cost is fixed per-layer overhead** (dispatch, copies,
   waits: ~7 ms per token), not expert math.

## Speculative decoding with DSpark: batched verification plan (2026-10-28)

Shadow mode (README step 19): with our exact numerics a verified step would yield 2.95 tokens
(chat) and 5.10 (code). This plan turns it into real speculative decoding, greedy, with a
strong gate: **speculative output must equal plain greedy output token for token**.

A verify step: the main model runs [anchor, d1..d5] at positions p..p+5 in one pass; with a_i
its top token after position p+i, it keeps the longest run d(i+1) = a(i) and emits a_0..a_k;
positions after p+k are discarded; a_k is the next anchor; the draft reruns from p+k.

What a rejection must undo:

| state | after rejected positions | approach |
|---|---|---|
| window ring (128 slots per layer) | a rejected token's slot overwrote an older position still in the window | stage the chunk's K/V; write only accepted positions into the ring (commit) |
| compressor groups, compressed KV, index keys | keyed by position, read only after the token completing them | nothing: the real tokens rewrite them (self-healing) |
| draft windows | - | seed only accepted positions |
| n_past, token history (engram) | - | truncate to the accepted prefix |

Phases:

0. **Precondition:** batched kernels must give each token the same bits as single-token runs
   (fp8 GEMV: lanes per row do not depend on M; rows per wave change at M >= 4). Gate: M = 1
   vs M = 6 outputs bitwise equal, for every batched kernel. Done for the fp8 GEMV: all 9 dense
   shapes x M = 2..8 bit-identical to single-token calls, including the switch to 4 rows per
   wave at M >= 4 (`bench-fp8 --bitwise`, `results/fp8-batch-bitwise.txt`; a control checks
   the test can see a difference). The kernels still to be batched (bf16 head, f32 GEMVs,
   hc/norm/rope rows, fp4 expert pairs) get the same test when written in phase 2.
1. **Verify loop, no batching:** the chunk's tokens through today's single-token path one by
   one; rollback by saving/restoring window slots; accept logic; draft re-anchoring. Gate:
   speculative = plain greedy output, several prompts, 256+ tokens, past the 128-token window
   wrap. No speedup (6x the work); it validates the loop. **Done:** identical output on 4 prompts
   (chat, code, math with thinking, Chinese; 300-600 tokens), after one fix: the compressor's
   partial groups are not self-healing (a rejected position can overwrite the slot of a kept
   one with the same pos % ratio), so commit rebuilds them (README step 20).
2. **Batched forward (M <= 6):** GEMVs for all tokens in one weight read (fp8, bf16 head);
   small kernels with rows = M; attention per query with staged keys in the same key order
   as M = 1 (bit-identical); compressor/indexer in position order; experts as (token, expert)
   pairs on GPUs and CPU, each expert read once. Gates: token-for-token as above, and a
   per-layer test of a chunk against the single-token path (bitwise).
3. **Faster draft:** the head read once for 5 positions, Markov step and argmax on the GPU.
   26 -> ~6 ms.
4. **Draft length from the confidence head:** verify fewer tokens when unsure (output stays
   greedy). **Done** (README step 25): sigmoid(conf) is calibrated as is; per step the n with
   the most expected tokens per ms (running cost averages), plus a backoff that skips the
   draft after n = 0. Chat +9%, code +25%, math +14% vs plain; zh -4%.
5. **Measure and document:** held-out prompts, study30 and oracle placements, profiles.
   **Done** (README step 26): 8 held-out prompts +7% to +33% (geometric mean +22%), oracle
   placement +19% / +71%; the cost model became a fitted line (it was stuck at n = 1 under the
   oracle placement).

Expected per verify step (6 tokens, sclk 2, after phases 2-3): fp8 GEMVs ~31 ms, head + small
kernels ~18, attention ~18, GPU experts ~12, CPU misses ~40-60 (study30) or ~10 (good
cache), draft ~6: ~125-145 ms (study30) or ~95 ms. That is chat ~21-24 t/s (~31 with a good
cache), code ~35-40 t/s (~54), against ~20 / ~17 today.

Risks: CPU misses scale with the tokens verified (speculation and the adaptive cache
reinforce each other; phase 4 matters for chat); attention per query may need a multi-query
kernel; bit-identity must hold in every batched kernel (fallback gate: the reference's error
level); the draft's 7.7 GB on GPU 0 costs ~6% of expert slots.

## Expert placement and cache: implementation plan (2026-10-27)

Goal: each layer's chosen experts split into GPU hits (on whichever die holds them) and CPU
misses, run concurrently. Estimates (README step 13 + unit costs): fixed set ~18 t/s, adaptive
~22 t/s on one machine.

Phase 1: fixed set, no swapping (**done**, README steps 14-18: steps 1-7 below, plus three fixes
the oracle test found: no die waits for the host, a faster kernel for 1-3 experts, 16 CPU
threads)

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

Phase 2: adaptive swapping (**next**; the detailed plan is "Adaptive expert cache" below. The
items here are the first sketch.)

7a. **Pinned staging for loads:** today's upload copies straight from the mmap (pageable,
    4.6 GB/s for all dies); the cache needs ~4.8 GB/s in the background. Copy through pinned
    buffers (host memcpy, then an async copy), and measure the effect on the CPU miss kernel.
8. **Cache manager on the host:** the host already knows each layer's misses; LRU per die
   picks victim slots.
9. **Background loads:** a low-priority copy stream per die (mmap -> pinned -> GPU -> repack),
   with a per-step load budget (e.g. <= 16 experts).
10. **Safe table updates:** only at step boundaries, stream-ordered; a victim leaves the table
    before its slot is overwritten, a loaded expert joins only after its copy completes.
11. **Measure:** live hit rate vs 95%, loads per token vs ~11, the loads' effect on the CPU
    miss kernel, t/s vs ~22.

Later refinements (each measured): peer copies for die 1 (XGMI pair) instead of host staging;
a host flag the CPU spins on instead of an event wait; fewer dies per layer to cut staging;
router + dispatch fusion (item 22a).

Risks: the host still syncs once per layer to learn the misses (~40 waits per step); the
background loads compete with the CPU miss kernel for memory bandwidth and with the staging
copies for PCIe.

## Adaptive expert cache: plan (2026-10-28)

Goal: the GPU expert pools follow the experts in use (LRU, warm-started from study30) instead of
a fixed placement. Expected with ~95% hits: steps close to the oracle runs, plain ~23-25 t/s
(today 17-21), `--spec` chat ~27-29 and code ~35-40 t/s (oracle: 29.3 / 40.8).

**The problem to solve first: an expert's bits depend on where it runs.** The GPU kernel sums
with fp16 dot instructions and a lane tree, the CPU kernel with AVX-512 FMAs and another tree,
and each die and the CPU pre-sum their own experts. Both are exact to the reference's error
level, not bit-identical to each other. With a fixed placement that is harmless; with an
adaptive cache, residency depends on when background loads finish, so repeated runs could
differ and `--spec` would no longer equal plain greedy.

- **A. Placement-invariant expert numerics (preferred).** Every expert writes its own output
  row; one kernel adds shared + routed in router-rank order; the CPU and GPU kernels use one
  summation order (the CPU emulating the GPU's, or both moving to exact integer sums per
  32-block). Output then does not depend on placement or cache at all: the gate becomes "any
  cache setting = the CPU-only run, byte for byte", and the cache can be fully asynchronous.
  Risk: the CPU kernel's speed.
- **B. Deterministic cache schedule (fallback).** Cache decisions only from committed tokens;
  a loaded expert counts as resident from position p + D exactly (the dispatch checks per
  token; the host waits if a load is late). Keeps today's kernels; output depends on the cache
  policy, as it depends on the placement today.

Phases:

0. **A or B, by measurement:** test whether v_dot2_f32_f16 rounds once or twice; prototype the
   CPU expert kernel in the GPU's order and measure it. Within ~15% of today's speed -> A,
   else B. **Done: A** (README step 28). Instead of emulating the GPU's fp16 dots, both sides
   compute exact integer block sums (VNNI / v_dot4_i32_i8) with one fixed conversion,
   accumulator layout and tree (`src/canon_dot.h`), plus a shared expf (`src/canon_math.h`).
   Whole expert block GPU = CPU bit for bit; CPU +5%, GPU +13%; accuracy unchanged.
1. **Simulation with realistic constraints** (`scripts/cache_sim.py`, CPU only): load latency, a
   load rate, speculative verify chunks (rejected drafts included). **Done** (README step 29):
   with a time model fitted on the measured runs (4.8% rms), held-out `--spec` goes from 18.8
   (static) to ~26.7 t/s at 50 loads/s and ~28.9 at 100 (unlimited LRU: 33.5, ~8 GB/s).
   Frequency (the most-used experts of the last 256 tokens replace the least-used residents,
   every 16 tokens) beats LRU by 2.5-5.5 t/s at limited rates; rejected drafts do not pollute
   (load from all forwarded tokens); latency of 1-8 steps costs little.
2. **Load path, measured alone.** **Done** (README step 30; `hip/expert_loader.{h,cpp}`):
   direct copies from the registered mmap (registration ~40 s, in the background; it turned out to stall the GPUs meanwhile: item 30), at most 2
   matrices in flight per loader, blocking waits. ~1,400 loads/s possible; live cost with
   `--spec` -1.1% at 100 loads/s, -5.2% at 200 (pinned: -9.0% at 100); output unchanged.
3. **Cache manager on the host.** 3a **done** (README step 31): `--canon` in the runtime, each
   expert writes its own row on any die or the CPU, one kernel sums shared + rows in router-rank
   order; the same output bytes with no placement, study30, an oracle placement, dies 1-3 only,
   and `--spec` at two placements. 3b **done** (README step 32;
   `hip/expert_cache.{h,cpp}`, `--cache-loads R`): frequency rebalance every 16 tokens over the
   last 256, loads at R per second once the expert tensors are registered (~40-90 s, stalling
   the GPUs meanwhile: item 30; no pinned
   loads), victims leave the table before their slot is overwritten, newcomers are published
   when their load has finished; table changes only between forwards. Held-out `--spec`, sclk 2:
   fast static 19.97 -> 25.55 t/s at 200 loads/s (+28%); sclk 8: 22.19 -> 29.09 (+31%); every
   output byte-identical to the static canonical run. Not done: placing newcomers so each
   layer stays spread over the dies (a newcomer takes its victim's slot).
4. **Tune** budget, admission rule and rebalance period (simulation, then live); expose them on
   the CLI with the tuned default. The knobs exist (`--cache-loads`, `--cache-period`,
   `--cache-window`). Open: the default rate (100-200 loads/s), hysteresis so the cache settles
   in long runs (today it keeps loading at the full rate: the boundary of the ranking is noisy),
   live vs simulated hits. **Deferred until after "Long context".**
5. **Measure:** plain and `--spec` on the held-out set; live hit rate vs simulated; loads per
   token; verify-step profile against the oracle's 121 ms.

**Aggressiveness is a setting, not a constant.** Updating is a spectrum: static study30
(~71% simulated hits, 0 loads), rare rebalancing (swap in the top of the last 256-1,024
tokens: ~94%, 12-15 loads per token), continuous LRU (~95%, ~11 loads per token; README
step 13). More is not always better: a load moves 18.8 MB (host copy + PCIe, competing with
the CPU miss kernel and the staging copies), a CPU miss costs ~0.27 ms per expert-token on the
critical path, so a load pays only if the expert is used several times before eviction; too
aggressive thrashes. Expected: diminishing returns, most of the gain from a small budget.
Knobs:
- **Load budget:** at most N experts per second (phase 1: the main knob; per step would tie it
  to the step length, ~3x longer with `--spec`); 0 = today's static placement.
- **Admission rule:** load only after k misses in a recent window (frequency) instead of on the
  first miss (LRU); filters one-off experts, which matters under speculation (rejected drafts
  route to experts that may never be needed again).
- **Rebalance period:** the batch alternative, every N tokens swap in the window's top
  non-resident experts.
Phase 1 sweeps them (hit rate vs loads per token: where the curve flattens), phase 4 picks the
default from live runs; a CLI option (e.g. `--cache-loads N`) keeps it adjustable. Under A every
setting gives the same output bits (a pure speed/resource trade-off); under B each setting is
deterministic but settings differ from each other.

Gates: A: any cache configuration byte-identical to the CPU-only run, plain and `--spec`.
B: repeated runs identical, `--spec` = plain under the same cache settings. Both: a stress test
with a tiny cache (constant evictions) and random placement.

Clock (README step 27): today sclk 2 -> 8 gives only +9% with `--spec`, because the CPU misses
(~70 ms of a 6-token verify step) do not scale with it; with the cache the step is mostly GPU
work, so phase 5 should measure sclk 2 and 8 again.

Risks: the CPU kernel's speed under A; memory bandwidth (loads + CPU misses share ~112 GB/s);
PCIe shared with the per-layer staging copies; hit rate under speculation below the
simulation's 95% (rejected drafts, late loads).

## Further optimization ideas (2026-10-29)

From the verify-step breakdown with the cache (README step 33: per verify step + draft ~122 ms
at sclk 8: dense part ~64%, CPU experts ~20%, draft ~10%, GPU experts ~3% visible). Brief; each
needs its own measurement first. Order: after "Long context".

**Decode profile (README step 49, sclk 6, the cache warm):** a verify step is 62 ms (chat, 2
tokens) to 134 ms (code, 6 tokens); the GPU waits 25-38% of it for the CPU's experts (~0.3 ms
each, the single-token kernel reading at ~63 GB/s); then the attention core (per token, ~10% on
code), the router (~4%), launches (at most 7-12%), the small-batch fp8 kernels (~5%). Order:
the CPU's misses first (fewer: cache tuning, A; cheaper: the single-token CPU kernel's memory
bandwidth), then the batched attention core.

**Done since (README steps 50-52):** the CPU kernel's preparation in parallel (+2.6%), the cache
at 800 loads/s and every 8 tokens (+5.6%; held-out mean 30.3 t/s at sclk 6), a per-layer miss
histogram (cost is linear, ~0.27 ms per distinct CPU expert, little hidden: the hit rate is the
target), `--trace` for Perfetto, and a prefetch of the next layer's likely CPU expert measured
and not built (the guess is right ~20-28% of the time). Step 53: the router's top-k in one launch
for all tokens (+4.6%, held-out mean 31.7 t/s). Step 54: co-selection is too weak to place by
(at most 1-3% even with perfect balance). Step 55: a die's expert time is ~30 us per (token,
expert) pair, but reading each expert once for all its tokens (bit-identical) gained only with
heavy sharing (-21%) and lost without (+23%): reverted, documented in `results/fp4-grouped/`.
Step 56: the CPU's verify-step experts on a static schedule (15-22% faster in a benchmark), 2 MB
pages (no change) and CPU settings (performance governor, C6 off: nothing end to end) all
dropped. Step 57: that "fixed cost" was the multi-token dot routine on one-token groups; those
now use the one-token routine (bit-identical): +1.1%, held-out mean 32.0 t/s. Step 59: decayed
counts equal the window, hysteresis margins only cost, a lower candidate threshold (`--cache-min
1`) gains hits but loses ~0.7% live: the policy stays; more GPU slots are the cache's lever.
Step 60: the memory ledger. Experts are 84% of GPU memory; 64k context costs only 21 slots;
the draft's 8 GB on GPU 0 is 7.2 GB of its own mxfp4 experts, its dense part already fp8 (step
61: nothing cheap to free). Step 62: the output head's multi-input GEMV shares activations
through LDS (bit-identical): head 3.3 -> 2.2 ms at 5 inputs, +1.4% held-out (32.5 t/s). Steps
63-64: layer L's router input through the next layers' routers guesses their experts well (one
layer ahead 72% of the chosen in the top 6; 71% of the CPU's experts in 3 guesses vs 20% by use
counts), but loads must start 3-4 layers ahead, where the guesses are weaker: a GPU prefetch
simulates to ~2.5% at best (bound 10%), likely 1-2% real. Not built. Step 65: the multi-input fp8
GEMV re-decodes inputs per lane group for 64-lane rows (bit-identical fixes: ~1% at 3-4 inputs,
none at 6); the GPU rarely waits for the host (<= 3-5%: engram rows, the sum's prelude). Next:
the batched attention core for verify chunks (and prompt chunks), in stages, each bit-identical:
(1) the window ring at window + chunk slots (done, step 66); (2) attn_core in phases with the
small per-token kernels batched (done, step 67); (3) one attention launch per chunk (done, step
68: attn.core 17.7 -> 7.8 ms at 6 tokens, held-out 32.48 -> 34.33 t/s, the prompt +6-12%); (4)
the indexer batched (past ~2k context; open). Step 69: traces regenerated. Step 70: every decode idea re-scored
(README step 70): next the handoffs around the CPU's experts (~9 ms per code step), then the
cross-die latency on the XGMI pair (for attention over dies 0-1), then a CPU-cache prefetch from
the lookahead guesses; verify steps in halves (the prompt's overlap) tried: exact, 10-13% slower.
Step 71: the handoffs: a short forward's copies back from the dies issued before the host waits
(the CPU's experts start ~5 us after staging instead of ~80-190): held-out 34.33 -> 34.98 t/s,
generation after a long prompt +8-10%.
Step 72: the held-out history: every decode change measured on the same 8 prompts at sclk 6
(older commits rebuilt and rerun overnight; `results/heldout-history/`), 10.06 t/s (step 6) to
the current point; each kept step since adds its point.
Step 73: the engram fetch (~1.4 ms of host page faults while the GPU idled at each step's start)
read with pread, after layer 0 is queued, copied on its own stream: a step reaches layer 1 ~1 ms
sooner; held-out 34.98 -> 35.09 t/s (steps 0.4-2.3% faster on 8 of 9 prompts).
Step 74: the indexer skipped while it keeps every compressed position (<= 512 keys; its scores
were computed and dropped): held-out 35.09 -> 35.94 t/s; batching it for long context is F below.
Step 75: the fp8 GEMV at 2-8 inputs (attn.out ~12 ms of a code step at ~270 GB/s): fp16 inputs,
shared memory, fewer registers, a register cap: none faster at M = 4-6; it needs a new work split.
The traces in `results/trace/` were regenerated with step 74's build (c4bd99b).
Step 76: attention's output projections over dies 0-1 (flags in fine-grained memory, ~6-8 us per
hand-off; exact): +0.6%, not kept (~55 us per layer of hand-offs, and die 1's copy of the weights
cost its expert cache 1.41 GiB); `results/decode-next/attn-split.patch`. With the core and q_b
split too (KV state on die 1) est. +4-6%.
Step 77: the CPU expert call from cache (one expert warm in L2 + L3) is only x1.2 faster than
from DRAM (a plain read: x3-6): DRAM is not its limit, so a prefetch to a large LLC or HBM waits
on a faster call. Step 78 takes the call apart.
Step 78: a warm one-token call is ~2x its compute; ~45% of it is reading activation digits the
team just wrote. Per-thread copies: -25% back to back, nothing after the runtime's idle gap
(end to end flat); not kept. After the runtime's idle gap a call loses only 5-10% (libgomp's
workers spin through it). Step 79 (perf counters): the digits read slowly because 16 threads wrote
them (one writer: no penalty; whole lines per thread: -40%); aligned 16-block chunks +0.2% held-out,
not kept. Step 80: an AVX-512 canon_act_blocks (the same bytes, ~12.5x faster) and one writer per
token / pair: a CPU call -5-14% after the gap, held-out 35.94 -> 36.31 t/s. Next on the CPU: a task
scheduler that claims ahead (overlap memory and compute), then a CPU weight layout.
Step 81: a call's first phase loses ~0.03-0.04 ms after the runtime's idle gap (the master's core
idles); C6 off, polled HIP waits and a worker as the writer do not help; a spinning host wait
helped an A/B but was -0.5% held-out. Not kept.
Step 82: the CPU's 32-row tasks handed out in pairs (dynamic, 2): contiguity, not prefetching the
next task, makes a cold call faster; a call with 2-6 experts -12-13% after the gap, held-out
36.31 -> 37.16 t/s (+2.3%). Next: the CPU weight layout (cheaper decode).
Step 83: the CPU weight layout benchmarked (bench only): the same bits, compute x1.8, a cold call
-9% (8 KB prefetch); est. +1.5-2.5%. Not deployed: it needs a repacked ~269 GB expert file as the
only source of expert weights (CPU kernels, the GPU loader's unpack, model loading).
Step 84: traces split the CPU's handoff back (new `scripts/trace_critpath.py`): the copy of the
CPU's rows waited behind the cache's background loads on die 0's copy engine; copied by a kernel
from the pinned buffer instead: held-out 37.16 -> 37.57 t/s. Left: the dies' own small copies
behind their loads (~3 ms per code step; kernels there and smaller load pieces were slower).
Step 85: the background loads paced by the layers (k matrices per layer right after it, or none
during the expert phase): slower or no change; not kept (`results/decode-next/load-gate.patch`).
Step 86: the draft in the traces (its parts, commit, the loads per die): drafts doubled while the
cache loaded into die 0; the home die now keeps its startup experts (dies 1-3 adapt): held-out
37.57 -> 40.82 t/s. Ideas, not built:
- how out of date die 0's set is: per rebalance its experts' share of the uses the best set of
  that size would serve, and the CPU pairs it would have absorbed (ms); if it grows in long runs,
  its loads at the start of the CPU's experts (die 0 idles ~0.8 ms there), one matrix per layer;
- die 0 seeded with the experts used most across kinds of prompts (the stable ones), dies 1-3
  left for the prompt's own;
- the draft's 384 experts (7.2 GB on die 0): measure its router's usage; if skewed, keep the
  used half (~190 slots freed; acceptance may drop, outputs cannot change), first a cache
  simulation of what the slots are worth;
- the draft's window seeded for all of a commit's kept tokens at once (commit's ~1 ms).
Step 87: the draft router's usage: concentrated per prompt, unstable across prompts (top 64 of
seven prompts cover 88% of the eighth's choices, 70% for the worst); fixed pruning not done.
Idea: a draft-expert cache following the prompt (~120 of 384 used per run; ~4-5 GB). The verify
step's tail: the host's argmax of all logits ~0.3-0.8 ms plus their copy ~0.2-0.3 ms.
Step 88: that argmax on the GPU (M ids back): held-out 40.82 -> 41.19 t/s.
Step 89: traces at step 88; the time outside the layers split: ~2-3 ms untraced, almost all the
head (the bf16 head kernel, as the draft's). Sections B and D below updated.
Step 90: the GPU clock: held-out 36.8 / 41.1 / 42.1 t/s at sclk 2 / 6 / 8; ~35% of a token's time at
sclk 6 scales with the clock, ~65% (the CPU's experts, handoffs, host, and HBM-bound GPU work such
as the fp8 GEMVs; step 94's correction) does not.
Step 91: the CPU layout as a copy in RAM: -7-12% per CPU expert in the same run, held-out +1.6%
with all layers, but 269 GiB, ~45 s per start and an unexplained abort; not kept
(`results/cpu-decode/layout-live.patch`).
Step 92: the draft's head over part of the vocabulary, simulated: every deployable subset loses
(32k -8.8%, with the run's own tokens -4.8%; an oracle at most +4%); not built. Open: an abort at
exit, rare (2 in ~40 runs), seen with and without step 91's code (`malloc_consolidate` / `double
free`, right after the expert cache's stats): a teardown bug to find.
Step 93: that abort not repeated in 56 runs (plain, gdb, ASan: no double free on every run); a
race at shutdown, likely; a stack handler and the exit code in the held-out summary kept for the
next one.
Step 94: step 70's verify halves traced: die 0 reads its weights once per half (two 3-token halves
~84 ms against 54.5 for one 6-token step), more than the CPU's time they could hide; closed.
Step 95: llama.cpp upstream reads our GGUF's Engram table (the PR's MXFP8, type 43) as plain fp8
and generates garbage; the fork stays until a newer model is quantized with the merged code
(`docs/llama-cpp-upstream.md`).
Step 96: the exit abort revisited: its window includes the loader's shutdown (step 93 had it after);
three teardown fixes (stop under the mutex, the loader threads' frees one at a time, the expert
tensors unregistered before the unmap); the handler prints the shutdown phase, the thread and the
loads per die. Still open until it either recurs with that output or stays away.
Step 97: the dies' results in smaller copies and the staged block in one: 41.10 -> 41.96 t/s.
The exit abort recurred with step 96's fixes: after main, in the HSA runtime's shutdown (documented,
not worked around).
Step 98: the compressor's projections once per chunk: 41.96 -> 42.52 t/s.
Step 99: the review's four leftovers: none kept (patches in results/decode-next/).

**Triage (2026-11-07, after README step 90).** A 6-token code step (~89 ms traced): die 0 waits
~25 ms for the CPU's experts (the part the GPU clock does not touch, step 90), die 0's dense front
~54 ms (fp8 GEMVs ~25), the draft 8.4, the tail after the CPU's rows 6.3; outside the layers ~2-3
ms untraced (step 89). Die 0 and the CPU still take turns.

Promising, in order:
1. Tried (step 91, not kept): +1.6% with all layers as a copy in RAM, too costly for the gain.
   Only as a repacked expert file that is the only source (the GPU loader reading it), if ever.
2. Closed (step 92, simulated): a vocabulary subset for the draft's head loses at every size.
3. Closed (step 94): verify halves double die 0's weight reads (+~30 ms per code step), more
   than the CPU's 25-30 ms they could hide.
4. Done (step 97): the tail was the dies' worst-case copies, not their loads: 3 rows per token
   first, the staged block in one copy, +2.1%. With it the compressor batched (step 98, +1.3%).
   The review's leftovers (step 99): hc kernels fused, the compressor's state per chunk (prompt
   +1.1%, decode 0), a gather stream per die: tried, not kept; the draft's head in fp8 waits for
   the multi-input fp8 kernel.
5. A simulation of ~190 more slots on die 0 (`scripts/cache_sim.py`): what a draft-expert cache
   would buy, before building one.
6. Commit's 0.86 ms wait: the draft's window seeded for all kept tokens at once (~1%).

Big, a design first: the fp8 GEMVs at 6 tokens (~25 ms; local variants failed in 65 and 75);
drafting ahead while the verify runs (~2-4% on code; memory and a second stream or die).

Backlog (less likely): cache policy tweaks (59 lost; 63-64 2.5% simulated), an LLC prefetch (77:
x1.2 ceiling), CPU micro-fixes around the idle gap (78, 79, 81), load pacing (85), attention over
dies 0-1 / TP (76: +0.6%), small-op fusion (65: launches not the limit), speculation tuning and
trees (94% of the hindsight bound), the fast path's activation (held-out runs `--canon`), PGO and
compile-time shapes, the batched indexer (long context), die 0's staleness metric (long runs).

**A. Tuning the expert cache** (the CPU experts' ~25 ms)
- Hysteresis: a newcomer must beat its victim by a margin, so the cache settles in long runs
  instead of churning at the noisy boundary; window length / decayed counts; the default rate
  (100-200 loads/s). Simulate first (`scripts/cache_sim.py`), then live.
- Place newcomers so each layer stays spread over the dies (today a newcomer takes its victim's
  slot).
- More slots: compressed caches in fp8/fp4 (part of "Long context") and anything else freed on
  die 0.
- The fast path's activation (`fq_fp8_host`, scalar, written in parallel 8-block tasks): the
  canonical path's step 80 (AVX-512, one writer per token / pair) has no fast-path counterpart;
  the CPU's fixed per-layer latency (~0.03-0.04 ms after the idle gap, step 81).
- Hardware (discussed, not planned): DDR5 or a multi-socket box with experts striped by rows
  across sockets (exact under the canonical numerics).

**B. The dense part of a verify step** (updated after README step 88: a 6-token code step's die 0
spans in step 86's traces, ms per step: attn.out 11.5, ffn.shared 8.9, attn.core 5.7, attn.q 5.1,
hc 5.0, the wait for dies 1-3's experts 5.1, attn.compressor 4.0, dispatch 2.5, router 2.4, the
head 2.2, attn.kv 1.6, engram 1.3, indexer 1.2)
- Done: the router's top-k in one launch (step 53), the head through LDS (step 62), the batched
  attention core (steps 66-68: 17.7 -> ~6 ms), the engram fetch off the critical path (step 73),
  the indexer skipped while it picks every position (step 74), the head's argmax on the GPU
  (step 88).
- Tried, not kept: verify steps in two halves (step 70), the fp8 GEMV for 2-8 inputs as local
  variants (steps 65, 75), attention's output over the XGMI pair 0-1 (step 76).
- Left: the fp8 GEMVs at 6 tokens (attn.out, shared expert, q: ~25 ms, weights read at ~270 GB/s
  against ~580 at one token) need a restructured kernel: more rows per wave without holding M
  decoded inputs (step 75's conclusion).
- Kernel fusion of the small ops (hc, norms, quantize; item 14): launches are not the limit
  (step 65: the host is ahead), so only the GPU time of the fused ops would be saved.
- Tensor parallelism (items 17, 18, 23): step 76 split attention's output over the pair 0-1 and
  lost to the exchange; not before the kernels above.

**C. Tuning speculation**
- The draft-length rule refits its verify-cost line online, so with the cache it already sees
  cheaper verifies; check that it now picks longer drafts and whether the backoff still pays.
- A tiny tree (top-2 at the first draft slot): the sandbox's offline study (`dspark-tree`, on
  V4-Flash) found trees buy little (+0.2 accepted tokens for one extra draft token, +0.7 for 25x
  the budget), so only the smallest trees are worth a test.
- Per-prompt-type expectations: code accepts ~2x more than chat; nothing to tune, but the
  held-out mix decides the average.
- Momentum (trust drafts more after recent successes): checked and rejected (README step 47):
  past success adds nothing beyond the confidence head; rechecked live with the cache (step 48):
  the same, and the head stays calibrated.
- The rule is at 94% of a hindsight lower bound per step (step 48): little left in draft-length
  choices; the cost side (verify step, draft) is where speculation still gains.

**D. A faster draft** (updated after README step 88: 8.3 ms of GPU time per call on die 0, GPU-bound
(step 86); 3 layers, fp8 dense weights (step 61), 128 mxfp4 experts top-3 per layer, all resident)
- Its parts per call (step 86's traces): the head 2.2 (the main model's bf16 head, 1.32 GB, for 5
  tokens), routed experts 1.7, the Markov part 1.0, attn.out 0.8, attn.core 0.75, attn.proj 0.6,
  shared expert 0.5, hc 0.3.
- Done: fp8 dense weights (an exact repack at load), the head through LDS (step 62), no cache
  loads into its die (step 86: drafts had doubled while die 0 loaded).
- The head over a subset of the vocabulary (the most frequent ~32k tokens by a row-index list,
  from prompts outside the held-out set): ~2.2 -> ~0.6 ms; outputs cannot change (the verify
  decides), its confidence is its own projection (not the logits); acceptance may drop on chat
  in other languages.
- Its experts in less memory (step 87: concentrated per prompt, not across prompts): a draft
  expert cache following the prompt (~4-5 GB of the 7.2 freed for the main model); a cache
  simulation first.
- Overlap: start drafting as soon as the verify's accepted prefix is known (a few ms), or draft
  ahead assuming every draft is accepted, kept only if it was; on another die it needs the
  draft's weights there.
- Batching it with the verify (one forward for both) is the larger structural idea, and the
  hardest.

**E. Build-level specialization** (2026-11-01, after README step 45's -O2 finding)
- PGO (`-fprofile-generate`, a prompt or two, `-fprofile-use`) for the host code. Expected
  small (0-5%): the hot CPU loops are hand-written AVX-512 whose instruction mix the numerics
  fix; PGO helps inlining, branch layout and unroll choices, which mattered once (-O2) and is
  fixed. Risk: GCC fuses multiply-adds by default (`-ffp-contract=fast`) and PGO changes
  inlining, so plain C++ numeric code (SiLU, fake quantization) could change bits between
  builds; the canonical numerics must stay bit-identical to the GPU's, so every PGO build
  would need the gates. First `-ffp-contract=off` on the numeric files (bits independent of the
  optimization level), then a ~30-minute experiment (speed and output bytes).
- Compile-time shapes (E = 5120, F = 2304, 6 experts per token) as template constants in the CPU
  kernels: full unrolling without runtime bounds; cheap, deterministic per build.
- The GPU side has no practical PGO in ROCm's device compiler; its equivalent is the kernel
  sweeps already done (README steps 39-40).

**F. The indexer batched over a verify step's tokens** (2026-11-03, long context only; README step 74)
- Today the indexer (layers 2, 8, 14, 20, 24, 28, 32, 36) runs one token at a time inside
  `attn_prep`: ~6 launches per token, `idx_q_b` (fp8) and `idx_proj` (bf16) read again for every
  token, ~76 us per token and source layer at short context (~3.4 ms of a 6-token code step).
  Step 74 skips it while every compressed position is picked (<= 512: up to ~1,024 tokens on
  layers 2-19, 512 on 20-39), so this only matters beyond that, where scoring also grows with
  the context (`k_index_scores` over n / ratio keys, then `k_topk`).
- Batched: the two projections, the rope and the fake quantization for all M tokens in one
  launch each (they do not depend on the compressor's per-token state; `q_b` is already batched
  the same way); one scores kernel for M queries with each token's own key count (a group
  completed mid-chunk adds a key for the later tokens only), reading every key once for all M;
  then the top-k per token (M blocks in one launch), the candidate blocks likewise.
- Exactness: each token's scores and picks must equal the one-token path bit for bit (the
  batched gemv rows already are; the scores kernel needs the same per-key reduction order, the
  top-k the same tie rule); gate with an `attn_check`-style comparison and the 6,106-token
  prompt.
- Measure first: decode t/s against position (step 34's 0.3k / 7.4k / 29.6k points) and the
  indexer's share of a verify step there.

## Long context: plan (2026-10-29; before any further cache tuning)

The model supports 1,048,576 positions (YaRN: a 65,536-token training context x 16). The runtime
stops at 4,096 tokens in all (prompt + generation; refused or capped with a message, never
silently wrong). Checked against the official `inference/model.py`, what already holds at any
length: YaRN only on the compressed layers (compress_rope_theta 160000, factor 16, beta 32/1),
plain rope on window-only layers; rope angles with full-precision cosf/sinf; the 128-slot window,
compressor groups, indexer top-512, engram, the draft's window. What is missing:

1. **Top-k over large candidate sets.** **Done** (README step 34). The indexer keeps the best 512 of all compressed
   positions (n / ratio: up to 1M on layers 20-39). Today's kernel holds every candidate in
   shared memory (`kTopkMaxN` = 4096). Needed: a global-memory top-k (radix select on the score
   bits, then the 512 survivors sorted), bit-identical selection to the reference's `topk`
   (ties by lower index), for N up to 1M; the CPU Engine the same.
2. **Candidate-block pre-filtering** (`select_candidate_blocks`): **done in the runtime** (step 34). layer 20's indexer scores
   blocks of 8 positions by their best score and keeps the top 2,048 blocks (16,384 positions);
   the later index sources (24, 28, 32, 36) mask everything outside them before their own top-k.
   A no-op below 16,384 compressed positions (all blocks kept), so it only changes outputs
   beyond 16k tokens. Runtime and CPU Engine.
3. **Compressed caches in their real formats.** Open (matters beyond ~64k). Compressed KV and index keys are fp32 on die 0
   (~0.6 GB at 64K, ~10 GB at 1M: ~500 expert slots). The reference stores them as fp8 / fp4 and
   our values are already rounded to those formats, so storing them so is exact and 4-8x
   smaller.
4. **`--max-ctx` up to the full range.** **Done:** the runtime defaults to 65,536 (the original
   training context; ~420 MB on GPU 0), any value up to 1M with `--max-ctx`.
5. **Gates.** (a) **done** (`ref/check_topk.py`: 46 of 46 cases, 513 to 1M positions); (b) and (c)
   **done differently** (README step 35): the whole reference at 20,000 tokens took 42 minutes,
   not a night, so the runtime was gated against it directly: per layer (teacher-forced) and end
   to end (9/9 greedy, KL as at short context), and every indexer disagreement ranked against
   the reference's own scores (near-ties). The CPU Engine still stops at 16k (not needed as a
   second path any more). The original plan: The Python reference runs one layer at a time on the CPU: a 20K-token golden run of
   the whole model is very slow. Instead: (a) kernel-level checks of top-k and candidate blocks
   against the reference's functions on synthetic scores at 4K-1M candidates (bit-identical
   indices); (b) a teacher-forced per-layer gate at long positions for a few attention layers
   (2, 20, 24) on a ~20K-token prompt; (c) end to end: the runtime vs the CPU Engine on a
   ~20K-token prompt (both ours, independent paths), plus greedy agreement on long documents.
6. **Measure** (first points, step 34: 18.5 / 17.5 / 17.2 t/s at ~0.3k / 7.4k / 29.6k
   positions, fast static): decode speed against position (indexer scoring grows linearly with context);
   prompt speed (token by token then: a 64K prompt took 1-2 h, which made item 20, batched
   prompt processing, the next priority; now ~50 t/s, README steps 37-41).

## Prompt processing: where it stands and the options (2026-10-30)

Done (README steps 37-39; every output byte-identical to token by token): chunks of up to 512,
CPU experts batched per expert, the CPU's experts overlapped with the GPU's work on the other
half of a chunk, an fp8 matrix-multiply kernel. 1,989-token prompt at sclk 2: 16.8 -> 57.1 t/s
(fast), 17.4 -> 40.5 (`--canon`).

Where a prompt token goes now (36.2 s for 1,989 tokens, fast): the CPU computes experts 30 s
(~15 ms per token), the GPU is busy ~27 s (~13.6 ms per token: attention core 10.2 s, attention
output 3.5, router 2.9, indexer 2.5, shared expert 1.9, compressor 1.7, q 1.6, the rest ~3) and
waits 9 s for the CPU. With `--canon` the CPU's share is about twice that (its kernel gains
little from batching).

### Estimate: stream the CPU's experts to the GPUs instead (no code yet)

Measured costs (README steps 30, 37): loading one expert (18.8 MB) runs at 1,437 loads/s over the
4 dies (27 GB/s): ~0.7 ms of transfer per expert, whatever its number of tokens. The CPU costs
0.086 ms per expert-token (fast) or 0.17 ms (canonical) once an expert has 8+ tokens, ~0.27 ms
with 1-2. The GPU's fp4 kernel is ~18 us per expert-token on one die. So streaming an expert beats
computing it on the CPU above ~8 of its tokens in the chunk (fast), ~4 (canonical).

A chunk of B tokens gives each expert ~B/64 tokens (6 of 384), the non-resident ones ~B/90 (the
GPUs already hold the popular ones: ~60% of expert-tokens on 43% of the experts):

| B | tokens per non-resident expert | stream them all? |
|---|---|---|
| 256 | ~3 | no (canonical: about even) |
| 512 | ~6 | about even (fast); yes (canonical) |
| 1,024 | ~11 | yes |
| 2,048+ | ~23 | clearly |

A hybrid moves the threshold down: per chunk and layer, stream the non-resident experts with the
most tokens, leave the rare ones to the CPU. Can the loads keep up? ~220 non-resident experts per
layer = ~150 ms of loads per layer, independent of B; the GPU's own work is ~0.34 ms x B per layer
(~175 ms at 512, ~700 at 2,048), so loading layer L + 1's experts while L computes hides them
from B ~ 450. Result: the CPU leaves prompt processing; the limit becomes the GPU's own work
(today ~13.6 + ~1 ms per token for the extra experts: ~65-70 t/s, and the same with `--canon`,
since the GPU's canonical expert kernel exists and is bit-identical). The CPU path tops out at
~85 t/s fast / ~45 canonical even with perfect batching; with streaming, a faster GPU side
(attention, matrix kernel) could reach ~200 t/s, the loads (~3 ms per token at B = 2,048) being
the floor.

Caveats: 1,437 loads/s was measured alone (copy engines and staging share PCIe with compute); the
direct path needs the 289 GB registered (item 30's ~90 s stall: fix first; the pinned path reaches
the same rate but costs ~4 CPU cores); spare slots per die to stream through (~20-40: 0.4-0.8 GB);
larger chunks cost ~2 MB per token on die 0; real routing is skewed, so measure the per-chunk
token counts on a long prompt before building it (cheap: a routing log and a script).

### All the options, by part

| # | option | part | expected | cost / risk | depends on |
|---|---|---|---|---|---|
| P1 | measure tokens per non-resident expert per chunk size on a long prompt | estimate | done (README step 42): at 512-token chunks ~110 non-resident experts per layer with ~10 tokens each; the hybrid (47% streamed) brings the expert side ~15 -> ~3 ms per token, the GPU's own work (~11 ms) then the limit: ~58 -> ~80 t/s, canonical too | - | - |
| P2 | stream non-resident experts to the GPUs for prompt chunks (hybrid: hottest streamed, rare ones on the CPU) | CPU experts | tried and dropped (README step 44): correct, 48.8 -> 65.3 t/s (`--canon`) at best, but loads ran 3x slower than standalone (unpacks queued behind the dies' compute); too complicated for the gain; code removed | - | - |
| P3 | fix the registration stall | infrastructure | done (README step 43; item 30) | - | - |
| P4 | larger chunks (1,024+) | CPU experts | tried, slower (README step 46): after step 45 the CPU kernel is arithmetic-bound per pair (0.114 -> 0.107 ms), and the overlap gets coarser; 78.6 -> 73.8 t/s fast | - | - |
| P5 | canonical CPU kernel tiled (rows x tokens, as the fast one) | CPU experts, canonical | done (README step 41): 0.17-0.19 -> 0.14 ms per expert-token, `--canon` prompt 40.4 -> 48.8 t/s, bit-identical | - | - |
| P6 | batched attention core (many queries per call; the window ring needs a separate staging area for the chunk's own keys) | GPU | attention core 10.2 s -> ~1-2 s | medium-large; outputs must stay bit-identical per query | - |
| P7 | batched router, indexer and compressor (per-token loops today) | GPU | ~7 s -> ~2 s | medium | - |
| P8 | fp8_gemm tuning | GPU | done as far as cheap knobs go (README step 40): 8/16-lane shapes +0.3-0.4x; 64-lane shapes unchanged (~20% of `dot2` peak at sclk 2; the instruction mix caps ~40%); no end-to-end change while the CPU is the limit | further gains need a different weight decode / formulation | worth more once the GPU is the limit |
| P9 | dense work of a chunk on several dies (layer pipeline over dies, or splitting the matrix rows) | GPU | up to ~2-4x on the GPU side | large; XGMI pairs only (0-1, 2-3), cross-pair copies over PCIe | P6-P8 first |

Order: P1 and P3 done; P2 tried and dropped (README step 44: no dynamic expert loading during the
prompt); the CPU kernel's in-place overhead found and fixed (README step 45: the runtime was
built with -O2; 55.5 -> 76.0 t/s fast, 48.7 -> 56.7 canonical); P4 tried, slower (step 46). Now
the fast numerics are balanced (CPU ~20.5 s, GPU ~21.5 s for 1,989 tokens at sclk 6) and `--canon`
is CPU-bound: further gains need the CPU kernel's arithmetic (its instruction mix) and the GPU
side (P6, P7) together; P9 last.

## History

Sections 1-6 below are the original brainstorm (2026-10-26) and the first re-prioritization;
their numbers are from their dates and describe paths that have since changed: the ggml GPU
path they start from was removed (README step 11; the CPU Engine is now the gate), the 32-thread
numbers were superseded by 16 threads (step 18), and the order in section 6 has been carried
out up to the adaptive cache. The overview above is kept current; the cost model above is from
2026-10-27, before the cache (today's breakdown: TIMING.md).

## 1. Where a decode token went at step 4 (ggml GPU path, 32 threads)

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
Step 76 tried the output projections alone over 0-1 (+0.6%, not kept; README step 76).

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
(Later: the prompt moved to the runtime and ggml's GPU path was removed, README steps 10-11;
the CPU Engine remains the reference.)
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
   the prefill path. (Done; the prefill moved too, README steps 6, 10, 11.)
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
2. **Per-layer expert split (EP over 4 dies + the CPU, concurrently).** Done (README step 14).
   - A static placement, uniform to start, so no routing statistics are needed.
   - The router's ids are dispatched to the owners (die or CPU) and the partials summed.
   - Per layer: one fan-out and one gather (host-staged to the other card).
   - Expected ~13-14 t/s.
3. **Routing statistics -> hot-set placement.** Done (steps 13-18). The biggest lever (~18-24 t/s). It also
   measures how well a hot set built from one prompt category serves another, which is the
   "prompt-specific" question answered with numbers.
4. **Single-die GPU overhead:** HIP graphs for the runtime's fixed launch sequence (positions
   moved into device memory), and fusing the small kernels. 35 -> ~25 ms. (Started, step 12:
   graphs save little on this ROCm; fusion estimated at ~2-3 ms.)
5. **Tensor parallelism:** 2-way over the XGMI pair, then 4-way.
6. **Separately:**
   - DSpark speculative decoding, which amortizes every weight read over several tokens (needs
     the 1017 MTP weights);
   - prefill;
   - a faster GPU fp4 kernel (it matters less once its work runs alongside the CPU);
   - GPU clock levels: measured in README step 9. The dense GPU part gains ~2 ms from level 4
     to 8; the GPU expert kernel gains ~10%. Revisit once more experts live on the GPUs.
