# hip-moe — the moe-serv kernels on Linux/ROCm

Standalone HIP benchmarks for the shapes that have recorded Vulkan/Windows
numbers (`moe-serv/docs/KERNEL.md`, `moe-serv/docs/MEASUREMENTS.md`,
`vk-latency/README.md`). Same hardware — Mac Pro 7,1, four Vega II dies
(gfx906) — different OS, driver, and shader compiler: Linux + ROCm 5.7
against Windows + AMD Vulkan. No ggml dependency: MXFP4 blocks are generated
synthetically and the CPU reference dequantizes the same bytes with ggml's
documented semantics (kvalues_fp4 + e8m0-half), so the correctness gate is
the probe's (`|diff| <= 1e-4 + 1e-3|ref|`).

Build and run:

    make                      # hipcc, --offload-arch=gfx906
    ./bin/hip-moe-bench all   # or: matmul / tp / tp4 / latency

The kernels are line-for-line ports of `moe-serv/shaders/*.comp` — the
surviving E2+E6 variant (paired-k uint2 loads, 4 columns/thread, 16-entry
broadcast LDS LUT) plus the TP block stages (fused GU → clamp+SwiGLU →
down slice → reduce×router-weight).

## Results, 2026-08-17 (ROCm 5.7.1, hipcc/LLVM 17)

Kernel time, 100 reps, GPU events, warm-up excluded. HIP numbers reproduce
to <1% across processes and across all four dies; the Vulkan column is the
recorded Windows result on the same silicon.

| shape                            | Vulkan/Win | HIP/Linux | delta |
|---|---|---|---|
| gate/up `k=4096 m=2048` ×6       | 95.8 µs    | **105.7 µs** | +10% |
| down `k=2048 m=4096` ×6          | 88.9 µs    | **81.0 µs**  | −9%  |
| unfused block (2×GU + down)      | 280.5 µs   | 292.4 µs     | +4%  |
| TP block, GPU time per die       | 113 µs     | **118.8 µs** | +5%  |

A sign flip between the two matmul shapes (compiler difference: hipcc
allocates 35 VGPRs → 7 waves/SIMD, where the Vulkan driver landed at 6);
net effect on the block is ~+4%, and on the fused TP pipeline ~+5%. The
compute story transfers: the kernel is the same speed on both stacks to
within the usual noise.

The launch/sync floors do NOT transfer — ROCm is 5–7× cheaper:

| null-kernel floor                  | Vulkan/Win | HIP/Linux |
|---|---|---|
| launch/submit, sustained           | 9 µs       | **0.8 µs** |
| launch → host observes done, polled | ~59 µs     | **8.3 µs** |
| same, blocking                     | ~65 µs     | **9.3 µs** |
| 4-die round, launch-all/wait-all   | 87 µs      | **42 µs**  |

End-to-end TP layer on all four dies for real (slice per die, per-rep H2D of
x, 4 kernels/die, D2H of 4×96 KB partials, host sum, polled): **~260–280 µs
wall** (median across two processes; p10–p90 ≈ 251–314). The comparable
Vulkan probe number is ~325 µs (212 µs TP-shaped round + 113 µs GPU), and
moe-serv measured **439 µs/layer in-process** on Windows. Border share here:
~145 µs against ROCm's ~42 µs null floor — so there is still ~100 µs of
non-null cost (transfers, 16 launches, host sum), but the whole layer is
~40% cheaper than the in-process Windows figure before any tuning.

Caveats, honestly held:

- This probe is cache-warm and maps no model. moe-serv's Windows border was
  **cache-eviction-priced** (~2× on every host phase with the trunk
  streaming between calls); that mechanism is OS-independent and should be
  expected to inflate an in-process ROCm border too. The probe-vs-probe
  comparison (266 vs 325) is the fair one; 266-vs-439 mixes probe vs process.
- The synthetic weights sit in the probe's magnitude regime (weight rms
  ~0.1, matching ggml-quantized uniform[-1,1] data). Larger scales fail the
  tolerance gate honestly through fp32 cancellation on near-zero outputs —
  that is data, not kernel (see `random_blocks` comment).
- `latency` uses spin-polling (`hipStreamQuery`), matching `moe_tp.h`'s
  polled-fence choice; blocking sync costs ~1 µs more per round here (it
  cost ~6 µs on Vulkan).

## What this says for the port

The decision-relevant deltas versus the Windows/Vulkan integration:

1. Kernel compute: parity (±5%). Nothing to re-tune before integrating —
   though the E1–E8 ledger was tuned under the Vulkan compiler, and hipcc's
   different allocation (7 waves/SIMD) means the occupancy lever (E6/E7)
   could land differently here; re-run that experiment only if the kernel
   becomes the bottleneck again.
2. The border — the dominant cost and the closed-as-structural chapter on
   Windows — is 5–7× cheaper at the floor and ~40% cheaper end-to-end. The
   levers that were parked as "reachable only with fewer submissions per
   token" (sentinel poll, merged submits) may simply not be needed on ROCm.
3. MoE at ~11% of decode on Windows was mostly border; if the in-process
   border shrinks proportionally, the TP path's +7.6% over stock should
   improve here. Needs the real model (in transfer) + a llama.cpp HIP build
   to confirm.

## ROCm 5.7.1 vs 6.3.4, same-day interleaved A/B (2026-08-17)

6.3.4 installed side-by-side (llama.cpp requires >= 6.1; 6.3 is the last
release whose stock rocBLAS ships gfx906 Tensile kernels — 156 files
confirmed present). Both binaries from the same source, each resolving its
own runtime via soname (libamdhip64.so.5 vs .so.6). Three pairs per row,
alternating; all 30 runs pass the correctness gate.

| row | ROCm 5.7.1 | ROCm 6.3.4 | verdict |
|---|---|---|---|
| gate/up matmul | 105.2-106.1 µs | 106.6-107.0 µs | +1.3%, resolved (LLVM 18 lands 36 vgprs vs 35) |
| down matmul | 81.3-81.7 | 81.2-81.4 | tie |
| TP block GPU /die | 118.7-118.8 | 118.4-118.6 | tie |
| null round trip, polled | 8.7 | 10.1-11.4 | **+2 µs, resolved** |
| 4-die null round | 40.6-42.6 | 46.2-46.9 | **+5 µs, resolved** |
| tp4 layer wall | 263-266 | **296-300** | **+13%, resolved** |

Compute transfers unchanged; the 6.3 runtime's dispatch path costs ~2 µs
more per round trip and ~33 µs more on the full 4-die TP layer (16 launches
+ 12 async copies per rep, so a per-call overhead of this size compounds).
Still 4-6x below RADV Vulkan on every row. Consequence: build llama.cpp
against 6.3.4, but keep 5.7.1 installed — it is the cheaper runtime for the
custom backend, and the A/B costs one extra `HIPCC=` build.

## Stock llama.cpp on ROCm 6.3.4 (2026-08-17)

Built master `60eeeb608` (2026-08-17) with the HIP backend against the
side-by-side 6.3.4 — the version gate (>= 6.1) and the hipBLAS 2.0 API rule
out 5.7. The recipe that keeps the alternatives symlink and the 5.7 install
out of the build:

    ROCM_PATH=/opt/rocm-6.3.4 HIP_PATH=/opt/rocm-6.3.4 \
    HIPCXX=/opt/rocm-6.3.4/lib/llvm/bin/clang++ \
    cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.4 \
          -DLLAMA_CURL=OFF
    cmake --build build-hip -j16

Validation, no model needed:

- All four dies enumerate: `4x gfx906, 32752 MiB each (131 GB), wave 64`.
- `test-backend-ops test -b ROCm0`: **12,926/12,926 passed**. The
  "not supported" lines (f16 ABS/SGN/NEG variants etc.) are declined ops
  that fall back to CPU — normal for any backend, not gfx906 rot.
- `SOLVE_TRI` passes on every shape — the one op with an open gfx906 issue
  upstream (rocBLAS strsm fallback); this master carries the custom-kernel
  path, so the known landmine is already defused.

Carried-over discipline for when the models arrive: a GPU-enabled build is
NOT a CPU baseline (bit-exact gates keep using CPU-only builds, as
logit-kld forces), and watch `sched_reserve: graph splits` whenever an op
falls back — backend-specific-op fallbacks are what shattered prefill on
the Windows Vulkan build.

## The model arrives: full-model baselines (2026-08-17)

Model: DS-V4-Flash-0731-UD-Q8_K_XL, 150.75 GiB, verified sizes + Linux
SHA-256 manifest (`checksums/DS-V4-Flash-0731-UD-Q8_K_XL.sha256`; Windows
cross-check pending). Stub regenerated with `moe-serv/make_stub.py` —
15.78 GiB, byte count matching the Windows instrument, layout check ok.
All rows: stock llama.cpp `build-hip`, `-lm none -t 16 -r 3..5`, two loads
per quoted config. Smokes first: stub loads across all four dies
(engagement from the memory breakdown), CPU and GPU greedy decodes agree
token-for-token over 24 tokens (weak, sampler-level — the KLD gate is still
owed), and the DS4-specific ops (HC_COMB, LIGHTNING_INDEXER) exist under
HIP because the backend compiles the CUDA sources — the ops whose absence
crippled the Vulkan backend.

| tg32, full model | t/s |
|---|---|
| Windows stock (CPU) | 3.46-3.64, 7-9% load-to-load |
| Windows best ever (moe-serv TP) | 3.92 |
| Linux stock CPU (`-ngl 0 -nopo 1`) | 3.50 / 3.59 |
| Linux `-ngl 99 -ncmoe 14 -ts 19/8/8/8` | 9.78 / 9.88 |
| Linux `-ngl 99 -ncmoe 13 -ts 19/8/8/8` | **10.12 / 10.00** |

pp512: 18.25 stock -> **76.4-76.7** at ncmoe 13 (Windows mirror best: 21.84).
Stub decode: CPU 30.2-30.5 (Windows band reproduced), full offload **89 t/s**.
`-ncmoe 12` OOMs under every viable `-ts` (9 heavy layers on one die is
31+ GiB before compute buffers) — ncmoe 13 is the capacity frontier of the
simple layer split. The `-ts` weighting exists because `-ncmoe` strips
experts from the *head* layers: die 0 absorbs all 13 light layers plus six
heavy ones (~24.5 GiB), dies 1-3 take eight heavy layers each (~28 GiB).

What the 10 t/s is and is not: stock llama.cpp, layer-split pipeline
(`-sm layer`) — no EP, no TP, three of four dies idle at any decode
instant, ggml's own HIP mul_mat_id for the on-die experts, CPU experts for
the first 13 layers. The 2.6x over the Windows-best is bought entirely by
capacity: experts in HBM instead of streaming through 75 GB/s DDR4, plus a
trunk that could never leave the CPU under Vulkan. Levers not yet pulled:
the 13 CPU expert layers (the dominant remaining cost), the idle pipeline,
and the custom kernel (2.5x over ggml's *Vulkan* mxfp4 kernel; whether
ggml's HIP kernel leaves the same gap is one `test-backend-ops perf -o
MUL_MAT_ID` away).

## The KLD gate on ncmoe 13 (2026-08-17)

`llama-perplexity -f gate_corpus.txt -c 512`, CPU base (`-ngl 0
--no-op-offload`) vs the serving config (`-ngl 99 -ncmoe 13 -ts 19/8/8/8`):
**mean KLD 8.0e-3 ± 0.7e-3, top-1 96.3 ± 0.8%**, PPL 6.18 vs 6.23 at
99.78% log-correlation.

The MoE-only yardsticks (repack 3.6e-5, mirror 6-8e-5, TP 1-1.8e-4) do not
apply: those varied only the expert matmuls, while this swaps the entire
stack's arithmetic. The recorded precedent for a full-stack swap is the
Apple-clang vs MSVC gap on the same CPU — 8.85e-3 mean KL, 96.67% top-1 —
and this result has the same magnitude and shape (median 4.4e-3, smooth
tail, no position-dependent spikes; the corrupt-weights tell is absent).
Verdict: two correct implementations disagreeing through 43 layers of
rounding. Config cleared.

Boundary of the claim, per the repo's own lesson: end-to-end KL at this
depth saturates and cannot certify kernel exactness — per-op correctness
rests on the 12,926-test backend suite and the token-identical greedy
smoke; this gate rules out the gross failure modes (wrong op, bad weights,
broken placement).

## The custom-kernel question, closed (2026-08-17)

`test-backend-ops perf -o MUL_MAT_ID -b ROCm0`, mxfp4 decode case
(n=1, m=k=2880, 4 of 32 experts): **63.2 µs = 279 GB/s**. Our custom
kernel's own band at its shapes is 253-330 GB/s — ggml's HIP mul_mat_id
lands inside it. The Vulkan-era 2.5x gap (ggml-vk ~163 GB/s) was a
ggml-vulkan weakness, not a ggml weakness; the CUDA-derived MMVQ kernels
on gfx906 already sit at the same latency-bound ~280 GB/s decode plateau
the E1-E8 ledger mapped. Porting the 2-pass kernel to HIP would buy ~0%.

Batch scaling is also healthy — n=8 costs 48 µs/token, n=512 reaches
7.4 TFLOPS — so there is no HIP analog of the Vulkan 8-token vector-path
cliff that forced moe-serv's 8-token chunking.

Remaining levers on this stack, in order of expected value: the 13 CPU
expert layers (dominant), the 3-of-4-idle layer-split pipeline, and
nothing else — the kernel and the border are both settled.

## DSpark speculative decoding (2026-08-17)

Unsloth ships a DSpark drafter extracted from the official checkpoint
(`dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, 10.9 GB, arch dflash; our
build b10472 clears every version window on the card). Q8_0 over BF16:
acceptance measured identical upstream, and gfx906 has no native BF16 —
Q8_0 reads half the bytes on the drafting critical path.

Geometry that made it fit: the target keeps its full ncmoe-13 placement;
the drafter's MXFP4 experts stay in host RAM (`--spec-draft-cpu-moe`) with
only its dense side (~1.5 GB) on the dies. Without that, the drafter tries
a single 10.2 GiB allocation on die 0 and the server dies; the target's
`-ts` does not shape the draft model.

Sweep, llama-server, 6 prompts x 512 greedy tokens each (Python, JS, C,
technical prose, short story, math proof), base 9.84 +- 0.02:

| n_max | mean t/s | speedup | acceptance |
|---|---|---|---|
| 2 | 13.71 | 1.39x | 0.78 |
| **3** | **15.06** | **1.53x** | 0.76 |
| 4 | 12.48 | 1.27x | 0.58 |
| 5 | 11.71 | 1.19x | 0.51 |
| 8 | crash | — | — |

n=3 is the optimum, same as upstream's B200 measurements; the falloff past
3 is steeper here (verify batches are relatively costlier with 13 expert
layers on the CPU). Content matters: creative prose is the consistent
floor (story: 0.66 at n=3, 0.39 at n=4); within code/technical prompts,
run-to-run variance swamps language differences because each config
generates different text. Peak observed: 16.4 t/s. **n_max > 5 does not
degrade — it kills the server**: the documented clamp-to-5 warning fires,
then the first request aborts with `MUL_MAT failed, ROCm error` (worth an
upstream report; something still sizes off the unclamped value). Caveat
carried from upstream #25618: speculative greedy output is not
bit-identical to non-speculative — KLD gates must hold the spec axis
fixed.

## DVFS: a measured null (2026-08-17)

rocm-smi shows sclk swinging 1000-1730 MHz as the layer-split pipeline
moves between dies, raising the ramp-latency question. Pinned
(`--setperflevel high`, verified 1730 MHz on all dies) vs same-day auto,
two loads each, ncmoe 13: tg32 10.19/10.17 vs 10.12/10.00, pp512
77.0 vs 76.6 — +1% decode, inside the load-to-load band, NOT RESOLVED.
The governor keeps up with this workload's burst structure; pinning is
not worth the idle power, and the probe numbers' pinned-clock rep loops
were not flattering the kernels.

Bottom line for the stack: **15.1 t/s mean / 16.4 peak decode** on the
150.75 GiB model — 3.8-4.2x the Windows-best 3.92 — from stock llama.cpp,
capacity placement, and the repo's own drafter.

## moe-ggml-bench: one V4-Pro MoE layer through llama.cpp's kernels (2026-08-18)

`make bin/moe-ggml-bench` — a standalone transcription of `build_moe_ffn`'s
deepseek4 path at DeepSeek-V4-Pro-0813 shapes (7168→3072, 384 experts,
top-6 sqrt-softplus routing with selection bias, norm + 2.5 scale, clamped
SwiGLU, +1 shared expert), synthetic MXFP4 weights (12.55 GiB, fits one
die), llama.cpp's own HIP kernels via the ggml backend API. Batch 1..8 —
the speculative-verification regime. Sanity: same graph on the CPU backend
with routing held fixed, scale-aware gate (rms(diff)/rms(ref) 2.3e-4, two
correct kernels).

| n | µs/graph | µs/token | uniq experts | eff. GB/s |
|---|---|---|---|---|
| 1 | 785 | 785 | 6 | 313 |
| 2 | 1139 | 570 | 12 | 400 |
| 4 | 2192 | 548 | 23 | 384 |
| 6 | 3166 | 528 | 32 | 366 |
| 8 | 4115 | 514 | 41 | 358 |

Findings: the Pro shape runs *better* than Flash's (313-400 GB/s vs 279 —
bigger matmuls amortize dispatch); per-token cost falls only 35% by n=8
because random routing keeps 48 draws ~85% distinct (41 unique of 384) —
real correlated routing reuses more, so the batch benefit here is a lower
bound. Extrapolated: 61 layers ≈ 48 ms/token of MoE at n=1 (~20 t/s
ceiling before trunk), ~31 ms in the n=8 verify regime — Pro is 3x Flash
per layer, tracking its 3x expert bytes (245 vs ~82 MB/token).

Two instrument traps hit and fixed before trusting any number, both
already in the repo's lesson book: `ggml_argsort_top_k` returns a view
(post-compute readback needs a `ggml_cont` snapshot — an output flag on a
view protects nothing), and a plain relative tolerance manufactures
failures on near-zero elements of +-hundreds-scale sums (gate normalized
by reference RMS instead).

## moe-ep-bench: expert parallelism on the Pro shape (2026-08-18)

`make bin/moe-ep-bench` — the same V4-Pro layer distributed EP-style: die d
owns experts [d·384/D, (d+1)·384/D), router + shared expert on die 0, and
each step routes on-die, reads ids/weights back, partitions (token, slot)
pairs by owner, runs a compact `mul_mat_id` batch per die (`--dies 2|4`),
and host-reduces the weighted outputs — moe-serv's run_device_compact on
stock kernels. Phase timers split router / prep+upload / submit / wait+read;
per-die pair counts and solo compute times show imbalance; a Monte-Carlo
pass through the real router gives its expectation. Sanity: CPU backend
with routing held fixed, all n green.

µs/token, chunked (see below), vs the single-die bench as D=1:

| n | 1 die | EP D=2 | EP D=4 |
|---|---|---|---|
| 1 | 785 | 916 | 1071 |
| 2 | 570 | 695 | 640 |
| 4 | 548 | 500 | **421** |
| 6 | 528 | 459 | **374** |
| 8 | 514 | 429 | **376** |

- **The compact layout has an 8-pair cliff on HIP.** `mul_mat_id` keeps its
  MMVQ fast path only while dst->ne[2] stays under a per-type cap
  (`get_mmvq_mmid_max_batch`); the compact form puts P there, and 9 pairs
  cost 3x what 8 do (2662 vs 683 µs). The standard [6, n] layout never
  trips it — which is why the single-die bench and the earlier "no HIP
  8-token cliff" verdict saw nothing. Fix: issue ≤8-pair chunks (moe-serv's
  chunking lesson, HIP edition) — restores ~82 µs/pair linearity for
  +16 µs of submissions.
- **EP loses at n=1, wins 27-29% at verify batches** (crossover n≈2-4;
  D=4 ≥ D=2 above it).
- **Communication dominates, not imbalance.** At D=4 n=8: compute critical
  path 1072 µs, everything else ~1930 (router round trip 684, host
  partition+upload 646, submits 92, readback ~480) — 64% of EP's time.
  Imbalance: measured sample max/ideal 1.08; Monte-Carlo expectation 1.30
  at D=4 n=8 (1.88 at n=1, falling with batch; D=2: 1.11-1.30) — ~30% of
  the compute phase, ~10% end-to-end, shrinking exactly where EP is viable.

Consequence: EP on stock kernels is compute-viable at speculative
verification batch sizes, and its budget is the border family again —
host round trips, staging, per-die submissions — the parts an integration
would attack by keeping routing on-die and overlapping transfers.

## Profiling EP, and the pinned-staging fix (2026-08-18)

`rocprofv3 --hip-trace --kernel-trace --memory-copy-trace` (ROCm 6.3.4,
`GGML_CUDA_DISABLE_GRAPHS=1` for per-kernel readability) on the D=4 run;
analysis from the CSVs, pftrace kept for perfetto. What the timeline showed
at n=4: per-die kernels run gapless (the router's ~20-kernel chain is only
~130 µs of GPU), but dies 1-3 overlapped at just **1.74-1.85x of 3** —
~340 µs of staggered starts — and the host API told the story: 20
`hipMemcpyAsync` calls averaging **87 µs each**, the signature of pageable
staging making "async" copies block. Also priced: 89 launches/step with
hipGraphs off vs ~5 with them on (submit phase 451 vs 82 µs) — keep EP
graphs capture-friendly.

The fix A/B (2x2: `--reorder` x `--pinned`, 50 reps, D=4, µs/graph):

| n | base | reorder | pinned | both |
|---|---|---|---|---|
| 1 | 1056 | 1001 | **809** | 819 |
| 4 | 1680 | 1662 | **1298** | 1364 |
| 8 | 3012 | 2806 | **2135** | 2164 |

**Pinned staging is the whole lever; the submission reorder is a null**
(and slightly negative on top of pinned). The diagnosis refines honestly:
the die-start stagger was never loop order — it was pageable copies
serializing the upload loop as a side effect. With staging in pinned host
buffers (`ggml_backend_dev_host_buffer_type`), the plain loop overlaps
everything on its own. Sanity gate green at every n on the pinned path.

Final EP table, pinned, µs/token vs the single die:

| n | 1 die | D=2 | D=4 |
|---|---|---|---|
| 1 | 785 | 782 | 809 |
| 2 | 570 | 598 | 500 |
| 4 | 548 | 417 | **324** |
| 6 | 528 | 407 | **296** |
| 8 | 514 | 380 | **267** |

EP is now at parity at n=1 and wins **41-48% at verify batches** — double
the pre-fix margin. Remaining known cost: the router round trip (~350 µs
of readback-sync, killable by keeping partition on-die). Lesson for any
integration and for the earlier tp4/EP numbers alike: on ROCm, staging
buffers must be pinned or every "async" copy is a serialization point.

## On-die partitioning: a measured null (2026-08-18)

`--ondie` restructures the step so the post-routing traffic is tiny: full x
goes to every die before routing (overlapped with the router), each die
gathers its pair inputs on-device (`get_rows`), folds pairs into per-token
partials with a [P, n] weight-matrix matmul, and the shared-expert readback
leaves the critical path. Gate green at every n. Result vs pinned host
partition: **+7%/+4%/tie/-3% at n=1/4/6/8** — a null.

Why, from the phases: the router phase only fell 452 → 422 µs (the sh_out
readback was worth ~30), while prep grew 86 → 226 (twelve tiny uploads +
the weight-matrix build cost more than the two bulk uploads they replaced)
and the gather/concat/fold chain added 20-130 µs to each die's graph (die 3
at n=4: solo 619 → 751).

The refined cost map after pinning: EP D=4 carries **~420 µs of serial
router dependency** (x upload + router compute + sync + tiny reads —
nothing downstream can start earlier) plus ~60 µs of readback tail. The
remaining attacks are architectural, not plumbing: replicate routing on
every die (parallel, zero round trip — blocked on ggml having no on-device
stream compaction for per-die pair selection), or fuse routing into the
per-die expert graphs. Both are integration designs, out of scope for this
instrument. Pinned host partition stays the recommended EP configuration;
`--ondie` remains in the tree as the documented null, with its small n=8
edge where the fold matmul amortizes.

## Shared expert off the critical path, and the CPU-offload yardstick (2026-08-18)

**`--shared-late`** (EP): the shared expert depends only on x, so it runs as
its own die-0 graph launched with the expert wave instead of inside the
serial router phase. The router phase fell exactly as predicted (451 →
246 µs — the true serial floor of centralized routing: x upload + routing
compute + sync + tiny reads), but only about a third of the evicted ~200 µs
was net gain: die 0 running its expert slice + shared (~650 µs) becomes the
critical path over die 3 (615), so the wait phase absorbed the rest. Net:
**-5 to -6.5% at n=2-6** (n=4: 315 µs/token, n=6: 276), +6% at n=1, tie at
n=8. Gate green. Dynamic shared placement (replicate 35 MB per die, pick
the least-loaded) was arithmetic-checked and buys nothing — die 0 already
is the least loaded at n=4, and at n=8 the shared work (~3.4
pairs-equivalent) overloads any die. Best config: shared-late for n=2-6,
plain pinned at the edges. Also a free-rider null: pinning the router's x
upload was worth ~nothing (452 → 451) — pageable staging only hurt at
volume.

**`--cpu` on moe-ggml-bench**: the same Pro-shape MoE layer through the
ggml CPU backend (16 threads, plain mxfp4 path — no CPU_REPACK, so mildly
pessimistic vs llama.cpp serving):

| n | µs/token | GB/s |
|---|---|---|
| 1 | 4465 | 55 |
| 4 | 3503 | 58 |
| 8 | 3179 | 59 |

Bandwidth-bound at ~59 GB/s effective, weak batch amortization (-29%/token
by n=8, pure expert-union effect). Cross-checks against Flash serving
(1.4 ms/layer × 2.8× expert bytes ≈ 3.9 ms). The hybrid-placement
yardstick: every Pro MoE layer offloaded to CPU costs ~4.5 ms/token at
decode vs 0.3-0.8 ms on-die — a 6-12× penalty per displaced layer, which
is why the ncmoe count dominates serving throughput.

## The repacked CPU kernels: the yardstick corrected (2026-08-18)

`--repack` on moe-ggml-bench puts the mxfp4 tensors in the CPU_REPACK
buffer type (fetched via `ggml_backend_dev_get_extra_bufts`, the same
route llama.cpp serving takes; engagement proven by the per-tensor
`repack ... with mxfp4_8x8` lines and the buffer name). Gate green
against the plain path.

| n | plain µs/token | repack µs/token | GB/s |
|---|---|---|---|
| 1 | 4465 | **3297** | 74.5 |
| 4 | 3503 | **2585** | 78 |
| 8 | 3179 | **2392** | 79 |

Repack is worth a consistent **~26%**, and the bandwidth column names the
mechanism: 74-80 GB/s effective vs the plain path's 59 — the repacked GEMM
removes the dequant inefficiency and runs at the machine's streaming
ceiling (nano-glm sustained 75; the 4 KiB-page probe 100). Corrected
yardsticks: a CPU-offloaded Pro MoE layer costs **~3.3 ms/token at
decode, ~2.4 ms at n=8**; a distinct cold expert ~450-470 µs — so in the
mixed-placement umbrella arithmetic one cold expert hides under the
~650 µs GPU wave with margin, two stick out only ~300 µs, and the
expected tax for a "1-2 cold pairs per batch" placement drops to ~3-8%.

Gate curiosity, recorded so nobody re-derives it: the repack-vs-plain
diff statistics printed identical to the GPU-vs-plain gate's (2.310e-4
to four digits). Both MMVQ and repack quantize activations to q8_1 while
the plain path quantizes differently, so both comparisons are dominated
by the same shared activation-quantization term; the 26% speed gap is
what proves the code paths differ.

## CPU threads: SMT is the last free lever, and the memory question closes (2026-08-18)

Thread/affinity sweep at n=4 (`--tsweep`, in-process; affinity as the
process axis): scaling is near-linear 8→16 threads (49.5→81.8 GB/s — each
core sustains only ~6 GB/s of demand misses), **32 threads (full SMT)
wins** by +14% over 16 (SMT doubles the outstanding-miss slots the GEMM
needs), affinity (`OMP_PROC_BIND=spread OMP_PLACES=cores`) is a null, and
**20 threads is a trap**: 71 GB/s, *worse* than 16 — asymmetric SMT
occupancy (4 cores doubled, 12 not) makes ggml's even row-split straggle.
Oversubscription must be symmetric.

Variance study, 5 independent loads at 32 threads (the house discipline —
and it paid: fresh runs beat the mid-sweep figure by ~5%, the same
cycling-state effect that made earlier 16-thread runs disagree):

| n | µs/token (5 loads) | spread | GB/s |
|---|---|---|---|
| 1 | 2644-2692 | 1.8% | 91-93 |
| 4 | **2140-2160** | 0.9% | 97.5-98.4 |
| 8 | 1994-2010 | 0.8% | 94-95 |

**The memory-access question is closed by saturation**: 92-98 GB/s against
the machine's measured 100-107 GB/s streaming ceiling (140.8 theoretical —
never reached by any software on this machine, including the dedicated
probe). Software prefetch is dead (the perf counters already showed the
HW prefetcher covering ~13/14 lines at demand time; the residual gap is
smaller than plausible graph overhead), and x86 offers no non-temporal
loads on write-back memory anyway — PREFETCHNTA only limits pollution.

Final CPU yardsticks (repack + 32 threads, cumulatively -40% from the
plain-16-thread starting point, all configuration and zero custom code):
a CPU-offloaded Pro MoE layer costs **2.65 / 2.15 / 2.00 ms/token at
n=1/4/8**; a distinct cold expert **~380-400 µs** — two of them (~780 µs)
barely stick out past the 650 µs GPU wave, putting the mixed-placement
tax for a "1-2 cold pairs" batch at **~2-5%**. (Both cold-expert claims
were later measured directly and corrected — see the umbrella chapter
below: a lone expert runs well under saturation, and the estimate that
follows from that is the per-pair price, not the per-expert one.)

## Mixed CPU+GPU EP: the umbrella, measured (2026-08-20)

The BACKLOG experiment: the CPU as a fifth EP target owning K "cold"
experts. `--cpu-experts K` on moe-ep-bench puts experts [384-K, 384) on
the CPU **only** (CPU_REPACK buffer — the layout llama.cpp serving runs),
and a persistent cv-woken worker thread computes their pairs concurrently
with the GPU wave. Instrument details that turned out to be load-bearing:

- **The A/B is interleaved rep by rep**, not sequential passes: with
  identical work in both arms (a run whose cold set caught zero pairs),
  sequential passes read a -5.4% "tax" at n=4 — GPU clock ramp across the
  first pass. Interleaved, the same null reads -0.5 to +0.4%, an order of
  magnitude under the effects being measured.
- **`--probe` replicates the router on the host** (seconds, no GPU) and
  prints realized cold pairs per (n, K) for the bench's fixed inputs —
  the first run used K=16, which intersects the routing exactly nowhere,
  and measured nothing. Choose K from the probe, not from E[C].
- The CPU graph's inputs are re-set every rep: ggml_gallocr recycles
  INPUT tensors' memory mid-graph (the apple-silicon-moe lesson; only
  OUTPUT flags protect).

**Result: the umbrella is real but shallow — cold pairs hide while their
summed serial CPU cost stays under the GPU wave, and every pair costs a
full ~550-600 µs, distinct expert or not:**

| n | GPU wave (µs) | cold pairs C | cpu-wall (µs) | cold tax |
|---|---|---|---|---|
| 2 | ~510 | 1 | 517-596 | **+1.4-1.5%** |
| 4 | ~860 | 1 | 490-589 | **+1.4-3.5%** |
| 6 | ~1220 | 2 | 1184 | **+0.8%** |
| 8 | ~1720 | 2 | 1197 | **+0.9%** |
| 2 | ~500 | 3 | 1356 | +67% |
| 4 | ~900 | 3 | 1751 | +47% |
| 6 | ~1290 | 5 | 2906 | +76% |
| 8 | ~1840 | 5 | 2861 | +33% |

(Top half K=32/48, bottom K=64; D=4, 16 CPU threads, default scheduling,
50 interleaved reps per arm, sanity gate green on every configuration
including the cold path.)

Mechanisms behind the numbers, each checked directly:

- **No same-expert amortization.** The repack `forward_mul_mat_id` runs
  one gemv per routed row (repack.cpp:4498) — rows are grouped per expert
  only to order the loop. Two pairs on the SAME cold expert cost ~250 µs
  marginal (an L3-warmed re-stream), not ~0. The backlog's "read it once,
  ~free" assumed a kernel that doesn't exist.
- **A lone expert doesn't reach the wall.** Solo price list (16 threads):
  1 distinct expert 488-1003 µs across runs (46-72 GB/s — placement
  lottery; pinning fixes the solo number), 2-4 experts 82-86 GB/s. The
  ~380-400 µs/expert yardstick assumed the saturated 92-98 GB/s regime,
  which one 35 MB stream never enters.
- **Stealing shrinks the umbrella.** The die that loses pairs to the CPU
  gets faster (681→390 µs at n=4/K=64), so cover is thinnest exactly when
  it's needed.
- **Capacity law.** Wave ≈ 215·n µs and pair ≈ 600 µs give a hiding
  capacity of ~n/2.8 pairs, while E[C] = 6nK/384 = nK/64 — both linear in
  n, so the viability criterion is on K alone: **K ≲ 20 hides in
  expectation at any batch size**, and larger n makes it more reliable
  (binomial spread narrows relative to capacity).
- **The serving trade.** The intended use case is miss tolerance, not
  offload: the cold set is chosen for genuine rarity, so most steps have
  C=0 and the design question is what the occasional miss costs. The
  answer measured here: a miss within capacity costs +0.6-3.5% *on the
  step it occurs*, so the amortized cost is (per-step hit rate) x
  ~30-50 µs — for a cold set hit once every few steps per layer, well
  under 1% mean — and an over-capacity burst is a bounded one-step hiccup
  (~600 µs per excess pair), not a persistent tax. Misses in different
  layers each hide under their own layer's wave, so the per-layer
  analysis is the right unit and taxes add linearly. The worst-case
  bound, for calibration: under *uniform* routing (this synthetic
  router), folding measured tax(C) through Binomial(6n, K/384) at n=4
  gives ≈ +3.5% mean for K=8 and ≈ +10% for K=16 — i.e. a cold set that
  isn't actually cold gets expensive fast, which is why the K ≲ 20
  criterion above is the uniform-routing ceiling and a measured hit
  histogram on the real model is what licenses a larger K.

Two more scheduler traps for the ledger, both hit while chasing this:

- **`GOMP_CPU_AFFINITY=0-15` (one thread per core) poisons the whole
  step**: prep 83→503-707 µs and wait+rd ×2.5 at n≥6 — the 16 pinned OMP
  threads own every physical core, so the main thread's prep and the ROCm
  busy-wait spinner starve; measured "tax" hit +133% with the CPU job
  itself finishing in ~525 µs. (Pinning *did* fix the single-expert solo
  cost, 714→488 µs — placement helps the worker and strangles everyone
  else.)
- **`OMP_WAIT_POLICY=passive` is worse**: the CPU side drops to 35 GB/s
  (team wake latency on every graph) and the run segfaulted on the first
  cold rep. Default GOMP (spin-then-sleep) with no pinning is the right
  configuration; the scheduler needs the freedom more than the worker
  needs the affinity.

One measurement footnote: the per-config `cpu-solo` column (measured
right after a GPU-only phase) reads 1.5-2x the in-loop `cpu-wall` —
the cores are cold when it runs (frequency ramp), so `cpu-wall`, taken
inside the alternating loop, is the deployed number.
