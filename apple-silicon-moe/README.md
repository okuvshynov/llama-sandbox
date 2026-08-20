# apple-silicon-moe — the hip-moe study on M2 Ultra

The hip-moe measurement program (one DeepSeek-V4-Pro MoE layer, batch 1..8,
GPU vs CPU, hybrid placement arithmetic) replicated on Apple Silicon:
Mac Studio M2 Ultra — 16P+8E CPU, 76-core GPU, 192 GB unified memory,
~800 GB/s theoretical fabric. Same graph, same shapes, same synthetic MXFP4
weights, same reporting as `hip-moe/src/moe_ggml_bench.cpp`, so rows compare
across machines.

Regimes (decided 2026-08-19):

1. **Backend**: Metal GPU vs CPU-plain vs CPU-repack (mxfp4 has a NEON
   repack path — `ggml/src/ggml-cpu/arch/arm/repack.cpp`, engages as
   `mxfp4_4x4`). No BLAS regime: Accelerate only serves large dense
   f32/f16 `mul_mat`, and the routed expert op is `mul_mat_id` over
   quantized weights — the BLAS backend never sees it.
2. **Batch n = 1..8** — the speculative-verification regime, with exact
   deduplicated expert-union bytes so effective bandwidth stays honest.
3. **CPU thread sweep** in place of the SMT sweep: no SMT here, asymmetric
   P/E cores instead.
4. **GPU+CPU concurrency**: both backends streaming their own full MoE
   layer at once — the fabric-contention question unified memory poses.
5. No EP/TP — one device.

Build and run:

    make                                   # needs ~/projects/3rd/llama.cpp/build (Metal+CPU, DL=OFF)
    ./bin/moe-ggml-bench                   # Metal sweep, n=1..8
    ./bin/moe-ggml-bench --repack --prio 2 --poll 100   # CPU, the low-variance config
    ./bin/ceilings                         # machine floors, no model
    ./bin/moe-contention-bench --secs 4    # GPU+CPU concurrently

llama.cpp build: `~/projects/3rd/llama.cpp` @ b062ba735 (2026-08-19),
Release, GGML_METAL + GGML_ACCELERATE + GGML_CPU_REPACK, shared libs,
GGML_BACKEND_DL=OFF. Exactness rests on test-backend-ops: 73/73 mxfp4
MUL_MAT_ID cases pass on MTL0. Sanity gates: Metal vs CPU rms(diff)/rms(ref)
9.9e-3 (both quantize activations differently — two correct kernels);
repack vs plain 1.9e-7 (same activation quantization on ARM).

## The machine's ceilings (2026-08-19, `bin/ceilings`)

| probe | result |
|---|---|
| CPU streaming, 16 threads | **246.5 GB/s** (flat at 20/24/32 — E cores add no bandwidth) |
| GPU streaming | **696–720 GB/s** |
| Metal null dispatch, same encoder | 1.24 µs |
| Metal encode+commit+wait round trip | **154 µs** (vs 9.3 µs HIP, ~65 µs Vulkan/Win) |
| Metal sustained commit, wait last | 17.9 µs |

The 154 µs round trip is the number any per-layer CPU⇄GPU sync design must
respect; ggml-metal amortizes it by encoding whole graphs per command buffer.

## Metal sweep (2026-08-19, 5 independent loads, spreads 0.2–4%)

| n | µs/token | GB/s eff | % of GPU ceiling |
|---|---|---|---|
| 1 | 802–834 | 294–306 | 42% |
| 4 | 484–488 | 413–417 | 58% |
| 8 | 417–418 | 472–474 | **66%** |

Unlike the CPU sides of both machines, Metal does NOT saturate its
streaming ceiling — a third of it is overhead (launch, sync, small router
nodes), so there is real kernel/batching headroom on the GPU. Cross-machine:
a single Vega II die ran this layer at ~1.5–3.3 ms/token (hip-moe, before
EP); the M2 Ultra GPU runs it at 0.42–0.81 ms.

## CPU: the scheduler is the instrument (2026-08-19)

Plain vs repack at 16 threads: repack (`mxfp4_4x4`) reaches ~192 GB/s at
n=8 vs plain's ~82–148 — same ~26%-class win as x86, arriving via NEON.

**Thread sweep**: 16 threads (= P cores) wins; 8 → 12 → 16 scales
near-linearly; **20 and 24 threads are the trap** — E-core spill makes
ggml's even row-split straggle, up to 2× worse (the Apple analog of
hip-moe's asymmetric-SMT trap). macOS offers no affinity; priority is the
only lever, and it matters more than anything else measured here:

- Default QoS: load-to-load spreads 20–55%, workers wander onto E cores.
- `--prio 2 --poll 100` (SCHED_FIFO 80 + aggressive spin): spreads
  collapse to ≤1–3%.

CPU-repack, prio 2 poll 100, 5 independent loads:

| n | mean µs/token | min µs/token | GB/s eff |
|---|---|---|---|
| 1 | 2126–2336 | 1184–1219 | 105–116 |
| 4 | 954–959 | 933–938 | 220–221 |
| 8 | 905–913 | 884–893 | **207–208** |

**The CPU side saturates**: 207–221 GB/s against the 246 GB/s ceiling
(84–90%) — the same closure as x86 (92–98 vs 100–107), at 2.2× the absolute
bandwidth. n=1 keeps a bimodal mean (floor 1.19 ms is rock-stable at 2.9%,
mean floats 1.6–1.9× above it under every prio/poll combination) — open
puzzle, see BACKLOG.

Two scheduler traps recorded (both cost an afternoon-hour each):

- **An idle poll=100 threadpool poisons the process**: FIFO-80 workers
  busy-spin from creation; 16 of them spinning through a multi-minute
  weight load left all later compute at a sticky 3.4× penalty (70 vs
  222 GB/s). Create the pool after the load, right before compute.
- **FIFO workers exile their own chief**: under concurrency, the calling
  thread (= ggml's worker 0) at default priority gets pushed to an E core
  by its own P-core-pinning workers, and the whole graph runs at chief
  speed: +663% CPU tax. Boost the driver threads too (SCHED_FIFO on the
  submitting threads) and the tax falls to ~16%.

## GPU+CPU concurrency: the fabric has room (2026-08-19)

`moe-contention-bench`: both backends stream their own 13.5 GB expert set
at n=4, solo then concurrently. Two loads of the winner agree to 0.5%.

| CPU config | Metal solo→conc GB/s | CPU solo→conc GB/s | combined conc |
|---|---|---|---|
| FIFO, drivers unboosted | 429 → 402 (+6.7% tax) | 221 → 29 (**+663%**) | 431 |
| default QoS | 429 → 380 (+12.7%) | 157 → 158 (−0.4%) | 538 |
| **FIFO + boosted drivers** | 430 → 378 (+13.9%) | 221 → 190 (+16.3%) | **568** |

The verdict: **unified-memory contention is mild** — with scheduling done
right, each side pays ~14–16% and the machine sustains 568 GB/s of useful
MoE streaming, 1.32× the GPU-alone figure. The hybrid-placement arithmetic
on this machine: a CPU-offloaded Pro MoE layer at n=4 costs ~1157 µs/token
concurrent (4627/4) vs Metal's ~558 (2231/4) — a 2.1× penalty per displaced
layer, far better than the 6–12× the Vega machine charges over PCIe, and
the CPU side adds real throughput instead of merely hiding under a wave.

## Splitting one request between GPU and CPU (2026-08-19, `moe-split-bench`)

The latency question: one n=4 request, one layer — partition its ~22 unique
experts between Metal and CPU (compact per-pair `mul_mat_id`, the hip-moe EP
layout, pair→token fold on-graph via a [P, n] weight matrix), router on the
CPU, both sides concurrent, host combine. Sanity-gated against the CPU
standard graph every configuration.

Winner (two loads, 3% apart): **k=17 of 22 experts on GPU, shared expert +
router + 5 experts on CPU, wall 1606-1656 µs vs Metal-alone 1938-1945 —
17-21% faster**, and 2.4× CPU-alone. Both sides finish together (GPU busy
~1410, CPU ~1340-1390, router 115-120 serial). With the shared expert on the
GPU instead, the optimum shifts to k=14 and the win shrinks to ~11%: the
shared expert is better spent on the side that also owns the router.

What the sweep taught, beyond the number:

- **GPU DVFS is the anti-split force.** The shared-expert-only Metal graph
  runs 474 µs in a tight loop but ~1060 µs inside the split loop — at
  partial duty cycle the GPU never ramps, so its work does not shrink
  pro-rata as experts move off it. This is also the measured size of the
  "sporadic request" tax the roundtrip estimates warned about.
- **Idle FIFO spinners poison the other side** (third scheduler trap): the
  first harness had worker threads busy-spinning at FIFO priority between
  epochs, and each side's spinner stole a P core from the side still
  working (CPU wall mean 2× its floor). Condition-variable wakeups
  (~10-20 µs) fixed it. Related: with cv-sleeping workers, pool poll=100
  beats poll=0 — the router's ~10 small nodes pay per-node wake latency
  otherwise (router 250 µs → 117 µs).
- **`ggml_gallocr` recycles INPUT tensors' memory mid-graph** (only
  OUTPUT-flagged tensors are protected, `ggml-alloc.c`). A graph whose
  inputs are set once and computed many times reads garbage from rep 2 on
  if later nodes reuse the slot — the CPU repack `mul_mat_id` asserted on
  out-of-range expert ids; Metal read the same garbage silently. Fix:
  allocate inputs in their own static buffer (`input_buf_t`), llama.cpp's
  own pattern. `moe_ggml_bench` is unaffected (its x is consumed by the
  graph-final shared expert, so nothing after it can reuse the slot; its
  ids-as-input graphs compute exactly once).

## Status / next

- [x] Port moe-ggml-bench, sanity gates green
- [x] Ceilings: CPU 246 GB/s, GPU ~710 GB/s, Metal floors
- [x] 5-load variance: Metal (≤4%), CPU repack prio2/poll100 (≤3% at n≥2)
- [x] Contention: ~15% mutual tax at the winning config, 568 GB/s combined
- [x] Split one request GPU+CPU: 17-21% latency win at k=17/22, shared+router on CPU
- [ ] BACKLOG: n=1 bimodality, contention n-sweep, Metal overhead decomposition,
      DVFS-aware split arithmetic for the full 61-layer pipeline
