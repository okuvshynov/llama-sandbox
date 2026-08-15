# The measurements

Every number this project quotes, in one place, each with its instrument and
its noise floor. The history and the reasoning live in `../PLAN.md` and the
commit messages; this file is what was measured.

**The discipline** (learned here, enforced by `bench.py`):

- `-lm none` always; a single mmap-backed timing on Windows is worthless.
- Every configuration on more than one load, round-robin — load-to-load spread
  is larger than within-run spread and is the real error bar.
- A run proves its own noise floor, and a delta below it is NOT RESOLVED. The
  real model's decode floor has ranged **0.3-9% by day**; the 4-layer stub
  (`make_stub.py`) holds 0.0-2.3% and is the decode instrument. Only a
  quantity that transfers (a fixed per-event cost, not a percentage) may be
  extrapolated from it.
- Same-day interleaved A/Bs only, for small effects: stock stub decode moved
  28.4 -> 30.8 t/s overnight, and even an unchanged DLL's phase profile moved
  429 -> 463 µs. Cross-day comparison of either t/s or timers is unquotable.
- Every run proves from its own log where the expert weights went and whether
  we computed; a run that cannot aborts instead of being tabulated. Six
  "passes" in this project have tested nothing or the wrong thing.

Model throughout: DeepSeek-V4-Flash UD-Q8_K_XL, 150.75 GiB, 43 expert layers;
stub = its first 4 layers, 15.78 GiB. Host: 16-core Xeon, four Vega II dies.

## What owning the weights costs (the repack forfeit)

llama.cpp rewrites MXFP4 experts into `CPU_REPACK` layout at load and runs a
different GEMM; we compute on the bytes as they sit in the file. That gap,
measured (`gate.py --vs-stock`): **mean KLD 3.6e-5, max 2.1e-3, top-1
99.804%** — the yardstick for what two *correct* kernels disagreeing looks
like on this machine. In throughput it is the `stock -> ours-off` column:
**-4.4 to -4.5%** of decode on both stub and full model.

## Decode, CPU-only host (why the stub exists)

Full model, two loads per configuration, tg32:

| config | load 1 | load 2 | mean | load-to-load |
|---|---|---|---|---|
| `stock` | 3.34 ± 0.05 | 3.59 ± 0.20 | 3.46 | 7.2% |
| `ours-off` | 3.47 ± 0.03 | 3.18 ± 0.03 | 3.33 | 8.7% |
| `ours-on` | 3.12 ± 0.04 | 3.39 ± 0.10 | 3.25 | 8.3% |

Nothing resolved: sorted, the six measurements interleave and the two loads
rank the configurations differently. The same measurement on the stub (floor
0.0-2.3%) resolves the repack at **3.2%** and bounds our split boundary at
**≤216 µs per split** — which scales to **≤3%** on the real model's 43 splits,
the kind of bound the big model could never produce about itself.

## Prefill on the dies, ggml kernels (`dies`)

Full model, 36 of 43 layers mirrored (9 per die), `--pp 512`, two loads,
spreads ≤1.1% — prefill measures well on the real model, unlike decode:

| pp512 | stock | ours-off | ours-on |
|---|---|---|---|
| t/s | 18.49 | 18.02 | **21.84** |

**Our compute +21.1%, net vs stock +18.1%.** The enabling fact, stub, by
batch size (our compute vs same weights on CPU):

| batch | 8 | 16 | 32 | 128 | 512 | decode |
|---|---|---|---|---|---|---|
| unchunked | +21.9% | -60.2% | -54.6% | -51.4% | — | -0.1% |
| 8-token chunks | **+26.7%** | **+32.9%** | **+35.1%** | **+30.1%** | **+27.2%** | -0.2% |

The cliff is `ggml_vk_use_mul_mat_vec_id` (`ggml-vulkan.cpp:10607`): the fast
vector path ends at 8 tokens. Issuing the block 8 tokens at a time is a graph
decision, not a shader.

## Decode on the dies, ggml kernels (single read-back)

Stub, per-layer, steady-state medians from the per-split profiler
(`MOESERV_PROFILE`):

| µs per layer | 6 reads | 1 read |
|---|---|---|
| compute | 1018 | 1106 |
| read-back | **530** | **149** |
| total | 1578 | **1279** |
| CPU layer | 1420 | 1403 |

Six `ggml_backend_tensor_get` calls at ~88 µs each were the border; reading
the terminals' common view-root once turned the layer from 10% slower than
the CPU to 9.7% faster, and stub decode from flat to **+6.9%**. First Vulkan
call costs 306 ms (pipeline compile) — quote medians.

## Decode TP, custom kernel (`tp-integrate`)

Kernel-level numbers (probe, synthetic weights): see `KERNEL.md` — the 2-pass
mxfp4 kernel runs the block's matmuls at 95.4/86.5 µs vs ggml's 163.8, and
the full TP block costs **113 µs/die**.

Full model, four configurations, two loads each, spreads 0.3-1.3% (a lucky
day; see the discipline note):

| tg32 | stock | ours-off | ours-on | ours-tp |
|---|---|---|---|---|
| mean t/s | 3.64 | 3.48 | 3.46 | **3.92** |

**TP vs stock +7.6%, TP vs the ggml path +13.0%.** 34 of 43 layers resident
under the 28000 MB/die budget (816 MiB per layer per die; 43 would need
34.3 GiB against 32 GB of HBM), 9 on the CPU delegate — 5474/1449 splits,
exactly the capacity arithmetic.

Per resident layer, phase timers: **439 µs** = stage 4.5 + submit 125.1 +
wait-first 225.7 (containing ~113 µs of GPU) + wait-rest+sum 83.9 — within 2%
of the stub's number, so the border is per-call, not per-model. Against the
CPU delegate's **1.38 ms/layer**: the block is **3.1x faster where resident**,
and MoE falls from ~21% of decode time to ~11%. Bounds on this host: perfect
residency ≈ +3%, MoE free ≈ +12%.

Two border fixes worth ~2 ms/call together, found by the phase timers:
persistent mapping, and HOST_CACHED memory for anything the CPU reads
(`MECHANISM.md` #9).

## The border (`36be37a`)

Same-day interleaved A/Bs on the stub, three pairs each, engagement asserted
per load. **Threaded per-die submit: refuted** — +0.2% against 0.4% spread,
phase totals equal; `vkQueueSubmit` costs ~35 µs serialized in the driver
whichever thread issues it. **Fence polling: kept** — `vkGetFenceStatus`
spins instead of blocking waits, ~+0.5%, poll ahead in 6 of 6 pairs across
both pair orders (both first A/Bs' second-in-pair had come out ahead, so the
order flip was the discriminating run). The remaining ~340 µs of border is
structural: serialized submit floor plus launch-to-completion latency,
reachable only with fewer submissions per token.

**Amendment (2026-08-14)**: a standalone null-shader probe
(`../../vk-latency/`, raw Vulkan, no ggml, same queue-family and polling
choices as `moe_tp.h`) measured the machine floor for these shapes:
**9 µs per `vkQueueSubmit`** (identical for 0/1/4 dispatches per cb),
**~59 µs** submit→fence polled (~65 blocking — poll saves ~6 µs/fence,
independently confirming the polling verdict above), **87 µs** for a 4-die
submit-all/wait-all round against 243 µs strictly serial (round trips
overlap; submits serialize at ~9 µs each). Our ~35 µs submits and ~310 µs of
non-GPU wait are therefore 3-6x above floor. The obvious reading — border
priced by what the call *carries* — was then **refuted** by the same tool's
TP-shaped ladder (same day): rebuilding our cb ingredient by ingredient
(5-binding descriptor sets, the exact copy/barrier pattern, all 4 dispatches,
real-sized 816 MiB buffer references, 26.3 GiB/die ballast, 16 spinning host
threads) moved submit only 9 → 11 µs and the full-shape 4-die round to
212 µs total (stage 2.5 / submit-all 47 under load / wait+sum ~161). Line
items: copies +14 µs (honest DMA), ballast +12-20 µs wait on the referencing
cb only, everything else free. 212 + our ~113 µs GPU ≈ 325 against the
measured ~440: the residual ~100 µs and the ~3x per-submit gap are properties
of the moe-serv *process* (150 GiB mapped, ggml graph around the call, submit
thread = ggml compute thread), not of the submitted Vulkan work. Next probe
lives inside moe-serv: `MOESERV_PROFILE` while varying `-t`, and audit what
besides `vkQueueSubmit` sits inside the submit phase timer. Full tables:
`vk-latency/README.md`, "The TP-shaped ladder".

Same day, `VK_EXT_calibrated_timestamps` (40 ns tick, 0.04 µs median
cross-clock deviation) decomposed the round trip. Null dispatch, per die,
medians: submit 9.3 / **launch 34.5** (submit returned → GPU starts) / GPU
1.6 / **signal 21** (GPU done → polled fence reads signaled) ≈ 67 total.
TP-shaped cb: launch 28 / GPU 17.7 (copy-in 6.1, dispatches 8.5, copy-out
3.0) / signal 22.5 ≈ 79. Two consequences: (a) the border floor is launch +
signal, both driver-path costs that **overlap across dies** — the 87 µs
4-die round pays them roughly once; (b) fence signaling costs ~21 µs *after*
the result bytes are already readable in host-cached memory, so a sentinel
word written by the cb after the result copy (fence kept only for cb reuse)
would let the host read ~20 µs earlier per call — a named, unmeasured lever
for `moe_tp_compute`. Oddity recorded: bigger cbs launch *faster* (28 vs
34.5), so cb-start cost is not monotone in cb size. Tables:
`vk-latency/README.md`, "Calibrated decomposition".

**The submit gap, closed (2026-08-14, same day).** The in-process ~21-24 µs
per submit against the probe's ~9-11 is **cache eviction**: our calls come
~8 ms apart with the trunk streaming memory in between, so the driver's
submit path runs cold every call. Evidence, in order: (a) the submit phase
timer was split (`moe_tp.h` now reports `reset` separately) —
`vkResetFences` is ~2 µs/die, innocent; (b) `-t` 4/8/16 moves submit <15% —
thread count innocent; (c) vk-latency's cold-cache rung (64 MB streamed
before each iteration) reproduces it: submit 9.3→17.5-19, launch
34.5→55-60, signal 21→25-33, every host phase roughly doubled. With cold
figures the stub profile reconciles with no residual: predicted submit
4×18≈72 vs measured 82-94; wait-first ≈ 57 launch + ~113 GPU + ~29 signal
≈ 199 vs measured 190-210; wait-rest+sum ≈93 vs measured 59-93. Border
chapter closed: it is cache-eviction-priced, not fixed; surviving levers
are fewer submissions per token and the sentinel poll (which removes the
~25-30 µs cold signal). Gate after the timer split: bit-identical.

## The correctness ledger

| path | comparison | result |
|---|---|---|
| CPU delegate | byte-equality of `--kl-divergence-base` files | **bit-identical** (`--tol 0`) |
| ggml-vulkan mirror | mean KLD vs same-placement CPU control | 6.2e-5 – 8.4e-5 across commits |
| TP, real weights | mean KLD vs same-placement CPU control | **1.070e-4** (build-vk host) / **1.780e-4** (CPU host) |
| CPU delegate, GLM-5.2 stub | byte-equality, `glm-L5.gguf` | **bit-identical** (`--tol 0`) |
| ggml-vulkan mirror, GLM-5.2 stub | mean KLD vs same-placement CPU control | **6.1e-5**, top-1 99.02% |

The GLM mirror row needs `GGML_VK_FORCE_MAX_ALLOCATION_SIZE=3221225472` in the
environment: one q6_K expert tensor is 2520 MiB against the driver's
`maxMemoryAllocationSize` of exactly 2 GiB, so without the override
ggml-vulkan refuses the buffer and the mirror **silently falls back to the
CPU delegate — and the gate goes green anyway**, because the delegate is
correct. The tell is the direction of the result: a mirror run that comes
back *bit-identical* did not run on the dies. (Engagement lines that prove it
did: `uploaded N GiB`, `splits computed — N on device(s)`.) ds4 never hit the
limit — its MXFP4 tensors are 1088 MiB — which is why this first appeared on
the second architecture.

## Breadth: GLM-5.2 stub bench (2026-08-14)

`bench.py --model glm-L5.gguf --build-dir build-vk --pp 512 --n 32`, two
loads per config, the allocation override set, mirror engagement verified in
each treated log (`14.77 GiB uploaded`, `334 splits on device(s), 0 on CPU`,
`512 tokens in chunks of 8` — the ds4 chunking policy unchanged).

| comparison | pp512 | tg32 |
|---|---|---|
| ours-off → ours-on | **+6.8%** (168.3 → 179.6, resolved) | **+8.9%** (25.00 → 27.22, resolved) |
| net vs stock | +9.1%, NOT RESOLVED (== 9.1% noise) | **+9.6%** (24.84 → 27.22, resolved) |
| stock → ours-off | not resolved | not resolved |

The decode result is a sign flip against ds4, whose full-model mirror decode
was slightly negative (3.48 → 3.46): on GLM the mirror wins decode outright.
Candidate mechanisms, not separated: q6_K's per-weight-heavier CPU
`mul_mat_id` (q6_K gets no CPU_REPACK on x86, so the CPU side runs plain
kernels), 8+1 experts per token vs 6+1, and the stub's inflated block share
(2 of 5 layers are MoE vs 43 of 79 on the real model). Stub percentages do
not transfer to the full model; the sign is the finding. Instrument note:
this stub's load-to-load spread is 1-9% (ds4's is ~0.3%), and its first load
is contaminated by the cold 18 GiB disk read (within-run sd ±7.9 vs ±0.25 on
the second load) — trust second-load spreads, or add loads.

The tolerance typed on every Vulkan/TP command line is 5e-4 — ~14x the repack
gap. The TP number has two spellings because the *host build* is an axis of
the run configuration: each reproduces exactly on its own instrument, and the
TP logit files are byte-identical across every DLL variant that never touched
arithmetic (old / workers / poll). A KLD change without a config change is a
red flag (repo `CLAUDE.md`: corrupt-weights tell); check the instrument first.
