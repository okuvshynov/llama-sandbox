# vk-latency

What does it cost to ask a GPU to do nothing? This tool submits a **null
compute shader** (one workgroup, empty `main`, no bindings) through
pre-recorded command buffers and times the host-visible pieces:

- **submit** — how long `vkQueueSubmit` takes to return
- **submit→fence** — from submit to the fence reading signaled (the launch
  round trip)

per die, per dispatch count (0 / 1 / 4 per command buffer), per wait mode
(polling `vkGetFenceStatus` vs blocking `vkWaitForFences`), and across
1 / 2 / 4 dies with the same submit-all-then-wait-all shape
[moe-serv](../moe-serv/) uses per layer.

## Why

moe-serv's TP decode path measured a ~440 µs per-layer call with only
~113 µs of GPU compute in it, and attributed the rest to the *border*:
~35 µs per submit (serialized in the driver), ~200 µs launch latency
(`moe-serv/docs/MEASUREMENTS.md`). Those numbers were measured inside a
llama.cpp backend with real buffers and real work. This tool measures the
same quantities with **nothing else present** — no ggml, no memory traffic,
no descriptors — so they become properties of the machine (driver + OS +
these dies), not of the backend. Any future "why is a call slow" question
starts by comparing against this floor.

Setup choices mirror `moe-serv/src/moe_tp.h` where they could matter:
discrete GPUs only, first compute-capable queue family, fences, poll =
`vkGetFenceStatus` + `YieldProcessor`.

## Build and run

```powershell
.\build.ps1        # -> build\bin\vk-latency.exe (+ shaders\null.spv)
.\build\bin\vk-latency.exe
```

Requires the Vulkan SDK (`glslc` compiles the shader at build time) and
Visual Studio; no other dependencies.

## Reading the output

Each row is 500 iterations after 50 warmup, reported as median [min .. p90].
Things worth comparing:

- **0 vs 1 dispatches**: the cost of the dispatch itself vs the cost of a
  submission carrying nothing at all.
- **1 vs 4 dispatches**: whether commands inside one submission are ~free
  (they should be — moe-serv records 4 dispatches per die per call).
- **poll vs block**: the kernel sleep/wake cost that made moe-serv keep
  fence polling.
- **submit-all vs serial total**: how much the four dies' round trips
  overlap when submitted back to back vs strictly one at a time.

## Baseline on this machine (2026-08-14)

All four dies behave identically; representative numbers (median):

| quantity | value |
|---|---|
| `vkQueueSubmit`, null cb | **~9 µs** per die (0, 1 or 4 dispatches — identical) |
| submit → fence signaled, poll | **~59 µs** |
| submit → fence signaled, blocking | ~65 µs (poll saves ~6 µs per fence) |
| 4-die round, submit-all then wait-all | **submit 37 µs, total 87 µs** |
| 4-die strictly serial | 243 µs (perfectly additive, ~60 µs per die) |

Readings: dispatches inside a submission are free at null scale; submits
serialize (~9 µs each even back to back) but the launch round trips overlap
almost completely (87 µs for four vs 243 serial); polling beats blocking by
a consistent ~6 µs per fence, matching moe-serv's decision to poll.

The lead this opens: moe-serv's TP call pays **~35 µs per submit and
~310 µs of non-GPU wait** against this **9 µs / ~50 µs** null floor — so
most of what moe-serv's border costs is *not* a fixed machine property of
submitting and fencing. It scales with something the null case lacks:
command-buffer content (buffer copies, barriers, descriptors), buffer
residency, or queue state. Measured on different days, so treat the gap as
a lead, not a conclusion — the next step is growing this tool's command
buffer toward moe-serv's shape (add copies, add a real buffer, add
barriers) until the gap appears, one ingredient at a time.

## The TP-shaped ladder (2026-08-14)

That decomposition is now in the tool. After the null scenarios it grows
the command buffer toward moe-serv's TP call
([`moe_tp.h` `setup_layer`](../moe-serv/src/moe_tp.h) is the template;
copies, barrier masks, binding tables and buffer sizes are mirrored
exactly, only the dispatch grids are cut to 1×1×1), one ingredient per
rung, and adds the four-phase round (stage / submit-all / wait+sum) that
`moe_tp_compute` runs per layer. Per-die results, poll wait, medians:

| rung | adds | submit | submit→fence | verdict |
|---|---|---|---|---|
| `+desc` | 5-binding descriptor set, push constants | 9.2 | 49.5 | descriptors free |
| `+copies` | 3 copy-ins, barriers, 96 KiB copy-out | 9.3 | ~63 | copies +~14 µs (GPU DMA, honest) |
| `+4disp` | all 4 dispatches, 4 sets, barriers between | 9.2 | ~63 | extra dispatches free |
| `+bigref` | plane/scale bindings → 512/32/256/16 MiB buffers | 9.2 | ~63 | referencing 816 MiB free |
| + ballast | +33 × 816 MiB alive per die (26.3 GiB) | 9.3 | **~76** | +12 µs, only on the big-ref cb (null unchanged) |

Readback sum of 96 KiB from HOST_CACHED: 25.3 µs, dead stable. Four-die
full-shape round: stage 2.5 / submit-all 37 / wait+sum 139 (161 with
ballast) / **total 179 (199 with ballast)** — against moe-serv's measured
4.5 / ~125 / ~310 / ~440 per layer (~113 µs of which is real GPU work the
ladder deliberately lacks).

A final rung mimics llama.cpp's threadpool: **16 host threads
busy-spinning** during the measured region (cumulative with ballast).
Effect: submit ~9 → ~11 µs/die (submit-all 37 → 47), wait +~2, round
total 204 → 212.

So the ladder **acquits nearly everything**:

- command-buffer content (descriptors, 4 dispatches, barriers): free
- copies: +~14 µs, which is honest GPU DMA time
- referencing 816 MiB: free; 26 GiB resident: +12-20 µs wait, on the
  referencing cb only (the null cb is unmoved)
- 16 spinning host threads: +1-2 µs submit, +10 µs on a 4-die
  submit burst

Stacking every ingredient: round total ~212 µs; add the ~113 µs of real
GPU compute the ladder deliberately lacks → ~325 µs. moe-serv pays ~440,
and its four submits alone cost ~125 µs where ours cost 47 under the
same-shaped load. The residual ~90-115 µs — and specifically the ~3x
per-submit gap — is not reproduced by cb shape, referenced size,
residency, or raw CPU contention. What this tool cannot mimic from
outside: the moe-serv *process* itself (150 GiB of mapped model memory,
the ggml graph executing around each call, the submit thread being a ggml
compute thread). Next probe accordingly lives inside moe-serv, not here —
e.g. `MOESERV_PROFILE` while varying `-t`, and checking what besides
`vkQueueSubmit` sits inside its submit phase timer.

## Calibrated decomposition (2026-08-14)

`VK_EXT_calibrated_timestamps` (present on this driver; 40 ns tick) reads
a (GPU tick, QPC) pair, so GPU timestamps land on the host clock and the
opaque submit→fence wait splits into named pieces. Calibration
max-deviation: **0.04 µs median** — negligible against everything below.
Per die, medians:

| phase | null 1-dispatch | TP +bigref |
|---|---|---|
| submit (host, `vkQueueSubmit`) | 9.3 | ~9 |
| **launch** (submit returned → GPU starts the cb) | **34.5** | **28** |
| GPU execution | 1.6 | 17.7 (copy-in 6.1 / dispatches 8.5 / copy-out 3.0) |
| **signal** (GPU done → polled fence reads signaled) | **21** | **22.5** |
| total | ~67 | ~79 |

Readings:

- The mystery of the ~59 µs null round trip is solved: it is **35 µs of
  launch latency plus 21 µs of fence-signal latency** around ~2 µs of
  GPU time. Both are driver/queue path costs, not GPU work; they are the
  structural floor on this driver, and they overlap across dies (the
  87 µs 4-die round = one launch + one signal, roughly, not four).
- **Fence signaling costs ~21 µs even when the CPU is spinning on
  `vkGetFenceStatus`** — the copy-back data has long since arrived in
  host-cached memory by then. That names a concrete lever for moe-serv:
  don't wait on the fence to *read the result* — have the cb write a
  sentinel word into the io buffer after the result copy and poll that
  (or poll a `VkEvent` set after the copy), keeping the fence only for
  cb-reuse bookkeeping. Potential ~20 µs per layer call.
- The bigger cb *launches faster* than the null one (28 vs 34.5 µs),
  echoing the `+desc`-faster-than-null oddity above; whatever the driver
  does at cb start is not monotone in cb size.

## The cold-cache rung (2026-08-14) — the submit gap, solved

The last unexplained item was moe-serv paying ~21-24 µs per submit
in-process against ~9-11 µs here under every mimicked condition. The one
condition no prior rung mimicked: moe-serv's calls arrive ~8 ms apart with
the trunk's memory-streaming compute in between, so the driver's
submit path runs **cold-cached** every call, while this tool submits every
~70 µs with everything hot. The rung streams 64 MB through the cache
before each iteration and repeats the calibrated measurement. Effect
(medians, per die):

| phase | hot | cold |
|---|---|---|
| submit | 9.3 | **17.5-19** |
| launch | 34.5 | **~55-60** |
| gpu (TP shape) | 17.7 | ~27-31 |
| signal | 21 | **25-33** |
| total (null / TP) | 67 / 79 | ~107 / ~131 |

Cold caches roughly double every host-side phase. And with these numbers
moe-serv's stub phase profile **reconciles completely** (measured the same
day with a reset/submit timer split; `vkResetFences` itself is ~2 µs/die,
innocent, and `-t` 4/8/16 moves submit by <15%): predicted submit
4×18 ≈ 72 vs measured 82-94; wait-first ≈ launch 57 + GPU ~113 + signal
~29 ≈ 199 vs measured 190-210; wait-rest+sum ≈ 93 vs measured 59-93.
There is no residual mystery in the border.

Consequence: the border is not a fixed driver property — it is
**cache-eviction-priced**, and any host whose CPU does real work between
calls pays roughly double the hot-loop figures. Levers that survive this
finding: fewer submissions per token (unchanged), and the sentinel-poll
idea (the ~25-30 µs cold *signal* is what it removes). Warming the driver
path between calls is not a lever — the trunk needs those caches.
