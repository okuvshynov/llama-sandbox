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
