# Running a Mixture-of-Experts block on four Vega dies — a tutorial

This is a guided tour of `moe-serv` for someone with a CS or math degree and
little systems experience. By the end you should understand what this project
computes, on what hardware, through which API, and why each optimization was
tried — including the ones that failed, which are at least half the lesson.

It is written against one concrete system: one specific model
(DeepSeek-V4-Flash), one specific machine (a 2019 Mac Pro with four AMD Vega
II GPU dies, running Windows), one specific host program (llama.cpp).
Concreteness is the point — every claim here was measured on this system, and
the numbers are what make the abstract ideas stick.

## Reading order

1. [What a Mixture-of-Experts model is](01-moe.md) — and why the expert
   weights dominate everything: 137 of 150 GiB, and most of every
   token's memory traffic.
2. [The hardware](02-hardware.md) — the four GPU dies, and the vocabulary
   you need to reason about them: wave, CU, LDS, occupancy, HBM.
3. [The model and its number format](03-model.md) — DeepSeek-V4-Flash's
   shapes, and MXFP4, the 4.25-bits-per-weight format the experts are
   stored in.
4. [Vulkan, and how a program talks to a GPU](04-vulkan.md) — buffers,
   memory types, command buffers, queues, fences, and where the
   microseconds go.
5. [How the backend plugs into llama.cpp](05-backend.md) — a DLL that an
   *unmodified* llama.cpp loads, hands 137 GiB of weights to, and asks to
   compute exactly one thing.
6. [The optimization campaign](06-optimization.md) — everything tried, with
   verdicts: what worked, what didn't, and the measurement discipline that
   separated the two.

## Conventions

Code links point at the exact commit this tutorial was written against
(`9ef58c9`), so line numbers stay true even as the code moves on:

> [`src/moe_tp.h:711`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_tp.h#L711)

Numbers are quoted with their instrument. "On the stub" means the 4-layer,
15.78 GiB cut of the model that loads in seconds and measures decode to
±0.3%; "on the full model" means all 150.75 GiB. Why both exist is part of
the story ([06](06-optimization.md#measure-honestly)).

Deeper reference material, written for people already working on the project,
lives one directory up: [`docs/MECHANISM.md`](../docs/MECHANISM.md),
[`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md),
[`docs/KERNEL.md`](../docs/KERNEL.md), and [`PLAN.md`](../PLAN.md).

## Amendments — learned since the draft, not yet folded in

The chapters are a snapshot; the project keeps moving. Findings that
qualify or contradict something a chapter says are collected here until the
next revision pass folds them into the text. When editing a chapter, check
this list first and delete the entries the edit absorbs.

- **2026-08-14 — the border is mostly not a fixed machine cost.** A
  standalone null-shader probe ([`vk-latency/`](../../vk-latency/), no ggml)
  measured the machine floor: **9 µs** per `vkQueueSubmit` (same for 0/1/4
  dispatches), **~59 µs** submit→fence polled, **87 µs** for a 4-die
  submit-all/wait-all round. Our TP call pays ~35 µs/submit and ~310 µs of
  non-GPU wait — 3-6x above that floor. The same tool's TP-shaped ladder then
  rebuilt our call ingredient by ingredient and *acquitted the command
  buffer* (descriptors, copies, dispatches, 816 MiB references, 26 GiB
  residency, spinning threads — all nearly free): the residual gap turned
  out to be **cache eviction** — the trunk's streaming between calls runs
  the driver's submit path cold, doubling every host phase (calibrated
  split: submit 9 / launch 35 / signal 21 hot, ~2x each cold). The border
  is self-interference-priced, not fixed; on a host whose trunk runs on
  its own device the same code drops from ~440 to ~310 µs/layer.
  Affects: [06 "The border"](06-optimization.md#the-border)
  ("what remains ... is structural"), [04 "What a call actually looks
  like"](04-vulkan.md#what-a-call-actually-looks-like) ("the fixed cost of
  talking to a GPU at all", the ~35 µs submit figure), and the ~35 µs quoted
  in [02 "The CPU still matters"](02-hardware.md#the-cpu-still-matters).
  Cross-day comparison, so a lead, not a verdict; details in
  [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md) ("The border",
  amendment) and `vk-latency/README.md`.
