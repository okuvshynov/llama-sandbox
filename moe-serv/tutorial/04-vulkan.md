# 4. Vulkan, and how a program talks to a GPU

## What Vulkan is

Vulkan is a low-level, cross-vendor API for GPUs. Where CUDA gives you a
`kernel<<<grid, block>>>(args)` call that hides everything, Vulkan makes
every step explicit: you enumerate devices, allocate memory, describe
bindings, record command buffers, submit them to queues, and synchronize by
hand. The payoff is control and portability (it is the only modern compute
API these 2019 AMD cards speak on Windows); the price is that nothing
happens implicitly, so *you* see — and pay — every cost.

The objects, in the order you create them:

| object | what it is |
|---|---|
| **instance** | your connection to the Vulkan loader; entry point to everything |
| **physical device** | one GPU as enumerated — each of our four dies is one |
| **device** | your opened handle to a physical device |
| **queue** | where you submit work for one device |
| **buffer** + **device memory** | a linear range of GPU-visible bytes; you allocate memory and bind a buffer to it |
| **shader module** | compiled GPU code (SPIR-V bytecode, compiled from GLSL by `glslc` at build time) |
| **pipeline** | a shader plus its fixed configuration, ready to run |
| **descriptor set** | the table that tells a pipeline which buffers its bindings point at |
| **command buffer** | a recorded list of GPU commands (copies, dispatches, barriers) |
| **fence** | a one-shot flag the GPU sets when a submission finishes; the CPU waits on it |

A **dispatch** launches a compute shader over a 3-D grid of workgroups
([02](02-hardware.md#gpu-vocabulary-gcn--vega-dialect)). One expert-block
call in this project is 4 dispatches per die, recorded once and replayed
every token.

## Memory types: the single most expensive lesson here

GPU-visible memory comes in flavors, and picking wrong is invisible until
measured:

- **DEVICE_LOCAL** — the die's own HBM. Fast for the GPU; the CPU cannot
  see it (on these cards). Weights live here.
- **HOST_VISIBLE | HOST_COHERENT** — CPU-mappable memory that on AMD is
  **write-combined**: CPU *writes* stream out fine, but CPU **reads run at
  ~150 MB/s** — roughly hard-disk speed.
- **HOST_VISIBLE | HOST_CACHED** — CPU-mappable *and* CPU-cached: reads at
  normal RAM speed.

The project once staged its GPU results through a write-combined buffer and
read them back on the CPU: 98 KB of reads cost **2 ms per call** and turned
the whole GPU path into a 13.6% *loss*. Switching that one buffer to
HOST_CACHED cut the cost 20x and flipped the benchmark to a win
([`src/moe_tp.h:525`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_tp.h#L525)).
Rule: any buffer the CPU will *read* must be HOST_CACHED.

Two related habits: map the staging buffer **once** at startup and keep the
pointer (`vkMapMemory` per call cost ~2 ms/layer on this driver), and copy
host↔device through explicit transfer commands inside the command buffer.

## What a call actually looks like

Per token, per resident layer, the backend does
([`src/moe_tp.h:711`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_tp.h#L711)):

1. **Stage** (~5 µs): memcpy the input vector, expert ids and router weights
   into each die's persistently-mapped buffer.
2. **Submit** (~140 µs for all four dies): `vkQueueSubmit` the pre-recorded
   command buffer on each die. ~35 µs each, and — measured, surprising —
   **serialized inside the driver**: submitting from four threads
   concurrently is no faster than one thread doing all four.
3. **Wait** (~220 µs, containing ~113 µs of actual GPU work): the GPU runs
   copies → 4 dispatches → copy-back; the CPU polls each die's fence.
   Polling (`vkGetFenceStatus` in a loop) beats blocking
   (`vkWaitForFences`) by skipping a kernel-level sleep/wake per fence
   ([`src/moe_tp.h:748`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_tp.h#L748)).
4. **Sum** (~40 µs): read each die's partial result (from HOST_CACHED
   memory!) and add the four vectors on the CPU.

Total ≈ 440 µs, of which only ~113 µs is GPU compute. The rest — staging,
submission, launch latency, readback — is the **border**: the fixed cost of
talking to a GPU at all. At batch size 1 the border dominates, and a large
part of the optimization story ([06](06-optimization.md#the-border)) is
about it. One more fixed cost worth knowing: the *first* call compiles
pipelines and costs ~306 ms — benchmark with medians or a warm-up.

## Coexistence

llama.cpp has its own Vulkan backend, and our DLL creates its **own separate
instance** for the TP path. Two Vulkan instances in one process is fine —
they only meet at the driver. This lets the host stay a plain CPU build
while the DLL still drives all four dies.
