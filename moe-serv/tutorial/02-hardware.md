# 2. The hardware

## The machine

A 2019 Mac Pro (7,1) running Windows:

- **CPU**: Intel Xeon W-3245, 16 cores. Six DDR4-2933 memory channels,
  12×64 GB = 768 GB of RAM — enough to hold the whole 150 GiB model in
  ordinary memory. Theoretical bandwidth ~140 GB/s; the model actually
  achieves ~75 GB/s during decode (dequantization and thread synchronization
  eat the rest — a measured fact, not a guess).
- **GPUs**: two AMD Radeon Pro **Vega II Duo** cards. Each card carries two
  independent GPU **dies**; the machine therefore has **four dies**, each
  with its own 32 GB of HBM2 memory. The dies do not share memory — for
  every purpose in this project they are four separate GPUs.

The headline asymmetry: each die's HBM2 delivers on the order of **1 TB/s**,
versus the CPU's ~75 GB/s. For a memory-bound workload
([01](01-moe.md#why-decode-is-a-memory-problem-not-a-compute-problem)),
that is the entire motivation for moving the expert block onto the dies —
*if* the weights fit (4×32 GB = 128 GB against 137 GiB of experts: almost,
not quite, and that "not quite" shapes a lot of decisions later).

## GPU vocabulary (GCN / Vega dialect)

GPUs run thousands of threads, but not independently — threads are grouped,
and the groups are what the hardware actually schedules. AMD's GCN
architecture (which Vega is) uses these terms:

| term | meaning |
|---|---|
| **lane** | one "thread" as seen by your shader code |
| **wave** (wavefront) | 64 lanes that execute in lockstep — one instruction pointer for all 64. The unit of scheduling. (NVIDIA calls this a *warp*, which is 32.) |
| **SIMD** | the 16-lane-wide ALU that executes a wave; one 64-lane wave takes 4 cycles per instruction |
| **CU** (compute unit) | 4 SIMDs plus shared resources; each Vega II die has 64 CUs |
| **LDS** (local data share) | 64 KB of fast scratch memory per CU, shared by the workgroup — what CUDA calls *shared memory* |
| **workgroup** | the software-side grouping of lanes (you pick the size) that shares LDS and can synchronize with `barrier()` |
| **occupancy** | how many waves each SIMD has resident at once; more waves means memory stalls in one can be hidden by running another |
| **HBM2** | the stacked DRAM on the GPU package; high bandwidth (~1 TB/s), still hundreds of cycles of latency |

Two facts about this hardware that the kernel work leaned on
([06](06-optimization.md)):

- **LDS reads broadcast**: if all 64 lanes of a wave read the *same word* of
  LDS, that is one access, not 64. Reading *different* words in the same
  bank conflicts and serializes. This decides how big a lookup table can be.
- **Occupancy saturates early here**: the custom kernel measured fastest at
  ~6 waves per SIMD; more gave nothing. Once memory latency is covered,
  extra waves just shrink each wave's register budget.

## What "instruction-bound" means, and why it mattered

A memory-bound kernel should run at the memory's speed. Measured on one die
with plain float32 weights, the expert multiply reaches **~701 GB/s** — near
the practical HBM ceiling, confirmed independently by an older Metal project
on the same dies (720-791 GB/s). So the hardware can stream.

But with *quantized* weights (4-8 bits), the stock kernels ran far below the
bytes/second they should have: they spent **16-23 ALU cycles per weight** on
unpacking arithmetic, where the streaming itself only needs ~2-4. The
bottleneck was instructions, not memory — **instruction-bound**. That
diagnosis (made with a 40 GB synthetic probe, minutes of work) is what
justified writing a custom kernel at all, and the custom kernel's whole
design is about spending fewer instructions per weight
([06](06-optimization.md#the-custom-kernel)).

## The CPU still matters

Every byte the GPU produces must be read back by the CPU, and every command
the CPU issues costs host-side microseconds (submitting one command buffer:
~35 µs, serialized in the driver). At one-token decode the GPU's actual
compute is ~113 µs per layer — so host-side "border" costs of a few hundred
microseconds can eat the entire GPU win. Sections [04](04-vulkan.md) and
[06](06-optimization.md#the-border) are largely about this fight.
