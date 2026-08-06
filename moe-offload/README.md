# moe-offload — what a Vulkan GPU charges for MoE expert offload

A cross-platform measurement, not a benchmark of kernels. It answers one
question that decides which OS the distributed work in
[../nano-glm/PLAN.md](../nano-glm/PLAN.md) should run on:

> For the dispatch pattern expert-parallel decode needs — per layer: upload one
> activation, run top-K experts, read K rows back, combine on the CPU — how much
> does the driver charge for the round trip, and does that differ between the
> AMD Vulkan driver on Windows and MoltenVK-on-Metal on macOS?

Same Mac Pro 7,1, same Vega II dies, same shapes. Any difference is the
software stack, which is the thing being chosen.

## What it emulates

One MoE layer of GLM-5.2 at decode (batch 1), split the way the plan splits it:

- **CPU**: router (top-K of E), shared expert, combine.
- **GPU**: the K selected routed experts — `up`, `gate`, swiglu, `down`.

Shapes default to the real model: `d_embd=6144`, `d_ffn=2048`, `top_k=8`.
Weights are int8 with a per-row scale — 8 bpw against the real Q6_K's 6.64, so
memory traffic is realistic, without nibble-unpack bugs to debug. The expert
kernel is one workgroup per output row with a shared-memory tree reduction:
vk-test measured that shape at ~230 GB/s on these dies versus ~4 GB/s
row-per-thread, so it is the right starting point, not a tuned kernel.

Every run cross-checks the GPU expert output against a CPU reference before
timing anything. That matters here specifically: this hardware has a documented
history of silent NaN under Metal (see the repo `CLAUDE.md`), and MoltenVK runs
Vulkan *on* Metal, so "it produced numbers" is not evidence of correctness.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Needs the Vulkan SDK (`VULKAN_SDK` set) and `glslc`. On macOS the SDK supplies
MoltenVK, the loader, and `glslc`; `brew install molten-vk vulkan-loader
vulkan-tools shaderc` also works, in which case set
`VK_ICD_FILENAMES=$(brew --prefix molten-vk)/share/vulkan/icd.d/MoltenVK_icd.json`.

On Windows, use `../nano-glm/build.ps1`'s toolchain discovery or a
`vcvars64` shell; the CMake is plain and cross-platform.

## Run

```bash
./build/moe-offload --list                       # devices, driver, limits
./build/moe-offload --experts 32 --iters 30      # the measurement
```

Options: `--device N`, `--experts N` (resident on the GPU), `--topk N`,
`--embd N`, `--ffn N`, `--layers N` (extrapolation only), `--iters N`,
`--threads N` (CPU shared expert), `--no-verify`.

## Reading the output

Each phase is timed separately, because "dispatch is slow" has different fixes
depending on where the time lands:

- **null dispatch** — submit + queue round trip with a trivial kernel. The
  floor no kernel work can go below. Multiply by layer count for the per-token
  tax of one submit per layer.
- **record / submit** — command-buffer construction and handing it to the
  driver. If MoltenVK is expensive at encode time, it shows here.
- **wait fence** — actual GPU execution.
- **upload / download** — PCIe round trips at the sizes the protocol moves
  (24 KiB up, `K x d_embd` floats back).
- **LAYER w/ cpu overlap** — the same layer with the CPU shared expert running
  between `submit()` and `waitFence()`, i.e. the `moe_send`/`moe_recv` trick in
  PLAN.md phase 3.

### Measured: both OSes, same Mac Pro 7,1, one Vega II die

`--experts 32 --topk 8 --iters 30`. Windows: AMD proprietary driver
21.30.44.22, api 1.2.196, subgroup 64, maxAlloc 2.00 GiB. macOS: MoltenVK,
api 1.2.357, subgroup **32**, maxAlloc **3.50 GiB**.

| phase (p50, us)    | Windows (AMD) | macOS (MoltenVK) |
|--------------------|--------------:|-----------------:|
| null dispatch      |         189.8 |            151.5 |
| upload x (24 KiB)  |          76.8 |             83.4 |
| download 192 KiB   |         261.0 |         **99.1** |
| record cmdbuf      |           9.7 |          **3.7** |
| submit             |          13.4 |             36.2 |
| wait fence         |     **755.4** |           3187.0 |
| **layer total**    |  **1109.9**   |           3482.7 |
| per token (x78)    |   86.6 ms     |         271.7 ms |

Both platforms pass the CPU cross-check with the *same* error
(`max|gpu-cpu| = 6.914e-06`, no non-finite values). **Vulkan-on-Metal computes
correctly on hardware where native Metal silently NaNs** — see the repo
`CLAUDE.md` AMD-Metal entry. That is a real result for PLAN.md phase 4.

What the split says:

- **Windows is 3.1x faster overall, and all of it is inside the fence** —
  actual GPU execution, 755 us vs 3187 us. Over the 302 MB each layer reads
  that is ~400 GB/s versus ~95 GB/s on identical silicon.
- macOS is *better* at the things that were expected to hurt it: readback
  2.6x faster, command-buffer encoding 2.6x faster, dispatch floor slightly
  lower. Only `submit` is worse (36 vs 13 us).
- Recording and submitting are nearly free on both (23 us / 40 us combined),
  so command-buffer construction is not the cost anywhere.
- A null dispatch costs more than a 24 KiB upload round trip on Windows, so
  ~110 us of the 190 us floor is kernel launch rather than queue round trip.
- The 192 KiB readback is round-trip-bound rather than bandwidth-bound on both
  (0.7 GB/s Windows, 1.9 GB/s macOS). This is the quantitative case for
  PLAN.md's partial-sum mode: return one combined row instead of K per-slot
  rows.
- Overlap hides the fence wait but not the transfers, which sit outside the
  submit/wait window.
- macOS allowing a **3.50 GiB** single allocation where Windows caps at 2.00
  inverts the stock-llama.cpp finding: a whole layer's `ffn_*_exps` (2.46 GiB
  at Q6_K) can be `-ot`-offloaded on macOS but not on Windows.

### Workgroup width

MoltenVK reports subgroup 32 for silicon the AMD driver calls 64, so the
kernel's reduction width was the obvious suspect for the fence gap. `--wg`
sweeps it. It is not the cause (wait fence p50, `--iters 20`, all verify OK):

| `--wg`     |    32 |    64 |   128 |   256 | spread |
|------------|------:|------:|------:|------:|-------:|
| Windows us | 873.5 | 753.8 | **717.4** | 874.4 |   22% |
| macOS us   | 3412.6 | **3182.4** | 3364.2 | 3210.1 |    7% |
| ratio      |  3.9x |  4.2x |  4.7x |  3.7x |        |

macOS is essentially flat — it barely responds to the knob — while Windows
leads by ~4x at every width. So the gap is not a tuning mistake in this
kernel's reduction; it is MoltenVK's MSL translation or the Metal compute path
itself. Windows also happens to be fastest at a width (128) that is neither
driver's reported subgroup size, so "match the subgroup size" is not a
reliable default either.

What this does not yet separate: whether Metal is slow at *streaming memory*
generally, or specifically at this kernel's constructs (byte unpack from
`uint[]`, threadgroup memory, barriers). A trivial read-and-sum kernel with no
reduction and no unpacking would tell them apart, and is the natural next
probe if anyone wants Metal to be fast rather than just wants to know it is
not.

## Which OS, then?

The GPU answer and the CPU answer point in opposite directions, so the choice
depends on whether GPUs matter for the phase being run:

| | macOS | Windows |
|---|---|---|
| CPU decode (llama-bench tg32 @32t) | **2.23 t/s** | 1.84 t/s |
| CPU timing stability | **+/- 0.01** | mmap swings 1.04-1.84 |
| GPU expert execution (this harness) | 3182 us | **755 us** |
| max single allocation | **3.50 GiB** | 2.00 GiB |
| Vulkan correctness | OK | OK |

**Phases 0-3 of PLAN.md are CPU and network only**, and there macOS is ~20%
faster with far more reproducible timings — which matters more than it sounds,
since those phases are judged by A/B comparisons of dispatch policies.
**Phase 4 is where Windows' 4x GPU lead would matter**, and it is explicitly
off the critical path.

Two things keep the GPU question secondary regardless of the 4x: expert
residency is the binding constraint (563 GiB of routed experts against 128 GiB
of VRAM per machine, so the GPU can only ever hold hot subsets), and a mixed
cluster is not an option — a Windows/MSVC and a macOS/clang build of the same
llama.cpp commit differ by 8.85e-3 mean KL, so the KL == 0 gates in phases 0-2
cannot pass across two different OSes.

## Caveats that limit what this can conclude

- **The CPU shared expert is a naive scalar int8 FFN** (~6.3 ms here) where a
  real AVX-512 Q6_K one would be well under a millisecond. So the reported
  "overlap saved %" understates what overlap is worth in a real engine — read
  the raw `wait fence` figure as the amount of GPU time that *can* be hidden,
  not the percentage.
- **Expert residency is the unsolved problem, not dispatch.** GLM-5.2 has
  563 GiB of routed experts against 128 GiB of VRAM per machine. This harness
  holds a small resident subset; a real deployment would have to stream weights
  or hold hot subsets, and streaming cost is not measured here.
- A single Vulkan allocation is capped at **2 GiB** on the AMD/Windows driver
  (`maxMemoryAllocationSize`), which is why weights are split into one buffer
  per projection rather than one per layer. A whole layer's 256 experts as a
  single tensor does not fit — the same limit that blocks `-ot` offload of
  `ffn_*_exps` in stock llama.cpp.
- One resident die only. Multi-GPU fan-out is not implemented.
