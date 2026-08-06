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

### Measured: Windows, AMD proprietary driver 21.30.44.22, one Vega II die

`--experts 32 --topk 8 --iters 30`, api 1.2.196, subgroup 64, maxAlloc 2.00 GiB:

| phase | p50 (us) |
|-------|---------:|
| null dispatch      |  189.8 |
| upload x (24 KiB)  |   76.8 |
| download 192 KiB   |  261.0 |
| record cmdbuf      |    9.7 |
| submit             |   13.4 |
| wait fence         |  755.4 |
| **layer total**    | **1109.9** |

Extrapolated to 78 MoE layers: **86.6 ms/token** of GPU-path work, against a
**14.8 ms/token** driver floor from one submit per layer.

What that says:

- Recording and submitting are nearly free (23 us combined). Command-buffer
  construction is not where the cost is on this driver.
- A null dispatch costs more than a 24 KiB upload round trip, so ~110 us of the
  190 us floor is kernel launch rather than queue round trip.
- The 192 KiB readback takes 261 us — about 0.7 GB/s, round-trip-bound rather
  than bandwidth-bound. This is the quantitative case for PLAN.md's partial-sum
  mode: returning one combined row instead of K per-slot rows.
- Overlap hides the fence wait but not the transfers, which sit outside the
  submit/wait window.

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
