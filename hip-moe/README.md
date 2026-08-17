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
