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

## ROCm 5.7.1 vs 6.3.4, same-day interleaved A/B (2026-08-17)

6.3.4 installed side-by-side (llama.cpp requires >= 6.1; 6.3 is the last
release whose stock rocBLAS ships gfx906 Tensile kernels — 156 files
confirmed present). Both binaries from the same source, each resolving its
own runtime via soname (libamdhip64.so.5 vs .so.6). Three pairs per row,
alternating; all 30 runs pass the correctness gate.

| row | ROCm 5.7.1 | ROCm 6.3.4 | verdict |
|---|---|---|---|
| gate/up matmul | 105.2-106.1 µs | 106.6-107.0 µs | +1.3%, resolved (LLVM 18 lands 36 vgprs vs 35) |
| down matmul | 81.3-81.7 | 81.2-81.4 | tie |
| TP block GPU /die | 118.7-118.8 | 118.4-118.6 | tie |
| null round trip, polled | 8.7 | 10.1-11.4 | **+2 µs, resolved** |
| 4-die null round | 40.6-42.6 | 46.2-46.9 | **+5 µs, resolved** |
| tp4 layer wall | 263-266 | **296-300** | **+13%, resolved** |

Compute transfers unchanged; the 6.3 runtime's dispatch path costs ~2 µs
more per round trip and ~33 µs more on the full 4-die TP layer (16 launches
+ 12 async copies per rep, so a per-call overhead of this size compounds).
Still 4-6x below RADV Vulkan on every row. Consequence: build llama.cpp
against 6.3.4, but keep 5.7.1 installed — it is the cheaper runtime for the
custom backend, and the A/B costs one extra `HIPCC=` build.

## Stock llama.cpp on ROCm 6.3.4 (2026-08-17)

Built master `60eeeb608` (2026-08-17) with the HIP backend against the
side-by-side 6.3.4 — the version gate (>= 6.1) and the hipBLAS 2.0 API rule
out 5.7. The recipe that keeps the alternatives symlink and the 5.7 install
out of the build:

    ROCM_PATH=/opt/rocm-6.3.4 HIP_PATH=/opt/rocm-6.3.4 \
    HIPCXX=/opt/rocm-6.3.4/lib/llvm/bin/clang++ \
    cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm-6.3.4 \
          -DLLAMA_CURL=OFF
    cmake --build build-hip -j16

Validation, no model needed:

- All four dies enumerate: `4x gfx906, 32752 MiB each (131 GB), wave 64`.
- `test-backend-ops test -b ROCm0`: **12,926/12,926 passed**. The
  "not supported" lines (f16 ABS/SGN/NEG variants etc.) are declined ops
  that fall back to CPU — normal for any backend, not gfx906 rot.
- `SOLVE_TRI` passes on every shape — the one op with an open gfx906 issue
  upstream (rocBLAS strsm fallback); this master carries the custom-kernel
  path, so the known landmine is already defused.

Carried-over discipline for when the models arrive: a GPU-enabled build is
NOT a CPU baseline (bit-exact gates keep using CPU-only builds, as
logit-kld forces), and watch `sched_reserve: graph splits` whenever an op
falls back — backend-specific-op fallbacks are what shattered prefill on
the Windows Vulkan build.

## The model arrives: full-model baselines (2026-08-17)

Model: DS-V4-Flash-0731-UD-Q8_K_XL, 150.75 GiB, verified sizes + Linux
SHA-256 manifest (`checksums/DS-V4-Flash-0731-UD-Q8_K_XL.sha256`; Windows
cross-check pending). Stub regenerated with `moe-serv/make_stub.py` —
15.78 GiB, byte count matching the Windows instrument, layout check ok.
All rows: stock llama.cpp `build-hip`, `-lm none -t 16 -r 3..5`, two loads
per quoted config. Smokes first: stub loads across all four dies
(engagement from the memory breakdown), CPU and GPU greedy decodes agree
token-for-token over 24 tokens (weak, sampler-level — the KLD gate is still
owed), and the DS4-specific ops (HC_COMB, LIGHTNING_INDEXER) exist under
HIP because the backend compiles the CUDA sources — the ops whose absence
crippled the Vulkan backend.

| tg32, full model | t/s |
|---|---|
| Windows stock (CPU) | 3.46-3.64, 7-9% load-to-load |
| Windows best ever (moe-serv TP) | 3.92 |
| Linux stock CPU (`-ngl 0 -nopo 1`) | 3.50 / 3.59 |
| Linux `-ngl 99 -ncmoe 14 -ts 19/8/8/8` | 9.78 / 9.88 |
| Linux `-ngl 99 -ncmoe 13 -ts 19/8/8/8` | **10.12 / 10.00** |

pp512: 18.25 stock -> **76.4-76.7** at ncmoe 13 (Windows mirror best: 21.84).
Stub decode: CPU 30.2-30.5 (Windows band reproduced), full offload **89 t/s**.
`-ncmoe 12` OOMs under every viable `-ts` (9 heavy layers on one die is
31+ GiB before compute buffers) — ncmoe 13 is the capacity frontier of the
simple layer split. The `-ts` weighting exists because `-ncmoe` strips
experts from the *head* layers: die 0 absorbs all 13 light layers plus six
heavy ones (~24.5 GiB), dies 1-3 take eight heavy layers each (~28 GiB).

What the 10 t/s is and is not: stock llama.cpp, layer-split pipeline
(`-sm layer`) — no EP, no TP, three of four dies idle at any decode
instant, ggml's own HIP mul_mat_id for the on-die experts, CPU experts for
the first 13 layers. The 2.6x over the Windows-best is bought entirely by
capacity: experts in HBM instead of streaming through 75 GB/s DDR4, plus a
trunk that could never leave the CPU under Vulkan. Levers not yet pulled:
the 13 CPU expert layers (the dominant remaining cost), the idle pipeline,
and the custom kernel (2.5x over ggml's *Vulkan* mxfp4 kernel; whether
ggml's HIP kernel leaves the same gap is one `test-backend-ops perf -o
MUL_MAT_ID` away).

## The KLD gate on ncmoe 13 (2026-08-17)

`llama-perplexity -f gate_corpus.txt -c 512`, CPU base (`-ngl 0
--no-op-offload`) vs the serving config (`-ngl 99 -ncmoe 13 -ts 19/8/8/8`):
**mean KLD 8.0e-3 ± 0.7e-3, top-1 96.3 ± 0.8%**, PPL 6.18 vs 6.23 at
99.78% log-correlation.

The MoE-only yardsticks (repack 3.6e-5, mirror 6-8e-5, TP 1-1.8e-4) do not
apply: those varied only the expert matmuls, while this swaps the entire
stack's arithmetic. The recorded precedent for a full-stack swap is the
Apple-clang vs MSVC gap on the same CPU — 8.85e-3 mean KL, 96.67% top-1 —
and this result has the same magnitude and shape (median 4.4e-3, smooth
tail, no position-dependent spikes; the corrupt-weights tell is absent).
Verdict: two correct implementations disagreeing through 43 layers of
rounding. Config cleared.

Boundary of the claim, per the repo's own lesson: end-to-end KL at this
depth saturates and cannot certify kernel exactness — per-op correctness
rests on the 12,926-test backend suite and the token-identical greedy
smoke; this gate rules out the gross failure modes (wrong op, bad weights,
broken placement).
