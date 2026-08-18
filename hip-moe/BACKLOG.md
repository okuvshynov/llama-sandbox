# Backlog — deferred experiments and follow-ups

Open items with enough context to pick up cold. Closed work lives in
README.md; this file holds what we chose not to do yet.

## Mixed CPU+GPU EP: the umbrella hypothesis

Add `--cpu-experts K` to moe-ep-bench: the CPU as a fifth EP target owning
K (cold) experts, its ggml graph launched concurrently with the GPU wave.
Prediction to falsify, from measured numbers (2026-08-18): a distinct cold
expert costs ~600-650 µs on CPU (35 MB at 59 GB/s), the n=4 GPU wave is a
~650 µs umbrella — so ONE distinct cold expert per step hides almost
completely, the second sticks out ~600 µs (+30% step). Expected tax for
"1-2 cold pairs/step" placement: 5-15% mean, with 1.3-1.65 ms jitter.
Design constraints already known: CPU graph on ~14-15 threads, blocking
GPU sync (+1 µs on HIP per latency bench), overlap mandatory. Also
verify: two tokens hitting the SAME cold expert read it once (~free).

## Architectural EP: kill the ~246 µs serial routing floor

The measured floor after --shared-late is x upload + routing compute +
sync + tiny reads. Two designs, both integration-scale:
- Replicated routing: every die holds the 11 MB router and computes top-k
  locally — zero round trip. Blocked on ggml having no on-device stream
  compaction for per-die pair selection (would need a custom kernel or a
  masked-compute trick that doesn't 4x the work).
- Router-fused per-die graphs (routing inside each die's graph, ids
  consumed on-device by mul_mat_id directly at full [6, n] shape with
  per-die expert masking).

## Upstream reports owed to llama.cpp

- `--spec-draft-n-max > 5` with the DSpark drafter does not clamp safely:
  the warning fires, then the first request dies with MUL_MAT failed /
  ROCm error (b10472, gfx906). Repro: chat.sh server + n_max 8.
- Compact-layout mul_mat_id cliff: ids [1, P] puts P in dst->ne[2] and
  falls off the MMVQ fast path past get_mmvq_mmid_max_batch — 9 pairs
  cost 3x what 8 do on gfx906/MXFP4. Maybe expected behavior, but the
  cliff shape is surprising; at minimum documentation-worthy.

## Serving-mode wait policy

The bench spins (ROCm blocking sync busy-waits by default); serving with
CPU expert layers wants blocking sync so the waiter doesn't compete with
the 16 expert threads. Measure decode t/s of the chat.sh config with
hipDeviceScheduleBlockingSync (env/patch) vs default — prediction: ~free
on HIP (poll-vs-block gap ~1 µs), frees a core.

## Housekeeping

- perf installed mid-session (linux-tools-generic, invoke the versioned
  binary directly — the T2 kernel has no matching package); sample the
  EP bench wait phase to visually confirm the ROCm spin loop.
- Windows-side SHA-256 cross-check of the DS-V4-Flash transfer against
  checksums/DS-V4-Flash-0731-UD-Q8_K_XL.sha256 (Linux-side manifest).
- Apple SSD (nvme0) health check from macOS someday — Linux no longer
  depends on it (boot migrated 2026-08-18), but macOS's ESP shares it.
- E6 occupancy experiment re-run under hipcc if the custom kernel ever
  matters again — the E1-E8 ledger was tuned under the Vulkan compiler
  (hipcc lands 35-36 VGPRs / 7 waves per SIMD vs Vulkan's 6).
