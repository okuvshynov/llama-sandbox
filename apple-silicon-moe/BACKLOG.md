# Backlog — deferred experiments and follow-ups

Open items with enough context to pick up cold. Closed work lives in
README.md.

## The n=1 CPU bimodality

CPU-repack at n=1: the per-rep floor is rock-stable (1184–1219 µs across 5
loads, 2.9% spread — ~207 GB/s, i.e. the same saturation as n≥2) but the
mean floats 1.6–1.9× above it, and NO prio/poll combination moves the gap
(prio 2/3 × poll 0/100 all measured, `results/` has the runs). n≥2 means sit
within 15% of their floors. So single-token graphs specifically alternate
between full-speed and ~half-speed reps. Suspects not yet separated: DVFS
idle transitions between short graphs, P-cluster migration, memory-controller
power states. Instrument: `--only 1` with a per-rep histogram (bimodal split
would name the duty cycle), powermetrics alongside.

## Contention n-sweep and the serving arithmetic

The 568 GB/s combined figure is n=4 vs n=4 with both sides running flat out.
Serving reality is one token wave: the GPU runs its layers, the CPU runs its
displaced layers, alternating — the tax profile vs (n_gpu, n_cpu) and with
idle gaps is what the ncmoe-style placement math actually needs. Also
unmeasured: whether the ~14% Metal tax persists when the CPU side only
computes intermittently (suspect: it shrinks — the taxes looked
scheduling-shaped, not bandwidth-shaped, since combined 568 is well under
both ceilings' sum ~950).

## Metal overhead decomposition

Metal MoE reaches only 66% of the GPU streaming ceiling at n=8 (both CPUs
saturate theirs). ~1/3 of GPU time is not memory movement. Decompose: null
graph of the same node count (launch floor ×20 nodes ≈ 25 µs — small),
router-only graph, experts-only graph; compare `mul_mat_id` alone vs the
full block. If the gap is kernel inefficiency rather than graph overhead,
a batched-expert layout question follows (Metal has no MMVQ-style cliff
documented — find where batch amortization actually comes from, since GB/s
climbs 303→473 across n=1→8).

## Scheduler traps: upstream relevance

Two macOS behaviors measured here that llama.cpp serving with `--prio` may
also hit (worth checking llama-server's threadpool lifecycle before
reporting): (a) poll=100 FIFO workers spinning while idle → sticky 3.4×
demotion of later compute (create pools after model load); (b) FIFO workers
+ default-priority chief → +663% under a concurrent GPU submitter (the
chief needs the boost too). Both reproduce in `moe-contention-bench` by
moving one line.

## Housekeeping

- moe-contention-bench asserts at exit (`ggml-metal-device.m:657`,
  residency-set count) — teardown order, results unaffected. Free buffers
  before exit or ignore.
- The Metal-side weight fill is single-threaded RNG (~1–2 min per 13.5 GB
  set); a parallel fill was tried, is NOT the earlier slowdown suspect it
  briefly looked like, but was reverted for parity — safe to restore if
  load time matters.
- hip-moe cross-reference: Vega single-die ggml numbers for this exact
  layer live in `hip-moe/README.md` §moe-ggml-bench; EP wave arithmetic in
  §moe-ep-bench.
