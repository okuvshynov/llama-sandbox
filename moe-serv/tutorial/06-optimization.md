# 6. The optimization campaign — what worked, what didn't

This section is the payoff: every strategy tried, with its verdict and its
mechanism. The failures are kept deliberately — almost every *guessed*
mechanism in this project turned out wrong, and the instrument that told the
truth was always a small, cheap, targeted measurement. Full experiment
ledger: [`docs/KERNEL.md`](../docs/KERNEL.md); every number with its noise
floor: [`docs/MEASUREMENTS.md`](../docs/MEASUREMENTS.md).

## Prefill: one graph decision, +21%

Running the block on the dies with ggml's stock kernels initially made
prefill 51% *slower*. The cause was a cliff in ggml's Vulkan backend: up to
8 tokens, `mul_mat_id` takes a fast "vector" path; past 8, a general path
that loses half its speed. Feeding the block **8 tokens at a time** keeps it
on the fast path always: **+21% prefill** end-to-end. No shader was written —
the win was noticing which code path the sizes select.

| verdict | strategy |
|---|---|
| ✅ +21% prefill | chunk the block to 8-token pieces |
| ❌ | offloading without chunking (-51%) |

## Decode, round 1: the border shows its teeth

Same setup, decode: the die computed the block **1.4x faster** than the CPU
and still *lost* end-to-end. The profiler showed why: results were read back
through six separate 88 µs calls — 530 µs of readback against ~1000 µs of
compute. The six outputs were views tiling one tensor, so reading the parent
**once** cut readback to 149 µs: decode went from flat to **+6.9%** (stub).

The pattern — *fast arithmetic eaten by a border cost* — repeats three more
times below. It is the single most transferable lesson here.

## The custom kernel

Stock quantized kernels were instruction-bound: 16-23 ALU cycles per weight
against the ~2-4 that streaming needs
([02](02-hardware.md#what-instruction-bound-means-and-why-it-mattered)).
The replacement
([`shaders/mxfp4_pass1.comp`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/shaders/mxfp4_pass1.comp))
is a two-pass matrix-vector multiply: pass 1 has each workgroup compute
partial dot products over a K-tile, pass 2 reduces the partials. Design
choices that survived measurement, and the experiments that didn't (E-numbers
from the ledger):

| verdict | idea | mechanism |
|---|---|---|
| ✅ 1.5-1.9x vs ggml | 2-pass K-tiled structure | borrowed from a prior Metal project on the same dies; single-pass one-workgroup-per-row was 10x slower there |
| ✅ | repack weights at load into two planes (nibbles / scales) | contiguous streams; no per-block byte fiddling in the inner loop |
| ✅ | 16-float dequant LUT in LDS | a whole wave reading one entry is a broadcast — free ([02](02-hardware.md)) |
| ✅ E6 | 4 columns per lane, 6 waves/SIMD | occupancy saturates at 6; more waves gave nothing |
| ❌ E5 | replace the LUT with bit arithmetic | 2x *worse* — the ALU was the scarce resource being spent |
| ❌ E8 | accumulate in packed f16 pairs | slower AND wrong (6.6e-2 error): halved precision, no speed |
| ❌ | a 256-entry LUT (two nibbles at once) | LDS bank conflicts — lanes now read *different* words |

Result: the block's matmuls at 95.4/86.5 µs vs ggml's 163.8 on one die.
Meta-lesson: every ❌ above was a plausible idea; each cost under an hour to
refute *by measurement on the real shapes*. `test-backend-ops`-style generic
shapes predicted nothing (they suggested 20x gaps where our shapes show
1.3-1.5x).

## Tensor-parallel across the dies

With the kernel fast, one die per layer leaves three idle. Two ways to use
four dies:

- **Expert-parallel**: each die holds ¼ of the *experts*. Simple, but a
  token's 6 experts land unevenly (3-2-1-0 is common), so the slowest die
  sets the pace.
- **Tensor-parallel (TP), chosen**: every die holds ¼ of *every expert* —
  a column slice of gate+up, and the matching rows of down. Perfectly
  balanced by construction, and because the split is along the summation
  dimension of *down*, each die produces a partial output vector and the
  CPU just adds four vectors — no cross-die communication at all.

Fusing gate and up into one matmul (they read the same input) is what made
TP win the head-to-head: 113 µs/die for the whole block. End-to-end on the
**full 150 GiB model**: **+7.6% vs stock, +13% vs the ggml-kernel path**;
the block itself 3.1x faster than the CPU where resident; MoE down from
~21% to ~11% of decode time. 34 of 43 layers fit in the four dies' VRAM;
the other 9 fall back to the CPU (a capacity wall: 43 layers would need
34.3 GiB per 32 GB die).

## The border

Phase timers on the TP call: ~440 µs per layer, only ~113 µs of it GPU
compute ([04](04-vulkan.md#what-a-call-actually-looks-like)). Attacks on
the rest:

| verdict | idea | mechanism |
|---|---|---|
| ✅ 20x on its phase | HOST_CACHED memory for the readback buffer | write-combined memory reads at ~150 MB/s ([04](04-vulkan.md#memory-types-the-single-most-expensive-lesson-here)) |
| ✅ | persistent mapping (map once, keep the pointer) | `vkMapMemory` cost ~2 ms/layer per call |
| ✅ ~+0.5% | poll fences instead of blocking waits | skips a kernel sleep/wake per fence |
| ❌ | submit from four threads, one per die | `vkQueueSubmit` is serialized *inside the driver* (~35 µs each regardless of caller); threads just added wake latency |

What remains (~140 µs submit floor + ~200 µs launch latency) is structural:
it shrinks only with *fewer submissions per token*, and llama.cpp hands us
one layer at a time.

## Measure honestly

The results above survived because of rules learned by breaking them:

- **A run proves its own noise floor.** Full-model decode timings varied
  0.3-9% load-to-load *by day*; the benchmark script refuses to print a
  delta smaller than the spread it just measured. The 4-layer stub
  ([03](03-model.md#the-stub-cutting-the-model-down-to-an-instrument)) is
  the precise instrument; only costs that transfer (fixed µs per event, not
  percentages) are extrapolated to the full model.
- **A/B on the same day, interleaved, one variable.** The machine drifts
  overnight more than most effects being measured. And "the same flag
  pointed elsewhere" is not one variable — llama.cpp silently *repacks*
  expert weights placed on its CPU backend, so the honest control keeps
  weights in *our* buffer and toggles only whether we compute.
- **Probe the mechanism at small scale before building on it.** A synthetic
  40 GB bandwidth probe (minutes) killed a huge-pages theory that had lived
  in a plan for weeks; the instruction-bound diagnosis that justified the
  custom kernel came from the same style of probe.

### Test what you think you're testing

The clamp bug ([03](03-model.md)) shipped because the test fixture placed
all experts on one device — the multi-device code path under test never ran,
and the suite was green. Separately, six different checks in this project
passed while testing nothing (backend never loaded, flag eaten by the shell,
process killed mid-run...). The fix is mechanical: every run must assert
*engagement* — from its own log — before its result counts, and abort
otherwise.

### Compare logits, not text

Early on, "generated text is identical" was taken as proof of bit-exactness.
It is not: greedy sampling maps a whole neighborhood of logits onto the same
token, so identical text survives real numerical differences. All
correctness claims here compare **logits** (via files that are a
deterministic function of them —
[05](05-backend.md#how-correctness-is-proven)); "bit-identical" is reserved
for byte-equal files, and even that is a property of the *whole run
configuration* — thread count, compiler, batch shape, even which host binary
ran the trunk — not of the model alone.

## The scoreboard

| stage | decode (full model) | prefill (full model) |
|---|---|---|
| stock llama.cpp, CPU | 3.64 t/s | 18.49 t/s |
| + dies, ggml kernels, chunked | ~3.46 | **21.84 (+18%)** |
| + TP custom kernel (`MOESERV_TP=1`) | **3.92 (+7.6%)** | — (falls back) |

And the honest framing from [01](01-moe.md#which-part-is-worth-accelerating):
on *this* machine the trunk now dominates (~89% of decode), so further
expert-block work pays little here — but 439 µs/layer with 113 µs of GPU in
it is the number that transfers to a machine whose trunk is fast, and
that is the machine this block is being built for.
