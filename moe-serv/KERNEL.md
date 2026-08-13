# Kernel tuning log — 2-pass MXFP4 vecmat on Vega II

One entry per attempt, in order, kept whether it worked or not — lyrae's
numbered experiment series is the model, and its negative results (single-pass
10x slower, fused-into-slow-matmul 6x slower) saved this project real time.

Standard measurement: `moe-probe --custom --reps 100`, GPU timestamps, warm-up
rep excluded, both decode shapes:

    gate/up   --k 4096 --m 2048     (2 of the block's 3 matmuls)
    down      --k 2048 --m 4096

Correctness on every run: max abs vs ggml `to_float` + f32 `mul_mat_id`, gate
`|diff| <= 1e-4 + 1e-3|ref|`. A speedup on a MISMATCH row does not exist.

Bounds to race against, per matmul (26.7 MB, 6 experts):

    ggml mxfp4 span     163.8 µs   (the kernel being replaced)
    f32 practical BW     ~38 µs    (26.7 MB at the ~700 GB/s f32 proves)
    1 TB/s byte bound    ~27 µs

## Baseline — commit `70c073f`

lyrae 2-pass structure: K tiles, partials, reduce. Two-plane repack (nibble
u32[K][M/8], scales u8[K/32][M]), 8 cols/thread, 16-float LUT in LDS, scale per
32-sub-block, f32 accumulators.

| shape | tile 32 | tile 64 | tile 128 |
|---|---|---|---|
| gate/up | 136.4 | 115.5 | **111.9** |
| down | — | **88.3** | — |

vs ggml: 1.46x / ~1.9x. Suspected limiter: inner-loop instruction stream —
per u32: 1 global load, 8x (shift, mask, LDS read, FMA), and the 16-entry LUT
sits in 16 LDS banks so a 64-lane wave conflicts up to 4-way on every read.

## E1 — LUT replicated 4x across banks: WORSE, reverted

Premise: 16-entry LUT in 16 banks conflicts up to 4-way for a 64-lane wave.
Result: gate/up 111.9 -> 118.1 µs (-5.5%), down 88.3 -> 98.0 (-11%).

The premise was wrong about GCN: LDS reads of the *same word* broadcast, and a
16-entry table in 16 banks means every access to bank b is to one word — the
baseline was conflict-free by construction. Replication only added an OR to
every one of the 8 reads per u32. Corollary that reorders the queue: the LUT
read is nearly free, so replacing it with arithmetic decode (~5 ALU ops) is
now expected to lose, and that experiment is skipped rather than run.

## E2 — paired-k uvec2 loads: kept, +2.3% net, and a diagnosis

One 8-byte load now covers 16 weights (rows k, k+1 x 8 columns); loop
iterations and load instructions halved.

| shape | before | after |
|---|---|---|
| gate/up (tile 128) | 111.9 | **107.0** |
| down (tile 64/128) | 88.3 | 91.1 |

Net on the block (2x gate/up + down): 312.1 -> 305.1 µs. Kept for the gate/up
win, but the real yield is the diagnosis: halving global-load instructions
changed almost nothing, so the limiter is the per-weight decode chain
(shift+mask+LDS+FMA, ~3 issue slots per weight), not the loads. Next
experiments should cut ops per weight, not bytes or loads.

## E3 — byte-pair vec2 LUT (256 entries): WORSE, reverted

One ds_read_b64 per two weights instead of two nibble reads; no unpack.
Result: gate/up 107.0 -> 127.9 µs (-20%), down 91.1 -> 117.0 (-28%).

Closes the loop with E1 from the other side: the 16-entry LUT is fast *because*
every lane reading entry i hits the same LDS word and broadcasts; 256 entries
under random byte indices spread over 512 words and conflict for real. Fewer
but conflicting reads lose to more but broadcast ones. Between E1 and E3 the
LUT is now known to be at its optimum size.

Standing diagnosis after E1-E3: not loads (E2), not LUT conflicts (E1/E3), not
occupancy (tile 64 runs 6 waves/SIMD vs tile 128's 3 and is *slower*), and a
naive VALU count puts the floor at ~14 µs — so the 107 µs is latency-bound:
every FMA depends on an LDS read ~30 cycles away, with only 3 waves/SIMD and 8
chains/thread to hide it. Next: attack latency (prefetch, then LDS-free
arithmetic decode — whose *throughput* argument E1 killed but whose *latency*
argument is untested).

## E4 — software prefetch of the next uvec2: WORSE, reverted

Gate/up 107.0 -> 109.5, down 91.1 -> 93.6. GCN issues loads asynchronously and
waits via s_waitcnt at first use, which the compiler already places after
independent work — the hardware was prefetching; the manual version only added
an address clamp per iteration. Latency, if it is the limiter, is not hidable
from *inside* one iteration's window.

## E5/E5b — LDS-free arithmetic decode: 2x WORSE, reverted

Constructing the doubled-e2m1 value as float bits (~6 VALU/weight, ~4-cycle
chain) instead of the LUT read: gate/up 107.0 -> 198.2 µs, down 91.1 -> 180.0.
E5b replaced the quarter-rate v_mul_lo_u32 in the constant select and changed
nothing (199.0), so the slowdown is the decode itself — plausibly VGPR
pressure from 16 inlined copies, or plain instruction count under a
stall-dominated issue schedule. Either way the latency hypothesis joins the
throughput one: the broadcast-LDS LUT beats arithmetic decode from both sides.

Ledger after five experiments: E2 kept (+2.3%), E1/E3/E4/E5 reverted. The
inner loop as first written — bfe, broadcast LDS read, FMA — has survived
every attempt to improve it locally. Remaining untried levers are structural:
occupancy (E6), packed-f16 math, or accepting 107/91 and integrating.
