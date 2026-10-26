# mlp-matmul — one dense MLP matmul through llama.cpp's HIP kernels

Studies the matmul `y = W x` of one dense MLP layer (Qwen3.8-27B) on the Mac Pro's
Vega II dies, for N = 1 (single-token decode), 2-8 (tiny batch / speculative
verify) and 16-512 (prefill-like). Uses ggml from `~/projects/llama.cpp` as a
library (`build-hip`: ROCm 6.3.4, gfx906, commit `ce8caa6e6`), so the kernels
are exactly the ones llama.cpp runs.

Plan, step by step:

1. extract one MLP layer to its own file - **done**
2. benchmark llama.cpp's kernel on one die, with correctness - **done** (sclk level 2)
3. split W across 2 / 4 dies, run the same kernel on each slice, merge, check the merge - **done**
4. N sweep on all of the above (1, 2-4, 8, 16-512) - **done** (sclk level 2)
5. repeat at sclk level 3 (compute and transfers)

## Tools

```bash
make
bin/mlp-extract list    ~/llms/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
bin/mlp-extract extract ~/llms/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf 40 data/q4-l40.gguf
bin/mm-bench data/q4-l40.gguf [--dev D] [--tensor gate|up|down|all] [--ns 1,2,4] [--no-check]
bin/ceiling [D]           # raw-HIP memory-bandwidth / FMA ceilings of one die
bin/mm-split data/q4-l40.gguf [--tensor gate] [--mode row,k] [--ns 1,4] [--configs 0/0,1/0,2/0,1,2,3]
python3 summarize.py results/split-q4-l40-sclk2.txt [--speedup] [--field e2e|comp|slice]
```

`mm-split` (one line per die config and N; see the header comment in `src/mm_split.cpp`):

- `row` mode: die d holds a contiguous range of output rows and gets all of x;
  the merge places the slices side by side and must be **bit-identical** to 1 die.
- `k` mode: die d holds a K range of every row (cut on 256-element quant-block
  boundaries) plus the matching part of x; the merge sums the partials in die
  order and must match 1 die to ~1e-6 (only the order of additions changes).
- Times: `slice` (one slice's kernel alone), `compute` (all dies at once) and
  `e2e` (H2D x -> compute -> D2H y into pinned memory -> host merge). Compute
  and e2e are issued two ways: `ser` from the main thread, `thr` with one
  spinning worker thread per die. A `phases` line splits the serial e2e into
  launch / h2d / d2h / merge.
- Cross-card merges go through host memory only (ROCm 6.3.4 crashes on
  device-to-device copies between the two Duo cards).
- `summarize.py` pivots a log into N x config tables (best of ser/thr) and lists
  any row that isn't `ok`.

- `mlp-extract` copies `blk.L.ffn_{gate,up,down}.weight` byte for byte (names,
  quant types and layout unchanged) and re-reads the output to verify every byte.
- `mm-bench` prints each card's sclk/mclk level in its header, then for each N:
  - `single`: wall time of one `ggml_backend_graph_compute` holding one matmul
    (launch + kernel + sync, what a decode step pays);
  - `stream`: time per matmul with 16 in one graph, close to the kernel time;
  - GB/s (weight bytes / stream time) and TFLOPS.
- Correctness: W is dequantized on the host with ggml's own `to_float`, and the
  reference is accumulated in double for **all** N columns. `relerr` is
  `||y-ref|| / ||ref||`, the pass gate is < 2e-2; the GPU quantizes x to q8_1,
  so ~4e-3 to 9e-3 is expected. A negative control (gate output scored against up's
  reference) must fail, and does (relerr ~1.4). `single` and `stream` outputs must
  also be bit-identical.
- `x` sits in its own static buffer, not in gallocr's (see the gallocr entry in
  `../CLAUDE.md`).

## The layers (Unsloth UD mixes vary the quant type per layer)

`n_embd` 5120, `n_ff` 17408. gate/up are K=5120 -> M=17408, down is 17408 -> 5120.

| file | source | layer | type (gate/up/down) | per tensor | layers sharing the type |
|---|---|---|---|---|---|
| `data/q4-l40.gguf` | UD-Q4_K_XL | 40 | q5_K | 58.4 MiB | 16 of 65 (most common) |
| `data/q4-l19.gguf` | UD-Q4_K_XL | 19 | iq4_xs | 45.2 MiB | 10 of 65 |
| `data/q2-l41.gguf` | UD-Q2_K_XL | 41 | iq3_xxs | 32.5 MiB | 4 of 65 |

`data/` is git-ignored; regenerate with the `extract` lines above.

## Ceilings (one die, sclk level 2 = 1204 MHz, HBM mclk fixed 1000 MHz)

`results/ceiling-sclk2.txt`, all four dies within ~4%:

- read 58 MiB (one weight tensor): ~70 us back-to-back, **~880 GB/s**
- read 1 GiB: ~790 GB/s; D2D copy: ~770 GB/s (read + write)
- FP32 FMA: **9.63 TFLOPS** (= 1175 MHz x 64 CU x 128 FLOP/clk, so the pin is live)

## Step 2 results: one die, sclk level 2

`results/single-*-sclk2.txt`. Stream time per matmul, us (gate; up is the same
within 1%, down is in the files):

| N | q5_K | iq4_xs | iq3_xxs | q5_K GB/s | q5_K TFLOPS |
|---|---|---|---|---|---|
| 1 | 155 | 119 | 321 | 395 | 1.2 |
| 2 | 186 | 148 | 357 | 330 | 1.9 |
| 3 | 225 | 233 | 381 | 272 | 2.4 |
| 4 | 312 | 241 | 415 | 196 | 2.3 |
| 8 | 496 | 344 | 523 | 124 | 2.9 |
| 16 | 546 | 586 | 909 | 112 | 5.2 |
| 32 | 1332 | 815 | 1021 | 46 | 4.3 |
| 64 | 1885 | 1493 | 1554 | 33 | 6.1 |
| 128 | 3122 | 2479 | 2586 | 20 | 7.3 |
| 256 | 5614 | 4444 | 4648 | 11 | 8.1 |
| 512 | 10644 | 8420 | 8769 | 5.8 | 8.6 |

Run to run: +-2% between processes and between die 0 and die 3.

What it says:

- **N=1 is at ~45% of the memory ceiling.** q5_K and iq4_xs both read weights at
  ~395 GB/s against ~880 for a plain read of the same bytes. iq3_xxs is far worse
  (106 GB/s, 321 us for a 32 MiB tensor): its decode is compute-bound, so the
  smallest file is the slowest at N=1.
- **Tiny batches are not free.** The vector kernel (MMVQ, N <= 8) re-decodes the
  weights per column: q5_K N=4 costs 2.0x N=1 and N=8 costs 3.2x. iq4_xs scales
  better (N=8 = 2.9x), iq3_xxs best (1.6x), because its decode cost is shared.
  Odd point: iq4_xs down at N=3 (281 us) is slower than at N=4 (238 us).
- **Large N reaches ~8.6-10.9 TOPS** with MMQ at N=512. That's ~22-28% of the
  int8 dot-product peak (~39 TOPS at this clock), but close to FP32 peak, and it
  scales slowly with N.
- **Per-call overhead** (`single` - `stream`) is ~25-30 us per graph compute:
  about 16% of an N=1 matmul.

## Steps 3-4 results: split across 2 and 4 dies, sclk level 2

`results/split-*-sclk2.txt`. All rows pass on all three layers: every row split
is bit-identical to 1 die, every K split is within ~2e-7 of it, both issue
paths give identical bits, and relerr against the reference is unchanged.

End-to-end us (best of serial/threaded issue), `gate` split by rows and `down`
split by K, the standard tensor-parallel layout:

| layer | tensor | N | 1 die | 0,1 (same card) | 0,2 (two cards) | 4 dies | 4-die speedup |
|---|---|---|---|---|---|---|---|
| q5_K | gate/row | 1 | 218 | 154 | 154 | 122 | 1.8x |
| q5_K | gate/row | 8 | 612 | 391 | 366 | 269 | 2.3x |
| q5_K | gate/row | 64 | 2385 | 1879 | 1690 | 1085 | 2.2x |
| q5_K | gate/row | 512 | 14749 | 10540 | 8568 | 6841 | 2.2x |
| q5_K | down/k | 1 | 234 | 154 | 151 | 128 | 1.8x |
| q5_K | down/k | 64 | 2597 | 1738 | 1457 | 1092 | 2.4x |
| iq4_xs | gate/row | 1 | 182 | 135 | 134 | 109 | 1.7x |
| iq4_xs | down/k | 1 | 182 | 129 | 127 | 113 | 1.6x |
| iq3_xxs | gate/row | 1 | 385 | 244 | 245 | 173 | 2.2x |
| iq3_xxs | down/k | 1 | 321 | 231 | 228 | 186 | 1.7x |

Note the 1-die e2e includes the same H2D/D2H/merge as the splits (~60 us at N=1
on top of the 155 us kernel), so it's slower than `mm-bench`'s numbers.

What it says:

- **The kernel slices scale; the fixed per-call costs don't.** At N=1 a 1/4 slice
  of q5_K gate runs in 48 us (full: 158 us), but every graph compute costs
  ~30 us of launch + sync, and each die's async launch costs ~17 us of host
  time. Issued from one thread, 4 dies wait 68 us just for launches. One
  spinning worker thread per die removes most of that (4-die N=1: 246 -> 122 us).
- **Decode-bound types gain the most.** iq3_xxs reaches 2.2x at N=1 because its
  kernel time is large next to the fixed costs; iq4_xs, the fastest type, only 1.7x.
- **Row-splitting `down` doesn't work for N=16-64** (0.9-1.0x): for this shape a
  half or quarter slice runs as long as the full matrix (q5_K N=16: 851 / 841 us
  against 882). MMQ runs one workgroup per (128-row x N) tile and each one walks
  the whole K=17408; with 5120 rows split 2 or 4 ways there are fewer tiles than
  the 64 CUs, so the time doesn't shrink. K-splitting `down` shortens that loop
  and scales 2.2-2.6x. Use rows for gate/up and K for down.
- **The two dies on one Duo card share one host PCIe link.** For large N, 0,2
  (two cards) beats 0,1 (one card) by 15-25%: D2H of 17 MB took 1.29 ms from
  0+1 (13 GB/s) and 0.65 ms from 0+2 (27 GB/s). Host-staged 2-way traffic
  should use one die per card.
- **K-split merges cost more at large N:** every die returns a full [M, N] partial
  (D x the bytes of a row split) and the host sums them. That's fine for `down`
  (M=5120); for gate/up use rows.
- **4-die prefill tops out near 2.2x**, partly from MMQ under-filling the CUs on a
  4352-row slice (34 tiles for 64 CUs) and partly from host transfers.

## Next

- **Per-die fused MLP**: gate+up stacked into one tensor and split by rows,
  SwiGLU local, down split by K over the same n_ff range. That's one graph, one
  launch and one partial per die per layer, instead of three round trips.
- **Quant-type survey** for the decode path (q4_0, q4_K, iq4_xs, q8_0 at N=1-8):
  every type measured so far reads weights at <= 45% of the 880 GB/s ceiling,
  and iq3_xxs at 12%.
- **2-way on an XGMI pair** with the reduction as a peer copy (no host trip),
  and 4-way with a hierarchical reduce (XGMI within the pair, host across cards).
- **sclk level 3** rerun of `ceiling`, `mm-bench`, `mm-split` and the transfer tests.
