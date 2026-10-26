# mlp-matmul — one dense MLP matmul through llama.cpp's HIP kernels

Studies the matmul `y = W x` of one dense MLP layer (Qwen3.8-27B) on the Mac Pro's
Vega II dies, for N = 1 (single-token decode), 2-8 (tiny batch / speculative
verify) and 16-512 (prefill-like). Uses ggml from `~/projects/llama.cpp` as a
library (`build-hip`: ROCm 6.3.4, gfx906, commit `ce8caa6e6`), so the kernels
are exactly the ones llama.cpp runs.

Plan, step by step:

1. extract one MLP layer to its own file - **done**
2. benchmark llama.cpp's kernel on one die, with correctness - **done** (sclk level 2)
3. split W across 2 / 4 dies, run the same kernel on each slice, merge, check the merge
4. N sweep on all of the above (1, 2-4, 8, 16-512)
5. repeat at sclk level 3 (compute and transfers)

## Tools

```bash
make
bin/mlp-extract list    ~/llms/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
bin/mlp-extract extract ~/llms/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf 40 data/q4-l40.gguf
bin/mm-bench data/q4-l40.gguf [--dev D] [--tensor gate|up|down|all] [--ns 1,2,4] [--no-check]
bin/ceiling [D]           # raw-HIP memory-bandwidth / FMA ceilings of one die
```

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
