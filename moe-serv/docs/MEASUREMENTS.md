# The measurements

Every number this project quotes, in one place, each with its instrument and
its noise floor. The history and the reasoning live in `../PLAN.md` and the
commit messages; this file is what was measured.

**The discipline** (learned here, enforced by `bench.py`):

- `-lm none` always; a single mmap-backed timing on Windows is worthless.
- Every configuration on more than one load, round-robin — load-to-load spread
  is larger than within-run spread and is the real error bar.
- A run proves its own noise floor, and a delta below it is NOT RESOLVED. The
  real model's decode floor has ranged **0.3-9% by day**; the 4-layer stub
  (`make_stub.py`) holds 0.0-2.3% and is the decode instrument. Only a
  quantity that transfers (a fixed per-event cost, not a percentage) may be
  extrapolated from it.
- Same-day interleaved A/Bs only, for small effects: stock stub decode moved
  28.4 -> 30.8 t/s overnight, and even an unchanged DLL's phase profile moved
  429 -> 463 µs. Cross-day comparison of either t/s or timers is unquotable.
- Every run proves from its own log where the expert weights went and whether
  we computed; a run that cannot aborts instead of being tabulated. Six
  "passes" in this project have tested nothing or the wrong thing.

Model throughout: DeepSeek-V4-Flash UD-Q8_K_XL, 150.75 GiB, 43 expert layers;
stub = its first 4 layers, 15.78 GiB. Host: 16-core Xeon, four Vega II dies.

## What owning the weights costs (the repack forfeit)

llama.cpp rewrites MXFP4 experts into `CPU_REPACK` layout at load and runs a
different GEMM; we compute on the bytes as they sit in the file. That gap,
measured (`gate.py --vs-stock`): **mean KLD 3.6e-5, max 2.1e-3, top-1
99.804%** — the yardstick for what two *correct* kernels disagreeing looks
like on this machine. In throughput it is the `stock -> ours-off` column:
**-4.4 to -4.5%** of decode on both stub and full model.

## Decode, CPU-only host (why the stub exists)

Full model, two loads per configuration, tg32:

| config | load 1 | load 2 | mean | load-to-load |
|---|---|---|---|---|
| `stock` | 3.34 ± 0.05 | 3.59 ± 0.20 | 3.46 | 7.2% |
| `ours-off` | 3.47 ± 0.03 | 3.18 ± 0.03 | 3.33 | 8.7% |
| `ours-on` | 3.12 ± 0.04 | 3.39 ± 0.10 | 3.25 | 8.3% |

Nothing resolved: sorted, the six measurements interleave and the two loads
rank the configurations differently. The same measurement on the stub (floor
0.0-2.3%) resolves the repack at **3.2%** and bounds our split boundary at
**≤216 µs per split** — which scales to **≤3%** on the real model's 43 splits,
the kind of bound the big model could never produce about itself.

## Prefill on the dies, ggml kernels (`dies`)

Full model, 36 of 43 layers mirrored (9 per die), `--pp 512`, two loads,
spreads ≤1.1% — prefill measures well on the real model, unlike decode:

| pp512 | stock | ours-off | ours-on |
|---|---|---|---|
| t/s | 18.49 | 18.02 | **21.84** |

**Our compute +21.1%, net vs stock +18.1%.** The enabling fact, stub, by
batch size (our compute vs same weights on CPU):

| batch | 8 | 16 | 32 | 128 | 512 | decode |
|---|---|---|---|---|---|---|
| unchunked | +21.9% | -60.2% | -54.6% | -51.4% | — | -0.1% |
| 8-token chunks | **+26.7%** | **+32.9%** | **+35.1%** | **+30.1%** | **+27.2%** | -0.2% |

The cliff is `ggml_vk_use_mul_mat_vec_id` (`ggml-vulkan.cpp:10607`): the fast
vector path ends at 8 tokens. Issuing the block 8 tokens at a time is a graph
decision, not a shader.

## Decode on the dies, ggml kernels (single read-back)

Stub, per-layer, steady-state medians from the per-split profiler
(`MOESERV_PROFILE`):

| µs per layer | 6 reads | 1 read |
|---|---|---|
| compute | 1018 | 1106 |
| read-back | **530** | **149** |
| total | 1578 | **1279** |
| CPU layer | 1420 | 1403 |

Six `ggml_backend_tensor_get` calls at ~88 µs each were the border; reading
the terminals' common view-root once turned the layer from 10% slower than
the CPU to 9.7% faster, and stub decode from flat to **+6.9%**. First Vulkan
call costs 306 ms (pipeline compile) — quote medians.

## Decode TP, custom kernel (`tp-integrate`)

Kernel-level numbers (probe, synthetic weights): see `KERNEL.md` — the 2-pass
mxfp4 kernel runs the block's matmuls at 95.4/86.5 µs vs ggml's 163.8, and
the full TP block costs **113 µs/die**.

Full model, four configurations, two loads each, spreads 0.3-1.3% (a lucky
day; see the discipline note):

| tg32 | stock | ours-off | ours-on | ours-tp |
|---|---|---|---|---|
| mean t/s | 3.64 | 3.48 | 3.46 | **3.92** |

**TP vs stock +7.6%, TP vs the ggml path +13.0%.** 34 of 43 layers resident
under the 28000 MB/die budget (816 MiB per layer per die; 43 would need
34.3 GiB against 32 GB of HBM), 9 on the CPU delegate — 5474/1449 splits,
exactly the capacity arithmetic.

Per resident layer, phase timers: **439 µs** = stage 4.5 + submit 125.1 +
wait-first 225.7 (containing ~113 µs of GPU) + wait-rest+sum 83.9 — within 2%
of the stub's number, so the border is per-call, not per-model. Against the
CPU delegate's **1.38 ms/layer**: the block is **3.1x faster where resident**,
and MoE falls from ~21% of decode time to ~11%. Bounds on this host: perfect
residency ≈ +3%, MoE free ≈ +12%.

Two border fixes worth ~2 ms/call together, found by the phase timers:
persistent mapping, and HOST_CACHED memory for anything the CPU reads
(`MECHANISM.md` #9).

## The border (`36be37a`)

Same-day interleaved A/Bs on the stub, three pairs each, engagement asserted
per load. **Threaded per-die submit: refuted** — +0.2% against 0.4% spread,
phase totals equal; `vkQueueSubmit` costs ~35 µs serialized in the driver
whichever thread issues it. **Fence polling: kept** — `vkGetFenceStatus`
spins instead of blocking waits, ~+0.5%, poll ahead in 6 of 6 pairs across
both pair orders (both first A/Bs' second-in-pair had come out ahead, so the
order flip was the discriminating run). The remaining ~340 µs of border is
structural: serialized submit floor plus launch-to-completion latency,
reachable only with fewer submissions per token.

## The correctness ledger

| path | comparison | result |
|---|---|---|
| CPU delegate | byte-equality of `--kl-divergence-base` files | **bit-identical** (`--tol 0`) |
| ggml-vulkan mirror | mean KLD vs same-placement CPU control | 6.2e-5 – 8.4e-5 across commits |
| TP, real weights | mean KLD vs same-placement CPU control | **1.070e-4** (build-vk host) / **1.780e-4** (CPU host) |

The tolerance typed on every Vulkan/TP command line is 5e-4 — ~14x the repack
gap. The TP number has two spellings because the *host build* is an axis of
the run configuration: each reproduces exactly on its own instrument, and the
TP logit files are byte-identical across every DLL variant that never touched
arithmetic (old / workers / poll). A KLD change without a config change is a
red flag (repo `CLAUDE.md`: corrupt-weights tell); check the instrument first.
