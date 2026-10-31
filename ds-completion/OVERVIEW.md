# ds-completion: overview (2026-10-28)

A ~20-minute read: what was built, how it works, what we measured, what we learned, and what
is next. How to run it: [QUICKSTART.md](QUICKSTART.md). Every step and number in detail:
[README.md](README.md). Plan and cost model: [PLAN.md](PLAN.md). Operand types and accumulation:
[NUMERICS.md](NUMERICS.md).

## 1. Goal and scope

Text completion with **DeepSeek-V4.1-Flash-1017** on one Mac Pro 2019, as fast as the machine
allows, with **outputs that match the official reference implementation**.

- Priorities, in order: correctness, code clarity, performance.
- Non-goals: other models, other hardware, serving features (batching users, sampling options).
  A CLI (`ds-completion -m model -p "prompt"`) is the interface.

| | |
|---|---|
| model | 40 layers, hidden 5120; MoE with 384 routed experts (top-6) + 1 shared; compressed sparse attention with an indexer; hyper-connections (4 copies); engram n-gram memory (189 GB table) in 2 layers |
| weights | GGUF (UD-Q8_K_XL, 474 GB), lossless against the official 1017 release: experts mxfp4 (269 GB), dense weights fp8 expanded to bf16 |
| machine | Xeon W-3245 (16 cores, AVX-512), 755 GB RAM (~112 GB/s), 2x Radeon Pro Vega II Duo = 4 GPU dies, 32 GB HBM each (gfx906: no fp8, no bf16 math, no matrix cores) |

## 2. Results

Decode speed (sclk level 2, 16 CPU threads, 128 generated tokens):

| mode | decode |
|---|---|
| llama.cpp fork (llama-cli, CPU) | 3.5 t/s |
| ours, CPU only | 3.5 t/s |
| ours, own GPU runtime, routed experts on the CPU | ~12 t/s |
| ours, + most-used experts placed on the 4 GPUs (`--placement`) | 17.7-21.7 t/s |
| ours, + speculative decoding with the model's DSpark draft (`--dspark ... --spec`) | 16-24 t/s; +22% (geometric mean) on 8 held-out prompts, +7% to +33% |
| **ours, + adaptive expert cache (`--canon --cache-loads 200`)** | **mean 25.6 t/s with `--spec` on the 8 held-out prompts (+28% over the line above); 29.1 at sclk 8; 20.4 plain at 100 loads/s** |
| upper bound: placement from the prompt's own routing (perfect knowledge) | ~25 t/s plain; 29-41 t/s with `--spec` |

(The speculative and cache numbers are 300-token runs on held-out prompts; README steps 26, 32.)

Prompt processing (chunks of 512, README steps 37-41): 53-57 t/s, 49-54 with `--canon`, output
byte-identical to token by token (which ran at ~17 t/s). Startup: ~15 s, plus 10-30 s to upload
experts with a placement; with the cache, ~40-90 s more while the experts' memory is registered
(the GPUs stall meanwhile: PLAN item 30). The fastest settings per job: QUICKSTART.md.

Accuracy, against the official implementation run on the CPU in fp32:

- **Per layer**, teacher-forced (each layer gets the reference's own input): attention and FFN
  outputs within ~1e-7 at decode, i.e. equal up to fp32 summation order. Larger errors appear only
  where an fp8 rounding step flips at a near tie, and they appear in every exact implementation.
- **End to end:** the top token matches the reference at 16 of 17 steps (the miss is a near
  tie), the same for every mode, the llama.cpp fork included. Median KL ~1e-5.
- **Deterministic:** repeated runs give bit-identical output.
- **Speculative decoding changes nothing in the output:** it is byte-identical to plain greedy
  decoding on every prompt tested (18 prompts; 4 of them at 5 fixed draft lengths too), because every batched kernel
  gives each token the same bits as a one-token call.

## 3. How a token is computed

```
token -> embedding (host) -> 40 x [ engram (layers 1, 14) -> attention block -> FFN block ] -> head -> logits
                                        all on GPU 0                   router on GPU 0, then:
                                                                        shared expert  -> GPU 0
                                                                        6 routed experts -> wherever they live:
                                                                          GPU 0..3 expert pools  (hits)
                                                                          CPU, from the mmap      (misses)
                                                                        sum in a fixed order on GPU 0
```

- **Dense part on GPU 0** (`hip/rt.cpp`, `hip/kernels.hip`, `hip/fp8_gemv.hip`): our own HIP
  kernels, one token per step, ~1,500 kernel launches per token. The dense linears come from a
  lossless fp8 repack (`fp8pack`, 6.4 GB instead of 14 GB bf16); the GEMV multiplies fp8 x fp8
  exactly with fp16 dot products and fp32 accumulation.
- **Routed experts on GPUs** (`hip/expert_pool.hip`, `hip/fp4_moe.hip`): each die holds a pool of
  expert slots (18.8 MB each). A placement file ranks (layer, expert) pairs by how often the router
  chose them in a 30-prompt study; at startup the top ones that fit are uploaded (6,626 experts =
  43%). The fp4 kernel is exact too (fp4 and fp8 values are exact in fp16).
- **Routed experts on the CPU** (`src/cpu_moe.cpp`): an AVX-512 kernel reading the mxfp4 weights in
  place from the memory-mapped GGUF; exact; ~85 GB/s.
- **Per layer, the split:** a dispatch kernel sorts the router's 6 experts into one list per die
  and one for the CPU. One 20 KB copy brings the input and lists to host memory; dies 1-3 get it
  from there (cards are not peer-connected), GPU 0 runs its own list next to the shared expert,
  and the host runs only the CPU's misses. Nothing on the GPUs waits for the host.
- **Prompt:** chunks of up to 512 tokens through the same path (README steps 37-41): the dense
  projections as fp8 matrix multiplies, the CPU's experts batched per expert and computed for
  one half of the chunk while the GPU runs the other half's attention; attention still per
  token inside the chunk. Every result bit-identical to token by token.
- **Speculative decoding** (`hip/dspark.cpp`, `Runtime::verify/commit`): the release's DSpark
  draft (3 small layers reading the main model's layers 37-39, 8.5 GB on GPU 0) guesses the
  next 5 tokens in ~12 ms. The main model then runs [last token, guesses] as one chunk:
  projections, the head and the expert block read their weights once for all tokens; attention
  runs per token in position order. The longest correct prefix plus the model's own next token
  is kept, the rest rolled back (window slots, compressor state). How many guesses to verify is
  chosen per step from the draft's confidence head, which turned out to be calibrated: the n
  with the most expected tokens per ms, with verify costs fitted during the run.
- **Reference path** (`src/forward.cpp`): the whole model as ggml CPU graphs, module by module like
  the official `model.py`. Slow (3.5 t/s), but it is the second implementation every runtime
  change is checked against.

## 4. How correctness is kept

- **Python reference** (`ref/ds_ref.py`): the official implementation, run on the CPU with the
  1017 weights (from the GGUF, verified lossless), in three variants: as released, fp32, and
  fp32 without fake quantization. It records every layer's intermediates ("goldens").
- **`ds-test`**: feeds each layer the reference's own input and compares its outputs, for the
  CPU path and the runtime (`--runtime`, with or without `--placement`); `--e2e` compares
  logits. Errors cannot accumulate across layers, so every row tests one layer.
- **Rule for every change:** placement-only or reordering changes must be bit-identical to the
  previous version; numerics changes must stay at the same error level against the reference.
  Every step in the README records its gate.
- **Batched kernels** (speculative verification) have bitwise tests against one-token calls
  (`bench-fp8 --bitwise`, `bench-kernels --bitwise`, `bench-expert-pool`), each with a control
  that the test can see a difference; the end-to-end gate is speculative output = plain greedy
  output, byte for byte.

## 5. What we learned (the surprises)

- **The local safetensors were a different checkpoint** (the base release, not 1017). The GGUF
  was verified against 1017 over HTTP range reads and became the source of truth.
- **gfx906 can do the reference's fp8/fp4 math exactly** without fp8 hardware: fp8 and fp4 values
  become fp16 by bit shifts, and fp16 products are exact in fp32.
- **GPU nondeterminism** came from rocBLAS split-K atomics (fixed by an environment variable while
  we used ggml's GPU path; our runtime has no atomics).
- **Owning the decode path beat staying inside ggml:** ggml's GPU path reached 6 t/s; our runtime
  10 t/s with the same expert placement. ggml's GPU path was then removed entirely.
- **HIP graphs do not help on this ROCm** (0.1-2 us saved per launch); only fewer, fused kernels do.
- **A routed-expert hot set does not transfer well between prompts** (71% of experts served at
  128 GB from other prompts' statistics), but **recent usage does** (an LRU cache: ~95%). Code,
  math and Chinese prompts transfer worst.
- **Perfect placement exposed the expert path's serialization**, then fixes showed **the CPU
  thread count mattered most**: `-t 32` (all logical CPUs) starved the host thread and the HIP
  runtime; `-t 16` made every configuration 25-50% faster.
- **Cold starts:** the model is memory-mapped; after a reboot the first run reads experts from
  NVMe. Mapping the CPU experts at load (`MADV_POPULATE_READ`, 1.2 s) also removes ~4k page
  faults per first-used expert.
- **The DSpark draft's confidence head is calibrated out of the box** (sigmoid(score) = the
  observed acceptance rate), so choosing the draft length per step needs no tuning; the official
  code computes it but never uses it.
- **Batching does not make GEMVs free on gfx906:** at 6 tokens the exact fp8 GEMV first cost 3x
  one token, bound by decoding the activations in every row (arithmetic), not by memory;
  decoding each tile once into shared memory fixed half of it.
- **A rare GPU memory fault** at startup came from copying experts straight out of the mmap;
  staging through pinned memory removed it (and made uploads 4x faster).
- **GPU clock** (sclk) matters for latency-bound kernels, little for bandwidth-bound ones:
  sclk 4-6 gives most of the speed at much less power than 8. With speculation too, the clock
  buys little while CPU expert misses are half of a verify step: level 2 -> 8 gives +8% plain,
  +9% `--spec`, for ~30% more power (README step 27); level 0 is slower at the same energy per
  token as level 2.

## 6. Where a token's time goes now (sclk 2, 16 threads)

| ms per token | experts on CPU only | study30 placement (81% on GPUs) | perfect placement (97%) |
|---|---|---|---|
| dense: attention, hyper-connections, norms, engram, head | 26.1 | 26.1 | 26.1 |
| shared expert + router + dispatch/copies | 8.5 | 8.7 | 8.7 |
| routed experts on GPUs (critical-path part) | - | 4.2 | ~5 |
| routed experts on the CPU (critical-path part) | 51.6 | 11.0 | ~2 |
| **step** | **86.6** | **50.6** | **~43** |

A floor of ~35 ms (~28 t/s) remains whatever the caching: the dense part, the shared expert, the
router and the per-layer dispatch.

With speculative decoding, a verify step of 6 tokens (`results/spec-profiles.txt`):

| ms per verify step (6 tokens) | study30 placement | perfect placement |
|---|---|---|
| routed experts on the CPU (critical path) | 68 | 9 |
| attention core (per token, in order) | 20 | 21 |
| fp8 projections: attention q + output, shared expert | 35 | 37 |
| router, hyper-connections, indexer, compressor, head, rest | ~48 | ~54 |
| **step** | **171** | **121** |

plus the draft, ~12 ms (at sclk 8 the GPU parts shrink ~25%; the CPU misses do not: 152 ms).
One token alone costs ~50 ms, so 6 tokens verified cost ~3.4x one; the
win depends on how many guesses are right (2 per step on chat, 4 on code).

With the adaptive cache (`--canon --cache-loads 200`, sclk 8; README step 33,
`results/verify-breakdown.txt`), the dense part is now most of a verify step:

| ms per verify step (6 tokens) | static study30 | + cache 200/s |
|---|---|---|
| **CPU experts** (the GPU waiting for them) | **77.6** | **24.8** |
| (host CPU expert compute, partly overlapped) | 85.4 | 32.3 |
| (expert uses per step on GPU / CPU) | 1,156 / 284 | 1,348 / 92 |
| **GPU experts** (visible part; the rest overlaps the shared expert) | 2.9 | 4.2 |
| attention: core (per token, in order) | 15.8 | 15.9 |
| attention: output projection | 11.1 | 11.2 |
| attention: q, indexer, compressor, kv | 15.0 | 15.2 |
| shared expert | 13.4 | 14.4 |
| router + dispatch + sum (per layer) | 10.5 | 10.6 |
| hyper-connections | 4.7 | 4.7 |
| output head | 3.7 | 3.7 |
| engram | 14.6 (cold table pages; ~3 warm) | 2.8 |
| **step** | **172.4** (~160 warm) | **110.8** |
| draft (separate) | 8.8 | 11.6 |

Per verify step plus draft (~122 ms): dense part ~64%, CPU experts ~20%, draft ~10%, GPU
experts ~3% visible.

## 7. What is next

1. **Long context** (PLAN.md "Long context", README step 34): the runtime now runs past
   4,096 tokens (65,536 by default, 1M with `--max-ctx`): a top-k for any number of positions and
   the candidate-block pre-filter beyond 16k, both checked against the reference's own
   functions; a 29.5k-token prompt decodes at 17.2 t/s; at 20k tokens the runtime gives the
   reference's greedy tokens with short-context KL (step 35). Left: compressed caches in fp8/fp4
   (for contexts far beyond 64k), faster prompt processing (in progress, README step 37:
   chunks of up to 512, CPU experts batched and overlapped with the GPU, an fp8 matrix-multiply
   kernel: ~17 -> 57 t/s so far, ~17 -> 49 with `--canon`).
2. **Prompt processing, next** (PLAN "Prompt processing: where it stands and the options"): the
   CPU's experts are the limit (~30 of 36 s for 1,989 tokens); options: larger chunks, streaming
   the CPU's experts to the GPUs for long prompts, then the GPU side (batched attention core).
3. **Adaptive expert cache, tuning** (README steps 28-32): built and working (`--canon
   --cache-loads R`: +28% with `--spec` at sclk 2, every output identical); left: the default
   rate, hysteresis so it settles in long runs, the fast path's CPU preparation (the same fix as
   the canonical one).
4. **The rest of a verify step:** fp8 projections over 64-lane rows are still ~2x one token at
   6 tokens; the router's 6 top-k launches; attention core per token.
5. **The ~35 ms floor:** a higher GPU clock (free, if the power is acceptable), fusing the router
   and dispatch, fusing small kernels; tensor parallelism across dies last.

Explored, not planned: a second machine (pipeline split; ~1-2 ms gain once the cache adapts),
RDMA between machines, an MI210 for the dense part (~33 t/s estimated), and a hypothetical
V4.1-Pro (engram on SSD is easy; experts barely fit in two machines' RAM).

## 8. Open questions for review

- Which prompts represent real use? The placement and the cache's starting set depend on it.
- Which GPU clock level is acceptable for power (sclk 2 today; 4-6 is the sweet spot; level 8
  costs ~17-25% more energy per token for ~8-9% speed while CPU misses dominate)?
- Is the remaining correctness risk acceptable: near-tie fp8 rounding flips make the top token
  differ from the reference occasionally (1 of 17 steps on the test prompt), in every
  implementation we compared, the llama.cpp fork included.

## 9. Code map

| path | what |
|---|---|
| `src/ds_completion.cpp` | the CLI |
| `src/model.{h,cpp}` | GGUF loading (zero-copy mmap), the model's schema and hyperparameters |
| `src/forward.{h,cpp}` | the CPU reference path (ggml graphs), used by tests and `bin/ds-completion` |
| `src/cpu_moe.{h,cpp}` | the exact AVX-512 routed-expert kernel |
| `src/fakequant.{h,cpp}` | the reference's fp8/fp4 rounding |
| `src/fp8pack.cpp` | lossless fp8 repack of the dense weights |
| `src/ds_test.cpp` | per-layer and end-to-end tests against the goldens |
| `hip/rt.{h,cpp}` | the GPU runtime: state, layer order, the expert split |
| `hip/kernels.{h,hip}` | the runtime's kernels (norms, attention, indexer, hyper-connections, router, dispatch) |
| `hip/fp8_gemv.{h,hip}` | exact fp8 x fp8 GEMV |
| `hip/fp4_moe.{h,hip}` | exact fp4 routed-expert kernel (4 launches) |
| `hip/expert_pool.{h,hip}` | expert slots per die, placement reading and assignment |
| `hip/dspark.{h,cpp}` | the DSpark draft (3 layers, Markov and confidence heads) |
| `hip/bench_*.cpp`, `src/bench_*.cpp` | benchmarks of each kernel |
| `ref/` | the Python reference, golden export, weight checks, logits comparison |
| `scripts/` | routing study and analysis, clock sweep, DSpark confidence calibration (`spec_conf.py`) |
| `placements/` | ranked expert placements (`study30.txt` is the default) |
| `results/` | raw outputs of every measurement quoted in the README |
