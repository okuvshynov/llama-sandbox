# ds-completion: overview (2026-11-18)

A ~20-minute read: what was built, how it works, what we measured, what we learned, and what
is next. First written 2026-10-28; brought up to date at step 156. How to run it: [QUICKSTART.md](../QUICKSTART.md). Where the time goes, one page: [docs/timing.md](timing.md). Every step and number in detail:
[docs/steps.md](steps.md). Plan and cost model: [docs/plan.md](plan.md). Operand types and accumulation:
[docs/numerics.md](numerics.md).

## 1. Goal and scope

Text completion with **DeepSeek-V4.1-Flash-1017** on one Mac Pro 2019, as fast as the machine
allows, with **outputs that match the official reference implementation**. Since step 111 also
on two of them, the model cut in two halves, with the same outputs.

- Priorities, in order: correctness, code clarity, performance.
- Non-goals: other models, other hardware, serving many users at once.
- One sequence at a time, text only. Greedy by default; temperature and top-p with a seed
  since steps 148-151.
- Three ways in, one engine: a program for one prompt (`ds-completion -m model -p "prompt"`),
  an interactive session (`-i`), and since steps 152-154 requests on stdin (`--requests`) with
  a small HTTP server in front that speaks the Anthropic Messages API.

| | |
|---|---|
| model | 40 layers, hidden 5120; MoE with 384 routed experts (top-6) + 1 shared; compressed sparse attention with an indexer; hyper-connections (4 copies); engram n-gram memory (189 GB table) in 2 layers |
| weights | GGUF (UD-Q8_K_XL, 474 GB), lossless against the official 1017 release: experts mxfp4 (269 GB), dense weights fp8 expanded to bf16 |
| machine | Xeon W-3245 (16 cores, AVX-512), 755 GB RAM (~112 GB/s), 2x Radeon Pro Vega II Duo = 4 GPU dies, 32 GB HBM each (gfx906: no fp8, no bf16 math, no matrix cores) |

## 2. Results

Decode speed (16 CPU threads; sclk level 2 and 128 generated tokens unless a row says otherwise):

| mode | decode |
|---|---|
| llama.cpp fork (llama-cli, CPU) | 3.5 t/s |
| ours, CPU only | 3.5 t/s |
| ours, own GPU runtime, routed experts on the CPU | ~12 t/s |
| ours, + most-used experts placed on the 4 GPUs (`--placement`) | 17.7-21.7 t/s |
| ours, + speculative decoding with the model's DSpark draft (`--dspark ... --spec`) | 16-24 t/s; +22% (geometric mean) on 8 held-out prompts, +7% to +33% |
| **ours, + adaptive expert cache (`--canon --cache-loads 800 --cache-period 8`)** | **mean 44.6 t/s with `--spec` on the 8 held-out prompts at sclk 6 (README steps 50, 53, 57, 62, 68, 71, 73, 74, 80, 82, 84, 86, 88, 97, 98, 101, 102, 104); at 200 loads/s 25.6 at sclk 2 (+28% over the line above), 28.0 at sclk 6; 20.4 plain at 100 loads/s** |
| for scale, at sclk 2 and before steps 50-104: placement from the prompt's own routing (perfect knowledge) | ~25 t/s plain; 29-41 t/s with `--spec` |

(The speculative and cache numbers are 300-token runs on held-out prompts; README steps 26, 32.)

Sampled (temperature 1, top-p 0.95; README steps 149-151): 43.1 t/s mean on the same prompts
against 44.9 greedy in the same session. The loss is the draws (~0.5 ms each on the host), not
acceptance: as many drafts are accepted as with greedy decoding.

Through the server (README step 153): one request of the benchmark it was written for, 1,305
prompt tokens and 42,960 tokens of reasoning ending in a tool call, at 38.5 t/s: 19 minutes.

Two machines (README steps 110-112; sclk 6, the same prompts): **mean 50.9 t/s** against 44.6
(+14%; +12% to +27% on code, math, SQL and JSON, +8% on chat, +2.5% on Chinese and Japanese,
nothing on a long reasoning run whose cache had adapted), every output byte-identical. Layers 0-19 run on the first machine,
layers 20-39 with the head and the draft on the second; once per step the residual stream (64 KB
per token) crosses 10 GbE. Each machine then holds ~88% of its half's experts on GPUs instead of
~42%: the gain is the CPU's experts, and the network costs ~0.5 ms of an ~83 ms step.

Prompt processing (chunks of 512, README steps 37-45, 68): 93 t/s, 63 with `--canon` (1,989 tokens, sclk 6),
output byte-identical to token by token (which ran at ~17 t/s). Startup: ~15 s, plus 10-30 s to upload
experts with a placement; with the cache, its loads start after the experts' memory is registered
(~95-135 s in the background; README step 43). The fastest settings per job: QUICKSTART.md.

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
  place from the memory-mapped GGUF; exact; a cold call ~0.25-0.3 ms per expert after the
  runtime's wait for the GPU (README steps 77-82: neither DRAM nor arithmetic alone sets it).
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
  next 5 tokens in ~8 ms. The main model then runs [last token, guesses] as one chunk:
  projections, the head and the expert block read their weights once for all tokens; attention
  runs per token in position order. The longest correct prefix plus the model's own next token
  is kept, the rest rolled back (window slots, compressor state). How many guesses to verify is
  chosen per step from the draft's confidence head, which turned out to be calibrated: the n
  with the most expected tokens per ms, with verify costs fitted during the run.
- **The next token** (`src/sampler.cpp`): the largest logit, or with `--temp` a draw from the
  softmax with top-p. The random number of a token is a hash of the seed and the token's
  position, not the next value of a generator, so one seed gives one output with and without
  speculation: the speculative loop draws the main model's token at each verified position
  and accepts a draft if it is that token.
- **Requests and the server** (`--requests`, `server/ds_server.py`): the engine reads a whole
  rendered prompt and its settings from stdin, rewinds its context to empty, and writes the
  reply's tokens as lines. The server in front converts an API request to the model's own
  message format and lets the model's `encoding.py` render the prompt (system block, tools,
  reasoning effort) and parse the reply (reasoning, text, tool calls). One request at a time;
  a signal cancels a request whose client left.
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

- **Stored outputs:** the held-out prompts' outputs are kept as hashes of their token ids
  (`results/record/reference/`); a run writes itself as a record (`--record`), and
  `scripts/run_matrix.py` runs prompts x configurations and compares. With `--canon` the output
  must be the stored one on any placement, with or without speculation or the cache.
- **Sampling and requests** are held to the same kind of gate: a sampled speculative run must
  equal the plain loop's with the same seed; a reply in `--requests` mode must equal a single
  run of that prompt, also after other requests, errors and cancels.

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
- **Sampling at temperature 1 did not lower the draft's acceptance** on the held-out prompts
  (2.80 tokens per step against 2.58 greedy; predicted: clearly fewer). The model is sure of
  most tokens, and where it is not, the draft was often wrong under greedy decoding too
  (README step 150).
- **129,280 scalar `exp` calls are a millisecond.** A draw from the whole vocabulary cost more
  than the copy of the logits it reads; a vector `exp` and stopping at the first rejected draft
  halved it (README step 151).
- **Two traces side by side name a cause in one line.** Runs that came out 10-14% slow were
  the engram rows read from disk, 1.2 -> 13 ms per step, three hours after a reboot; every
  other span was level (README step 155).

## 6. Where a token's time goes

Today, at sclk 6 with speculation and the cache (the page for this: [docs/timing.md](timing.md)):

    step  ~=  draft (~8 ms)  +  verify: ~38 ms for one token, ~+9 ms per token after it

A step yields ~2 tokens on chat and Chinese/Japanese and 3.3-4.5 on code and structured text.
A code prompt's 6-token verify step, 85 ms (the traces of README steps 105 and 155):

| part | ms |
|---|---|
| die 0's dense part (shared expert 8, attention's output 8, its core 6, its query 5, hyper-connections 5, the rest) | 48 |
| die 0 waiting for the CPU's experts | 27 |
| die 0 waiting for the other dies' experts | 5 |
| outside the layers (embedding, head, ids to the host, the cache's step) | 5 |

So the dense part on one die is over half of a step and the CPU's misses a third; on two
machines the CPU's part is gone and each half's layers take ~36 ms. Sampling adds ~0.5 ms per
draw, one per accepted draft and one more.

When this page was first written (sclk 2, before the cache) a plain step was 87 ms with every
expert on the CPU, 51 with the static placement, and ~43 with perfect knowledge of the
experts: the CPU's experts were the cost then, and the cache and steps 50-104 moved it to the
dense part.

## 7. What is next

Optimization is paused (the person, 2026-11-16); the work since is the engine as something
other programs can use.

1. **How the programs are organized.** One prompt, chat and the server are modes of one loop in
   `apps/ds_completion.cpp`, and the chat template exists twice (in C++ for `-i`, in the
   model's `encoding.py` for the server). Proposed, not decided: the request protocol as the
   one way to talk to the engine, chat as a client of it like the server, and for that a
   context kept between requests where their prompts share a prefix.
2. **The server:** tool results in a conversation, the OpenAI endpoint the benchmark's other
   script needs, a reply streamed as it is generated, sampling with `--spec` on two machines.
3. **Where the time is, when optimization resumes** (the list: [docs/plan.md](plan.md)):
   - **die 0's dense part** (~48 ms of a code step): attention over the XGMI pair 0-1 with
     the core split by heads (step 76 split only the output projections: +0.6%, not kept);
     the fp8 GEMV at 2-8 inputs needs a new work split (steps 65, 75);
   - **the CPU's experts** (27 ms of that step): steps 77-82 took a CPU call apart (+3.4%); a
     CPU weight layout was tried and not kept (steps 83, 91); on two machines they are
     nearly gone;
   - **two machines** (+14%): the halves take turns, so one machine is always idle; an
     overlap between the halves is to be estimated first;
   - **the draft:** a block of 5 tokens; on code the rule verifies all 5 on about half of
     the steps (step 146 estimated what a perfect draft-length rule would give).
4. **Long context** (steps 34-35): runs to 1M with `--max-ctx`, gated against the reference at
   20k tokens. Left: compressed caches in fp8/fp4 (far beyond 64k), the indexer batched over a
   verify step past 512 keys.

Explored, not planned: RDMA between machines, an MI210 for the dense part (~33 t/s estimated),
more hardware for the experts, and a hypothetical V4.1-Pro (engram on SSD is easy; experts
barely fit in two machines' RAM).

## 8. Open questions for review

- Which prompts represent real use? The placement and the cache's starting set depend on it.
- Which GPU clock level is acceptable for power (sclk 6 for every measurement since step 48; 4-6 is the sweet spot; level 8
  costs ~17-25% more energy per token for ~8-9% speed while CPU misses dominate)?
- Should chat need Python and the model's `encoding.py` (as the server does), or stay inside
  the C++ program?
- Is the remaining correctness risk acceptable: near-tie fp8 rounding flips make the top token
  differ from the reference occasionally (1 of 17 steps on the test prompt), in every
  implementation we compared, the llama.cpp fork included.

## 9. Code map

Libraries and programs are apart (README step 139): `src/` is the CPU-side library and `hip/`
the GPU runtime library, neither has a `main`; `apps/` has the programs one runs, `tests/` the
test programs, `bench/` the benchmarks. `scripts/` has the tools still in use; a script written
for one step lies with its results.

| path | what |
|---|---|
| `apps/ds_completion.cpp` | the CLI, top to bottom: the options (`Args`, `parse_args`), the record (`write_record`: `--record` writes a run as one JSON object, `docs/record.md`), the two generation loops (`generate_spec`, `generate_plain`), then `run`: load, the turns (one prompt, `-i`'s lines or `--requests`' requests), the summaries |
| `apps/think_view.h` | how a reply is printed: `--think-show`, the reasoning as one moving line or hidden (README step 147) |
| `apps/record.h` | what `--record` writes beyond the program's own variables: the id hashes, the GPU clock, the storage reads |
| `apps/ds_inspect.cpp` | load the model, or some layers, through the real loader and show what it found |
| `src/canon_dot.{h,cpp}`, `src/canon_math.h` | `--canon`: one specification of an expert's arithmetic that the CPU and the GPU kernels both reproduce bit for bit, so an expert gives the same bits wherever it runs |
| `src/fatal.h` | where the process is, for the handler that reports a fatal signal |
| `src/model.{h,cpp}` | GGUF loading (zero-copy mmap), the model's schema and hyperparameters |
| `src/forward.{h,cpp}` | the CPU reference path (ggml graphs), used by tests and `bin/ds-completion` |
| `src/cpu_moe.{h,cpp}` | the exact AVX-512 routed-expert kernel |
| `src/fakequant.{h,cpp}` | the reference's fp8/fp4 rounding |
| `src/sampler.{h,cpp}` | the next token from the logits: greedy, or temperature and top-p with a random number per (seed, position) (README steps 148-151); `tests/test_sampler.cpp` checks it |
| `apps/fp8pack.cpp` | lossless fp8 repack of the dense weights |
| `tests/ds_test.cpp` | per-layer and end-to-end tests against the goldens |
| `tests/test_fakequant.cpp`, `tests/test_topk.cpp` | the rounding ops and the indexer's selection against the reference's, bit for bit |
| `hip/rt.{h,cpp}` | the GPU runtime: state, layer order, the expert split |
| `hip/rt_peer.cpp` | the runtime as one half of a model on two machines: the front's messages, the back's `serve` (`hip/peer.h`: the protocol) |
| `hip/rt_trace.cpp` | the runtime's observability: the startup memory ledger, `--profile`'s groups, `--trace` (`hip/rt_internal.h`: the helpers the runtime's files share) |
| `docs/requests.md` | `--requests`: the protocol between the engine and a server's front (README step 152) |
| `docs/server.md`, `server/` | the HTTP front: `ds_server.py` (Anthropic Messages over `--requests`), and its test without a model (`test_server.py` with `fake_engine.py`; README steps 153-154) |
| `docs/rocprof.md` | the GPU side under AMD's `rocprofv3`: how to run it, its files, how to read them (README step 140) |
| `hip/kernels.{h,hip}` | the runtime's kernels (norms, attention, indexer, hyper-connections, router, dispatch) |
| `hip/fp8_gemv.{h,hip}` | exact fp8 x fp8 GEMV |
| `hip/fp4_moe.{h,hip}` | exact fp4 routed-expert kernel (4 launches) |
| `hip/expert_pool.{h,hip}` | expert slots per die, placement reading and assignment |
| `hip/expert_cache.{h,cpp}` | the adaptive cache: which experts to swap in and out, from recent use |
| `hip/expert_loader.{h,cpp}` | the background loads into pool slots, one thread per die |
| `hip/dspark.{h,cpp}` | the DSpark draft (3 layers, Markov and confidence heads) |
| `hip/peer.h` | two machines: the messages between the front and the back half (TCP) |
| `bench/bench_*` | benchmarks of each kernel and of the loader, CPU and GPU (the Makefile knows which need `HIP=1`) |
| `ref/` | the Python reference, golden export, weight checks, logits comparison |
| `scripts/` | the tools still in use ([scripts/README.md](../scripts/README.md)): the runner of a matrix of runs (`run_matrix.py`) and the reader of their records (`records.py`; README steps 128-129), the chat launcher (`chat.sh`), the held-out measurement (`spec-measure.sh`) and the two-machine one (`split-measure.sh`), the trace tools, `route_stats.py`. A script written for one step lies with its results under `results/` (README step 130) |
| `placements/` | ranked expert placements (`study30.txt` is the default) |
| `results/` | raw outputs of every measurement quoted in the steps ([docs/reproduce.md](reproduce.md): which command produced which) |
