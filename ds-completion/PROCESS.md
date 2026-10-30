# How this was done: method and decision log

This is the story of the project as a sequence of decisions: at each point, the question we
asked, what we expected, what we measured, and what we decided. It is meant for someone who
wants to learn the *approach* (how to make a large model fast on odd hardware without losing
correctness), not only the result. Details and every number: [README.md](README.md) (the lab
notebook, steps 1-30). Plans and cost models as they evolved: [PLAN.md](PLAN.md). How to rerun
everything: [REPRODUCE.md](REPRODUCE.md). A 20-minute summary: [OVERVIEW.md](OVERVIEW.md).

Sources: the repository (docs, scripts, results, 50+ commits) and the visible messages of the
working sessions (a person directing, an AI assistant, Claude, implementing and measuring), 2026-10-26 to -29.

## 1. The method in ten rules

Each rule is followed by where it mattered most.

1. **Build the correctness ladder before any speed work.** A slow but trusted reference, gates
   per layer and end to end, and a rule for every later change: bit-identical where only
   placement or batching changed, a stated tolerance where kernels changed.
   (README "Correctness ladder", steps 2-3; every step since re-runs the gates.)
2. **Verify your inputs, not only your code.** The local safetensors turned out to be a
   different checkpoint from the GGUF; we proved which one the GGUF was by reading byte ranges
   of the official release over HTTP. (Step 1, "Which checkpoint".)
3. **Prefer exactness that hardware makes cheap.** gfx906 has no fp8 or fp4, but those values
   become fp16 exactly and fp16 products are exact in fp32. So the GPU computes the reference's
   quantized math exactly instead of approximately. Exactness later paid for itself: it made
   batching, speculation and placement-invariance testable as "the same bytes". (Steps 5, 7, 28.)
4. **Measure ceilings before optimizing.** Before building expert placement, put whole layers
   of experts on the GPUs (no statistics needed) to see what moving work there buys; before
   building a cache, run a perfect-knowledge (oracle) placement as the upper bound.
   (Steps 8, 15.)
5. **Keep a cost model and attack its largest term.** Every planning point started from "where
   does a token's time go", measured (profiles, single-kernel benchmarks) and extrapolated from
   unit costs (an expert: 0.27 ms on the CPU, ~0.065 ms on a GPU, ~1.4 ms to copy).
   (PLAN "Current cost model", OVERVIEW section 6.)
6. **Measure the cheap unknown that decides the design.** The routing study (30 prompts, a few
   hours, no runtime changes) decided between a fixed hot set and an adaptive cache before
   either was built. (Step 13.)
7. **Simulate before building, and calibrate the simulator on reality.** The cache was
   simulated with load rates, latency and speculative chunks, with a time model fitted on
   measured runs (and checked against an independent measurement when the fit was confounded).
   (Step 29.)
8. **Isolate risky mechanisms and measure them alone first.** The load path was benchmarked
   standalone, then live with loads into slots nothing reads, before any cache logic existed.
   (Step 30.)
9. **Make every risky optimization prove "nothing changed".** Speculative decoding must equal
   plain greedy byte for byte; with canonical numerics any placement must equal the CPU-only
   run byte for byte; background loads must not change a byte. (Steps 20, 28, 30, phase 3.)
10. **Record negative results and surprises as carefully as wins.** HIP graphs did not help,
    32 threads were slower than 16, pre-decoded activations were slower, a faster clock did not
    help speculation more. (OVERVIEW section 5; README "tried and measured no better" notes.)

## 2. How the collaboration worked

- **The person** set goals and priorities (correctness, clarity, performance; a CLI, one
  sequence), asked the questions that steered the work (several of the turning points below
  started as "I wonder if..."), did everything needing hands or privileges (installing the
  Infinity Fabric bridges, setting GPU clock levels with sudo, the Python environment,
  downloading models), and decided when to commit.
- **The assistant** read the code and papers, implemented, ran the measurements, reported
  numbers with their uncertainty, and proposed the next step with an estimate.
- **The cadence:** one step at a time; "commit, then do X"; docs updated with every commit;
  plans written down before implementing (PLAN.md) and revised when measurements disagreed.
- **Hypotheses were written before measuring** where possible, so the result could prove them
  wrong (section 4 lists predictions against outcomes).

## 3. Decision log

### Prerequisites: the machine (2026-10-26, separate repositories)

- **Q:** Do the newly installed Infinity Fabric (XGMI) bridges work under Linux? **Found:**
  after reseating, two pairs (dies 0-1, 2-3), not one group of four; ROCm crashes on cross-card
  runtime copies. **Decided:** within a pair GPU-to-GPU, across cards through host memory.
  (`macpro-2019-vega-if`.)
- **Q:** How fast are skinny matmuls (batch 1-8) on gfx906 with ggml? **Found:** quantized
  GEMV at ~45-60% of HBM bandwidth; a ceiling probe at ~880 GB/s per die.
  (`llama-sandbox/mlp-matmul`.) This set expectations for everything below.

### Phase A: correctness first (steps 1-3)

- **Q:** Can the official Python implementation be run at all, to serve as a reference?
  **Problem:** its kernels (tilelang) do not run on gfx906, and the model is far larger than
  any GPU. **Decided:** a pure-torch CPU version of its kernels, run one layer at a time,
  decoding only the experts actually used; golden activations per layer and logits end to end.
- **Q:** Which checkpoint is the GGUF? **Found:** the local safetensors are the base release,
  the GGUF is 1017; proved by HTTP range reads of the official 1017 files. **Decided:** the GGUF
  is the source of truth; references read 1017 weights remotely when needed.
- **Q:** Should our CPU forward emulate the reference's fp8/fp4 fake quantization?
  **Measured:** KL 1.8e-2 with it vs 3.0e-2 without. **Decided:** emulate it exactly.
- **Result:** CPU-only 3.47 t/s, the same as the llama.cpp fork, matching the reference per layer.

### Phase B: the first GPU path and its ceiling (steps 4-9)

- **Q:** Dense part on one GPU through ggml, experts on the CPU? **Predicted:** >10 t/s with
  overlap. **Measured:** 6.0 t/s; GPU and CPU take turns, the GPU part is half launches.
  **Decided:** write a plan before more code (PLAN.md: experts, fp8 repack, tensor parallelism,
  fusion, speculation, prefill), starting with what does not depend on routing.
- **Q:** Can the dense weights go back to fp8 losslessly and be computed exactly?
  **Found:** yes (repack verified byte-wise; an exact fp8 GEMV, 311 us per layer vs ggml's 832).
- **Q:** Own the decode path or stay in ggml? **Decided:** own it (every op in our HIP kernels
  on one die): 9.4 -> ~10 t/s, deterministic, and later ggml's GPU path was removed.
- **Q:** An exact CPU expert kernel? **Built:** AVX-512 fp4 x fp8, exact (ggml's re-quantizes
  activations to int8, ~1e-2 off).
- **Q (the person's):** Can we test a ceiling by loading whole layers of experts onto the GPUs,
  without routing statistics? **Measured:** 13 layers -> 11.4 t/s; each layer moved saves
  ~1.4 ms. Extrapolated: all 40 layers -> ~20 t/s. **Learned:** experts cost 5x more to copy
  than to compute on the CPU, so experts can only help from GPUs if already resident.
- **Q:** What does the GPU clock buy? **Measured** (levels 0-8, set by hand): the dense part
  38 -> 29 ms; most of the gain by level 4-6.

### Phase C: re-prioritizing from measurements (steps 10-13)

- **Q (the person's):** With these numbers, should tensor parallelism, expert parallelism or
  CPU/GPU work-sharing come first? **Reasoned from the cost model:** tensor parallelism saves
  ~8-10 ms of a ~100 ms step, so it waits; splitting each layer's experts between GPUs and CPU
  concurrently makes the step cost the slower side, not the sum; its value depends on how
  concentrated routing is, which nobody had measured.
- **Decided:** free GPU memory first (prompt through the runtime, drop ggml's dense copies:
  -14 GB on die 0), then measure routing.
- **Q:** Do HIP graphs remove the launch overhead? **Measured:** 0.1-2 us per launch on this
  ROCm: no. Only fusion reduces launches.
- **Q:** Routing study (30 prompts, 9 categories): how much do hot experts catch, and does a
  hot set transfer between prompts? **Found:** in-sample 80% at 128 GB, but only 71% from
  other prompts (code, math, Chinese worst); an LRU cache warm-started from other prompts
  reaches ~95% with ~11 loads per token. **Decided:** the end state is an adaptive cache, built
  in stages, starting from a fixed set (study30).

### Phase D: experts on all GPUs, and the oracle bound (steps 14-18)

- **Built:** expert pools (slot per expert), a ranked placement file, a dispatch kernel that
  splits each token's experts into per-die and CPU lists, all running concurrently, summed in a
  fixed order. Gate: four placements (all CPU, all GPU, random, real) pass per layer.
- **Measured:** study30 17.7-21.7 t/s (at 16 threads).
- **Q (the person's):** What is the best caching could ever do? **Built:** an oracle placement
  from the prompt's own routing. **Measured:** only ~16 t/s with 97% of experts on GPUs, far
  below the ~24 predicted. **Learned:** the expert path itself was serialized. Three fixes
  followed, each measured: no die waits for the host; a faster kernel for few experts; and the
  largest, **16 CPU threads instead of 32** (32 starved the host thread and the HIP runtime):
  every configuration +25-50%, oracle ~25 t/s.

### Phase E: speculative decoding (steps 19-27)

- **Q (the person's):** A DSpark draft model was released for 1017; can it work here?
  **Estimated:** with study30 it barely pays on chat (~22 vs ~20 t/s) because CPU misses grow
  with the number of verified tokens; with a good cache it is the biggest lever (~30-40).
  **Decided:** first measure acceptance in *shadow mode* with our own exact numerics (not the
  fork's numbers): 2.95 / 5.10 tokens per main step (chat / code).
- **Plan:** batched verification must be **bit-identical** to plain greedy decoding. Phase 0
  proved the fp8 GEMV gives each token the same bits at 2-8 tokens before anything else was
  built.
- **Built in order, each measured:** the speculative loop (output identical); a chunk forward;
  a batched expert block (verify step 267 -> 207 ms); a faster draft (26 -> 11.7 ms); batched
  hyper-connection kernels (207 -> 181 ms); a multi-token fp8 kernel (181 -> 171 ms).
- **Surprise:** batching does not make GEMVs free on gfx906: at 6 tokens the fp8 GEMV cost 3x
  one token, bound by arithmetic (decoding activations per row), not memory.
- **Q:** How long should drafts be? **Found:** the draft's confidence head is calibrated as is
  (sigmoid(score) = acceptance rate, AUC 0.875). **Decided:** choose the draft length per step
  by expected tokens per unit time, with a verify-cost line fitted online.
- **Measured on 8 held-out prompts:** +7% to +33%, geometric mean +22%; with the oracle
  placement up to +71% (code). Output identical to plain greedy on every prompt.
- **Hypothesis (the person's):** more GPU compute (higher clock) should help speculation more
  than plain decoding. **Measured (sclk 0/2/8):** +8% plain, +9% speculative from level 2 to 8:
  no. **Why:** CPU misses (~70 ms of a 170 ms verify step) do not scale with the GPU clock.
  This pointed straight at the cache.

### Phase F: the adaptive expert cache (steps 28-30, phase 3 in progress)

- **Q:** What could go wrong with a cache that changes what runs where while decoding?
  **Found:** an expert's bits depend on where it runs (GPU and CPU kernels sum in different
  orders), so output would depend on load timing, and speculation would stop matching plain
  decoding. **Options:** (A) make expert numerics placement-invariant, (B) make the cache
  schedule deterministic. **Decided:** decide by measurement, A if it costs under ~15%.
- **Q (the person's):** Is aggressiveness a setting rather than a constant? Rarely updating a
  placement should beat never updating it. **Decided:** load budget, admission rule and
  rebalance period become knobs; 0 = today's static placement.
- **Phase 0, measured:** both sides compute exact integer block sums (VNNI on the CPU,
  v_dot4_i32_i8 on the GPU) with one fixed conversion and summation tree: GPU = CPU bit for bit,
  CPU +5%, GPU +13% (later +2%). **Decided: A.** The fast kernels stay the default
  (the person's request: test the approach, keep the fast path).
- **Phase 1, simulated:** routing logs now include verify chunks and rejected drafts; a time
  model fitted on 16 measured runs (4.8% rms). **Found:** the load rate (loads/s, not per step)
  is the knob; frequency-based rebalancing beats LRU by 2.5-5.5 t/s at limited rates; rejected
  drafts do not pollute; held-out `--spec` 18.8 -> ~29 t/s at 100 loads/s. **Lesson on method:**
  the first fits were confounded (chunk size vs misses) and predicted nonsense at the extremes;
  pinning the GPU cost from the oracle runs' per-step data fixed it.
- **Phase 2, measured:** copying straight from the registered model mapping is ~8x cheaper for
  the CPU than through pinned memory; live, 100 loads/s cost ~1% with identical output. Two
  bugs only measurement found: an unbounded copy queue stalls a die's other copies whatever the
  stream priority; spinning waits in loader threads cost the CPU kernel 5%.
- **Phase 3a, built and gated:** `--canon` in the runtime (each expert writes its own row, one
  kernel sums shared + rows in router-rank order; the host copies back only the rows each die
  computed, sized from the dispatch counts it already has). **Gate:** the same bytes with all
  experts on the CPU, two placements, three dies only, and `--spec` at two placements
  (results/canon-gate.txt). Batching the canonical GPU kernel over experts also cut its cost
  from +13% to +2%. The end-to-end speed cost against the fast kernels: phase 3b below. **Decided:** build the cache manager (3b) on top.

- **Phase 3b, built and measured:** the cache manager (frequency rebalance, loads only once the
  expert memory is registered, table changes only between forwards). First live run: canonical
  numerics cost 9-11% end to end, far above the kernel benchmarks' +5% / +2%. **A profile found
  why:** the canonical CPU path prepared its inputs on one thread, which dominates with the
  runtime's usual 1-2 misses per layer (the benchmark used 6), and the combine kernel ran one
  block per token. Fixed: canonical costs 1-2%, and its CPU kernel is now faster than the fast
  one at 1-2 experts. **Measured** (held-out, `--spec`): fast static 19.97 -> 25.55 t/s with the
  cache at 200 loads/s, every output identical; at sclk 8 22.19 -> 29.09.
- **Hypothesis revisited (the person's, step 27):** a higher clock helps speculation more. Without
  the cache: no (+9% vs +8%). With it: partly (+14% for `--spec`), because the cache turned CPU
  misses into GPU work.
- **A bug in the measuring method:** compiling during a timing run (CPU-bound timings): the run
  was stopped and redone.
- **Q (the person's):** do we support the model's full context? **Found** (against the official
  code): everything implemented matches at any position, but the runtime stops at 4,096 tokens
  and the candidate-block pre-filter (>16k) is missing. **Decided:** long context before any
  cache tuning (PLAN "Long context").

- **Q (the person's):** with the cache and speculation, where does a step go now? **Measured**
  (6-token verify steps, sclk 8): the dense part ~64%, CPU experts ~20% (77.6 -> 24.8 ms), the
  draft ~10%, GPU experts ~3% visible. **Decided:** record the next levers by part (PLAN
  "Further optimization ideas": cache tuning, the dense part and TP, speculation, the draft),
  then long context first.

- **Long context, steps 1, 2, 4 (README step 34):** the reference's selection rules were read
  first (sorted top-k indices; candidate blocks with the newest block pinned), then built as one
  radix-select kernel used twice, then **gated against the reference's own functions** on the
  same scores before any end-to-end run (46 of 46; the one apparent mismatch was a tie at the
  512th value, where torch's choice is unspecified, so the check now detects ties itself). End
  to end: 7.4k and 29.5k-token prompts give accurate summaries at 17.5 / 17.2 t/s. **Open:** a
  numerical end-to-end gate past 16k needs a very long reference run.

- **The long-context gate (README step 35).** **Predicted:** 6-10 h for a 20k-token reference
  run (scaled from 22 minutes for 1,100 tokens). **Measured:** 42 minutes; the short run was
  fixed cost. Memory was watched while it ran (it rose to ~500 GB: freed heap the allocator
  kept, identified from the process's memory map, not a leak), with a guard to stop it before
  the system ran out. **Result:** 9/9 greedy tokens, KL at the short-context level, per-layer
  errors as at 1,100 tokens. The indexer overlap fell (0.97-0.99 -> 0.85-0.94); rather than
  explain it away (at 1,100 tokens 512 of 550 were picked, so overlap was nearly free), the
  reference was rerun recording its own scores, and every disagreement was ranked against them:
  all just past the cutoff, gaps under 1% of the top-512 range in the median. **Caught on the
  way:** the per-layer tool silently skipped most columns of a `--keep-last` golden in runtime
  mode (blank cells, not wrong numbers); fixed before reading any result.

- **Q (the person's):** a short prompt that reasons for long: can it come from a file, and is
  the reasoning effort controlled? **Found:** the official encoding takes an effort of 1-100
  (default 75, the value `--think` had fixed). **Done:** `-f` and `--effort`, checked against
  `encode_messages` (same tokens). **Noted:** a slow first prompt after a memory-heavy job is the
  model's pages coming back from disk, not the runtime (README step 36).

- **Batched prompt processing (README step 37).** Order chosen by payoff for the person: prompt
  speed was the practical limit of long context. **First the cheapest change:** prompt chunks
  through the existing verify path (bit-identical per token): +26% / +55%, every output the same.
  **Then a profile before designing more:** CPU experts were 53% of the time and did not
  amortize. **Predicted:** chunks of 64 would share expert reads. **Measured:** no change; with
  384 experts, 64 tokens give each CPU expert about one token. **Then** chunks of 512: still no
  change, and a kernel benchmark showed why: batched, the kernel was compute-bound, not reading
  the weights; tiling 4 rows x 4 tokens fixed it (0.27 -> 0.086 ms per expert-token at 16
  tokens, bit-identical). **Caught on the way:** a "difference" in the fast path that was the
  expert placement changing with buffer sizes (the reason canonical numerics exist): the gate
  compares with `--canon` across builds. And a separate stall: expert registration blocks the
  GPU for ~90 s (step 36's slow prompt).

- **CPU/GPU overlap (README step 38).** The profile showed the two sides taking turns, each about
  half the time. **Predicted:** up to ~2x by overlapping halves of a chunk. **Measured:** 1.5x;
  the overlap is nearly complete (the GPU waits 6 of 40 s), but each half gives the CPU half the
  tokens per expert, so the CPU's own time grew slightly; both sides are now equal, which says
  the next gains need both (GPU kernels and CPU sharing). The restructuring kept every output
  identical (gate: with and without overlap, and canonical against the token-by-token build).

## 4. Predictions against outcomes

Estimates were made before measuring wherever possible; this is how they fared.

| step | prediction | outcome | why the difference |
|---|---|---|---|
| dense on GPU via ggml (4) | >10 t/s with overlap | 6.0 t/s | no overlap; half the GPU time was launches |
| whole layers on GPUs (8) | all 40 layers -> ~20 t/s | 13 layers 11.4 t/s, consistent | |
| per-layer split, study30 (14) | ~18 t/s | 17.7-21.7 t/s (after fix 3) | |
| oracle placement (15) | ~24 t/s | 16 t/s, then ~25 after three fixes | the expert path was serialized; 32 threads starved the host |
| speculation with study30 (19) | chat ~22 vs ~20 t/s | +7% to +33%, gmean +22% | draft length chosen per step (phase 4) |
| higher clock helps speculation more (27) | yes (the person's hypothesis) | no: +8% plain, +9% spec | CPU misses do not scale with the GPU clock |
| canonical numerics cost (28) | within 15% | CPU +5%, GPU +13% (+2% after batching) | |
| load cost at 100 loads/s (30) | 2.7% (the simulation's pessimistic case) | ~1.1% with direct copies | |
| canonical cost end to end (32) | ~5% (kernel benchmarks) | 9-11%, then 1-2% after two fixes | benchmarks used 6 experts; the runtime's CPU gets 1-2 |
| cache, `--spec`, 100 loads/s (29 -> 32) | ~28.9 t/s, 87% hits (1,024 tokens, no canonical cost) | 24.5 t/s, 80% hits (300 tokens) | shorter runs adapt for a larger share of the run |
| CPU/GPU overlap of two half-chunks (38) | up to ~2x | 1.5x | half the tokens per CPU expert; both sides now ~33 s |
| chunks of 64 share CPU expert reads (37) | most of the 53% saved | nothing | 384 experts: ~1 token per CPU expert per chunk; then compute-bound until tiled |
| reference run, 20k tokens (35) | 6-10 h (scaled from 1,100 tokens) | 42 min | the 1,100-token run was mostly fixed cost |

## 5. Mistakes and how they were caught

- **Wrong checkpoint** (local safetensors vs GGUF): caught by per-layer gates disagreeing,
  confirmed against the official files.
- **Thread count**: 32 threads looked natural for 32 logical CPUs; the oracle bound falling
  short of its estimate exposed it.
- **Comparing speculative output to the wrong baseline**: the draft's memory changes the
  placement, so plain and speculative runs must both load the draft (until canonical
  numerics, which remove the dependence).
- **A confounded simulation fit**: caught by checking the model at the extremes (it predicted
  speculation slower than plain at full hits) and against independent per-step data.
- **Unbounded asynchronous queues** and **spinning waits**: caught only by measuring under
  load, not by unit tests.
- **False alarms from the measuring scripts** (a trailing newline in a prefix comparison):
  every "differs" was investigated before being believed or dismissed.

## 6. What a reader should take away

- Exactness and determinism are not a cost here; they are what made aggressive optimizations
  (batching, speculation, moving experts at runtime) safe to try and easy to verify.
- The largest wins came from measurements that changed the plan (routing transfer, the oracle
  bound, the thread count, the clock result), not from the first idea.
- Simple models of where time goes, kept up to date and checked against measurements, chose
  almost every next step.
