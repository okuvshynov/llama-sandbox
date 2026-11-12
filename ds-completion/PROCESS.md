# How this was done: method and decision log

This is the story of the project as a sequence of decisions: at each point, the question we
asked, what we expected, what we measured, and what we decided. It is meant for someone who
wants to learn the *approach* (how to make a large model fast on odd hardware without losing
correctness), not only the result. Details and every number: [README.md](README.md) (the lab
notebook, steps 1-83). Plans and cost models as they evolved: [PLAN.md](PLAN.md). How to rerun
everything: [REPRODUCE.md](REPRODUCE.md). A 20-minute summary: [OVERVIEW.md](OVERVIEW.md).

Sources: the repository (docs, scripts, results, 100+ commits) and the visible messages of the
working sessions (a person directing, an AI assistant, Claude, implementing and measuring), 2026-10-26 to 11-07.

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

### Phase F: the adaptive expert cache, and the decode work after it (steps 28-83)

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

- **fp8 matrix multiply (README step 39).** The constraint first: bit-identical results, so the
  kernel keeps the GEMV's per-output order and is free only in tiling. A sweep, then two fixes
  found by reading the kernel (idle lanes in short K tiles) and the compiler's resource report
  (registers: pipelining made it slower). **Predicted** ~3 ms per prompt token saved on the GPU;
  **measured** ~3.5 ms (34 -> 27 s), but end to end only +10%: with the GPU faster the CPU's
  experts set the pace. Tuning the kernel further waits until the CPU side is faster.

- **Q (the person's):** at what chunk size does moving the CPU's experts to the GPUs beat
  computing them on the CPU? **Estimated** from measured unit costs (no code): a load is a fixed
  ~0.7 ms per expert, the CPU ~0.09-0.17 ms per expert-token, so ~4-8 tokens per expert, i.e.
  chunks of ~500-1,000; the loads hide behind the GPU's work from ~450. **Decided** (the
  person's): document it with all the options (PLAN P1-P9), then first the pure optimizations
  (kernel tuning: P8, P5, P6) before restructuring.

- **P8, kernel tuning (README step 40).** Each knob tested for speed and bit-identity together
  (0 differences in ~400 configuration x tensor runs). Hypotheses tested and rejected: that
  decoding the inputs per block was the limit (decoding once globally: not faster), that shared
  memory capacity was (scales moved out: mixed). Kept only what helped (a register cap for the
  narrow shapes); the rest was removed from the code and recorded in the results file, so the
  reasoning survives without the dead code. Stopped at diminishing returns, since end to end the
  CPU sets the pace.

- **P5, the canonical CPU kernel (README step 41).** **Predicted** from the fast kernel's history:
  the same flaw (one row at a time, activations re-read per row), so tiling should help; ~25% at
  most, since its integer formula costs more per product. **Measured:** 0.17 -> 0.14 (~20%), and
  an L1 capacity effect found on the way (16 tokens slower than 8) fixed by blocking the tokens.
  End to end the canonical prompt gained the same ~20%, since there the CPU is the limit.

- **Docs check (the person's request).** QUICKSTART, the README status, OVERVIEW and PLAN still
  described the prompt as token by token, the cache's startup as ~30-40 s (it is ~40-90 s and
  stalls the GPUs) and `--canon` as ~2% slower (true for generation, not the prompt), and the
  "fastest" command was not the fastest. QUICKSTART now gives the fastest settings per job
  (long generation: `--spec --canon --cache-loads 200`; short runs and long prompts: `--spec`
  without the cache and `--canon`).

- **P1, streaming the experts: measure before building (README step 42).** The person asked about
  the design (does parallelism matter; TP2 over the XGMI pairs?). Answered from the structure
  (experts are independent: no TP on the expert side; the dense part is where TP2 or a pipeline
  would go), then measured what the estimate had assumed: real routing of two long prompts.
  **Found:** a long document uses fewer experts with more tokens each than the per-token average
  (10 vs 6 at 512), which favours streaming. **Calibrated** the model against a profiled run
  of the same prompt before trusting it (the CPU model was 1.65x optimistic), and tried a lower
  load rate for sensitivity. **Decided:** the hybrid at today's chunk size; P3 first.

- **P3, the registration stall (README step 43).** Three hypotheses, each tested by a probe in
  a standalone benchmark before changing the runtime: HIP calls blocked (no), page faults blocked
  (no), mmap/munmap blocked (yes, seconds). The runtime's own profile had pointed at the host
  (microsecond GPU groups taking 20+ s), which is what made the third hypothesis worth testing.
  Also found by the benchmark: a copy may not span two registrations, which the fix (pieces)
  had to handle. Checked after: no stall, loads still work and are as fast, output unchanged.

- **P2, streaming during the prompt: tried and dropped (README step 44).** Built behind a flag
  and kept correct from the first run (canonical output byte-identical), so every stage was a
  pure speed question. The model from step 42 had used the standalone load rate; in the runtime
  the loads were 3x slower and the instrumentation (time from submit to done, time waited after
  the CPU's share) showed it at once. Four fixes were tried one at a time against that number
  (home die out, loader pipelining, fewer CPU threads, stream priority); one helped. **Decided
  (the person's):** too complicated for its gain; removed, documented with its measurements.
  **Lesson:** a unit cost measured alone (the load rate) does not carry over to a resource
  shared with the work it overlaps; measure it in place before building on it.

- **The CPU kernel's in-place overhead (README step 45).** The cheapest option on the list,
  chosen after P2. Instrumented first (time per phase): the overhead theories (grouping,
  barriers) were wrong, the kernels themselves were slow. Then each difference between the
  benchmark and the runtime was removed one at a time: the workload (a replay of the runtime's
  exact pairs), the overlap, thread placement, and the build flags, which were it. **Lesson:** a
  benchmark is only a proxy if it is built the same way as the program it stands for; the
  runtime had been 1.6x slower than its own kernel's benchmark since step 37. Checked after:
  every gate and output byte-identical.

- **P4, chunks of 1,024 (README step 46).** Predicted from step 37's reasoning (more tokens per
  weight read); measured slower. The premise had changed with step 45: the kernel became
  arithmetic-bound, so the per-pair saving was ~6%, outweighed by a coarser overlap. Reverted;
  a reminder that an option's expected gain must be re-derived after the profile changes.

- **Q (the person's): does speculation need momentum, or does the draft handle it?** Answered by
  a test on existing logs rather than by argument: does past success add predictive power beyond
  the confidence head, judged on prompts left out of the fit (so an in-sample improvement from an
  extra parameter is not mistaken for signal)? It does not. Recorded as a checked-and-rejected
  idea, with what is still unchecked (long runs, the cache).
- **TIMING.md (the person's request):** a one-page "where the time goes", for a reader who wants
  the feel of the numbers (what a millisecond is worth, what-ifs) before the details.

- **The live speculation measurement (README step 48).** Planned to answer "does it adapt?"
  (calibration, behaviour over time, distance to the best choice per step, momentum), and it
  did. It also turned up two things nobody asked about:
  - **A repetition loop.** I first called it greedy behaviour without checking; the person
    asked whether it was a bug. The checks then separated the parts: plain vs speculative (same
    bytes), old vs new build (same bytes), and the independent CPU path on the same context
    (also loops). On the way, a prefix ending in a space looked like evidence against the loop
    until the token boundary was fixed. **Lesson:** "it's the model" is a claim to test, and the
    referee must see exactly the same tokens.
  - **A speed regression since step 37,** visible only because the per-prompt numbers were
    compared with step 32's. Separated from run-to-run noise (up to ~20% between single runs) by
    repeating old and new builds three times each, then traced to a per-step copy sized by the
    largest chunk. **Lesson:** the byte gates protect correctness, not speed: a change that
    grows a buffer must also be checked for the copies sized by it, and generation speed belongs
    in the checks after any change to shared code.

- **A fresh decode profile before choosing (README step 49).** Step 33's breakdown was four
  builds old; the person asked for the ideas not yet explored, and the answer depended on
  numbers I did not have (launch overhead was a guess). The installed profiler could not trace
  the runtime, so the two missing numbers were measured directly: a launch counter (LD_PRELOAD
  shim) and a launch microbenchmark. **Found:** the CPU's misses had grown back to the largest
  item on code (step 33 measured chat); launches matter less than the count suggested.

- **Ask what a cost depends on before optimizing it (README steps 50-52).** Two plausible ideas
  were checked by measurement before any build: the person suspected one miss per layer would
  hide behind the GPU's work and several would not; a per-layer histogram showed the cost is
  linear and almost nothing hides, which pointed back to the hit rate (the cache tuning: +5.6%).
  The person proposed prefetching the next layer's likeliest CPU expert; a bar was set first
  (~60% right), the predictability was logged in the existing profile, and it came out ~20-28%:
  not built. A trace (`--trace`, Perfetto) was added because the overlap question was easier to
  see than to tabulate. The baseline run separating the kernel gain from the cache gain was lost
  once to a waiting loop whose `pgrep -f` matched itself (as `pkill -f` had before).

- **Look at the picture, then ask what the slow thing is (README steps 53-54).** The trace made
  the router stand out (as slow as an expert); its size said it should not be, so the cause had
  to be structure: one single-wave launch per token. The person then asked whether co-selected
  experts form clusters to place by; the analysis set its own bar (a perfect balance) before any
  build and showed at most 1-3%, and its cost fit exposed the real GPU-side item (an expert's
  reads are not shared across its tokens). A cold page cache after a reboot was caught by a
  mixed result (two prompts faster, three slower, outputs identical) and `fincore`.

- **A fit is not a mechanism (README step 55).** Step 54's cost fit (~30 us per pair, ~0 per
  distinct expert) was read as "the weights are read once per pair"; the bit-identical grouped
  kernel showed otherwise within two benchmarks, and two cheap probes (more rows per lane group,
  decoding once) ruled out the obvious alternatives. With no profiler for this runtime the next
  step would have been guessing, so the attempt was reverted and kept as a patch with its numbers.

- **Make the benchmark look like the runtime before trusting it (README step 56).** The static
  schedule won 15-22% back to back; adding the runtime's idle gap reversed it, CPU settings
  restored it, and the runtime still showed nothing. Each step was cheap and kept the result
  honest; the change, the runtime flag and the settings were all removed, and the benchmark mode
  that found it stays. The person applied the sudo settings one at a time, as with the GPU clock.

- **Split a cost before naming it (README step 57).** Step 56 called the one-expert call's extra
  0.25 ms "fixed per-call cost" and listed wake-ups and barriers; splitting the call into its
  phases next to the one-token kernel and an empty parallel region showed neither: one phase on
  the wrong routine. The fix was two lines; the end-to-end gain (+1.1%) was a fraction of the
  per-call one, as the layers without CPU experts predict.

- **Check a comparison's knobs are matched (README step 59).** Decay first looked worse than the
  window; the candidate threshold ("at least 2 uses") means more for decayed scores. Matched, the
  two were the same. The one knob that moved hits, the threshold, was then settled live, where
  the simulator's free-vs-charged loads could not.

- **Measure the budget before planning to shrink a part of it (README step 60).** The next
  cache lever was assumed to be fp8 compressed caches; a ledger from the driver's own numbers
  (per startup phase, three context lengths) showed them at 0.04-0.43 GB and the draft at 8 GB.
  The person asked for the visualization first, which is what found it.
  The ledger's own follow-up then made the mistake it had caught: the draft's 8 GB was called
  bf16 from its file's name; reading the loader (step 61) showed fp8 and mxfp4 already.

- **Profile the part before parallelizing it (README step 62).** The person proposed expert
  parallelism for the draft; its existing per-part profile showed the experts at 16% and the
  output head at 27%, and the head's slowness was the same L2-bound pattern as before: inputs
  re-read per wave. The fix kept every row's arithmetic, so the gate was one benchmark mode.

- **A good predictor is not yet a good policy (README steps 63-64).** The person's idea held up
  (the next layers' experts guessed 4-7x better than by counts), and the simulation said what it
  is worth: the bound for perfect guesses is 10%, the policies reach a quarter of it. Two of my
  own errors were caught on the way: the first tally read the cache's residency after the step
  had published its loads, and the first simulation queued loads that could not land in time
  (deadline-blind), which made even the perfect predictor look bad. And `pkill -f` matched its
  own shell again: stop a process by its PID.

- **Unblock before batching (README step 66).** The attention core could not be batched while
  the window ring was exactly the window; the first stage changes only the ring and proves it
  exact (prompt gate, long context with `--canon`, held-out prompts), so the later stages start
  from a known-good base. A fast-mode difference was explained (4 fewer expert slots) before
  being accepted, by checking the placement in both runs.

- **Stages that each prove themselves (README steps 66-68).** The plan put every risk in its own
  stage with its own gate (the ring, then the phases, then the launch), and stage 3 carried a
  check mode comparing each batched token against a one-token launch (0 of ~260,000 differ).
  Stage 2 gained 2% before the main change. Stages 2 and 3 share a commit: stage 3 was written
  in the same files before stage 2 was committed.
  The traces were then regenerated (README step 69) and compared span by span with the old ones:
  the attention core is the only span that moved by more than ~1 ms, which is what the change
  claims and nothing else.
- **Revisit dropped ideas when the profile shifts (README step 70).** After the attention work
  the person asked to re-score every decode idea, the dropped ones included. Two moved up (a
  CPU-cache prefetch, now with a better predictor; attention over two dies, now that attention
  is a third of a step), one appeared that no summary had separated (the CPU's handoffs, ~8%),
  and a one-line probe of an untried one (verify steps in halves) was exact but 10-13% slower:
  measured in minutes instead of planned for days.
- **Moving work is not removing it (README step 71).** The first fix for the CPU's handoffs
  moved the dies' copies earlier and the tail shrank in the trace, but nothing got faster: the
  HIP calls that issue them (~10-15 us each) had moved in front of the CPU's experts. A
  temporary host timer showed it (189 us before the CPU starts); issuing the copies while the
  host is ahead of the GPU, with a bound of rows instead of the counts, removed it. A single
  prompt's t/s also misled once more: the draft-length rule took 66 steps instead of 64, hiding
  a 2.7% faster step; the 8 prompts and ms per step settled it.
- **One ruler for the whole history (README step 72).** The person asked for the held-out speed
  per step. Only steps 48-71 shared a clock and prompt set; the earlier numbers mixed sclk 2 and
  8 and other settings, so they are rerun from their commits rather than adjusted by a guessed
  factor. Each old commit was built and smoke-tested first, so the overnight run does not fail
  at its first point.
  The run took ~5 h, not ~12, and its reruns of steps 48-74 matched their original numbers within
  0.2 t/s, which is what lets the old and new points share one chart. A small status script
  (progress, a stall warning) made the long run checkable at a glance; its first version
  counted the summary's header line as a prompt, caught when it printed "9 of 8".
- **Ask what the time is made of before moving it (README step 73).** The engram fetch looked
  like ~2 ms of host work to hide behind layer 0; moved there it gained only 0.3 ms, because it
  was longer than layer 0's front and delayed that layer's CPU experts. A fault counter showed
  the ~1.4 ms was ~200 minor page faults on a 189 GB mapping (one per row read); pread removed
  them, and the move then fit. The trace found the last piece (the copy waiting on die 0's
  stream). The end result is small (+0.3% mean, ~1 ms per step); the person's question about
  huge pages was answered from the access pattern (scattered first touches) rather than tried.
- **Ask whether the work is used at all (README step 74).** Asked whether the indexer was serial,
  the per-layer trace answered yes (~76 us per token), but the larger finding was that at the
  held-out lengths every compressed position is kept, so its result was never used. Skipping it
  was three lines and exact by construction (+2.4%); batching it, the answer to the question as
  asked, only pays at long context and went to the backlog with its exactness requirements.
- **Measure the limiter before fixing it (README step 75).** The fp8 kernel's own comment blamed
  decoding; removing the decode made it slower. The register count then pointed at occupancy, and
  raising it also made it slower. Four cheap variants, each with its bits checked, were enough to
  show the kernel needs a different work split; stopped there rather than restructure it.
  Step 65 had already measured part of this (and concluded "not done"); the README's earlier
  steps should be searched for a kernel before an option built on it is ranked first.
- **Price the hand-off and the memory, not just the work moved (README step 76).** The split
  of attention's output was estimated from the work it moves (~140 us per layer at 6 tokens) less
  a hand-off cost measured alone. The real sequence cost ~55 us per layer of extra kernels and
  waits, and the moved weights took 1.41 GiB from die 1's expert cache, which the estimate left
  out; together they left +0.6%. A bench of the whole sequence on the real shapes (bits checked)
  showed the first in minutes; the trace's CPU expert pairs showed the second.
- **Find the ceiling before building toward it (README step 77).** A CPU with a large cache
  looked like the way to keep experts on the CPU: prefetch the guessed ones during attention. Its
  ceiling is the kernel on an expert already in cache, a few minutes to measure here (one expert
  fits): x1.2, against x3-6 for a plain read. A first reading ("bound by its arithmetic") was
  one step too quick: the next measurement put the arithmetic at about half the warm call
  (step 78).
- **Reproduce under the runtime's conditions before believing a bench (README step 78).** The
  warm call's 2x was reproduced in a bare loop and pinned on reading freshly written activations;
  per-thread copies took 25% off back to back, and the end to end A/B gave nothing. The bench had
  shown why in its 1.5 ms gap case (the runtime idles that long between calls), which was read as
  noise. Also: a reboot empties the page cache; the first run after it (38.4 instead of 43 t/s)
  was set aside and the cache warmed before measuring.
  The suspected after-gap cost then measured at 5-10%; the open question (why fresh activations
  read slowly) was left for hardware counters instead of a fifth variant.
- **Counters turn guesses into one experiment (README step 79).** Four variants had failed to
  explain the slow reads. The counters (machine clears, HITM loads, L2-miss stalls) pointed at
  sharing, and one more variant (a single writer) isolated it: the lines were slow because many
  threads wrote them. Installing perf took one apt command; it would have saved the earlier guesses.
- **Make the fix cheap enough, then apply the finding directly (README step 80).** The counters
  said one writer removes the penalty; it was too slow only because its code was scalar. A
  vector version checked byte for byte against the scalar one (kept as the specification) on 2M
  blocks of edge cases made one writer cost 7 us, and the finding applied as is: +1.0%, where
  the workarounds (copies, flushes, aligned chunks) had given 0-0.2%.
- **Two prompts are not the held-out set (README step 81).** A spinning host wait won both
  cpp-matrix pairs (+1.5%) and was flat on tcp-udp; on all 8 prompts it was -0.5%. The A/B is
  for a quick no; a keep needs the full set. Also: one variable per sudo request (C6 off was
  answered in one bench run and switched back before the next test).
- **Bench the alternatives next to the plan (README step 82).** The plan was to prefetch the next
  task; the same bench run also tried static, guided and larger chunks. Prefetching only hurt;
  contiguity was the effect, and the plain OpenMP setting that keeps most of it after the gap
  (chunks of two) gave +2.3%. A one-constant change, found because the bench measured the cases
  the plan did not expect.
- **Look at the distribution, not the median (README step 84).** A breakdown script showed ~160
  us per layer for a copy of a few hundred KB; its median was 44 us, the cost was the tail (p99
  1.5 ms), and one run without background loads named the cause. A fix that removes a delay can
  expose the next one: half the gain came back as waiting for the other dies, whose copies
  share the same problem.
- **Correlate before fixing (README step 86).** Half the drafts were twice as slow, in turns. Two
  guesses were wrong (the host launching too slowly: the issue span had been measured through a
  blocking copy; the unpack kernel: 7x faster, no change), and one more trace track (each load,
  per die) settled it: 36 of 36 slow drafts overlapped a load into die 0, none of the fast ones.
  The fix was to stop loading there, +8.7%, the largest step since the cache's tuning.

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
| one CPU miss per layer hides behind the GPU's work (50) | partly | 0.05-0.15 of ~0.35-0.48 ms | the GPU's work alongside (shared expert, each die's experts) is shorter than one miss |
| fp8 compressed caches free expert slots (59 -> 60) | the next lever | 21 slots at 64k | the caches are small; the draft is 8 GB |
| the batched attention core (66-68) | attn.core 17.7 -> 6-9 ms, +6-8% on code | 7.8 ms, +5.7% mean (+10% on cpp-matrix), the prompt +6-12% | as planned; the stages kept every output identical |
| the fp8 projections lose the step 62 way (65) | 3-7% | ~1% at 3-4 inputs, 0 at 6 | the cost is decoding inputs per lane group, not L2 reads; the fixes keep the bits only partly |
| launches / host gaps (65) | <= 7-12% | <= 3-5% | the host is usually 0.1-1 ms ahead; it waits on its own work, not on launches |
| the next layers' experts from this layer's router input (63) | "demonstrated for some models" (the person) | 72% one layer ahead, 4-7x the use counts for CPU experts | the residual stream changes little per layer |
| a GPU prefetch from those guesses (64) | ~15% if a third of misses covered | 2.5% simulated (bound 10%) | loads need 3-4 layers' notice, where guesses are weaker; wrong loads take link time |
| attention's output over dies 0-1 (76) | ~3.5% of a code step | +0.6% | ~55 us per layer of hand-off kernels and waits; the weights on die 1 cost its cache 1.41 GiB (CPU expert pairs +4.4%) |
| a CPU expert from cache vs DRAM, the ceiling of an LLC prefetch (77) | ~3x (assumed, for a large-LLC EPYC) | x1.2 | DRAM is not the call's limit: half of a warm call is neither DRAM nor arithmetic (step 78) |
| per-thread activation copies in the CPU call (78) | ~8-10% on code (from -0.1 ms per expert back to back) | flat | the runtime's calls follow a ~1.5 ms idle gap, after which the copy costs what it saves |
| aligned 16-block activation chunks (79) | ~40% of the read penalty, ~1-2% end to end | +0.2% held-out (chat +0.5-1.1%) | the remaining penalty is the spread over 16 writers; the cold path and the gap dominate a call |
| one writer per token / pair, AVX-512 digits (80) | the bench's -0.1 ms per call; +2-4% | +1.0% held-out (A/B +1.2-1.7%) | a call is ~1 per layer with CPU experts; the rest of it (cold memory, compute) is unchanged |
| a spinning host wait after the idle gap (81) | ~1-2% (the bench's ~0.02 ms per call, and a quicker handoff) | -0.5% held-out (A/B: code +1.5%) | a spinning host thread takes a core the CPU's experts share; the A/B's code gain did not carry over |
| a claim-ahead scheduler prefetching the next task (82) | -10-25% per cold CPU expert, 3-8% end to end | the prefetch slower; chunks of 2 tasks instead: +2.3% held-out | the cold starts cost less than lost contiguity; prefetch bursts compete with the streams |
| a CPU weight layout (83, bench) | decode ~0.03 ms cheaper per expert, 2-4% | compute x1.8, a cold call -9%, warm -18%; est. +1.5-2.5%, not deployed | a cold call is mostly DRAM time now; deploying needs a second 269 GB expert store |
| the CPU's rows copied by a kernel, not the copy engine (84) | most of ~5-6 ms per code step, +3-5% | +1.1% held-out | the dies' results then arrive last in ~10% of layers: their own copies wait behind their loads |
| background loads paced by the layers (85) | the dies' ~3 ms per code step of waiting | slower or no change | one matrix per layer starves the cache; more bunch up against the next layer's attention; holding loads back in the expert phase is too late |
| the home die keeps its startup experts (86) | the slow drafts (~3%), +4-5% in traces | +8.7% held-out | die 0's loads had also slowed its own dense work, not only the draft |
| the draft's experts pruned by usage (87, measured only) | skewed: half of them for ~3% more capacity | skewed within a prompt, not across prompts: top 64 cover 70-89% of an unseen prompt's choices | a draft for every kind of prompt needs most of its experts; the per-prompt working set is the small one |
| the verify step's argmax on the GPU (88) | ~1% on code, ~0.8% on chat | +0.9% held-out (+1.1-1.5% on code, +0.6% on chat) | as estimated from the step's tail |
| the time outside the layers (89, traces) | ~5-6 ms per code step, the head and overhead | ~2-3 ms untraced, almost all the head; the trace's bookkeeping 1.2-1.7 ms | a traced run's own overhead looked like the step's; the head kernel is what is left |
| the GPU clock level (90: sclk 2 / 6 / 8) | a third or so of the time scales with the clock | 36.8 / 41.1 / 42.1 t/s: ~35% of a token's time at sclk 6 scales with it | the fit on two levels predicts the third within 0.1%; the ~65% is the CPU side and HBM-bound GPU work (step 94) |
| the CPU layout as a copy in RAM (91, not kept) | +1.5-2.5% (step 83's bench), +0.5-1% with half the layers | -7-12% per CPU expert in the same run (control 0%); all layers +1.6% held-out | down gains, gate + up is memory-bound after the gap; 269 GiB, ~45 s per start and an unexplained abort for 1.6% |
| the draft's head over the most frequent ~32k tokens (92, simulated) | ~1.5-2% (the head's 2.2 ms at a quarter) | -8.8% at 32k, -0.5% at 96k; at best +4% with an oracle | 12% of emitted tokens fall outside 32k, and each cuts a chain of accepted drafts; the Markov part was a second V-wide cost |
| a rare abort at exit (93, investigated) | a double free in our teardown, found with gdb or ASan | not repeated in 56 runs (14 plain, 30 gdb, 10 ASan) | rare and timing-dependent: a race at shutdown; a stack handler kept for the next one |
| step 70's verify halves, traced (94) | overlap the CPU's ~25 ms with die 0; the loss was host blocking or doubled per-call costs | -13 / -17%; die 0 +28 ms per code step, less overlap (4 vs 8 ms) | die 0 reads its weights once per half: two 3-token halves cost ~84 ms against 54.5 for one 6-token step |
| llama.cpp upstream instead of the fork (95) | upstream merged DSV4.1, so the fork can go | upstream reads our GGUF silently wrong: garbage | the files' type 43 is the PR's MXFP8, upstream's plain fp8; fork kept, pinned by hash for new machines |
| the exit abort revisited (96) | step 93 missed something in the teardown | its window was too narrow (the stats line comes before the loader stops); concurrent loader frees alone: 7,800 clean | three teardown fixes (stop under the mutex, serial frees, unregister); the handler prints phase, thread and loads |
| the dies' results in smaller copies (97) | a review of the traces span by span: the tail's copies took the worst-case size | 3 rows per token first, a second copy for the rest (3.5%); the staged block in one copy: +2.1% | kept; the exit abort seen again: after main, in the HSA runtime's shutdown |
| the compressor's projections per chunk (98) | the same review: the one span still growing per token | both weights read once per chunk, the state token by token: +1.3% | kept |
| the review's leftovers (99) | four smaller items from step 97's review | hc kernels fused: 35.2 vs 35.7 us; the compressor's state per chunk: decode 0.0%, prompt +1.1%; a gather stream per die: -0.5%; the draft's head in fp8: slower with today's fp8 kernel (estimate) | none kept; patches saved |
| traces at step 98 (100) | the traces were step 88's | regenerated; a 6-token code step 89.4 -> 87.1 ms (tail 6.3 -> 4.4, front 53.9 -> 52.3) | PLAN B refreshed |
| padded weight rows (101) | what limits the fp8 GEMV at 6 inputs: a ceiling probe, then why more rows per wave slowed bare loads | rows a multiple of 4096 bytes apart read at 279 GB/s, 715 with 64 bytes of padding; the attention output matrices padded: +2.4% (short steps up to +5.6%) | kept; the other kernels' matrices and the 6-input kernel next |
| the other kernels; wave slots (102) | step 101's padding on the f32 / bf16 matrices and the expert slots; then why the compressor's GEMV took 51 us at any input count | padding helps only the router; a launch of 2017-2048 waves (the GPU's wave slots filled exactly) takes ~48 us even empty, found by sweeping wave counts and counting a run's launches with a preload shim; the light ones sent as two launches: +1.0% together | kept (split for the compressor and the quantize, router / compressor rows padded); expert slots padded and the fp8 GEMVs split: not kept |
| the fp8 GEMVs by wave count (103) | do query B and output A pay for launching on multiples of 2048 waves? the same kernel over a matrix a few waves shorter | no (a little faster on the multiple); but the several-input kernels step every ~512 waves: a launch just over a multiple pays a whole extra round (query A 30.0 us at 640 waves, 21.6 at 512) | measured only; ~1-1.5% if the tail round went away, part of the several-input kernel's design |
| the several-input fp8 kernel (104) | what exactly fixes a 64-lane row's bits, and can a faster work split keep it? | the 64 partial sums and their tree, not which lane computes them: 16 or 32 physical lanes each carry 4 or 2 of the sums (the shared-memory kernel plus an accumulator index); bit-identical in every layout; attention output A 113 -> 79 us at 6 inputs; +1.3% held-out | kept; less than the bench suggested (traces next); the prompt kernel could use it |
| expert parallelism for the draft (62) | proposed by the person | ~1 ms at best | experts are 16% of the draft; the head (27%) was the item |
| the head through LDS (62) | ~1-2% | +1.4% held-out, +2% on code | as estimated from the per-part times |
| the draft in fp8 frees ~4 GB (60 -> 61) | ~210 slots | nothing | its dense part was already fp8 and its experts mxfp4: assumed from the file's name |
| decayed counts / hysteresis for the cache (59) | settle the churn, some gain | decay = window, margins lose | the loads they skip were useful; misses left are unpredictable |
| one-token groups on the one-token routine (57) | the ~0.25 ms "fixed cost" removed | -35% per one-expert call, +1.1% end to end | the cost was one phase (down) on the wrong routine, not fixed overhead |
| static schedule for verify-step CPU experts (56) | ~7% of a code step | 15-22% in the benchmark, 0-1% slower end to end | the benchmark lacked the gap before each call and the runtime's other threads |
| reading each expert once on the dies (55) | ~20 us saved per shared pair | -21% at heavy sharing, +23% without | a pair's cost was not the weight reads (a fit read as a mechanism) |
| batched router top-k (53) | ~4% (5-6 ms per code step) | +4.6% held-out, +1-4% prompt | as estimated; the prompt hides most of it behind the CPU |
| co-selected experts on different dies (54) | some gain if clusters exist | <= 0.3% (bound 1-3%) | weak affinities; the CPU sets a layer's pace |
| next layer's CPU expert guessable for a prefetch (52) | >= 60% needed | 17-28% (top-1) | the CPU's experts are the tail the cache did not keep |
| the long proof's loop is a bug (48) | possible | the model's greedy behaviour | both implementations continue the loop from the same tokens |
| chunks of 1,024 cut CPU expert time 20-30% (46) | 20-30% | ~6% per pair, prompt 6-12% slower | the kernel had become arithmetic-bound (step 45) |
| runtime's CPU experts = the benchmark x 1.65 (42) | overhead around the kernel | the build: -O2 vs -O3 | the runtime was compiled with the common CXXFLAGS (-O2) |
| loads hidden behind the GPU's work (42 -> 44) | ~0.7 ms per load (standalone rate) | 1.9-2.4 ms | the unpack kernels queue behind the dies' own compute |
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
- **A trace that looked right but did not nest** (README step 51): a check over every span found
  ~1% of groups 0.1 us past their layer, from rounding start and length separately; and a run
  ~15% slower than the day before was a cold page cache after a reboot, not the change (rerun:
  back to the earlier speed).
  The per-die expert counts added later were checked the same way: in every layer the dies' and
  the CPU's pairs add up to 6 x tokens.
  The person then spotted layer 13's experts inside layer 14's span: die 0's track was anchored
  at the time its first event was issued, 0.3-0.4 ms before it ran. Fixed by anchoring where the
  host waits on the GPU anyway, and checked by causality (a wait ends after what it waits for).
- **An overhead claimed without a like-for-like run** (README step 58): the trace's cost was
  written as 1-3% from runs of different days; measured properly (untraced, `--profile`,
  `--trace` on the same prompts and build) it is ~7%, most of it the profile's events.
- **Each summary answers a question someone asked** (README step 62): TIMING.md gained a
  section on background loads when the person asked how long moving an expert takes; every
  number in it was already measured (steps 30, 42-44, 59) except today's load rate, read from
  the latest runs' logs.
- **A summary document goes stale quietly** (README step 58): TIMING.md's tables kept numbers
  from four clock settings and a dozen steps back while every later step appended a sentence; the
  person noticed. Refreshed from one build and one clock, with the data in `results/`.
- **False alarms from the measuring scripts** (a trailing newline in a prefix comparison):
  every "differs" was investigated before being believed or dismissed.

## 6. What a reader should take away

- Exactness and determinism are not a cost here; they are what made aggressive optimizations
  (batching, speculation, moving experts at runtime) safe to try and easy to verify.
- The largest wins came from measurements that changed the plan (routing transfer, the oracle
  bound, the thread count, the clock result), not from the first idea.
- Simple models of where time goes, kept up to date and checked against measurements, chose
  almost every next step.
