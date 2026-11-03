# Where the time goes

A one-page feel for the numbers: what a millisecond is worth, and what changes if a part gets
faster or slower. Details and measurements: [README.md](README.md) (steps cited), plans:
[PLAN.md](PLAN.md). Generation and prompt numbers are at sclk 6 (the GPU clock level).

## Generation (one sequence, speculative decoding with the adaptive cache)

All at sclk 6 with today's settings (`--spec --canon --cache-loads 800 --cache-period 8`; README
steps 57-58). A step drafts up to 5 tokens, then verifies the anchor and n drafts in one forward:

    step  ~=  draft (~9-14 ms)  +  verify: ~40 ms for one token, ~+14 ms per token after it

(verify steps of a chat prompt, `--profile`, medians: 1 token 40 ms, 2: 56, 3: 70, 4: 85, 5: 94;
a code prompt's 6-token step 108). A step yields the accepted drafts plus one token of its own:
**~2 tokens per step on chat and Chinese/Japanese, 3.3-4.5 on code and structured text**. Two
typical steps, from the held-out run (README step 57):

| | tokens per step | draft | verify | step | t/s |
|---|---|---|---|---|---|
| code (cpp-matrix), mostly 6 tokens verified | 4.48 | 12 ms | ~109 ms | 121 ms | 36.9 |
| chat (tcp-udp), mostly 2-3 verified | 2.16 | 10 ms | ~65 ms | 75 ms | 28.9 |

Held-out mean: 32.0 t/s (26.7-38.6); 32.5 since the output head reads its inputs through LDS
(README step 62: -1.1 ms per draft call, -1.6 ms per 6-token verify step). For scale: step 48 measured 28.0 at the same clock before
the cache tuning and the router and CPU-routine fixes (steps 50, 53, 57).

**What a millisecond is worth:** on the code step 1 ms is 0.8% (~0.3 t/s), on the chat step 1.3%
(~0.4 t/s). For the draft (holding draft length and acceptance fixed):

| draft per step | code step | code t/s | chat step | chat t/s |
|---|---|---|---|---|
| 1 ms | 110 ms | 40.7 (+10%) | 66 ms | 32.8 (+14%) |
| **today (12 / 10 ms)** | **121 ms** | **36.9** | **75 ms** | **28.9** |
| 24 ms | 133 ms | 33.6 (-9%) | 89 ms | 24.3 (-16%) |

(A cheaper draft also gets used more: the rule would draft on more steps and verify more, so the
gain at 1 ms would be somewhat larger.)

**Where a verify step goes** (`--profile`, sclk 6, medians of the run's second half; profiling
adds ~5%, README step 58):

| part | code, 6 tokens | | chat, 2 tokens | |
|---|---|---|---|---|
| attention: core, output, q, indexer, compressor, kv | 45.6 ms (core 17.7, out 11.9) | 39% | 21.4 ms (out 7.3, core 5.6) | 33% |
| other dense: shared expert, hyper-connections, head, router, dispatch, engram, sum | 27.9 ms (shared 10.9) | 24% | 17.8 ms (shared 5.5) | 28% |
| CPU experts (die 0 waiting for them) | 26.2 ms | 22% | 11.2 ms | 17% |
| GPU experts (die 0 waiting for the other dies) | 5.9 ms | 5% | 4.1 ms | 6% |
| draft | 12.2 ms | 10% | 10.0 ms | 16% |
| **verify + draft** | **~118 ms** | | **~65 ms** | |

So the dense part is ~60% of a step, the CPU's misses about a fifth: per layer a CPU miss costs
~0.27 ms per distinct expert plus ~0.1 ms for having any, little of it hidden behind the GPUs
(README step 50), so every miss avoided is ~0.27 ms off the step. The attention core runs token
by token, so it grows fastest with the tokens verified (5.6 -> 17.7 ms from 2 to 6). Kernel
launches (~3,000-3,900 per step at ~2.4 us) are at most 7-12% (step 49). On the dies an expert
costs ~30 us per (token, expert) pair, mostly arithmetic rather than weight reads (steps 54-55).
On the CPU a verify step's expert streams at ~55-62 GB/s against ~90-105 GB/s for a plain read;
a static schedule, 2 MB pages and CPU power settings did not close that end to end (step 56).

## Prompt (chunks of 512 tokens)

The CPU and the GPU work on different halves of each chunk at the same time (README step 38),
so a prompt token costs roughly the slower of the two, plus a little:

    per prompt token  ~=  max(CPU experts, GPU work) + ~1-2 ms

1,989-token prompt, sclk 6 (wall: README step 53's runs without profiling; parts: `--profile`,
README step 58, which adds ~5%):

| per prompt token | fast | `--canon` |
|---|---|---|
| CPU: routed experts not on a GPU (~40% of expert uses) | 10.4 ms | 15.0 ms |
| GPU: attention core 4.0, attention output 1.4, indexer 1.0, compressor 0.7, shared expert 0.7, q 0.5, router 0.2, the rest 1.1 | 9.6 ms | ~10.3 ms |
| **wall** | **12.0 ms (~83 t/s)** | **16.8 ms (~59.5 t/s)** |

(The router was 1.1 ms per token before its top-k went into one launch, README step 53.) The two
sides are close in the fast numerics, the CPU now slightly the slower, so halving the GPU's work
alone gains nothing:

| if | wall per token | t/s |
|---|---|---|
| today (fast) | 12.0 ms | 83 |
| CPU experts 2x faster | ~11.2 (now the GPU sets the pace) | ~89 |
| GPU work 2x faster | ~12.0 (the CPU sets the pace) | ~83 |
| both 2x faster | ~6.8 | ~147 |

With `--canon` the CPU is the limit (15.0 vs ~10.3 ms): its exact integer kernel costs ~1.5x the
fast one per (token, expert) pair, so only the CPU side helps there.

**Scale:** a 30,000-token document takes ~6 min at ~83 t/s (it took ~30 min token by token).
10,000 tokens of reasoning take ~5 min at ~32 t/s.

## Seeing a step: traces

`results/trace/tcp-udp.json.gz` (chat) and `cpp-matrix.json.gz` (code), 300 tokens each with
speculation and the tuned cache (README steps 51, 58), open in ui.perfetto.dev. Per layer: die
0's "layer N" span with its groups nested (attention, router, shared expert, the wait for the
CPU, the sum); the host's "CPU experts" span with how many experts and tokens; each die's routed
experts with theirs. The things above show up directly: a verify step is ~40 layers of 1.6-2.5
ms; in a layer with CPU experts die 0 sits in `ffn.wait_for_cpu_experts` for most of the CPU's
span, while the dies' experts finish in 0.1-0.35 ms.

Measuring costs time: `--profile` 5-6% of generation speed, `--trace` 7% (28.9 / 36.9 t/s
untraced vs 26.8 / 34.5 traced; README step 58), so a trace's times run a few percent long; its
proportions hold. `--trace FILE` writes the file for any run.

## Background expert loads (the adaptive cache)

A load copies one expert (gate, up, down: 3 x 6.27 MB = 18.8 MB of mxfp4) from the model file in
the page cache to a GPU over PCIe and unpacks it into a free slot, on a low-priority stream; one
loader thread per GPU (README steps 30, 32). The copy engine reads host memory directly (the
expert tensors are registered with HIP), so the CPU only issues API calls (~1 ms per load).

| | per load | rate |
|---|---|---|
| one GPU alone | ~2 ms (9.2 GB/s) | 491 loads/s |
| all four GPUs | ~0.7 ms of host-link time each | 1,437 loads/s (27 GB/s) |
| during the prompt, competing with its copies | 1.9-2.4 ms | |

(Standalone at sclk 2, README steps 30, 42-44.) A loaded expert is used from the next forward
after its copy and unpack finish; its victim leaves the routing table before its slot is
overwritten, so outputs never depend on load timing.

**What the cache asks for:** every 8 tokens it swaps the most-used experts not on a GPU for the
least-used ones that are, at up to 800 loads/s. In 300-token runs it starts 310-465 loads/s,
**~11-12 per generated token**, about a quarter of what the links could carry (README step 62's
runs). A token takes ~31 ms; a missed expert costs the CPU ~0.27 ms on the critical path, a load
~2 ms of one GPU's copy engine in the background.

**What loads cost generation:** not nothing. Loading at 100 / 200 loads/s cost a decode 1-2% / 5%
when only loads were added (README step 30), and the copies share the PCIe links and host memory
with the CPU's expert reads. Letting the cache load ~60% more often (`--cache-min 1`) gained 1-1.6
points of experts on GPUs and lost ~0.7% (README step 59). The tuned rate and threshold are where
loads still pay.

**At startup:** before the first load the 289 GB of expert tensors must be registered with HIP,
~95-135 s in the background (103 s today; 64 MB pieces so the rest of the process is not
stalled, README step 43); generation runs meanwhile on the startup placement. `--cache-wait`
waits for it, for measurements.

Could loads be timed by the experts the next layers will choose? Layer L's router input passed
through layer L+j's router guesses them well (README step 63), but a load needs ~2.9 ms, so it
must start 3-4 layers ahead, where the guesses are weaker: a prefetch simulates to ~2.5% of
decode time at best against a 10% bound for perfect guesses (step 64).

## Memory

Expert slots fill 84% of the four GPUs (6,126 slots of 18.8 MB at 4k context). GPU 0 also holds
the draft (8.0 GB, mostly its own 384 mxfp4 experts), the dense fp8 weights (6.85), the small
weights (1.9) and activations (0.8),
so it has 831 slots against 1,765 on each other GPU; 64k context costs only 0.4 GB (README step
60, `results/gpu-memory/`). The runtime prints this ledger at startup.

## Fixed costs

Model load ~9-20 s (from the page cache; minutes if it must come back from disk), expert upload
for a placement 5-30 s, the cache's expert registration ~100 s in the background (its loads
start only after it). For short runs these dominate: a 20-token prompt and 100 tokens of answer
take ~5 s of work after ~20-30 s of startup.
