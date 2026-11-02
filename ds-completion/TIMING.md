# Where the time goes

A one-page feel for the numbers: what a millisecond is worth, and what changes if a part gets
faster or slower. Details and measurements: [README.md](README.md) (steps cited), plans:
[PLAN.md](PLAN.md). Generation numbers are at sclk 2 unless noted, the prompt at sclk 6.

## Generation (one sequence, speculative decoding with the adaptive cache)

A step drafts up to 5 tokens, then verifies the anchor and n drafts in one forward:

    step  ~=  draft (~12-16 ms)  +  verify: ~50 ms  +  ~20 ms per draft verified

and yields the accepted drafts plus one token of its own: **2-4 tokens per step** (code and
structured text high, chat ~2-3, Chinese/Japanese ~2). A typical step, 2.6 drafts verified and
3.2 tokens: 12 + 50 + 52 = **~114 ms -> ~28 t/s** (measured: 25.6 t/s mean on the held-out
prompts at sclk 2, 28.0 at sclk 6; README steps 32, 48). Without speculation a token costs ~49 ms
(~20 t/s with the cache); without the cache ~55-60 ms (~17-18 t/s).
(README steps 25-26, 32; fitted verify lines 41-68 + 15-24 n ms per prompt.)

**What a millisecond is worth:** at ~114 ms per step and ~3.2 tokens, 1 ms per step is ~0.9%,
**~0.25 t/s**. For the draft (holding draft length and acceptance fixed):

| draft per step | step | t/s | |
|---|---|---|---|
| 1 ms | 103 ms | 31.1 | +11% |
| **12 ms (today)** | **114 ms** | **28.1** | |
| 24 ms | 126 ms | 25.4 | -10% |

(A cheaper draft also gets used more: the rule would draft on more steps and verify more, so the
gain at 1 ms would be somewhat larger.)

**Why the fixed ~50 ms and the ~20 ms per extra token:** the fixed part is one token's pass,
mostly the dense part (attention projections, shared expert, router, output head: ~26 ms of fp8
GEMVs and small kernels on die 0) plus the experts. Each extra verified token adds what is not
shared across tokens: the attention core runs token by token, the fp8 projections cost ~2x one
token at 6, and more experts miss the GPUs. A 6-token verify step at sclk 8 (README step 33):

| part | ms | share |
|---|---|---|
| dense: attention (core 16, output 11, q/indexer/compressor 15) | 42 | 34% |
| dense: shared expert, router + dispatch + sum, hyper-connections, head, engram | 36 | 30% |
| CPU experts (the GPU waiting for them) | 25 | 20% |
| draft | 12 | 10% |
| GPU experts (visible part) | 4 | 3% |
| **step + draft** | **~122** | |

So the dense part is two thirds of a step; the cache cut the CPU experts from 78 ms to 25. On
code, where a step verifies 6 tokens more often, the CPU's misses are larger: 50 of 134 ms at
sclk 6 (README step 49). Kernel launches (~3,000-3,900 per step at ~2.4 us) are at most 7-12%.
Per layer, a CPU miss costs ~0.27 ms per distinct expert plus ~0.1 ms for having any, and only
~0.05-0.15 ms of it hides behind the GPUs' work (README step 50): every miss avoided is ~0.27 ms
off the step. With the cache tuned (800 loads/s, every 8 tokens) and the router's top-k in one
launch the held-out mean is 31.7 t/s at sclk 6 (README steps 50, 53). On the dies an expert costs
~30 us per (token, expert) pair, mostly arithmetic rather than weight reads (steps 54-55). To see a step: `--trace` (README step 51).

## Prompt (chunks of 512 tokens)

The CPU and the GPU work on different halves of each chunk at the same time (README step 38),
so a prompt token costs roughly the slower of the two, plus a little:

    per prompt token  ~=  max(CPU experts, GPU work) + ~1-2 ms

1,989-token prompt, sclk 6 (README steps 45-46):

| per prompt token | fast | `--canon` |
|---|---|---|
| CPU: routed experts not on a GPU (~40% of expert uses) | 10.3 ms | 15.1 ms |
| GPU: attention core 4.0, attention output 1.4, router 1.1, indexer 1.0, compressor 0.7, shared expert 0.7, q 0.5, the rest 1.4 | 10.8 ms | ~11 ms |
| **wall** | **13.1 ms (76 t/s)** | **17.6 ms (57 t/s)** |

The two sides are balanced in the fast numerics, so halving either one alone gains only ~15%:

| if | wall per token | t/s |
|---|---|---|
| today (fast) | 13.1 ms | 76 |
| CPU experts 2x faster | ~11.5 (now the GPU sets the pace) | ~87 |
| GPU work 2x faster | ~11 (the CPU sets the pace) | ~90 |
| both 2x faster | ~6.5 | ~150 |

With `--canon` the CPU is the limit (15.1 vs ~11 ms): its exact integer kernel costs ~1.5x the
fast one per (token, expert) pair, so only the CPU side helps there.

**Scale:** a 30,000-token document takes ~6.5 min at 76 t/s (it took ~30 min token by token).
10,000 tokens of reasoning take ~6 min at ~28 t/s.

## Fixed costs

Model load ~9-20 s (from the page cache; minutes if it must come back from disk), expert upload
for a placement 5-30 s, the cache's expert registration ~100 s in the background (its loads
start only after it). For short runs these dominate: a 20-token prompt and 100 tokens of answer
take ~5 s of work after ~20-30 s of startup.
