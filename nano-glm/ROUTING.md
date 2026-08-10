# Routing study: which experts does GLM-5.2 actually pick?

PLAN.md step 3 rests on an assumption that had never been measured — that a
GPU-resident subset of experts is "worth roughly the resident fraction and no
more", i.e. that routing is close to uniform. Within one continuation it is
not: a 23% resident subset catches **58.4%** of selections and token-to-token
overlap is **11x** what independent routing would give.

**Across prompts it collapses.** The cross-prompt section finds a placement
built from other prompts catching **28.2%** where picking at random catches
23.1%. Both readings are kept because the difference between them is the
finding: routing is concentrated and *prompt-specific*, and only the first half
of that helps a static placement.

## The run

```
nano-glm (build-trace, MSVC, 16 threads) -i results/corpus/nano_01_prose.bin -n 1024
  --expert-log prose-n1024.trace
python expert_stats.py prose-n1024.trace          -> prose-n1024.stats.txt
python expert_stats.py prose-n1024.trace --null   -> prose-n1024.null.txt
```

1138 positions (114 prompt + 1024 generated) x 75 MoE layers x 8 of 256
experts = 682,800 selections, 35.6 expected per (layer, expert) cell. Local MoE
path, greedy, 1.69 tok/s. Length was chosen so that a split-half check still
leaves ~18 selections per cell in each half — enough to rank a layer's head
against its tail, which is all the residency question needs.

Everything is reported against **`--null`**: the same script over i.i.d.
uniform draws of the identical shape. That is not decoration. At this sample
size the *in-sample* residency column reads 30.4% for a 23% subset with no skew
at all, so any in-sample number below ~30% would have been noise dressed as a
finding. All headline numbers below are out of sample.

Two controls say the concentration is real and not an artifact of greedy
decoding falling into a loop:

- 493 of 1024 generated tokens are distinct; repeated 4-grams are 0.8%.
- the prompt segment and the generated segment agree closely (overlap 35.7% vs
  35.2%, rank-0 repeat 24.8% vs 25.7%) — natural text and model output route
  the same way.

## Skew

| | measured | uniform |
|---|---|---|
| entropy per layer | **6.98 bits** (6.24 - 7.84) | 8.00 |
| experts carrying 50% of selections | **36** of 256 | 128 |
| experts carrying 90% of selections | 134 of 256 | 230 |
| busiest expert's share | **5.94%** (1.05 - 11.11) | 0.39% |
| experts never selected in 1138 positions | ~13 per layer | 0 |

There is a clear depth profile. The first five MoE layers (3-7) route almost
uniformly — entropy 7.81-7.84 bits, consecutive overlap 3.8-8.7% against an
independent baseline of 3.1%. Skew appears at layer 8, peaks around layers
17-21 (entropy 6.24-6.72, only 14-24 experts for half the mass), relaxes
slightly, and holds near 6.6-7.0 bits for the remaining sixty layers. Whatever
the early layers are doing, they are not specializing.

## Static residency (the step 3 number)

Rank each layer's experts on the first half of the run, hold the top f% in
VRAM, score the hit rate on the second half:

| resident | hit rate | null | per-layer spread | global budget |
|---|---|---|---|---|
| 5% | **27.0%** | 5.1% | 9.3 - 45.3 | 27.3% |
| 10% | **38.7%** | 10.3% | 17.0 - 56.0 | 38.8% |
| **23%** (128 GiB VRAM) | **58.4%** | 23.1% | 34.3 - 73.0 | 58.4% |
| 33% | **69.1%** | 32.9% | 44.9 - 82.7 | 69.4% |
| 50% | **82.3%** | 50.0% | 63.5 - 92.0 | 83.3% |

**Within one continuation a resident subset is worth ~2.5x its size**, not 1x.
Read on, though: across prompts it is worth 1.2x, and that is the number a
static placement actually gets.

The last column is a negative result worth keeping: spending one global budget
across all layers — letting the skewed middle layers take capacity from the
near-uniform early ones — is worth **nothing** (58.4% vs 58.4%; +1.0 point at a
50% budget). Only five of seventy-five layers are near-uniform, and taking
their capacity costs about what it gains. Uniform per-layer placement is both
simpler and equally good.

That arithmetic used to continue "so the MoE-bound ceiling moves 3.42 -> ~8
tok/s". Two later measurements retired it: `nano-bench` found decode bound by
total bytes rather than MoE bytes and only ~75% memory anyway, and the
cross-prompt section below found 58.4% unreachable by a static placement. The
current numbers are in that section and in PLAN.md's budget.

## Locality

| | measured | independent |
|---|---|---|
| consecutive-token set overlap | **35.2%** (3.8 - 52.3) | 3.1% |
| rank-0 expert repeats | **25.6%** (1.3 - 61.4) | 0.4% |
| distinct experts in a 8-token window | 36.6 | 57.4 |
| distinct experts in a 32-token window | **92.1** | 163.3 |
| distinct experts in a 128-token window | 171.2 | 251.6 |

Consecutive tokens share a third of their experts, and one token in four keeps
the same top-ranked expert as its predecessor. Locality rises with depth in
step with skew: 3.8% at layer 3, ~25% by layer 15, and 36-52% from layer 28
onward.

This is the first evidence for the prefetch idea in PLAN.md's "Ideas,
unproven": "keep the previous token's experts warm" is a predictor with 35%
recall against a 3% base rate, available for free, before any speculative
router runs. Note what it does *not* say — a hit rate is not a saving. Prefetch
only pays where the fetch has somewhere slower to come from, which is Kimi-K3
with experts on storage, not GLM-5.2 with experts in DRAM.

## Cross-prompt: it does not transfer

Same question as the table above, asked across five corpus prompts instead of
one continuation (`-n 256` each, `residency_study.py`). Hit rate with the top
f% of experts per layer resident:

| resident | prose | code | math | history | french | pooled | **unseen** | null |
|---|---|---|---|---|---|---|---|---|
| 5% | 9.3 | 14.1 | 14.4 | 28.2 | 26.6 | 12.4 | **7.0** | 5.1 |
| 10% | 16.4 | 23.6 | 23.8 | 39.6 | 38.3 | 20.6 | **13.4** | 10.2 |
| 23% | 32.3 | 42.2 | 41.5 | 59.4 | 58.7 | 37.9 | **28.2** | 23.1 |
| 33% | 42.8 | 53.5 | 52.0 | 69.8 | 69.4 | 48.6 | **38.5** | 32.9 |
| 50% | 60.9 | 69.6 | 68.1 | 82.6 | 82.8 | 65.1 | **55.8** | 50.1 |

The five named columns rank and score inside one prompt. **pooled** ranks on
all five and scores each. **unseen** ranks on four and scores the fifth — the
only column that describes a placement meeting a workload it was not built
from. **null** is the same machinery over uniform draws, and lands on the
resident fraction as it should.

So, at 50% resident: **83%** if the placement is tuned for that exact prompt,
**65%** if tuned for a corpus containing it, **56%** for a prompt it has never
seen, against **50%** for picking experts at random.

**The unseen column is the one that matters, and it is worth ~6 points over
random at every size.** A static placement gets almost all of its value from
holding *some* fraction of the experts closer, and almost none from holding the
*right* ones.

Two things the per-prompt columns show on the way past. Prompts differ a lot —
history and French are twice as easy to serve as prose at 5% resident — and the
ordering follows how concentrated each one's routing is (code spreads over 47
experts per layer for half its mass, history over 31). And the gap between any
named column and `unseen` is the whole finding: within a prompt the signal is
large, between prompts it nearly vanishes.

The one exception worth knowing: **prose and history** transfer to each other
far better than any other pair (45.6% at 23% resident, resident-set overlap
0.34 against 0.13 for independent choices). Both are English prose about
cities. Routing is domain-specific rather than universal, so a placement
specialised per workload is a real possibility even though a single static one
is not.

Transfer matrix, per-prompt diversity metrics, Jaccard overlaps and a
by-depth breakdown: `results/residency/study.txt` (`--detail`).

## Would a cache do better than a fixed placement?

The traces are an access sequence, so this needs no new runs (`cache_sim.py`).
Each layer gets its own cache of f% of 256 experts, cold-started with a random
resident set. Hit rate, and what it costs:

| resident | policy | hit rate | DRAM GB/tok | PCIe GB/tok | s/token |
|---|---|---|---|---|---|
| 23% | static (fixed, random) | 22.9% | 14.57 | 0 | 0.193 |
| 23% | random eviction | 58.0% | 7.94 | 7.94 | 0.716 |
| 23% | **LRU** | **63.3%** | 6.93 | 6.93 | 0.625 |
| 23% | LFU | 48.1% | 9.81 | 9.81 | 0.885 |
| 50% | static | 49.9% | 9.46 | 0 | 0.126 |
| 50% | **LRU** | **83.6%** | 3.10 | 3.10 | 0.280 |

For scale: reading all 18.90 GB of routed experts from DRAM, as today, is
0.251 s/token, and a whole token is 0.516 s.

**LRU wins the hit rate outright and loses the race anyway.** At 23% resident
it hits 63.3% where the best *deployable* static placement manages 28.2% and
even a within-prompt oracle only reaches 58.4% — recency beats any fixed
ranking, which is what the 35% consecutive-token overlap predicted. But a
static miss is a DRAM read, while a cache miss is a DRAM read **and** a PCIe
install, and at 13 GB/s against 75.4 GB/s that install makes each miss **6.8x**
more expensive.

The break-even is not close. To beat the deployable static placement a cache
would need a **89.4%** hit rate at 23% residency, or **93.5%** at 50%. LRU
reaches 63.3% and 83.6%. In wall-clock terms it is 3.5x slower than static at
23% and 2.5x at 50% — and worse than doing nothing at all.

So PLAN.md's "static, not LRU" survives, now for a measured reason rather than
an asserted one.

Two smaller findings. **LFU is actively bad and gets worse**: 63% in the first
tenth of the run decaying to 41% by the last, because early-hot experts
accumulate counts that later evidence cannot overcome, and the cache ossifies.
And **there is no warmup to speak of** — a 59-entry cache filling at 8 accesses
per token is warm within about seven tokens, so LRU's first band already reads
62% against a 61% steady state. The interesting curve was LFU's decay, not
anyone's warmup.

### When would a cache pay?

When installing into the fast tier is cheap relative to the miss it saves.
Here it is the opposite: the install crosses the very link that makes misses
expensive. Invert the ratio — experts on NVMe at ~7 GB/s with DRAM as the cache,
which is Kimi-K3's shape — and an install is a DRAM write at 75 GB/s against a
miss that costs ten times more. There LRU's hit-rate advantage would convert
directly into time saved. **The policy is not the thing that transfers between
tiers; the bandwidth ratio is.**

### What that costs step 3

At the measured 75.4 GB/s, 38.93 GB/token, and the ~75% memory share
`nano-bench --pages` established:

| resident 23% chosen by | hit rate | DRAM/token | decode |
|---|---|---|---|
| a within-prompt oracle | 58.4% | 27.89 GB | 2.47 tok/s (+27%) |
| **leave-one-out, deployable** | 28.2% | 33.60 GB | **2.16 tok/s (+12%)** |
| **23% picked at random** | 23.1% | 34.58 GB | **2.12 tok/s (+9%)** |

Residency still pays. It pays for holding 23% of the bytes closer, not for
holding the right 23%.

## What this does not show

- **Five prompts, one model, one language mix.** The corpus is prose, code,
  math, history and French; a workload that stayed inside one domain would look
  more like the prose/history pair than like the average.
- **370 positions per prompt**, so 5.8 selections per (layer, expert) cell in
  each half against 35.6 in the study above. That thins the ranking and pushes
  every hit rate down, which is why the within-prompt diagonal reads 46.8% here
  and 58.4% there. The null and the diagonal bracket it: signal-above-null is
  the figure to carry forward, not the absolute.
- The split falls at position 185, so the ranking half contains all the prompt
  tokens and the scoring half is pure generation. Consistent across every cell,
  but it makes the diagonal a slightly conservative ceiling.
- **One decoding path.** Greedy. Sampling would spread the token distribution
  and could spread routing with it.
- **Ids only.** The trace records which experts, not their routing weights, so
  "how much would dropping the tail experts cost" is not answerable from it.

## Files

Everything below is under `results/expert-routing/`, which this project does
not track — regenerate with the two commands at the top. Conclusions live
here; bytes live there.

- `prose-n1024.stats.txt` — full report, including the 75-row per-layer table
- `prose-n1024.null.txt` — same script, uniform draws, same shape
- `prose-n1024.counts.csv` — per (layer, expert) selection counts, 19,200 rows.
  The one worth keeping if any: 196 KB that reproduces every residency and
  skew number here without the model or the 15-minute run.
- `prose-n1024.trace` — the raw trace, 2.9 MB
- `ab-plain.bin` / `ab-trace.bin` — the byte-identical pair proving the traced
  build does not perturb the model
