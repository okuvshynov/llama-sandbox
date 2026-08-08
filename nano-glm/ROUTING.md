# Routing study: which experts does GLM-5.2 actually pick?

PLAN.md step 3 rests on an assumption that had never been measured — that a
GPU-resident subset of experts is "worth roughly the resident fraction and no
more", i.e. that routing is close to uniform. It is not. A 23% resident subset
catches **58.4%** of selections, and the token-to-token overlap of expert sets
is **11x** what independent routing would give.

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

**A resident subset is worth ~2.5x its size**, not 1x. What 128 GiB of VRAM
buys is not 23% of expert traffic but 58% of it.

The last column is a negative result worth keeping: spending one global budget
across all layers — letting the skewed middle layers take capacity from the
near-uniform early ones — is worth **nothing** (58.4% vs 58.4%; +1.0 point at a
50% budget). Only five of seventy-five layers are near-uniform, and taking
their capacity costs about what it gains. Uniform per-layer placement is both
simpler and equally good.

What this does to the ceiling, as an upper bound: MoE reads fall from
21.66 GB/token to 9.01 GB, i.e. 122 ms at the 74 GB/s this machine sustains, so
the MoE-bound ceiling moves **3.42 -> ~8 tok/s**. The resident half is not the
new bottleneck (12.65 GB/token across four Vega II dies at ~1 TB/s each is
~3 ms), but this ignores dispatch cost entirely — see ../../../moe-offload for
what a Vulkan round trip actually costs, which is what turns this bound into a
prediction.

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

## What this does not show

- **One prompt.** The split-half test measures stability *within* a single
  continuation, which is the easy case: same topic, same register. A static
  VRAM placement is chosen once and must serve every workload, so the number
  that decides step 3 is the cross-prompt one — rank on prose, score on code.
  Treat 58.4% as an upper bound until that is run. It is cheap: five corpus
  prompts at `-n 256` is about half an hour.
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
