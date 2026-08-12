# Optimization backlog

Work that would make remote MoE *faster*. None of it is on the critical path in
`PLAN.md`, which is about making it *work*.

## Where DeepSeek-V4-Flash stands today

UD-Q8_K_XL, 150.7 GiB, on the Mac Pro (16 physical cores, four Vega II dies).
Two scoreboards, kept apart because they are measured by different harnesses and
a single table would invite a ratio that is not there — see "Comparing the two"
below.

### 1. llama.cpp, best of what it can do on this model

`llama-bench -t 16 -r 5 -p 128 -n 32 -lm none`. Measured at `132bcc2`.

| | best configuration | t/s |
|---|---|---|
| **prefill** | CPU only, `GGML_CPU_REPACK=ON` — no GPU involved | **22.16** ± 1.44 |
| **decode** | Vulkan, `-ngl 99 -ncmoe 24 -nopo 1 -ts 24/6/6/7` | **4.64** ± 0.02 |

```powershell
.\build_bench.ps1                 # the two CPU builds
.\build_bench.ps1 -Vulkan         # and the Vulkan one
.\bench_ds4.ps1 -Repack           # prefill winner is the repack-on row
.\bench_ds4.ps1 -Vulkan           # the -ngl / -nopo / -ncmoe configurations
```

Two things not to over-read. The decode winner sits in a **3.6-4.6 t/s band**
across the whole `-ncmoe` family whose ordering did not survive repeating one
configuration on a second load (10% load-to-load), so treat 4.64 as the top of a
band and not as a tuned optimum. And GPU offload buys *only* decode here: the
best prefill with a GPU in it was 16.68 (`-ngl 32 -nopo 1`), 75% of the CPU-only
figure, and every other GPU configuration was worse.

### 2. nano-glm, best of what expert-parallel can do

`testdata-deepseek4/01_prose.bin`, 111 prompt + 32 generated, `-t 16`, one
discarded warm-up and two measured passes. Measured at `906b965`.

| | best configuration | t/s |
|---|---|---|
| **prefill** | routed experts on the four dies, 93.75% resident | **8.45** |
| **decode** | all local, CPU only — no server | **2.195** |

```powershell
# best decode - all local, no server
build\bin\nano-glm.exe -m <ds4.gguf> -i testdata-deepseek4\01_prose.bin `
    -n 32 -t 16 -o results\out.bin

# best prefill - routed experts over the dies. The server comes from the
# separate Vulkan tree (build.ps1 -Vk) because every trunk binary aborts when a
# GPU device is registered; leave it running and use a second shell for the
# client. Expect ~90s to load and ~20s per die to upload.
build-vk\bin\moe-server.exe -m <ds4.gguf> --gpu-experts 240 --gpu-devices 4 -t 16
build\bin\nano-glm.exe -m <ds4.gguf> -i testdata-deepseek4\01_prose.bin `
    -n 32 -t 16 --moe-addr 127.0.0.1:5711 -o results\out.bin

# both of the above plus the forced-placement ladder, unattended
python split_study.py --model <ds4.gguf>
```

**No single configuration wins both**, and that is the standing tension rather
than an oversight: the split is 2.4x on prefill and 0.76x on decode, so which to
run depends on how much you generate per prompt token. On this workload's mix
(111 + 32) the split wins on aggregate, 4.43 against 3.12 t/s.

Every run prints the phase table (`lib/phase_timer.h`) and, over RPC, the MoE
call accounting, so a regression says *where* rather than only *that*.

### Comparing the two

Not directly, from the tables above: llama-bench measures pp128/tg32 with
weights in ordinary memory, nano-glm answers a 111-token prompt with weights
mmapped, and the load mode alone is worth ~1.45x of prefill. Dividing one column
by the other would fold a harness difference into an implementation claim.

The like-for-like comparison exists and is further down this file: `rescore
--sim-gen` runs llama.cpp over the *same token ids* in the same prefill-then-step
shape, which is what `bench_ds4.ps1 -Split` measures. On 143 positions of
`01_prose`, shipped llama.cpp does 21.0 s against nano-glm-plus-server's ~32 s —
so **roughly 65%**, and the sections below decompose what the remaining gap is
made of (kernels, weight residency, and a 1.9x that turned out to be in
nano-glm's own local expert path rather than anywhere exotic).

## Why this is a separate list

Every number below is measured on one machine, one model, one quantization, and
one workload shape: a single prompt answered once. That shape is narrow enough
that a projected "+12%" or "+40%" should not decide what gets built.

- **Multi-turn changes the economics.** Between turns there is think-time and
  user latency to hide expert movement behind, so a swap that looks unaffordable
  inside a token is free between them. And within a conversation the topic
  holds — the one strong transfer we measured was prose↔history, two English
  prose prompts about cities, at 45.6% against 23% for random. Same-conversation
  reuse is the case most likely to work and the one the single-turn study cannot
  see.
- **Quantization moves the thresholds.** Q4_K experts change both the bytes per
  token and how many fit in VRAM at once. 30% resident versus 40% resident is
  not a small difference when the curve is as steep as `ROUTING.md`'s.
- **Architecture moves everything.** A different model, a trunk on a GPU, a
  second machine, or experts on NVMe instead of DRAM each re-rank this list.
  The LRU result below is the clearest case: it loses badly against VRAM over
  PCIe and would win against DRAM over NVMe, on identical routing data.

So the benchmark's job (`nano-bench`, `PLAN.md` step 11) is **not to justify
optimizations — it is to catch pessimizations.** 1.932 tok/s at 75.2 GB/s with
a 0.4% spread is a number to defend while the infrastructure is built. A change
that quietly costs 10% should be visible; a change that might gain 10% can wait
until the thing works end to end.

## The backlog

Numbers are stable identifiers shared with `PLAN.md`; these moved here when the
plan narrowed to infrastructure.

### 12. Where the missing 25% goes

The largest and least understood gap. `nano-bench --pages` measures the model's
own access pattern sustaining ~100 GB/s with ordinary 4 KiB pages; the model
sustains 75.4. **A quarter of decode is not memory movement** and nothing
accounts for it.

Candidates, roughly by expected size:

- **Q6_K dequantization** — 210 bytes per 256 weights, unpacked on every read.
  The one candidate a lower-bit quant would change for a second reason.
- **Per-node barriers** — 78 layers x tens of graph nodes, each with a thread
  barrier. `ggml_barrier` at 16 threads does not scale with bytes.
- **Attention and the KV path**, which the byte budget excludes entirely.

Attribute before optimising: time the graph node-by-node, or run one MoE layer
against a raw stream of the same bytes. The answer decides whether the lever is
a kernel, a fusion, or nothing. This also retired the "expert FFN is ~95%+
memory movement" rule of thumb — the split is closer to 75/25.

### 9. Latency hiding around the shared expert

Issue the MoE request, run the shared expert on the client while it is in
flight, collect after. A `moe_send`/`moe_recv` split around the shared-expert
branch gets it for free: the CPU backend executes graph nodes in order, and the
other fifteen workers are already waiting out the round trip
(`lib/moe_client.h`).

Worth about the shared expert's own read — 3.05 GB over 75 layers is 40.7 MB,
which at 75.4 GB/s is ~0.54 ms against a 3.2 ms request, so of order 15% on
loopback and more over a real link. That is arithmetic from two measurements,
not a measured speedup; `nano-bench` could settle it once it speaks
`--moe-addr`.

**The largest confirmed win on this list**, and small: one graph restructuring,
no new subsystem. Must not change a bit — same corpus gate.

### 3-adjacent. What GPU offload is actually worth — measured

Step 3 shipped a Vulkan backend that can hold resident experts. The obvious
question — how much is that worth, and would more dies help — was unanswerable
from the real system, because `devices[0]` holds every expert and takes the
complement of whatever the GPUs take, pinning the CPU's share to `1 - residency`.

`moe-server --force-split a,b,c,d` (`TESTING.md`) breaks the coupling: slot *s*
of every token goes where the pattern says, regardless of routing, with the
expert remapped into that device's resident range. **The output is wrong by
construction**; the work has the right shape and cost, which is all a timing
needs. `04_history`, 151 fixed ids, k=52 over four dies, one discarded warm-up
and three measured passes:

| forced split | slots on CPU | mean | sd | vs CPU-only |
|---|---|---|---|---|
| CPU only | 8 | 37.9 s | 1.6 | 1.00x |
| `0,0,0,0` — dies idle | 8 | 35.4 s | 1.6 | 1.07x |
| `8,0,0,0` — one die | 0 | 20.2 s | 0.2 | 1.87x |
| `4,4,0,0` — two dies | 0 | 20.6 s | 0.1 | 1.84x |
| `2,2,2,2` — four dies | 0 | 20.0 s | 0.1 | **1.89x** |
| `2,2,2,1` | 1 | 22.1 s | 0.2 | 1.71x |

**One die is as fast as four.** 20.2 / 20.6 / 20.0, with sd 0.1-0.2 — equal, and
tightly enough measured to mean it. A single Vega absorbs the entire expert
workload of a 151-token prefill at the speed four do sharing it, so the dies are
nowhere near throughput-bound; they are bound by something fixed per dispatch
(../moe-offload measured ~190 us for a null dispatch). **Adding dies buys VRAM
capacity, not speed** — which is the opposite of the intuitive reading of
"expert parallelism", and worth knowing before anyone buys hardware for it.

**Time is linear in the slots left on the CPU**, at 2.24 s per slot
((37.9 - 20.0)/8). The model `T ~= 20.0 + 2.24 x cpu_slots` predicts the
one-slot case at 22.2 s against 22.1 s measured. So offload pays *strictly in
proportion* to how much work leaves the CPU — no threshold, no cliff.

**And that is the discouraging part.** Only ~22% of experts fit in 125 GiB of
VRAM at Q6_K, so cpu_slots ~= 6.2 and the model predicts ~1.11x — consistent
with the 1.06x measured on the real residency path (`PLAN.md` step 3). Q4_K
experts would reach ~33% residency and ~1.18x. The 1.89x ceiling needs
essentially every expert resident, i.e. ~4.5x more VRAM than exists here.

Consequences for this list (revised after the decode measurement below):
- **More dies: marginal.** Nothing in prefill, +4.9% in decode. Buy them for
  VRAM capacity, not for parallelism.
- **Smaller expert quant: yes**, and it now has numbers attached in both
  regimes.
- **Fewer, larger dispatches** is the lever on the GPU side: one die at 4x the
  prefill work costs the same, and decode is dispatch-bound enough that four
  dies help. Fusing up+gate, or a layer at a time, would test it.
- The ceiling is Amdahl on the trunk — it still runs on the client CPU and is
  ~40% of prefill and ~50% of decode (measured; see the correction below).
  `PLAN.md` step 4 (trunk on GPU) is the only thing that moves it.

Everything above is **prefill** (151 tokens in one batch). Decode was measured
next and disagrees with it in two places, so read the two together.

#### Decode, and two corrections to the above

14-token prompt + 32 generated, one token per step, same k=52 over four dies.
Reported as tok/s because forced-split output is garbage and could in principle
trip EOS early (it did not — every run generated all 32):

| forced split | slots on CPU | tok/s | sd | vs CPU-only | ms/token |
|---|---|---|---|---|---|
| CPU only | 8 | 1.317 | 0.005 | 1.00x | 759 |
| `8,0,0,0` one die | 0 | 1.793 | 0.005 | 1.36x | 558 |
| `2,2,2,2` four dies | 0 | 1.880 | 0.008 | **1.43x** | 532 |
| `2,2,2,1` | 1 | 1.857 | 0.005 | 1.41x | 538 |
| `1,1,1,1` | 4 | 1.633 | 0.005 | 1.24x | 612 |

**Correction 1: "one die is as fast as four" is prefill-only.** In decode four
dies beat one by 4.9% (1.880 vs 1.793), which is far outside sd 0.005-0.008.
The reason is the same fixed-dispatch story seen from the other side: a decode
layer gives each die *two pairs*, so there is essentially no work to be
throughput-bound by and four concurrent dispatches genuinely overlap. Prefill
hands one die enough work that a second adds nothing.

**Correction 2: the CPU cost is sublinear in decode, not linear.** Marginal
cost of a slot left on the CPU: the **1st costs 6.6 ms**, slots 2-4 about
20 ms each, slots 5-8 about 37 ms each. Against prefill's flat 2.24 s per slot,
that curve is the thread-per-device overlap working — the first CPU slot hides
almost entirely behind the GPUs' work and only becomes visible once the CPU's
share exceeds theirs. Keeping *a few* experts on the CPU is close to free;
keeping most of them is not.

**And the prediction in the previous version of this section was wrong.** It
said decode "should help *more*" because decode has worse arithmetic intensity
on the CPU. It helps **less**: 1.43x against prefill's 1.89x. The reason is
that decode's CPU-side MoE is memory-bound, where the GPU's edge is only VRAM
versus DRAM (~4.3x here), while prefill's CPU-side MoE is compute-bound, where
the GPU's edge is much larger. Worth keeping as a reminder that "memory-bound
work benefits more from offload" does not follow.

#### Correction: the trunk split, measured rather than inferred

An earlier version of this section put the trunk at ~53% of prefill and ~61% of
decode. **Both were wrong**, from treating the full-offload wall time as if it
were the trunk when it still contains several seconds of GPU MoE. The client's
own RPC accounting gives it directly — `MoE RPC: ... total = Xs server + Ys
network` for prefill, and `rtt p50` x 75 layers for decode:

| | wall / token | MoE (RPC) | trunk | trunk share |
|---|---|---|---|---|
| prefill, CPU only | 37.9 s | 22.3 s | 15.6 s | |
| prefill, `2,2,2,2` | 20.0 s | **4.9 s** | 15.1 s | **~40%** |
| decode, CPU only | 759 ms | 371 ms | 388 ms | |
| decode, `2,2,2,2` | 532 ms | **155 ms** | 377 ms | **~50%** |
| decode, `8,0,0,0` | 558 ms | 177 ms | 381 ms | |
| decode, `1,1,1,1` | 612 ms | 234 ms | 378 ms | |

The trunk lands at 377-388 ms across four independent decode configurations,
which is what makes the decomposition trustworthy.

**The GPU MoE is not free**, and it is worth being explicit because the shape
of the numbers invites that reading. It is 25% of the offloaded prefill and 29%
of the offloaded token. What offload buys is **4.6x** on prefill MoE and
**2.4x** on decode MoE. Decode works out at ~2.1 ms/layer including router and
transfer, the same ballpark as ../moe-offload's independently measured 1.11 ms
for a fully-resident layer.

With a *completely* free MoE the ceilings would be 2.5x (prefill) and 2.0x
(decode); the measured 1.89x and 1.43x are what a 4.6x/2.4x MoE actually gives.

Note "trunk" is not only attention: it is MLA plus the DSA lightning indexer,
**the shared expert** (which stays on the trunk by design, `moe_block.h`), the
154880-wide output head, embeddings and norms. The shared expert is itself an
offload candidate and nothing has measured it.

At realistic residency (~22%, so ~6.2 slots left on the CPU) the curve gives
~694 ms, i.e. **~1.09x** — in line with the ~1.11x the prefill model predicts,
and with the 1.06x measured on the real residency path.

### 3-adjacent. Better expert placement

Step 3 itself (a Vulkan backend that can hold resident experts) is
infrastructure and lives in `PLAN.md`. *Which* experts to hold is optimization,
and the current answer is discouraging in a specific, escapable way:

| resident 23% chosen by | hit rate | decode |
|---|---|---|
| a within-prompt oracle | 58.4% | 2.47 tok/s (+27%) |
| leave-one-out, deployable | 28.2% | 2.16 tok/s (+12%) |
| picking at random | 23.1% | 2.12 tok/s (+9%) |

All the routing intelligence is worth three points over random
(`ROUTING.md`). Escapes worth trying, in order:

- **Per-conversation placement.** Routing is domain-specific rather than
  universal. A backend that observes the first turn and swaps during think-time
  is a different proposition from a placement fixed at boot.
- **Q4_K experts** — a third off both DRAM traffic and resident size, moving
  along the residency curve rather than up it.
- **After a trunk move** (`PLAN.md` step 4) the MoE bytes are the whole cost,
  so the same hit rate is worth proportionally more.

### 8. Huge pages for the expert store

Probed and parked. The mechanism was plausible — 31.5 MB per expert against a
1536-entry L2 TLB, and Intel's L2 streamer not prefetching across a 4 KiB page
boundary — but `nano-bench --pages` found 4 KiB pages already reaching ~100 GB/s
at 16-32 threads, with shuffled expert-sized blocks matching a flat sequential
sweep. The headroom the mechanism was invoked to explain is not there.

Unfinished: the 2 MB arm needs `SeLockMemoryPrivilege` ("Lock pages in memory"
in `secpol.msc`, then log out and in). Expect confirmation rather than a
surprise. Would also need the non-mmap load path, which nothing else wants now —
and that is not incidental: **neither Windows nor macOS offers huge pages for
file-backed mmap at all**, so there is no version of this that keeps the current
loader.

### Caching instead of placement

Measured and rejected for this tier pair (`cache_sim.py`). LRU wins the hit
rate outright — 63.3% at 23% residency against 28.2% for the best deployable
static placement, beating even a within-prompt oracle, because recency beats
any fixed ranking. It loses anyway: a static miss is a DRAM read, a cache miss
is a DRAM read *and* a PCIe install, so each miss costs 6.8x more. Break-even
needs an 89.4% hit rate; LRU delivers 63.3% and comes out 3.5x slower than
static. LFU is worse and *decays* over a run, 63% to 41%, as early-hot experts
ossify.

**Carry the reasoning, not the verdict.** A cache pays when installing into the
fast tier is cheap relative to the miss it saves. Here the install crosses the
very link that makes misses expensive. With experts on NVMe and DRAM as the
cache — Kimi-K3's shape — an install is a DRAM write against a miss costing ten
times more, and LRU's advantage would convert directly into time saved. The
policy does not transfer between tiers; the bandwidth ratio decides.

### Speculative routing

With the router on the backend, run layer N+1's router on layer N's activation
while idle and prefetch the likely experts. Buys nothing while experts are in
DRAM — bandwidth-bound, and prefetch adds no bandwidth — and prefetching into
VRAM has the same install problem as caching into it. Its home is the NVMe
tier.

`ROUTING.md` puts a floor under the predictor half: keeping the previous
token's experts warm already hits 35% against a 3% base rate, free, with no
speculative router at all. Anything clever has to beat that, not uniform.

### Backend micro-optimisations

Each is listed with **the condition that would make it matter**, because the
condition is the useful part: the backend gets faster in later phases and
anything at 1% becomes 10% once compute drops 10x. The percentages below were
taken against a warm CPU backend at **3072 us/layer**; GPU-resident experts have
since moved that to ~2100 us/layer on decode, so the denominators are already
shrinking.

- **Graph cache** keyed on `(layer, n_tokens)`. Building the graph cost
  16-43 us/request when it was one ~15-node graph — 0.5-1.4% then, and it would
  be 5-14% at ~300 us/layer. Cost of doing it: 75 layers x every batch shape of
  compute buffers held resident, which is why it was not done up front.

  **Both sides of that ratio have moved and it is now unmeasured.** A layer is
  no longer one graph: it is a router graph plus one per device, so `N+1` per
  layer — five at four dies. And the denominator fell to ~2100 us. Re-measure
  before acting; `t_route_us` no longer reports construction time either, so the
  old instrumentation will not give the answer.

- **Zero-copy request/response.** Two 24 KiB memcpys per request: recv into
  `in_buf` then `tensor_set`, `tensor_get` into `out_buf` then send. Could recv
  straight into the allocated input tensor and send straight from the output
  tensor. Sub-microsecond today; matters only once per-request cost approaches
  the transfer cost.

- **Fusing up+gate.** They are independent — both read only `x` — but sit in
  separate graph nodes with a barrier between them, so their weight streams do
  not overlap. Fusing would double memory-level parallelism and drop a barrier.
  Speculative: needs a custom op and the win is unknown. Worth trying while we
  are still at ~57% of theoretical bandwidth. The GPU measurements give this a
  second motive — decode is dispatch-bound enough that four dies help, so fewer
  and larger dispatches is the lever on that side too.

- **Large pages for the expert store.** Neither Windows nor macOS offers huge
  pages for *file-backed* mmap, so this can only exist in the non-mmap load
  path. 4 KiB pages break the L2 streamer about once per output row and cost
  ~7900 TLB entries per expert. See "8. Huge pages" above: the mechanism was
  probed and the headroom it was invoked to explain is not there.

- **f16 on the wire.** Halves transfer, costs exactness — measure with
  `compare.py`, do not assume.

Recorded as *not* worth doing, so it is not re-derived: **request pipelining**.
The trunk is strictly sequential — layer i+1 needs layer i's output — so one
sequence never has more than one request in flight. Concurrency here only pays
if the backend serves several independent sequences at once.

## DeepSeek-V4-Flash: what the split costs, and where the gap to llama.cpp is

PLAN.md step 14 chose this model for a regime GLM-5.2 cannot reach: 150.7 GiB,
so its experts nearly fit on the four Vega dies. Measured at **93.75% expert
residency** (240 of 256 experts per layer, 40 served layers, 119.5 GiB on the
GPUs), one prompt of 111 tokens plus 32 generated, warm, two passes agreeing
within 2%:

| setup | prefill | decode | 143 positions | aggregate |
|---|---|---|---|---|
| llama.cpp **as shipped** (repack ON) | | | 21.0 s | **6.81 t/s** |
| llama.cpp, repack OFF (nano-glm's kernels) | | | 35.4 s | 4.04 t/s |
| nano-glm, all local | 3.8 t/s | 1.97 t/s | 45.4 s | 3.15 t/s |
| nano-glm + moe-server, 93.75% on GPU | **8.3 t/s** | **1.50 t/s** | 34.7 s | **4.12 t/s** |

**The split helps prefill and hurts decode.** Prefill 2.2x, decode 0.76x — the
first time an offload measurement here has gone *negative* on decode. It is not
a contradiction of the earlier 1.43x decode ceiling: that was GLM-5.2 at 22%
residency, whose experts are 31.5 MB each. DeepSeek's are 12.75 MB, so the
per-layer work shrank while the per-layer round trip did not, and 40 sequential
RPCs per token now cost more than the GPU saves. Whether a real two-machine
topology moves this either way is unmeasured: loopback has no network latency
but does share CPU and memory bandwidth between the two ends.

Net on this workload it is a 1.31x win over local — and still **60% of shipped
llama.cpp**.

**Two corrections from the forced-split study below, neither of which changes
the numbers in that table but both of which change what they mean.** Prefill's
2.2x is mostly *not* the GPUs: 1.80x of it is the server's expert path measured
with the dies deliberately idle, and only 1.30x is the dies. And the decode
figure has since widened rather than narrowed — the allocator fix took the
*local* baseline from 1.97 to 2.195 t/s, so the split now sits at 0.77x of a
faster thing.

### What the dies are actually worth here — and the 1.9x that is not them

`split_study.py` runs the ladder `moe-server --force-split` makes possible, and
it is deliberately **not** the study GLM-5.2 got. There, forced placement existed
to reach distributions residency could not: at ~22% resident, the real system
could never put every slot on a die. DeepSeek-V4-Flash is the opposite case — at
240 of 256 experts resident (93.75%) a token's 6 slots already land on a die
~5.6 times out of 6 — so the interesting direction is *downward*, forcing slots
back onto the CPU to price one.

`01_prose`, 111 prompt + 32 generated, one discarded warm-up and two measured
passes per rung, `--gpu-experts 240 --gpu-devices 4`:

| configuration | prefill | decode | rtt p50 |
|---|---|---|---|
| CPU only, no server | 3.55 | **2.195** | — |
| natural (routing decides) | 8.45 | 1.670 | 2073 us |
| `--force-split 2,2,1,1` — 0 slots on CPU | 8.30 | 1.685 | 2076 us |
| `--force-split 1,1,1,1` — 2 on CPU | 7.95 | 1.655 | 2074 us |
| `--force-split 1,1,0,0` — 4 on CPU | 6.90 | 1.605 | 2532 us |
| `--force-split 0,0,0,0` — dies idle | 6.40 | 1.540 | 3235 us |

**Natural routing already equals forced full placement** (8.45 vs 8.30 prefill,
1.670 vs 1.685 decode, both inside the pass-to-pass spread). At 93.75% residency
there is nothing left for more VRAM to buy on this model — which is the question
"would more dies help" asked and answered from the other end than the GLM study
asked it.

**And most of the prefill win is not the GPUs.** The `0,0,0,0` rung holds every
expert in VRAM and sends no work there, so it isolates the split's own path:

| | prefill | vs local |
|---|---|---|
| local | 3.55 | 1.00x |
| split, **dies idle** | 6.40 | **1.80x** |
| split, all six slots on dies | 8.30 | 1.30x *more* |

So of the 2.2x this file previously credited to offload, **1.80x is the server's
expert path and only 1.30x is the four Vega dies.**

The client's own instrumentation says where that goes, without a subtraction
across runs: `lib/phase_timer.h` gives total compute and `moe_stats` gives the
RPC total inside it, so trunk and experts separate directly.

| | client trunk | expert work | total compute |
|---|---|---|---|
| local | ~26.5 s | **18.7 s** | 45.21 s |
| split, dies idle | 26.9 s | **9.9 s** (server CPU) | 37.16 s |
| split, all on dies | 26.1 s | **5.0 s** (four dies) | 31.37 s |

The trunk is unchanged across all three, as it must be. **The same expert
arithmetic takes 18.7 s in the client and 9.9 s in the server, on the same
cores** — and the dies then take that 9.9 s to 5.0 s. The 1.9x is a pure
software gap with no GPU and no network in it, and it is the largest single
lever this file has ever identified. The obvious suspect is that `moe-server`
compacts rows per expert before its matmuls while the client's local block runs
`ggml_mul_mat_id` over the trunk graph, but that is a hypothesis: this
measurement establishes the gap, not its cause.

Two consequences. It is worth more than the split for most workloads, because
**decode never beats local anyway** — the best split rung is 1.685 against
local's 2.195 (0.77x), and closing a 1.9x on local experts moves the number that
is already winning. And it means the split's headline prefill figure has been
crediting the hardware for something the software was doing.

**Marginal cost of a slot the CPU has to serve**, from the ladder: **~9.3 ms per
slot per decode step**, and rising — 5.4 ms/slot over the first two, 9.5 over the
middle two, 13.2 over the last two. Same increasing shape GLM-5.2 showed
(6.6 ms for the first, ~20 for slots 2-4, ~37 for 5-8), which is the
thread-per-device overlap working: the first CPU slots hide behind the dies'
work and only become visible once the CPU's share exceeds theirs. Prefill runs
~0.66 s per slot over the range, too noisy at two passes to claim curvature.

### A decode step, fully accounted — and the 8-pair cliff

`lib/phase_timer.h` separates prefill from decode, `--moe-log` records the
server's own stages per call (`moe_log_stats.py` reads it), and `moe-bench`
times the expert kernel with synthetic weights and no model. Together they close
a decode step. Per generated token, `01_prose`:

| | local | split |
|---|---|---|
| client host-side (build/alloc/input/read/free) | 6.9 ms | 13.3 ms |
| client compute | 436.7 ms | 576.1 ms |
| — of which RPC, 40 calls x 2.08 ms | — | 83.3 ms |
| **total** | **443.6** | **589.4** |

and inside those 83.3 ms: **route 8.0, server compute 69.0, network+scheduling
6.2**, parse and serialize ~0.

**The network is not the problem.** 6.2 ms of 589. A free interconnect changes
almost nothing, so the "40 sequential RPCs per token" framing this file has
carried was aimed at the wrong term.

**The trunk inflates by ~103 ms doing identical work.** Subtracting round trips
leaves a 492.8 ms trunk in the split case, against a local *total* compute of
436.7 ms that already includes >=41 ms of expert reads (40 layers x 6 experts x
12.75 MB = 3.06 GB at 75.4 GB/s). So the split's decode penalty is
**71% trunk inflation, 29% round trips**:

```
  589.4 - 443.6 = +145.8 ms
     -41  expert work leaves the CPU
     +83  round trips
    +103  client trunk, same work
```

It is the *interaction*, not the second process. Running the client **locally
while an idle server holds 150 GiB and 120 GiB of VRAM** gives 436.8 ms/token —
if anything faster than the 443.6 with no server at all. So this does not go
away on a second machine. The leading suspect is ggml's threadpool spinning all
sixteen cores through each of the forty blocking calls; that is what step 9's
`moe_send`/`moe_recv` overlap would remove, which promotes it from "the largest
confirmed win, and small" to the largest lever on decode.

**And the server's 69 ms of compute is mostly not the GPU.** `moe-bench` runs
the same five-op graph `run_device_compact` builds, on one die with the pair
count a die actually receives at decode:

| pairs per die | per dispatch | GB/s |
|---|---|---|
| 1 | 373.6 us | 35.8 |
| 2 | 451.6 us | 59.2 |
| 4 | 599.1 us | 89.3 |
| **8** | **857.0 us** | **124.3** |
| **9** | **8780.4 us** | — |
| 16 | 12400.6 us | 17.2 |
| 166 (a 111-token prefill) | 78802.0 us | 28.2 |

At decode a die gets 1-2 pairs, so the kernel costs 374-452 us x 40 layers =
**15-18 ms per token against the 69 ms the server charges**. ~50 ms/token is
server-side overhead around the dispatch — thread fan-out and join per layer,
the gather into `gathered_x`, the scatter-add combine — and none of it has been
measured yet.

**The cliff is one line of llama.cpp.** `ggml_vk_use_mul_mat_vec_id` is
`src2->ne[1] <= 8` (`ggml-vulkan.cpp:10607`); `src2` is the id tensor and its
`ne[1]` is exactly the pair count. At 8 pairs Vulkan runs `mul_mat_vec_id`, at 9
it switches to the tiled `mul_mat_id_q_f16`, and on these dies — which report
`matrix cores: none`, so no coopmat path exists — the tiled kernel is **10.2x
slower for one extra pair**.

Prefill is entirely on the wrong side of it. A die handed 166 pairs takes
78.8 ms; the same work as 21 dispatches of 8 would be 21 x 857 us = **18 ms, a
4.4x**, and the server already has the pair list in hand to chunk it. That is
the largest single number on this page and it needs no new kernel — only a cap
on how many pairs go into one `mul_mat_id`.

It also explains why the dies were only worth 1.30x on prefill: they were
running their bad path for every prefill-sized batch.

### Where the other 40% is, and why it is mostly not the kernels

The obvious suspect is `GGML_CPU_REPACK`, which nano-glm cannot use: it rewrites
quantized weights into a blocked layout at load and runs a different GEMM.
Turning it off costs llama.cpp 1.69x on the workload above, which looks like the
whole story and is not. Repacking **allocates** its 140,352 MiB of converted
weights, so it also moves them out of the file mapping into ordinary memory.
Those are two effects and they separate cleanly — same binary, `llama-bench -r
5`, only the load mode changing:

| repack OFF | pp128 | tg32 |
|---|---|---|
| mmap | 12.41 ± 1.29 | 2.91 ± 0.02 |
| `-lm none` (ordinary memory) | 17.99 ± 0.09 | 3.36 ± 0.02 |
| **residency alone** | **1.45x** | **1.15x** |

and then, both in ordinary memory, only the flag changing:

| `-lm none` | pp128 | tg32 |
|---|---|---|
| repack OFF | 17.99 ± 0.09 | 3.36 ± 0.02 |
| repack ON | 22.16 ± 1.44 | 3.73 ± 0.04 |
| **kernels alone** | **1.23x** | **1.11x** |

So for prefill the *larger* half of llama.cpp's advantage is not its kernels at
all — it is that repacking incidentally stops the weights being mmap-backed.

That matters because the two halves have different prices. The kernel half is
closed to nano-glm by construction: a repacked layout is a second copy of the
weights, which a 583 GiB model and a remote-expert design cannot afford. **The
residency half is not.** Nothing stops nano-glm reading weights into ordinary
memory when the model fits, and this machine has 768 GB against DeepSeek's 150.7
GiB. `llama-bench --load-mode none` already showed the same effect on GLM-5.2
(1.90 ± 0.08 steady against 1.04-1.84 mmap-backed) and it was filed there as a
*measurement* fix; it is also a throughput one.

Not proposed for building yet, and deliberately: it helps exactly the models
small enough not to need this project, and Kimi-K3 at ~1.5 TB — the reason the
experts live elsewhere — cannot use it at all. What it does is stop the gap to
llama.cpp being attributed to the split.

### Caveats on the numbers above

- **Aggregate hides the split.** 4.12 vs 4.04 t/s reads as parity with
  repack-off llama.cpp and is a 2.2x prefill win against a 24% decode loss. The
  ratio of the two depends entirely on how many tokens are generated; at 111
  prompt and 32 generated it lands near parity, and a longer generation would
  make the split lose.
- **One prompt, one length.** `01_prose` only. The prefill/decode balance is the
  whole result, so a corpus sweep would say more than more repetitions of this.
- ~~**nano-glm rebuilds its graph every chunk**~~ — **settled, and it was not the
  cause.** `lib/phase_timer.h` splits a chunk into build / alloc / input /
  compute / read / free, always on, and building the ~6000-node graph is **0.4%**
  of a run (5.7 ms against a 1452 ms chunk). Caching it would buy nothing, and
  it is harder here than in glm-dsa anyway: the shape depends on the compressor
  plans as well as on (n_tokens, n_kv), because a ratio-4 block closes on one
  decode step in four.

  What the same profile did find is the reverse of the guess. **`free` was
  37 ms/chunk** — the largest host-side cost and four times the 9 ms `alloc` it
  undid — because `ggml_gallocr` was constructed and destroyed per chunk,
  releasing the compute buffer only to re-commit it. Keeping the allocator in
  `ds4_state` removes it. Two binaries from one tree differing in that alone,
  run **alternately, three times each** on `01_prose` (111 prompt + 32
  generated):

  | | alloc+free | compute | prefill | decode |
  |---|---|---|---|---|
  | `ggml_gallocr` per chunk | 1.356 s | 46.26 s | 3.5 t/s | 1.947 t/s |
  | kept in `ds4_state` | 0.099 s | **44.77 s** | 3.6 t/s | **2.217 t/s** |
  | | | | | **+13.9%** |

  All six outputs byte-identical to the golden set, and no overlap between the
  two sets of three on any column.

  **Half of the win is somewhere no phase timer could have attributed it.**
  `compute` is 45 ms/chunk lower with the allocator reused, against 38 ms/chunk
  of `alloc`+`free` removed directly. Releasing the compute buffer every chunk
  means the next chunk soft-faults it back in on first touch, and those faults
  land inside `ggml_backend_graph_compute`. The phase table could only show the
  direct half; the A/B was needed for the rest, which is an argument for keeping
  both tools rather than either.

  Worth keeping as a general lesson: **the expensive host-side thing was not the
  one that looked expensive.** 6000 nodes of graph construction sounds costly and
  is not; freeing a compute buffer sounds free and is not.

### llama.cpp's own GPU offload on the same model, and the flag that decides it

The section above measures nano-glm's split — routed experts behind a socket,
one dispatch per layer — against llama.cpp running everything on the CPU. The
obvious control was never run: **what does llama.cpp do with these four GPUs?**
It has a Vulkan backend and 124.7 GiB of free VRAM (four dies, 31.17 GiB each)
to point at a 150.7 GiB model, so five sixths of it could be resident.

`build_bench.ps1 -Vulkan` builds a third llama-bench differing from the repack-ON
one only in `GGML_VULKAN`; `bench_ds4.ps1 -Vulkan` runs the configurations. All
`-lm none -t 16 -r 5 -p 128 -n 32`, on the four Vega II dies:

| configuration | VRAM | splits @bs128 | pp128 | tg32 |
|---|---|---|---|---|
| CPU-only build, repack ON | — | — | **22.16** ± 1.44 | 3.73 ± 0.04 |
| CPU-only build, repack OFF | — | — | 17.99 ± 0.09 | 3.36 ± 0.02 |
| `-ngl 0` | 0 | 1780 | 5.37 ± 0.14 | 3.39 ± 0.02 |
| `-ngl 12` | 39.1 GiB | 968 | 9.07 ± 0.37 | 2.39 ± 0.02 |
| `-ngl 24` | 80.6 GiB | 602 | 10.64 ± 0.63 | 2.58 ± 0.02 |
| `-ngl 32` | 108.3 GiB | 358 | 11.87 ± 0.45 | 3.12 ± 0.01 |
| `-ngl 34`, `-ngl 36` | — | — | OOM during load | |
| `-ngl 0 -nopo 1` | 0 | — | 17.96 ± 0.14 | 3.41 ± 0.02 |
| `-ngl 32 -nopo 1` | 108.3 GiB | 10 | 16.68 ± 0.59 | 3.08 ± 0.04 |
| `-ncmoe 43` | 12.7 GiB | 169 | 8.55 ± 0.36 | 4.24 ± 0.02 |
| `-ncmoe 43 -nopo 1` | 12.7 GiB | 94 | 15.91 ± 0.49 | 4.20 ± 0.02 |
| `-ncmoe 36 -nopo 1` | 35.0 GiB | 80 | 15.37 ± 0.31 | 4.46 ± 0.04 |
| `-ncmoe 30 -nopo 1 -ts 30/3/3/7` | 55.4 GiB | 68 | 11.58 ± 0.14 | 3.58 ± 0.02 |
| — same flags, second load | 55.4 GiB | 68 | 11.08 ± 0.05 | 3.94 ± 0.06 |
| `-ncmoe 24 -nopo 1 -ts 24/6/6/7` | 75.0 GiB | 56 | 14.54 ± 0.28 | 4.64 ± 0.02 |
| `-ncmoe 19 -nopo 1 -ts 19/8/8/8` | 95.2 GiB | 46 | 15.05 ± 0.35 | 3.65 ± 0.02 |

**Read the `-ncmoe` rows as a band, not an ordering.** The repeated row is there
because the ordering looked meaningful and is not: the same configuration,
same flags, same buffer allocation to the MiB and the same 68 splits, decoded at
3.58 ± 0.02 on one load and 3.94 ± 0.06 on the next. **10% load-to-load, against
within-run sds of 2-6%.** `-lm none` was adopted to kill exactly this kind of
variance and it only kills it *within* an invocation; every other row here is a
single load, so differences below ~10% between them are not evidence. The
`-ncmoe` family lands somewhere in **3.6-4.6 t/s** and which member is best was
not determined — note that the largest VRAM configuration measured (`-ncmoe 19`,
95.2 GiB, near the ceiling) is *not* the fastest, which is the same
"adding dies buys capacity, not speed" result as the forced-split study above.

What survives that caveat is everything the rest of this section rests on: the
`-ngl` family (2.4-3.1) and the `-ncmoe` family (3.6-4.6) do not overlap, and
the `op_offload` effect is a factor of 3.3.

Four things come out of it, in rough order of how much they cost to learn.

**1. `op_offload` costs 3.3x of prefill with nothing offloaded, and one flag
recovers all of it.** `-ngl 0` on the Vulkan binary prefills at 5.37 t/s against
the CPU-only build's 17.99 — same source, same flags but `GGML_VULKAN`, no layer
deliberately on a GPU. `-nopo 1` restores it exactly: 17.96 ± 0.14. The
mechanism is two lines. `ggml-backend.cpp:959` offers any op whose **weights sit
in a host buffer** to a higher-priority backend, and
`ggml-vulkan.cpp:18511` accepts once the op's batch reaches
`op_offload_min_batch_size` (default 32, `GGML_OP_OFFLOAD_MIN_BATCH`). DeepSeek-V4
has two ops Vulkan cannot run at all — `DSV4_HC_COMB` on every layer,
`LIGHTNING_INDEXER` on the ratio-4 layers, both present in CPU, CUDA and Metal —
so every accepted op is dragged straight back and prefill shatters into 1780
graph splits. Decode, at batch 1, is below the threshold and untouched: 1 split,
3.39 t/s. This is the same scheduler policy the repo `CLAUDE.md` records for
Metal on this machine, where it produced NaN logits rather than a slowdown.

Two consequences worth separating. The split count is the *diagnostic* — it
tracked prefill inversely across every configuration measured, 1780 splits to
5.37 t/s and 10 splits to 16.68 — and the reason `bench_ds4.ps1` now records it
beside each timing. And a Vulkan-enabled build is **not** a superset of a CPU-only
one: merely registering the backend is a large regression on a model with
CPU-only ops, whatever `-ngl` says.

**2. `-ngl` is the wrong knob for this architecture.** 108.3 GiB of weights on
the GPUs (`-ngl 32`) decodes *worse* than 12.7 GiB (`-ncmoe 43`): 3.12 against
4.24. Every `-ngl` value tested decodes worse than `-ngl 0`. The reason is the
model's shape — 3.19 of the 3.46 GiB per layer is routed experts — combined with
the finding two sections up that a decode-step expert dispatch is too small to
pay for itself, while a 128-token one is. `-ngl` cannot express "attention yes,
experts no"; `-ncmoe` can, and `-ngl 99 -ncmoe 43` puts every layer's attention
on the GPU for a tenth of the VRAM.

**3. Nothing beats the CPU at prefill; decode gains roughly 1.1-1.2x.** The best
prefill measured is `-ngl 32 -nopo 1` at 16.68, still 0.75x of the CPU-only
repack-ON build, and no configuration came close to 22.16. Decode is the only
place the GPUs pay: the `-ncmoe` band is 3.6-4.6 against 3.73, so **about
1.0-1.24x** with the load-to-load spread included. The two halves point opposite
ways, so the answer depends on the workload:

```
CPU-only:                   P/22.16 + G/3.73
-ncmoe, decode at 4.64:     P/14.54 + G/4.64     faster once  G > 0.45 x P
-ncmoe, decode at 4.00:     P/15.37 + G/4.00     faster once  G > 1.10 x P
```

So the GPUs win once you generate somewhere between half and one token per
prompt token, and the width of that range is the load-to-load variance, not a
modelling choice. On `01_prose` (111 prompt, 32 generated) the ratio is 0.29 and
the CPU wins under either. On a chat-shaped 20-prompt/200-generated it inverts
under both. That is the same prefill-versus-decode tension the split measurement
has, arrived at from a completely different direction, which is mild evidence it
is a property of the model rather than of either implementation.

**4. `-ncmoe` and multi-GPU layer splitting interact badly.** `-ncmoe 30`, `24`,
`20` and `12` all abort during load while `-ncmoe 36` fits in 35.0 GiB of 124.7.
The `-v` log says why: llama.cpp splits *layers* across devices evenly, but
`-ncmoe` makes per-layer size wildly uneven (3.46 GiB carrying its experts, 0.27
without), so at `-ncmoe 36` all seven expert-carrying layers landed on Vulkan3
(26.02 GiB) while Vulkan0-2 held 2.98 GiB each. `-ts` moves the boundaries to
compensate and the aborting configurations then load — note it is
**slash**-separated (`-ts 30/3/3/7`), because llama-bench reads a comma as "run
this configuration once per value", so `-ts 30,3,3,7` silently becomes four
single-device runs that put everything on Vulkan0 and fails with an OOM naming
the device you meant to leave empty.

Worth saying plainly that this only buys the *ability* to use the VRAM, not a
result: `-ncmoe 19 -ts 19/8/8/8` reaches 95.2 GiB resident, the most this model
can use here, and decodes at 3.65 — below the 75 GiB and 35 GiB configurations.

**None of this can happen to nano-glm**, and the reason is structural rather
than lucky. `models/deepseek4/eval.h` computes on a single CPU backend with
`ggml_backend_graph_compute` — there is no `ggml_backend_sched`, so there is no
`op_offload` heuristic and no device transition to pay for. `models/glm_dsa`
does build a scheduler, and *aborts* if any GPU device is present. The GPU is
reached only through `moe-server`, as one explicit dispatch per layer carrying
work chosen by the router. This measurement is the case for that design: given
the same hardware and a far larger VRAM budget, the general-purpose scheduler
reaches 0.75x prefill and 1.0-1.24x decode, where the targeted split reaches
2.2x and 0.76x. Note the two disagree about *which half* the GPU helps, and
they are not measuring the same thing — llama-bench's pp128/tg32 against a 111+32
workload — so the honest reading is that neither approach makes this model fast
on this hardware, and each is bounded by a different thing: the scheduler by
graph splits and dispatch granularity, the split by 40 sequential RPCs per
token.

## Housekeeping that is not optimization

Small, unglamorous, and cheap to fold into whatever touches the file next:

- `per_slot` return mode is declared in the protocol and unimplemented.
- `moe_client::rtt_us` and `::log` grow without bound.
- `elapsed_us` returns `uint32_t` — about 71 minutes of microseconds.
