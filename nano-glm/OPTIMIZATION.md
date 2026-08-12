# Optimization backlog

Work that would make remote MoE *faster*. None of it is on the critical path in
`PLAN.md`, which is about making it *work*.

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
- **nano-glm rebuilds its graph every chunk**, including once per decode token —
  ~6000 nodes for 43 layers, where glm-dsa caches by (n_tokens, n_kv).
  `models/deepseek4/eval.h` says it is "left out until there is a measurement
  saying it matters". This is the first measurement that could, and it does not
  separate that cost from the rest; it remains a candidate for the local-path
  gap, not a demonstrated cause.

## Housekeeping that is not optimization

Small, unglamorous, and cheap to fold into whatever touches the file next:

- `per_slot` return mode is declared in the protocol and unimplemented.
- `moe_client::rtt_us` and `::log` grow without bound.
- `elapsed_us` returns `uint32_t` — about 71 minutes of microseconds.
