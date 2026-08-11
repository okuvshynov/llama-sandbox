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

Consequences for this list:
- **More dies: no.** Capacity only.
- **Smaller expert quant: yes, linearly**, and it now has a number attached.
- **Fewer, larger dispatches** is the lever on the GPU side, since one die at 4x
  the work costs the same. Fusing up+gate, or a layer at a time, would test it.
- The 1.89x ceiling is Amdahl on the trunk: attention still runs on the client
  CPU and is ~53% of wall time here. `PLAN.md` step 4 (trunk on GPU) is what
  moves it.

Caveat: measured on **prefill** (151 tokens in one batch), where the CPU reads
each expert once and amortizes it across the tokens that routed there. Decode
has far worse arithmetic intensity on the CPU side, so offload should help
*more* there. Not yet measured; do that before generalizing these ratios.

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
surprise. Would also need the non-mmap load path, which nothing else wants now.

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

All recorded in `apps/moe-server/main.cpp` with the condition that would make
each matter, because the condition is the useful part — the backend gets faster
in later phases and a 1% cost becomes 10% when compute drops 10x.

- **Graph cache** keyed on (layer, n_tokens): 16-43 us/request today, 0.5-1.4%.
- **Zero-copy request/response**: two 24 KiB memcpys per request.
- **Fusing up+gate**: independent, but separated by a barrier, so their weight
  streams do not overlap.
- **f16 on the wire**: halves transfer, costs exactness — measure with
  `compare.py`, do not assume.

Recorded as *not* worth doing: request pipelining. The trunk is strictly
sequential, so one sequence never has more than one request in flight.

## Housekeeping that is not optimization

Small, unglamorous, and cheap to fold into whatever touches the file next:

- `per_slot` return mode is declared in the protocol and unimplemented.
- `moe_client::rtt_us` and `::log` grow without bound.
- `elapsed_us` returns `uint32_t` — about 71 minutes of microseconds.
