# Plan: remote MoE evaluation

## Goal

Run MoE models whose experts do not fit next to the compute that wants them.
The routed experts move behind a **network service** on a machine that can hold
them; the trunk runs wherever trunk work is fastest.

```
  trunk host                              MoE backend
  ┌────────────────────────────┐   TCP    ┌──────────────────────────┐
  │ modern GPU (RTX 6000 Pro   │  x  ──►  │ Mac Pro 2019             │
  │ Blackwell class): attention│  75 req  │ 768GB DDR4: ALL routed   │
  │ + KV cache + shared expert │  /token  │ experts (563 GiB)        │
  │ ~21 GB of trunk weights    │  ◄── y   │ router, experts, combine │
  └────────────────────────────┘          │ 4x Vega II: hot subset   │
                                          └──────────────────────────┘
```

Two **roles**, not two peers: one machine holds every expert, so there is
nothing for the client to shard. The question is narrow — *how fast can the
trunk get answers out of a remote MoE backend?* GLM-5.2 is the testbed
(bit-exact baseline via ../logit-kld); Kimi-K3 (~1.5TB,
https://huggingface.co/moonshotai/Kimi-K3) is why the experts live elsewhere.

The router sits on the backend, so a request is one activation and a response
is one combined row. That costs model-agnosticism — gating, selection bias,
weight norm and the x2.5 scale become backend knowledge — which is acceptable
only because K3 shares DeepSeek-lineage gating. Revisit if a target model's
gating diverges.

## Invariants

Constraints that hold across every step:

- **KL == 0 wherever the whole path is CPU.** Steps 0-2, and the CPU path
  inside 3, must not perturb a single bit against the ../logit-kld baseline.
  Once GPUs carry real work the bar becomes bit-exact within the CPU path plus
  a *measured* KL bound against that same build's CPU run.
- **A reference without provenance is not a reference.** Every stored baseline
  carries model hashes, llama.cpp commit, compiler, thread count, batch size,
  OS and host; the comparison refuses to run on a mismatch rather than
  reporting a difference. `results/corpus/` had to be discarded for exactly
  this — the files predate the shard repair and nothing in them says so.
- **Same OS and toolchain on both ends** for any bit-exact claim. FP
  contraction is a compiler choice, so two builds of one llama.cpp commit
  disagree on identical hardware. The magnitude once recorded here (8.85e-3
  mean KL) is withdrawn: those artifacts date from the day before the corrupt
  shards were found, and while the evidence leans clean — they predate the
  reboot that exposed the corruption, and corruption alone flipped no top-1
  tokens where that comparison flipped 3.3% — it is unverified. Re-measure
  before quoting it: one llama.cpp run per platform over the same ids, both
  models now checked against `checksums/`.
- **Run configuration is part of the contract.** Thread count, batch shape and
  toolchain all move logits. Pin them before blaming the code. (repo
  `CLAUDE.md`)
- **Measure every deliberate trade** — f16 on the wire, GPU experts,
  server-side combine — with `compare.py` against the bit-exact run.
- **No mmap for the expert store.** Windows timings swing 1.04-1.84 t/s on
  identical configs from standby-list state alone; the backend needs an
  explicit non-mmap load path and should record which mode a run used.
- **No throughput number without a stated residency regime.** Cold, hot
  subset and whole-model warm differ by more than any optimisation in this
  plan claims to be worth — 16x between cold and warm at the layer level.
  A tok/s figure that does not say which regime it came from is noise with
  a decimal point (step 11).
- **Optimise bytes, not flops — but it is not 95/5.** The expert FFN was
  assumed to be ~95%+ memory movement. Measured (`nano-bench --pages`), a
  null-compute stream of the same bytes in the same shape runs at ~100 GB/s
  where the model sustains 75.4, so roughly **a quarter of decode time is not
  memory** — dequantization, per-node barriers across 78 layers, attention.
  Bytes still dominate; the tail is bigger than the slogan implied.

## The budget

Everything here is measured on this machine unless the row says otherwise.
Byte counts come from walking the loaded GGUF's tensor map (`nano-bench`), not
from tensor-offset arithmetic, so they follow the model rather than a note.

| quantity | measured |
|---|---|
| bytes / token, decode | **38.93 GB** = dense 20.03 (shared expert 3.05) + routed 18.90, being 8/256 of 604.81 |
| one routed expert (gate+up+down) | 31.5 MB |
| **sustained bandwidth** | **75.4 GB/s** on the model's own access pattern |
| decode, hot subset (regime A, ~39.7 GB resident) | **1.941 tok/s** |
| decode, whole model (regime B, ~466 GB resident) | **1.932 tok/s**, spread 0.4% |
| prefill, cold -> warm | 0.9 -> **6.9 tok/s** |
| warmup | one repetition for decode (1263 -> 515 ms), three for prefill |
| MoE-only bytes / token | 21.95 GB -> 291 ms -> **3.44 tok/s**, the ceiling once the trunk is elsewhere |
| backend (moe-server), warm / cold | 3.07 ms/layer, 80.6 GB/s / 52 ms/layer |
| RPC overhead, loopback warm | **114 us, 3.6%** of a layer |
| network+queueing over a corpus prompt | flat 0.7s while compute fell 108s -> 30s |
| **requests / token** | **75, strictly sequential** |
| wire / request | 24 KiB each way at decode; n_tokens x 24 KiB at prefill, 2.8 MB at 114 tokens |

Still estimates, flagged as such because nothing here has run them:

| quantity | estimate | rests on |
|---|---|---|
| compute budget / request | 3.88 ms -> 258 req/s | the MoE-only ceiling divided by 75 |
| decode wire cost on 10GbE | 3.7 MB/token, ~2.9 ms | line rate, never measured on these NICs (step 2) |
| prefill wire cost on 10GbE | ~2.2 ms at 114 tokens | same |
| trunk on a Blackwell-class card | ~10 ms, negligible | vendor bandwidth, no such card here (step 4) |

Two results from step 11, one of them a surprise:

- **74 GB/s was a good assumption.** It came from a synthetic sweep; the
  model's real access pattern sustains 75.4. Decode is bandwidth-bound and
  nothing else.
- **Working-set size does not matter once resident.** Regimes A and B differ by
  12x in footprint — 39.7 GB against ~466 GB — and by **0.5%** in throughput.
  There is no locality bonus to be had in DRAM: keeping *hot* experts hot buys
  nothing, and residency helps only by removing bytes from the DRAM path. That
  rules out the optimistic reading of the routing study, and it is why step 3's
  value is computed from bytes alone.

Regime B's 0.4% spread, against the 40% swings that motivated step 11, is what
makes a 10% change resolvable at all.

**This machine is not MoE-bound; it is bound by everything.** MoE is 56% of the
bytes a token reads and the dense trunk is 44%, which is exactly why step 4
matters and why the 3.44 tok/s figure above is a bound for *that* machine, not
this one. The earlier claim that the system is "entirely MoE-bound" was an
artifact of counting only MoE bytes.

Weight split, from the same tensor walk: routed experts 604.81 GB, shared
expert 3.05 GB, everything else 17.99 GB (of which a 1.01 GB embedding table
that decode reads one row of). Per *token*, though, the split is 56/44 as
above — near-uniform routing would make it 50/50, and the ratio is what decides
how much of the win lives on each side of the wire.

### What the router does

Measured, not assumed — 1138 positions x 75 layers x 8 of 256 experts, one
prose continuation, out of sample against a same-shape uniform null
(full study: `ROUTING.md`):

| quantity | measured | uniform routing |
|---|---|---|
| entropy per layer | 6.98 bits | 8.00 |
| experts carrying half the selections | **36** of 256 | 128 |
| hit rate of a 23% resident subset | **58.4%** | 23.1% |
| consecutive-token expert overlap | **35.2%** | 3.1% |
| distinct experts per 32-token window | 92 | 163 |

Routing is strongly concentrated and strongly autocorrelated, except in the
first five MoE layers (3-7), which are near-uniform on every metric. Spending
a VRAM budget globally rather than per layer exploits that and is worth
nothing (58.4% either way), so placement can stay uniform per layer.

## Steps

**Numbers are stable identifiers, not an ordering.** Code comments and commit
messages point at "step 1", "step 3"; renumbering would falsify them. The
order below is the execution order.

Step 2 moved to the back because nothing depends on it: 3, 6, 8, 9, 10 and 11
all run on this one machine, so machine B is needed by exactly one item, and
that item is the least informative one left. Step 5 has since made its
correctness half a one-line invocation.

Step 11 sat ahead of 3 and 8 for the same reason 5 sat ahead of 7: both are
claims about memory behaviour, and neither could be judged against a baseline
that moved 40% between runs. It is done, and the baseline now holds to 0.4%.

10 and 8 then moved ahead of 6 and 3, both being self-contained and ready.
8 did not survive contact: a 40 GB probe showed 4 KiB pages already reaching
~100 GB/s in the model's own access shape, so it is parked at the back and
**step 12 replaces it** — the same probe found the model running at 75% of
that, and a quarter of decode turning out not to be memory movement is a
larger and better-located target than page size ever was.

The dependency that matters is preserved: 6 still comes before 3.

### 0. Host-side dispatch, in-process — **Done**

`ggml_mul_mat_id` became `ggml_custom_4d` ops resolving each (token, slot,
expert) triple on the host — the client shape with the network removed. Gate:
903 positions, KL 0.0. Branch `phase0-moe-dispatch` (2aeceef) has the kernel
contract that makes a callback bit-exact.

### 1. RPC proof of concept, local, CPU-only — **Done**

`moe-server` holds the router and every expert, takes `(layer, x)`, returns one
combined row per token; the client is a `ggml_custom_4d` node at `n_tasks = 1`
in its place, so both paths live in one binary. Gate: 743 positions,
12,375 RPCs, KL 0.0. `lib/moe_proto.h`, `apps/moe-server/main.cpp` (deferred
optimisations documented there), `--moe-addr` / `--moe-log`.

### 5. Test harness and golden set — **Done**

`gate.py` with four named tests — `smoke`, `aa`, `rpc`, `llamacpp` — over a
committed golden set in `testdata/` (1.6 MB: 6 prompts as raw token ids, the
nano-glm outputs, the llama.cpp references, and `provenance.json`). The gate
checks provenance *before* it compares bytes and refuses on a mismatch, so a
different compiler or thread count is an explicit refusal rather than a
difference that looks like yours. Rules, costs and re-baselining: `TESTING.md`.

Passed: 6 prompts, 761 positions, KL == 0 against llama.cpp `rescore
--sim-gen`; the RPC path byte-identical to the local one. All three setups
agree, established with two edges rather than three.

Protocol **v2** adds a fingerprint handshake, sorting a mismatch into
always-fatal (structural — the client's graph assumes those hparams),
fatal-under-`--strict` (reproducibility), and informational. Runtime flag, not
build-time, because it touches nothing numeric — so the gate exercises the
shipping binary. Verified in both directions: `--strict` refuses a thread-count
mismatch, the lenient default warns and proceeds.

`lib/build_info.h`, `gate.py`, `TESTING.md`, `testdata/`. Two things naming the
tests bought: `rpc` exists at all (the numbered scheme had no slot for the
local-vs-backend edge, half of what step 1 built), and `gate.py rpc
--moe-addr <host>` *is* step 2's correctness argument, already written.

### 7. Core library, and an app that completes prompts — **Done**

**Done: the split.** `lib/` is the engine — model loader, trunk graph, routed
block, wire protocol, remote-MoE client, fingerprint, trace — and `apps/` the
programs that drive it (`nano-glm`, `moe-server`), joined by a `nano-lib`
INTERFACE target. `lib/README.md` maps the files and records the one
constraint: the headers hold `static` state, so the first app needing a second
translation unit must convert `nano-lib` to a real static library rather than
reaching for `inline`. Gated at each stage — move, then extract — against the
step 5 golden set.

**Done: `nano-chat`** — byte-level BPE from the GGUF (`lib/vocab.h`), GLM-5.2's
chat format as token ids (`lib/chat_glm.h`), greedy decode, streamed output.
`--dry-run` tokenizes from shard 1 alone, so checking what the template built
costs a second rather than a model load, and its ids feed straight into
`nano-glm -T` — which is how an interesting generation becomes a reproducible
logits test.

`nano-glm`'s contract stays frozen: ids in, lkldtopk out, greedy = stored
top-1, no sampler. The rule that forces that split is **the bit-exactness
contract is defined over a fixed token sequence, so anything able to change
that sequence lives outside the tool that produces reference numbers** — a
sampler with RNG, a template tweak, a tokenizer version bump each silently
invalidate every stored reference if they hide behind a flag in one binary.
A sampler belongs in `nano-chat` when it arrives, and costs nothing there.

Tokenizer agreement is measured, not assumed: `tokenizer_check.py` versus
`llama-tokenize`, **28/28 cases and 864 tokens exact**, including CJK, emoji
and combining marks. `\p{L}`/`\p{N}` come from llama.cpp's own tables via
`lib/gen_unicode_ranges.py`, so the two cannot drift apart on a Unicode
revision. Kept out of the logits gate on purpose.

Still open, small: a detokenizer for lkldtopk files, so corpus output is
readable (the routing study wanted it and read raw ids instead).

### 11. A benchmark with stated residency regimes — **Done**

`apps/nano-bench`, repetitions inside one process, every repetition printed and
the median taken from the back half only — the warmup curve is the signal, and
folding it into a mean is what produced the 40% swings this step existed to
replace. Byte counts come from the loaded GGUF (walk the tensor map, `*_exps`
counted at `n_expert_used/n_expert`), so a quant change updates the budget
instead of silently invalidating it.

| regime | working set | result |
|---|---|---|
| `--hot` one position re-decoded | ~39.7 GB | 1.941 tok/s, 75.5 GB/s, spread 9.5% |
| `--full` 256 tokens x 4 passes | ~466 GB | **1.932 tok/s, 75.2 GB/s, spread 0.4%** |

Findings are in the budget above. The short version: 74 GB/s was a good
assumption, decode is purely bandwidth-bound, and **a 12x difference in working
set is worth 0.5%** — so there is no locality prize in DRAM and residency helps
only by removing bytes.

Warmup is one repetition for decode and three for prefill, which also explains
every harness number ever recorded here: 0.94-1.69 tok/s was page-in, not
compute, and prefill swings 0.9 -> 6.9 tok/s across passes.

Not done, and no longer urgent: the **non-mmap load path**. It was in scope as
the way to guarantee residency, but repetition achieves that already and the
0.4% spread says the standby list is not interfering. It survives only as huge
pages' prerequisite (step 8), which is now the sole reason to build it.

Also open: `--moe-addr` support, so the RPC path can be measured in the same
regimes.

### 10. C++ unit tests — **Now**

The gate is all end-to-end: every check costs a model load and answers in
minutes, so anything it does not cover is untested in practice. `lib/` is
eleven headers now, and the list of things worth asserting has grown with it:

- **`fill_hadamard`** — orthonormal, symmetric, every entry exactly ±1/√n. The
  power-of-two abort too, which is the one branch nothing has ever taken.
- **`pretok_split`** — the GLM-4 alternation, especially the two rules that
  need backtracking. `tokenizer_check.py` covers these against llama.cpp but
  costs a minute and a model path; a unit test costs neither.
- **`nano_kv_parse` and the handshake drift comparison** — the step 5 coverage
  gap, stated at the time: `--strict` is exercised for `n_threads` only,
  because forcing a compiler or model mismatch needs two builds or two model
  copies. Tested directly, it needs neither, and no socket either.
- **wire header layout**, **hparam parsing against a synthetic GGUF**, and the
  statistics in `expert_stats.py` against traces with known structure.

Not a `--selftest` flag on the shipping binary: tests belong outside the
artifact under test, and a flag that only ever runs in CI is dead weight in
every other invocation.

Unblocked by 7 — the units exist, and a test target links `nano-lib` the same
way `nano-chat` does. The remaining friction is that the pieces are `static`,
which a test TU can include but not link against; the first real test is also
the nudge to convert `nano-lib` into a compiled library (`lib/README.md` says
to do that rather than reach for `inline`).

### 12. Where the missing 25% goes — **Next**

`nano-bench --pages` says the model's own access pattern sustains ~100 GB/s
with ordinary 4 KiB pages, and `nano-bench --hot/--full` says the model
sustains 75.4. So a quarter of decode is not memory movement, and nothing in
the plan currently accounts for it. Candidates, roughly in order of how much
they could be worth:

- **Q6_K dequantization.** 210 bytes per 256 weights, unpacked on every read.
  Real work, and the one thing a lower-bit quant would change for a second
  reason beyond bytes.
- **Per-node barriers.** 78 layers x tens of graph nodes, each with a thread
  barrier; `ggml_barrier` at 16 threads is not free and does not scale with
  bytes.
- **Attention and the KV path**, which the byte budget excludes entirely.

Cheap to attribute before optimising: time the graph node-by-node
(`ggml_backend_sched` already knows the split), or run one MoE layer in
isolation against a raw stream of the same bytes. The answer decides whether
the lever is a kernel, a fusion, or nothing.

Worth doing before step 3: a residency win is a *bytes* win, and if bytes are
only 75% of the time then step 3's +40% is really +30%.

### 6. Cross-prompt residency — **Planned**

`ROUTING.md`'s 58.4% is measured *within* one continuation — same topic, same
register, the easy case. A static VRAM placement is chosen once and serves
every workload, so the number that decides step 3 is rank-on-prose /
score-on-code. Until it exists, treat 58.4% as an upper bound.

Five corpus prompts at `-n 256` with `--expert-log`, then compare rankings
across the five `counts.csv` files. Half an hour of tracing, and it is the
cheapest decision-relevant measurement left in the plan.

### 3. Vulkan experts inside the backend — **Planned**

Hold a resident expert subset on the 4 Vega dies, serve misses from DRAM. 23%
of the routed 604.81 GB fits in 128 GiB of VRAM, and `ROUTING.md` says that
catches **58% of selections, not 23%** — which is what makes this step worth
doing rather than marginal, so that number is load-bearing and step 6 has to
land first.

What it is worth depends on which machine, and the plan used to quote only the
larger figure without saying so. At the measured 75.4 GB/s and 38.93 GB/token:

| | DRAM / token | tok/s |
|---|---|---|
| today | 38.93 GB | 1.94 |
| **+ residency, trunk still local** | 27.89 GB | **2.71 (+40%)** |
| + step 4, no residency | 21.95 GB | 3.44 |
| + residency and step 4 | 10.91 GB | **6.92** |

So this step is worth **+40% now** and roughly **2x after step 4** — both
worth having, but they are not the same claim. (The 3.44 also lands on the
budget's MoE-only ceiling, which is the same arithmetic from the other
direction.) And because A and B tied, these follow from bytes alone:
there is no additional gain from the resident set being the *hot* one.

Runs entirely on this machine: backend-side, client on loopback. No dependency
on step 2.

Placement is **static, not LRU** — PCIe is far slower than DRAM, so cache
refill never pays; and uniform per layer, since a global budget measured the
same (`ROUTING.md`). Q4_K experts are the composing lever, roughly a third off
both DRAM traffic and resident size. The PCIe ratio and the Q4_K figures are
../moe-offload's, not measured here; nothing in this repo has yet run a byte
over PCIe.

Byte identity ends here for the expert path, so the gate changes shape:
compare against **the same build's CPU run over the same ids**, never a
historical file; measure the GPU path's own reproducibility floor first
(GPU-vs-GPU across batch shapes and workgroup sizes) or its KL against CPU
means nothing; gate on per-position max, not mean, since one mis-routed token
vanishes in an average over 743 positions. The RPC boundary is the useful
diagnostic: a server compare-mode evaluating a request on both CPU and GPU
localizes divergence to a layer instead of smearing it across the forward pass.

### 9. Latency hiding around the shared expert — **Planned, independent**

Issue the MoE request, run the shared expert on the client while it is in
flight, collect after. A `moe_send`/`moe_recv` split around the shared-expert
branch gets it for free, because the CPU backend executes graph nodes in order
and the other fifteen workers are already waiting out the round trip.

Worth about the shared expert's own read: 3.05 GB over 75 layers is 40.7 MB,
which at the measured 75.4 GB/s is ~0.54 ms against a 3.2 ms request, so of
order 15% on loopback. That is arithmetic from two measurements, not a measured
speedup — `nano-bench` can settle it once it speaks `--moe-addr`. Depends on
nothing; listed late because it is an optimisation, not information. Must not
change a bit: same corpus gate.

### 8. Huge pages for the expert store — **Parked, probably dead**

The idea: 31.5 MB per expert against a 1536-entry L2 TLB, and Intel's L2
streamer does not prefetch across a 4 KiB page boundary, so streaming one
expert restarts it ~7,700 times at 4 KiB against ~15 at 2 MB. That mechanism
would have explained a 75.4 GB/s model against 140.8 GB/s of theoretical
DDR4-2933.

**Probed before rewriting anything** — `nano-bench --pages`, 40 GB, no model,
because the model-scale version costs a non-mmap loader, ~583 GiB read per run
and 583 GiB locked non-pageable:

| threads | sequential | expert-shaped blocks |
|---|---|---|
| 8 | 81.7 GB/s | 84.5 GB/s |
| 16 | 100.6 | 98.9 |
| 32 | 102.8 | 106.9 |

**With 4 KiB pages.** Two things follow, and both cut against the step:

- **The access pattern is free.** Shuffled expert-sized blocks match a flat
  sequential sweep, so the scattered structure of `mul_mat_id` costs nothing to
  begin with.
- **4 KiB already reaches ~100 GB/s**, 73% of theoretical and inside the normal
  75-85% band for six-channel Cascade Lake. The headroom the mechanism was
  invoked to explain is not there, so large pages have at most a small ceiling
  to raise — and the model is not reaching the current ceiling anyway.

What the probe did find is a **different and larger gap**: the model sustains
75.4 GB/s where its own access pattern allows ~100, so ~25% of decode is
compute and per-node overhead rather than memory. That is now the more
promising direction, and it is not this step.

Remaining, if anyone wants to close it: the 2 MB arm needs
`SeLockMemoryPrivilege`, which this account does not hold — grant "Lock pages
in memory" in `secpol.msc` and log out and in, then `nano-bench --pages` runs
both arms. Expect it to confirm the above rather than overturn it. The non-mmap
load path is no longer blocked on anything; it just has no reason left.

### 2. Backend on another machine — **Deferred**

Direct-attach 10GbE, TCP_NODELAY, jumbo frames. Verify machine B's model from
disk against `checksums/GLM-5.2-UD-Q6_K.sha256`, bring it up on the same
build, then

    python gate.py rpc --moe-addr <host>:5711

— the correctness half is already written and needs no new code. What remains
is measurement: RTT distribution, sustained req/s against the 256 the CPU
ceiling allows, end-to-end tok/s.

Deferred because most of it is arithmetic at line rate — decode would add
~20 us of transfer and prefill ~2.2 ms at 114 tokens, neither measured, and
the only real unknown is whether 24 KiB to 2.8 MB messages reach line rate on
these NICs at all. Its KL == 0 gate also tests a
configuration we do not intend to ship: once the trunk is on a GPU (step 4),
end-to-end byte identity is gone by construction. What it *uniquely* de-risks
is cross-machine identity, and the step 5 handshake answers that on connect,
without a 10GbE run.

**Trigger**: when the throughput number is wanted for a writeup, or when the
trunk actually moves off this machine.

Cheap cleanups to fold in whenever it happens: rename the server's `t_route`
(it times graph construction, not routing); `per_slot` return if the debug path
is wanted; the unbounded `moe_client::rtt_us` / `::log` growth and the
`uint32_t` wrap in `elapsed_us`.

### 4. Trunk on a modern GPU — **Planned**

Attention + KV + shared expert onto a Blackwell-class card; nano-glm needs a
GPU backend (it currently aborts if one is present — the abort is a
correctness-testing guard, not a position). End-to-end bit-exactness ends here
by construction, and llama.cpp CPU becomes the only remaining independent
oracle, which is why the golden set has to be frozen with full provenance
*before* the dependency is dropped.

## Ideas, unproven

- **Speculative routing.** With the router on the backend, run layer N+1's
  router on layer N's activation while idle and prefetch the likely experts.
  Buys nothing while experts are in DRAM (bandwidth-bound; prefetch adds no
  bandwidth), and the idle window it needs shrinks to ~3% once the trunk is
  fast. Its home is K3, where experts exceed RAM and prefetch targets storage.
  The routing study puts a floor under the *predictor* half: keeping the
  previous token's experts warm already hits 35% against a 3% base rate, free
  and with no speculative router at all. A speculative router has to beat that,
  not uniform.
- **Backend micro-optimisations** — graph caching, zero-copy request/response,
  fusing up+gate. All noise against today's 3072 us/layer; each is recorded in
  `apps/moe-server/main.cpp` with the condition that would make it matter, since the
  backend gets faster in later phases and a 1% cost becomes 10% when compute
  drops 10x.

## Parked

Client-side expert sharding, static-vs-replicated dispatch policies, balanced
slot assignment across peers, asymmetric split-ratio tuning. All assumed
experts spread over machines the client chooses between; with one backend
holding everything, none applies. Global expert ids in the protocol are what
lets this return as a backend-side concern without a client change.

## Links

- client seam: `apps/nano-glm/main.cpp` — `moe_rpc_cb` (its comment documents the
  ggml custom-op contract), combine at `cur_experts`, shared expert at
  `ffn_up_shexp`
- bit-exact kernel contract: `ggml/src/ggml-cpu/ggml-cpu.c`
  `ggml_compute_forward_mul_mat_id_one_chunk`; public traits in
  `ggml/include/ggml-cpu.h` (`ggml_get_type_traits_cpu`)
- custom ops: `ggml/include/ggml.h` — `ggml_custom_4d`; dispatch:
  `ggml/src/ggml-cpu/ops.cpp` `ggml_compute_forward_custom`
- tests: `TESTING.md` — what each named test establishes, which to run after
  which change, what a provenance refusal means; `gate.py`, `testdata/`
- build fingerprint: `lib/build_info.h` — one definition behind `--version`,
  the provenance sidecar and the wire handshake
- verification: ../logit-kld — `compare.py`, `rescore --sim-gen`, corpus in
  `prompts/`, noise floors in its README
- routing study: `ROUTING.md`; trace `lib/expert_trace.h` (`build.ps1 -Trace`,
  `--expert-log`), analysis `expert_stats.py`
- GPU cost model: ../moe-offload/README.md
- platform traps (AMD-Metal NaN, mmap variance, core counts, run config):
  repo `CLAUDE.md`
