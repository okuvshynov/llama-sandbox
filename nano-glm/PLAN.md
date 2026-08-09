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
  │ ~17 GB of trunk weights    │  ◄── y   │ router, experts, combine │
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
- **Same OS and toolchain on both ends** for any bit-exact claim: a
  Windows/MSVC and a macOS/clang build of one llama.cpp commit differ by
  8.85e-3 mean KL on identical hardware.
- **Run configuration is part of the contract.** Thread count, batch shape and
  toolchain all move logits. Pin them before blaming the code. (repo
  `CLAUDE.md`)
- **Measure every deliberate trade** — f16 on the wire, GPU experts,
  server-side combine — with `compare.py` against the bit-exact run.
- **No mmap for the expert store.** Windows timings swing 1.04-1.84 t/s on
  identical configs from standby-list state alone; the backend needs an
  explicit non-mmap load path and should record which mode a run used.
- **Optimise bytes, not flops.** The expert FFN is ~95%+ memory movement.

## The budget

GLM-5.2 UD-Q6_K, 75 MoE layers, at the 74 GB/s this Mac Pro sustains:

| quantity | value |
|---|---|
| one Q6_K expert | 31.09 MB |
| MoE reads / token | 21.66 GB → 293 ms → **3.42 tok/s ceiling** |
| trunk on a Blackwell-class card | ~10 ms — negligible |
| **requests / token** | **75, strictly sequential** |
| **compute budget / request** | **3.90 ms → 256 req/s** |
| wire / request, **decode** | 24 KiB each way → 3.7 MB/token, ~2.9 ms on 10GbE |
| wire / request, **prefill** | n_tokens x 24 KiB — ~2.8 MB at 114 tokens (measured) |

Measured in step 1, superseding the estimates above where they differ:

| quantity | measured |
|---|---|
| backend, warm | **3.07 ms/layer, 80.6 GB/s** -> 4.16 tok/s MoE-only |
| backend, cold | 52 ms/layer — 16x, all page-in |
| RPC overhead, loopback warm | **114 us, 3.6%** of a layer |
| network+queueing over a corpus prompt | flat 0.7s while compute fell 108s -> 30s |

The system is entirely MoE-bound, so the figure of merit is the backend's
sustained request rate. RPC overhead is pure addition, and its *share* grows
as the backend gets faster — the 0.7s above was 0.7% of the first corpus
prompt and 2.5% of the fifth, for identical work.

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

Weight split (from the GGUF tensor offsets): routed experts 563.27 GiB /
96.6%, shared expert 2.84 GiB, trunk 16.76 GiB. Per *token*, though, the trunk
is ~45% of bytes read — which is why moving it to fast silicon removes the
ceiling that expert offload alone cannot.

## Steps

**Numbers are stable identifiers, not an ordering.** Code comments and commit
messages point at "step 1", "step 3"; renumbering would falsify them. The
order below is the execution order.

Step 2 moved to the back because nothing depends on it: 3, 6, 7, 8, 9 and 10
all run on this one machine, so machine B is needed by exactly one item, and
that item is the least informative one left. Step 5 has since made its
correctness half a one-line invocation.

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

### 6. Cross-prompt residency — **Now**

`ROUTING.md`'s 58.4% is measured *within* one continuation — same topic, same
register, the easy case. A static VRAM placement is chosen once and serves
every workload, so the number that decides step 3 is rank-on-prose /
score-on-code. Until it exists, treat 58.4% as an upper bound.

Five corpus prompts at `-n 256` with `--expert-log`, then compare rankings
across the five `counts.csv` files. Half an hour of tracing, and it is the
cheapest decision-relevant measurement left in the plan.

### 7. Core library, and an app that completes prompts — **In progress**

**Done: the split.** `lib/` is the engine — model loader, trunk graph, routed
block, wire protocol, remote-MoE client, fingerprint, trace — and `apps/` the
programs that drive it (`nano-glm`, `moe-server`), joined by a `nano-lib`
INTERFACE target. `lib/README.md` maps the files and records the one
constraint: the headers hold `static` state, so the first app needing a second
translation unit must convert `nano-lib` to a real static library rather than
reaching for `inline`. Gated at each stage — move, then extract — against the
step 5 golden set.

**Next: `nano-chat`** — tokenizer, chat template, sampling, streaming text.
`nano-glm`'s contract stays frozen: ids in, lkldtopk out, greedy = stored
top-1, no sampler. The rule that forces that split is **the bit-exactness
contract is defined over a fixed token sequence, so anything able to change
that sequence lives outside the tool that produces reference numbers** — a
sampler with RNG, a template tweak, a tokenizer version bump each silently
invalidate every stored reference if they hide behind a flag in one binary.

Two bridges: `nano-chat --dump-ids` turns any interesting behaviour into a
reproducible logits test, and a detokenizer for lkldtopk files makes corpus
output readable (the routing study wanted this and had to read raw ids).

The tokenizer brings its own correctness question — do our ids match
llama.cpp's for the same text? Vocab only, no model, seconds, and deliberately
*not* part of the logits gate, so a tokenizer bug cannot present as a numerics
failure.

### 3. Vulkan experts inside the backend — **Planned**

Hold a resident expert subset on the 4 Vega dies, serve misses from DRAM. 23%
of 563 GiB fits in 128 GiB of VRAM, and the routing study says that catches
**58% of selections, not 23%** — DRAM traffic falls 21.66 -> 9.01 GB/token,
moving the MoE-bound ceiling from 3.42 to ~8 tok/s before dispatch cost. That
is what makes this step worth doing rather than marginal, so the number is
load-bearing and step 6 has to land first.

Runs entirely on this machine: backend-side, client on loopback. No dependency
on step 2.

Placement is **static, not LRU** — PCIe is ~5.7x slower than DRAM, so cache
refill never pays; and uniform per layer, since a global budget measured the
same. Q4_K experts are the composing lever (~32% less DRAM traffic, ~33%
resident). Cost model and per-phase dispatch numbers: ../moe-offload.

Byte identity ends here for the expert path, so the gate changes shape:
compare against **the same build's CPU run over the same ids**, never a
historical file; measure the GPU path's own reproducibility floor first
(GPU-vs-GPU across batch shapes and workgroup sizes) or its KL against CPU
means nothing; gate on per-position max, not mean, since one mis-routed token
vanishes in an average over 743 positions. The RPC boundary is the useful
diagnostic: a server compare-mode evaluating a request on both CPU and GPU
localizes divergence to a layer instead of smearing it across the forward pass.

### 8. Huge pages for the expert store — **Planned**

31 MB per expert against a 1536-entry L2 TLB, and 4 KiB pages break the L2
streamer about once per output row. Self-contained, backend-side, independent
of everything else.

Prerequisite is the non-mmap load path the invariants already require: neither
Windows nor macOS offers huge pages for file-backed mappings. That path is
worth having anyway — it is what makes Windows timings reproducible.

### 10. C++ unit tests — **Planned**

The gate above is all end-to-end: every check costs a model load and answers
in minutes. Unit-level invariants deserve a real test target instead — the
Hadamard matrix being orthonormal, symmetric and made of exact ±1/√n entries;
wire-header layout; hparam parsing against a synthetic GGUF; the routing
statistics in `expert_stats.py` against traces with known structure.

Not a `--selftest` flag on the shipping binary: tests belong outside the
artifact under test, and a flag that only ever runs in CI is dead weight in
every other invocation.

Sequenced after 7 because unit tests need units, and that is exactly what the
core-library extraction produces — today `fill_hadamard` is a `static` function
inside `lib/nano_graph.h` and nothing else can reach it. A few pieces are testable
sooner (`moe_proto.h` is already a standalone header, `expert_stats.py` needs
no C++ at all) if the rest slips.

### 9. Latency hiding around the shared expert — **Planned, independent**

Issue the MoE request, run the shared expert on the client while it is in
flight, collect after. A `moe_send`/`moe_recv` split around the shared-expert
branch gets it for free, because the CPU backend executes graph nodes in order
and the other fifteen workers are already waiting out the round trip.

Worth ~0.5 ms against a 3.2 ms request — **16% on loopback today**, more once
the link is slower. Depends on nothing; it is listed late because it is an
optimisation, not information. Must not change a bit: same corpus gate.

### 2. Backend on another machine — **Deferred**

Direct-attach 10GbE, TCP_NODELAY, jumbo frames. Verify machine B's model from
disk against `checksums/GLM-5.2-UD-Q6_K.sha256`, bring it up on the same
build, then

    python gate.py rpc --moe-addr <host>:5711

— the correctness half is already written and needs no new code. What remains
is measurement: RTT distribution, sustained req/s against the 256 the CPU
ceiling allows, end-to-end tok/s.

Deferred because most of it is arithmetic — decode adds ~20 us of transfer,
prefill ~2.2 ms at 114 tokens, and the only real unknown is whether 24 KiB to
2.8 MB messages reach line rate on these NICs. Its KL == 0 gate also tests a
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
