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
nothing for the client to shard. GLM-5.2 is the testbed (bit-exact baseline via
../logit-kld); Kimi-K3 (~1.5TB, https://huggingface.co/moonshotai/Kimi-K3) is
why the experts live elsewhere.

The router sits on the backend, so a request is one activation and a response
is one combined row. That costs model-agnosticism — gating, selection bias,
weight norm and the x2.5 scale become backend knowledge — which is acceptable
only because K3 shares DeepSeek-lineage gating. Revisit if a target model's
gating diverges.

**This plan is about making that work, not about making it fast.** Enough
measurement exists to say the shape is viable and to catch a regression; what
it cannot do is justify optimizations one at a time, because every number here
comes from a single machine, model, quantization and workload shape. Anything
whose purpose is speed lives in `OPTIMIZATION.md` — including several things
measured in detail and then deliberately not built.

## Invariants

- **KL == 0 wherever the whole path is CPU.** Nothing in steps 0-2 may perturb
  a bit against the ../logit-kld baseline. Once a GPU carries real work the bar
  becomes bit-exact within the CPU path plus a *measured* KL bound against that
  same build's CPU run.
- **A reference without provenance is not a reference.** Every stored baseline
  carries model hashes, llama.cpp commit, compiler, thread count, batch size,
  OS and host; the comparison refuses to run on a mismatch rather than
  reporting a difference. `results/corpus/` had to be discarded for exactly
  this — the files predate the shard repair and nothing in them says so.
- **Same OS and toolchain on both ends** for any bit-exact claim. FP
  contraction is a compiler choice, so two builds of one llama.cpp commit
  disagree on identical hardware. The magnitude once recorded here (8.85e-3
  mean KL) is withdrawn as unverified — those artifacts predate the corrupt
  shards. Re-measure before quoting it.
- **Run configuration is part of the contract.** Thread count, batch shape and
  toolchain all move logits. Pin them before blaming the code (repo
  `CLAUDE.md`).
- **Measure every deliberate trade** — f16 on the wire, GPU experts,
  server-side combine — with `compare.py` against the bit-exact run.
- **No throughput number without a stated residency regime.** Cold, hot-subset
  and whole-model-warm differ by more than any optimisation is worth. A tok/s
  figure that does not say which regime it came from is noise with a decimal
  point (`nano-bench`).
- **Defend the baseline; do not chase it.** The bench exists to catch
  pessimization. A change that quietly costs 10% must be visible; a change that
  might gain 10% can wait until the system works end to end.

## What is measured

Facts that constrain the design. Per-optimisation payoffs have moved to
`OPTIMIZATION.md`.

| quantity | measured |
|---|---|
| bytes / token, decode | **38.93 GB** = dense 20.03 (shared expert 3.05) + routed 18.90, being 8/256 of 604.81 |
| sustained bandwidth | **75.4 GB/s** on the model's own access pattern; ~100 GB/s is available to that pattern |
| decode, whole model warm | **1.932 tok/s**, spread 0.4% |
| decode, hot subset (~39.7 GB resident) | 1.941 tok/s — a 12x smaller working set is worth 0.5% |
| prefill, cold -> warm | 0.9 -> **6.9 tok/s** |
| MoE share of a token's bytes | 56%; the dense trunk is 44% |
| backend (moe-server), warm / cold | 3.07 ms/layer, 80.6 GB/s / 52 ms/layer |
| RPC overhead, loopback warm | **114 us, 3.6%** of a layer |
| **requests / token** | **75, strictly sequential** |
| wire / request | 24 KiB each way at decode; n_tokens x 24 KiB at prefill, 2.8 MB at 114 tokens |

Consequences worth keeping in view:

- **Decode is bandwidth-bound, and about a quarter of it is not memory** —
  compute and per-node overhead. The old "expert FFN is ~95%+ memory movement"
  rule is retired.
- **This machine is not MoE-bound**; it is bound by everything. Which is why
  moving the trunk (step 4) matters, and why MoE-only ceilings describe *that*
  machine rather than this one.
- **Working-set size does not matter once resident**, so residency helps only
  by removing bytes from the DRAM path, never by improving locality.
- **Expert placement does not transfer between prompts** — 28.2% against 23.1%
  for random (`ROUTING.md`). Read that as a statement about *single-turn*
  workloads: a conversation has think-time to move experts in and same-topic
  reuse to exploit, and neither is measured yet (step 13).

## Steps

**Numbers are stable identifiers, not an ordering.** Code comments and commit
messages point at them; renumbering would falsify those. Steps 8, 9 and 12 now
live in `OPTIMIZATION.md` under the same numbers.

### Done

- **0. Host-side dispatch, in-process.** `ggml_mul_mat_id` became
  `ggml_custom_4d` ops resolving each (token, slot, expert) triple on the host
  — the client shape with the network removed. 903 positions, KL 0.0. Branch
  `phase0-moe-dispatch` (2aeceef) has the kernel contract that makes a callback
  bit-exact.
- **1. RPC proof of concept, local.** `moe-server` holds the router and every
  expert; the client is a `ggml_custom_4d` node in its place, so both paths
  live in one binary and local-vs-remote is a direct A/B. 743 positions,
  12,375 RPCs, KL 0.0. `lib/moe_proto.h`, `apps/moe-server/`.
- **5. Test harness and golden set.** `gate.py`, four named tests, a committed
  golden set, and a provenance record checked before any byte comparison.
  Protocol v2 adds a fingerprint handshake. `TESTING.md`.
- **7. Core library and apps.** `lib/` is the engine; `apps/` are nano-glm
  (validation), nano-chat (tokenizer, chat template, streaming) and moe-server.
  Tokenizer agrees with llama.cpp on 28/28 cases. `lib/README.md`.
- **11. `nano-bench`.** Throughput in a named residency regime, plus a
  no-model memory probe. A regression guard, not a design input.
- **6. Cross-prompt residency.** Placement does not transfer; `ROUTING.md`.

### 10. C++ unit tests — **Planned**

The gate is all end-to-end: every check costs a model load and answers in
minutes, so anything it does not cover is untested in practice. `lib/` is
eleven headers, and the list worth asserting has grown with it:

- **`fill_hadamard`** — orthonormal, symmetric, every entry exactly ±1/√n, and
  the power-of-two abort, the one branch nothing has ever taken.
- **`pretok_split`** — the GLM-4 alternation, especially the two rules that
  need backtracking. `tokenizer_check.py` covers these but costs a minute and a
  model path; a unit test costs neither.
- **`nano_kv_parse` and the handshake drift comparison** — the step 5 coverage
  gap: `--strict` is exercised for `n_threads` only, because forcing a compiler
  or model mismatch needs two builds. Tested directly it needs neither, nor a
  socket.
- **wire header layout**, **hparam parsing against a synthetic GGUF**, and
  `expert_stats.py` against traces with known structure.

Not a `--selftest` flag on the shipping binary: tests belong outside the
artifact under test. The pieces are `static`, which a test TU can include but
not link against, so the first real test is also the nudge to make `nano-lib` a
compiled library (`lib/README.md`).

### 2. Backend on another machine — **Planned**

The point of the whole exercise, and the last piece of the core path that has
never run. Direct-attach 10GbE, TCP_NODELAY, jumbo frames.

1. Verify machine B's model from disk against
   `checksums/GLM-5.2-UD-Q6_K.sha256` before trusting anything it computes.
2. Bring it up on the same build. Under `--strict` the v2 handshake refuses a
   mismatch, which is how a toolchain difference announces itself instead of
   surfacing later as a KL failure.
3. `python gate.py rpc --moe-addr <host>:5711` — the correctness half needs no
   new code.
4. Measure what loopback cannot: RTT distribution over a real link, sustained
   req/s against the 258 the CPU allows, end-to-end tok/s.

Expect prefill transfer to be the new cost: ~2.8 MB per request at 114 tokens
against 24 KiB at decode. If it dominates, the fixes in order of cheapness are
f16 on the wire (measure the KL), chunking the prefill, or overlapping transfer
with compute.

### 3. Vulkan experts inside the backend — **Now**

The backend should use the GPUs it has: four Vega II dies, 128 GiB of VRAM,
idle while the CPU streams experts from DRAM.

**Support, not speedup** — though ../moe-offload suggests it may be both. One
fully-resident layer, 8 experts on *one* die, Windows AMD driver: 1.11 ms end
to end including transfers, against the 3.07 ms per layer in the table above.

**That ~2.8x is a cross-harness comparison and is not yet ours.** The 1.11 ms
was measured in ../moe-offload, a different program with a different graph and
its own timing; only the 3.07 ms is in-tree. Two numbers from two harnesses
divided by each other is precisely the shape of mistake `OPTIMIZATION.md` and
the bench exist to prevent, so treat it as *inherited motivation* and re-measure
in-tree at increment 2 before it justifies a design choice. It is a reason to
build the thing, not a number to plan against.

../moe-offload also found Vulkan-on-Metal computing *correctly* on hardware
where native Metal silently NaNs (repo `CLAUDE.md`), so the macOS path is
usable even at 3.1x slower inside the fence — same caveat on the figure.

Which experts to hold is `OPTIMIZATION.md`; this step assigns them trivially
and moves on.

#### The shape the code has to take

`eval_layer` builds one graph and runs it on one backend
(`apps/moe-server/main.cpp`). Two things break that:

**ggml graphs are sequential.** One `ggml_backend_graph_compute` is one
backend, in order. Parallelism across devices therefore means *several graphs*,
one per device, and `ggml_backend_graph_compute_async` is merely
`graph_compute` without the synchronize — whether it actually returns early is
backend-specific. So: **one host thread per device**, join, combine. That is
also the only formulation that generalizes off the GPU.

**Routing has to move to the host.** Today `build_moe_block` routes *inside*
the graph — `argsort_top_k` is a node — so which experts a token needs is not
known until the graph runs. Partitioning work across devices needs it before
the per-device graphs are built. So the layer splits in two: a small router
graph (a 6144x256 matmul, sigmoid, bias, top-k) whose output is read back, then
per-device expert graphs. Step 0's `phase0-moe-dispatch` branch (2aeceef)
already prototyped host-side dispatch and is the reference for doing it without
perturbing the numbers.

#### Device abstraction

A device is *a backend, the experts it holds, and a thread to drive it*:

```
struct moe_device {
    ggml_backend_t backend;      // CPU, Vulkan, later a NUMA-pinned CPU
    ggml_gallocr_t galloc;
    std::vector<uint8_t> meta;   // its own graph scratch
    // expert ids resident here; the rest fall through to the DRAM device
};
```

Deliberately not GPU-shaped, because a **4-socket NUMA server** is the same
problem: a node is a backend with thread affinity and expert weights allocated
in its own memory, and the partition-then-combine machinery is identical. The
only GPU-specific part is that VRAM holds ~23% of the experts at Q6_K, so a
device's set is a subset and misses fall through to a DRAM-resident device.

#### Combine and determinism

Each device sums its own slots' weighted rows and returns one row; the host
adds the N partials **in device order**. Deterministic run to run, which is
what matters, and it costs N rows of readback rather than 8 — ../moe-offload
measured a 192 KiB 8-slot readback at 261 us against 33 us for one row, so this
is the difference between ~20 ms and ~2.5 ms per token of pure readback.

It is *not* bit-identical to the CPU's pairwise combine, and cannot be: the
expert FFN itself runs in different arithmetic. That is why this step's gate
changes shape (below).

#### Increments, each gated

1. **Restructure with one CPU device** — **done**. Host-side routing, the
   `moe_device` abstraction, combine — no Vulkan, in `apps/moe-server/main.cpp`.
   `gate.py rpc` byte-identical 6/6, which is the whole point of doing it first.

   Two things were deliberately left out, because with one device neither can
   be *tested*: threading (increment 3) and residency partitioning — device 0
   holds every expert, so any partition is the identity. The seam is marked in
   `run_device`, along with what makes increment 2's version hard: residency is
   per **expert** while slots are per **token**, so which device owns a given
   (token, slot) pair varies across the batch. A per-device slot subset is
   therefore not well defined, and the honest options are compaction (gather
   the pairs a device owns into a dense `ids`) or computing everything
   everywhere and zeroing the weights — the second defeats the purpose.

   `t_route_us` now times the router rather than graph construction, so the
   backend can report the split. Measured cold (0.4 tok/s, weights paging in):
   router p50 228 us vs experts 21985 us at decode, 2051 vs 426796 at prefill —
   so the router is low single-digit percent of a layer, and *higher* than the
   1.14% measured, since paging inflates the denominator. Re-measure warm
   before optimizing against it. The real design consequence is not the cost
   but the shape: the read-back is a hard per-layer sync point that did not
   exist before.
2. **One Vulkan device holding a subset**, misses falling through to the CPU
   device. Gate becomes the KL bound below.
3. **N devices, one thread each**, trivial expert assignment (`e % n_devices`).
   Measure whether four dies beat one; expert parallelism within a layer is the
   thing being tested. Also where the per-layer router sync can be overlapped:
   layer i+1's router genuinely cannot start early (it needs layer i's output),
   but a device's expert work can start as soon as its slice of the decision
   lands.

#### Build

Vulkan has to stay off in the default tree: nano-glm aborts when a GPU device
is present, and that guard exists because of the AMD-Metal NaN incident. A
separate `build-vk` tree — as `build.ps1 -Trace` already does for the trace
build — keeps the CPU gate's guarantees while `moe-server` gets a GPU.

#### Gate

Byte identity ends here for the expert path. Compare against **the same
build's CPU run over the same ids**, never a historical file. Measure the GPU
path's own reproducibility floor first — GPU-vs-GPU across batch shapes and
workgroup sizes — or its KL against CPU means nothing. Gate on per-position
max, not mean, since one mis-routed token vanishes in an average over 743
positions. The RPC boundary is the useful diagnostic: a server compare-mode
evaluating a request on both CPU and GPU localizes divergence to a layer
instead of smearing it across the forward pass.

### 13. Multi-turn conversations — **Planned**

`nano-chat` is single-turn: one prompt, one response, exit. Real use is a
conversation, and that changes what the system can *do*, not only how fast it
is.

- **KV reuse across turns.** The prefix is unchanged, so only the new turn
  needs prefilling; without it every turn re-reads the whole context.
- **A conversation is where expert movement becomes affordable.** Between turns
  there is think-time and user latency to hide a swap behind; inside a token
  there is not. Every placement and prefetch idea that looked unaffordable in
  `OPTIMIZATION.md` deserves re-asking here, and the one strong transfer result
  we have — prose↔history, two prompts on the same subject — is exactly the
  same-topic reuse a conversation provides.
- Turn-boundary handling: the template's `<|user|>` / `<|assistant|>` markers
  and the eot token the model emits.

Infrastructure, and a prerequisite for measuring anything about realistic
workloads.

### 4. Trunk on a modern GPU — **Planned**

Attention + KV + shared expert onto a Blackwell-class card. nano-glm needs a
GPU backend and currently aborts if one is present — a correctness-testing
guard, not a position. This is what makes the MoE-only ceiling the operative
one, since the dense trunk is 44% of per-token bytes today.

End-to-end bit-exactness ends here by construction, and llama.cpp CPU becomes
the only remaining independent oracle — which is why the golden set must be
frozen with full provenance *before* the llama.cpp dependency is dropped.

## Parked

Client-side expert sharding, static-vs-replicated dispatch policies, balanced
slot assignment across peers, asymmetric split-ratio tuning. All assumed
experts spread over machines the client chooses between; with one backend
holding everything, none applies. Global expert ids in the protocol are what
lets this return as a backend-side concern without a client change.

## Links

- optimization backlog: `OPTIMIZATION.md`
- tests: `TESTING.md` — named tests, provenance refusals, re-baselining;
  `gate.py`, `testdata/`
- routing and residency studies: `ROUTING.md`; `expert_stats.py`,
  `residency_study.py`, `cache_sim.py`, trace in `lib/expert_trace.h`
- library map and model-specificity tiers: `lib/README.md`
- build fingerprint: `lib/build_info.h` — `--version`, provenance, handshake
- client seam: `lib/moe_client.h` — `moe_rpc_cb` documents the ggml custom-op
  contract; protocol in `lib/moe_proto.h`
- verification: ../logit-kld — `compare.py`, `rescore --sim-gen`, noise floors
- GPU cost model: ../moe-offload/README.md
- platform traps: repo `CLAUDE.md`
