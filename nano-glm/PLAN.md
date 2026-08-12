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

### 14. A second architecture: DeepSeek-V4-Flash — **Now** (branch `deepseek-v4-flash`)

`unsloth/DeepSeek-V4-Flash-0731-GGUF`, UD-Q8_K_XL, 5 shards, 150.7 GiB at
`D:\llms\ds-v4-flash`. Three reasons it is worth the port, in order:

1. **It is the second model**, which is the only way to find out which of
   `lib/`'s tier boundaries were real. `lib/README.md` scores its own
   prediction; the short version is that the "DeepSeek lineage" family tier was
   wishful and the loader had a latent alignment bug.
2. **~91% expert residency** on this hardware, against GLM-5.2's 22%. Every
   offload conclusion in `OPTIMIZATION.md` is bounded by that 22% and says so.
   This model reaches the regime those measurements could not see.
3. **A high-residency thread-overhead check.** The per-layer thread spawn cost
   ~3-7% of a decode layer at 20% residency; whether that holds when the GPUs
   hold nearly everything is unmeasured.

**Done:** `lib/models/{glm_dsa,deepseek4}/`, the generic loader split into
`gguf_store.h`, `moe_shape.h` for the client/backend contract, the deepseek4
hparams and loader, its MoE block, `moe-server` dispatching on architecture,
and `nano-probe` / `gguf_peek.py` to see what a checkpoint actually contains.
GLM-5.2 stayed byte-identical through all of it (`gate.py rpc`, 6/6).

**Hash-routed layers stay on the client.** Layers 0-2 pick experts from a
token-id lookup, and the wire protocol carries activations, not token ids.
Rather than grow the request, `moe_shape.n_dense_lead` reports 3 for this model
— the field already means "leading layers the backend does not serve", and the
client already honours it. It makes the trunk slightly bigger and the protocol
not at all.

**The trunk, tensor by tensor.** `logit-kld`'s `dump` captures llama.cpp's own
intermediates and `dump_inspect.py` compares them, so each tensor is checked the
moment it is written rather than at the logits, where KL saturates
(`OPTIMIZATION.md`). `apps/ds4-port` is the harness and goes away when the graph
is complete. **Layers 0, 1 and 2 are done: 184/184 tensors bit-identical** over
a 384-token prompt — both hyper-connection halves, all four norms, the q/kv
construction, the attention core, the grouped-LoRA output, the router, hash
routing, the routed experts, the shared expert, `l_last`, the overlap
compression that folds every 4 tokens into one key for both the attention and
the indexer, the DSA lightning indexer itself, and the attention over the
concatenated raw and compressed key sequences.

Layer 2 was checked **again at 2560 tokens** (47/47), where its 640 compressed
blocks exceed `indexer_top_k = 512` and the indexer's selection genuinely drops
128 of them. Below that length the top-k selects everything and cannot fail.

Seven of layer 2's tensors are excluded from the comparison and cannot be
otherwise: they are shaped by llama.cpp's KV-cache padding and their tails hold
whatever the buffer held. Everything downstream of them is compared and exact —
see `dump_inspect.py --exclude` and `models/deepseek4/graph.h`.

Layers 0-1 are every layer of the simple shape: `compress_ratios` is
`[0, 0, 4, 128, 4, 128, ...]`. The prompt length is itself part of the check:
five tokens make one compressed block, and one block cannot distinguish a block
index from a block's first token position; under 128 tokens the sliding window
never bites, so a plain causal mask passes. 384 tokens give 96 blocks and three
window widths. Note that a dump at that length needs an explicit
`--max-elem` — the 4M default truncates `attn_raw` at exactly token 128, and
`dump_inspect.py` used to compare the overlap and call it a pass.

Getting there required the reference to be built with `GGML_CPU_REPACK OFF`
(`logit-kld/CMakeLists.txt`). llama.cpp repacks MXFP4 experts into `mxfp4_8x8`
at load and runs a different GEMM; nano-glm mmaps weights as they sit in the
file and cannot follow. It never came up with GLM-5.2 because Q6_K only repacks
under NEON. Note for step 11 and for the measurements below: a *performance*
comparison against llama.cpp must use a repack-enabled build, since that is
what llama.cpp actually ships.

**Next, in order:**

- **Layer 3**, a ratio-128 compressor layer: the same compressor at a different
  ratio, no indexer, and llama.cpp's `build_hca_attention` rather than
  `build_csa_lid_attention` — no top-k, so the whole compressed sequence is
  visible. Note `build_hca_compressed_kv_from_state` is a *different* function
  from the overlap one already ported: ratio-128 blocks do not overlap, so it
  reads `ratio` rows rather than `2*ratio` and there is no prev/cur split.
  Layers alternate 4/128 from there, so this is the last new attention shape.
- Then the head (`hc_head`, `result_norm`, `result_output`).
- **A golden set.** llama.cpp supports `LLM_ARCH_DEEPSEEK4`, so `gate.py
  llamacpp` can create one exactly as it did for glm-dsa. The verification
  methodology survives the second model unchanged, which was not guaranteed.
- **The two measurements** this model was chosen for, once a trunk exists to
  drive the server end to end.

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

   *Build tree done.* `build.ps1 -Vk` -> `build-vk\`, `NANO_VULKAN=ON` ->
   `GGML_VULKAN=ON`, and that tree builds **moe-server alone**: a Vulkan build
   registers the GPUs and every trunk binary aborts when a GPU device is
   present (`lib/nano_graph.h`), so the server is the only binary that can hold
   one. It pairs with `build\bin\nano-glm.exe` as client, which keeps the
   client on exactly the numerics the golden set was made with. `vulkan` is now
   in the build fingerprint and in `NANO_REPRO_KEYS`, so a strict client
   *refuses* a Vulkan backend — deliberately, since that pairing cannot be
   byte-compared and the refusal is the signal to use the KL gate.

   `moe-server --devices` (no model load) reports what a build sees:

       4x GPU  Vulkan0-3   31.2 / 32.0 GiB free   AMD Radeon Pro Vega II Duo
       1x CPU  CPU        743.9 / 767.9 GiB free   Xeon W-3245
       fp16 1 | bf16 0 | int dot 1 | matrix cores none | warp 64

   **Sizing, and it is sobering.** At 31.5 MB per expert (`OPTIMIZATION.md`),
   31.2 GiB of VRAM holds ~1060 experts against 75 layers x 256 = 19200, i.e.
   **~5.5% on one die**, ~22% across all four. So increment 2 exercises
   correctness and the fallthrough path thoroughly, and will be *slower* than
   CPU-only, because the GPU does a twentieth of the work while adding a
   transfer and a sync. That is expected and is not a reason to stop; the
   speed question belongs to increment 3 and to placement in `OPTIMIZATION.md`.

   **Design decision: real compaction, not weight-zeroing.** Residency is per
   *expert*, slots are per *token*, so a device's share is a set of
   (token, slot) pairs that varies across the batch. The cheap alternative —
   keep the full 8 slots on every device and zero the weights of the ones it
   does not own — is correct but makes each device do the whole layer's work,
   so the GPU would be pure overhead and the CPU would save nothing. Instead
   each device gets a compacted `x`, `ids` and `weights` covering only its
   pairs, and the host scatter-adds the returned rows. Increment 3 needs the
   same machinery for expert parallelism within a layer, so the cheap version
   would be built to be thrown away.

   *Implemented and measured.* `--gpu-experts k` uploads k experts of every
   MoE layer (26.06 GiB at k=12, 16.3 s) and the CPU device keeps the
   byte-identical full path whenever it still owns every pair — so `gate.py
   rpc` with no GPU is unchanged, verified 6/6.

   **The measurements.** `vk_check.py`, k=12, one die, scoring **fixed token
   ids** taken from the golden (top-1 agreement / KL max):

   | prompt          | determinism | shape floor  | compaction alone | GPU alone    | end to end   |
   |-----------------|-------------|--------------|------------------|--------------|--------------|
   | smoke (18)      | **0** /100% | 94.4% /3.4e-2| 100%  /7.5e-3    | 94.4%/1.2e-2 | 94.4%/1.1e-2 |
   | 04_history (151)| **0** /100% | 98.7% /2.8e-2| 97.4% /6.6e-2    | 97.4%/5.4e-2 | 96.7%/8.9e-2 |
   | 01_prose (146)  | **0** /100% | 97.3% /4.4e-2| 93.2% /4.4e-2    | 95.2%/5.8e-2 | 95.2%/5.5e-2 |

   The GPU took a measured **4.37%** of (token, slot) pairs at k=12, matching
   the 12/256 residency, so the router picks resident experts about as often as
   chance.

   Three conclusions, in decreasing order of how well they are supported:

   - The Vulkan path is **bit-reproducible** — KL exactly 0 across 315
     positions, and also at k=2. That is the one unambiguous result, and it is
     what makes any gate possible at all.
   - **The split dominates, not the driver.** `compaction alone` is comparable
     to `GPU alone` everywhere and worse than it on `01_prose`. Without the
     second-CPU-device control the whole difference would have been filed
     against the GPU.
   - **The measurement is saturated, so it cannot certify correctness.**
     Raising the GPU's share 5.5x (k=2 -> k=12) moved nothing, and the shape
     floor — no GPU involved — lands in the same band. 75 layers amplify any
     perturbation to the same ceiling. End-to-end logit KL therefore cannot
     distinguish a subtly wrong GPU from a correct one; this supports
     "deterministic and plausible", not "verified".

   **Per-layer compare-mode settled it.** `--compare` evaluates every layer on
   both the full CPU path and the split path, hands the trunk the *CPU* answer,
   and records the deviation. Returning the CPU answer is the whole trick:
   every layer then receives identical input, so each measurement is local and
   nothing compounds. Per-layer relative RMS over `smoke`:

   | path                       | median   | max      | worst layer |
   |----------------------------|----------|----------|-------------|
   | compaction (CPU -> CPU)    | 2.06e-08 | 7.64e-08 | 58          |
   | GPU                        | 1.82e-03 | 1.54e-02 | 58          |

   Five orders of magnitude apart — where the end-to-end KL had called them
   4.4e-2 and 5.8e-2, i.e. indistinguishable. That gap is the justification for
   building this mode.

   - **The compaction is correct.** 2e-08 is f32 reassociation and nothing
     else, so `01_prose`'s 93.2% top-1 was chaotic amplification of a 1e-08
     perturbation, not a bug. The open question from the previous commit is
     closed.
   - **The GPU has a real ~1.8e-03 per-layer difference**, which is a
     precision/algorithm difference rather than a wrong answer.

   Two internal consistency checks passed: layers 42, 46 and 54 are *exactly*
   zero in both runs — the router never picked a resident expert there, so the
   split device got no work — and the worst layers coincide (58, 45, 64),
   because error tracks how much work the split device actually did.

   **Where the GPU difference does *not* come from.** Four hypotheses, four
   negative results, overall rel RMS each time — recorded so nobody runs them
   again:

   | configuration                          | rel RMS  |
   |----------------------------------------|----------|
   | default                                | 8.44e-04 |
   | `GGML_VK_DISABLE_F16=1`                | 8.02e-04 |
   | `GGML_VK_DISABLE_MMVQ=1`               | 8.45e-04 |
   | `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`| 9.97e-04 |

   fp16 storage was the obvious suspect — Vega advertises `fp16: 1` — and it is
   **falsified**: disabling it changes nothing. Neither does dropping the
   quantized matvec path, and disabling integer dot products makes it slightly
   *worse*.

   So the ~1e-3 is **intrinsic to ggml's Vulkan Q6_K `mul_mat_id` on this
   driver**, not a toggle anyone left in the wrong position. It is stable,
   deterministic and proportional to the work the device does. Whether it is
   *acceptable* is a separate question this measurement does not answer; what
   it does is put a number on the cost of moving experts to the GPU, so the
   trade is explicit rather than discovered later.

   **Measure over fixed ids, never over free generation.** The first version of
   `vk_check.py` gave each configuration the same *prompt* and let it generate
   its own continuation. Four of eighteen comparisons then produced no number
   at all — the token sequences had diverged and `compare.py` correctly refused
   them. The instructive one was `04_history` "compaction alone", where **both
   sides ran on the CPU** with identical arithmetic: reassociating a token's
   8-term expert sum flipped one greedily-sampled token, and the continuations
   parted from there. Free generation cannot measure numerical divergence in a
   system where numerical divergence changes what is generated. The script now
   scores the golden's full id sequence with `-n 0`.

   Two controls earned their keep and should not be removed:

   - **A second CPU device** (`--cpu-experts k`) running the identical
     compaction. Without it the whole CPU-vs-GPU difference would have been
     filed against the GPU, when in fact compaction alone accounts for most of
     it — splitting a token's 8-term sum into (7)+(1) is a reassociation, and
     75 layers amplify it.
   - **A shape floor that actually varies the shape.** The first attempt
     compared `-b 512` against `-b 16` on a 14-token prompt: both prefill in
     one chunk, so it reported a flat 0.0 that looked like a perfectly
     deterministic GPU and was two identical runs. The split batch must be
     smaller than the prompt.

   Incidental but worth keeping: the CPU control assigned **478** pairs to its
   split device where the GPU run assigned **472**, same k and same placement.
   Numerical divergence changes a routing decision, which changes which pairs
   exist — MoE routing turns continuous differences into discrete ones. It did
   not move top-1 here, but it is why "small KL" is a weaker guarantee in an
   MoE than elsewhere.
3. **N devices, one thread each** — **done**. `--gpu-devices n` spreads experts
   `0..k-1` round-robin over n dies, each driven by its own host thread;
   `devices[0]` runs on the caller's thread since it has the largest share.
   One `ggml_backend_graph_compute` is one backend in order, and `..._async` is
   only that call minus the synchronize, so several graphs on several threads
   is the formulation that actually overlaps — and the one that generalizes to
   NUMA nodes rather than only to GPUs.

   **Correctness, via `--compare`.** Same expert set {0..11}, one die versus
   four:

   |        | rel RMS  | split pairs                       |
   |--------|----------|-----------------------------------|
   | 1 die  | 8.44e-04 | 482                               |
   | 4 dies | 8.21e-04 | 129 + 118 + 132 + 103 = **482**   |

   Identical pair total and identical DRAM fallthrough (10318), so the
   round-robin loses none and duplicates none; the small RMS change is the
   combine going from one partial to four. A broken local-index map would have
   read the wrong expert's weights and shown order-1 error, not 1e-3.

   Incidental: compare mode reports **482** pairs where the free-running GPU
   and CPU-control runs reported 472 and 478. Returning CPU activations to the
   trunk removes the routing feedback, so 482 is the canonical count and the
   earlier spread was that feedback, measured.

   **Does it go faster? Barely, and the honest answer is "not established".**
   `04_history`, 151 fixed ids, k=52 over four dies (13 experts/die, 28.23 GiB
   each, 112.9 GiB total), one discarded warm-up then three measured passes:

   | config          | passes           | mean   | sd    |
   |-----------------|------------------|--------|-------|
   | CPU only        | 36.5 33.0 33.4   | 34.3 s | 1.6 s |
   | 4 dies, k=52    | 33.8 31.3 31.5   | 32.2 s | 1.1 s |

   1.06x. The difference is ~1.9 sigma with overlapping ranges, so it is
   suggestive and not more than that. A first single-shot pair had said 1.10x;
   its CPU sample (37.6 s) fell outside all three later reps, which is the repo
   `CLAUDE.md` warning about single mmap-backed timings on Windows arriving
   exactly on schedule. Do not quote a speedup from one pass.

   **The ceiling is residency, not parallelism**, and that is the useful part.
   `devices[0]` holds every expert and takes the *complement* of what the GPUs
   take, so at 19.85% of pairs on the GPUs the CPU still does 80.15% and the
   best attainable speedup is **~1.25x** however many dies are added. Four dies
   raise how much can be *resident* (~22% of experts in 125 GiB), not how much
   of the work can move. We are getting roughly a quarter of that headroom; the
   rest is dispatch, transfer and sync not hidden behind the CPU's share.

   Getting past 1.25x is therefore not a threading problem. It needs the CPU to
   stop being the fallthrough for everything — a placement question
   (`OPTIMIZATION.md`), or a smaller expert quant on the GPU side so more fits.

   **`--force-split` settled what the dies are worth.** Forcing the work
   distribution regardless of routing (wrong output, right cost — `TESTING.md`)
   gives a full-offload ceiling of **1.89x on prefill** and **1.43x on decode**,
   both bounded by the trunk still running on the client CPU (~40% of prefill
   and ~50% of decode, measured from the RPC accounting). The GPU MoE is *not*
   free — it is 25-29% of the offloaded run; what offload buys is 4.6x on
   prefill MoE and 2.4x on decode MoE. Four dies are worth nothing on prefill
   and +4.9% on decode, so they buy VRAM capacity rather than parallelism. At
   the ~22% residency that actually fits, both regimes land near **1.1x**.
   Numbers, the sublinear decode curve, and two claims of mine the data
   falsified: `OPTIMIZATION.md`, "What GPU offload is actually worth".

   Still open here: the per-layer router read-back is a hard sync point, and a
   device's expert work could start as soon as its slice of the decision lands.
   Not attempted yet.

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
