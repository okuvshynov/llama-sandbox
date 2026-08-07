# Plan: remote MoE evaluation

## Goal

Run MoE models whose experts do not fit next to the compute that wants them,
by putting the routed experts behind a **network service** and keeping the
trunk on whatever machine is best at trunk work.

Target topology:

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

Two **roles**, not two peers. GLM-5.2 is the testbed (bit-exact baseline via
../logit-kld); Kimi-K3 (~1.5TB, https://huggingface.co/moonshotai/Kimi-K3) is
the reason the experts have to live somewhere else.

**The router lives on the backend.** The request is one activation and the
response is one combined row — "activation in, layer output out" — so expert
ids and routing weights never cross the wire, routing histograms are collected
where the experts are, and the backend is free to use its own routing
knowledge (see the speculative-prefetch note at the end). The cost is that the
backend stops being model-agnostic: sigmoid gating, the selection bias, weight
normalisation and the x2.5 scale all become its business. That was an explicit
non-goal in the sharded draft ("the server carries over to Kimi unchanged"),
and it is a real loss — mitigated only by Kimi-K3 being DeepSeek-lineage and
sharing that gating structure. Revisit if a target model's gating diverges.

## Why remote-MoE rather than sharding

Earlier drafts of this plan split experts across two peer machines and spent
most of their complexity on the *dispatcher*: static shard vs replicated,
balanced slot assignment, asymmetric split ratios. That was solving the wrong
problem for GLM-sized models. One Mac Pro holds **all 563 GiB** of routed
experts in RAM, so there is nothing to shard on the client side. What is left
is a much narrower question:

> how fast can the trunk get answers out of a remote MoE backend?

Sharding does not disappear; it becomes an *internal* concern of the backend
(which experts sit in the 4 GPUs vs in DRAM), invisible to the protocol. If a
model ever exceeds one machine's RAM, a second backend is an addition to the
service, not a change to the client.

The other half of the reframing: per *token* the trunk is ~45% of the bytes
read (~17 GB against ~21.7 GB of experts), even though experts are 96.6% of
the weights. Expert offload alone therefore caps the achievable speedup —
moving the trunk onto fast silicon is what removes the ceiling, and a modern
card holds the whole 17 GB trivially.

## The budget

Measured on GLM-5.2 UD-Q6_K (582.87 GiB, from the GGUF tensor offsets):
routed experts 563.27 GiB / 96.6%, shared expert 2.84 GiB, trunk 16.76 GiB.
nano-glm runs 78 blocks, the first 3 dense, so **75 MoE layers**.

Per token, at the 74 GB/s this Mac Pro sustains (derived from llama.cpp's
measured 1.84 tok/s):

| quantity | value |
|---|---|
| one Q6_K expert | 31.09 MB |
| routed expert reads | 18.65 GB |
| shared expert | 3.01 GB |
| **MoE total** | **21.66 GB → 293 ms → 3.42 tok/s ceiling** |
| trunk on a Blackwell-class card (~1.8 TB/s) | ~10 ms — negligible |

So the system is **entirely MoE-bound**, and the figure of merit is the
backend's sustained request rate:

- **75 requests per token, strictly sequential** — layer *i+1* cannot start
  until layer *i*'s experts return. There is no batching across layers.
- **3.90 ms per request** of compute budget at the CPU ceiling → **256 req/s**.
- RPC overhead is pure addition to that. At 200 us/round trip the cost is ~5%;
  it only becomes structural if the backend gets much faster than DRAM.

Wire, per request: 24 KiB out (n_embd f32), 24 KiB back (the combined row).
With the router on the backend, that is the whole protocol payload — 3.7 MB
per token, ~2.9 ms on 10GbE, ~1% of the budget.

Returning the 8 per-slot rows instead would cost 192 KiB back, 16.6 MB per
token, 13.3 ms — worth keeping as a debug mode (it is what lets the client
reproduce the combine independently) but not the default.

**The combined return can be bit-exact**, which the earlier plan assumed it
could not: the backend applies the routing weights and pairwise-adds the slots
in index order, exactly as the baseline combine does. Byte-identity then
survives the 8x traffic reduction rather than trading against it. Verify it,
do not assume it.

## Verification methodology

The KL == 0 gate survives, but it has to be scoped per component, because the
end state deliberately puts the trunk on different silicon:

- **Phases 1-2 are all-CPU, one toolchain: end-to-end bit-exactness applies.**
  A remote backend on a second machine must run the *same OS and compiler* or
  the gate fails for reasons unrelated to the RPC — a Windows/MSVC and a
  macOS/clang build of the same llama.cpp commit differ by 8.85e-3 mean KL on
  identical hardware (see repo CLAUDE.md).
- **Phase 3 (Vulkan experts) and phase 4 (GPU trunk) cannot be bit-exact.**
  There the bar becomes: bit-exact within the CPU path, and a *measured* KL
  bound end to end, read against the documented noise floors in
  ../logit-kld/README.md.
- Every deliberate trade — f16 on the wire, GPU experts, server-side combine —
  is measured with compare.py against the bit-exact run, never assumed.

Run configuration is part of the contract: thread count, batch shape and
toolchain all move logits. Pin them on both ends before blaming the code.

## Phases

### Phase 0 — host-side dispatch, in-process (DONE, branch `phase0-moe-dispatch`)

The routed block's three `ggml_mul_mat_id` calls became three
`ggml_custom_4d` ops whose callback reads the expert ids on the host, resolves
each (token, slot, expert) triple, evaluates it, and writes the row back to
its slot. This is the client shape with the network removed, and it is the
seam everything below plugs into.

Gate passed: 903 positions across the 5-prompt corpus, KL exactly 0.0, all
931,896 payload bytes identical to llama.cpp; both threading branches
(item-parallel on prefill, row-split on decode) exercised.

Two findings worth carrying:

- Bit-exactness through a callback requires **reusing ggml's kernels**, not
  reimplementing them: `ggml_get_type_traits_cpu()` gives `from_float` and
  `vec_dot`, and `mul_mat_id` calls `vec_dot(..., nrc = 1)` once per output
  element. swiglu stays a ggml op — its vectorised SiLU uses a polynomial exp
  a scalar `expf()` will not match.
- The in-graph alternative (per-shard `ggml_mul_mat_id` over `ggml_view_3d`
  slices) would have cost **4x the expert compute**: a reused graph has fixed
  topology, so every shard must evaluate all 8 slots and discard most.

### Phase 1 — RPC proof of concept, local, CPU-only

Split the phase 0 callback into client and server across loopback TCP.

1. Server: holds the router and the expert tensors. Takes `(layer, x)`, runs
   the router, evaluates the selected experts, combines, returns one row —
   all on the same ggml kernels. Stateless: no KV, no history between calls.
2. Client: the custom-op callback serialises the request, blocks, and drops
   the returned row where the combine used to land.
3. Protocol v1 (below): versioned, length-prefixed, explicit dims, loud
   errors.
4. **Gate: KL == 0 over the 5-prompt corpus**, same machine, same binary.
5. Measure the floor: round-trip latency, serialise/deserialise cost, and the
   per-request overhead that phase 2 will pay again over a real link.

Two things to settle here, where they are cheap: that the server-side combine
is byte-identical to the client-side one, and that moving the router across
the boundary changes nothing (it is the same matmul on the same activation —
if this is not bit-exact, something is wrong with the transport, not the
router).

### Phase 2 — the backend on another machine

6. Direct-attach 10GbE, TCP_NODELAY, jumbo frames. Same OS and toolchain on
   both ends (see Verification).
7. **Gate: KL == 0 still** — the RPC must not perturb a single bit.
8. Measure what actually matters: sustained req/s, per-request RTT
   distribution, and tok/s end to end. Compare against the 256 req/s the CPU
   ceiling allows; the gap is the RPC tax.
9. Latency hiding: issue the request, run the shared expert on the client
   while it is in flight, then collect. The CPU backend executes graph nodes
   in order, so a `moe_send` / `moe_recv` split around the shared-expert
   branch gets this for free.

### Phase 3 — Vulkan experts inside the backend

10. Hold a resident expert subset in the 4 Vega dies and serve hits from
    there, misses from DRAM. **23% of the 563 GiB fits** in 128 GiB of VRAM,
    so the CPU still does the bulk; this is worth roughly the resident
    fraction, no more.
11. Placement is **static, not LRU**: PCIe (~13 GB/s) is ~5.7x slower than
    DRAM (~74 GB/s), so streaming a missed expert to a GPU costs ~2.4 ms
    against ~0.42 ms to compute it on the CPU — and the DMA consumes the same
    DRAM bandwidth it is trying to save. Cache *refill* never pays.
12. Trust is earned the same way as everywhere else: A/A NaN scan, then a
    measured KL bound against the bit-exact run. MoltenVK and the AMD Vulkan
    driver both compute correctly here (../moe-offload), unlike native Metal.
13. Lower-bit experts are the other lever, and they compose: Q4_K (~4.5 bpw)
    cuts DRAM traffic ~32% *and* raises the resident fraction to ~33%. Cost
    in KL is exactly what logit-kld measures.

### Phase 4 — trunk on a modern GPU

14. Move attention + KV cache + shared expert onto the trunk host — the router
    and the combine already live on the backend. ~17 GB fits any
    Blackwell-class card with room for KV.
15. nano-glm needs a GPU backend for this; today it aborts if one is present.
16. End-to-end bit-exactness ends here by construction — switch to the
    KL-bounded gate and keep the CPU path as the reference.
17. Only now does RPC overhead become a large fraction of the token budget,
    because the trunk stops costing 226 ms. Re-measure before optimising.

## Protocol v1 (design requirements, not final wire format)

Much smaller than the sharded design needed, because there is one backend, it
routes for itself, and the client chooses nothing:

- Request: `{version, layer, x: f32[n_embd x n_tokens], return_mode}`.
- Response: one combined row per token, or the `n_used` per-slot rows plus the
  expert ids under `return_mode = per_slot` (debug: lets the client reproduce
  the combine independently). Either way the header carries **server-side
  durations** — parse, route, compute, serialise, on the server's own clock.
- Batched requests for prefill: `n_tokens > 1` in one call.
- Version field; dims explicit; hard errors on mismatch.

Expert ids stay **global** wherever they do appear, so a future backend that
shards internally maps them itself and the client never learns about it.

## Latency & routing logging

Durations only — never absolute cross-machine timestamps, so no clock sync is
needed: the client measures RTT on its own clock, the server reports its
internal durations, and `network + queueing = RTT − server_total`.

- Per-RPC: `(token_idx, layer, n_slots, bytes_out, bytes_in, rtt_us,
  t_serialize_us, t_deserialize_us, srv_parse_us, srv_route_us,
  srv_compute_us, srv_serialize_us, srv_gpu_hits, srv_cpu_misses)`.
- Per-token: trunk compute, MoE wait, overlap efficiency.
- Routing: selected expert ids per (layer, token), logged **on the backend**
  now that it routes — drives the residency question in phase 3 (are the
  marginals flat enough that a static subset is as good as anything smarter?).
- In-memory append on the hot path, JSONL at run end, analysis in Python.

## Backend implementation notes

- **Do not rely on mmap for the expert store.** On Windows, identical
  `llama-bench` invocations returned 1.04 ± 0.29 and 1.84 ± 0.02 t/s purely
  from standby-list state, while `--load-mode none` gave a steady
  1.90 ± 0.08; macOS shows no such spread. A backend holding 563 GiB resident
  and measured on throughput needs an explicit non-mmap load path, and each
  run should record which mode it used.
- Huge pages are worth testing: 31 MB per expert against a 1536-entry L2 TLB
  means every expert access thrashes it.
- The expert FFN is ~95%+ memory movement, not arithmetic — 31 MB streamed
  against ~6-125 us of actual MAC work. Optimise bytes, not flops.
- **Speculative routing (idea, unproven).** With the router on the backend, it
  can run layer N+1's router on layer N's activation while otherwise idle, and
  prefetch the experts it expects. Activations drift slowly between layers, so
  the prediction may be good — measurable cheaply by logging the overlap
  between the speculative top-k and the real one. Two caveats: it buys nothing
  while experts are in DRAM (the workload is bandwidth-bound and prefetch does
  not add bandwidth), and the idle window it relies on shrinks to almost
  nothing once the trunk is on a fast GPU. Its real home is Kimi-K3, where
  experts exceed RAM and prefetch targets storage rather than cache.

## Parked (revisit only if a model outgrows one backend)

Client-side expert sharding, static-vs-replicated dispatch policies, balanced
slot assignment across peers, and asymmetric split-ratio tuning. All of it
assumed experts spread over machines the client had to choose between; with
one backend holding everything, none of it applies. The protocol keeps global
expert ids specifically so this can come back as a backend-side concern
without a client change.

## References (searchable anchors, not line numbers)

- client seam: `src/nano_glm.cpp` — `moe_proj_cb`, the combine at
  `cur_experts`, shared expert at `ffn_up_shexp` (branch
  `phase0-moe-dispatch`).
- kernel contract for bit-exactness: `ggml/src/ggml-cpu/ggml-cpu.c`
  `ggml_compute_forward_mul_mat_id_one_chunk`; public traits in
  `ggml/include/ggml-cpu.h` (`ggml_get_type_traits_cpu`).
- custom ops: `ggml/include/ggml.h` — `ggml_custom_4d`, `GGML_OP_CUSTOM`
  (callback gets `ith`/`nth`; inputs via `dst->src[i]`).
- CPU execution order: `ggml/src/ggml-cpu/ggml-cpu.c`
  `ggml_graph_compute_thread`, `ggml_barrier` — nodes run strictly in order,
  which is what makes the send/recv overlap trick work.
- verification: `../logit-kld` — `compare.py`, `rescore --sim-gen`, the
  5-prompt corpus in `prompts/`, noise floors in its README.
- GPU cost model for phase 3: `../moe-offload` — per-phase dispatch/transfer
  costs on both platforms, and the 2 GiB max-allocation limit on the AMD
  driver that forces expert tensors to be split per projection.
- run-configuration rules (toolchain, thread count, batch shape all move
  logits): repo `CLAUDE.md`.
