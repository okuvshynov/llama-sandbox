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

- **KL == 0 while everything is CPU.** Phases 1-2 must not perturb a single
  bit against the ../logit-kld baseline. Once GPUs enter (3-4) the bar becomes
  bit-exact within the CPU path plus a *measured* KL bound.
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
| wire / request | 24 KiB out, 24 KiB back → 3.7 MB/token, ~2.9 ms on 10GbE |

The system is entirely MoE-bound, so the figure of merit is the backend's
sustained request rate. RPC overhead is pure addition: ~5% at 200 us/round
trip, structural only once the backend outruns DRAM.

Weight split (from the GGUF tensor offsets): routed experts 563.27 GiB /
96.6%, shared expert 2.84 GiB, trunk 16.76 GiB. Per *token*, though, the trunk
is ~45% of bytes read — which is why moving it to fast silicon removes the
ceiling that expert offload alone cannot.

## Steps

### 0. Host-side dispatch, in-process — **Done**

The routed block's `ggml_mul_mat_id` calls became `ggml_custom_4d` ops whose
callback resolves each (token, slot, expert) triple on the host. This is the
client shape with the network removed, and the seam everything below plugs
into. Gate passed: 903 corpus positions, KL exactly 0.0, 931,896 bytes
identical to llama.cpp.

Branch `phase0-moe-dispatch` (2aeceef) — see that commit for the kernel
contract that makes a callback bit-exact, and why the in-graph alternative
would have cost 4x the expert compute.

### 1. RPC proof of concept, local, CPU-only — **Next up**

Split the phase 0 callback into client and server across loopback TCP. The
branch above needs rebasing onto main first, and its callback has to absorb
the router and the combine rather than just the three projections.

**Server**: holds the router and the expert tensors. Takes `(layer, x)`,
routes, evaluates the selected experts, combines, returns one row — same ggml
kernels throughout. Stateless: no KV, no history between calls.

**Client**: the custom-op callback serialises, blocks, and drops the returned
row where the combine used to land.

**Protocol v1** (requirements, not a wire format):

- Request `{version, layer, x: f32[n_embd x n_tokens], return_mode}`.
- Response: one combined row per token, or `n_used` per-slot rows plus expert
  ids under `return_mode = per_slot` — a debug mode that lets the client
  reproduce the combine independently.
- Header carries **server-side durations**: parse, route, compute, serialise,
  on the server's own clock.
- `n_tokens > 1` in one call for prefill. Version field, explicit dims, hard
  errors on mismatch. Expert ids stay **global** wherever they appear, so a
  backend that later shards internally maps them itself.

**Logging** (durations only — no cross-machine clock sync; `network + queueing
= RTT − server_total`): per-RPC `(token_idx, layer, bytes_out, bytes_in,
rtt_us, t_serialize_us, t_deserialize_us, srv_parse_us, srv_route_us,
srv_compute_us, srv_serialize_us)`; per-token trunk/wait/overlap; routing ids
per (layer, token), logged on the backend. In-memory on the hot path, JSONL at
run end, analysis in Python.

**Gate**: KL == 0 over the 5-prompt corpus, same machine, same binary.

Two questions to settle here, where they are cheap:

- Is the server-side combine byte-identical to the client-side one? It should
  be if the backend applies routing weights and pairwise-adds slots in index
  order, exactly as the baseline does — that makes the 8x traffic reduction
  free rather than a trade. Verify, do not assume.
- Does moving the router across the boundary change anything? It is the same
  matmul on the same activation, so a failure here indicts the transport, not
  the routing.

**Measure**: round-trip latency, serialise/deserialise cost, and the
per-request overhead phase 2 will pay again over a real link.

### 2. Backend on another machine — **Planned**

Direct-attach 10GbE, TCP_NODELAY, jumbo frames; same OS and toolchain both
ends. Gate: KL == 0 still. Measure sustained req/s, RTT distribution, and
end-to-end tok/s against the 256 req/s the CPU ceiling allows — the gap is the
RPC tax. Then latency hiding: issue the request, run the shared expert on the
client while it is in flight, collect after. A `moe_send`/`moe_recv` split
around the shared-expert branch gets this for free, since the CPU backend
executes graph nodes in order.

### 3. Vulkan experts inside the backend — **Planned**

Hold a resident expert subset on the 4 Vega dies, serve misses from DRAM. 23%
of 563 GiB fits in 128 GiB of VRAM, so this is worth roughly the resident
fraction and no more. Placement is **static, not LRU** — PCIe is ~5.7x slower
than DRAM, so cache refill never pays. Q4_K experts are the composing lever
(~32% less DRAM traffic, ~33% resident). Cost model and per-phase dispatch
numbers: ../moe-offload.

### 4. Trunk on a modern GPU — **Planned**

Attention + KV + shared expert onto a Blackwell-class card; nano-glm needs a
GPU backend (it currently aborts if one is present). End-to-end bit-exactness
ends here by construction.

## Ideas, unproven

- **Speculative routing.** With the router on the backend, run layer N+1's
  router on layer N's activation while idle and prefetch the likely experts.
  Buys nothing while experts are in DRAM (bandwidth-bound; prefetch adds no
  bandwidth), and the idle window it needs shrinks to ~3% once the trunk is
  fast. Its home is K3, where experts exceed RAM and prefetch targets storage.
  Cheap to test: log the overlap between speculative and real top-k.
- **Huge pages** for the expert store — 31 MB per expert against a 1536-entry
  L2 TLB.

## Parked

Client-side expert sharding, static-vs-replicated dispatch policies, balanced
slot assignment across peers, asymmetric split-ratio tuning. All assumed
experts spread over machines the client chooses between; with one backend
holding everything, none applies. Global expert ids in the protocol are what
lets this return as a backend-side concern without a client change.

## Links

- client seam: `src/nano_glm.cpp` — `moe_proj_cb`, combine at `cur_experts`,
  shared expert at `ffn_up_shexp` (branch `phase0-moe-dispatch`)
- bit-exact kernel contract: `ggml/src/ggml-cpu/ggml-cpu.c`
  `ggml_compute_forward_mul_mat_id_one_chunk`; public traits in
  `ggml/include/ggml-cpu.h` (`ggml_get_type_traits_cpu`)
- custom ops: `ggml/include/ggml.h` — `ggml_custom_4d`; execution order:
  `ggml_graph_compute_thread`, `ggml_barrier`
- verification: ../logit-kld — `compare.py`, `rescore --sim-gen`, corpus in
  `prompts/`, noise floors in its README
- GPU cost model: ../moe-offload/README.md
- platform traps (AMD-Metal NaN, mmap variance, core counts, run config):
  repo `CLAUDE.md`
