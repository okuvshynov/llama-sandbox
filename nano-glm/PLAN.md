# Plan: distributed MoE evaluation (expert parallelism)

## Goal

Run truly massive MoE models — target: Kimi-K3 (~1.5TB of weights,
https://huggingface.co/moonshotai/Kimi-K3) — across two machines that
individually cannot hold them:

- 2× Mac Pro 2019 (Intel Xeon, same ISA), 768GB DDR4 each (1536GB total);
- 2× Vega II Duo per machine (4 GPU dies × 32GB HBM = 128GB per machine,
  256GB total) — **later phase only**, see "GPUs" below;
- dual built-in 10GbE between the machines; TCP for activations.

Strategy: expert parallelism. The routed-expert FFN is ~95%+ of the weights
in DeepSeek-lineage MoE models and is architecturally the *simple* part
(the per-model differences live in attention/trunk). We build a dedicated
**expert-shard evaluation service** that only knows how to run routed
expert FFNs over a set of expert weights it holds; the client (nano-glm)
keeps the trunk: attention, router, shared expert, combine.

All experimentation happens on GLM-5.2 first (smaller, already ported,
bit-exact baseline exists via ../logit-kld). Kimi-K3 is DeepSeek-lineage:
same routed swiglu experts + shared expert, so the server carries over;
only the client trunk is new work.

Development style, per repo conventions: manual/explicit version first,
promote to a proper ggml backend only if the data says so; orchestration
(launching servers, shard configs, analysis) in external scripts, C++ kept
minimal.

## Why not the ggml scheduler / meta backend (for now)

- `ggml_backend_sched` (ggml-backend.cpp) does placement and partitioning,
  not parallel execution: splits run sequentially in submission order
  (`ggml_backend_sched_compute_splits`); the `parallel` flag only enables
  pipeline parallelism across *successive graph evaluations* via input
  copies + events (`n_copies`, `GGML_SCHED_MAX_COPIES`) — useless for
  single-stream decode and for synchronous backends.
- Upstream's answer to multi-GPU is the **meta backend**
  (ggml-backend-meta.cpp, `GGML_BACKEND_DEVICE_TYPE_META`): one virtual
  backend wrapping N devices, split-axis propagation per op
  (`ggml_backend_meta_get_split_state`, `handle_mul_mat`), concurrent
  per-device subgraphs, allreduce via `ggml_backend_comm_allreduce_tensor`
  (NCCL on CUDA, generic fallback). Policy is an application callback
  (`llama_meta_device_get_split_state` in llama-model.cpp — per-tensor-name
  regexes choosing split axes). It validates the "looks sequential from
  the scheduler, hides parallelism inside" design.
- But the meta backend today treats `GGML_OP_MUL_MAT_ID` like `MUL_MAT`
  (tensor-parallel splits of every expert); true expert parallelism
  (split along the expert dim, `ne[2]`) is not implemented, and the meta
  backend has no cross-machine story. For a 2-node CPU cluster, a manual
  client/server design is less work and fully observable. If this ever
  moves upstream, EP maps onto the meta backend's existing
  `GGML_BACKEND_SPLIT_AXIS_2` + `PARTIAL` + allreduce machinery.

## Architecture

```
machine A                                machine B
┌──────────────────────────────┐         ┌────────────────────┐
│ nano-glm client              │   TCP   │ expert-shard server│
│  trunk: attn, router, shexp, │◄───────►│  (expert weights)  │
│  combine, KV cache, sampling │         └────────────────────┘
│  dispatcher (slot → server)  │
│ expert-shard server (local)  │
└──────────────────────────────┘
```

**Server contract** (deliberately narrow): holds a declared set of expert
tensors per layer (`ffn_{up,gate,down}_exps` slices or full copies);
evaluates `(layer, expert_id, x) → down-projection row` for an explicit
list of (token, slot, expert_id) work items. Stateless (no KV), config is
`(layer → expert set)`. All gating/model-specific logic stays client-side,
so the server is model-agnostic across GLM/Kimi-style MoE.

**Dispatcher** (client-side policy, the load-balancing brain): decides per
token/layer which server evaluates which selected slot. Because slot rows
are assembled client-side into the baseline combine order, *any* partition
of slots is numerically valid — static sharding, balanced dispatch over
replicas, and hybrids are all dispatch policies against one protocol.

**Client integration**: the routed-expert block in nano_glm.cpp (search
`ffn_up_exps` / the `ggml_mul_mat_id` calls in `build_graph`) is replaced
by custom-op nodes (`ggml_map_custom3`): callback sends work items to all
servers, waits, assembles the `experts` tensor; the pairwise-add combine
(search `cur_experts`) stays byte-for-byte as today. Graph topology stays
constant across tokens, so graph reuse keeps working. Later, a `moe_send`
node placed before the shared-expert branch and a `moe_recv` node after it
overlap the RPC with shexp compute (the CPU backend executes nodes in
order — free latency hiding, no threads).

## Verification methodology

The whole point of nano-glm is a bit-exact baseline; distribution must not
lose it prematurely:

- Both machines are Mac Pro 7,1 running the same binary → slot rows computed
  remotely are **bit-identical** to local ones. The bit-exact bar survives
  crossing the network. Same ISA is necessary but **not sufficient**: the
  Windows port (2026-08-06) measured llama.cpp disagreeing with *itself* by
  8.85e-3 mean KL between Apple-clang and MSVC builds on the same Xeon W-3245
  at the same AVX-512 level (compiler FMA contraction), and ggml's matmul
  partitioning makes the logits depend on **thread count** too. So the run
  configuration must be pinned across the cluster: same ISA, same toolchain,
  same `-t`. All three are free here (identical machines, one binary, one
  launch config) — but a KL==0 gate below will fail on a mismatch of any of
  them, for reasons having nothing to do with the sharding math. Check the run
  configuration first when a gate fails.
- Bit-exactness holds as long as (a) per-slot rows travel unsummed and
  (b) the combine runs client-side in baseline op order.
- When we deliberately trade exactness for speed (partial sums, f16 wire),
  the regression is *measured*, not assumed: ../logit-kld `compare.py`
  against the bit-exact run, read against the documented batch-shape noise
  floors. Every phase below ends with a logit-kld gate on the 5-prompt
  corpus.

## Phases

### Phase 0 — shard decomposition in-graph (one process, no network)

Validate the sharding math with zero distribution complexity.

1. Rewrite the routed block as S=4 shard branches: `ggml_view_3d` slices
   of the expert tensors along `ne[2]` (contiguous at stride `nb[2]`, no
   repacking of mmap'd weights), id remap/mask to local range, per-shard
   `ggml_mul_mat_id`, slot rows reassembled into the full `experts`
   layout; combine unchanged.
2. Gate: 5-prompt corpus, KL == 0, output bit-identical to baseline.

This nails id remapping, non-local-slot masking, and slot reassembly —
the parts where a bug is a silently wrong answer.

### Phase 1 — process split, localhost TCP

3. Extract the expert-shard server binary (reuses nano-glm's loader; mmaps
   the same GGUF, touches only its expert ranges — partial page-cache
   residency is free).
4. Protocol v1 (below), lkldtopk discipline: versioned, length-prefixed,
   explicit dims, loud errors.
5. Client custom-op integration; dispatcher with static-shard policy.
6. Gate: 2 local server processes, bit-identity again.

### Phase 2 — second machine

7. Direct-attach 10GbE (TCP_NODELAY, jumbo frames). Two run modes:
   - **static shard**: experts split ~50/50 (~225GB/machine for GLM);
   - **replicated + dynamic dispatch**: full expert copy on both machines
     (~450GB each, + ~30GB trunk on A — fits in 768GB), dispatcher
     assigns slots at runtime for perfect balance.
8. Gate: bit-identity still (same ISA/binary), plus the first real
   experiment: static vs dynamic on identical prompts — tok/s + balance
   histograms from the logs.

### Phase 3 — performance mode (each step logit-kld-quantified)

9. Partial-sum responses (server pre-reduces its slots): ~8× less return
   traffic; costs bit-exactness → measure KL.
10. `moe_send`/`moe_recv` split to overlap RPC with shared-expert compute;
    logs report overlap efficiency.
11. f16/bf16 activations on the wire → measure KL.
12. Batched prefill requests + bin-packing dispatch (keep each selected
    expert's tokens on one machine; balance total (token, slot) pairs).
13. Tune the asymmetric split ratio (machine A also runs trunk/shexp, so
    optimum may be 3/5 not 4/4) — driven by the latency logs.

### Phase 4 — Kimi-K3 and GPUs

14. K3 fit: ~1.4TB of experts does NOT split 50/50 into 768GB alongside
    the trunk → asymmetric static split (A holds fewer experts) and/or
    one quant step down; partial replication (static halves + a
    replicated overlap set for dispatch smoothing) only if the GLM
    routing histograms show it has headroom (DeepSeek-lineage models are
    trained load-balanced; per-expert marginals may be too flat).
15. K3 client trunk port (DeepSeek-V3-style MLA; much of nano-glm's GLM
    code is adjacent). Server unchanged.
16. GPUs — **Vulkan on the Vega IIs**, not Metal (Metal on this hardware
    silently NaNs large batches; see repo CLAUDE.md "AMD-Metal" entry).
    Vulkan is a different kernel stack, so the Metal history doesn't
    transfer — but trust is earned the same way: A/A gate (same shard on
    Vulkan vs CPU: NaN scan, then KL vs the bit-exact baseline) before
    any result is believed. Candidate roles: hosting the trunk, or hot
    expert subsets; 8×32GB granularity makes full expert residency
    awkward. Strictly off the critical path.

## Protocol v1 (design requirements, not final wire format)

Absorb from day one (cheap now, painful to retrofit):

- Request: `{layer, work items: [(token_idx, slot_idx, expert_id)], x:
  f32[n_embd × n_tokens]}` — explicit slot assignment (dispatcher decides,
  server obeys), global expert ids (server maps through its own table; no
  remap coupling between client and shard layout).
- Response: rows per work item + **server-side durations in the header**
  (parse, compute, serialize — measured on the server's steady clock).
- Server config declares held experts per layer (range or "all").
- Version field; dims explicit; hard errors on mismatch.

## Latency & routing logging

Durations only — never absolute cross-machine timestamps (no clock sync
needed: client measures RTT on its clock, server reports its internal
durations, `network + queueing = RTT − server_total` by subtraction).

- Per-RPC record: `(token_idx, layer, server_id, n_slots, bytes_out,
  bytes_in, rtt_us, t_serialize_us, t_deserialize_us, srv_parse_us,
  srv_compute_us, srv_serialize_us)`.
- Per-token record: trunk compute, MoE wait, overlap efficiency.
- Routing record: the selected expert ids per (layer, token) — feeds the
  imbalance analysis, the per-expert frequency histogram (K3 partial
  replication question), and static-vs-dynamic comparisons.
- Mechanics: in-memory append in the hot path, JSONL flush at run end
  into the run's results dir; aggregation/percentiles/plots in an
  external Python script.

## Performance expectations (GLM-5.2, decode batch 1)

- Wire per token per MoE layer: ~28KB out (n_embd f32 + ids), ~224KB back
  in per-slot mode (8 × n_embd f32); ~78 layers → ~20MB/token ≈ 16ms on
  10GbE + ~78 RTTs ≈ ~10-15ms direct-attach. Against a ~500ms/token
  compute budget: ~5% overhead. Partial-sum mode: ~4.4MB/token.
- Static 128/128 sharding pays E[max(k, 8−k)] ≈ 5.1 of 8 expert-reads per
  layer (Binomial(8, ½), each layer is a barrier) → ~25-30% slower than
  balanced dispatch's 4/4, before any routing skew. This is the case for
  replicated + dynamic dispatch.
- Ceiling: decode is weight-bandwidth-bound (~66GB/s effective per
  machine, from the observed 2 tok/s × ~33GB active). EP splits expert
  reads across machines → ceiling ≈ 2× minus network: GLM 2 → ~3.5 tok/s
  is success. K3 on CPUs lands in low single digits tok/s; 10+ needs the
  GPU phase.

## References (searchable anchors, not line numbers)

- ggml scheduler: `ggml/src/ggml-backend.cpp` —
  `ggml_backend_sched_split_graph` (placement passes, op_offload),
  `ggml_backend_sched_compute_splits` (sequential split loop, MoE
  used-expert copy optimization), `GGML_SCHED_MAX_COPIES` / `n_copies`
  (what `parallel=true` actually does), `GGML_SCHED_DEBUG` env var
  (dump split assignments).
- meta backend (upstream TP): `ggml/src/ggml-backend-meta.cpp` —
  `ggml_backend_meta_get_split_state`, `handle_mul_mat`,
  `GGML_BACKEND_SPLIT_AXIS_*` / `MIRRORED` / `PARTIAL`,
  `ggml_backend_comm_allreduce_tensor`; policy callback
  `llama_meta_device_get_split_state` in `src/llama-model.cpp`.
- custom ops: `ggml/include/ggml.h` — `ggml_map_custom3`,
  `GGML_OP_CUSTOM` (callback receives `ith`/`nth`; do IO on `ith == 0`).
- CPU execution model: `ggml/src/ggml-cpu/ggml-cpu.c` —
  `ggml_graph_compute_thread`, `ggml_barrier` (all threads cooperate on
  one node at a time; nodes strictly in order — what makes the
  send/recv-split overlap trick work).
- nano-glm seam: `src/nano_glm.cpp` — routed-expert block (search
  `ffn_up_exps`, `ggml_mul_mat_id`), baseline combine (search
  `cur_experts`), shared expert (search `ffn_up_shexp`).
- verification harness: `../logit-kld` — `compare.py`, the batch-shape
  noise-floor discussion in its README, the 5-prompt corpus in
  `prompts/`.
