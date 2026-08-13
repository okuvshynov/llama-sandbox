# moe-serv — a ggml backend that claims the MoE block

A shared library that llama.cpp loads at runtime, takes ownership of the routed
expert weights, and computes the whole expert block. Nothing else. Every other
part of the model stays llama.cpp's problem.

## Why this instead of a trunk

`../nano-glm` established that a bit-exact port is possible and what the MoE
split costs, and that took a trunk: tokenizer, KV cache, attention at three
compression ratios, hyper-connections, a lightning indexer, an output head. All
of it existed to get to the expert block, and all of it is a second thing to
debug when a number looks wrong.

The expert block is the part worth owning. It is where the bytes are (137 of
DeepSeek-V4-Flash's 150.7 GiB), it is where hardware-specific tuning pays, and
it is small enough to verify exhaustively.

That only works if llama.cpp will hand us exactly the expert block and keep
everything else. **It will, using mechanisms that already exist** — a documented
environment variable for loading an out-of-tree backend, and the same
tensor-override flag `-cmoe` is built from. No llama.cpp patch, no fork, no
maintained diff against upstream. The next section is what was read to confirm
that, with file and line references, because the whole project rests on it.

## The mechanism, verified against llama.cpp @ 6a32c29a7

**Loading.** `ggml-backend-reg.cpp:588` reads `GGML_BACKEND_PATH` and calls
`ggml_backend_load` on it, after all built-in backends. The library exports two
symbols: `ggml_backend_score()` (0 means "not usable here") and
`ggml_backend_init()` returning a `ggml_backend_reg_t`.

**Claiming the weights.** `-ot` / `--override-tensor` binds a tensor-name regex
to a *buffer type*, resolved by name. `common/arg.cpp:252` calls
`ggml_backend_load_all()` **before** building its name -> buft map from every
registered device, so a `GGML_BACKEND_PATH` backend is already present and
targetable. `-cmoe` is nothing but this with a canned regex:

    LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps"
    -cmoe    ->  { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() }
    -ncmoe N ->  the same, per layer, for the first N

so our invocation is that regex pointed at us instead of the CPU. Owning the
weights (rather than borrowing them via the ACCEL/host-buffer trick BLAS uses)
is what makes residency ours to decide later, with no per-op copies.

**Getting the ops.** `ggml_backend_sched_backend_from_buffer`
(`ggml-backend.cpp:884`) returns *the highest-priority backend that supports both
the weight's buffer type and the op*. We own the expert weights, so the three
`mul_mat_id` nodes are ours after pass 1. Pass 2 then expands that assignment
across **adjacent unassigned** nodes (`ggml-backend.cpp:1113`), which is how the
weightless ops in between join the same split — one split, one `graph_compute`,
one submit per layer.

**Configuring it from a script.** `-ot` carries
`.set_env("LLAMA_ARG_OVERRIDE_TENSOR")` and `common_params_parse` applies
environment variables *before* command-line args, so a launcher sets the default
and the user's own flags still override. One exception, and it is the tool we
reach for first: **`llama-bench` has its own parser** (no `common_params_parse`,
its only `getenv` is `HF_TOKEN`), so it needs `-ot` passed explicitly.

    export GGML_BACKEND_PATH=.../moeserv.dll
    export LLAMA_ARG_OVERRIDE_TENSOR='\.ffn_(up|down|gate|gate_up)_(ch|)exps=MoE'
    llama-cli   -m model.gguf -p "..."                  # env is enough
    llama-bench -m model.gguf -ot '...=MoE'             # must be explicit

## What "the MoE block" is, as ops

From `build_moe_ffn` (`src/llama-graph.cpp`), the expert half after routing:

| op | notes |
|---|---|
| `MUL_MAT_ID` | up, gate, down — three of them, ours by weight ownership |
| `CLAMP` | deepseek4 only (`swiglu_clamp_exp` = 10.0); glm-dsa has none |
| `GLU` | `ggml_swiglu_split` |
| `MUL` | the router-weight multiply |
| `ADD` | the expert sum, `n_expert_used - 1` of them |
| `ADD_ID` | expert bias — neither of our two models has one |
| views | `reshape_3d`, `view_2d` — pass 2 skips view ops, they never break a split |

**We claim `MUL_MAT_ID`, `CLAMP`, `GLU`, `MUL` and not `ADD`.** That covers
everything expensive in one split; the expert sum is `n_expert_used - 1` adds of
`[n_embd, n_tokens]` and stays on the CPU. `ADD` is deliberately excluded
because it is also the residual add, and pass 2's expansion does **not** reset
`cur_backend_id` when it meets an op it cannot place
(`ggml_backend_sched_set_if_supported`, `ggml-backend.cpp:1047`) — so a claim on
`ADD` risks reaching past the block into the trunk. Revisit only if a
measurement asks: claiming `ADD` would let a remote service return the summed
`[n_embd, n_tokens]` instead of `[n_embd, n_expert_used, n_tokens]`, which is
`n_expert_used`x less payload. That is a transport question and there is no
transport yet.

## Steps have names, not numbers

`../nano-glm/PLAN.md` numbered its steps, and then reality reordered them: step
3 shipped before step 2, step 14 was inserted years after step 11, and
`OPTIMIZATION.md` grew entries called "3-adjacent" because there was nowhere
else to put them. Code comments referring to "step 9" now need a lookup, and
some of them are wrong. A number claims an ordering the work does not actually
have.

So the steps below are named after what they do. Names survive being
reprioritised, dropped or split, and a comment saying "see `tape`" stays true
whatever happens to the order.

## `handshake` — it loads, and llama.cpp offers us the tensors

A library that registers one device and claims nothing.

- CMake project building `moeserv` as a shared library against the same
  `LLAMA_CPP_DIR` ggml that everything else here uses. No llama.cpp source
  changes, ever — if something needs one, that is a finding to write down, not a
  patch to carry.
- Exports `ggml_backend_score` and `ggml_backend_init`.
- One `ggml_backend_reg`, one `ggml_backend_device`, named `MoE`.
- A buffer type named `MoE` whose allocator is, for now, plain host memory:
  `malloc`, `memcpy` for `set_tensor`/`get_tensor`, `is_host` true.
- `supports_op` returns **false for everything**. `supports_buft` true only for
  our own.

Done when:
- `-ot "x=NOSUCHBUFT"` errors and lists `MoE` among "Available buffer types",
  which proves registration and name resolution without running a model.
- A real run with `-ot exps=MoE` produces the same output as one without — a
  backend that claims no ops must be invisible even when explicitly targeted.

**DONE**, and it did more than this section expected.

DeepSeek-V4-Flash, `llama-completion`, greedy, fixed seed, 24 tokens:

    load_backend: loaded MoE backend from ...moeserv.dll
    load_tensors:   CPU_Mapped model buffer size =  46935.53 MiB
    load_tensors:   CPU_Mapped model buffer size =  46309.94 MiB
    load_tensors:   CPU_Mapped model buffer size =  46086.90 MiB
    load_tensors:   CPU_Mapped model buffer size =  11769.43 MiB
    load_tensors:          MoE model buffer size = 140352.00 MiB

Generated text byte-identical to the stock run.

**The prediction in the first draft was wrong.** It said llama.cpp would refuse
to place a weight in a buffer whose device cannot run the op, so `-ot` would be
ignored and the run would fall back to the CPU. It is not ignored: `-ot` is an
explicit override and llama.cpp honours it, so **137 GiB of expert weights are
already in our buffer with `supports_op` returning false for everything**.

Correctness survives for a different reason than the one written down: our
buffer is host memory with `is_host` true, `ggml_backend_cpu_device_supports_buft`
accepts any host buffer, and `ggml_backend_sched_backend_from_buffer` therefore
hands every op to the CPU — which reads our tensors in place. Nothing is copied
and nothing changes.

Two consequences. **`passthrough` is smaller than planned**: weight ownership is
already working, so what is left is turning `supports_op` on for the block and
writing `graph_compute`. And **`is_host` is now load-bearing** — the moment the
buffer stops being host-visible (`dies`), the CPU can no longer read it and
every op we do not claim becomes a copy. That is the real content of the weight
placement decision deferred to `dies`, and it is sharper than "device memory or
mirror".

## `passthrough` — own the experts, compute them, change nothing

Claim the block and delegate the arithmetic to a `ggml_backend_cpu` instance we
hold internally.

This is the whole point of the increment: our buffer is host memory and the CPU
backend can read it directly, so `graph_compute` hands the subgraph straight to
CPU. **Same kernels, same data, same order — the result is bit-identical to not
loading us at all, by construction.** Any logit difference is a plumbing bug and
cannot be arithmetic. That is the cheapest possible way to prove the hard parts:
weight ownership, buffer lifetime, split boundaries, in/out tensor handling.

- `supports_op` true for `MUL_MAT_ID`, `CLAMP`, `GLU`, `MUL`; also the free
  structural ops BLAS accepts (`NONE`, `RESHAPE`, `VIEW`, `PERMUTE`,
  `TRANSPOSE`) so views do not fragment anything.
- Guard `supports_op` on shapes we actually mean: `MUL_MAT_ID` with `src0` in
  our buffer. An op we accept and cannot compute is a crash at best.
- `graph_compute` forwards the subgraph to the internal CPU backend.
- Report what we claimed at load: layer count, tensor count, bytes, and the
  op histogram. Silence here would make `dies` undebuggable.

Done when:
- **End-to-end output is bit-identical** with and without the backend, greedy,
  fixed seed. Byte comparison, not KL.
- **One split per MoE layer**, and nothing claimed outside the block.
- The split count and throughput are recorded, not tuned. `passthrough` is
  expected to be *slower* than plain CPU — an extra split boundary buys nothing
  yet.

**DONE.** DeepSeek-V4-Flash, `llama-completion`, greedy, fixed seed:

    MoE: first split has 13 nodes: MUL x1 MUL_MAT_ID x3 VIEW x6 CLAMP x2 GLU x1
    sched_reserve: graph splits = 87
    BIT-IDENTICAL generated text: 647 chars

The histogram is exactly the expert block — up, gate, down, deepseek4's two
SwiGLU clamps, the gate, the router-weight multiply — arriving as **one split**.
87 splits is 43 layers x 2 + 1, i.e. one MoE split per layer alternating with
CPU and nothing claimed outside.

### `test-backend-ops` does not work here, and that was a planning error

This file said llama.cpp's own conformance suite would give us per-op checking
for free. It does not, for a structural reason worth keeping.

`supports_op` is called on **every tensor in the context before anything is
allocated** (`tests/test-backend-ops.cpp:1355`), so a weight has no buffer at
that moment. Our `supports_op` requires `MUL_MAT_ID`'s `src0` to be *in our
buffer*, so we answer no to everything, and the suite reports
`Backend MoE: OK` over **0/0 tests**. A green tick on an empty set — the same
shape of non-result this project has now hit four times.

The guard is not the thing to change. `MUL_MAT_ID` appears only in MoE blocks,
so claiming it regardless of ownership would claim blocks whose weights are on
the CPU, and merely loading the library would alter a run — exactly the property
`handshake` exists to establish. Relaxing the guard to light up a test would
trade a real invariant for a green tick.

So the correctness contract loses its cheapest layer and the other two carry it:
end-to-end byte identity (which `passthrough` passes), and **`tape`**, which is
now the only per-op check that can see this backend, because it replays real
tensors that really are in our buffer. That raises `tape` from convenience to
requirement and it should come next.

What the suite did establish, for what it is worth: the device enumerates, the
buffer type allocates and frees across 16125 op cases, and nothing crashes.
That is a smoke test of the buffer, not conformance of the compute.

## `tape` — capture and replay

`graph_compute` sees every tensor the block consumes and produces. Write them
out, and the backend becomes reproducible without llama.cpp or a model.

- `MOESERV_CAPTURE=<dir>` dumps, per call: the op list, shapes and types, expert
  weights (once, by hash — they do not change), activations, ids, router
  weights, and the output.
- A standalone `moe-replay` reads a capture and runs it against any compute
  path, comparing to the recorded output and timing it.
- This is the unit-test corpus *and* the benchmark harness, taken from real runs
  of both models rather than invented. It is what makes `dies` and `shaders`
  safe to iterate on: a kernel change is checked in seconds against real
  tensors, not by reloading 150 GiB.

Captures are large, so: weights by content hash and stored once; activations
capped by a sampling rule; both documented in the format header.

## Later — bullets, deliberately

Named, unordered, and expected to be reprioritised — which is the point of not
numbering them.

- **`dies`** — compute claimed ops on the Vega II dies via Vulkan. Weight
  placement becomes a real decision: our buffer allocates device memory
  directly, or holds host memory and mirrors. Gate is `passthrough`'s plus a
  documented numerical floor instead of bit-identity.
- **`residency`** — the model does not fit. Which experts sit on which die, and
  what happens to the rest. `../nano-glm/OPTIMIZATION.md` has measurements to
  start from.
- **`wire`** — whether the service is in-process or a separate process reached
  over a socket. Deliberately undecided until `passthrough` and `dies` are done
  and we know what a call actually costs.
- **`shaders`** — custom kernels for these dies if the generic path leaves
  something on the table. `../nano-glm/OPTIMIZATION.md` records a 10.2x cliff at
  9 pairs per dispatch in `ggml_vk_use_mul_mat_vec_id`, which is the first thing
  to look at — and is a chunking decision, not a shader, so it may belong in
  `dies`.
- **`breadth`** — models beyond the two below. Only once those are solid.

## Correctness contract

Two layers, not the three this file originally listed:

1. End-to-end output, fixed seed, with and without the backend. Bit-identical
   through `passthrough`; a stated floor after.
2. Capture replay (`tape`) — real tensors from real runs, checked against
   recorded output. **Load-bearing**, because it is the only per-op check that
   can see an ownership-gated backend; see the `passthrough` section for why
   `test-backend-ops` cannot.

`test-backend-ops` is still worth running as a smoke test of the buffer type —
it allocates and frees across 16125 cases — but it reports OK over zero tests
and must never be quoted as conformance.

Neither layer depends on `../nano-glm`.

## Scope: two models, on purpose

**Everything here targets exactly two checkpoints**: `glm-dsa` (GLM-5.2,
UD-Q6_K) and `deepseek4` (DeepSeek-V4-Flash, UD-Q8_K_XL). They are the two that
have been run, gated and measured on this machine, and nothing is claimed about
anything else. "Support" means a passing correctness contract against a real
checkpoint, not that the code has no reason to fail.

That said, the expert block is one of the *least* model-specific parts of a
modern MoE transformer. It is a routed `mul_mat_id` triple with a gated
activation between them, and llama.cpp builds it for every MoE architecture from
one shared `build_moe_ffn`. So the reasonable expectation is that a third model
mostly works, and the reasonable posture is to not say so until one has been
run. The differences that *do* exist are per-model and small — which is exactly
why the op histogram is reported at load rather than assumed.

Between our two:

- deepseek4 clamps its SwiGLU at 10.0, glm-dsa does not — so `CLAMP` appears in
  one and not the other. This exact difference cost `../nano-glm` four commits
  when an architecture-neutral path silently omitted it and no gate covered the
  configuration that would have caught it.
- deepseek4's first three layers are hash-routed. The routing happens before the
  block and produces ids like any other; the expert half is unchanged.
- Expert counts and `n_expert_used` differ (256/8 vs 256/6). Nothing in the
  block should care, and `test-backend-ops` varies both.

Things `build_moe_ffn` can emit that **neither** of our two uses, so they are
untested and must be refused rather than mishandled: fused `gate_up_exps`,
expert bias (`ADD_ID`), `weight_before_ffn`, and the non-SwiGLU gates (GEGLU,
REGLU, `swiglu_oai`). `supports_op` should return false for a block shaped in
any of those ways, so an unported model falls back to the CPU and is slow rather
than wrong.

## Relationship to ../nano-glm

**No dependency, in either direction.** Copy anything worth copying — the wire
protocol, the compacted expert graph, the residency machinery — but as source
that then belongs here. The two projects have different lifetimes: nano-glm is
a reference that has already done its job, this is meant to be used.

## What this file deliberately does not decide

Transport, weight placement, kernel strategy, and residency policy. Each is a
performance decision, each depends on measurements that do not exist yet, and
each is cheaper to make once `passthrough` works than before.
