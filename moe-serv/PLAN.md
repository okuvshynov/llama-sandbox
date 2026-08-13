# moe-serv — plan

## Goal

A shared library that unmodified llama.cpp loads at runtime, which takes
ownership of the routed expert weights and computes the whole MoE block —
nothing else. The trunk, tokenizer, KV cache and head stay llama.cpp's problem.

The expert block is the part worth owning: it holds the bytes (137 of
DeepSeek-V4-Flash's 150.7 GiB), it is where hardware-specific tuning pays, and
it is small enough to verify exhaustively. `../nano-glm` proved a bit-exact port
is possible and measured what the MoE split costs, but it needed a whole trunk
to get there, and all of that is a second thing to debug when a number looks
wrong.

How llama.cpp is persuaded to hand us exactly that block — `GGML_BACKEND_PATH`,
`-ot`, and the scheduler's buffer-based assignment — is in `README.md`, with
file:line references. No llama.cpp patch, no fork, no maintained diff.

## Invariants

- **No llama.cpp source changes, ever.** If something appears to need one, that
  is a finding to write down, not a patch to carry.
- **Two models, named**: `glm-dsa` (GLM-5.2, UD-Q6_K) and `deepseek4`
  (DeepSeek-V4-Flash, UD-Q8_K_XL). "Supported" means a passing correctness
  contract against a real checkpoint. The expert block is among the least
  model-specific parts of a MoE transformer, so a third model probably mostly
  works — do not say so until one has been run.
- **Refuse what is untested.** `build_moe_ffn` can emit shapes neither model
  uses: fused `gate_up_exps`, expert bias (`ADD_ID`), `weight_before_ffn`, and
  the non-SwiGLU gates. `supports_op` must return false for those, so an
  unported model falls back to the CPU and is slow rather than wrong.
- **Claim only what descends from weights we own.** `MUL_MAT_ID` appears only in
  MoE blocks; claiming it regardless of ownership would mean merely loading the
  library alters a run. Never relax this to make a test light up.
- **`ADD` stays unclaimed.** It is also the residual add, and pass 2 does not
  reset its running backend id when it meets an op it cannot place
  (`ggml_backend_sched_set_if_supported`), so claiming it could reach past the
  block into the trunk.
- **No dependency on `../nano-glm`, in either direction.** Copy what is worth
  copying, as source that then belongs here.

## Correctness contract

1. End-to-end output, fixed seed, with and without the backend. Bit-identical
   through `passthrough`; a stated numerical floor after.
2. Capture replay (`tape`) — real tensors from real runs against recorded
   output. **Load-bearing**: it is the only per-op check that can see an
   ownership-gated backend.

`test-backend-ops` does **not** work here and must never be quoted as
conformance — it calls `supports_op` before allocating anything, so our
ownership guard answers no and it reports OK over 0/0 tests. Useful only as a
smoke test that the buffer type allocates and frees.

Every check must assert the backend was *engaged* — the loader line, the `MoE
model buffer size` line, the `first split has` line — and abort rather than
compare when that assertion fails. Four separate "passes" in this project have
so far tested nothing at all.

## Steps

Named, not numbered: `../nano-glm/PLAN.md` numbered its steps and reality
reordered them, leaving code comments pointing at "step 9" and an
`OPTIMIZATION.md` section called "3-adjacent". Names survive reprioritisation.

### `handshake` — Done (`3b228f2`)

Library registers, exposes a buffer type named `MoE` that `-ot` resolves, claims
no ops. Verified invisible when explicitly targeted.

Surprise worth carrying: `-ot` is honoured **regardless of `supports_op`**, so
140352 MiB of expert weights landed in our buffer before we claimed a single op.
The run stayed correct because the buffer is host memory and the CPU backend
accepts any host buffer. That made `passthrough` the smaller half, and it makes
**`is_host` load-bearing** — see `dies`.

### `passthrough` — Done (`bd7abcc`)

Claims `MUL_MAT_ID`, `CLAMP`, `GLU`, `MUL` (guarded by ownership) and computes
by handing the split to a CPU backend from the host's registry.

    MoE: first split has 13 nodes: MUL x1 MUL_MAT_ID x3 VIEW x6 CLAMP x2 GLU x1
    sched_reserve: graph splits = 87        (43 layers x 2 + 1)
    generated text bit-identical to stock

One split per MoE layer, nothing outside it. Throughput recorded, not compared —
an extra split boundary buys nothing yet and it is expected to be slower.

### `tape` — In progress: capture done, replay remaining

Capture what `graph_compute` already sees, and replay it without llama.cpp or a
model. Promoted from convenience to requirement by the `test-backend-ops`
finding: this is now the only per-op check that can see this backend.

**Capture.** `MOESERV_CAPTURE=<dir>` writes, per call: op list with shapes,
types and `op_params` (the GLU variant lives there), the ids and router weights,
the activations, and the output. Expert weights are the problem — 137 GiB, and
identical on every call — so they are written **once, keyed by content hash**,
with each record referencing the hash. Activations need a cap or a sampling rule
or a 24-token run fills a disk; both belong in the format header rather than in
someone's memory.

**Capture: done** (`src/moe_capture.h`, `tape_inspect.py`). A record of a
deepseek4 decode split reads back as exactly the expert half of `build_moe_ffn`
— `MUL_MAT_ID`, `CLAMP`, `MUL_MAT_ID`, `CLAMP`, `GLU`, `MUL_MAT_ID`, `MUL`, then
six terminal `VIEW`s — with all six leaves captured (three mxfp4 weight tensors,
activations, ids, router weights). Two format bugs that reading it back caught:

- The op *number* was recorded and the reader decoded `MUL_MAT_ID` as `op50`,
  because the enum is internal and had moved. The name is now recorded too.
- "Capture the last node" was the wrong output rule: this split's terminals are
  six views of the experts tensor, all consumed *outside* our claim, so a replay
  would have compared one sixth of the result and passed. The rule is now
  "every node nothing in this split consumes".

**Replay: remaining.** A standalone `moe-replay` reads a capture, rebuilds the
graph from the recorded node list, runs it against a chosen compute path, and
compares to the recorded terminals. Rebuilding rather than hard-coding the five
ops is the point — see the open question below.

**The capture must be complete enough to recompute the block independently**,
not merely to re-run our own graph — because of what `dies` needs, below.

**Captures are bounded by a budget, and that is the whole size policy.** The
first attempt on a 4-token run wrote **140 GB**: content addressing dedupes a
weight across calls, but each of the 43 layers has different experts and
llama.cpp hands `mul_mat_id` the *whole* 256-expert tensor — ~1.07 GB, three per
layer. `MOESERV_CAPTURE_MAX_RECORDS` and `MOESERV_CAPTURE_MAX_MB` stop the
capture early and say so.

A handful of records is what replay needs, so this is sufficient rather than a
compromise: the point is to check a kernel against real tensors, not to archive
a run. Keep it this way until something actually needs more.

If a larger corpus is ever wanted, the move is to **slice each `mul_mat_id`'s
weights to the experts its ids name** and remap the ids to `0..k-1` — ~80 MB per
decode record instead of 3.2 GB, and a faithful reduction rather than a
truncation. Not built, not needed yet, and it would put `mul_mat_id`-specific
knowledge into an otherwise generic capture, so it should wait for a reason.

Open questions:

- What identifies a record — layer index is not in the graph the backend sees.
  Call ordinal is available and probably enough.
- Whether replay rebuilds the graph from recorded metadata or hard-codes the
  five ops. Rebuilding is generic and tests *the graph the backend built*;
  hard-coding would re-introduce exactly the duplication that cost `../nano-glm`
  a 5.6% RMS bug.

Done when: a capture from each of the two models replays bit-identically
against the CPU path, **and a deliberately corrupted capture does not**. The
second half is the negative control, and given that four checks in this project
have passed while testing nothing, it is not optional.

### Why `tape` gates `dies`: a Vulkan kernel cannot be tested against CPU

`passthrough` is bit-identical because it delegates to the same kernels. Vulkan
will not be, so `dies` needs a tolerance — and "within epsilon of CPU" is a weak
test whose pass band is the size of the bugs it should catch. This repo has the
scar: end-to-end KL **saturates** on a deep model
(`../nano-glm/OPTIMIZATION.md`), and a **mathematically-invariant wrong graph
hides inside the precision noise floor** (repo `CLAUDE.md` — a structurally
wrong Hadamard rotation sat at exactly F16 epsilon, where "the magnitude looks
like precision" was evidence about magnitude, not cause).

What makes a numeric test well-posed *here* is that the block is a **pure
function of its inputs**: the router runs outside our claim, so expert ids
arrive as data and there is no argmax-flip nondeterminism to chase.

So replay should compute a **high-precision reference** from the captured
inputs — dequantize exactly (MXFP4/Q6_K dequant is exact into f32 and is
therefore not a divergence source), then matmuls and activation in f64 — and
compare *both* paths against it:

    err_cpu    = |cpu    - f64_reference|      the floor, measured not assumed
    err_vulkan = |vulkan - f64_reference|

The question becomes **"is the Vulkan error the same order as the CPU's own
error"** rather than "is Vulkan close to CPU". A differently-rounded kernel
lands on the floor; a structurally wrong one does not, even when its absolute
magnitude looks like precision.

Report **max-abs and max-rel, never a mean** — a mean over 4096 values hides the
one wrong element, which is what a mis-indexed view produces.

### `dies` — Planned, next

Compute claimed ops on the four Vega II dies via Vulkan.

The real decision is **weight placement, and `is_host` is what makes it sharp**.
Today the buffer is host memory, which is why every op we do *not* claim is free
— the CPU reads our tensors in place. The moment the buffer becomes device
memory, `ggml_backend_cpu_device_supports_buft` stops accepting it and each
unclaimed op becomes a copy. So the choice is not "device memory or mirror" but
"how much does the block have to claim before device-resident weights stop
costing more than they save".

Two known constraints, both from `../nano-glm/OPTIMIZATION.md`: a 10.2x cliff at
9 pairs per `mul_mat_id` dispatch (`ggml_vk_use_mul_mat_vec_id`), which is a
chunking decision rather than a shader; and 240 of 256 experts fitting in
124.7 GiB of VRAM at Q8, so partial residency is the normal case and not an
edge.

Gate: the two-sided comparison `tape` provides — Vulkan's error against the f64
reference must be the same order as the CPU's own, not merely close to CPU. Plus
`passthrough`'s bit-identity gate, which does **not** retire when `dies` lands:
the CPU delegation path stays, so it remains a regression check on the plumbing,
and the tolerance only ever applies to the Vulkan path.

### Later — one line each

- **`residency`** — which experts sit on which die, and what happens to the rest.
- **`wire`** — in-process or a separate process over a socket. Undecided until a
  call's real cost is known.
- **`shaders`** — custom kernels for these dies, if the generic path leaves
  something. May not exist as a step; the known win is chunking, which is `dies`.
- **`breadth`** — a third model, once two are solid.

## Links

- `README.md` — build, run, how the mechanism works, and six things that are not
  obvious about writing an out-of-tree ggml backend.
- `../nano-glm/OPTIMIZATION.md` — the measurements this project starts from:
  what the dies are worth, the 8-pair cliff, expert residency at Q8.
- Commits `b6f46b1` (plan), `3b228f2` (`handshake`), `bd7abcc` (`passthrough`).
