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

**`gate.py` against a stub model.** Two runs of stock `llama-perplexity`, both
placing the routed experts in our buffer with `-ot`, differing only in whether
we claim any ops (`MOESERV_DISABLE=1` is the control). Each writes its
log-probabilities with `--kl-divergence-base`; those files are a deterministic
function of the logits, so identical files mean our compute changed nothing.
Bit-identical through `passthrough`; a stated numerical floor after.

The harness is llama.cpp. Nothing here defines the graph, the ops or the
arithmetic a second time, because a wrong test that passes is worse than no
test. `make_stub.py` is what makes it affordable — the first four layers of
DeepSeek-V4-Flash are ~16 GiB and load in seconds, against 150 GiB and minutes.

**The control is not stock llama.cpp, and that is not a shortcut.** llama.cpp
overrides `-ot exps=CPU` to **CPU_REPACK** and rewrites MXFP4 experts into a
blocked layout multiplied by a different GEMM (repo `CLAUDE.md`). We cannot
repack — owning the weights is the point, and a second 137 GiB copy is not
available — so that comparison differs in weight layout as well as in backend.
It is worth knowing and `--vs-stock` measures it; it is not the gate.

`test-backend-ops` does **not** work here and must never be quoted as
conformance — it calls `supports_op` before allocating anything, so our
ownership guard answers no and it reports OK over 0/0 tests. Useful only as a
smoke test that the buffer type allocates and frees.

Every check must assert the backend was *engaged* — and, since today, also
where the weights went, which is the assertion that would have caught the
CPU_REPACK control being wrong the first time instead of after an hour. Runs
that cannot prove both from their own log abort the gate rather than being
compared. Five separate "passes" in this project have now tested nothing at
all, or tested the wrong thing.

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

One split per MoE layer, nothing outside it. Throughput recorded, not compared —
an extra split boundary buys nothing yet and it is expected to be slower.

Bit-identity was claimed here on **generated text**, which `gate` has since
shown to be the weaker statement: at the logit level our compute is bit-exact
against the same-placement control, and stock llama.cpp differs from both
because of the repack. Text identity would have survived either outcome.

### `gate` — Done

`gate.py` + `make_stub.py`: the correctness contract above, running in minutes
against a cut-down model rather than in an hour against 150 GiB. Result:

    ctl    experts -> MoE         our compute: no
    moe    experts -> MoE         our compute: yes
    PASS: our compute is bit-identical to llama.cpp's on the same weights

Three things it established that were previously assumed.

**Our compute is exactly llama.cpp's.** Same weights, same threads, one
unsplit graph versus a claimed split — byte-identical log-probabilities. The
split boundary costs nothing arithmetically.

**Owning the weights does cost something, and it is now measured.** Against
stock (`--vs-stock`), where llama.cpp overrides `-ot exps=CPU` to **CPU_REPACK**
and multiplies MXFP4 experts by a different GEMM: mean KLD 3.6e-5, max 2.1e-3,
top-1 agreement 99.804% over four layers. That is the price of ownership, not a
defect, and it will not go away — so **the project's headline claim is
"bit-identical arithmetic on the weights as they sit in the file", never
"bit-identical to a stock run"**.

**A prefix of layers is a valid model; a renumbered layer is not.** Every
per-layer thing llama.cpp reads is indexed by layer number, so a prefix needs no
metadata surgery beyond `block_count` and cutting the arrays whose length must
equal it. Four layers is the minimum for deepseek4: the compressor kinds appear
at layers 2 (ratio 4) and 3 (ratio 128), and a stub with neither divides by zero
building an empty KV cache.

Two mistakes worth keeping. `-ot exps=CPU` was used as the control for an hour
on the belief that naming a buffer type bypasses the repack selection — it does
not, and the run says so in one `-v` line that was not being read. And the
`passthrough` bit-identity claim rested on generated text, which would have
looked the same either way; the gate now asserts where the weights went, not
only that something was computed.

### `tape` — Done, and no longer a correctness path

`MOESERV_CAPTURE=<dir>` records a split generically — node list with ops,
shapes, `op_params` and source indices, plus the data of every tensor the split
does not produce or does not consume, content-addressed into `blobs/`.
`tape_inspect.py` reads it back. One record per distinct split shape, so a run
of any length yields one prefill and one decode record and then costs nothing.

It was built to be half of a replay harness, and `gate` is a better answer to
the same question: llama.cpp is the harness, so nothing has to define the graph
a second time. What survives is inspection — it is how we know the backend is
handed exactly the expert half of `build_moe_ffn` and nothing else — and a
plausible input for a standalone kernel benchmark under `dies`. **If neither
earns its keep, delete it**; a second thing that looks like a correctness path
is a liability.

Kept from building it: captures are bounded by `MOESERV_CAPTURE_MAX_RECORDS` /
`_MAX_MB` because the first unbounded one was 140 GB — llama.cpp hands
`mul_mat_id` the whole 256-expert tensor, 1.07 GB, three per layer.

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

**Gate: `gate.py --tol`, and the tolerance has to be argued rather than picked.**
The CPU path keeps its `--tol 0` bit-identity check — it does not retire when
Vulkan lands, it becomes the regression test on the plumbing. For Vulkan the
question is what number to allow, and two things already measured bound it:

- The **repack gap** (mean KLD 3.6e-5, max 2.1e-3 over four layers) is a
  same-machine, same-model example of *two correct kernels disagreeing*. A
  Vulkan difference of that order is evidence of nothing except different
  rounding; one much larger is worth investigating.
- End-to-end KLD **saturates** on a deep model (`../nano-glm/OPTIMIZATION.md`),
  and a **mathematically-invariant wrong graph hides inside the precision noise
  floor** (repo `CLAUDE.md`, the Hadamard incident). So a passing tolerance at
  the logits is necessary and nowhere near sufficient.

Which means the stub earns a second job: at four layers there is far less
amplification than at 43, and the layer count is a knob (`make_stub.py
--layers`). If a Vulkan difference grows with depth faster than the repack gap
does, that is a signal the logits alone cannot give. Whether that is enough, or
whether `dies` needs a per-op comparison after all, is the open question — and
if it is the latter, `tape` is the thing to revive rather than replace.

### Later — one line each

- **`residency`** — which experts sit on which die, and what happens to the rest.
- **`wire`** — in-process or a separate process over a socket. Undecided until a
  call's real cost is known.
- **`shaders`** — custom kernels for these dies, if the generic path leaves
  something. May not exist as a step; the known win is chunking, which is `dies`.
- **`breadth`** — a third model, once two are solid.

## Links

- `README.md` — build, run, how the mechanism works, and the things that are not
  obvious about writing an out-of-tree ggml backend.
- `gate.py` / `make_stub.py` — the correctness gate and the stub it runs on,
  each carrying its own reasoning at the top.
- `../nano-glm/OPTIMIZATION.md` — the measurements this project starts from:
  what the dies are worth, the 8-pair cliff, expert residency at Q8.
- Commits `b6f46b1` (plan), `3b228f2` (`handshake`), `bd7abcc` (`passthrough`),
  `9fe5558` (`tape` capture).
