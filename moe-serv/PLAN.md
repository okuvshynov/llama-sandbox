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

### `tape` — Removed (`9fe5558`, `0ea5520`)

A capture format for the split `graph_compute` is handed, built to be half of a
replay harness. `gate` answered the same question better — llama.cpp is the
harness, so nothing has to define the graph a second time — leaving `tape` with
no job that justified carrying it. Deleted rather than kept "in case": it is one
`git show 9fe5558` away, and a second thing that looks like a correctness path
is a liability.

Two findings from it are worth having without the code. The block the backend
receives is exactly the expert half of `build_moe_ffn` and nothing else —
`MUL_MAT_ID`, `CLAMP`, `MUL_MAT_ID`, `CLAMP`, `GLU`, `MUL_MAT_ID`, `MUL`, then
six terminal `VIEW`s. And any future scheme that captures a `mul_mat_id`'s
inputs faces the same wall: llama.cpp hands it the *whole* 256-expert tensor,
1.07 GB, three per layer, so an unbounded capture of a 4-token run wrote 140 GB.
Slicing the weights to the experts the ids name (~80 MB per decode record) is
the way through, and it is the reason to revive this only with a purpose in
hand — see the open question under `dies`.

### `bench` — Done

`bench.py`: decode throughput for the gate's three configurations, CPU only.
Numbers and reasoning in `README.md`; the two results that matter here are that
**our split costs the real model ~0.4% and at most ~3%** of decode, and that
**this cannot be measured on the real model at all**.

That second one shapes everything after it. On DeepSeek-V4-Flash the
load-to-load spread is 7-9% while the effects are 2-6%, so the three
configurations interleave and the two loads rank them differently — one
3.1-3.6 t/s band and no ordering. More loads is the wrong answer at sqrt(n).
The 4-layer stub has a 0.0-2.3% spread, resolves the repack difference (3.2%),
and bounds the per-split cost at ≤216 µs, which scales to the real model by
split count. **So the stub is the measuring instrument and the real model is the
sanity check, for performance as well as for correctness.**

`bench.py` now refuses to print a delta smaller than the noise it just measured.
The first version printed three confident percentages under a table whose own
spread contradicted all of them.

### `dies` — In progress

Compute claimed ops on the four Vega II dies via Vulkan, reusing ggml's own
kernels. They are generic and not tuned for Vega II; that is fine as a start and
`shaders` is where it stops being fine.

**The ceiling first, because it decides what "good" means.** DeepSeek-V4-Flash
is 90.9% experts *by size*, but a decode step reads only 6 of 256 experts and
100% of everything else:

| | GiB read per token | share |
|---|---|---|
| routed experts (6/256) | 3.21 | **20.2%** |
| attention, shared expert, output head | 12.70 | 79.8% |

So if the dies made our whole block **free**, decode goes 3.46 -> 4.34 t/s: a
**1.25x ceiling** before any transfer or dispatch cost. `../nano-glm`'s best
llama.cpp Vulkan decode was 4.64 t/s at `-ngl 99 -ncmoe 24 -nopo 1
-ts 24/6/6/7` — which offloads *attention* and leaves experts on the CPU, i.e.
the other 80%. **For decode, the block we own is the less valuable half**, and
the two approaches compose rather than compete (`-ngl` is llama.cpp's business).
Prefill inverts this: at batch 512 nearly every expert is touched.

Nothing about that changes the plan — capacity is still the reason this project
exists, since the experts are precisely what does not fit — but it does mean a
decode result near 4.3 t/s is a *success*, not a disappointment, and that the
prefill number is the one to watch.

**Placement: whole layers, packed in device order, remainder on the CPU.**
3.188 GiB of experts per layer against 31.73 GiB per die gives **9 layers/die,
36 of 43 on the GPUs (84%)**, ~3 GiB/die spare. Three measurements decide this,
all from `../nano-glm`:

- **One die is as fast as four** (20.2 / 20.6 / 20.0 s, sd 0.1-0.2). The dies are
  bound by something fixed per dispatch, not by throughput — *adding dies buys
  VRAM capacity, not speed* — so the policy should minimise dispatches.
- **Routing skew does not transfer across prompts**: a placement built from other
  prompts catches 28.2% of selections against 23.1% for random. Hot-expert
  placement is ~5pp for a lot of machinery. Not built.
- **Time is linear in the slots left on the CPU**, so the only thing worth
  maximising is resident fraction.

Striping each layer's experts across all four dies reaches 91% residency but
costs ~3.3 dispatches per layer instead of 1. That extra 7pp is worth ~7% of
expert time = **1.4% of decode**, which does not buy 3.3x the dispatches on
dispatch-bound hardware. Same arithmetic rejects per-expert placement. Revisit
only if a measurement says dispatch is cheap.

**Mechanism: mirror into VRAM, do not move.** The buffer stays host memory with
`is_host` true, and the layers we place get a second copy uploaded to a die.
Costs 137 GiB host + up to 115 GiB VRAM, which this machine has. The
alternative — making our buffer type device memory — cannot express "84% on GPU,
16% on CPU" at all, because llama.cpp allocates one buffer per buffer type, and
it would drop `is_host`, which is what keeps every unclaimed op free and the
correctness gate unchanged.

Per split we then either delegate to the CPU as today, or rebuild the received
graph against the die's tensors, upload the activations, compute, and read the
terminals back. Rebuilding generically from the cgraph — not hard-coding the
five ops — for the reason `tape` was built that way.

**Host build requirement, and it is not optional.** `dies` needs llama.cpp built
with `GGML_VULKAN=ON` (`llama.cpp/build-vk`), and **every run must pass
`-nopo 1`**. Otherwise `op_offload` hands host matmuls to the Vulkan device at
batch >= 32, they bounce off the two ops Vulkan cannot run (`DSV4_HC_COMB` every
layer, `LIGHTNING_INDEXER` on ratio-4 layers), and prefill fragments into ~1780
splits — 5.37 t/s against 17.99 (repo `CLAUDE.md`). `gate.py` and `bench.py`
take `--build-dir` so both hosts are reachable, and the CPU-only host stays the
baseline.

Increments, in order:

1. **Done.** Enumerate the dies, compute and log the placement, change nothing
   else — a placement that is only printed must not move a logit, and the gate
   says it did not. All three branches exercised on the stub by shrinking usable
   VRAM with `MOESERV_RESERVE_MB` rather than building bigger models.
2. **Done.** Upload the mirror; still compute on the CPU. **12.75 GiB at
   3.2-3.5 GiB/s**, so the real model's 115 GiB costs ~35 s of load, once.
   Freed and reset when the model buffer goes, so a second load re-probes.
   Untested: the fallback that sends a die's layers back to the CPU when
   allocation fails. It cannot be reached by shrinking the reserve — that
   changes the *plan*, not the outcome of allocating it — and needs a driver
   that reports more free VRAM than it will hand out.
3. **Done.** Compute placed layers on their die (`src/moe_run_vk.h`): rebuild
   the received split against the mirror, upload the inputs, compute, read the
   terminals back. Rebuilt generically from the cgraph — nothing in that file
   names an op. Measured against the same-placement control on the 4-layer stub:

   | configuration | mean KLD | max KLD | top-1 |
   |---|---|---|---|
   | repack gap, CPU vs CPU_REPACK (the yardstick) | 3.6e-5 | 2.1e-3 | 99.804% |
   | 4 layers on Vulkan0 | 8.4e-5 | 1.13e-2 | 99.804% |
   | 4 layers, one per die | 8.4e-5 | 1.13e-2 | 99.804% |
   | 1 layer on Vulkan0, 3 on CPU | 8.1e-5 | 1.11e-2 | 99.216% |

   **Vulkan is 2.3x the repack gap on the mean and 5.4x on the max — the same
   order**, which is the criterion: the repack gap is two *correct* kernels
   disagreeing on this machine and model, so a difference of that size is
   evidence of different rounding and nothing else.

   The four-dies row is identical to the one-die row to every printed digit, so
   the dies agree with each other exactly. That is the same-arithmetic control
   `../nano-glm` had to learn to run: without it, a difference between dies would
   have been charged to Vulkan.

   **Necessary, not sufficient.** End-to-end KLD saturates on a deep model and a
   mathematically-invariant wrong graph sits inside the precision band (repo
   `CLAUDE.md`). Four layers is chosen partly because it amplifies less than 43;
   whether that is enough is still the open question below.

4. **Done, and it found the wall.** `bench.py --build-dir build-vk --ngl 0`, the
   4-layer stub, ours-off (CPU) against ours-on (dies):

   | batch | our compute |
   |---|---|
   | pp4 | **+15.6%** |
   | pp8 | **+21.9%** |
   | pp16 | **-60.2%** |
   | pp32 | -54.6% |
   | pp128 | -51.4% |
   | tg32 (decode) | -0.1% |

   **The cliff is `ggml_vk_use_mul_mat_vec_id`**: `src2->ne[1] <= 8`
   (`ggml-vulkan.cpp:10607`), where `src2` is the ids tensor and `ne[1]` is the
   token count. At 8 tokens or fewer the dies use the vector path and beat 16
   CPU cores by ~20%; at 9 or more they take the general path and lose half.
   This is `../nano-glm`'s 10.2x cliff, found independently and from the other
   side.

   **Decode gains nothing** (-0.1%), which is the more sobering number: batch 1
   is firmly on the fast path, so the die is simply no faster than the CPU for
   one token's experts. At ~1.45 ms per layer either way, the dies are bound by
   dispatch and transfer, not by arithmetic — exactly what "one die is as fast
   as four" predicted.

   So `shaders`/chunking is not a later refinement, it is the next step: split a
   large `mul_mat_id` into dispatches of at most 8 tokens and the prefill loss
   should become a prefill win. That is a graph decision, not a shader.

5. Then the real model, which is only worth loading once a configuration wins on
   the stub — 7-9% load-to-load spread there cannot see a 20% effect reliably,
   let alone confirm one.

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
if it is the latter, the thing to reach for is `git show 9fe5558`, not a new
format.

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
