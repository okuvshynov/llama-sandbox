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

**Where it stands:** the block is claimed, mirrored onto four Vega II dies and
computed there, bit-exact against the CPU on the same weights, and worth **+21%
of prefill** on DeepSeek-V4-Flash. Decode is unchanged and, for the reason in
`dies` below, will stay that way without a different idea.

## Invariants

- **No llama.cpp source changes, ever.** If something appears to need one, that
  is a finding to write down, not a patch to carry.
- **Two models, named**: `glm-dsa` (GLM-5.2, UD-Q6_K) and `deepseek4`
  (DeepSeek-V4-Flash, UD-Q8_K_XL). "Supported" means a passing correctness
  contract against a real checkpoint. The expert block is among the least
  model-specific parts of a MoE transformer, so a third model probably mostly
  works — do not say so until one has been run. **Only deepseek4 has ever been
  run**; see `breadth`.
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
- **Every fall-back is a fall-back to the CPU, never a skip.** A layer that does
  not fit, an op a die declines, a shape the chunker cannot read: all of them
  must be slow and right.
- **No dependency on `../nano-glm`, in either direction.** Copy what is worth
  copying, as source that then belongs here.

## Correctness contract

**`gate.py` against a stub model.** Two runs of stock `llama-perplexity`, both
placing the routed experts in our buffer with `-ot`, differing only in whether
we claim any ops (`MOESERV_DISABLE=1` is the control). Each writes its
log-probabilities with `--kl-divergence-base`; those files are a deterministic
function of the logits, so identical files mean our compute changed nothing.

The harness is llama.cpp. Nothing here defines the graph, the ops or the
arithmetic a second time, because a wrong test that passes is worse than no
test. `make_stub.py` is what makes it affordable — four layers of
DeepSeek-V4-Flash load in seconds against 150 GiB in minutes.

- **CPU path: `--tol 0`.** Bit-identical, and it does not retire now that Vulkan
  works — it is the regression test on the plumbing.
- **Vulkan path: `--tol 5e-4`**, typed on the command line every time rather
  than defaulted. That is ~14x the **measured repack gap** (mean KLD 3.6e-5,
  max 2.1e-3), which is what two *correct* kernels disagreeing looks like on this
  machine; the Vulkan path measures 6.2e-5. Necessary, not sufficient:
  end-to-end KLD saturates on deep models and a mathematically-invariant wrong
  graph sits inside the precision band (repo `CLAUDE.md`).

**The control is not stock llama.cpp.** llama.cpp overrides `-ot exps=CPU` to
`CPU_REPACK` and multiplies MXFP4 experts by a different GEMM, which we cannot
do. `--vs-stock` measures that gap; it is not the gate.

`test-backend-ops` does **not** work here and must never be quoted as
conformance — it calls `supports_op` before allocating anything, so our
ownership guard answers no and it reports OK over 0/0 tests.

Every check asserts **where the weights went and whether we computed**, from
each run's own log, and aborts rather than comparing when it cannot. Five
"passes" in this project have tested nothing or the wrong thing.

## Steps

Named, not numbered: `../nano-glm/PLAN.md` numbered its steps and reality
reordered them, leaving code comments pointing at "step 9".

### `handshake` — Done (`3b228f2`)

Registers, exposes a buffer type named `MoE` that `-ot` resolves, claims no ops.
Surprise worth carrying: `-ot` is honoured **regardless of `supports_op`**, so
140352 MiB of experts landed in our buffer before a single op was claimed — the
run stayed correct only because the buffer is host memory, which makes `is_host`
load-bearing.

### `passthrough` — Done (`bd7abcc`)

Claims `MUL_MAT_ID`, `CLAMP`, `GLU`, `MUL` guarded by ownership; one split per
MoE layer, nothing outside it. Its "bit-identical" was measured on generated
*text*, which `gate` later showed to be the weaker claim.

### `gate` — Done (`0ea5520`)

`gate.py` + `make_stub.py`, the contract above. Established that our compute is
bit-identical to llama.cpp's on the same weights, that a **prefix** of layers is
a valid model needing no metadata surgery beyond `block_count`, and that the
repack gap is 3.6e-5 mean KLD — the yardstick every tolerance since is argued
from.

### `tape` — Removed (`9fe5558`, `c5ec045`)

A capture/replay format, superseded by `gate` before it was finished. `git show
9fe5558` if a per-op comparison is ever needed.

### `bench` — Done (`615d67c`)

`bench.py`. Established the measurement discipline that everything after it
relied on: **decode on the real model cannot resolve anything below ~10%**
(7-9% load-to-load against 2-6% effects), so the stub is the instrument and only
a quantity that transfers may be extrapolated. Prefill turned out to be the
exception — see `dies`.

### `dies` — Done (`f312726`, `11626d1`, `ebf83f6`, `57c5b85`, `b9e0cec`, `6689bdc`)

The expert block on four Vega II dies, with ggml's own kernels. Whole layers,
packed in device order, mirrored into VRAM; the rest on the CPU. On
DeepSeek-V4-Flash that is **9 layers per die, 36 of 43, 7 on the CPU**, exactly
as the capacity arithmetic predicted before it met the real model.

    pp512        stock 18.49    ours-off 18.02    ours-on 21.84
    our compute  +21.1%         net vs stock +18.1%        (spreads <= 1.1%)

Three findings that shape what comes next:

- **The cliff.** `ggml_vk_use_mul_mat_vec_id` takes the vector path only at
  <= 8 tokens (`ggml-vulkan.cpp:10607`). Unchunked, prefill *lost* 51%; issuing
  the block 8 tokens at a time wins 21-35%. A graph decision, not a shader —
  which is why `shaders` is now less urgent than it looked.
- **Decode cannot be fixed by chunking.** Batch 1 is already on the fast path,
  and a die is no quicker than 16 CPU cores for one token's experts (~1.45 ms
  per layer either way). The block is also only ~20% of the bytes a decode step
  reads, capping any decode win at 1.25x.
- **Prefill measures well on the real model** (0.0-1.1% load-to-load), unlike
  decode. The "stub is the instrument" rule is a decode rule.

### `decode` — Planned, next

Profiled with `MOESERV_PROFILE=<prefix>` (`src/moe_prof.h`), which writes one
CSV row per split with microsecond phase timings for both paths — `dev = -1` is
the CPU delegate, so the die and the 16 cores it replaced are in one file.
Decode, stub, one layer on a die and three on the CPU, **steady-state medians**:

| | CPU | vk0 |
|---|---|---|
| compute | 1420 | **1018** |
| read-back | — | **530** |
| build + alloc + upload + free | — | 19 |
| **per layer, µs** | **1420** | **1578** |

**The die wins the arithmetic and loses it back at the border.** Compute is 1.4x
faster; the read-back costs 530 µs and turns the layer into an 11% net loss,
which is why decode measured flat.

530 µs for 98 KB is 185 MB/s — not bandwidth, but **six separate
`ggml_backend_tensor_get` calls at ~88 µs each**, one per terminal, each
stalling the pipeline. Those six terminals are views that tile one parent
tensor exactly, so one contiguous read of the parent would do. Projected:
1125 µs per layer, **+26% against the CPU** — decode's first real win.

So the next step is: when a split's terminals are all views of one node, read
that node once instead. Generic (walk `view_src` to a root that is itself a node
here), and it must not change a bit — the six reads and the one read cover the
same bytes.

Watch out for two things the profile also showed. The **first call costs 306 ms**
(Vulkan pipeline compilation) and dragged the *mean* to 4730 µs against a median
of 1580 — quote medians here, or drop the warm-up. And `alloc` is 4 µs steady
against 2873 µs on the first call, so the kept allocator is doing its job.

### `breadth` — Planned, after that

Run GLM-5.2 (`glm-dsa`, UD-Q6_K, 583 GiB). The invariants name two models and
only one has ever been run, so every generalisation this project has made is
currently untested: a second architecture is the cheapest way to find which of
them were really deepseek4 in disguise.

Specifically at risk, in rough order of likelihood:

- **The chunker's token-dimension inference.** It matches each tensor's extents
  against the ids tensor's token count and refuses when two dimensions match.
  GLM-5.2 has a different `n_expert_used` and `n_ff_exp`, so both the match and
  the refusal need to be seen happening.
- **`supports_op`'s claim set.** glm-dsa gates with sigmoid rather than
  sqrt-softplus and has no SwiGLU clamp, so its block is a different op sequence
  — the guard should claim it or decline it cleanly, and "decline" must mean the
  CPU path, not a wrong answer.
- **`make_stub.py`'s prefix rule.** Whether GLM-5.2 has per-layer arrays that
  `get_key_or_arr` sizes to `block_count`, and whether a short prefix is a
  loadable model at all — deepseek4 needed four layers for reasons specific to
  its compressor kinds.
- **Placement arithmetic** at a different expert size and layer count.

Order: build a stub, get `gate.py --tol 0` green on the CPU path, then the
Vulkan path, then `bench.py` prefill. Only load the 583 GiB model once the stub
is green, and expect the load-to-load problem to be worse there, not better.

### Later — one line each

- **`residency`** — which experts sit on which die, once something makes the
  static whole-layer placement look wrong.
- **`wire`** — in-process or a separate process over a socket. Still undecided,
  and now informed by a measured per-split cost.
- **`shaders`** — custom kernels for these dies. Demoted: the generic ones win
  prefill once the graph keeps them on the vector path.

## Links

- `README.md` — build, run, how the mechanism works, the numbers, and the things
  that are not obvious about writing an out-of-tree ggml backend.
- `gate.py` / `bench.py` / `make_stub.py` — each carries its reasoning in its
  header, including the measurement discipline the benchmarks depend on.
- `../nano-glm/OPTIMIZATION.md` — where the dies' dispatch-bound behaviour, the
  8-pair cliff and the routing-skew result were first measured.
- Repo `CLAUDE.md` — the traps this project keeps meeting: repack, op_offload,
  load-to-load variance, and controls that test nothing.
