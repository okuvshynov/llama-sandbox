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

**Where it stands:** the block is claimed and computed two ways — mirrored onto
four Vega II dies with ggml's kernels (**+21% of prefill**), or decode-only
tensor-parallel across all four dies with a custom mxfp4 kernel (`MOESERV_TP=1`,
**+7.6% of decode vs stock** on the full model, the block itself 3.1x faster
where resident) — both verified against llama.cpp's CPU on the same weights.

**Optimising the block is the right thing even where this machine cannot show
it.** A decode step here reads 6 of 256 experts and all of the trunk, so even a
free block caps at ~1.12x under TP. That is this machine's balance, not the
approach's: the trunk's weights are 13.7 GiB and would fit on one modern GPU,
where they cost a fraction of the ~230 ms/token they cost on these cores. In
that configuration the expert block is most of the time — and it currently runs
at ~440 µs/layer of which only ~113 µs is GPU arithmetic. The border is ~3x the
compute. It was measured structural against our own call shapes; a null-shader
baseline (`../vk-latency/`, 2026-08-14) put the *machine* floor at 9 µs/submit
and 87 µs for a full 4-die round, and its TP-shaped ladder then rebuilt our
call ingredient by ingredient and **acquitted the command buffer**: submit
stays ~9-11 µs through descriptors, copies, big buffer references, 26 GiB
residency and 16 spinning threads. The residual — ~3x on submit, ~100 µs of
wait — lives in the moe-serv *process* (mapped model, ggml graph around the
call), so the next probe is inside moe-serv, plus the standing lever of fewer
submissions per token. See `decode-kernel`, `tp-integrate`, `border`.

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

`bench.py`, and the measurement discipline everything after it relies on: a
run proves its own noise floor and deltas below it are not reported; the stub
is the decode instrument (the real model's floor has ranged 0.3-9% by day);
only a quantity that transfers may be extrapolated. Prefill is the exception —
it measures to ~1% on the real model.

### `dies` — Done (`f312726`..`6689bdc`)

The block on four Vega II dies with ggml's kernels, whole layers mirrored into
VRAM (9 per die, 36 of 43 on the real model). Prefill **+21.1%** (net +18.1%
vs stock) once the block is issued in 8-token chunks —
`ggml_vk_use_mul_mat_vec_id`'s vector path ends at 8 tokens and the general
path loses half. Decode stayed flat; that story continues at `decode`.

### `decode` — Done (single read-back)

Six per-terminal read-backs cost 530 µs; reading their common view-root once
cut that to 149 and turned stub decode from flat to **+6.9%**. Standing
lessons: quote medians (first Vulkan call = 306 ms of pipeline compile), and a
phase timer bills work to where it runs, not to what caused it. Superseded by
`decode-kernel` / `tp-integrate`.

### `decode-kernel` — Done (E1-E9, ledger with postmortems in `docs/KERNEL.md`)

f32 is bandwidth-bound at ~701 GB/s; stock quantized kernels are
instruction-bound. The custom 2-pass mxfp4 kernel (`shaders/`) runs the
block's matmuls 1.5-1.9x faster than ggml's, and **TP-within-expert** beats
expert-parallel — perfect balance, no cross-die reduction, host sums four
partials. Full block in the probe: 113 µs/die. q4_0 requantisation accuracy
unmeasured — parked with the branch question.

### `tp-integrate` — Done (`dbfed68`, `0ae4a45`, `3969696`)

The TP engine in the DLL behind `MOESERV_TP=1` (`src/moe_tp.h`): lazy
per-layer repack, pre-recorded command buffers, host sum, CPU fallback for
anything unparseable or over budget. Full model: **3.92 t/s, +7.6% vs stock,
+13.0% vs the ggml path** — 34 of 43 layers resident, the block 3.1x faster
where resident, MoE down from ~21% to ~11% of decode time. Gate on real
weights: 1.070e-4. The hard-won specifics (HOST_CACHED for any CPU-read
buffer, the `-b` pin that makes `--ubatch 1` real, phase-timer numbers) are
in the commit messages.

### `border` — Done (`36be37a`)

Threaded per-die submit refuted — `vkQueueSubmit` serializes in the driver at
~35 µs whichever thread issues it. Fence polling kept: ~+0.5% on the stub,
poll ahead in 6/6 interleaved pairs across both pair orders. The remaining
border is structural (serialized submit floor + launch latency) and shrinks
only with fewer submissions per token, which the scheduler's
one-layer-at-a-time contract forbids. Postmortem in the commit message.
**Amendment (2026-08-14):** "structural" held only relative to our own call —
`../vk-latency/` (null shader, no ggml) measured the machine floor at
9 µs/submit, ~59 µs submit→fence, 87 µs for a 4-die round, against our
~35 µs/submit and ~310 µs non-GPU wait. Its TP-shaped ladder then rebuilt our
cb ingredient by ingredient (descriptors, exact copies+barriers, 4 dispatches,
real 816 MiB references, 26 GiB ballast, 16 spinning threads) and acquitted
all of it: full-shape 4-die round 212 µs, +113 µs GPU ≈ 325 vs our 440, submit
~11 µs vs our ~31. The residual is process-specific, not Vulkan-work-specific;
next probe is inside moe-serv (`MOESERV_PROFILE` vs `-t`; audit what besides
`vkQueueSubmit` the submit phase timer covers). Calibrated timestamps
(`VK_EXT_calibrated_timestamps`, 0.04 µs cross-clock error) then split the
round trip: submit 9 / **launch 28-35** / GPU / **fence-signal 21** µs — the
result bytes sit in host-cached memory ~21 µs before a polled fence admits it.
Named lever for the TP path: have the cb write a sentinel word after the
result copy and poll that to read results, fence only for cb-reuse
(~20 µs/layer, unmeasured in moe-serv). Launch+signal overlap across dies, so
the 4-die round pays them ~once, not 4x.

### `breadth` — Planned, next up

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

- **`prefill`** — parked. Known: chunking to 8 tokens re-reads each expert's
  weights once per chunk (~34x at pp512); lyrae's gather-scatter (expert-major,
  8.7x at batch 2048) is the structural fix to evaluate.
- **`residency`** — parked (`032d4c5`): full TP residency is a capacity wall
  (43 x 816 MiB = 34.3 GiB per die against 32 GB of HBM). Budget raise ~+1.5%
  and mixed CPU/die layers ~+3% assessed and not taken — the whole prize is
  ≤ +5% and it is the one optimisation that does not transfer.
- **`wire`** — in-process or a separate process over a socket; informed now by
  a measured per-call cost.

## Links

- `README.md` — what this is, build, run, and how to reproduce the current
  state.
- `docs/MECHANISM.md` — how llama.cpp is persuaded, and the out-of-tree
  backend traps. `docs/MEASUREMENTS.md` — every number with its instrument
  and noise floor. `docs/KERNEL.md` — the kernel experiment ledger.
- `gate.py` / `bench.py` / `make_stub.py` — each carries its reasoning in its
  header, including the measurement discipline the benchmarks depend on.
- `../nano-glm/OPTIMIZATION.md` — where the dies' dispatch-bound behaviour, the
  8-pair cliff and the routing-skew result were first measured.
- Repo `CLAUDE.md` — the traps this project keeps meeting: repack, op_offload,
  load-to-load variance, and controls that test nothing.
