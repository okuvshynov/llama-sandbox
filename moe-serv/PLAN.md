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
at 439 µs/layer of which only ~113 µs is GPU arithmetic, so the border is still
~3x the compute and that is the part worth having. See `decode-kernel` and
`tp-integrate`.

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
exception — see `dies`. (The `tp-integrate` run then measured 0.3-1.3%
load-to-load on the same model, so the floor is not a constant of the machine;
the rule stays — a run proves its own noise floor, and deltas below it are not
reported.)

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

### `decode` — Done (single read-back)

The per-split profiler (`MOESERV_PROFILE=<prefix>`, `src/moe_prof.h`) showed
the die winning the arithmetic (1018 vs 1420 µs/layer) and losing it back to
six 88 µs read-backs; reading the terminals' common view-root once cut
read-back 530 -> 149 µs and turned stub decode from flat to **+6.9%**. Two
standing lessons: the first Vulkan call costs 306 ms of pipeline compile, so
quote medians; and a phase timer bills work to where it runs, not to what
caused it. Superseded by `decode-kernel` / `tp-integrate`.

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

### `decode-kernel` — Done (branch `moe-q40-experiment`, ledger in `KERNEL.md`)

Nine kernel experiments, one commit each, postmortems in `KERNEL.md`. What
survived: **f32 is bandwidth-bound at ~701 GB/s** (the practical ceiling for
this gather; lyrae's dense Metal kernels on the same dies: 720-791) and every
stock quantized kernel is **instruction-bound** at 16-23 lane-cycles/weight; a
custom **2-pass K-tiled mxfp4 kernel** (two-plane repack, 16-float LDS LUT, 4
cols/thread — `shaders/mxfp4_pass1.comp`) runs the block's matmuls at
95.4/86.5 µs against ggml's 163.8; and **TP-within-expert** — each die holds a
column slice of gate+up and the matching down k-rows for *all* experts — beats
expert-parallel: perfect balance, no cross-die reduction, the host sums four
partial vectors. Full block on 4 dies in the probe: 113 µs/die, max abs 4.3e-4
against a ggml f32 block reference (`--tp` in `src/moe_probe.cpp`). The q4_0
requantisation accuracy is still unmeasured — parked with the branch question.

### `tp-integrate` — Done (`dbfed68`, `0ae4a45`, this commit)

The TP engine lives in the DLL behind `MOESERV_TP=1` (`src/moe_tp.h`): its own
VkInstance beside the host's, lazy per-layer repack (~816 MB/die/layer, paid in
warm-up), one pre-recorded command buffer per (layer, die), host sum into the
MUL node. Anything unparseable or over budget falls back to the CPU delegate,
per the invariant. Gate on real weights: mean KLD 1.070e-4 (tol 5e-4), 4096
splits, 0 fallbacks — via `gate.py --tp --ubatch 1`, where pinning `-b` to the
ctx is load-bearing (`n_seq_max = 4` otherwise hands every call 4 tokens and
the gate passes while testing nothing). Two border fixes found by phase
timers, worth ~2 ms/call together: persistent mapping, and **HOST_CACHED
memory for any buffer the CPU reads** — plain VISIBLE|COHERENT is
write-combined on AMD, ~150 MB/s to read (lyrae's storageModeShared, in
Vulkan spelling).

**Full model** (tg32, 2 loads/config, spreads 0.3-1.3%):

    stock 3.64    ours-off 3.48    ours-on 3.46    ours-tp 3.92
    TP vs ours-on +13.0%    TP vs stock +7.6%    repack forfeit -4.4%

34 of 43 layers resident under the 28000 MB/die budget, 9 on the CPU delegate
(5474/1449 splits — exactly the capacity arithmetic). Per resident layer:
**439 µs** (stage 4.5, submit 125.1, wait-first 225.7 — containing ~113 µs of
GPU — wait-rest+sum 83.9) against **1.38 ms** on the CPU delegate: the block
is 3.1x faster where resident, and MoE falls from ~21% of decode time to
~11%. Headroom left on this host is bounded — perfect residency ≈ +3%, MoE
free ≈ +12% — but the border is still ~3x the GPU arithmetic, and submit
batching / fence polling transfer to hardware where the trunk is fast.

### Later — one line each

- **`prefill`** — parked. Known: chunking to 8 tokens re-reads each expert's
  weights once per chunk (~34x at pp512); lyrae's gather-scatter (expert-major,
  8.7x at batch 2048) is the structural fix to evaluate.
- **`border`** — submit batching and fence polling; the per-call 439 µs is
  ~3x its GPU arithmetic, capped at a few percent end-to-end here.
- **`residency`** — parked, assessed 2026-08-13. Full residency is physically
  impossible: 43 layers x 816 MiB = 34.3 GiB per die against 32 GB of HBM, so
  9 layers on the CPU is a wall, not a budget choice. What exists, none of it
  taken: raising `MOESERV_TP_BUDGET_MB` toward the die's true allocatable
  ceiling (~31 GiB?) fits ~38 layers for ~+1.5% at zero code, unmeasured;
  mixed CPU/die layers — dies take a column share, the CPU computes the rest
  *between submit and fence-wait*, so the CPU slice overlaps the ~330 µs
  border — pencil out to ~+3% end-to-end for genuinely fiddly code (the down
  matrix needs a k-range slice of quantized data; 5-way host sum). The whole
  9-layer prize is 12.4 ms of a 257 ms token, ≤ +5%, and it is the one
  optimisation that does not transfer to a fast-trunk target, where border
  work does.
- **`wire`** — in-process or a separate process over a socket; informed now by
  a measured per-call cost.

## Links

- `README.md` — build, run, how the mechanism works, the numbers, and the things
  that are not obvious about writing an out-of-tree ggml backend.
- `gate.py` / `bench.py` / `make_stub.py` — each carries its reasoning in its
  header, including the measurement discipline the benchmarks depend on.
- `../nano-glm/OPTIMIZATION.md` — where the dies' dispatch-bound behaviour, the
  8-pair cliff and the routing-skew result were first measured.
- Repo `CLAUDE.md` — the traps this project keeps meeting: repack, op_offload,
  load-to-load variance, and controls that test nothing.
