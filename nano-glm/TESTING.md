# Testing nano-glm

Everything here runs through `gate.py`. It exists because assembling a gate by
hand each time is how `results/corpus/` became unusable: those reference files
were produced the evening before the corrupt-shard repair, nothing in them
records that, and the page cache means it is not even knowable from the file
whether they read good weights. A reference that cannot state the configuration
it was made under is not a reference.

## Two different questions

They get conflated whenever a gate is ad hoc, and they want opposite responses
to a legitimate change:

| | fixed reference, byte comparison | independent implementation |
|---|---|---|
| asks | did *my change* alter the output? | is nano-glm still *correct*? |
| a failure means | you did something | the port is wrong |
| on an intended change | re-baseline it | never re-baseline it |
| tests | `smoke`, `aa`, `rpc` | `llamacpp` |

## The tests

```bash
python gate.py                    # smoke — the default
python gate.py aa
python gate.py rpc llamacpp       # several at once
python gate.py all
```

| name | needs | cost | establishes |
|---|---|---|---|
| `smoke` | model | ~2 min | one 14-token prompt, 18 positions, byte-identical to the golden |
| `aa` | model | ~15 min | the whole 6-prompt set, byte-identical |
| `rpc` | model + `moe-server` | ~20 min | the set through the backend equals the local path, byte for byte |
| `llamacpp` | model + `rescore` | ~40 min | llama.cpp re-derives the set at KL == 0 — and **creates** the golden |

Three setups must agree: **A** llama.cpp, **B** nano-glm local, **C** nano-glm
over RPC. Byte equality is transitive, so two edges suffice and B is the hub
because it is cheapest to reproduce. `llamacpp` is A↔B, `rpc` is B↔C, and A↔C
follows without ever being run.

### Restricting to one prompt

```bash
python gate.py rpc --only smoke
```

Six prompts is twenty minutes; one is two. Use it while iterating. It refuses
to combine with `--update-golden` — a partial golden set would pair a fresh
provenance record with stale files it does not describe, which is the exact
defect this harness exists to prevent.

## Which test after which change

| you changed | run | why |
|---|---|---|
| trunk graph (attention, rope, norms) | `llamacpp` | only an independent implementation can catch a wrong port |
| `moe_block.h` | `llamacpp` | B and C share the header, so they cannot disagree with each other |
| protocol, server, client seam | `rpc` | the trunk is untouched; A↔B still holds from the golden |
| comments, asserts on untaken paths | `smoke` | cheap proof of no effect |
| ggml bump, compiler, thread count, machine | `--update-golden`, then `rpc` | the golden is invalid; re-derive it before anything else |
| GPU expert path | `vk_check.py` **and** `rpc` | see below: bytes still apply with no GPU, and only KL applies with one |

## The GPU expert path: `vk_check.py`

Byte identity ends when experts run on a GPU, so this lives outside `gate.py`
rather than inside it — mixing "are the bytes the same" with "is the difference
smaller than the noise" would blur the one thing `gate.py` is good at. A strict
client *refuses* a Vulkan backend outright (`vulkan` is in `NANO_REPRO_KEYS`),
and that refusal is the signal to come here.

```bash
python vk_check.py --gpu-experts 12 --only smoke     # floors, then three servers
python vk_check.py --floor-only                      # just the floors
```

It runs three servers against one CPU client build, and the middle one is the
point:

| run | what it isolates |
|---|---|
| CPU, no split | the reference — byte-identical to the golden |
| CPU, split onto a **second CPU device** | compaction alone: same arithmetic, so any difference is the partition and scatter |
| CPU, split onto the GPU | compaction + GPU |

**Do not drop the CPU control.** Measured at k=2, compaction alone accounted
for most of what naively looked like GPU error; without the control the whole
of it would have been filed against the driver.

Every configuration scores the **golden's full token ids** with `-n 0`, rather
than being given a prompt and left to generate. That is not a detail. The first
version did the latter, and four of eighteen comparisons then produced no
number at all, because the sequences had diverged and `compare.py` refused
them — including one where *both sides ran on the CPU* and only the partition
differed. A reassociation flipped one greedily-sampled token and the
continuations parted. **Free generation cannot measure numerical divergence in
a system where numerical divergence changes what is generated.**

Two floors run first, and they answer different questions. *Determinism* — same
server, same batch, twice — asks whether the GPU repeats itself at all; it
measured exactly 0 across 315 positions. *Shape* — prefill in one chunk versus
several — is the bar any CPU-vs-GPU figure has to clear. The split batch must
be **smaller than the prompt**: a first attempt compared `-b 512` with `-b 16`
on a 14-token prompt, which prefills in one chunk either way, and reported a
flat 0.0 that read as a flawless GPU and was two identical runs.

**Read `max`, not `mean`.** One mis-routed token vanishes in an average, and
`compare.py`'s mean can go negative — a top-128 truncation artifact, not a real
divergence.

**Known limit of this test.** Raising the GPU's share of the work 5.5x did not
move any figure, and the floor — no GPU involved — sits in the same band. The
measurement is saturated: 75 layers amplify any perturbation to the same
ceiling. So `vk_check.py` establishes *deterministic and plausible*, **not
correct**. For that, use the compare-mode below.

## `moe-server --compare`: the measurement that is not saturated

```bash
moe-server -m <model> --gpu-experts 12 --compare                  # or --cpu-experts 12
moe-server -m <model> --gpu-experts 52 --gpu-devices 4 --compare  # N dies, one thread each
```

Evaluates every layer on **both** the full CPU path and the split path, hands
the trunk the **CPU** answer, and prints per-layer error at disconnect. Roughly
2x slower.

Returning the CPU answer is the point, not a safety measure: every layer then
receives identical input, so each layer's number is its own. Feed the split
result forward instead and layer i+1's "error" includes everything layers 0..i
did — which is precisely the compounding that makes end-to-end KL saturate.

How much that matters, on `smoke`, per-layer relative RMS:

| path | median | max | end-to-end KL said |
|---|---|---|---|
| compaction (CPU→CPU) | 2.06e-08 | 7.64e-08 | 4.4e-2 |
| GPU | 1.82e-03 | 1.54e-02 | 5.8e-2 |

End-to-end called them indistinguishable. Per-layer separates them by five
orders of magnitude, and shows the compaction is exact to f32 reassociation
while the GPU has a real ~1e-3 difference.

Two sanity checks it also gives you for free: layers where the router never
picked a resident expert read *exactly* 0, and the worst layers are the ones
the split device did most work in. If either stops holding, distrust the run
before distrusting the driver.

### Run it for every new architecture, on a split configuration

Not optional, and this is the case that proves it. `run_device_compact` — the
path a device takes when it holds only *some* experts — is deliberately
arch-neutral: it moves weights and calls `mul_mat_id`, and knows nothing about
which model it is serving. That is the right design and it has one failure mode,
which deepseek4 hit: **an op the architecture needs and the neutral path does
not have.** deepseek4 clamps its SwiGLU at ±10.0; `run_device_compact` built
`mul_mat_id -> swiglu_split` with nothing between, which is correct for glm-dsa
and wrong here.

Nothing caught it for four commits. `gate.py rpc` starts a server with no
`--gpu-experts` or `--cpu-experts`, so every expert lives on device 0 and takes
the *other* path — the one that calls `build_ds4_moe_experts` and does clamp.
**The gate's server configuration never reaches the code the gate exists to
check.** Only a split configuration does, and only `--compare` looks there.

So: a new architecture is not ported until `--compare` has been run against a
split configuration. Use `--cpu-experts` rather than `--gpu-experts` for it —
identical arithmetic on both sides means any residual is the graph.

The residual is not zero even when correct: a token whose experts land on two
devices has its contributions summed by the host scatter-add instead of ggml's
pairwise adds, so summation order differs and the result drifts at rounding
level. What a missing op looks like, measured on this one with only 12.67% of
pairs on the split device:

| `--cpu-experts 32 --compare`, `01_prose` | worst layer max abs | overall rel RMS |
|---|---|---|
| clamp missing | 2.241e+02 | 5.598e-02 |
| clamp present | 1.221e-04 | 1.873e-08 |

Six orders of magnitude on both, so a single run at any point would have found
it. Read the **max** by preference — it is the more sensitive of the two when a
fault is rare — but do not wait for a subtle signal: an omitted op in this path
is not subtle.

## `moe-bench`: the expert kernel, without the model

```bash
build-vk\bin\moe-bench.exe --list
build-vk\bin\moe-bench.exe --experts 60 --pairs 2 --iters 100     # decode
build-vk\bin\moe-bench.exe --experts 60 --pairs 166               # 111-token prefill
```

`moe-server` needs ~3 minutes to load 150 GiB and fill four dies before it can
be timed, which is the wrong edit cycle for a matmul. This allocates weights of
the right shape and MXFP4 type, fills them synthetically, and runs the same five
ops `run_device_compact` builds. Seconds, from nothing.

`--pairs` is the (token, expert) pair count *one die* receives, because the
server dispatches to its dies in parallel and a layer's wall time is the slowest
one: 1-2 at decode, ~166 for a 111-token prompt over four dies.

**The graph is copied, not shared.** If `run_device_compact` changes, this must
change with it or it silently measures something else.

## `moe_log_stats.py`: where an RPC step's time goes

```bash
nano-glm.exe ... --moe-addr 127.0.0.1:5711 --moe-log results\rpc.jsonl
python moe_log_stats.py results\rpc.jsonl
```

Splits the per-call records by `n_tokens` — prefill calls carry the whole
prompt, decode calls carry one row, and a mean over both describes neither — and
reports the server's four stages plus `rtt` minus their sum, which is socket
time and scheduling.

## `split_study.py`: the whole forced-split ladder, unattended

```bash
python split_study.py --model <ds4.gguf>          # local, natural, then the ladder
python split_study.py --model <m> --ladder 1,1,1,1 0,0,0,0
```

Starts a `build-vk` server per rung, runs the CPU-tree client against it with one
discarded warm-up and two measured passes, and stops the server (waiting out the
unmap, which is minutes and otherwise locks the exe — see Troubleshooting). Every
pass is printed, never just the mean, and the summary carries a `tg spread`
column so a difference smaller than the pass-to-pass noise is visible as such.

The GLM-5.2 study this replaces was run by hand, which is why it was run once.

## `--force-split`: timing a distribution that residency cannot reach

```bash
moe-server -m <model> --gpu-experts 52 --gpu-devices 4 --force-split 2,2,2,2
```

**This computes the wrong answer on purpose. Never use it to produce logits.**

The problem it solves: `devices[0]` holds every expert and takes the complement
of whatever the GPUs take, so the CPU's share is pinned to `1 - residency` and
no measurement can separate "the dies are slow" from "the CPU still has 80% of
the work". Forced placement breaks that coupling — slot *s* of every token goes
where the pattern says regardless of routing, and the expert is remapped into
that device's resident range.

The work then has exactly the right shape and cost — same matmul dimensions,
same transfers, same pairs per device — and only the weights are wrong. That is
precisely enough to time a distribution, and not nearly enough to trust an
output. `--compare` refuses to run alongside it, because there is nothing
meaningful to compare a deliberately wrong answer against.

With 8 experts used per token, the interesting patterns are:

| pattern   | meaning                                        |
|-----------|------------------------------------------------|
| `0,0,0,0` | devices exist but do nothing — isolates the machinery's overhead |
| `8,0,0,0` | one die does everything, CPU idle              |
| `4,4,0,0` | two dies                                       |
| `2,2,2,2` | four dies, CPU idle — the parallelism ceiling  |
| `2,2,2,1` | seven slots on dies, **one** left on the CPU   |

The last two are the pair worth running. The gap between them is the price of
any CPU involvement at all, and it is what decides whether partial offload is
worth pursuing or whether the DRAM stream dominates regardless.

## Provenance, and what a refusal means

`testdata/provenance.json` records what the golden set was made with: compiler,
ggml commit, build flags, thread count, host, and every model shard's size and
mtime. The gate compares that **before** comparing bytes.

Refusal fields — these change the bytes for reasons that have nothing to do
with your change, so comparing across them measures the configuration:

    compiler   ggml_commit   blas   llamafile   n_threads   model shard size/mtime

`vulkan` is a refusal field on the **client's handshake** (`NANO_REPRO_KEYS`)
but not in `gate.py`'s golden comparison: the client is always CPU-only, so it
does not change the client's numerics — it means the *backend* cannot be
byte-compared, and a strict client should say so rather than fail obscurely.

Deliberately *not* refusal fields:

- **`git_rev`** — running at a different revision is the entire point.
- **`trace`** — a `-DNANO_EXPERT_TRACE` build was measured byte-identical to a
  plain one, so it only warns.
- **`host`, OS, core counts** — informational.

A refusal looks like this and exits 2:

```
REFUSED: the golden set was made under a different configuration,
so comparing against it would measure that, not your change.
  - n_threads: golden 16, now 32
```

Options, in order of preference: fix the configuration (usually `-t`); re-derive
the golden if the change is intended and permanent; or `--allow-drift` to
downgrade refusals to warnings when you are deliberately investigating.

The model check is size and mtime, not a hash — 583 GiB per gate run is not
affordable, and sampling a few MiB would have missed the ~1.9 KB of real
corruption we hit. It does not prove the bytes are good; `checksums/` does that,
occasionally and deliberately. What it proves is "these are the files the
reference was made with", which is the failure that actually bit us: the repair
replaced two shards, so a stale reference now announces itself.

## Re-baselining

```bash
python gate.py --update-golden
```

Runs `llamacpp` over every prompt and writes `testdata/` only if all of it
passes — the outputs are staged and copied at the end, so a divergence found
late cannot leave a half-replaced golden set with no provenance beside it.

Commit `testdata/` with the change that made it necessary, and say in the
message *why* the golden moved. A golden set that changes without an
explanation is indistinguishable from a regression that was papered over.

## Against another machine

```bash
python gate.py rpc --moe-addr 10.0.0.2:5711
```

With `--moe-addr` the gate uses an already-running backend instead of spawning
one locally. This is PLAN.md step 2 in a single flag; what remains for that
step is the network measurement, not the correctness argument.

Verify the remote model from disk against `checksums/GLM-5.2-UD-Q6_K.sha256`
before trusting anything it computes.

## What the handshake checks

`moe-server` and the client exchange fingerprints on connect (protocol v2).
Three tiers:

- **always fatal** — arch, `n_embd`, `n_layer`, `n_dense_lead`, `n_expert`,
  `n_expert_used`, `n_ff_exp`, `expert_scale`, `expert_norm`. The client's graph
  assumes these; per-request validation only covers `n_embd` and the layer
  range, which two entirely different models can share. Without this, pointing
  `--moe-addr` at the wrong backend gives fluent, confident, wrong output.
- **fatal under `--strict`** — compiler, ggml commit, blas, llamafile,
  n_threads, model name/bytes/shards. Valid to run, but bit-exactness is void.
  Not fatal by default because Q4_K experts behind a Q6_K trunk is a *planned*
  configuration (PLAN.md step 3), not a mistake.
- **informational** — printed either way.

The gate's `rpc` test always passes `--strict`. The client defaults to lenient
on purpose; the guarantee that a gate run is never lenient belongs in the
script, not in the binary's default.

## The tokenizer, separately

```bash
python tokenizer_check.py            # 28 cases, ~1 min
python tokenizer_check.py --verbose  # show the ids around a disagreement
```

Compares `nano-chat --raw --dry-run` against `llama-tokenize --ids --no-bos
--no-parse-special` over the corpus plus cases chosen to reach the parts of the
pre-tokenizer regex English prose never does: CJK, Japanese, Korean, Cyrillic,
Greek, astral-plane emoji, combining marks, digit-run chunking, and the two
whitespace rules that need backtracking.

Deliberately **not** part of the logits gate. A tokenizer bug that fed the gate
would present as a numerics failure and send you looking at the graph; keeping
the two apart means a broken tokenizer says "tokenizer".

Needs a target llama.cpp does not build by default:

```bash
cmake --build <llama.cpp>/build --target llama-tokenize
```

Current: 28/28 cases, 864 tokens, 0 differences. It has already earned its
keep once — it caught that Windows was mangling non-ASCII `argv` through
cp1252 before the tokenizer ever ran (repo `CLAUDE.md`). The tell was that
every ASCII case passed and every non-ASCII one failed.

## Not covered here

- **Unit-level invariants** — Hadamard orthonormality, wire-header layout,
  hparam parsing, the routing statistics. These want a C++ test target, not a
  flag on the shipping binary: PLAN.md step 10.
- **`nano-chat`'s generation.** Only its tokenizer is checked. The decode path
  it shares with `nano-glm` is covered by the gate above; the chat template is
  eyeballed against the GGUF's Jinja and has no automated check.
- **Performance.** Nothing here is a benchmark, and gate timings are not
  comparable between runs: each test cold-starts through 583 GiB, and a single
  mmap-backed Windows timing is not worth reading into (repo `CLAUDE.md`).
- **A port under construction.** `gate.py` needs a complete graph and a golden
  set; a half-written one has neither, and its logits would saturate anyway
  (`OPTIMIZATION.md`). While a model is being ported the check is per tensor:
  `logit-kld`'s `dump` captures llama.cpp's intermediates and
  `dump_inspect.py` compares them, driven by a throwaway harness
  (`apps/ds4-port` for deepseek4). That is a scaffold, not a test — it goes
  away when `gate.py llamacpp` can build a golden set for the model.

## Troubleshooting

**`LINK : fatal error LNK1104: cannot open file 'bin\moe-server.exe'`** — a
server from an earlier run is still alive. A process holding 583 GiB of views
takes minutes to unmap, and `Stop-Process` returns long before it is gone.
`Get-Process moe-server`, then `Wait-Process -Id <id> -Timeout 600`. The gate
does this for servers it spawned and reports how long it took.

**`Cannot open include file: 'winsock2.h'`** — building outside the MSVC
environment. Go through `build.ps1`, which wraps configure and build in one
`cmd /c` under `vcvars64`.

**Everything differs, and the diff looks like noise** — check the run
configuration before the code. Thread count, batch shape and toolchain all move
logits; `CLAUDE.md` has the measurements. The gate refuses precisely to stop
this from being a debugging session.

**Only the expert matmuls differ, by ~1e-6** — llama.cpp repacked the weights.
`GGML_CPU_REPACK` is ON by default and rewrites some quant types into a blocked
layout with its own GEMM; nano-glm mmaps weights as they sit in the file and
cannot match one. Whether it engages depends on the quant type *and* the ISA,
so a model that has always agreed proves nothing about the next one. Check with
`grep "repack: repack tensor"` on the reference log; `logit-kld/CMakeLists.txt`
forces it off and says why.

## Two golden sets

A golden set belongs to one model on one machine, so each architecture has its
own directory and `--testdata` selects it. Everything else about the gate is
shared.

    python gate.py aa                                   # GLM-5.2, testdata/
    python gate.py aa --testdata testdata-deepseek4                       --model D:\llms\ds-v4-flash\...-00001-of-00005.gguf

The corpus is the same five prompt texts plus a smoke prefix, re-tokenized per
model: the texts are shared, the ids are not, and the bit-exactness contract is
defined over ids. `dump --tokenize` produces them, and reads the prompt from a
**file** — Windows builds `argv` through the ANSI code page and the French
prompt does not survive it (`CLAUDE.md`).

## Current status

**GLM-5.2 UD-Q6_K** (14 shards, 582.88 GiB), `testdata/`: 6 prompts, 761
positions, every one verified at KL == 0 against llama.cpp `rescore --sim-gen`.
`rpc` passes byte-for-byte against it, so all three setups agree.

**DeepSeek-V4-Flash UD-Q8_K_XL** (5 shards, 150.7 GiB),
`testdata-deepseek4/`: 6 prompts, 793 positions, all at KL == 0, and `rpc`
passes byte-for-byte as well — so all three setups agree for this model too.
Its hash-routed layers (0-2) stay on the client and the server is told so by
`moe_shape.n_dense_lead`, which is the same mechanism glm-dsa's dense lead
uses.

Both on Windows / MSVC 19.50 / ggml `6a32c29a7` / 16 threads.

macOS has no golden set yet — it needs its own, built by the same command, for
the reason in `CLAUDE.md`: a different toolchain is a different set of numbers.
