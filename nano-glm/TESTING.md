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

## Provenance, and what a refusal means

`testdata/provenance.json` records what the golden set was made with: compiler,
ggml commit, build flags, thread count, host, and every model shard's size and
mtime. The gate compares that **before** comparing bytes.

Refusal fields — these change the bytes for reasons that have nothing to do
with your change, so comparing across them measures the configuration:

    compiler   ggml_commit   blas   llamafile   n_threads   model shard size/mtime

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

## Not covered here

- **Unit-level invariants** — Hadamard orthonormality, wire-header layout,
  hparam parsing, the routing statistics. These want a C++ test target, not a
  flag on the shipping binary: PLAN.md step 10.
- **Tokenizer agreement with llama.cpp** — arrives with the tokenizer
  (PLAN.md step 7), and stays out of the logits gate so a tokenizer bug cannot
  present as a numerics failure.
- **Performance.** Nothing here is a benchmark, and gate timings are not
  comparable between runs: each test cold-starts through 583 GiB, and a single
  mmap-backed Windows timing is not worth reading into (repo `CLAUDE.md`).

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

## Current status

Golden set: 6 prompts, 761 positions, every one verified at KL == 0 against
llama.cpp `rescore --sim-gen`. `rpc` passes byte-for-byte against it, so all
three setups agree. Windows / MSVC 19.50 / ggml `6a32c29a7` / 16 threads /
GLM-5.2 UD-Q6_K, 14 shards, 582.88 GiB.

macOS has no golden set yet — it needs its own, built by the same command, for
the reason in `CLAUDE.md`: a different toolchain is a different set of numbers.
