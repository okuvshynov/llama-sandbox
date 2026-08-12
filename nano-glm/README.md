# nano-glm — minimal CPU-only MoE inference on bare ggml

An experiment in going back to basics: llama.cpp grew from a single-file
inference engine into a framework (142 model architectures, a KV-cache zoo,
15+ accelerator backends, batch/memory abstractions). nano-glm asks how small
an engine gets if the compute layer is kept and the framework is dropped: it
links only `ggml` (kernels, GGUF reader, backend scheduler) and reimplements
the thin slice of llama.cpp that two large MoE models on CPU actually need —
shard loader, single-sequence KV cache, forward graph, greedy loop.

Two architectures, both bit-identical to llama.cpp:

| | | |
|---|---|---|
| **GLM-5.2** | `glm-dsa`, UD-Q6_K, 582.88 GiB | MLA, DSA lightning indexer |
| **DeepSeek-V4-Flash** | `deepseek4`, UD-Q8_K_XL, 150.7 GiB | hyper-connections, three KV compression ratios, hash-routed layers |

The name is now a historical accident; the second model is what turned the
tier boundaries in `lib/` from a guess into a measurement.

Laid out as a library and the apps that drive it:

```
lib/     the engine — GGUF store, per-architecture model/graph/cache, routed
         expert block, wire protocol, remote-MoE client, tokenizer,
         fingerprint, routing trace
apps/    nano-glm     the validation harness: token ids in, lkldtopk out
         nano-chat    single-turn chat: text in, streamed text out
         nano-bench   throughput in a named residency regime
         nano-probe   what a checkpoint contains and what it would cost to hold
         moe-server   the MoE backend: one activation in, one combined row out
         moe-bench    the backend's expert kernel with synthetic weights and no
                      model at all, so tuning it is an edit cycle of seconds
                      rather than the three minutes moe-server needs to load
         ds4-port     temporary: the deepseek4 porting harness, per-tensor
                      comparison against llama.cpp. Goes away when nothing
                      needs it.
```

The split is along the line the bit-exactness contract draws. `lib/` is
mechanism; `apps/` is policy — what to read, what to emit, when to stop. Client
and backend build the routed block from one definition, so they cannot drift
apart.

`lib/` is in three tiers: the fingerprint, wire protocol, RPC client, GGUF
store and routing trace know nothing about any model; `moe_shape.h` is the
dimensions a client and a backend must agree on; and everything genuinely
one-model lives under `models/<arch>/`. [lib/README.md](lib/README.md) has the
file-by-file map and scores the predictions this layout was built on — the
short version being that the "DeepSeek lineage" family tier was wishful (two
models in that lineage still share no arithmetic), the KV cache was correctly
predicted to be the thing a second model disturbs, and the ~500-line estimate
for a second model was low by 4x.

Correctness bar: **bit-identical logits** vs the llama.cpp-based `collect`
baseline from `../logit-kld`, verified over the same prompts with
`compare.py` / `cmp`. That is why the forward graph is an op-for-op port of
llama.cpp's glm-dsa trunk graph (`src/models/glm-dsa.cpp` plus the
`llm_graph_context` helpers inlined), with the same op order, the same
seemingly-redundant views, and the same runtime configuration the baselines
ran under:

- flash attention ON (f16 masks, F32 FA precision) — what `auto` resolved to;
- fused lightning indexer (`ggml_lightning_indexer`) ON;
- F16 K caches (MLA: 576/row; indexer: 128/row with the arch-forced 128×128
  orthonormal Walsh-Hadamard rotation), zero-initialized, n_kv padded to 256;
- BLAS (Accelerate) offload for batches ≥ 32 via `ggml_backend_sched` with
  the same [BLAS, CPU] backend priority as llama.cpp;
- llamafile sgemm ON, Metal OFF (CPU-only by construction — see repo
  CLAUDE.md for the AMD-Metal NaN story).

What is deliberately **not** here: tokenizer (raw token ids are the
interface, same policy as logit-kld — take them from an existing lkldtopk
file or pass a comma list), chat templates, samplers (greedy = stored top-1),
batching across sequences, KV shifting/defrag, the NextN/MTP draft head, and
every architecture but the two above.

Where this is heading: [PLAN.md](PLAN.md) — remote MoE evaluation. The routed
experts move behind a network service on a machine that can hold them, the
trunk runs wherever trunk work is fastest, and these two are the testbed for
Kimi-K3-scale models that fit nowhere else. That plan is about making it
*work*; anything whose purpose is speed lives in
[OPTIMIZATION.md](OPTIMIZATION.md), where the measurements say what each idea
would be worth on this one machine, model and workload — which is exactly why
none of them is on the critical path.

## Build

```bash
cmake -B build -DLLAMA_CPP_DIR=$HOME/projects/llama.cpp
cmake --build build -j
```

Only `${LLAMA_CPP_DIR}/ggml` is used; ggml stays unmodified, so upstream
kernel improvements keep flowing.

On Windows (MSVC), `build.ps1` wraps the same two commands — it locates
`vcvars64.bat` and VS's bundled ninja, which CMake cannot find on its own:

```powershell
.\build.ps1                       # nano-glm
.\build.ps1 -Project logit-kld    # the verification tools, same toolchain
.\build.ps1 -Vk                   # Vulkan moe-server only, into build-vk\
```

`-Vk` is a separate tree because a Vulkan-enabled build registers the GPUs, and
every trunk binary aborts when a GPU device is present
(`lib/models/glm_dsa/graph.h`).
Only `moe-server` can hold one, so that tree builds only `moe-server` and it is
paired with `build\bin\nano-glm.exe` as the client — the client then keeps
exactly the numerics the golden set was made with. `moe-server --devices` lists
what a build can see, without loading a model.

Executables land in `build\bin\` next to the ggml DLLs so Windows resolves
them without PATH juggling.

The script also passes `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, which has to be a
*cache* variable: llama.cpp sets it as a plain variable inside ggml's directory
scope, so without this `build/compile_commands.json` lists ggml's sources but
none of ours. Point an editor at that file (VS Code: a `.vscode/
c_cpp_properties.json` with `compileCommands` listing each subproject's copy —
gitignored, since the paths are absolute) and includes resolve.

## Run

```bash
# prompt token ids from an existing logit-kld collect file (its n_prompt tokens):
./build/nano-glm -m .../GLM-5.2-UD-Q6_K-00001-of-00014.gguf \
    -i ../logit-kld/results/glm-5.2-ud-q6_k.n256.bin -n 256 -o nano.bin

# or raw ids:
./build/nano-glm -m .../GLM-5.2-UD-Q6_K-00001-of-00014.gguf -T "151331,98328,3837" -n 32
```

Flags: `-o` output lkldtopk file, `-n` tokens to generate, `-k` top-K stored
(128), `-c` context (4096, auto-raised), `-b` prompt chunk size (512), `-t`
threads (all cores). Generated token ids stream to stdout (no tokenizer —
no text); summary to stderr.

Output is an `lkldtopk` v1 file (writer shared with logit-kld), so the whole
logit-kld toolchain applies: `inspect.py` for sanity, `compare.py` against
any collect/rescore file over the same token sequence.

## Chat

```bash
./build/bin/nano-chat -m <model> -p "why do cities grow near rivers?" -n 200
./build/bin/nano-chat -m <model> -p "..." --no-think     # skip the reasoning pass
./build/bin/nano-chat -m <model> -f prompt.txt --raw     # plain completion
./build/bin/nano-chat -m <model> -p "..." --dry-run      # tokenization only, ~1s
```

Byte-level BPE read from the GGUF, GLM-5.2's chat format applied as token ids,
greedy decoding, tokens streamed as they are produced. `--dry-run` reads only
shard 1 (9.4 MB of metadata) so you can see what the template built without
touching 583 GiB of weights.

The tokenizer agrees with llama.cpp on **28/28 cases, 864 tokens, exactly** —
CJK, Japanese, Korean, Cyrillic, Greek, astral-plane emoji, combining marks and
the whitespace rules included. Re-check with `python tokenizer_check.py`.

Why this is a separate binary from `nano-glm`: the bit-exactness contract is
defined over a *fixed token sequence*, so a tokenizer, a chat template or a
sampler — anything that can change which tokens get evaluated — has to sit
outside the tool that produces reference numbers, or every stored reference
quietly rots. `nano-chat --dry-run` prints ids that `nano-glm -T` accepts, which
is how an interesting generation becomes a reproducible logits test.

## Routing trace

Which experts the router picks is not observable from timings, and it decides
whether a GPU-resident expert subset or a prefetcher can pay (PLAN.md steps 3-4).
A separate build records it:

```powershell
.\build.ps1 -Trace     # -> build-trace\bin, cmake -DNANO_EXPERT_TRACE=ON
.\build-trace\bin\nano-glm.exe -m <model> -i <prompt.bin> -n 1024 \
    -o run.bin --expert-log run.trace
python expert_stats.py run.trace
```

Separate build tree, not a runtime flag: tracing keeps an intermediate tensor
alive so it survives to be read after `ggml_backend_graph_compute`, which
changes how the graph is allocated. That is a change the default build — the
one the bit-exactness gate runs against — should not carry. The two binaries
sit side by side so the claim can be checked rather than assumed; on Windows /
MSVC / 16 threads a traced and an untraced run of the same prompt are byte-
identical.

The trace is a text file, position-major, ~2.4 KB per position:

```
# n_layer=78 n_dense_lead=3 n_expert=256 n_expert_used=8 n_prompt=114
p 0 785                              <- position, token id
l 3 250,194,214,140,171,63,161,205   <- layer, expert ids in router rank order
```

`expert_stats.py` reports per-layer and aggregate skew (entropy, how many
experts carry 50%/90% of selections), static-residency hit rates measured
*out of sample*, and locality (consecutive-token overlap, distinct experts per
sliding window) — each against what independent uniform routing would give,
since a locality number means nothing without that baseline. `--null` re-runs
every metric on uniform draws of the same shape, which is how you tell a
finding from a sample-size artifact.

Two studies build on the same traces and need no further runs:

```bash
python residency_study.py results/residency/*.trace   # does a placement transfer?
python cache_sim.py       results/residency/*.trace   # LRU/LFU instead of static?
```

Findings in [ROUTING.md](ROUTING.md), and they cut against each other in a
useful way. Within one continuation routing is strongly concentrated — a 23%
resident subset catches 58% of selections. Across prompts that collapses to
28%, against 23% for picking at random. An LRU cache recovers it and more
(63%), but each of its misses costs a PCIe install on top of the DRAM read, so
it is 3.5x slower than a fixed placement despite hitting twice as often.

## Verification

```bash
python gate.py                  # smoke: one prompt, 18 positions, ~2 min
python gate.py aa               # the whole prompt set, bytes, ~15 min
python gate.py rpc              # the set through moe-server, ~20 min
python gate.py llamacpp         # re-derived by llama.cpp at KL == 0, ~40 min
```

Four named tests, a golden set in `testdata/`, and a provenance record the gate
checks *before* it compares bytes — so a mismatched compiler, thread count or
model is a refusal rather than a difference that looks like yours.

```bash
python vk_check.py --gpu-experts 12    # the GPU expert path, where bytes stop applying
```

Separate from `gate.py` on purpose: once experts run on a GPU the question
changes from "are the bytes identical" to "is the difference smaller than the
difference this pipeline has with itself", and those want different machinery.
It measures two floors first, and runs the same split onto a second **CPU**
device as a control — which is what shows how much of an apparent GPU error is
really just the change in summation order.

**[TESTING.md](TESTING.md)** has the full picture: which test to run after
which change, what a refusal means, how to re-baseline, how to point the same
test at another machine, and what `vk_check.py` can and cannot establish.

Current status: 6 prompts, 761 positions, KL == 0 against llama.cpp, and the
RPC path byte-identical to the local one.

### Comparing by hand

`gate.py` automates all of this; the manual form is here because it is what a
one-off investigation looks like, and because the reasons behind it are what
the gate's refusals are enforcing.

```bash
# A/A vs the llama.cpp baseline: same prompt ids, same batch shapes
./build/nano-glm -m <model> -i <baseline.bin> -n 256 -o nano.bin
python ../logit-kld/compare.py <baseline.bin> nano.bin   # expect KL ~ 0
cmp <baseline.bin> nano.bin                              # bit-identity modulo model_desc
```

The graph replicates llama.cpp's execution shape for the collect workload
(prompt prefilled in `-b` chunks, generation one token per batch), which the
logit-kld A/A study showed is the regime where CPU llama.cpp is bit-exact
across runs. Divergence beyond the `model_desc` header bytes means a port
bug, and per-position KL points at where.

**Bit-exactness is per-platform and per-thread-count.** The baseline must come
from a llama.cpp built with the *same toolchain* and run at the *same* `-t`:

- toolchain — llama.cpp disagrees with itself across macOS/Apple clang and
  Windows/MSVC by 8.85e-3 mean KL on identical hardware and the same commit
  (compiler FMA contraction, not ISA). A cross-platform comparison measures the
  compiler, not the port.
- threads — ggml partitions matmul work by thread count, so `-t 16` and `-t 32`
  give different logits; runs at a fixed count are bit-identical regardless of
  page-cache warmth. Default is physical cores, ignoring SMT siblings.

The portable way to produce a matching reference on any platform is to rescore
nano-glm's *own* output, which sidesteps greedy divergence entirely:

```bash
../logit-kld/build/bin/rescore -m <model> -i nano.bin --sim-gen -o ref.bin
python ../logit-kld/compare.py nano.bin ref.bin          # expect KL == 0 exactly
```

Verified on Windows (MSVC, 16 threads, GLM-5.2 UD-Q6_K): 270/270 top-1, KL
exactly 0.0, all 278,640 payload bytes identical — the same bar the macOS build
met. See ../logit-kld/README.md "What must match for a comparison to mean
anything".

## Performance

```bash
./build/bin/nano-bench -m <model> -i testdata/01_prose.bin --hot  -r 24
./build/bin/nano-bench -m <model> -i testdata/01_prose.bin --full -n 256 -r 4
./build/bin/nano-bench --pages --gb 40        # memory probe, no model needed
```

**Quote no throughput number without its residency regime.** Cold, hot-subset
and whole-model-warm differ by more than any optimisation here is worth, and
mixing them is what made every earlier figure unreliable. `nano-bench` prints
every repetition and takes its median from the back half, so warmup stays
visible instead of being averaged in.

Windows / MSVC / 16 threads / GLM-5.2 UD-Q6_K, 38.93 GB read per token:

| regime | working set | result |
|---|---|---|
| `--hot`, one position re-decoded | ~39.7 GB | 1.941 tok/s, 75.5 GB/s |
| `--full`, 256 tokens x 4 passes | ~466 GB | **1.932 tok/s, 75.2 GB/s**, spread 0.4% |
| prefill, cold to warm | | 0.9 → 6.9 tok/s |

A 12x difference in working set is worth 0.5%: once resident, footprint does
not matter, and there is no locality prize to be won in DRAM.

`--pages` measures the machine rather than the model — 40 GB, no weights:

| threads | sequential | expert-shaped blocks |
|--------:|-----------:|---------------------:|
| 8       | 81.7 GB/s  | 84.5 GB/s            |
| 16      | 100.6      | 98.9                 |
| 32      | 102.8      | **106.9**            |

So the scattered block pattern `mul_mat_id` produces costs nothing, ordinary
4 KiB pages already reach ~73% of the 140.8 GB/s this DDR4-2933 tops out at,
and the model's 75.4 GB/s is **~75% of what its own access pattern allows** —
the missing quarter is compute and per-node overhead, not memory.

Older figures, kept for the record: the same 270 one-token decodes with growing
context, measured *while paging 583 GiB in*, which is why they sit so far below
everything above.

| threads | nano-glm | llama.cpp `rescore --sim-gen` |
|--------:|---------:|------------------------------:|
| 16      | 1.27 t/s | 1.2 t/s                       |
| 32      | 1.34 t/s | 1.3 t/s                       |

Warmed steady-state reference, stock `llama-bench`, both OSes on the *same*
Mac Pro 7,1 (build `6a32c29a7`, 5 reps, mmap):

| test  | macOS (clang, Accelerate) | Windows (MSVC, no BLAS) |
|-------|---------------------------|-------------------------|
| tg32  | 2.00 @16 · **2.23** @32   | 1.58 @16 · 1.84 @32     |
| pp128 | **7.13** @16 · 6.43 @32   | 6.24 @16 · **7.26** @32 |

Reading these:

- Harness-style runs (the table above this one) measure below `llama-bench`
  because they page 583 GiB in *during* the measured window.
- macOS decode leads by ~15-20%, and the cause is not mmap alone: Windows
  `--load-mode none` (weights read into ordinary RAM) reaches 1.90 t/s vs
  1.84 for warm mmap. What mmap costs on Windows is *stability*, not average
  throughput — repeated identical invocations returned 1.04 ± 0.29 and
  1.84 ± 0.02, where macOS holds ± 0.01. Treat any single Windows mmap
  timing as suspect; re-run it.
- SMT helps decode a little on both (+11% / +16%) but its prefill sign is
  platform-dependent (macOS −10%, Windows +16%), so physical cores is the
  *fastest* prefill setting on macOS.

The default is physical cores — see `../logit-kld/src/cpu_topology.h`, and
note that changing `-t` changes the logits.

## Scope notes

- hparams are read from GGUF metadata but structural assumptions are
  asserted loudly (arch == glm-dsa, sigmoid expert gating, the degenerate
  freq_scale==1 rope case, power-of-two indexer key length). Anything else
  aborts rather than mis-computes.
- The indexer-types layout (which layers run a full lightning indexer vs
  reuse the previous top-k) comes from GGUF metadata, with the GLM-5.2
  default pattern as fallback — same BC rule as llama.cpp.
- Sparse DSA attention is exercised end-to-end even when n_kv < indexer
  top_k (2048): the top-k mask path always runs; it just selects everything
  for short sequences.
