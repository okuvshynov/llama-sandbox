# nano-glm — minimal CPU-only GLM-5.2 inference on bare ggml

An experiment in going back to basics: llama.cpp grew from a single-file
inference engine into a framework (142 model architectures, a KV-cache zoo,
15+ accelerator backends, batch/memory abstractions). nano-glm asks how small
a *single-model* engine gets if the compute layer is kept and the framework
is dropped: it links only `ggml` (kernels, GGUF reader, backend scheduler)
and reimplements the thin slice of llama.cpp that GLM-5.2 on CPU actually
needs — shard loader, single-sequence KV cache, forward graph, greedy loop —
in one source file.

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
every non-glm-dsa architecture.

Where this is heading: [PLAN.md](PLAN.md) — distributed MoE evaluation
(expert parallelism) across two Mac Pros, with GLM-5.2 as the testbed and
Kimi-K3-scale models as the goal.

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
```

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

## Verification

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

nano-glm tracks stock llama.cpp rather than beating it — the point is a minimal
engine with a bit-exact baseline, and the kernels are ggml's either way. On the
same workload (270 one-token decodes with growing context, GLM-5.2 UD-Q6_K,
582.87 GiB, Xeon W-3245 / 16C32T, Windows MSVC build):

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
  freq_scale==1 rope case). Anything else aborts rather than mis-computes.
- The indexer-types layout (which layers run a full lightning indexer vs
  reuse the previous top-k) comes from GGUF metadata, with the GLM-5.2
  default pattern as fallback — same BC rule as llama.cpp.
- Sparse DSA attention is exercised end-to-end even when n_kv < indexer
  top_k (2048): the top-k mask path always runs; it just selects everything
  for short sequences.
