# llama.cpp: upstream, the DSV4.1 PR commit, and the one tensor between them

**Now (2026-11-22, step 175): ds-completion builds against upstream llama.cpp, pinned to one
commit of its main branch (`e626db7ca`; QUICKSTART names it).** The one tensor upstream reads
wrong in our files (the Engram table) is decoded by our own code, on the GPU runtime and on the
CPU path, and checked at load (step 168). The PR's commit also still works, with the same
results, and only llama.cpp's *own* programs on these files (`llama-cli`, our
`bin/llama-logits`) need it: "The move" at the end. The sections between are the history, as
written at the time.

**Decision (2026-11-09), replaced by the above: keep the fork at `b72050d03`.** Upstream llama.cpp merged DeepSeek-V4.1
support, but it cannot read the GGUF files we use (Unsloth's DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL):
it loads them without an error and generates garbage. These files and that commit belong together.
When a newer DeepSeek model (Flash or full) is quantized with the merged code, we move to upstream.

**Checked again (2026-11-20):** upstream still reads these files wrong, but we do not need it to
read them. With a small change on our side (the Engram rows decoded by our own code, not by
ggml) `ds-completion` built against upstream master gives the stored outputs: "A trial against
upstream" below. Nothing is changed in the code yet, and the decision above stands until it is.

## What ds-completion takes from llama.cpp

- **The tokenizer:** `llama_model_load_from_file` with `vocab_only`, `llama_tokenize`,
  `llama_token_to_piece` (`apps/ds_completion.cpp`).
- **GGUF reading and ggml tensor metadata** (`src/model.cpp`): every tensor's type, shape, data
  pointer (the mmap) and row strides, computed by ggml from the tensor's type.
- **ggml's CPU ops**: until step 194, for the CPU reference path (`src/forward.cpp`, removed). Now only one benchmark's baseline (`bench/bench_cpu_moe.cpp`).

The GPU runtime (`hip/`) and the CPU expert kernels are our own code; they read raw bytes.

## The fork

- `~/forks/llama.cpp`, branch `deepseek41`: one commit, `b72050d03ae14bb3cb5138ecefab3f3f65157d31`
  ("model : add DeepSeek-V4.1-Flash (deepseek41)", Daniel Han, 2026-10-19) on upstream
  `797de63b2` (2026-10-17). It is the head of [PR #30817](https://github.com/ggml-org/llama.cpp/pull/30817),
  the PR Unsloth's guide builds and that made these GGUF files.
- Reproduced on a fresh machine by its hash (GitHub serves it; checked 2026-11-09):

```bash
git clone https://github.com/ggml-org/llama.cpp ~/forks/llama.cpp
cd ~/forks/llama.cpp
git fetch origin b72050d03ae14bb3cb5138ecefab3f3f65157d31
git checkout -b deepseek41 b72050d03ae14bb3cb5138ecefab3f3f65157d31
cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32
```

  (`git fetch origin pull/30817/head` gives the same commit today, but a PR head can move; the
  hash cannot.)

**Checked (2026-11-09, on tomb):** a fresh clone of upstream plus that commit, built as above, with
`ds-completion` built against it (`make FORK=<clone> BIN=bin-fresh`, `make HIP=1 FORK=<clone>
BIN=bin-fresh-hip`):

| check | result |
|---|---|
| `bin-fresh/fp8pack` regenerates `dense-fp8.gguf` | byte-identical to ours (sha256 `31865c3e...`), 20 s |
| held-out cpp-matrix and tcp-udp, the held-out command (`--spec --canon --cache-loads 800 --cache-period 8 --cache-wait`) | outputs byte-identical to `results/cache-test/plain-canon-static/`; 49.60 and 35.47 t/s |
| its `llama-completion` (greedy, 32 tokens) | the same text as our fork's |

## What goes wrong with upstream

Tested with upstream `a4d4f2fd4` (2026-11-09), CPU build, `llama-completion -p "The three largest
cities in Ukraine are" -n 32 --temp 0 --top-k 1 -t 16`:

| build | output |
|---|---|
| fork `b72050d03` | `: Kyiv, Lviv, and Odesa. Kyiv is the capital and largest city. ...` |
| upstream `a4d4f2fd4` | `ademyunkyEditing其三hots埠乘积iline代入 sund SUS BAL...` |

Both load in ~170 s and run at ~3.6 t/s, and the upstream log has no error or warning.

**The cause: one tensor, and a type id that changed meaning.** Of all the files' tensors, exactly
one has ggml type id 43: the Engram table `per_layer_token_embd.weight` (256 x 768,022,850 values,
the whole of shard 8):

- In the PR (and the files), **43 is MXFP8**: blocks of 32 e4m3 codes plus one e8m0 scale byte,
  33 bytes per 32 values. The tensor is 202,758,032,400 bytes.
- In merged upstream ggml, **43 is `F8_E4M3`**: plain fp8, 1 byte per value, no scale. Read so, the
  same tensor is 196,613,849,600 bytes: every row lands misaligned and the scale bytes are read as
  weights. The data still fits in the file, so nothing checks it.

(Both readings from the two versions' `gguf-py` over the same files; `GGMLQuantizationType` 43.)
Upstream's converter (`conversion/deepseek41.py`) never writes MXFP8: it multiplies the e4m3 codes
by their scales and stores the Engram table as BF16 (exact, but ~393 GB for this table). The PR's
MXFP8 type was not merged, and its id was later given to `F8_E4M3`.

**Known upstream:** [issue #31385](https://github.com/ggml-org/llama.cpp/issues/31385) (open since
2026-10-29): Unsloth's UD-Q4_K_XL first failed with `tensor 'per_layer_token_embd.weight' has
invalid ggml type 43`, then, once `F8_E4M3` was merged as 43, loaded and produced gibberish. A
maintainer called the MXFP8 Engram an unofficial quantization scheme. Only the quants whose Engram
is MXFP8 are affected (UD-Q3_K_XL, UD-Q4_K_XL, UD-Q8_K_XL); the others store it as Q5_K, Q6_K or
IQ4_NL, which upstream reads (the reporter has UD-Q2_K_XL working). Unsloth also ships the Engram
alone in other precisions ([guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash): MXFP8 202.8
GB, Q6_K 161.3, Q5_K 135.2, IQ4_NL 110.6) with a script to swap it in.

## Ways to upstream, if we ever need one before a new model

1. A narrow shim in our loader: the Engram tensor with type 43 whose region in the file holds 33
   bytes per 32 values read as the pre-merge MXFP8 (the GPU runtime already decodes those bytes;
   the CPU path would need its own row gather instead of `ggml_get_rows`). No new files, exact.
2. Unsloth's Q6_K Engram file: upstream reads it, but it is lossy, so every output changes and
   our byte-identical references are lost.
3. The Engram table repacked into our own `I8` tensor in a separate file, as `fp8pack` does for the
   dense weights: clean for upstream, ~203 GB more disk.

Also worth having upstream, for everyone with these files: the `deepseek41` loader refusing a
type-43 Engram whose size on disk does not match, instead of producing garbage (not filed).

## A trial against upstream (2026-11-20)

The question: a person who wants to run this (the pi harness on one machine, say) has to fetch
a PR's commit by its hash and build it. Could any recent llama.cpp do?

**Upstream has not changed for these files.** At master `e626db7ca` (2026-11-20, 353 commits
after `a4d4f2fd4`) type 43 is still `F8_E4M3`, and no MXFP8 type exists in `ggml.h`. Upstream's
own programs still misread the Engram table.

**What we take from the fork that upstream lacks is small.** Every ggml function our code calls
is in upstream's `ggml.h` (`ggml_dsv4_hc_pre`, `ggml_swiglu_clamp`, `ggml_mul_mat_id` and the
rest). Three places depend on type 43 being MXFP8:

| place | what it takes from ggml |
|---|---|
| `src/model.cpp`, the Engram tensor's check | the name `GGML_TYPE_MXFP8` |
| `hip/rt.cpp`, `fetch_engram_rows` (the GPU runtime) | the type's `to_float` and the row's size in bytes (`nb[1]`) |
| `src/forward.cpp`, `Engine::engram` (the CPU-only path; removed in step 194) | `ggml_get_rows` on the table |

**The trial** (`results/dist/trunk-trial.patch`, on a copy of our sources at `40f345e`; the
repository's code is as it was): the first two places only. The type is checked as the number
43, and a new 20-line header decodes a row itself: blocks of 33 bytes, one e8m0 scale and 32
e4m3 codes, value = e4m3(code) x 2^(scale - 127); a row of 256 values is 264 bytes. Built
against upstream `e626db7ca` (CPU, static libraries, as the fork is built):

| check | result |
|---|---|
| `make` and `make HIP=1` (`ds-completion`, `ds-inspect`, `fp8pack`) | no errors |
| the 8 held-out prompts, the held-out command | every output byte-identical to `results/cache-test/plain-canon-static/` (`results/dist/trunk-trial.txt`) |
| `fp8pack` regenerates `dense-fp8.gguf` | byte-identical to ours, 18.8 s |

The Japanese and Chinese prompts are among the eight, so upstream's tokenizer gives the same
ids for them as the fork's. The speeds in the results file are single runs after a restart
and are not a measurement.

**Not covered by the trial:**

- The CPU-only path (as it was then; removed in step 194). `Engine::engram` still gathers the rows with `ggml_get_rows`, which on
  upstream would read them as plain fp8, with no error. It needs the same decoder (the rows
  as an input tensor filled by our code).
- The gates against the reference (`ds-test` per layer and end to end). ggml's CPU kernels have
  a month of upstream changes since the fork's base (2026-10-17), and the CPU path's numbers
  may have moved.
- The other programs (`ds-test`, `llama-logits`, the benchmarks) were not built.
- A check at load that the tensor is what we decode: its size in the file is 33 bytes per 32
  values. The trial trusts the type number.
- Two machines, sampling, `--requests`: none of them reads the table another way, but none
  was run.

**What it would take to move** (way 1 of the section above, now known to be enough):

1. Our own decoder on both paths, and the size check at load. The code then builds against
   the pinned commit and against upstream alike, and nothing changes for a setup on the pin.
2. The gates and the held-out set against upstream; then QUICKSTART and `reproduce.md` name a
   tested upstream commit in place of the PR's.

What we would give up: the pin keeps upstream's changes out. Following upstream, a later commit
can break the build or move the CPU reference's numbers; the tested commit in the documents is
the way back.

**To repeat the trial:** upstream at `e626db7ca` built with the two `cmake` lines above;
`git apply ds-completion/results/dist/trunk-trial.patch` on `40f345e`; `make FORK=<upstream>
BIN=bin-trunk && make HIP=1 FORK=<upstream> BIN=bin-trunk-hip`; the held-out command per prompt
of `results/spec-heldout/prompts.txt` (as `scripts/spec-measure.sh` runs it, with
`--cache-loads 800 --cache-period 8`, without `--spec-log`), each output against the stored
one with `cmp`; `bin-trunk/fp8pack -m $M -o <file>` and `cmp` with `dense-fp8.gguf`.

## The move (2026-11-21, step 168)

The trial's two steps, done:

- **The decoder** (`src/model.h`, `src/model.cpp`): `mxfp8_to_float` and `mxfp8_row_bytes`. A
  block is 33 bytes, one e8m0 scale then 32 e4m3 codes; value = e4m3(code) x 2^(scale - 127),
  exact in fp32, the same values as the PR's `dequantize_row_mxfp8`.
- **Both paths use it.** The GPU runtime (`hip/rt.cpp`, `fetch_engram_rows`) in place of the
  type's `to_float` and `nb[1]`. The CPU path (`src/forward.cpp`, `Engine::engram`; removed in step 194) gathered and
  decoded the rows on the host and gives them to its graph as an input, in place of
  `ggml_get_rows` on the table.
- **The type is the number 43**, and the loader checks what the number cannot say: the tensor's
  place in its file (up to the next tensor or the file's end) must hold 33 bytes per 32 values,
  or the load fails with a message. Plain fp8 under the same number would be smaller.
- `ds-inspect` prints the table's size from the layout, not from `ggml_nbytes`.

Nothing in the code reads the table through ggml now; the rule is in a comment at the tensor
(`src/model.h`).

**Checked** (`results/dist/trunk.txt`; three builds: before the change on the pin, the change on
the pin, the change on upstream `e626db7ca`):

| check | the change, pin | the change, upstream |
|---|---|---|
| `make`, `make HIP=1`: every program | no warnings | builds; two deprecation warnings in `ref/llama_logits.cpp` |
| per-layer gate (`hello-fp32`), CPU path and GPU runtime | 201 rows each, equal to before | equal |
| end-to-end logits, both paths | the same bytes | the same bytes |
| the 8 held-out prompts, 2 runs each | 16 of 16 the stored outputs | 16 of 16 |
| `fp8pack` regenerates `dense-fp8.gguf` | (not run again) | byte-identical, 19.1 s |

So the month of upstream changes to ggml's CPU kernels did not move the CPU path on this gate.
Speeds are within about 2% of step 138's on both; one prompt read 3.4% slow on the pin build
and normal in three repeats (the file's section 4).

**What still needs the PR's commit:** llama.cpp's own reading of these files. `bin/llama-logits`
(`ref/llama_logits.cpp`) and the CPU baseline (`llama-cli`, `llama-completion`) run llama.cpp's
implementation of the model, which upstream gets wrong for a type-43 Engram table. Built
against upstream, `llama-logits` compiles and would give wrong logits with no error.

**Not checked:** the longer golden sets against upstream; the size check against a file that
fails it; two machines and `--requests` with the upstream build. Every result in this
repository before this step was made with the pin, and this machine's build stays on it.
(As of step 168. Steps 174-175, below, ran the longer golden sets and `--requests` and moved this
machine's build to upstream; two machines left the code in step 178.)

**Following upstream from here:** a later commit can break the build or move the CPU path's
numbers. `e626db7ca` is the way back; after a move to a newer one, the gate of this section.

**Since step 175 (2026-11-22)** upstream `e626db7ca` is the only llama.cpp the instructions name.
Every check was run again with it as the build in `bin/` and `bin-hip/`
(`results/dist/upstream.txt`): the longer golden sets to 20,000 tokens, the held-out prompts,
the request and server tests. `bin/llama-logits` is no longer built by `make` or `make tests`;
`make bin/llama-logits LLAMA_CPP=<a checkout of the PR's commit>` builds it when llama.cpp's own
logits are wanted again.

**Step 176:** the checkout is `~/projects/llama.cpp` and the Makefile's variable `LLAMA_CPP`
(`FORK` in the commands above works again since step 182; from step 176 until then the Makefile ignored it). One more user of the PR's code turned up: the
Python reference (`ref/weights.py`) imported the checkout's `gguf-py` and used its `MXFP8` type
for the Engram table. It reads the table by its layout now, as `src/model.h` does.

## Other notes

- The other files are not affected: `dense-fp8.gguf` (`fp8pack`) is all `I8`, the DSpark draft
  `F32`, `BF16` and `MXFP4`, all with the same meaning upstream.
- Before step 168 our code did not build against upstream: `src/model.cpp` named
  `GGML_TYPE_MXFP8`, the only compile error.
