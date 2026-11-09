# llama.cpp: why ds-completion stays on the DSV4.1 PR commit

**Decision (2026-11-09): keep the fork at `b72050d03`.** Upstream llama.cpp merged DeepSeek-V4.1
support, but it cannot read the GGUF files we use (Unsloth's DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL):
it loads them without an error and generates garbage. These files and that commit belong together.
When a newer DeepSeek model (Flash or full) is quantized with the merged code, we move to upstream.

## What ds-completion takes from llama.cpp

- **The tokenizer:** `llama_model_load_from_file` with `vocab_only`, `llama_tokenize`,
  `llama_token_to_piece` (`src/ds_completion.cpp`).
- **GGUF reading and ggml tensor metadata** (`src/model.cpp`): every tensor's type, shape, data
  pointer (the mmap) and row strides, computed by ggml from the tensor's type.
- **ggml's CPU ops**, for the CPU-only path (`src/forward.cpp`) and the reference checks.

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

## Other notes

- The other files are not affected: `dense-fp8.gguf` (`fp8pack`) is all `I8`, the DSpark draft
  `F32`, `BF16` and `MXFP4`, all with the same meaning upstream.
- Our code does not build against upstream as is: `src/model.cpp` names `GGML_TYPE_MXFP8`.
