# 5. How the backend plugs into llama.cpp

## ggml in three sentences

llama.cpp's math library is **ggml**. A model forward pass is built as a
**compute graph** — tensors as nodes, operations (`mul_mat`, `add`, `glu`,
...) connecting them — and then executed by **backends** (CPU, Vulkan, CUDA,
...). A **scheduler** decides which backend runs which node and splits the
graph accordingly.

The MoE-specific operation is `mul_mat_id`: "multiply by matrix number *i*
from this stack of 256", where the ids come from the router at runtime.
It appears three times per expert block (gate, up, down).

## The whole trick, honestly

This project never patches llama.cpp. It is a DLL that a stock llama.cpp
loads and defers to, through three mechanisms that already exist
([`docs/MECHANISM.md`](../docs/MECHANISM.md) has file:line references):

1. **Loading**: set the environment variable `GGML_BACKEND_PATH` to the DLL
   and ggml dlopens it as an extra backend at startup.
2. **Owning the weights**: llama.cpp's `-ot` flag maps tensor names (by
   regex) to a *buffer type* — a named memory pool. Our backend registers a
   buffer type called `MoE`; `-ot "exps=MoE"` places every expert weight in
   it. The buffer is ordinary host memory (`is_host = true`), which is
   load-bearing: anything we *don't* claim can still be computed by the CPU
   backend, in place, with no copy.
3. **Getting the ops**: the scheduler assigns an operation to the backend
   that owns its weights. Owning the experts makes the three `mul_mat_id`
   nodes ours, and adjacent glue ops (clamp, activation, the router-weight
   multiply) join the same **split**. The result: llama.cpp hands us the
   expert block — 13 nodes per layer — and nothing else
   ([`src/moe_backend.cpp:303`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_backend.cpp#L303)).

## What we do with a split

Three paths, chosen at runtime:

- **CPU delegate** (always available): hand the split to a CPU backend
  instance. Bit-identical to llama.cpp computing it itself — this is the
  correctness baseline and the universal fallback.
- **Mirror path** (default on the dies): copy each layer's expert weights
  into a die's HBM and run the split there with ggml's own Vulkan kernels,
  chunked 8 tokens at a time. This is the prefill winner (+21%).
- **TP path** (`MOESERV_TP=1`, decode only): the custom kernel, all four
  dies computing every layer together
  ([`src/moe_tp.h`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_tp.h),
  design in [06](06-optimization.md#tensor-parallel-across-the-dies)).

One invariant governs all three: **every fallback is a fallback to the CPU,
never a skip**. A layer that doesn't fit in VRAM, a graph shape we don't
recognize, a Vulkan error — all of them must produce a *slow and correct*
run, never a wrong one. The parser that decides whether a split is exactly
the expert block we support is deliberately paranoid
([`src/moe_tp.h:301`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/src/moe_tp.h#L301)).

## How correctness is proven

The test harness is llama.cpp itself
([`gate.py`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/gate.py)):
run `llama-perplexity` twice on the stub — once with the backend told to
compute nothing (`MOESERV_DISABLE=1`), once computing — and compare the
files of per-token log-probabilities each run writes. Identical files mean
our compute changed nothing. The CPU delegate passes **byte-identical**; the
GPU paths pass within a tolerance calibrated against how much two *known
correct* kernels differ on this machine (~3.6e-5 mean KL divergence).

Two testing principles worth stealing for any project:

- **The harness must not re-define the thing under test.** No replay format,
  no re-implemented reference graph — the comparison is llama.cpp against
  llama.cpp, differing in exactly one variable.
- **Every run must prove, from its own log, that it tested what it meant
  to.** Which buffer did the weights land in? Did the backend actually
  compute? Six times in this project a check came back green while testing
  nothing — a backend never loaded, a flag eaten by the shell, a pipeline
  that killed the process it measured. The scripts now abort rather than
  compare when engagement can't be shown.
