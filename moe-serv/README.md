# moe-serv — a ggml backend that claims the MoE block

`moeserv.dll` is loaded by an **unmodified** llama.cpp at runtime, takes
ownership of the routed expert weights, and computes the expert block. Nothing
else: the trunk, tokenizer, KV cache and head stay llama.cpp's.

Two compute paths, selectable at runtime, both verified against llama.cpp's
CPU on the same weights:

- **default** — the block on the four Vega II dies with ggml's kernels,
  chunked to their fast path: **+21% prefill** on DeepSeek-V4-Flash.
- **`MOESERV_TP=1`** — decode-only tensor-parallel across all four dies with
  a custom mxfp4 kernel: **+7.6% decode vs stock** on the full model, the
  block itself 3.1x faster where resident.

## Build

```powershell
.\build.ps1                  # -> build\bin\moeserv.dll (+ shaders\)
.\build.ps1 -Clean
```

Needs `LLAMA_CPP_DIR` (defaults to the tree beside this repo). Only ggml is
used — no llama headers, no llama linkage, and no patch to llama.cpp ever.

## Run

```powershell
$env:GGML_BACKEND_PATH = "...\moe-serv\build\bin\moeserv.dll"
llama-bench      -m <model> -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=MoE"
llama-completion -m <model> -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=MoE" -p "..."

$env:MOESERV_TP = "1"        # the TP decode path (four dies, custom kernel)
```

The regex is `-cmoe`'s own (`LLM_FFN_EXPS_REGEX` in `common/common.h`) pointed
at `MoE`. Knobs: `MOESERV_DISABLE=1` (own the weights, compute nothing — the
control), `MOESERV_TP_BUDGET_MB` (per-die VRAM budget, default 28000; layers
that do not fit fall back to the CPU), `MOESERV_SHADERS` (shader dir if not
next to the DLL), `MOESERV_PROFILE=<prefix>` (per-split phase CSV).

**Before believing any run**: check its log for
`load_backend: loaded MoE backend from ...` and, for TP, a nonzero
`MoE-TP: N splits on the dies`. A missing `-ot` means the backend silently
never loads (`docs/MECHANISM.md` #1).

## Reproduce the current state

```powershell
# the instrument: a 4-layer prefix of the model, ~16 GiB, loads in seconds
python make_stub.py <model-00001-of-000NN.gguf> D:\llms\stub\ds4-L4.gguf --layers 4

# correctness (each asserts placement + engagement from its own log)
python gate.py                                     # CPU path: bit-identical
python gate.py --build-dir build-vk --tol 5e-4     # ggml-vulkan mirror path
python gate.py --tp --ubatch 1 --tol 5e-4          # TP path, decode shape

# second architecture (GLM-5.2, q6_K experts): stub is 3 dense + 2 MoE layers.
# The mirror gate needs the allocation override — one expert tensor is
# 2520 MiB against the driver's 2 GiB maxMemoryAllocationSize; without it the
# mirror silently falls back and the gate is green for the wrong reason.
python make_stub.py <GLM-...-00001-of-00014.gguf> D:\llms\stub\glm-L5.gguf --layers 5
python gate.py --model D:\llms\stub\glm-L5.gguf    # CPU path: bit-identical
$env:GGML_VK_FORCE_MAX_ALLOCATION_SIZE = "3221225472"
python gate.py --model D:\llms\stub\glm-L5.gguf --build-dir build-vk --tol 5e-4

# throughput (stub ~15 min; full model ~2 h)
python bench.py --model D:\llms\stub\ds4-L4.gguf --tp
python bench.py --tp                               # full model, decode
python bench.py --build-dir build-vk --pp 512      # full model, prefill
```

Current full-model numbers (tg32, two loads each, deltas resolved above the
run's own noise floor):

| stock | ours-off | ours-on | ours-tp |
|---|---|---|---|
| 3.64 | 3.48 | 3.46 | **3.92** |

`llama-perplexity` and `llama-bench` come from a **CPU-only** llama.cpp build
(`build`); the `build-vk` host is needed only for the ggml-vulkan mirror path
and always gets the no-op-offload flag, which the scripts add.

## Where things are written down

- `PLAN.md` — goal, invariants, the correctness contract, every step with its
  status, and what is deliberately parked.
- `docs/MECHANISM.md` — how an unmodified llama.cpp is persuaded to hand over
  exactly this block, and nine things about out-of-tree ggml backends that
  each cost an hour to learn.
- `docs/MEASUREMENTS.md` — every number quoted anywhere, with its instrument
  and noise floor, plus the measurement discipline itself.
- `docs/KERNEL.md` — the kernel experiment ledger (E1-E9), wins and
  postmortems alike.
- `gate.py` / `bench.py` / `make_stub.py` — each carries its reasoning in its
  header.
