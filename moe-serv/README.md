# moe-serv — a ggml backend that claims the MoE block

`moeserv.dll` is loaded by an **unmodified** llama.cpp at runtime, takes
ownership of the routed expert weights, and computes the expert block. Nothing
else: the trunk, tokenizer, KV cache and head stay llama.cpp's.

See `PLAN.md` for why this shape, what the mechanism is, and where it is going.
This file is how to build and run it, and the things that cost an hour to find
out.

## Build

```powershell
.\build.ps1                  # -> build\bin\moeserv.dll
.\build.ps1 -Clean
```

Needs `LLAMA_CPP_DIR` (defaults to the tree beside this repo). Only ggml is
used — no llama headers, no llama linkage, and no patch to llama.cpp ever.

## Run

```powershell
$env:GGML_BACKEND_PATH = "...\moe-serv\build\bin\moeserv.dll"
llama-bench      -m <model> -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=MoE"
llama-completion -m <model> -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=MoE" -p "..."
```

The regex is `-cmoe`'s own (`LLM_FFN_EXPS_REGEX` in `common/common.h`) pointed at
`MoE` instead of the CPU, so the two are directly comparable: swap `=MoE` for
`=CPU` and you have the baseline.

## Four things that are not obvious

**1. `GGML_BACKEND_PATH` is not honoured by every tool.** `llama_backend_init`
calls `ggml_backend_load_all()` **only if `ggml_backend_reg_count()` is zero**
(`src/llama.cpp`), and in a normal build the static CPU backend has already
registered — so the count is nonzero and the variable is never read. It is
honoured when something else calls `load_all()`:

| | how |
|---|---|
| `llama-bench` | calls it unconditionally |
| common-arg tools | only via a handler that calls it — `-ot`, `--device`, `--list-devices` |

Since every real invocation here passes `-ot`, this works. But a run with only
`GGML_BACKEND_PATH` set and no `-ot` silently does not load the backend, which
makes any comparison against it meaningless. **Check for the
`load_backend: loaded MoE backend from ...` line before believing a result.**

**2. Three compile definitions are needed and one fails silently.**

| define | what it does |
|---|---|
| `GGML_BACKEND_DL` | gates `GGML_BACKEND_DL_IMPL`, which generates the `ggml_backend_init` / `ggml_backend_score` entry points |
| `GGML_BACKEND_SHARED` | gates `GGML_BACKEND_API` being a visibility attribute at all — without it the macro is a bare `extern` |
| `GGML_BACKEND_BUILD` | selects `dllexport` over `dllimport` within that |

Omitting `GGML_BACKEND_SHARED` compiles and links cleanly, and the loader then
says `failed to find ggml_backend_init`. Nothing warns that an entry point was
declared and never exported.

**3. `enum` is required on some ggml types in C++.** `ggml-backend.h` declares
both an enum `ggml_backend_dev_type` and a *function* of the same name. C keeps
those in separate namespaces; C++ lets the function hide the type, so
`static ggml_backend_dev_type f(...)` does not compile and
`static enum ggml_backend_dev_type f(...)` does. ggml's own backends are written
that way — it reads as house style and is a requirement.

**4. Do not let a second `ggml-base.dll` sit next to `moeserv.dll`.** The build
tree produces one as a dependency. It happens to be harmless today because
Windows resolves the import to the *host* llama.cpp's copy, which is what we
want — two ggml instances would mean two registries and the backend would
register into the wrong one. That is the loader's search order rather than
anything arranged here, so treat it as a hazard: if `MoE` stops appearing in
`-ot`'s buffer type list after a build change, this is the first thing to check.

## Correctness

Three layers, none of which depend on `../nano-glm`:

1. `GGML_BACKEND_PATH=... test-backend-ops` — llama.cpp's own per-op
   conformance suite, which enumerates registered devices and compares each
   against CPU.
2. End-to-end output with and without the backend, greedy and fixed seed.
   Bit-identical through `passthrough`; a stated numerical floor after.
3. Capture replay (`tape`) — real tensors from real runs, checked against
   recorded output.

## Status

**`handshake` done.** The library registers, exposes a buffer type named `MoE`
that `-ot` resolves, and claims no ops. Pointing `-ot` at it on
DeepSeek-V4-Flash places **140352 MiB of expert weights in our buffer** and
produces byte-identical output to a stock run.

That last part is worth reading twice: llama.cpp honours `-ot` even though our
`supports_op` returns false for everything, so we already *own* the weights. The
run stays correct because our buffer is host memory — the CPU backend accepts
any host buffer type and computes on our tensors in place, with no copy. So
weight ownership is proven before a single op is claimed, and `is_host` is
load-bearing from here on.

Next is `passthrough` in `PLAN.md`, which is now the smaller half: claim the
block's ops and implement `graph_compute`.
