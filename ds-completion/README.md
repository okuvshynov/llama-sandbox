# ds-completion: DeepSeek-V4.1-Flash inference for the Mac Pro (755 GB RAM, 4x Vega II)

An inference engine for one model, DeepSeek-V4.1-Flash-1017, written from scratch for one
machine: a Mac Pro 2019 with two Radeon Pro Vega II Duo cards, or two such machines.

    ./ds-completion -m <model> -p "prompt"

Priorities, in order: correctness (outputs that match the official reference implementation),
clarity, performance. Non-goals: other models, other quantizations, other hardware, serving
many users at once.

How to build and run it: [QUICKSTART.md](QUICKSTART.md).

## Status (2026-11-18)

- **One machine:** 44.6 t/s mean (34-55) on the 8 held-out prompts with speculative decoding
  and the adaptive expert cache, every output byte-identical to plain greedy decoding
  (step 104). CPU only: 3.5 t/s, as the llama.cpp fork.
- **Two machines** (the model in two halves, layers 0-19 and 20-39, the residual stream over
  10 GbE): 50.9 t/s mean (35-63) on the same prompts, the same outputs (steps 110-112).
- **Exact and deterministic:** fp8 and fp4 weights computed exactly on hardware without them;
  with `--canon` the output does not depend on where an expert runs. Gated per layer and end to
  end against the official implementation.
- **Prompt processing** in chunks of 512: 93 t/s on a 1,989-token prompt, 63 with `--canon`
  (steps 37-45, 68).
- **Long context:** 65,536 tokens by default, up to 1M; gated against the reference at 20,000
  tokens (steps 34-35).
- **Sampling:** temperature and top-p with a seed; one seed gives one output with or without
  speculative decoding (steps 148-151). Greedy is the default.
- **Interactive sessions** (`-i`, `scripts/chat.sh`): chat and thinking mode over several
  turns (steps 108-109).
- **A server:** the engine answers requests from stdin (`--requests`, step 152), and
  `server/ds_server.py` puts the Anthropic Messages API in front of it: one request at a time,
  tools, tool results and reasoning effort rendered by the model's own `encoding.py`
  (steps 153-154, 157). The pi coding harness runs against it (step 157); a request that goes on from the
  last one runs only its new tokens (step 158), and a new conversation starts from a point
  saved at the end of the shared system prompt and tools (step 160). A stream carries the
  reasoning and the text as they are generated (step 161). `GET /metrics` has counters and
  gauges in the Prometheus format (step 165).

## The product and the tools around it

Three kinds of code live here, and the directories alone do not say which is which
(file by file: [docs/overview.md](docs/overview.md), "Code map"):

| | what | where |
|---|---|---|
| **The product** | the engine with the GPU runtime (`bin-hip/ds-completion --rt ...`), the HTTP server in front of it, and `fp8pack`, which makes one of its files | `apps/ds_completion.cpp`, `apps/fp8pack.cpp`, `hip/`, `src/` except `forward.*`, `server/ds_server.py`, `scripts/serve.sh`, `scripts/chat.sh`, `placements/` |
| **What checks it** | a second implementation of the model on the CPU (ggml graphs), the test programs, the official Python reference and the golden data it writes | `src/forward.*`, `tests/`, `ref/`, `server/test_server.py`, `server/fake_engine.py` |
| **Tools** | benchmarks, the runner of measurements and the readers of records and traces, the inspector, every measurement's raw output | `bench/`, `scripts/` (the rest), `apps/ds_inspect.cpp`, `results/` |

The second and third may depend on the product; the product should not depend on them. It
still does in four small places, listed in the code map; removing them is in
[docs/plan.md](docs/plan.md), "The product and the tools".

## The documents

Two in this directory: this page and [QUICKSTART.md](QUICKSTART.md) (what it needs, how to
build, every option, the current results, the limits).

| | |
|---|---|
| [docs/steps.md](docs/steps.md) | the lab notebook: every step in order, what was done, measured and decided. "README step N" and "step N", in code comments, results files and the other documents, mean its section "Step N" (it was this file until step 156) |
| [docs/overview.md](docs/overview.md) | a 20-minute read: what was built, how a token is computed, what we learned, the code map |
| [docs/timing.md](docs/timing.md) | where the time goes, on one page |
| [docs/numerics.md](docs/numerics.md) | operand types, how they map to the GPUs, accumulation |
| [docs/plan.md](docs/plan.md) | the items and their status, the plans and cost models |
| [docs/process.md](docs/process.md) | how the work was approached: the method, the decision log, predictions against outcomes, mistakes |
| [docs/reproduce.md](docs/reproduce.md) | how to rebuild the setup and which command produced which results file |
| [docs/record.md](docs/record.md) | `--record`: a run as one JSON object |
| [docs/requests.md](docs/requests.md), [docs/server.md](docs/server.md) | `--requests`, the protocol between the engine and a front; the HTTP front |
| [docs/pi.md](docs/pi.md) | the pi coding harness on this engine: the setup, step by step |
| [docs/rocprof.md](docs/rocprof.md) | the GPU side under AMD's profiler |
| [docs/llama-cpp-upstream.md](docs/llama-cpp-upstream.md) | llama.cpp: upstream or the PR's commit, what upstream reads wrong in these files, and how we came not to depend on it |
| [scripts/README.md](scripts/README.md) | the tools still in use |

Until step 156 the documents of `docs/` with a lower-case name of one word stood in this
directory in capitals, and code comments and results files still name them so: "PLAN.md" is
`docs/plan.md`, and the same for OVERVIEW, TIMING, NUMERICS, PROCESS and REPRODUCE.

## Docs and commits

Every commit that changes code, scripts or results updates, in the same commit:
`docs/steps.md` (a step: what was done and measured), `docs/process.md` (the question,
prediction, result and decision), `docs/reproduce.md` (new scripts and results and what
produces them), and `QUICKSTART.md`, `docs/overview.md`, `docs/plan.md`, `docs/timing.md` and
this page's status when what they say changed. New results files start with a `#` header naming the command or script, the commit,
the clock level and the date. (The rule for the assistant is in the sandbox's CLAUDE.md.)
