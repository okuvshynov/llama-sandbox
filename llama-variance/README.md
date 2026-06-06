# llama-variance

Single-shot variance study for local OpenAI-compatible chat servers.

This isn't really a benchmark — there's no leaderboard, no model ranking.
The question it tries to answer is *"how much of a local model's apparent
score on a coding-style task is signal vs. sampling noise?"*. Each run
makes one HTTP request with `n=N` completions, scores each completion
independently against the same test corpus, and appends one JSONL row
per completion. Sweep that across (temperature, top_p, top_k, seed, …)
and you have a distribution to look at.

## Project lineage

llama-variance copies pieces from two sibling projects but keeps zero
runtime dependencies on either — they're allowed to drift.

- From [validation-bench](../validation-bench/): the task structure
  (`data/specs/<spec>/`, `data/envs/<env>/`, `data/tasks/<spec>-<env>/`),
  the docker sandbox flags, the strict `valid`/`invalid` scoring contract,
  and the `submit` tool shape.
- From [server-n-bench](../server-n-bench/): the OpenAI-style `n=N`
  parameter idea — one HTTP request that fans out to N parallel slots,
  so all N draws share identical sampling-param treatment and prompt
  processing happens once.

The differences from validation-bench are deliberate:

- **One turn only.** No compile-feedback loop, no resubmission. The
  preamble tells the model that explicitly — first `submit` call is the
  measurement, full stop.
- **N completions in one request** instead of N attempts. Lets us study
  the per-sampling-param distribution at the cost of one HTTP round trip.
- **Just one task today**: `toml-1.0-cpp17` (write a TOML v1.0 validator
  in C++17). Adding another (spec, env) cell is a matter of dropping
  files into `data/`.

## Setup

```bash
docker build -t vb-sandbox-cpp17 data/envs/cpp17/
```

The image is the same one validation-bench uses (`vb-sandbox-cpp17`) —
the Dockerfile is duplicated here so the projects are independent, but
either project can use the prebuilt image.

The test corpus is already checked in under `data/specs/toml-1.0/tests/`
(copied from validation-bench's `.cache/toml-test/tests/`, pinned at
upstream commit `0ee318a`). `tests.jsonl` is the manifest the scorer reads.

The TOML specification text (`data/specs/toml-1.0/spec_body.md`) and the
test corpus are both MIT-licensed upstream content vendored verbatim;
see `data/specs/toml-1.0/THIRD-PARTY-NOTICE` for the per-source license
and provenance pins.

## Server setup

llama-server must be started with `-np >= N` so it has enough slots to
serve `n=N` completions in parallel. The standard `n=N` perf advice from
server-n-bench applies — `--flash-attn`, `--kv-unified`, etc. as desired.

```bash
llama-server -m model.gguf -np 16 --jinja
```

`--jinja` is what enables tool-call support in the OpenAI-compatible
chat endpoint (the model needs a chat template that emits the
function-call tags).

## Run

```bash
python run.py --model qwen3-coder --n 16 --temperature 0.7 \
    --jsonl results/sweep.jsonl
```

Each invocation issues one `n=N` request, then scores each choice
sequentially through one Sandbox (with `begin_submission` restarting
the container between scorings, same as validation-bench). The headline
output is the JSONL — one row per completion, carrying its own
confusion matrix + sampling params + server meta.

Sweep example — shell-driven so each row is a fresh request:

```bash
for t in 0.0 0.3 0.5 0.7 1.0 1.3; do
  for r in 1 2 3; do
    python run.py --model qwen3-coder --n 16 --temperature $t \
        --jsonl results/sweep.jsonl
  done
done
```

Targeting a remote llama-server (e.g. a runpod tunnel; HTTPS works as-is):

```bash
python run.py --base-url https://abc-8080.proxy.runpod.net/ \
    --model qwen3-coder --n 16 --temperature 0.7 \
    --jsonl results/sweep.jsonl
```

## JSONL row shape

One row per choice in the `n=N` response. Top-level fields:

| field | notes |
|-------|-------|
| `ts` | epoch seconds at the time of the request |
| `task`, `spec`, `env` | identifies the (spec, env) cell |
| `base_url`, `model`, `build_info`, `total_slots` | server provenance from `/props` |
| `completion_idx`, `n_total` | which of the N choices this row is |
| `sampling_params` | the full sampling dict (`max_tokens`, `n`, plus any of `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`) |
| `finish_reason` | `"tool_calls"` on a clean submit; `"length"` if `max_tokens` ran out before submit |
| `model_seconds` | wall time of the one HTTP request (same value on every row from the same call) |
| `tokens_predicted` | per-choice predicted token count, via the server's `/tokenize` endpoint applied to `content + reasoning_content + tool_call.arguments`. Tends to slightly undercount vs request-level `completion_tokens` because chat-template envelope tokens around tool-calls / reasoning blocks aren't part of the text bodies we re-tokenize; small (~1%) drift is expected. |
| `tokens_content`, `tokens_reasoning`, `tokens_tool_args` | the three-way split that sums to `tokens_predicted`. Useful for thinking-mode studies (where the bulk usually lives in `tokens_reasoning`). |
| `compiled` | bool |
| `tp`, `fn`, `fp`, `tn`, `passed`, `total`, `mcc` | confusion matrix; absent when `compiled=false` |
| `error` | set when the choice didn't yield a compilable submission (`no_tool_call`, `wrong_tool:X`, `bad_args_json`, `no_source_code`, `compile_error`, `compile_timeout`) |
| `prepare_seconds`, `tests_seconds`, `score_wall` | per-completion scoring breakdown |
| `usage`, `timings` | recorded only on the row with `completion_idx=0`. **Gotcha**: at `n>1`, llama.cpp's OAI chat handler returns slot-0's `usage`/`timings` unchanged (see `tools/server/server-context.cpp` ~line 3260 — it appends the other slots' choices into arr[0] but never touches its usage block). So `usage.completion_tokens` and `timings.predicted_n` are the **single-slot** counts, not the sum across N. For true per-choice tokens use the `tokens_predicted` field below. |
| `note` | optional free-form tag from `--note` (e.g. machine name, experiment label) |

## Tracked datasets

`results/*.jsonl` files in this repo are check-in-quality runs kept for
reference. All current datasets target Qwen3.6-27B with the UD-`{Q8,Q6,Q5}`_K_XL
quant family (per-dataset blurb says which) on llama-server `b9048-5207d120e`
(M2 Ultra, 64 slots) at server-default sampling (`max_tokens=65536`, no
temperature/top_p/etc. overrides). PNGs and other ad-hoc artifacts under
`results/` are gitignored (see `.gitignore`) — only the `.jsonl` datasets
are checked in.

- `results/res.jsonl` — first smoke runs, 52 rows: 28 rows from 7 × `n=4`
  plus 24 rows from 3 × `n=8`. The variance signal is already loud at
  this size: 23/52 rows hit `compile_error` before reaching the corpus;
  of the 29 that compiled, MCC spans `-0.80 – 0.68` and `passed` spans
  `63 – 583` out of 678. The bottom tail includes a near-inverted
  validator (MCC `-0.80`, 63/678) — same prompt, same sampling params
  as the `0.68` row. Each request's rows share a `ts` field, so dedupe
  on `ts` if you ever suspect a double-append. No `tokens_*` fields
  (collected before the tokenize-based per-choice counting landed).

- `results/res_reasoning_on.jsonl` — 76 rows from 12 requests
  (11 × `n=4`, 1 × `n=32`) collected after the `/tokenize` per-choice
  counting landed, so every row carries `tokens_predicted` plus the
  `content` / `reasoning` / `tool_args` split. Reasoning tokens are
  ~76% of total tokens per choice on average — confirms the bulk of
  thinking-mode emission lives in `reasoning_content`. 39/76 rows
  compiled (51%); MCC spans `-1.00 – +0.64`, median `+0.33`. Mean MCC
  is lower than `res.jsonl`'s n=4 subset by ~0.10–0.18 with marginal
  statistical significance (Mann-Whitney p ≈ 0.03), but the new data is
  a tight 28-hour collection block while `res.jsonl` is spread across
  ~4 days — so time-correlated environmental noise (server uptime,
  RNG continuation, thermals) is a plausible confound. Don't read the
  gap as a model-quality change from the build that introduced
  tokenize-counting; `/tokenize` runs after the chat response and
  cannot affect generation. Includes one MCC=-1.0 row (the new
  near-inverted floor, beneath res.jsonl's -0.80).

- `results/res_reasoning_off.jsonl` — 300 rows, first quant-cross-section
  with reasoning disabled (Qwen3 nothink path: `tokens_reasoning == 0` on
  every row, all output goes to `tokens_tool_args`). 100 rows per quant at
  `Q8_K_XL` / `Q6_K_XL` / `Q5_K_XL`, mixed `n=4` and `n=32` requests, same
  build/sampling as the other datasets. Compile-error mass climbs
  monotonically as precision drops: Q8 78%, Q6 84%, Q5 87% of completions
  hit `compile_error` before reaching the corpus. Conditional on compiling,
  the upper MCC tail tops out around `+0.65` across all three quants —
  the differences between quants in `[0, 0.65]` are noisy at n=100. Useful
  as the reasoning-off counterpart to `res_reasoning_on.jsonl` (which is
  Q8 only) and as the first dataset where multiple quants live in one
  file. See `plot_mcc_cdf.py` for the per-quant ECDF view.

## Plots

Ad-hoc analysis scripts live next to `run.py` and write PNGs into
`results/` (gitignored). Currently:

- `plot_mcc_cdf.py` — per-quant ECDF of MCC for one results JSONL,
  treating compile-error rows as MCC = -1 so compile rate and corpus
  quality summarize on a single curve. Defaults to
  `results/res_reasoning_off.jsonl` → `results/mcc_cdf.png`; override via
  `--input`/`--output`/`--title`. Pools rows on the `model` field, so
  new rows of an existing quant fold into the same curve on rerun
  without code changes.

## Adding a task

The same (spec, env) decomposition validation-bench uses applies here.
To add e.g. `yaml-1.2` C++17 support:

1. Copy `data/specs/yaml-1.2/` from validation-bench (or wherever — this
   project doesn't run `setup.sh`; whatever ends up under
   `data/specs/<name>/tests/` plus a matching `tests.jsonl` is enough).
2. Create `data/tasks/yaml-1.2-cpp17/{task.json, preamble.md}` —
   `task.json` is `{"spec": "yaml-1.2", "env": "cpp17"}`, `preamble.md`
   inherits the single-shot wording from `toml-1.0-cpp17`.
3. Run with `--task yaml-1.2-cpp17`.
