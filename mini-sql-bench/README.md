# mini-sql-bench

A one-task smoke harness on top of [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent).

This is **not** a benchmark — there's no leaderboard and no model ranking. Its only job is
to confirm that the mini-swe-agent agentic harness drives a real, Docker-isolated task
end-to-end **against whatever model provider you point it at**. A correct answer proves the
whole pipeline works — model wiring, the step loop, the sandbox, and the submit contract —
not that the model is good at SQL (the task is trivial and deterministic).

The task: a pre-baked SQLite database ships inside the container; the agent inspects it,
writes a query, and reports the answer.

## How it works

- `run.py` builds the Docker image, constructs a mini-swe-agent model + docker environment
  from `config.yaml`, runs `DefaultAgent` on the single task, then scores the answer.
- The agent works one bash command at a time inside the container. It finishes by running
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /workspace/answer.txt` — mini-swe-agent's
  `DockerEnvironment` recognizes that magic first line and ends the run.
- Scoring reads `/workspace/answer.txt` straight out of the container (falling back to the
  submission text) and compares it to `task.expected` in `config.yaml`.
- Isolation matches ProgramBench's approach: `--network none --cpus 2 --memory 2g
  --user agent --cap-drop SYS_PTRACE`. The model API call happens host-side, so the
  container needs no network.

## Where each step runs (host vs. container)

There's a strict split: the **model API call is the only thing that happens on the host**,
and **every command the model chooses runs inside the sandbox container** — which is started
with `--network none`, so it can't reach the network or the model. The loop alternates
between the two sides until the model submits.

Below is an actual run against the local Qwen3.6-27B server (4 model calls). Each row is one
action, in order; exactly one side is active.

| #  | host                                                                   | container                                                                       |
|----|------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| 1  | `docker build` → image with the SQLite DB baked in                     |                                                                                 |
| 2  | start sandbox: `docker run -d … mini-sql-bench:task sleep 1h`           |                                                                                 |
| 3  | **call model** (turn 1) → returns a `bash` tool call                   |                                                                                 |
| 4  |                                                                        | `sqlite3 /workspace/company.db ".schema"`                                       |
| 5  | feed output back, **call model** (turn 2) → tool call                  |                                                                                 |
| 6  |                                                                        | `sqlite3 … "SELECT * FROM departments;"`                                         |
| 7  | feed output back, **call model** (turn 3) → tool call                  |                                                                                 |
| 8  |                                                                        | `sqlite3 … "SELECT SUM(e.salary) … WHERE d.name='Engineering';"` → `405000`      |
| 9  | feed output back, **call model** (turn 4) → submit tool call           |                                                                                 |
| 10 |                                                                        | `echo 405000 > answer.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat answer.txt` |
| 11 | env sees `COMPLETE_…` as the first output line → raises `Submitted`, loop ends |                                                                          |
| 12 |                                                                        | score read: `cat /workspace/answer.txt` → `405000`                              |
| 13 | normalize & compare to `expected`; write `results/<ts>.{result,traj}.json` |                                                                             |

Reading top-to-bottom, host and container strictly alternate through the loop (rows 3–10):
the host calls the model, the container runs whatever the model picked. That alternation —
model on the host, its chosen commands sandboxed and offline — is exactly the isolation the
smoke test exercises.

## Setup

```bash
python -m pip install -r requirements.txt   # mini-swe-agent (PyPI) + pyyaml
docker build -t mini-sql-bench:task .        # also done automatically by run.py
```

Docker must be running. The image is Debian-based on purpose (Alpine's busybox `timeout`
misbehaves under the mini-swe-agent wrapper — see the repo CLAUDE.md).

## Run — provider-agnostic

```bash
# Local OpenAI-compatible server (e.g. llama-server started with --jinja for tool calls)
python run.py --model openai/qwen3-coder --base-url http://localhost:8080/v1

# Anthropic (reads ANTHROPIC_API_KEY from the environment)
python run.py --model anthropic/claude-sonnet-4-6

# OpenAI (reads OPENAI_API_KEY)
python run.py --model openai/gpt-4o
```

The model name is a [LiteLLM](https://docs.litellm.ai/) name (`provider/model`). For a
local / 3rd-party OpenAI-compatible endpoint, pass `--base-url`; the runner then sets an
explicit (dummy by default) API key so it never silently falls back to `OPENAI_API_KEY` for
a non-OpenAI server. The model must support tool calls — for llama-server that means
starting it with `--jinja` and a tool-capable chat template.

Useful flags: `--no-build` (reuse an existing image), `--step-limit`, `--cost-limit`,
`--temperature`, `--api-key`, `--note`.

## Output

Each run writes two files under `results/` (gitignored):

- `<ts>.result.json` — `{model, base_url, exit_status, submission, answer, expected,
  correct, cost, n_calls, ...}`
- `<ts>.traj.json` — the full mini-swe-agent trajectory (inspect with `mini-extra inspector`)

`run.py` exits 0 when `correct` is true, 1 otherwise. The console prints a one-line summary.

## The task & changing it

`seed.sql` is the single source of truth for the data and is baked into the image at build
time. The question and the hand-computed answer both live in `config.yaml`'s `task` block.
**If you edit `seed.sql`, recompute and update `task.expected` to match** (the shipped
answer is `405000` — the total Engineering salary). Rebuild with `docker build` or just let
`run.py` rebuild.

To sanity-check the baked DB independent of any model:

```bash
docker run --rm mini-sql-bench:task \
  sqlite3 /workspace/company.db \
  "SELECT SUM(salary) FROM employees e JOIN departments d ON e.dept_id=d.id WHERE d.name='Engineering';"
# -> 405000
```
