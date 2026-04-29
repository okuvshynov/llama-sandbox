# validation_bench

AI coding benchmark harness that evaluates models on code generation tasks via OpenAI-compatible API with tool calling.

## Setup

```bash
pip install -r requirements.txt
./setup.sh
docker build -t vb-sandbox-cpp17 envs/cpp17/   # C++17 tasks
docker build -t vb-sandbox-lua   envs/lua/     # Lua tasks
```

`setup.sh` clones [toml-test](https://github.com/toml-lang/toml-test) at a pinned commit into `.cache/toml-test`, generates `tests.jsonl` for each spec from the upstream file lists, and symlinks `specs/<spec>/tests/` to the cached corpus. Run it once after cloning the repo, or again after bumping the pinned commit.

## Layout

The benchmark is decomposed along two orthogonal axes:

- `specs/<spec>/` — what's being implemented (e.g. `toml-1.0`, `lua-5.4`). Owns the spec text embedded in the prompt body, the test corpus, and the `tests.jsonl` index.
- `envs/<env>/` — the implementation language family (e.g. `cpp17`, `lua`). Owns the docker image (Dockerfile + meta.json declaring `prepare_cmd`, `run_cmd`, source filename).
- `tasks/<spec>-<env>/` — a (spec, env) cell. Holds only `task.json` (a 2-key pointer to spec/env) and `preamble.md` (the small per-cell prose that combines them).

Add a new env (e.g. `cpp20`, `rust`, `go`):

1. `envs/<env>/Dockerfile` — base image + compiler/runtime
2. `envs/<env>/meta.json` — `language`, `docker_image`, `source_filename`, `prepare_cmd`, `run_cmd`, `compile_cmd`
3. For each spec the env should support, create `tasks/<spec>-<env>/{task.json, preamble.md}`. The composer wires up the rest.

Add a new spec (e.g. `json`, `yaml`):

1. `specs/<spec>/spec.md` — the reference text to embed in the prompt
2. `specs/<spec>/meta.json` — `display_name`, `has_spec_body`, `oracle`
3. Source the corpus into either `specs/<spec>/corpus/` (hand-curated, label-validated by `setup.sh`) or by extending `setup.sh`'s upstream-fetch path.
