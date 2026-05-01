# validation_bench

AI coding benchmark harness that evaluates models on code generation tasks via OpenAI-compatible API with tool calling.

## Setup

```bash
pip install -r requirements.txt
./setup.sh
docker build -t vb-sandbox-cpp17  data/envs/cpp17/   # C++17 tasks
docker build -t vb-sandbox-lua    data/envs/lua/     # Lua tasks
docker build -t vb-sandbox-erlang data/envs/erlang/  # Erlang/OTP tasks
docker build -t vb-sandbox-go     data/envs/go/      # Go 1.23 tasks
```

`setup.sh` populates the test corpora for the upstream-sourced specs:

- [toml-test](https://github.com/toml-lang/toml-test) → `data/specs/toml-{1.0,1.1}{,-nospec}/tests/` (pinned at `0ee318a`)
- [yaml-test-suite](https://github.com/yaml/yaml-test-suite) (`data` branch) → `data/specs/yaml-1.2{,-nospec}/tests/` (pinned at `6ad3d2c`); 1.3-only tests are filtered out

For each, it clones into `.cache/`, generates `tests.jsonl`, and symlinks `data/specs/<spec>/tests/` to the cached corpus. The hand-curated specs (`lua-5.4`, `palindrome`) re-derive labels via a configured oracle. Run it once after cloning the repo, or again after bumping a pinned commit.

## Layout

The benchmark is decomposed along two orthogonal axes:

- `data/specs/<spec>/` — what's being implemented (e.g. `toml-1.0`, `lua-5.4`, `yaml-1.2`). Owns the spec text embedded in the prompt body, the test corpus, and the `tests.jsonl` index.
- `data/envs/<env>/` — the implementation language family (e.g. `cpp17`, `lua`, `erlang`, `go`). Owns the docker image (Dockerfile + meta.json declaring `prepare_cmd`, `run_cmd`, source filename).
- `data/tasks/<spec>-<env>/` — a (spec, env) cell. Holds only `task.json` (a 2-key pointer to spec/env) and `preamble.md` (the small per-cell prose that combines them).

Add a new env (e.g. `cpp20`, `rust`, `go`):

1. `data/envs/<env>/Dockerfile` — base image + compiler/runtime
2. `data/envs/<env>/meta.json` — `language`, `docker_image`, `source_filename`, `prepare_cmd`, `run_cmd`, `compile_cmd`
3. For each spec the env should support, create `data/tasks/<spec>-<env>/{task.json, preamble.md}`. The composer wires up the rest.

Add a new spec (e.g. `json`, `yaml`):

1. `data/specs/<spec>/spec.md` — the reference text to embed in the prompt
2. `data/specs/<spec>/meta.json` — `display_name`, `has_spec_body`, `oracle`
3. Source the corpus into either `data/specs/<spec>/corpus/` (hand-curated, label-validated by `setup.sh`) or by extending `setup.sh`'s upstream-fetch path.

Other directories worth knowing about: `examples/` — a small museum of interesting model submissions surfaced by past runs (failure modes, recovery patterns, scoring quirks). Each subdirectory has the source the model produced plus a write-up of what's interesting about it.
