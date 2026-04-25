# validation_bench

AI coding benchmark harness that evaluates models on code generation tasks via OpenAI-compatible API with tool calling.

## Setup

```bash
pip install -r requirements.txt
./setup.sh
docker build -t vb-sandbox -f Dockerfile .          # C++ tasks
docker build -t vb-sandbox-lua -f Dockerfile.lua .  # Lua tasks
```

`setup.sh` clones [toml-test](https://github.com/toml-lang/toml-test) at a pinned commit into `.cache/toml-test`, generates `tests.jsonl` for each task from the upstream file lists, and symlinks task test data into the cache. Run it once after cloning the repo, or again after bumping the pinned commit.

Each task directory has a `task.json` declaring its language, docker image, source filename, and prepare/run commands. Add a new language by writing a `Dockerfile.<lang>`, building its image, and creating a `tasks/<name>/task.json` that points at it.
