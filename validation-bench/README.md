# validation_bench

AI coding benchmark harness that evaluates models on code generation tasks via OpenAI-compatible API with tool calling.

## Setup

```bash
pip install -r requirements.txt
./setup.sh
docker build -t vb-sandbox .
```

`setup.sh` clones [toml-test](https://github.com/toml-lang/toml-test) at a pinned commit into `.cache/toml-test`, generates `tests.jsonl` for each task from the upstream file lists, and symlinks task test data into the cache. Run it once after cloning the repo, or again after bumping the pinned commit.
