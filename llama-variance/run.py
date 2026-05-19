#!/usr/bin/env python3
"""Single-shot variance study for local OpenAI-compatible servers.

One HTTP request with `n=N` chat completions, all sampled from the same
prompt at identical sampling params. Each choice's `submit` tool_call
yields one source file, which we compile + score against the spec's test
corpus sequentially. Every choice produces one JSONL row carrying its own
matrix + sampling params + server meta, so the resulting file is a
distribution over the score random variable.

This is *not* a benchmark for ranking models — it's a tool for staring
at the per-(sampling-param) score distribution and understanding how
much of a given model's apparent ability is signal vs. sample noise.

Why n=N (one HTTP request) rather than N independent requests:
  the server processes the prompt once and batches the N generations
  across slots. Cheaper than N separate calls, and identical sampling-
  param treatment for all N draws (no risk of one slot getting different
  defaults). See server-n-bench/bench_n.py for the same idea applied to
  pure throughput.

Tool-calling note: llama.cpp's server supports `n>1` together with
tool_choice="required", and each choice in the response carries its own
tool_calls array. We don't stream — non-streaming response shape is
trivial to parse when N choices come back at once.

Usage:
  python run.py --model qwen3-coder --n 16 --temperature 0.7 \\
      --jsonl results/sweep.jsonl

  # Sweep temperatures (shell-driven so each row is a fresh request):
  for t in 0.0 0.3 0.5 0.7 1.0; do
    for r in 1 2 3; do
      python run.py --model qwen3-coder --n 16 --temperature $t \\
          --jsonl results/sweep.jsonl
    done
  done
"""
import argparse
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import lib


DEFAULT_BASE = "http://127.0.0.1:8080"
USER_AGENT = "llama-variance/0.1"


def _http(method: str, base_url: str, path: str,
          body: dict | None = None, timeout: float = 600) -> dict:
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    ctx = ssl.create_default_context() if url.startswith("https://") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode())


def get_props(base_url: str) -> dict:
    """Pull model name, build_info, total_slots from /props."""
    return _http("GET", base_url, "/props")


def post_chat_completions(base_url: str, payload: dict,
                          timeout: float) -> dict:
    return _http("POST", base_url, "/v1/chat/completions",
                 body=payload, timeout=timeout)


def parse_choice(choice: dict) -> tuple[str | None, str | None, str | None]:
    """Return (source_code, finish_reason, error). On a clean submit tool_call,
    error is None. On any deviation (no tool_call, wrong tool name, bad JSON
    arguments, missing source_code), source_code is None and error is set."""
    msg = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None, finish_reason, "no_tool_call"
    tc = tool_calls[0]
    name = (tc.get("function") or {}).get("name")
    if name != "submit":
        return None, finish_reason, f"wrong_tool:{name}"
    raw_args = (tc.get("function") or {}).get("arguments") or ""
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return None, finish_reason, "bad_args_json"
    source_code = args.get("source_code")
    if not isinstance(source_code, str) or not source_code.strip():
        return None, finish_reason, "no_source_code"
    return source_code, finish_reason, None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="toml-1.0-cpp17",
                   help="Task directory under data/tasks/ (default: toml-1.0-cpp17)")
    p.add_argument("--base-url", default=DEFAULT_BASE,
                   help=f"OpenAI-compatible chat endpoint base (default {DEFAULT_BASE})")
    p.add_argument("--model", default=None,
                   help="Model id sent in the chat payload. If omitted, the "
                        "value reported by /props is used.")
    p.add_argument("--n", type=int, default=16,
                   help="Number of completions per request (OpenAI `n`). "
                        "Must be <= server -np. Default 16.")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Server-side sampling seed. Note: with the same seed "
                        "the N choices in one request will still differ "
                        "(the seed governs the RNG stream, not the per-slot "
                        "draw).")
    p.add_argument("--timeout", type=float, default=1800,
                   help="HTTP timeout for the chat call (seconds).")
    p.add_argument("--docker-timeout", type=float, default=600)
    p.add_argument("--jsonl", type=Path, default=None,
                   help="Append one row per completion to this JSONL file.")
    p.add_argument("--note", default=None,
                   help="Free-form tag stamped into each row's `note` field "
                        "(e.g. machine name, experiment label).")
    args = p.parse_args()

    config, prompt, tests_root = lib.load_task(args.task)
    tests_file = tests_root / "tests.jsonl"
    if not tests_file.exists():
        sys.exit(f"missing tests file: {tests_file}")
    tests = lib.load_tests(tests_file)

    print(f"task           : {args.task}   ({len(tests)} tests)")
    print(f"server         : {args.base_url}")

    try:
        props = get_props(args.base_url)
    except (urllib.error.URLError, socket.timeout) as e:
        sys.exit(f"failed to GET /props at {args.base_url}: {e}")
    server_model = props.get("model_path") or props.get("default_generation_settings", {}).get("model") or ""
    build_info = props.get("build_info", "")
    total_slots = props.get("total_slots", 0)
    model = args.model or server_model or "local-model"
    print(f"model          : {model}   (build {build_info}, {total_slots} slots)")

    if args.n > total_slots and total_slots > 0:
        sys.exit(f"--n={args.n} exceeds total_slots={total_slots}: restart "
                 f"server with -np >= {args.n}")

    # Build sampling-params dict — only include fields the user passed so
    # server defaults win on anything left blank.
    sampling: dict = {"max_tokens": args.max_tokens, "n": args.n}
    if args.temperature is not None: sampling["temperature"] = args.temperature
    if args.top_p is not None:       sampling["top_p"] = args.top_p
    if args.top_k is not None:       sampling["top_k"] = args.top_k
    if args.min_p is not None:       sampling["min_p"] = args.min_p
    if args.repeat_penalty is not None: sampling["repeat_penalty"] = args.repeat_penalty
    if args.seed is not None:        sampling["seed"] = args.seed

    print(f"sampling params: " + ", ".join(f"{k}={v}" for k, v in sampling.items()))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [lib.SUBMIT_TOOL],
        "tool_choice": "required",
        "stream": False,
        **sampling,
    }

    t_model = time.perf_counter()
    try:
        resp = post_chat_completions(args.base_url, payload, args.timeout)
    except (urllib.error.URLError, socket.timeout) as e:
        sys.exit(f"chat/completions request failed: {e}")
    model_seconds = time.perf_counter() - t_model

    choices = resp.get("choices") or []
    # llama.cpp's response carries a non-OpenAI `timings` block at the top
    # level; preserve it in the row metadata for prompt/cache breakdown.
    timings = resp.get("timings") or {}
    usage = resp.get("usage") or {}

    print(f"\n  model time     : {model_seconds:6.2f}s  (one request, n={args.n})")
    print(f"  choices        : {len(choices)}")
    if timings:
        print(f"  prompt_n       : {timings.get('prompt_n', '?')}  "
              f"cache_n={timings.get('cache_n', '?')}  "
              f"predicted_n={timings.get('predicted_n', '?')}")

    # Score each choice sequentially through one Sandbox. begin_submission
    # restarts the container between submissions to keep the pids cgroup
    # from saturating across the corpus runs.
    sandbox = lib.Sandbox(config=config, startup_timeout=args.docker_timeout)
    sandbox.start()

    ts = int(time.time())
    try:
        for i, choice in enumerate(choices):
            source_code, finish_reason, parse_err = parse_choice(choice)

            row: dict = {
                "ts": ts,
                "task": args.task,
                "spec": config.spec,
                "env": config.env,
                "base_url": args.base_url.rstrip("/"),
                "model": model,
                "build_info": build_info,
                "total_slots": total_slots,
                "completion_idx": i,
                "n_total": args.n,
                "sampling_params": sampling,
                "finish_reason": finish_reason,
                "model_seconds": round(model_seconds, 3),
            }
            if args.note is not None:
                row["note"] = args.note
            # Aggregate token counts apply to the whole request; we record
            # them only on completion 0 to avoid N-fold double-counting in
            # downstream sums.
            if i == 0:
                if usage:
                    row["usage"] = usage
                if timings:
                    row["timings"] = {
                        "prompt_n": timings.get("prompt_n"),
                        "cache_n": timings.get("cache_n"),
                        "predicted_n": timings.get("predicted_n"),
                    }

            if parse_err is not None:
                row["error"] = parse_err
                row["compiled"] = False
                print(f"  choice#{i:2d}  {parse_err}")
            else:
                t0 = time.perf_counter()
                result = lib.handle_submit(source_code, tests, sandbox, tests_root)
                wall = time.perf_counter() - t0
                row["compiled"] = result.compiled
                row["prepare_seconds"] = round(result.prepare_seconds, 3)
                row["tests_seconds"] = round(result.tests_seconds, 3)
                row["score_wall"] = round(wall, 3)
                if result.compiled:
                    m = result.matrix
                    row.update({
                        "tp": m.tp, "fn": m.fn, "fp": m.fp, "tn": m.tn,
                        "passed": m.passed, "total": m.total,
                        "mcc": round(m.mcc, 6),
                    })
                    print(f"  choice#{i:2d}  {m.passed}/{m.total}  "
                          f"MCC={m.mcc:+.4f}  prep={result.prepare_seconds:.2f}s  "
                          f"tests={result.tests_seconds:.2f}s")
                else:
                    row["error"] = ("compile_timeout"
                                    if "timed out" in result.compiler_output
                                    else "compile_error")
                    print(f"  choice#{i:2d}  {row['error']}  "
                          f"prep={result.prepare_seconds:.2f}s")

            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with open(args.jsonl, "a") as f:
                    f.write(json.dumps(row) + "\n")
    finally:
        sandbox.stop()

    print()


if __name__ == "__main__":
    main()
