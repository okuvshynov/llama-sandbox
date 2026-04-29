#!/usr/bin/env python3
"""Fireworks AI validation benchmark — dedicated script using the OpenAI SDK.

The `fireworks-ai` Python SDK is still alpha and auto-generated; the inference
path is just the OpenAI-compatible endpoint at
https://api.fireworks.ai/inference/v1. This script targets that endpoint
directly with the OpenAI SDK (same pattern as moonshot/llama_cpp here) and
plumbs the Fireworks-specific knobs through `extra_body`:

- reasoning_effort (low/medium/high) — mutually exclusive with `thinking`
- thinking={type:"enabled", budget_tokens} — Anthropic-style manual budget
- reasoning_history (disabled/interleaved/preserved) — multi-turn reasoning handling
- non-OpenAI sampling: top_k, min_p, repetition_penalty
- prompt_cache_key — KV-cache session affinity across runs

Preserves `reasoning_content` from streamed deltas on the assistant message so
multi-turn tool use on thinking models (Kimi-K2-Thinking, GLM-4.6/4.7,
GPT-OSS, Qwen3-thinking) carries prior reasoning when reasoning_history is on.
Reuses Sandbox / scoring / attempt-id / log-persistence from validation_bench_lib.py.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from validation_bench_lib import (
    Sandbox, Submission, AttemptResult, InfraFailure,
    SUBMIT_TOOL, TaskConfig, VB_VERSION,
    handle_submit, format_tool_result, load_tests,
    make_attempt_id, save_attempt_log, _log,
)
from composer import load_task, spec_dir


DEFAULT_API_BASE = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL_PREFIX = "accounts/fireworks/models/"


def normalize_model_id(model: str) -> str:
    """Accept either the full `accounts/fireworks/models/<name>` form or a bare
    `<name>`; return the canonical full form that Fireworks' API expects."""
    if model.startswith(FIREWORKS_MODEL_PREFIX) or "/" in model:
        return model
    return FIREWORKS_MODEL_PREFIX + model


def fireworks_slug(model: str, reasoning_effort: str | None, thinking_budget: int | None) -> str:
    """fireworks-<short>[-<effort>] or fireworks-<short>-think<budget>.

    Strip the `accounts/fireworks/models/` prefix so the slug mirrors the
    model's catalog name; fall back to the last path segment for custom
    account-hosted models.
    """
    short = model
    if short.startswith(FIREWORKS_MODEL_PREFIX):
        short = short[len(FIREWORKS_MODEL_PREFIX):]
    elif "/" in short:
        short = short.rsplit("/", 1)[1]
    base = f"fireworks-{short}"
    if reasoning_effort:
        return f"{base}-{reasoning_effort}"
    if thinking_budget is not None:
        return f"{base}-think{thinking_budget}"
    return base


def build_extra_body(
    reasoning_effort: str | None,
    thinking_budget: int | None,
    reasoning_history: str | None,
    top_k: int | None,
    min_p: float | None,
    repetition_penalty: float | None,
    prompt_cache_key: str | None,
) -> dict | None:
    """Assemble the Fireworks-specific extra_body payload. Returns None if
    nothing non-OpenAI needs to be sent."""
    body: dict = {}
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if thinking_budget is not None:
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    if reasoning_history is not None:
        body["reasoning_history"] = reasoning_history
    if top_k is not None:
        body["top_k"] = top_k
    if min_p is not None:
        body["min_p"] = min_p
    if repetition_penalty is not None:
        body["repetition_penalty"] = repetition_penalty
    if prompt_cache_key is not None:
        body["prompt_cache_key"] = prompt_cache_key
    return body or None


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    sampling_params: dict,
    extra_body: dict | None,
    tool_choice: str | None,
    turn: int,
) -> tuple[dict, str]:
    """Stream one turn. Returns (assistant_message_dict, finish_reason).

    Preserves reasoning_content when the server emits it (Kimi-K2-Thinking,
    GLM-4.6+, GPT-OSS, Qwen3-thinking all use this field). The legacy
    deepseek-r1 inlines thinking as <think>…</think> inside content — that
    passes through untouched.
    """
    kwargs = dict(sampling_params)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        **kwargs,
    )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    chars = 0
    chunks_seen = 0
    last_log = time.time()

    for chunk in stream:
        chunks_seen += 1
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        if delta is None:
            continue

        if getattr(delta, "content", None):
            content_parts.append(delta.content)
            chars += len(delta.content)

        # Fireworks surfaces reasoning on a non-standard field; the OpenAI SDK
        # type doesn't declare it, so hasattr/getattr.
        if hasattr(delta, "reasoning_content"):
            rc = getattr(delta, "reasoning_content")
            if rc:
                reasoning_parts.append(rc)
                chars += len(rc)

        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index
            slot = tool_calls.get(idx)
            if slot is None:
                slot = {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                tool_calls[idx] = slot
            if tc.id and not slot["id"]:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["function"]["name"] = tc.function.name
                if tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments
                    chars += len(tc.function.arguments)

        now = time.time()
        if now - last_log >= 5:
            _log(f"  turn {turn}: streaming... {chars} chars, {chunks_seen} chunks")
            last_log = now

    msg: dict = {"role": "assistant"}
    if content_parts:
        msg["content"] = "".join(content_parts)
    if reasoning_parts:
        msg["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg, finish_reason or "stop"


def run_attempt_fireworks(
    client: OpenAI,
    model: str,
    user_prompt: str,
    tests: list[dict],
    config: TaskConfig,
    max_turns: int,
    sampling_params: dict,
    extra_body: dict | None,
    tool_choice: str | None,
    attempt_dir: Path,
    tests_root: Path,
    attempt_id: str,
    docker_timeout: float = 600,
) -> tuple[AttemptResult | None, InfraFailure | None]:
    """Return (result, failure). Both can be non-None when an api_error hit
    mid-attempt after some submissions were already graded — we keep the
    per-turn rows from those submissions instead of discarding everything.
    """
    sandbox = Sandbox(config=config, startup_timeout=docker_timeout)
    sandbox.start()

    attempt_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = attempt_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    submission_count = 0
    submission_results: list[Submission] = []
    api_error: Exception | None = None
    error_turn = -1
    start = time.time()
    turn = 0
    nudged = False

    def flush_log():
        save_attempt_log(attempt_dir, messages)

    flush_log()

    for turn in range(max_turns):
        try:
            assistant_msg, finish_reason = stream_completion(
                client=client,
                model=model,
                messages=messages,
                tools=[SUBMIT_TOOL],
                sampling_params=sampling_params,
                extra_body=extra_body,
                tool_choice=tool_choice,
                turn=turn,
            )
        except Exception as e:
            _log(f"  API error on turn {turn}: {e}")
            api_error = e
            error_turn = turn
            break

        messages.append(assistant_msg)
        flush_log()

        if finish_reason == "length":
            _log(f"  turn {turn}: response truncated (max_tokens too low)")

        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            # With reasoning models on Fireworks, tool_choice="required" is
            # accepted but the model can still emit its full answer as
            # content instead of calling submit — especially when
            # reasoning_history is "preserved". One-nudge-per-attempt
            # recovery consumes a turn from the budget.
            if not nudged:
                _log(f"  turn {turn}: no tool_call — nudging model to use submit")
                messages.append({
                    "role": "user",
                    "content": f"You responded without calling the `submit` tool. Please call `submit` now with your complete {config.language} source code.",
                })
                flush_log()
                nudged = True
                continue
            _log(f"  turn {turn}: no tool_call after nudge — ending attempt")
            break

        for tool_call in tool_calls:
            tc_id = tool_call["id"]
            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]

            if name != "submit":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": f"Unknown tool: {name}. Use the `submit` tool.",
                })
                flush_log()
                continue

            try:
                args = json.loads(arguments)
                source_code = args["source_code"]
            except (json.JSONDecodeError, KeyError) as e:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": f"Invalid tool arguments: {e}. Pass source_code as a string.",
                })
                flush_log()
                continue

            submission_count += 1
            sub_dir = submissions_dir / str(submission_count)
            sub_dir.mkdir()
            (sub_dir / config.source_filename).write_text(source_code)

            result = handle_submit(source_code, tests, sandbox, tests_root)
            (sub_dir / "compiler.txt").write_text(result.compiler_output)
            if result.compiled:
                (sub_dir / "tests.txt").write_text(result.test_output)

            tool_result_str = format_tool_result(result)

            if result.compiled:
                submission_results.append(Submission(turn=turn, matrix=result.matrix))
                m = result.matrix
                status = f"{m.passed}/{m.total} (TP={m.tp} FN={m.fn} FP={m.fp} TN={m.tn}) MCC={m.mcc:.3f}"
            else:
                error = "compile_timeout" if "timed out" in result.compiler_output else "compile_error"
                submission_results.append(Submission(turn=turn, error=error))
                status = error.upper()

            _log(f"  turn {turn}, submission {submission_count}: {status}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": name,
                "content": tool_result_str,
            })
            flush_log()

        if submission_results and submission_results[-1].matrix:
            m = submission_results[-1].matrix
            if m.passed == m.total and m.total > 0:
                break

    elapsed = time.time() - start
    sandbox.stop()
    flush_log()

    failure: InfraFailure | None = None
    if api_error is not None:
        error_type = "timeout" if "timeout" in str(api_error).lower() else "api_error"
        failure = InfraFailure(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            turn=error_turn,
            error_type=error_type,
            error_message=str(api_error),
        )

    if submission_count == 0:
        if failure is None:
            failure = InfraFailure(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                turn=turn,
                error_type="no_submissions",
                error_message="Model completed without making any submissions",
            )
        return None, failure

    result = AttemptResult(
        attempt_id=attempt_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed, 1),
        submissions=submission_results,
    )
    return result, failure


def main():
    parser = argparse.ArgumentParser(description="Fireworks AI validation benchmark (OpenAI-compatible endpoint).")
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--model", required=True,
                        help="Fireworks model ID. Bare names are auto-prefixed with "
                             "'accounts/fireworks/models/' (e.g. 'glm-4p6' -> "
                             "'accounts/fireworks/models/glm-4p6').")
    # Reasoning controls — reasoning_effort and thinking-budget are mutually
    # exclusive server-side. Both go in extra_body; omit both for non-thinking
    # models or to accept Fireworks' defaults.
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                        help="Fireworks `reasoning_effort`. Mutually exclusive with --thinking-budget.")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Fireworks `thinking.budget_tokens` (>=1024). Mutually exclusive with "
                             "--reasoning-effort.")
    parser.add_argument("--reasoning-history", choices=["disabled", "interleaved", "preserved"],
                        default=None,
                        help="Fireworks multi-turn reasoning handling. Omit for server default.")
    # Standard sampling.
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Omit to use the server default.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling. Omit to use the server default.")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max tokens per response. Default: 32768. Bump for thinking budgets.")
    # Non-OpenAI sampling — accepted by Fireworks over the OpenAI-compat
    # endpoint; sent via extra_body since the SDK strips unknown top-level args.
    parser.add_argument("--top-k", type=int, default=None,
                        help="Fireworks `top_k` (non-OpenAI). 0 disables.")
    parser.add_argument("--min-p", type=float, default=None,
                        help="Fireworks `min_p` (non-OpenAI, 0..1).")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Fireworks `repetition_penalty` (non-OpenAI; 1.0 = off, 0..2).")
    parser.add_argument("--prompt-cache-key", default=None,
                        help="Fireworks `prompt_cache_key` for KV-cache session affinity across "
                             "calls (same key = same cache lane).")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--tool-choice", choices=["required", "auto", "none"], default="required",
                        help="Default: required. Fireworks accepts 'required' even with reasoning "
                             "models; switch to 'auto' if a model trips on it.")
    parser.add_argument("--timeout", type=float, default=600,
                        help="OpenAI client timeout (seconds) for Fireworks API calls. Default: 600.")
    parser.add_argument("--docker-timeout", type=float, default=600,
                        help="Timeout (seconds) for `docker run` when starting the sandbox container. Default: 600.")
    parser.add_argument("--api-base", default=os.environ.get("FIREWORKS_BASE_URL", DEFAULT_API_BASE),
                        help=f"API base URL. Default: {DEFAULT_API_BASE} (env: FIREWORKS_BASE_URL).")
    parser.add_argument("--api-key", default=None, help="Or set FIREWORKS_API_KEY env var")
    parser.add_argument("--slug", default=None, help="Override derived slug")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default=None,
                        help="Attempt-log base dir (default: ~/.vb-data, env VB_DATA_DIR)")
    args = parser.parse_args()

    if args.reasoning_effort is not None and args.thinking_budget is not None:
        print("Error: --reasoning-effort and --thinking-budget are mutually exclusive.",
              file=sys.stderr)
        sys.exit(2)

    tasks_dir = Path(__file__).parent / "tasks" / args.task
    if not tasks_dir.is_dir():
        print(f"Error: task directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)
    config, user_prompt = load_task(tasks_dir)
    tests_root = spec_dir(config.spec)
    tests_file = tests_root / "tests.jsonl"
    if not tests_file.exists():
        print(f"Error: missing file: {tests_file}", file=sys.stderr)
        sys.exit(1)
    tests = load_tests(tests_file)

    # Deliberately no OPENAI_API_KEY fallback: this script targets Fireworks
    # only; silently falling back would ship a real OpenAI key to Fireworks.
    api_key = args.api_key or os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set (and --api-key not passed).", file=sys.stderr)
        sys.exit(1)

    model = normalize_model_id(args.model)

    # OpenAI-native sampling goes in sampling_params; Fireworks-only knobs go in
    # extra_body so the OpenAI SDK doesn't strip them.
    sampling_params: dict = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        sampling_params["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p

    extra_body = build_extra_body(
        reasoning_effort=args.reasoning_effort,
        thinking_budget=args.thinking_budget,
        reasoning_history=args.reasoning_history,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        prompt_cache_key=args.prompt_cache_key,
    )

    tool_choice = None if args.tool_choice == "none" else args.tool_choice

    # Provenance: everything that went on the wire, merged into one dict on the
    # results row. extra_body fields are prefixed so they don't collide with
    # standard sampling when multiple providers are compared.
    results_params: dict = dict(sampling_params)
    if extra_body:
        for k, v in extra_body.items():
            results_params[k] = v

    slug = args.slug or fireworks_slug(model, args.reasoning_effort, args.thinking_budget)
    results_base = Path(__file__).parent / args.results_dir
    results_file = results_base / "results.jsonl"
    data_dir_base = Path(
        args.data_dir or os.environ.get("VB_DATA_DIR", "") or Path.home() / ".vb-data"
    )
    data_dir_base.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = data_dir_base / "failures.jsonl"

    mode_bits = []
    if args.reasoning_effort:
        mode_bits.append(f"reasoning_effort={args.reasoning_effort}")
    if args.thinking_budget is not None:
        mode_bits.append(f"thinking_budget={args.thinking_budget}")
    if args.reasoning_history:
        mode_bits.append(f"reasoning_history={args.reasoning_history}")
    mode_str = " ".join(mode_bits) if mode_bits else "no reasoning knobs"
    print(f"Running task '{args.task}' with Fireworks model '{model}' ({mode_str})")
    sampling_str = ", ".join(f"{k}={v}" for k, v in results_params.items())
    print(f"Attempts: {args.n_attempts} | Max turns: {args.max_turns} | Sampling: {sampling_str}")
    print(f"Debug logs base: {data_dir_base}")
    print(f"Results: {results_file}")
    print("-" * 60)

    client = OpenAI(api_key=api_key, base_url=args.api_base, timeout=args.timeout)

    def save_result(r: AttemptResult):
        base = {
            "vb_version": VB_VERSION,
            "task": args.task,
            "spec": config.spec,
            "env": config.env,
            "model": model,
            "slug": slug,
            "sampling_params": results_params,
            "attempt_id": r.attempt_id,
            "attempt_timestamp": r.timestamp,
            "attempt_elapsed_seconds": r.elapsed_seconds,
        }
        with open(results_file, "a") as f:
            for s in r.submissions:
                row = {**base, "turn": s.turn}
                if s.matrix is not None:
                    m = s.matrix
                    row.update({"tp": m.tp, "fn": m.fn, "fp": m.fp, "tn": m.tn,
                                "mcc": round(m.mcc, 6)})
                else:
                    row["error"] = s.error
                f.write(json.dumps(row) + "\n")

    def save_failure(fail: InfraFailure):
        with open(failures_file, "a") as f:
            f.write(json.dumps({
                "vb_version": VB_VERSION,
                "timestamp": fail.timestamp,
                "turn": fail.turn,
                "error_type": fail.error_type,
                "error_message": fail.error_message,
            }) + "\n")

    try:
        for i in range(args.n_attempts):
            _log(f"\n--- Attempt {i + 1}/{args.n_attempts} ---")
            attempt_id = make_attempt_id(args.task, slug)
            attempt_dir = data_dir_base / attempt_id
            _log(f"  Debug logs: {attempt_dir}")
            result, failure = run_attempt_fireworks(
                client=client,
                model=model,
                user_prompt=user_prompt,
                tests=tests,
                config=config,
                max_turns=args.max_turns,
                sampling_params=sampling_params,
                extra_body=extra_body,
                tool_choice=tool_choice,
                attempt_dir=attempt_dir,
                tests_root=tests_root,
                attempt_id=attempt_id,
                docker_timeout=args.docker_timeout,
            )
            if failure is not None:
                save_failure(failure)
                _log(f"  Infrastructure failure: {failure.error_type}: {failure.error_message}")
            if result is not None:
                save_result(result)
                _log(f"  [{result.attempt_id}] saved to {results_file}")
    except KeyboardInterrupt:
        print("\n\nInterrupted!")


if __name__ == "__main__":
    main()
