#!/usr/bin/env python3
"""Local llama.cpp server validation benchmark — dedicated script using the openai SDK.

Targets OpenAI-compatible endpoints exposed by llama.cpp's built-in server (and,
by extension, any other OpenAI-compatible local runner — vllm, lmstudio, etc.).
Sampling parameters default to "not sent" so the server's configured defaults
win on anything not explicitly overridden at the CLI. Preserves reasoning_content
when the server emits it (e.g., Qwen3 / DeepSeek R1 thinking variants), so
multi-turn contexts carry the model's prior reasoning.
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

    normalize_openai_chat_usage, normalize_llama_cpp_timings,
    derive_slug, auto_detect_model,
)
from composer import load_task, spec_dir


DEFAULT_API_BASE = "http://localhost:8080/v1"


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    sampling_params: dict,
    tool_choice: str | None,
    turn: int,
) -> tuple[dict, str, dict | None]:
    """Stream one turn. Returns (assistant_message_dict, finish_reason).

    Preserves reasoning_content on the returned assistant dict when the server
    emits it — sent back on subsequent turns so thinking-capable local models
    (R1/Qwen3-thinking etc.) see their own prior reasoning.
    """
    kwargs = dict(sampling_params)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason: str | None = None
    last_usage = None
    last_timings: dict | None = None
    chars = 0
    chunks_seen = 0
    last_log = time.time()

    for chunk in stream:
        chunks_seen += 1
        if getattr(chunk, "usage", None) is not None:
            last_usage = chunk.usage
        # llama.cpp's server emits a non-OpenAI `timings` block on the response
        # (one per chunk on streaming). Pydantic v2 captures unknown fields in
        # model_extra when the model is configured with extra="allow"; the
        # OpenAI SDK uses that, so chunk.timings shows up there.
        extra = getattr(chunk, "model_extra", None) or {}
        if extra.get("timings"):
            last_timings = extra["timings"]
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

        # Some local servers surface reasoning on a non-standard field; the
        # OpenAI SDK type doesn't declare it, so hasattr/getattr.
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
    # Prefer llama.cpp's timings block (its canonical shape) over OpenAI-style
    # usage; fall back to the latter for servers that don't emit timings.
    usage = (normalize_llama_cpp_timings(last_timings)
             or normalize_openai_chat_usage(last_usage))
    return msg, finish_reason or "stop", usage


def run_attempt_llama_cpp(
    client: OpenAI,
    model: str,
    user_prompt: str,
    tests: list[dict],
    config: TaskConfig,
    max_turns: int,
    sampling_params: dict,
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

    # Persist attempt logs from turn 0 so a killed/mid-flight local run
    # leaves an inspectable transcript on disk.
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
        t_model = time.perf_counter()
        try:
            assistant_msg, finish_reason, turn_usage = stream_completion(
                client=client,
                model=model,
                messages=messages,
                tools=[SUBMIT_TOOL],
                sampling_params=sampling_params,
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

        usage_to_attach: dict | None = turn_usage
        model_seconds_to_attach: float | None = time.perf_counter() - t_model

        if finish_reason == "length":
            _log(f"  turn {turn}: response truncated (max_tokens too low)")

        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            # Local models (especially thinking-variant GGUFs like Qwen3 or R1
            # distills) sometimes emit the full solution as content instead of
            # calling submit. Same one-nudge-per-attempt pattern as the other
            # extracted scripts; the retry consumes a turn from the budget.
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
                submission_results.append(Submission(
                    turn=turn, matrix=result.matrix,
                    usage=usage_to_attach,
                    model_seconds=model_seconds_to_attach,
                    prepare_seconds=result.prepare_seconds,
                    tests_seconds=result.tests_seconds,
                ))
                m = result.matrix
                status = f"{m.passed}/{m.total} (TP={m.tp} FN={m.fn} FP={m.fp} TN={m.tn}) MCC={m.mcc:.3f}"
            else:
                error = "compile_timeout" if "timed out" in result.compiler_output else "compile_error"
                submission_results.append(Submission(
                    turn=turn, error=error,
                    usage=usage_to_attach,
                    model_seconds=model_seconds_to_attach,
                    prepare_seconds=result.prepare_seconds,
                    tests_seconds=result.tests_seconds,
                ))
                status = error.upper()

            _log(f"  turn {turn}, submission {submission_count}: {status}")
            usage_to_attach = None  # consumed; same API call backs all submissions in this turn
            model_seconds_to_attach = None

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
    parser = argparse.ArgumentParser(
        description="Local llama.cpp server validation benchmark (OpenAI-compatible)."
    )
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--model", default=None,
                        help="Model ID. Omit to auto-detect from the server's /models endpoint.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE,
                        help=f"OpenAI-compatible base URL. Default: {DEFAULT_API_BASE} (llama.cpp default).")
    parser.add_argument("--api-key", default=None,
                        help="API key. Most local servers ignore the value; a fixed placeholder "
                             "('local-placeholder') is used when this flag is omitted. "
                             "Intentionally does NOT read OPENAI_API_KEY — we don't want a real "
                             "OpenAI key mis-sent to a local/third-party endpoint.")
    # Sampling params default to None — when unset we don't send them, so the
    # server's launch-time config wins. Pass an explicit value to override.
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Omit to use the server's configured value.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling. Omit to use the server's configured value.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling. Omit to use the server's configured value.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per response. Omit to use the server's configured n_predict.")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--tool-choice", choices=["required", "auto", "none"], default="required",
                        help="tool_choice to send. Default: required (forces a submit call). "
                             "Use 'auto' if the local server rejects 'required'; 'none' to omit entirely.")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="OpenAI client timeout (seconds). Default: 3600. Bump to 86400+ for slow local models.")
    parser.add_argument("--docker-timeout", type=float, default=600,
                        help="Timeout (seconds) for `docker run` when starting the sandbox container. Default: 600.")
    parser.add_argument("--slug", default=None, help="Override the auto-derived results slug")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default=None,
                        help="Attempt-log base dir (default: ~/.vb-data, env VB_DATA_DIR)")
    args = parser.parse_args()

    tasks_dir = Path(__file__).parent / "data" / "tasks" / args.task
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

    # Deliberately no OPENAI_API_KEY env fallback: this script targets arbitrary
    # OpenAI-compatible endpoints (local llama.cpp, vllm, lmstudio, etc.), and
    # silently falling back would ship a real OpenAI key to whichever server the
    # user points at. Pass --api-key explicitly if the endpoint needs auth.
    api_key = args.api_key or "local-placeholder"

    model = args.model or auto_detect_model(args.api_base, api_key)

    # Build sampling_params with only user-provided overrides; server config
    # wins on anything omitted. Recorded verbatim into the results row so
    # provenance is honest about what was actually sent.
    sampling_params: dict = {}
    if args.temperature is not None:
        sampling_params["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    if args.max_tokens is not None:
        sampling_params["max_tokens"] = args.max_tokens

    tool_choice = None if args.tool_choice == "none" else args.tool_choice

    slug = args.slug or derive_slug(model)
    results_base = Path(__file__).parent / args.results_dir
    results_file = results_base / "results.jsonl"
    data_dir_base = Path(
        args.data_dir or os.environ.get("VB_DATA_DIR", "") or Path.home() / ".vb-data"
    )
    data_dir_base.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = data_dir_base / "failures.jsonl"

    print(f"Running task '{args.task}' against '{args.api_base}' "
          f"with model '{model}' (slug={slug})")
    sampling_str = (", ".join(f"{k}={v}" for k, v in sampling_params.items())
                    if sampling_params else "server config")
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
            "sampling_params": sampling_params,
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
                if s.usage is not None:
                    row.update(s.usage)
                if s.model_seconds is not None:
                    row["model_seconds"] = round(s.model_seconds, 3)
                if s.prepare_seconds is not None:
                    row["prepare_seconds"] = round(s.prepare_seconds, 3)
                if s.tests_seconds is not None:
                    row["tests_seconds"] = round(s.tests_seconds, 3)
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
            result, failure = run_attempt_llama_cpp(
                client=client,
                model=model,
                user_prompt=user_prompt,
                tests=tests,
                config=config,
                max_turns=args.max_turns,
                sampling_params=sampling_params,
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
