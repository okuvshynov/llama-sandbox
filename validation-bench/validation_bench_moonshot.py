#!/usr/bin/env python3
"""Moonshot/Kimi (K2.5 / K2.6 / K2-Thinking) validation benchmark — native API.

Separate from validation_bench.py (which routes through litellm/anthropic) to keep
Moonshot's thinking semantics explicit: reasoning_content via hasattr/getattr,
temperature=1.0 for thinking per the benchmark-best-practice guide, and Preserved
Thinking via thinking.keep="all" (documented for K2.6 only — omitted on K2.5).
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


DEFAULT_API_BASE = "https://api.moonshot.ai/v1"


def moonshot_slug(model: str, mode: str) -> str:
    """moonshot-<model>-<mode>, or moonshot-kimi-k2-thinking (forced thinking)."""
    if model == "kimi-k2-thinking":
        return f"moonshot-{model}"
    return f"moonshot-{model}-{mode}"


def mode_to_extra_body(mode: str, preserve: bool, model: str) -> dict | None:
    """Map (mode, preserve, model) to the Moonshot-specific thinking extra_body.

    kimi-k2-thinking has thinking forcibly enabled server-side: pass nothing.
    thinking.keep="all" (Preserved Thinking) is documented for kimi-k2.6 only; it
    is omitted on kimi-k2.5 since the parameter is not documented there.
    """
    if model == "kimi-k2-thinking":
        return None
    if mode == "instant":
        return {"thinking": {"type": "disabled"}}
    body: dict = {"thinking": {"type": "enabled"}}
    if preserve and model == "kimi-k2.6":
        body["thinking"]["keep"] = "all"
    return body


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    extra_body: dict | None,
    sampling_params: dict,
    tool_choice: str | None,
    turn: int,
) -> tuple[dict, str]:
    """Stream one turn. Returns (assistant_message_dict, finish_reason).

    Preserves reasoning_content (required for Preserved Thinking in multi-turn
    tool use on K2.6).
    """
    kwargs = dict(sampling_params)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    # Moonshot rejects tool_choice="required" when thinking is enabled
    # (k2.6 with mode=thinking, or kimi-k2-thinking). Caller passes None in
    # that case; otherwise "required" forces a tool call in instant mode.
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

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

        # Moonshot ships reasoning on this field; the OpenAI SDK type doesn't
        # declare it, so hasattr/getattr per the Moonshot docs.
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


def run_attempt_moonshot(
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

    # Write logs directly into the attempt dir from turn 0 so a killed or
    # mid-flight run leaves an inspectable transcript on disk.
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
                extra_body=extra_body,
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

        if finish_reason == "length":
            _log(f"  turn {turn}: response truncated (max_tokens too low)")

        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            # K2.5/K2.6 in thinking mode occasionally emit the full solution
            # as inline content instead of calling submit. tool_choice cannot
            # be "required" with thinking enabled, so we nudge once and let
            # the retry consume a turn from the budget.
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
    parser = argparse.ArgumentParser(description="Moonshot/Kimi validation benchmark (native API).")
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--model", default="kimi-k2.6",
                        help="Kimi model (kimi-k2.5, kimi-k2.6, or kimi-k2-thinking; default kimi-k2.6)")
    parser.add_argument("--mode", choices=["thinking", "instant"], default="thinking",
                        help="K2.5/K2.6 only; ignored for kimi-k2-thinking (default: thinking)")
    parser.add_argument("--no-preserve-thinking", dest="preserve_thinking",
                        action="store_false", default=True,
                        help="Disable Preserved Thinking (thinking.keep=all). K2.6-only; has no effect on K2.5. Default: enabled.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature (default: 1.0 thinking / 0.6 instant)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max tokens (Moonshot benchmark guide: >= 16000)")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600,
                        help="OpenAI client timeout (seconds) for Moonshot API calls. Default: 600.")
    parser.add_argument("--docker-timeout", type=float, default=600,
                        help="Timeout (seconds) for `docker run` when starting the sandbox container. Default: 600.")
    parser.add_argument("--api-base", default=os.environ.get("MOONSHOT_BASE_URL", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=None, help="Or set MOONSHOT_API_KEY env var")
    parser.add_argument("--slug", default=None, help="Override derived slug")
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

    api_key = args.api_key or os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("Error: MOONSHOT_API_KEY not set (and --api-key not passed).", file=sys.stderr)
        sys.exit(1)

    if args.temperature is not None:
        temperature = args.temperature
    elif args.mode == "thinking":
        temperature = 1.0
    else:
        temperature = 0.6

    extra_body = mode_to_extra_body(args.mode, args.preserve_thinking, args.model)
    # Moonshot rejects tool_choice="required" when thinking is on; send it
    # only in instant mode where thinking is explicitly disabled.
    is_forced_thinking_model = args.model == "kimi-k2-thinking"
    thinking_enabled = is_forced_thinking_model or args.mode == "thinking"
    tool_choice = None if thinking_enabled else "required"

    sampling_params = {
        "max_tokens": args.max_tokens,
        "temperature": temperature,
        "top_p": args.top_p,
    }

    # Provenance fields written alongside sampling_params in each result row.
    effective_mode = "thinking" if is_forced_thinking_model else args.mode
    effective_preserve = (
        args.preserve_thinking and effective_mode == "thinking"
        and not is_forced_thinking_model and args.model == "kimi-k2.6"
    )
    results_params = {
        **sampling_params,
        "mode": effective_mode,
        "preserve_thinking": effective_preserve,
    }

    slug = args.slug or moonshot_slug(args.model, effective_mode)
    results_base = Path(__file__).parent / args.results_dir
    results_file = results_base / "results.jsonl"
    data_dir_base = Path(
        args.data_dir or os.environ.get("VB_DATA_DIR", "") or Path.home() / ".vb-data"
    )
    data_dir_base.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = data_dir_base / "failures.jsonl"

    print(f"Running task '{args.task}' with Moonshot model '{args.model}' "
          f"mode={effective_mode}"
          + (" (preserve)" if effective_preserve else ""))
    sampling_str = ", ".join(f"{k}={v}" for k, v in sampling_params.items())
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
            "model": args.model,
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
            result, failure = run_attempt_moonshot(
                client=client,
                model=args.model,
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
