#!/usr/bin/env python3
"""DeepSeek validation benchmark — dedicated script using the OpenAI SDK.

DeepSeek's API is OpenAI-compatible (nested tool shape, `{role:tool, ...}`
results) with two DeepSeek-specific bits that this script handles explicitly:

- **Thinking toggle** (V4+ models): `extra_body={"thinking":{"type":"enabled"|"disabled"}}`.
  Legacy `deepseek-chat` has no thinking support; legacy `deepseek-reasoner`
  has thinking permanently on. The `--mode` flag is ignored (no param sent)
  for both legacy names so passing `--mode thinking` with `deepseek-chat`
  doesn't surface a confusing server error.
- **`reasoning_content` field** on streaming deltas, sibling of `content`.
  Preserved across turns by default (matches DeepSeek's own requirement that
  V4 thinking echoes `reasoning_content` back on tool-result turns, and is
  symmetric with Moonshot's Preserved Thinking). `deepseek-reasoner` rejects
  echoed `reasoning_content` with HTTP 400, so preservation is force-disabled
  there — passing `--no-preserve-reasoning` has no effect on that model.

**Caveat**: `deepseek-reasoner` does not support function calling at all, so
it's effectively unusable for this benchmark (every attempt will infra-fail at
turn 0). Script runs it if asked but prints a warning at startup.
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


DEFAULT_API_BASE = "https://api.deepseek.com"

# Legacy pre-V4 model names where the thinking toggle doesn't apply.
# deepseek-chat has no thinking, deepseek-reasoner has it always on.
LEGACY_MODELS = {"deepseek-chat", "deepseek-reasoner"}
# Models where echoed reasoning_content on assistant history returns HTTP 400.
NO_REASONING_ECHO_MODELS = {"deepseek-reasoner"}


def deepseek_slug(model: str, mode: str) -> str:
    """deepseek-<model>[-<mode>]. Mode suffix is omitted for legacy models
    where thinking behavior is fixed by the model name."""
    base = f"deepseek-{model}" if not model.startswith("deepseek-") else model
    if model in LEGACY_MODELS:
        return base
    return f"{base}-{mode}"


def mode_to_extra_body(mode: str, model: str) -> dict | None:
    """Map (mode, model) to the DeepSeek `thinking` extra_body.

    Legacy models have fixed thinking behavior server-side; sending the param
    would either be ignored or rejected, so we skip it.
    """
    if model in LEGACY_MODELS:
        return None
    if mode == "instant":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "enabled"}}


def stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
    extra_body: dict | None,
    sampling_params: dict,
    tool_choice: str | None,
    preserve_reasoning: bool,
    turn: int,
) -> tuple[dict, str]:
    """Stream one turn. Returns (assistant_message_dict, finish_reason).

    Captures `reasoning_content` from deltas when `preserve_reasoning` is on —
    V4 thinking requires it echoed back across tool-result turns. Forced off
    for `deepseek-reasoner` (which rejects echoed reasoning_content).
    """
    kwargs = dict(sampling_params)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
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

        # DeepSeek surfaces reasoning on this field; the OpenAI SDK type
        # doesn't declare it, so hasattr/getattr per DeepSeek docs.
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
    if reasoning_parts and preserve_reasoning:
        msg["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg, finish_reason or "stop"


def run_attempt_deepseek(
    client: OpenAI,
    model: str,
    user_prompt: str,
    tests: list[dict],
    config: TaskConfig,
    max_turns: int,
    sampling_params: dict,
    extra_body: dict | None,
    tool_choice: str | None,
    preserve_reasoning: bool,
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
                extra_body=extra_body,
                sampling_params=sampling_params,
                tool_choice=tool_choice,
                preserve_reasoning=preserve_reasoning,
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
            # DeepSeek V4 thinking models occasionally emit a solution as
            # inline content instead of calling submit. One nudge per
            # attempt; the retry consumes a turn from the budget.
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
    parser = argparse.ArgumentParser(description="DeepSeek validation benchmark (OpenAI-compatible endpoint).")
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--model", required=True,
                        help="DeepSeek model ID (e.g. deepseek-v4-flash, deepseek-v4-pro, "
                             "deepseek-chat, deepseek-reasoner). "
                             "Note: `deepseek-reasoner` does not support function calling.")
    parser.add_argument("--mode", choices=["thinking", "instant"], default="thinking",
                        help="V4+ models only — sends `thinking={type:enabled|disabled}`. "
                             "Ignored for legacy deepseek-chat / deepseek-reasoner. Default: thinking.")
    parser.add_argument("--no-preserve-reasoning", dest="preserve_reasoning",
                        action="store_false", default=True,
                        help="Disable echoing `reasoning_content` back on subsequent turns. "
                             "V4 thinking REQUIRES preservation on tool-result turns — disable at "
                             "your own risk. Force-disabled for deepseek-reasoner regardless.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Omit for server default. Ignored by deepseek-reasoner.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling. Ignored by deepseek-reasoner.")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max output tokens. Default 32768. deepseek-reasoner max: 64000.")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--tool-choice", choices=["required", "auto", "none"], default=None,
                        help="Default: auto-derived — 'auto' when thinking is on (DeepSeek rejects "
                             "'required' + thinking, same as Moonshot), 'required' otherwise. Override "
                             "to force a specific value.")
    parser.add_argument("--timeout", type=float, default=600,
                        help="OpenAI client timeout (seconds) for DeepSeek API calls. Default: 600.")
    parser.add_argument("--docker-timeout", type=float, default=600,
                        help="Timeout (seconds) for `docker run` when starting the sandbox. Default: 600.")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_API_BASE),
                        help=f"API base URL. Default: {DEFAULT_API_BASE} (env: DEEPSEEK_BASE_URL). "
                             "Use `{DEFAULT_API_BASE}/beta` for strict-mode tools.")
    parser.add_argument("--api-key", default=None, help="Or set DEEPSEEK_API_KEY env var")
    parser.add_argument("--slug", default=None, help="Override derived slug")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default=None,
                        help="Attempt-log base dir (default: ~/.vb-data, env VB_DATA_DIR)")
    args = parser.parse_args()

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

    # Deliberately no OPENAI_API_KEY fallback — this script targets DeepSeek
    # and silently falling back would ship a real OpenAI key to api.deepseek.com.
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set (and --api-key not passed).", file=sys.stderr)
        sys.exit(1)

    if args.model == "deepseek-reasoner":
        print("Warning: deepseek-reasoner does not support function calling; every attempt "
              "will infra-fail at turn 0.", file=sys.stderr)

    # deepseek-reasoner rejects echoed reasoning_content; override user flag.
    preserve_reasoning = args.preserve_reasoning and args.model not in NO_REASONING_ECHO_MODELS

    extra_body = mode_to_extra_body(args.mode, args.model)

    sampling_params: dict = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        sampling_params["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p

    # Thinking is on when the model is deepseek-reasoner (always on server-side)
    # or a V4+ model with --mode thinking. DeepSeek rejects tool_choice=required
    # when thinking is on, so we fall back to "auto" by default in that case
    # and rely on the nudge loop for the occasional no-tool-call turn.
    thinking_on = (
        args.model == "deepseek-reasoner"
        or (args.model not in LEGACY_MODELS and args.mode == "thinking")
    )
    if args.tool_choice is None:
        tool_choice = "auto" if thinking_on else "required"
    else:
        tool_choice = None if args.tool_choice == "none" else args.tool_choice

    # Provenance fields written alongside sampling_params in each result row.
    effective_mode = "thinking" if args.model == "deepseek-reasoner" else args.mode
    results_params: dict = dict(sampling_params)
    if args.model not in LEGACY_MODELS:
        results_params["mode"] = effective_mode
    results_params["preserve_reasoning"] = preserve_reasoning

    slug = args.slug or deepseek_slug(args.model, effective_mode)
    results_base = Path(__file__).parent / args.results_dir
    results_file = results_base / "results.jsonl"
    data_dir_base = Path(
        args.data_dir or os.environ.get("VB_DATA_DIR", "") or Path.home() / ".vb-data"
    )
    data_dir_base.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = data_dir_base / "failures.jsonl"

    mode_str = effective_mode if args.model not in LEGACY_MODELS else "(fixed by model)"
    print(f"Running task '{args.task}' with DeepSeek model '{args.model}' "
          f"mode={mode_str}"
          + (" (preserve_reasoning)" if preserve_reasoning else ""))
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
            result, failure = run_attempt_deepseek(
                client=client,
                model=args.model,
                user_prompt=user_prompt,
                tests=tests,
                config=config,
                max_turns=args.max_turns,
                sampling_params=sampling_params,
                extra_body=extra_body,
                tool_choice=tool_choice,
                preserve_reasoning=preserve_reasoning,
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
