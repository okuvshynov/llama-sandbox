#!/usr/bin/env python3
"""Anthropic/Claude validation benchmark — dedicated script using the native SDK.

Separate from validation_bench.py's litellm-backed AnthropicProvider path (which
uses tool_choice={"type":"any"}, drops thinking blocks from the assistant message,
and never passes the `thinking` request parameter) so that extended thinking,
signed multi-turn thinking continuity, and the real native request shape are
exercised correctly. Reuses the provider-agnostic pieces (Sandbox, scoring,
result writing, attempt-id generation, log persistence) from validation_bench.py.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

from anthropic import Anthropic

from validation_bench import (
    Sandbox, Submission, AttemptResult, InfraFailure,
    COMPILE_CMD,
    handle_submit, format_tool_result, load_tests,
    make_attempt_id, save_attempt_log, _log,
)


# Anthropic tool shape: {name, description, input_schema} — differs from the
# OpenAI {type:function, function:{name, description, parameters}} shape used
# elsewhere in this repo. Kept local so the script is self-contained.
SUBMIT_TOOL_ANTHROPIC = {
    "name": "submit",
    "description": "Submit source code for compilation and testing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "Complete source code to compile and test.",
            }
        },
        "required": ["source_code"],
    },
}


def anthropic_slug(model: str, thinking_mode: str) -> str:
    """anthropic-<model>-<thinking_mode>.

    Prefixed with "anthropic-" to keep results rows distinct from the existing
    litellm-path rows (which use bare "claude-*" slugs). Model name is kept
    verbatim so the exact invoked model ID is visible in the slug.
    """
    return f"anthropic-{model}-{thinking_mode}"


def build_thinking_param(mode: str, budget: int) -> dict | None:
    """Map (mode, budget) to the Anthropic `thinking` request field.

    - disabled: {"type": "disabled"}
    - enabled:  {"type": "enabled", "budget_tokens": budget}   (manual; deprecated on 4.6, 400s on 4.7)
    - adaptive: {"type": "adaptive"}                            (model decides; 4.6+; mandatory on 4.7)
    """
    if mode == "disabled":
        return {"type": "disabled"}
    if mode == "enabled":
        return {"type": "enabled", "budget_tokens": budget}
    return {"type": "adaptive"}


def _blocks_to_dicts(blocks) -> list[dict]:
    """Serialize a list of SDK content blocks to plain JSON-able dicts.

    Pydantic objects -> dicts with exactly the fields the API expects when the
    blocks are sent back in a later turn (thinking.signature, tool_use.id,
    redacted_thinking.data, etc.).
    """
    return [b.model_dump(mode="json", exclude_none=True) for b in blocks]


def stream_completion(
    client: Anthropic,
    model: str,
    system: str | None,
    messages: list[dict],
    tools: list[dict],
    thinking: dict | None,
    tool_choice: dict,
    max_tokens: int,
    sampling_params: dict,
    turn: int,
) -> tuple[dict, str]:
    """Stream one turn. Returns (assistant_message_dict, stop_reason).

    The returned assistant dict has Anthropic's native shape:
        {"role": "assistant", "content": [ ...blocks... ]}
    with thinking / redacted_thinking / text / tool_use blocks all preserved —
    required for multi-turn continuity with tool use under extended thinking.
    """
    kwargs = dict(sampling_params)
    if thinking is not None:
        kwargs["thinking"] = thinking
    if system is not None:
        kwargs["system"] = system

    chars = 0
    chunks_seen = 0
    last_log = time.time()

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    ) as stream:
        for event in stream:
            chunks_seen += 1
            etype = getattr(event, "type", None)
            if etype == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", None)
                if dtype == "text_delta":
                    chars += len(delta.text)
                elif dtype == "thinking_delta":
                    chars += len(getattr(delta, "thinking", ""))
                elif dtype == "input_json_delta":
                    chars += len(getattr(delta, "partial_json", ""))
            now = time.time()
            if now - last_log >= 5:
                _log(f"  turn {turn}: streaming... {chars} chars, {chunks_seen} chunks")
                last_log = now
        final = stream.get_final_message()

    msg_dict = {"role": "assistant", "content": _blocks_to_dicts(final.content)}
    stop_reason = final.stop_reason or "end_turn"
    return msg_dict, stop_reason


def run_attempt_anthropic(
    client: Anthropic,
    model: str,
    system: str | None,
    user_prompt: str,
    tests: list[dict],
    max_turns: int,
    max_tokens: int,
    sampling_params: dict,
    thinking: dict | None,
    tool_choice: dict,
    attempt_dir: Path,
    task_dir: Path,
    attempt_id: str,
    docker_timeout: float = 600,
) -> tuple[AttemptResult | None, InfraFailure | None]:
    """Return (result, failure). Both can be non-None when an api_error hit
    mid-attempt after some submissions were already graded — we keep the
    per-turn rows from those submissions instead of discarding everything.
    """
    sandbox = Sandbox(startup_timeout=docker_timeout)
    sandbox.start()

    attempt_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = attempt_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    # First user message uses a content-block list so we can attach cache_control
    # to the stable task prompt prefix (cheap, high-leverage cache hit on retries).
    messages: list[dict] = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": user_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
    }]

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
            assistant_msg, stop_reason = stream_completion(
                client=client,
                model=model,
                system=system,
                messages=messages,
                tools=[SUBMIT_TOOL_ANTHROPIC],
                thinking=thinking,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                sampling_params=sampling_params,
                turn=turn,
            )
        except Exception as e:
            _log(f"  API error on turn {turn}: {e}")
            api_error = e
            error_turn = turn
            break

        messages.append(assistant_msg)
        flush_log()

        if stop_reason == "max_tokens":
            _log(f"  turn {turn}: response truncated (max_tokens too low)")

        tool_use_blocks = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            # Same nudge pattern as the moonshot script: when thinking is on, we
            # can only pass tool_choice={"type":"auto"}, so the model can opt out
            # of tool use. One nudge per attempt; the retry consumes a turn.
            if not nudged:
                _log(f"  turn {turn}: no tool_use — nudging model to use submit")
                messages.append({
                    "role": "user",
                    "content": "You responded without using the `submit` tool. Please use `submit` now with your complete C++ source code.",
                })
                flush_log()
                nudged = True
                continue
            _log(f"  turn {turn}: no tool_use after nudge — ending attempt")
            break

        # All tool_result blocks for this turn go into a single user message,
        # matching Anthropic's "one user turn carries all tool results" shape.
        tool_result_blocks: list[dict] = []
        for block in tool_use_blocks:
            tu_id = block["id"]
            name = block["name"]
            input_data = block.get("input", {}) or {}

            if name != "submit":
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": f"Unknown tool: {name}. Use the `submit` tool.",
                    "is_error": True,
                })
                continue

            source_code = input_data.get("source_code")
            if not isinstance(source_code, str):
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": "Invalid tool arguments: pass source_code as a string.",
                    "is_error": True,
                })
                continue

            submission_count += 1
            sub_dir = submissions_dir / str(submission_count)
            sub_dir.mkdir()
            (sub_dir / "solution.cpp").write_text(source_code)

            result = handle_submit(source_code, tests, sandbox, task_dir)
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

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu_id,
                "content": tool_result_str,
            })

        messages.append({"role": "user", "content": tool_result_blocks})
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
        # No rows worth keeping — prefer the api_error cause if we have one,
        # otherwise record the "model never submitted" cause.
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
    parser = argparse.ArgumentParser(description="Anthropic/Claude validation benchmark (native SDK).")
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude model ID (default: claude-sonnet-4-6)")
    parser.add_argument("--thinking", choices=["adaptive", "enabled", "disabled"], default="adaptive",
                        help="Thinking mode. adaptive: model decides (4.6+, mandatory on 4.7). "
                             "enabled: manual --thinking-budget. disabled: off. Default: adaptive.")
    parser.add_argument("--thinking-budget", type=int, default=10000,
                        help="budget_tokens for --thinking enabled (>=1024, must be < --max-tokens). Default: 10000.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Default: 1.0.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling. Anthropic recommends using either temperature or top_p, not both.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling (advanced).")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600,
                        help="Anthropic client timeout (seconds) for API calls. Default: 600.")
    parser.add_argument("--docker-timeout", type=float, default=600,
                        help="Timeout (seconds) for `docker run` when starting the sandbox container. Default: 600.")
    parser.add_argument("--api-key", default=None, help="Or set ANTHROPIC_API_KEY env var")
    parser.add_argument("--slug", default=None, help="Override derived slug")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default=None,
                        help="Attempt-log base dir (default: ~/.vb-data, env VB_DATA_DIR)")
    args = parser.parse_args()

    tasks_dir = Path(__file__).parent / "tasks" / args.task
    if not tasks_dir.is_dir():
        print(f"Error: task directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)
    prompt_file = tasks_dir / "prompt.txt"
    tests_file = tasks_dir / "tests.jsonl"
    for f in [prompt_file, tests_file]:
        if not f.exists():
            print(f"Error: missing file: {f}", file=sys.stderr)
            sys.exit(1)
    user_prompt = prompt_file.read_text().replace("{compile_cmd}", COMPILE_CMD)
    tests = load_tests(tests_file)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set (and --api-key not passed).", file=sys.stderr)
        sys.exit(1)

    thinking = build_thinking_param(args.thinking, args.thinking_budget)
    # Only tool_choice "auto"/"none" are accepted when thinking is on; when off
    # we force tool use with {"type":"any"} (symmetric with moonshot's "required"
    # in instant mode).
    thinking_enabled = args.thinking != "disabled"
    tool_choice = {"type": "auto"} if thinking_enabled else {"type": "any"}

    sampling_params: dict = {"temperature": args.temperature}
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k

    # Provenance fields written into each result row's sampling_params.
    results_params: dict = {
        **sampling_params,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
    }
    if args.thinking == "enabled":
        results_params["thinking_budget"] = args.thinking_budget

    slug = args.slug or anthropic_slug(args.model, args.thinking)
    results_base = Path(__file__).parent / args.results_dir
    results_file = results_base / "results.jsonl"
    data_dir_base = Path(
        args.data_dir or os.environ.get("VB_DATA_DIR", "") or Path.home() / ".vb-data"
    )
    data_dir_base.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = data_dir_base / "failures.jsonl"

    mode_suffix = ""
    if args.thinking == "enabled":
        mode_suffix = f" budget={args.thinking_budget}"
    print(f"Running task '{args.task}' with Anthropic model '{args.model}' "
          f"thinking={args.thinking}{mode_suffix}")
    sampling_str = ", ".join(f"{k}={v}" for k, v in results_params.items())
    print(f"Attempts: {args.n_attempts} | Max turns: {args.max_turns} | Sampling: {sampling_str}")
    print(f"Debug logs base: {data_dir_base}")
    print(f"Results: {results_file}")
    print("-" * 60)

    client = Anthropic(api_key=api_key, timeout=args.timeout)

    def save_result(r: AttemptResult):
        base = {
            "task": args.task,
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
            result, failure = run_attempt_anthropic(
                client=client,
                model=args.model,
                system=None,
                user_prompt=user_prompt,
                tests=tests,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                sampling_params=sampling_params,
                thinking=thinking,
                tool_choice=tool_choice,
                attempt_dir=attempt_dir,
                task_dir=tasks_dir,
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
