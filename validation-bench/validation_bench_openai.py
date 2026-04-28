#!/usr/bin/env python3
"""OpenAI validation benchmark — dedicated script using the official SDK.

Replaces the litellm-based codex/chat-completions routing in validation_bench.py
with an explicit branch on the model ID, so each API surface is visible in this
file instead of hidden behind litellm's translation layer.

Two paths:
- **Chat Completions** (non-codex: gpt-5, gpt-5.5, gpt-5-mini, o3, o4-mini, …)
  Uses `client.chat.completions.create(..., stream=True)`. `reasoning_effort`
  is flat, `max_tokens`, tool shape is nested `{type:function, function:{…}}`,
  tool results are `{role:"tool", tool_call_id, content}`. Reasoning traces
  are NOT surfaced on stream deltas — only the final assistant message has them.
- **Responses** (codex: gpt-5-codex, gpt-5.3-codex, …) — `gpt-5-codex` is
  Responses-only, others are accepted on both; this script routes any
  `"codex" in model.lower()` to Responses.
  Uses `client.responses.stream(...)`. `reasoning={"effort":…}` is nested,
  `max_output_tokens`, tool shape is flat `{type:function, name, parameters}`,
  tool results are `{type:function_call_output, call_id, output}`. Reasoning
  IS streamed (reasoning_summary_text.delta / reasoning_text.delta events).

Reasoning-item replay (Responses path only): the script requests
`include=["reasoning.encrypted_content"]` on each call and captures the
full reasoning items emitted on `response.output_item.done`. They're stored
on the assistant message under a custom `reasoning_items` key and echoed
back in the same order on subsequent turns so the model sees its own prior
chain of thought — critical for multi-turn tool use where the reasoning
*led to* the tool call. Encrypted content is passed back unchanged per
OpenAI's cookbook guidance; tampering with it triggers server-side
signature rejection.

The assistant message is normalized to OpenAI-chat shape internally so the
attempt loop, log persistence, and results format stay identical across paths.
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
from composer import load_task


# SUBMIT_TOOL (from validation_bench_lib) is the nested Chat Completions shape.
# Responses API wants it flat — no inner "function" key.
SUBMIT_TOOL_RESPONSES = {
    "type": "function",
    "name": "submit",
    "description": "Submit source code for compilation and testing.",
    "parameters": {
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


REASONING_EFFORT_CHOICES = ["none", "minimal", "low", "medium", "high", "xhigh"]


def is_codex_model(model: str) -> bool:
    """Codex models route to the Responses API. `gpt-5-codex` is Responses-only;
    later codex variants accept both, but we always route to Responses for them
    so reasoning traces are visible on stream."""
    return "codex" in model.lower()


def openai_slug(model: str, reasoning_effort: str | None) -> str:
    """`<model>[-<effort>]`. Model IDs are already friendly (gpt-5, gpt-5.3-codex,
    o4-mini) so no translation needed beyond the optional effort suffix."""
    return f"{model}-{reasoning_effort}" if reasoning_effort else model


# ---------- Chat Completions path (non-codex) ----------


def stream_completion_chat(
    client: OpenAI,
    model: str,
    messages: list[dict],
    sampling_params: dict,
    reasoning_effort: str | None,
    tool_choice: str | None,
    turn: int,
) -> tuple[dict, str]:
    """Stream one turn via Chat Completions. Returns (assistant_dict, finish_reason)."""
    kwargs = dict(sampling_params)
    # Reasoning-era models (gpt-5.x, o-series) reject the legacy `max_tokens`
    # with HTTP 400 and require `max_completion_tokens`. Always translate —
    # the new field has been accepted on all chat-completions-supporting
    # models since it was introduced, so there's no downside.
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[SUBMIT_TOOL],
        stream=True,
        **kwargs,
    )

    content_parts: list[str] = []
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
    if tool_calls:
        msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return msg, finish_reason or "stop"


# ---------- Responses API path (codex) ----------


def messages_to_responses_input(messages: list[dict]) -> list[dict]:
    """Translate OpenAI-chat-shape messages into Responses `input` items.

    Prior assistant turns containing tool_calls expand into one
    `function_call` item per call; role="tool" messages become
    `function_call_output` items keyed by call_id. Reasoning items captured
    from prior turns (stored on the assistant dict under `reasoning_items`)
    are emitted first in their original stream order — this matches the
    order Responses itself emits them in `response.output` and lets the
    model see its own prior chain of thought on the next turn.
    """
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            # Reasoning items replay first — OpenAI's cookbook says to pass
            # them back unchanged ("append response.output directly"); the
            # `encrypted_content` field is a signed blob, so we forward the
            # dict verbatim rather than re-minting fields.
            for item in m.get("reasoning_items", []) or []:
                items.append(item)
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls", []) or []:
                items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"] or "",
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m["tool_call_id"],
                "output": m["content"],
            })
    return items


def stream_completion_responses(
    client: OpenAI,
    model: str,
    messages: list[dict],
    sampling_params: dict,
    reasoning_effort: str | None,
    tool_choice: str | None,
    turn: int,
) -> tuple[dict, str]:
    """Stream one turn via the Responses API. Returns (assistant_dict in OpenAI
    chat shape, finish_reason) so the attempt loop stays provider-agnostic."""
    kwargs = dict(sampling_params)
    # Responses uses max_output_tokens; translate if the caller passed max_tokens.
    if "max_tokens" in kwargs:
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    # Request the signed reasoning blob so we can replay reasoning items on
    # subsequent turns. Works with both store=True and store=False; without
    # it the encrypted_content field is absent and replay fails signature
    # verification server-side.
    kwargs["include"] = ["reasoning.encrypted_content"]

    input_items = messages_to_responses_input(messages)

    text_parts: list[str] = []
    # Keyed by item_id since parallel tool calls interleave delta events across
    # different output_index values.
    tool_items: dict[str, dict] = {}
    tool_order: list[str] = []
    # Reasoning items captured on `output_item.done` — stored in original
    # stream order so replay preserves the reasoning -> tool_call relationship
    # Responses expects.
    reasoning_items: list[dict] = []
    chars = 0
    events_seen = 0
    last_log = time.time()

    with client.responses.stream(
        model=model,
        input=input_items,
        tools=[SUBMIT_TOOL_RESPONSES],
        **kwargs,
    ) as stream:
        for event in stream:
            events_seen += 1
            etype = getattr(event, "type", None)
            if etype == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    tool_items[item.id] = {
                        "id": item.call_id,
                        "type": "function",
                        "function": {"name": item.name or "", "arguments": ""},
                    }
                    tool_order.append(item.id)
            elif etype == "response.output_item.done":
                # Reasoning items are fully populated here (including the
                # signed `encrypted_content` when requested via `include`).
                # Serialize with exclude_none so we forward exactly what the
                # API sent — tampering via field mutation breaks the signature.
                item = event.item
                if getattr(item, "type", None) == "reasoning":
                    reasoning_items.append(item.model_dump(mode="json", exclude_none=True))
            elif etype == "response.function_call_arguments.delta":
                slot = tool_items.get(event.item_id)
                if slot is not None:
                    slot["function"]["arguments"] += event.delta
                    chars += len(event.delta)
            elif etype == "response.output_text.delta":
                text_parts.append(event.delta)
                chars += len(event.delta)
            elif etype in ("response.reasoning_summary_text.delta",
                           "response.reasoning_text.delta"):
                # Count reasoning tokens toward the heartbeat. The text itself
                # rides on the reasoning item captured at output_item.done,
                # so we don't need to buffer deltas here.
                delta_text = getattr(event, "delta", "")
                chars += len(delta_text)

            now = time.time()
            if now - last_log >= 5:
                _log(f"  turn {turn}: streaming... {chars} chars, {events_seen} events")
                last_log = now

        final = stream.get_final_response()

    msg: dict = {"role": "assistant"}
    text = "".join(text_parts)
    if text:
        msg["content"] = text
    if reasoning_items:
        msg["reasoning_items"] = reasoning_items
    if tool_items:
        msg["tool_calls"] = [tool_items[i] for i in tool_order]

    # Normalize finish state to Chat Completions' finish_reason vocabulary so
    # the attempt loop's "length" / "tool_calls" checks don't need to branch.
    status = getattr(final, "status", None)
    if status == "incomplete":
        details = getattr(final, "incomplete_details", None)
        reason = getattr(details, "reason", "") if details else ""
        finish = "length" if "max_output_tokens" in (reason or "") else "stop"
    elif tool_items:
        finish = "tool_calls"
    else:
        finish = "stop"
    return msg, finish


# ---------- Shared attempt loop ----------


def run_attempt_openai(
    client: OpenAI,
    model: str,
    user_prompt: str,
    tests: list[dict],
    config: TaskConfig,
    max_turns: int,
    sampling_params: dict,
    reasoning_effort: str | None,
    tool_choice: str | None,
    attempt_dir: Path,
    task_dir: Path,
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

    codex = is_codex_model(model)

    def flush_log():
        save_attempt_log(attempt_dir, messages)

    flush_log()

    for turn in range(max_turns):
        try:
            if codex:
                assistant_msg, finish_reason = stream_completion_responses(
                    client=client, model=model, messages=messages,
                    sampling_params=sampling_params,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice, turn=turn,
                )
            else:
                assistant_msg, finish_reason = stream_completion_chat(
                    client=client, model=model, messages=messages,
                    sampling_params=sampling_params,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice, turn=turn,
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
            # Reasoning models occasionally emit the full solution as content
            # instead of calling submit — especially on codex where
            # tool_choice behavior around reasoning items can be subtle. One
            # nudge per attempt; retry consumes a turn from the budget.
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
        description="OpenAI validation benchmark — routes codex models to the "
                    "Responses API and everything else to Chat Completions."
    )
    parser.add_argument("--task", required=True, help="Task name (directory under tasks/)")
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--model", required=True,
                        help="OpenAI model ID (e.g. gpt-5, gpt-5.5, gpt-5-mini, o3, o4-mini, "
                             "gpt-5-codex, gpt-5.3-codex). Models containing 'codex' route "
                             "to the Responses API automatically.")
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_CHOICES, default=None,
                        help=f"Reasoning effort. Choices: {', '.join(REASONING_EFFORT_CHOICES)}. "
                             "Sent as flat `reasoning_effort` on Chat Completions and nested "
                             "`reasoning.effort` on Responses. Omit to use the model default.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Reasoning models may reject this.")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max output tokens. Translated to `max_output_tokens` on the "
                             "Responses path. Default: 32768. Bump for high-effort codex runs.")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--tool-choice", choices=["required", "auto", "none"], default="required",
                        help="Default: required. Both APIs accept it on reasoning models.")
    parser.add_argument("--timeout", type=float, default=600,
                        help="OpenAI client timeout (seconds). Default: 600.")
    parser.add_argument("--docker-timeout", type=float, default=600,
                        help="Timeout (seconds) for `docker run` when starting the sandbox. Default: 600.")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"),
                        help="Override API base URL (env: OPENAI_BASE_URL). Rarely needed.")
    parser.add_argument("--api-key", default=None, help="Or set OPENAI_API_KEY env var")
    parser.add_argument("--slug", default=None, help="Override derived slug")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default=None,
                        help="Attempt-log base dir (default: ~/.vb-data, env VB_DATA_DIR)")
    args = parser.parse_args()

    tasks_dir = Path(__file__).parent / "tasks" / args.task
    if not tasks_dir.is_dir():
        print(f"Error: task directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)
    tests_file = tasks_dir / "tests.jsonl"
    if not tests_file.exists():
        print(f"Error: missing file: {tests_file}", file=sys.stderr)
        sys.exit(1)
    config, user_prompt = load_task(tasks_dir)
    tests = load_tests(tests_file)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set (and --api-key not passed).", file=sys.stderr)
        sys.exit(1)

    sampling_params: dict = {"max_tokens": args.max_tokens}
    if args.temperature is not None:
        sampling_params["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p

    tool_choice = None if args.tool_choice == "none" else args.tool_choice

    codex = is_codex_model(args.model)
    api_path = "responses" if codex else "chat.completions"

    # Provenance fields written alongside sampling_params in each result row.
    results_params: dict = dict(sampling_params)
    if args.reasoning_effort is not None:
        results_params["reasoning_effort"] = args.reasoning_effort
    results_params["api"] = api_path

    slug = args.slug or openai_slug(args.model, args.reasoning_effort)
    results_base = Path(__file__).parent / args.results_dir
    results_file = results_base / "results.jsonl"
    data_dir_base = Path(
        args.data_dir or os.environ.get("VB_DATA_DIR", "") or Path.home() / ".vb-data"
    )
    data_dir_base.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    failures_file = data_dir_base / "failures.jsonl"

    effort_bit = f" reasoning_effort={args.reasoning_effort}" if args.reasoning_effort else ""
    print(f"Running task '{args.task}' with OpenAI model '{args.model}' "
          f"(api={api_path}{effort_bit})")
    sampling_str = ", ".join(f"{k}={v}" for k, v in results_params.items())
    print(f"Attempts: {args.n_attempts} | Max turns: {args.max_turns} | Sampling: {sampling_str}")
    print(f"Debug logs base: {data_dir_base}")
    print(f"Results: {results_file}")
    print("-" * 60)

    # `base_url=None` lets the SDK use its default (api.openai.com). The env
    # var path still works via OPENAI_BASE_URL inside the SDK; we only
    # forward args.api_base when the user passed it explicitly.
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
            result, failure = run_attempt_openai(
                client=client,
                model=args.model,
                user_prompt=user_prompt,
                tests=tests,
                config=config,
                max_turns=args.max_turns,
                sampling_params=sampling_params,
                reasoning_effort=args.reasoning_effort,
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
