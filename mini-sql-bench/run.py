#!/usr/bin/env python
"""mini-sql-bench: drive mini-swe-agent through one deterministic SQLite task to verify
the agent harness works end-to-end against an arbitrary model provider.

The task body runs fully inside a Docker container (`--network none`); only the model
API call happens host-side. A correct answer proves the whole pipeline — model wiring,
the agentic loop, the docker sandbox, and the submit contract — works for whatever
provider/model was passed in.

Examples:
    # Local OpenAI-compatible server (e.g. llama-server --jinja)
    python run.py --model openai/qwen3-coder --base-url http://localhost:8080/v1

    # Anthropic (reads ANTHROPIC_API_KEY)
    python run.py --model anthropic/claude-sonnet-4-6

    # OpenAI (reads OPENAI_API_KEY)
    python run.py --model openai/gpt-4o
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments import get_environment
from minisweagent.models import get_model

HERE = Path(__file__).resolve().parent
IMAGE = "mini-sql-bench:task"


def build_image() -> None:
    print(f"[build] docker build -t {IMAGE} .", flush=True)
    subprocess.run(["docker", "build", "-t", IMAGE, "."], cwd=HERE, check=True)


def build_task_prompt(task_cfg: dict) -> str:
    """The single user message handed to the agent (instance_template is just {{task}})."""
    return (
        f"{task_cfg['question'].rstrip()}\n\n"
        "Reminder: when you have the answer, write ONLY it to /workspace/answer.txt, then "
        "submit by running exactly, as a single command:\n"
        "    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /workspace/answer.txt\n"
    )


def normalize(text: str) -> str:
    """Last non-empty line, stripped — robust to a trailing newline or stray blank lines."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def sum_tokens(messages: list[dict]) -> dict:
    """Sum litellm usage across all model calls. Provider-independent (Anthropic, OpenAI,
    and local llama-server all populate response.usage), so this is reported alongside cost
    -- cost depends on provider pricing that can change, token counts don't.

    prompt_tokens is cumulative-by-turn (each turn re-sends the growing context), so the
    sum is the total input processed across the run, not the unique prompt size.
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    have_any = False
    for m in messages:
        if m.get("role") != "assistant":
            continue
        usage = (m.get("extra", {}) or {}).get("response", {}) or {}
        usage = usage.get("usage") if isinstance(usage, dict) else None
        if not isinstance(usage, dict):
            continue
        have_any = True
        for k in totals:
            v = usage.get(k)
            if isinstance(v, (int, float)):
                totals[k] += int(v)
    return totals if have_any else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="LiteLLM model name, e.g. anthropic/claude-sonnet-4-6 or openai/<served-name>")
    ap.add_argument("--base-url", default=None, help="api_base for an OpenAI-compatible server (local / 3rd-party)")
    ap.add_argument("--api-key", default=None, help="API key. For --base-url endpoints, defaults to a dummy so it never falls back to OPENAI_API_KEY")
    ap.add_argument("--config", default=str(HERE / "config.yaml"), help="Path to config.yaml")
    ap.add_argument("--output", default=str(HERE / "results"), help="Output directory")
    ap.add_argument("--step-limit", type=int, default=None, help="Override agent.step_limit")
    ap.add_argument("--cost-limit", type=float, default=None, help="Override agent.cost_limit (dollars; 0 = no limit)")
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature passed to the model")
    ap.add_argument("--note", default="", help="Free-form label recorded in the result row")
    ap.add_argument("--no-build", action="store_true", help="Skip `docker build` (use an already-built image)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    task_cfg = cfg["task"]

    if not args.no_build:
        build_image()

    # --- model config (provider-agnostic) ---
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["model_name"] = args.model
    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    if args.temperature is not None:
        model_kwargs["temperature"] = args.temperature
    if args.base_url:
        # Pointing at a local / 3rd-party OpenAI-compatible server. Set api_base and a key
        # explicitly so litellm never silently picks up OPENAI_API_KEY for a non-OpenAI
        # endpoint (see memory feedback_api_key_env_fallback).
        model_kwargs["api_base"] = args.base_url
        model_kwargs["api_key"] = args.api_key or "sk-noop"
    elif args.api_key:
        model_kwargs["api_key"] = args.api_key
    model_cfg["model_kwargs"] = model_kwargs

    # --- environment config (docker) ---
    env_cfg = dict(cfg.get("environment", {}))
    env_cfg["image"] = IMAGE

    # --- agent config ---
    agent_cfg = dict(cfg.get("agent", {}))
    if args.step_limit is not None:
        agent_cfg["step_limit"] = args.step_limit
    if args.cost_limit is not None:
        agent_cfg["cost_limit"] = args.cost_limit

    ts = int(time.time())
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"{ts}.traj.json"
    agent_cfg["output_path"] = str(traj_path)

    model = get_model(config=model_cfg)
    env = get_environment(env_cfg, default_type="docker")

    task_prompt = build_task_prompt(task_cfg)
    agent = DefaultAgent(model, env, **agent_cfg)

    print(f"[run] model={args.model} base_url={args.base_url or '-'}", flush=True)
    info = agent.run(task_prompt)
    exit_status = info.get("exit_status", "")
    submission = info.get("submission", "")

    # Score on the container's answer.txt as source of truth (robust to how the model
    # phrased its submit command); fall back to the submission text.
    answer = ""
    try:
        out = env.execute({"command": f"cat {task_cfg['db_path'].rsplit('/', 1)[0]}/answer.txt 2>/dev/null || true"})
        answer = out.get("output", "")
    except Exception as e:  # container may already be gone on hard errors
        print(f"[warn] could not read answer.txt from container: {e}", flush=True)
    if not normalize(answer):
        answer = submission

    expected = str(task_cfg["expected"])
    got = normalize(answer)
    correct = got == normalize(expected)

    result = {
        "ts": ts,
        "model": args.model,
        "base_url": args.base_url,
        "note": args.note,
        "exit_status": exit_status,
        "submission": submission,
        "answer": got,
        "expected": normalize(expected),
        "correct": correct,
        "cost": round(agent.cost, 6),
        "tokens": sum_tokens(agent.messages),
        "n_calls": agent.n_calls,
        "trajectory": traj_path.name,
    }
    result_path = out_dir / f"{ts}.result.json"
    result_path.write_text(json.dumps(result, indent=2))

    tok = result["tokens"]
    tok_str = (
        f"tokens={tok['total_tokens']} (in={tok['prompt_tokens']} out={tok['completion_tokens']})"
        if tok else "tokens=n/a"
    )
    print(
        f"[done] correct={str(correct).lower()} exit={exit_status} "
        f"answer={got!r} expected={normalize(expected)!r} "
        f"cost=${agent.cost:.4f} {tok_str} calls={agent.n_calls}",
        flush=True,
    )
    print(f"[done] result: {result_path}", flush=True)
    print(f"[done] trajectory: {traj_path}", flush=True)
    return 0 if correct else 1


if __name__ == "__main__":
    sys.exit(main())
