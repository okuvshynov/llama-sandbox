#!/usr/bin/env python
"""Compose a runtime task config + prompt from (spec, env) pieces.

The validation-bench matrix has two orthogonal axes:
- spec: what's being implemented (e.g. "toml-1.0", "lua-5.4"). Owns the
  reference text that goes into the prompt body and (eventually) the
  test corpus.
- env:  the implementation language family (e.g. "cpp", "lua"). Owns
  the docker image, source filename, prepare/run commands, and the
  per-language framing in the prompt preamble.

This module reads `specs/<spec>/` and `envs/<env>/`, plus a per-cell
preamble (the small "you are an expert C++ programmer..." block that
genuinely varies per (spec, env) pair), and produces:

- a TaskConfig identical in shape to load_task_config's output, with
  spec/env stamped from the inputs;
- a rendered prompt = preamble + spec body, with the same separator
  convention as the existing prompt.txt files.

Step 2 of the spec/env decomposition: composer infrastructure exists
alongside the existing tasks/ tree. Step 3 will cut over by deleting
the tasks/<name>/prompt.txt + task.json files in favor of composed
equivalents.

Layout this module reads:

    specs/
      <name>/
        meta.json     # display_name, has_spec_body, oracle, ...
        spec_body.md  # the long markdown body for the prompt; empty for
                      # nospec variants (has_spec_body=false)
    envs/
      <name>/
        meta.json     # language, docker_image, source_filename,
                      # prepare_cmd, run_cmd, compile_cmd, ...
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from validation_bench_lib import TaskConfig


# Joins per-cell preamble and spec body. The "---\n" sits between two
# blank lines, matching the existing prompt.txt convention so composed
# output is byte-identical to hand-written prompts.
SEPARATOR = "\n\n---\n\n"


HERE = Path(__file__).resolve().parent


@dataclass
class SpecContent:
    name: str
    meta: dict
    body: str  # may be empty (nospec variants)


@dataclass
class EnvContent:
    name: str
    meta: dict


def load_spec(name: str, root: Path = HERE / "specs") -> SpecContent:
    d = root / name
    if not d.is_dir():
        raise FileNotFoundError(f"spec dir not found: {d}")
    meta = json.loads((d / "meta.json").read_text())
    body_path = d / "spec_body.md"
    body = body_path.read_text() if body_path.exists() else ""
    return SpecContent(name=name, meta=meta, body=body)


def load_env(name: str, root: Path = HERE / "envs") -> EnvContent:
    d = root / name
    if not d.is_dir():
        raise FileNotFoundError(f"env dir not found: {d}")
    meta = json.loads((d / "meta.json").read_text())
    return EnvContent(name=name, meta=meta)


def compose_task_config(spec: SpecContent, env: EnvContent) -> TaskConfig:
    """Build a TaskConfig from spec/env metadata. The `extra` dict carries
    env-meta fields (compile_cmd, language_display_name, ...) so they remain
    available for {placeholder} substitution in the prompt preamble — same
    behavior as today's task.json `extra` mechanism."""
    em = env.meta
    extras = {
        k: v for k, v in em.items()
        if k not in ("language", "docker_image", "source_filename",
                     "prepare_cmd", "run_cmd",
                     "test_timeout_seconds", "prepare_timeout_seconds")
    }
    return TaskConfig(
        language=em["language"],
        docker_image=em["docker_image"],
        source_filename=em["source_filename"],
        prepare_cmd=em.get("prepare_cmd"),
        run_cmd=em["run_cmd"],
        spec=spec.name,
        env=env.name,
        test_timeout_seconds=em.get("test_timeout_seconds", 5.0),
        prepare_timeout_seconds=em.get("prepare_timeout_seconds", 30.0),
        extra=extras,
    )


def compose_prompt(spec: SpecContent, env: EnvContent, preamble: str) -> str:
    """preamble + SEPARATOR + spec.body (when has_spec_body); preamble alone
    for nospec specs. Trailing newline behavior matches existing prompts."""
    if spec.meta.get("has_spec_body", True) and spec.body:
        return preamble.rstrip("\n") + SEPARATOR + spec.body
    return preamble


def split_prompt(prompt_text: str) -> tuple[str, str | None]:
    """Inverse of compose_prompt: split a prompt at the first SEPARATOR
    occurrence into (preamble, spec_body). Returns (preamble, None) if no
    separator is present (nospec-style prompt)."""
    if SEPARATOR in prompt_text:
        preamble, body = prompt_text.split(SEPARATOR, 1)
        return preamble, body
    return prompt_text, None


def main():
    p = argparse.ArgumentParser(
        description="Compose a task config + prompt from (spec, env) pieces.")
    sub = p.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show",
                          help="Print the composed prompt + TaskConfig for a (spec, env) pair.")
    show.add_argument("--spec", required=True)
    show.add_argument("--env", required=True)
    show.add_argument("--preamble", type=Path, required=True,
                      help="Per-cell preamble file (the env-specific intro block).")

    validate = sub.add_parser("validate",
                              help="Verify the composer can reproduce existing tasks/<name>/prompt.txt byte-for-byte.")
    validate.add_argument("--task", required=True,
                          help="Existing task dir name (under tasks/), e.g. toml-1.0-cpp.")

    args = p.parse_args()

    if args.cmd == "show":
        spec = load_spec(args.spec)
        env = load_env(args.env)
        preamble = args.preamble.read_text()
        prompt = compose_prompt(spec, env, preamble)
        config = compose_task_config(spec, env)
        print(f"=== TaskConfig ===")
        print(config)
        print(f"\n=== Prompt ({len(prompt)} chars, {prompt.count(chr(10))+1} lines) ===")
        print(prompt)
        return

    if args.cmd == "validate":
        task_dir = HERE / "tasks" / args.task
        if not task_dir.is_dir():
            print(f"Task dir not found: {task_dir}", file=sys.stderr)
            sys.exit(1)
        existing_prompt = (task_dir / "prompt.txt").read_text()
        existing_config = json.loads((task_dir / "task.json").read_text())
        spec = load_spec(existing_config["spec"])
        env = load_env(existing_config["env"])

        # Extract preamble from existing prompt and recompose, verify byte
        # equivalence.
        preamble, existing_body = split_prompt(existing_prompt)
        composed = compose_prompt(spec, env, preamble)
        prompt_match = composed == existing_prompt

        # Body must match the spec_body.md the composer would inject.
        if spec.meta.get("has_spec_body", True):
            body_match = existing_body == spec.body
        else:
            body_match = existing_body is None

        # TaskConfig fields from compose_task_config should match what
        # load_task_config returns for this task.
        composed_config = compose_task_config(spec, env)
        from validation_bench_lib import load_task_config
        existing_loaded = load_task_config(task_dir)
        # Compare envelope: language, docker_image, source_filename, prepare_cmd,
        # run_cmd, spec, env. `extra` may differ if task.json carries fields not
        # in env meta (e.g. compile_cmd present in both for now); check key by key.
        config_diffs = []
        for fld in ("language", "docker_image", "source_filename",
                    "prepare_cmd", "run_cmd", "spec", "env"):
            if getattr(composed_config, fld) != getattr(existing_loaded, fld):
                config_diffs.append(
                    f"{fld}: composed={getattr(composed_config, fld)!r} "
                    f"existing={getattr(existing_loaded, fld)!r}")
        # Composed extras may be a *superset* of existing extras — env meta
        # can carry forward-compatible fields (e.g. language_display_name)
        # that older task.json files don't have. Only flag a mismatch if a
        # key exists in BOTH and the values disagree.
        for k in composed_config.extra:
            if k not in existing_loaded.extra:
                continue
            if existing_loaded.extra[k] != composed_config.extra[k]:
                config_diffs.append(
                    f"extra[{k!r}]: composed={composed_config.extra[k]!r} "
                    f"existing={existing_loaded.extra[k]!r}")

        ok = prompt_match and body_match and not config_diffs
        verdict = "OK" if ok else "MISMATCH"
        print(f"[{verdict}] {args.task}: prompt_match={prompt_match} "
              f"body_match={body_match} config_diffs={len(config_diffs)}")
        for d in config_diffs:
            print(f"    {d}")
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()
