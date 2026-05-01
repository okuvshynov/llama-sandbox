#!/usr/bin/env python3
"""Shared library for the per-provider benchmark scripts.

Imported by `validation_bench_{anthropic,moonshot,llama_cpp,fireworks,openai,deepseek}.py`.
No CLI entry point lives here anymore — each provider has its own dedicated script
(see `EXAMPLES.md`). This file is deliberately framework-free: no litellm, no
provider SDKs, just Sandbox + scoring + dataclasses + slug/attempt helpers that
every runner reuses verbatim.
"""

import datetime
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# Flip on with `VB_VERBOSE=1` (any truthy string). When set, handle_submit
# writes extra diagnostic files into each per-submission directory:
#   sandbox_state_before_prepare.txt  — container state (PIDs cgroup, /work
#                                        listing+disk, ps, memory) right
#                                        before the prepare_cmd runs
#   prepare_raw.txt                   — separated rc / stdout / stderr from
#                                        prepare_cmd. Distinguishes "compile
#                                        failed silently" from "compile
#                                        produced no diagnostics" — the
#                                        ambiguity that compiler.txt alone
#                                        can't resolve when it's empty.
#   sandbox_state_after_failed_prepare.txt   — only when prepare fails
#   sandbox_state_after_tests.txt            — only when run_tests ran
# Costs: a handful of cheap `docker exec` probes per submission. Off by
# default so production runs aren't slowed by debug instrumentation.
VB_VERBOSE = bool(os.environ.get("VB_VERBOSE"))


# Stamped into every results row + failures row so past runs can be filtered
# by harness version if a scoring/prompt/behavior change later breaks
# comparability. Bump on: prompt template changes, scoring logic changes,
# tool schema changes, sandbox compile flags, or any other change that
# could affect the numbers for a given (task, model, slug) triple. Also
# bump on milestone changes that expand the harness's surface area
# (new task type, new sandbox image, new provider script) — the numbers
# are still comparable, but the version then marks "what was available
# at the time of this run".
# Rows without this field predate the scheme — treat as "pre-0.0.2".
#
# Version log:
#   0.0.2 — per-task config (task.json), Lua sandbox / Lua-as-implementation
#   0.0.3 — lua-5.4-cpp task (Lua syntax validation as a target spec),
#           hand-curated corpus path in setup.sh
#   0.0.4 — spec/env fields added to task.json + result rows; numbers for
#           existing (task, model, slug) triples unchanged, schema gained
#           two derivable fields for cross-axis aggregation. Past rows
#           backfilled by scripts/migrate_results_spec_env.py.
#   0.0.5 — per-turn token usage stamped on result rows: input_tokens,
#           output_tokens, reasoning_tokens, cached_tokens. Numbers for
#           existing (task, model, slug) triples unchanged. Older rows
#           lack the new fields; analysis tools should treat them as
#           null/unavailable.
#   0.0.6 — Sandbox.prepare() now captures stdout + stderr from the
#           prepare_cmd, not stderr alone. Affects models' compile-error
#           feedback for toolchains that write diagnostics to stdout
#           (erlc, possibly others). For toolchains that use stderr only
#           (clang++, luac5.4), behavior is identical and numbers for
#           (task, model, slug) triples on those envs are unchanged.
#           Erlang/OTP env added at the same milestone — no historical
#           rows existed for it, so nothing to compare against.
#   0.0.7 — per-turn timing stamped on result rows: model_seconds (API
#           streaming wall time, attached to the first Submission of
#           each turn so sums don't double-count, mirroring usage),
#           prepare_seconds (Sandbox.prepare() wall time), tests_seconds
#           (run_tests wall time). All three are float|None, rounded to
#           ms (3 decimals). Numbers for existing (task, model, slug)
#           triples unchanged — purely additive schema fields. Older
#           rows lack the new fields; analysis tools should treat them
#           as null/unavailable.
#   0.0.8 — Three coupled changes that all address "the harness was
#           silently misclassifying broken-parser cases as successes":
#           (a) Scoring contract is now print-based instead of
#               exit-code-based. The parser must print EXACTLY 'valid' or
#               'invalid' (case-sensitive, optional surrounding whitespace)
#               to stdout. Any other output — empty, mixed case, debug
#               chatter, multi-line — fails the test regardless of expected
#               label. Removes the prior loophole where a hanging or
#               crashing parser was credited with "correct rejection" on
#               invalid inputs (rc != 0 → counted as TN under the old rule,
#               regardless of cause).
#           (b) Sandbox.run_input wraps run_cmd with `timeout -s KILL <secs>s`
#               so the per-test timeout fires *inside* the container.
#               Without the wrapper, subprocess.run's outer timeout only
#               killed the host-side docker-exec process; the in-container
#               child got orphaned to PID 1. On tasks where models emit
#               parsers with infinite loops on adversarial inputs (e.g.
#               yaml-1.2-cpp17/-go), orphan processes accumulated and
#               saturated the container's PIDs cgroup, causing later
#               submissions in the same attempt to fail with EAGAIN on fork.
#           (c) FAIL feedback lines now report semantic detail
#               ("timeout", "killed by signal N", "got '<output>'") rather
#               than raw exit codes. Sharper signal for the model on what
#               kind of bug to fix.
#
#           Schema: Sandbox.run_input now returns (rc, stdout_bytes) instead
#           of int. Internal API change only; harness consumers (provider
#           scripts) don't call run_input directly — they go through
#           handle_submit / run_tests.
#
#           Compat break: numbers for ANY (task, model, slug) triple where
#           the parser previously hung, crashed, or got partial credit
#           through exit-code conflation will change. yaml-1.2-cpp17/-go
#           are most affected (parsers there commonly hang on edge cases).
#           Clean-parser tasks (palindrome, well-behaved toml/lua impls)
#           are essentially unchanged — they already exit cleanly with the
#           right code; the only difference is they must now print 'valid'
#           or 'invalid' to stdout. All 15 task preambles updated to
#           specify the print contract.
#   0.0.9 — Tightens 0.0.8: a test now passes iff verdict matches AND
#           the process exits cleanly (rc == 0). Closes a hole in 0.0.8
#           where a parser that printed the right verdict and then
#           crashed/timed-out/exited-nonzero was counted as success.
#           Especially important for Erlang: the BEAM does not auto-halt
#           after `main/0` returns, so without an explicit `halt(0)` the
#           runtime idles until the in-container per-test timeout fires
#           (rc=124) — under 0.0.8 the verdict-only check would have
#           credited that as a clean classification. All 15 preambles
#           updated to require clean exit; erlang preambles now mention
#           `halt(0)` explicitly.
#   0.0.10 — Container is now restarted between submissions of the same
#           attempt (Sandbox.begin_submission, called from handle_submit).
#           Closes a leak that 0.0.8's `timeout -s KILL` wrapper didn't:
#           PID 1 in the sandbox is `sleep infinity` and doesn't reap
#           zombies, while Go's runtime maps each goroutine to an OS
#           thread that counts against --pids-limit=256. After a few
#           hundred run_tests invocations the cgroup saturates and every
#           subsequent `docker exec` — the next prepare_cmd in particular
#           — fails with `sh: can't fork: Resource temporarily unavailable`,
#           which surfaced as silent compile errors (rc != 0, zero stdout
#           and stderr) starting with the second submission of an attempt.
#           Restart is ~1s of docker stop+run, amortized across multi-
#           minute submissions; the first submission of an attempt skips
#           the restart since the provider just started a fresh container.
#
#           Compat break: any prior (task, model, slug) attempt where
#           submissions ≥2 hit the silent-compile-error mode would now
#           run cleanly. Affected configurations: yaml-1.2-go and other
#           Go tasks where parsers produced many timeouts in turn 0;
#           cpp17 / lua / erlang variants are typically less affected
#           because their runtimes spawn fewer OS threads per process.
#           Scoring rules from 0.0.9 are unchanged.
#   0.0.11 — validation_bench_openai.py: Chat Completions code path removed.
#           All OpenAI requests go through the Responses API regardless of
#           model. Per OpenAI's docs ("Reasoning models work better with
#           the Responses API. While the Chat Completions API is still
#           supported, you'll get improved model intelligence and
#           performance by using Responses"), and per actual usage of this
#           script (only reasoning models — gpt-5.x, codex variants — have
#           ever been benchmarked through it), the Chat Completions branch
#           was vestigial and gated reasoning behavior on gpt-5.4 / gpt-5.5.
#
#           Compat break: numbers for any (model, slug) cell previously run
#           through Chat Completions may differ — gpt-5.5 in particular
#           reported ~512 reasoning_tokens per turn over Chat Completions
#           and is expected to use significantly more reasoning tokens at
#           the same effort level over Responses. The `sampling_params.api`
#           field is still stamped on every row ("responses" since 0.0.11)
#           so old chat.completions rows remain identifiable. Scoring rules
#           and sandbox lifecycle (0.0.10) are unchanged. Only the OpenAI
#           provider script is affected; fireworks / deepseek / moonshot /
#           llama_cpp continue using Chat Completions (their endpoints
#           don't support a Responses analog).
VB_VERSION = "0.0.11"


@dataclass
class ConfusionMatrix:
    tp: int = 0  # valid correctly accepted
    fn: int = 0  # valid incorrectly rejected
    fp: int = 0  # invalid incorrectly accepted
    tn: int = 0  # invalid correctly rejected

    @property
    def passed(self) -> int:
        return self.tp + self.tn

    @property
    def total(self) -> int:
        return self.tp + self.fn + self.fp + self.tn

    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient (phi coefficient)."""
        denom_sq = ((self.tp + self.fp) * (self.tp + self.fn)
                    * (self.tn + self.fp) * (self.tn + self.fn))
        if denom_sq == 0:
            return 0.0
        return (self.tp * self.tn - self.fp * self.fn) / math.sqrt(denom_sq)


@dataclass
class TestResult:
    compiled: bool
    compiler_output: str
    test_output: str
    matrix: ConfusionMatrix
    # Per-submission timing measured by handle_submit, in seconds.
    # prepare_seconds covers Sandbox.prepare() (compile / syntax-check);
    # tests_seconds covers run_tests() over the full corpus. Both are 0.0
    # when the corresponding step was skipped (e.g. tests_seconds is 0.0
    # on a compile failure).
    prepare_seconds: float = 0.0
    tests_seconds: float = 0.0


@dataclass
class Submission:
    turn: int
    matrix: ConfusionMatrix | None = None  # None for failed submissions
    error: str | None = None  # e.g. "compile_error", "compile_timeout"
    # Per-turn API token usage in normalized form. The dict carries
    # {input_tokens, output_tokens, reasoning_tokens, cached_tokens}; any
    # field can be None on providers that don't expose it. Recorded on
    # the *first* Submission of a turn (subsequent submissions in the
    # same turn share the API call, so their usage is left as None).
    usage: dict | None = None
    # Wall time spent calling the model's streaming API for this turn.
    # Recorded on the *first* Submission of a turn for the same reason as
    # `usage` — sum-aggregation across rows counts each API call once.
    model_seconds: float | None = None
    # Wall time spent in Sandbox.prepare() and run_tests() for THIS
    # submission. Per-submission, not per-turn — each submit triggers its
    # own compile + test cycle, so each row carries its own breakdown.
    prepare_seconds: float | None = None
    tests_seconds: float | None = None


@dataclass
class AttemptResult:
    attempt_id: str
    timestamp: str  # ISO 8601, recorded when attempt completes
    elapsed_seconds: float
    submissions: list[Submission]


@dataclass
class InfraFailure:
    timestamp: str       # ISO 8601
    turn: int            # which turn failed
    error_type: str      # "api_error", "timeout", etc.
    error_message: str


def _log(msg: str):
    print(msg, flush=True)


# ---------- token-usage normalization ----------------------------------------
#
# Each provider exposes streaming token counts in its own shape. Runners
# capture the raw object during stream iteration and run it through one of
# the helpers below to land in a single normalized dict:
#
#     {input_tokens, output_tokens, reasoning_tokens, cached_tokens}
#
# Any field that the provider doesn't expose is left as None — e.g. Anthropic
# bundles thinking tokens into output_tokens (no separate reasoning count),
# llama.cpp doesn't track reasoning tokens at all, etc.

def _safe_attr(obj, *names):
    """Walk a chain of attributes, returning None if any link is missing/None."""
    for name in names:
        if obj is None:
            return None
        obj = getattr(obj, name, None)
    return obj


def normalize_openai_chat_usage(usage) -> dict | None:
    """Convert an OpenAI Chat Completions `CompletionUsage` (or any provider
    returning the same shape: Fireworks, Moonshot, DeepSeek) to the
    normalized usage dict."""
    if usage is None:
        return None
    cached = (_safe_attr(usage, "prompt_tokens_details", "cached_tokens")
              # Moonshot uses a flat top-level cached_tokens
              or getattr(usage, "cached_tokens", None)
              # DeepSeek uses prompt_cache_hit_tokens
              or getattr(usage, "prompt_cache_hit_tokens", None))
    return {
        "input_tokens":     getattr(usage, "prompt_tokens", None),
        "output_tokens":    getattr(usage, "completion_tokens", None),
        "reasoning_tokens": _safe_attr(usage, "completion_tokens_details", "reasoning_tokens"),
        "cached_tokens":    cached,
    }


def normalize_openai_responses_usage(usage) -> dict | None:
    """Convert an OpenAI Responses-shaped `ResponseUsage` to the normalized dict."""
    if usage is None:
        return None
    return {
        "input_tokens":     getattr(usage, "input_tokens", None),
        "output_tokens":    getattr(usage, "output_tokens", None),
        "reasoning_tokens": _safe_attr(usage, "output_tokens_details", "reasoning_tokens"),
        "cached_tokens":    _safe_attr(usage, "input_tokens_details", "cached_tokens"),
    }


def normalize_anthropic_usage(start_usage: dict | None,
                              last_delta_usage: dict | None) -> dict | None:
    """Combine the `usage` from an Anthropic `message_start` event (initial
    input + first output token) with the `usage` from the *last*
    `message_delta` event (cumulative output, possibly updated input).
    Anthropic includes thinking tokens in output_tokens — no separate
    reasoning count is exposed on the wire."""
    if start_usage is None and last_delta_usage is None:
        return None
    def pick(field):
        # Prefer the delta's value (cumulative / final) over message_start's.
        if last_delta_usage is not None and last_delta_usage.get(field) is not None:
            return last_delta_usage.get(field)
        if start_usage is not None:
            return start_usage.get(field)
        return None
    return {
        "input_tokens":     pick("input_tokens"),
        "output_tokens":    pick("output_tokens"),
        "reasoning_tokens": None,
        "cached_tokens":    pick("cache_read_input_tokens"),
    }


def normalize_llama_cpp_timings(timings: dict | None) -> dict | None:
    """Convert llama.cpp's non-OpenAI `timings` block to the normalized dict.
    Per llama.cpp: `prompt_n` is new prompt tokens processed (not from
    cache); `cache_n` is the count reused from cache; `predicted_n` is
    output tokens. Total input = prompt_n + cache_n."""
    if timings is None:
        return None
    prompt_n = timings.get("prompt_n", 0) or 0
    cache_n  = timings.get("cache_n",  0) or 0
    return {
        "input_tokens":     prompt_n + cache_n,
        "output_tokens":    timings.get("predicted_n"),
        "reasoning_tokens": None,
        "cached_tokens":    cache_n if cache_n > 0 else None,
    }


SUBMIT_TOOL = {
    "type": "function",
    "function": {
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
    },
}


def load_tests(tests_file: Path) -> list[dict]:
    """Load test cases from JSONL file."""
    tests = []
    with open(tests_file) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


@dataclass
class TaskConfig:
    """Per-task sandbox config, loaded from `<task>/task.json`.

    `prepare_cmd` is the optional first phase (compile / syntax-check / bytecode-precompile).
    Setting it to None means "no prep" — the source file is just dropped into /work and
    `run_cmd` is invoked per test (suitable for interpreters where you'd rather see syntax
    errors surface on the first run than spend time on a separate parse-only pass).

    `extra` carries arbitrary task-specific keys that authors expose as `{key}` placeholders
    in `prompt.txt` (e.g. a display-friendly `compile_cmd` distinct from the full prepare_cmd).

    `spec` and `env` are the two-axis decomposition of what the task tests:
    `spec` is what's being implemented (e.g. "toml-1.0", "lua-5.4"), `env` is
    the implementation language family (e.g. "cpp17", "lua"). Both are stamped
    into every result row so analysis can aggregate across one axis while
    holding the other constant. They're required as of vb_version 0.0.4.
    """
    language: str
    docker_image: str
    source_filename: str
    prepare_cmd: str | None
    run_cmd: str
    spec: str
    env: str
    test_timeout_seconds: float = 5.0
    prepare_timeout_seconds: float = 30.0
    extra: dict = field(default_factory=dict)


_TASK_CONFIG_KNOWN = {
    "language", "docker_image", "source_filename", "prepare_cmd", "run_cmd",
    "spec", "env",
    "test_timeout_seconds", "prepare_timeout_seconds",
}


def load_task_config(task_dir: Path) -> TaskConfig:
    """Read task.json from a task directory.

    Two shapes are accepted:

    1. **Composed** (canonical from vb_version 0.0.4 onward): task.json
       contains only `{"spec": ..., "env": ...}`. Env-specific fields
       (language, docker_image, source_filename, prepare_cmd, run_cmd,
       …) are pulled from `envs/<env>/meta.json`. Per-cell `task.json`
       fields take precedence over env meta if both are present, so an
       individual cell can still override e.g. `prepare_cmd` if it
       genuinely diverges from the env default.

    2. **Inline legacy**: task.json declares language, docker_image,
       prepare_cmd, run_cmd, source_filename inline alongside spec/env.
       Treated as authoritative; envs/ is not consulted. Kept so
       hand-written task.json files still work in tests and ad-hoc
       experiments.
    """
    config_file = task_dir / "task.json"
    if not config_file.exists():
        raise FileNotFoundError(f"task.json not found in {task_dir}")
    data = json.loads(config_file.read_text())
    if "spec" not in data or "env" not in data:
        raise ValueError(f"{config_file}: task.json must declare 'spec' and 'env'")

    if "language" not in data:
        # Composed shape: pull env meta from envs/<env>/meta.json
        env_meta_file = (Path(__file__).resolve().parent
                         / "data" / "envs" / data["env"] / "meta.json")
        if not env_meta_file.exists():
            raise FileNotFoundError(
                f"composed task.json requires {env_meta_file}; ensure the env "
                f"'{data['env']}' has a meta.json under envs/")
        env_meta = json.loads(env_meta_file.read_text())
        # task.json fields take precedence over env meta for true overrides.
        data = {**env_meta, **data}

    extras = {k: v for k, v in data.items() if k not in _TASK_CONFIG_KNOWN}
    return TaskConfig(
        language=data["language"],
        docker_image=data["docker_image"],
        source_filename=data["source_filename"],
        prepare_cmd=data.get("prepare_cmd"),
        run_cmd=data["run_cmd"],
        spec=data["spec"],
        env=data["env"],
        test_timeout_seconds=data.get("test_timeout_seconds", 5.0),
        prepare_timeout_seconds=data.get("prepare_timeout_seconds", 30.0),
        extra=extras,
    )


def render_prompt(prompt_text: str, config: TaskConfig) -> str:
    """Substitute {field} placeholders from TaskConfig + extras into prompt text."""
    fields = {
        "language": config.language,
        "source_filename": config.source_filename,
        "prepare_cmd": config.prepare_cmd or "",
        "run_cmd": config.run_cmd,
    }
    fields.update(config.extra)
    for k, v in fields.items():
        prompt_text = prompt_text.replace(f"{{{k}}}", str(v))
    return prompt_text


class Sandbox:
    """Docker container sandbox for preparing and running untrusted code.

    Generic over language: per-task config supplies the docker image, source filename,
    prepare command (compile / syntax-check, may be None for interpreters), and
    per-test run command.
    """

    def __init__(self, config: TaskConfig, startup_timeout: float = 600):
        self.config = config
        self.container_id: str | None = None
        self.startup_timeout = startup_timeout
        # Counts submissions begun on the current Sandbox (across restarts).
        # Drives begin_submission's "restart between submissions" policy:
        # the very first submission of an attempt runs on the container the
        # provider just started; every subsequent one gets a fresh container.
        self._submissions_begun = 0

    def start(self):
        try:
            result = subprocess.run(
                ["docker", "run", "-d", "--rm",
                 "--network=none",
                 "--memory=512m",
                 "--cpus=1",
                 "--pids-limit=256",
                 "--read-only",
                 "--tmpfs=/work:rw,exec,size=64m",
                 "--tmpfs=/tmp:rw,size=64m",
                 self.config.docker_image, "sleep", "infinity"],
                capture_output=True, text=True,
                timeout=self.startup_timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"`docker run` did not return within {self.startup_timeout}s "
                "(daemon hang or image pull stuck?)"
            )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start sandbox: {result.stderr}")
        self.container_id = result.stdout.strip()

    def stop(self):
        if self.container_id:
            subprocess.run(["docker", "kill", self.container_id],
                           capture_output=True)
            self.container_id = None

    def begin_submission(self):
        """Prepare the container for a new submission. On all but the first
        submission of the attempt, this restarts the container — `docker
        kill` + `docker run` — to clear any process / disk / cgroup state
        leaked by the prior submission's run_tests cycle.

        Why this is necessary: PID 1 in the sandbox is `sleep infinity`,
        which doesn't reap zombies, and the Go runtime in particular
        maps each goroutine to an OS thread that counts against the
        --pids-limit=256 cgroup. After a few hundred test invocations
        with timeouts, the cgroup saturates and *every* subsequent
        `docker exec` (the next prepare_cmd, our verbose probes, etc.)
        fails with `sh: can't fork: Resource temporarily unavailable`.
        Restarting the container is the cheapest correct fix: ~1s of
        docker stop+run amortized over a multi-minute submission.
        """
        if self._submissions_begun > 0:
            self.stop()
            self.start()
        self._submissions_begun += 1

    def _exec(self, cmd: list[str], input_data: bytes | None = None,
              timeout: float = 30) -> subprocess.CompletedProcess:
        full_cmd = ["docker", "exec"]
        if input_data is not None:
            full_cmd.append("-i")
        full_cmd.extend([self.container_id] + cmd)
        return subprocess.run(full_cmd, input=input_data,
                              capture_output=True, timeout=timeout)

    def prepare(self, source_code: str,
                verbose_dir: Path | None = None) -> tuple[bool, str]:
        """Write source into container and run prepare_cmd. Returns (success, output).

        Equivalent to "compile" for compiled languages (clang++, go build) and to a
        syntax check for interpreters that have one (luac -p, py_compile). When
        prepare_cmd is None, only the file write happens and we report success.

        When `verbose_dir` is given (only when VB_VERBOSE is set), writes a
        `prepare_raw.txt` with separated rc / stdout / stderr from prepare_cmd.
        Solves the "compiler.txt is empty" ambiguity: an empty compiler.txt
        could mean either "compile succeeded silently" (e.g. `go build` on a
        clean source) or "compile died silently" (OOM, EAGAIN on fork from
        PIDs cgroup exhaustion). The raw rc disambiguates.
        """
        src_path = f"/work/{self.config.source_filename}"
        write = self._exec(["sh", "-c", f"cat > {src_path}"],
                           input_data=source_code.encode())
        if write.returncode != 0:
            return False, f"Failed to write source: {write.stderr.decode()}"

        if not self.config.prepare_cmd:
            return True, ""

        try:
            comp = self._exec(
                ["sh", "-c", f"cd /work && {self.config.prepare_cmd}"],
                timeout=self.config.prepare_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            if verbose_dir is not None:
                (verbose_dir / "prepare_raw.txt").write_text(
                    f"rc=<host-timeout>\n"
                    f"prepare_timeout_seconds={self.config.prepare_timeout_seconds:g}\n"
                )
            return False, f"Preparation timed out ({self.config.prepare_timeout_seconds:g}s limit)."

        if verbose_dir is not None:
            stdout_text = comp.stdout.decode("utf-8", errors="replace")
            stderr_text = comp.stderr.decode("utf-8", errors="replace")
            (verbose_dir / "prepare_raw.txt").write_text(
                f"rc={comp.returncode}\n"
                f"stdout_bytes={len(comp.stdout)}\n"
                f"stderr_bytes={len(comp.stderr)}\n"
                f"--- stdout ---\n{stdout_text}\n"
                f"--- stderr ---\n{stderr_text}\n"
            )

        # Capture both channels — toolchains differ on which one diagnostics
        # land on. clang++ and luac5.4 write errors to stderr, but erlc writes
        # them to stdout. Concatenating gives the model the actual compiler
        # output regardless of channel; for tools that use only one channel
        # the other is empty.
        return comp.returncode == 0, (comp.stdout + comp.stderr).decode()

    def snapshot_state(self) -> str:
        """Probe container resource state for verbose-mode debugging. Returns
        multi-line text suitable for direct write to a .txt log file.

        Cheap (~5 short docker execs, each timeout-bounded). Defensive against
        a degraded container — every probe is wrapped so a single failure
        doesn't sink the whole snapshot. Useful when prepare/run failures
        suggest resource exhaustion (PIDs cgroup, /work disk, memory) rather
        than a real compile/test bug.
        """
        parts = []

        def probe(label: str, shell_cmd: str):
            try:
                r = self._exec(["sh", "-c", shell_cmd], timeout=5)
                out = (r.stdout + r.stderr).decode("utf-8", errors="replace").rstrip()
                parts.append(f"--- {label} (rc={r.returncode}) ---\n{out}")
            except subprocess.TimeoutExpired:
                parts.append(f"--- {label} (probe timed out) ---")
            except Exception as e:
                parts.append(f"--- {label} (probe error: {e!r}) ---")

        # cgroup v2 puts pids.{current,max} at the root; v1 nests under pids/.
        # Try both; the missing path just shows "No such file" which is fine.
        probe("pids cgroup",
              "for f in /sys/fs/cgroup/pids.current /sys/fs/cgroup/pids.max "
              "/sys/fs/cgroup/pids/pids.current /sys/fs/cgroup/pids/pids.max; "
              'do echo \"$f:\"; cat \"$f\" 2>&1; done')
        probe("processes", "ps -ef 2>&1 || ps 2>&1")
        probe("/work listing", "ls -la /work 2>&1; echo; du -sh /work 2>&1")
        probe("/work disk", "df -h /work 2>&1")
        probe("memory", "head -10 /proc/meminfo 2>&1")

        return "\n\n".join(parts) + "\n"

    def run_input(self, input_data: bytes) -> tuple[int, bytes]:
        """Run config.run_cmd in /work with input via stdin.

        Returns (exit_code, stdout_bytes). exit_code is -1 on outer timeout
        (the inner cgroup timeout should fire first; outer is safety net).

        Wraps run_cmd with `timeout -s KILL <secs>s` so the kill propagates
        *inside* the container. Without it, only the docker-exec process on
        the host gets killed when subprocess.run hits its outer timeout — the
        in-container child gets orphaned to PID 1 (sleep infinity) and keeps
        running. Under buggy parsers (e.g. infinite loops on adversarial
        YAML inputs), orphaned `./solution` processes accumulate quickly:
        each Go binary spawns ~5 OS threads, and the container's PIDs cgroup
        counts threads, so ~50 orphans saturates the default --pids-limit=256.
        Subsequent `go build` calls in the same attempt then fail with EAGAIN
        on fork.

        `timeout` is available on both alpine (busybox) and debian-bookworm-slim
        (coreutils) base images with identical `-s SIGNAL <secs>s CMD`
        invocation.
        """
        secs = self.config.test_timeout_seconds
        wrapped = f"cd /work && timeout -s KILL {secs:g}s {self.config.run_cmd}"
        try:
            proc = self._exec(
                ["sh", "-c", wrapped],
                input_data=input_data,
                timeout=secs + 2,
            )
            return proc.returncode, proc.stdout
        except subprocess.TimeoutExpired:
            return -1, b""


VERDICT_VALID = "valid"
VERDICT_INVALID = "invalid"
VERDICTS = (VERDICT_VALID, VERDICT_INVALID)


def _fail_detail(rc: int, stdout: bytes, verdict: str | None) -> str:
    """Compose a one-line explanation of why a test didn't pass, suitable for
    the FAIL feedback line that goes back to the model."""
    if verdict in VERDICTS:
        # Got a clean verdict from stdout. Either it was the wrong content,
        # or the process didn't exit cleanly. Both are failures.
        if rc == 0:
            return f"got {verdict!r}"  # wrong verdict, clean exit
        if rc == 124 or rc == -1:
            return f"got {verdict!r} but timed out"
        if rc >= 128:
            return f"got {verdict!r} but killed by signal {rc - 128}"
        return f"got {verdict!r} but exited with rc={rc}"
    # Otherwise the output didn't match either verdict. Diagnose why.
    if rc == 124 or rc == -1:
        return "timeout (no verdict printed)"
    if rc >= 128:
        sig = rc - 128
        return f"killed by signal {sig} (no verdict printed)"
    if not stdout.strip():
        return f"no output (exit={rc})"
    # Some output was produced but it doesn't match either verdict literal.
    try:
        text = stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        text = "<non-utf8 output>"
    if len(text) > 80:
        preview = text[:60] + "..." + text[-15:]
    else:
        preview = text
    return f"got {preview!r} (expected exactly 'valid' or 'invalid')"


def run_tests(sandbox: Sandbox, tests: list[dict], tests_root: Path) -> tuple[str, ConfusionMatrix]:
    """Run all test cases against the prepared solution in sandbox, return
    (output_text, matrix). `tests_root` is the directory under which each
    test's `input_file` is resolved — typically `specs/<spec>/`, since
    the corpus is a property of the spec rather than the (spec, env) cell.

    Scoring contract (vb_version 0.0.9+): a test passes iff
        (1) stdout, after .strip(), equals exactly `valid` or `invalid`
            (case-sensitive), AND it equals the expected label, AND
        (2) the process exits cleanly (rc == 0).
    Both conditions must hold. A parser that prints the right verdict
    but then crashes / times out / exits non-zero fails the test — the
    process must classify *and* terminate cleanly. Important for Erlang
    in particular, where the BEAM does not auto-halt after `main/0`
    returns and an explicit `halt(0)` is required."""
    matrix = ConfusionMatrix()
    lines = []

    for t in tests:
        input_data = (tests_root / t["input_file"]).read_bytes()

        tid = t.get("id", "?")
        label = t["label"]
        expected = t["expected"]

        rc, stdout = sandbox.run_input(input_data)

        # Decode the verdict. Strict matching: stdout.strip() must equal one
        # of the two literals exactly (case-sensitive).
        try:
            verdict_text = stdout.decode("utf-8", errors="replace").strip()
        except Exception:
            verdict_text = ""
        verdict = verdict_text if verdict_text in VERDICTS else None

        passed = (rc == 0) and (verdict == expected)
        if passed:
            if expected == "valid":
                matrix.tp += 1
            else:
                matrix.tn += 1
        else:
            if expected == "valid":
                matrix.fn += 1
            else:
                matrix.fp += 1
            detail = _fail_detail(rc, stdout, verdict)
            lines.append(f"FAIL {tid}: {label} ({detail}, expected {expected!r})")

    lines.append(f"{matrix.passed}/{matrix.total} passed")
    return "\n".join(lines), matrix


def handle_submit(source_code: str, tests: list[dict], sandbox: Sandbox,
                  tests_root: Path,
                  sub_dir: Path | None = None) -> TestResult:
    """Run prepare + tests for one submission. When VB_VERBOSE and `sub_dir`
    are both set, also dumps sandbox state snapshots into `sub_dir` for
    debugging — see VB_VERBOSE comment at top of file."""
    sandbox.begin_submission()

    verbose = VB_VERBOSE and sub_dir is not None

    if verbose:
        (sub_dir / "sandbox_state_before_prepare.txt").write_text(sandbox.snapshot_state())

    t0 = time.perf_counter()
    compiled, compiler_output = sandbox.prepare(
        source_code, verbose_dir=sub_dir if verbose else None,
    )
    prepare_seconds = time.perf_counter() - t0

    if not compiled:
        if verbose:
            (sub_dir / "sandbox_state_after_failed_prepare.txt").write_text(sandbox.snapshot_state())
        return TestResult(
            compiled=False,
            compiler_output=compiler_output,
            test_output="",
            matrix=ConfusionMatrix(),
            prepare_seconds=prepare_seconds,
            tests_seconds=0.0,
        )

    t1 = time.perf_counter()
    test_output, matrix = run_tests(sandbox, tests, tests_root)
    tests_seconds = time.perf_counter() - t1

    if verbose:
        (sub_dir / "sandbox_state_after_tests.txt").write_text(sandbox.snapshot_state())

    return TestResult(
        compiled=True,
        compiler_output=compiler_output,
        test_output=test_output,
        matrix=matrix,
        prepare_seconds=prepare_seconds,
        tests_seconds=tests_seconds,
    )


def format_tool_result(result: TestResult) -> str:
    parts = []
    if not result.compiled:
        parts.append("COMPILATION FAILED")
        parts.append(result.compiler_output)
        return "\n".join(parts)

    m = result.matrix
    parts.append(f"Compiled successfully. Test results: {m.passed}/{m.total} passed.")
    if m.passed < m.total:
        # Include FAIL lines so the model can fix bugs
        for line in result.test_output.splitlines():
            if line.startswith("FAIL "):
                parts.append(line)
    return "\n".join(parts)


def auto_detect_model(api_base: str, api_key: str) -> str:
    """Auto-detect model from a local OpenAI-compatible server via /models endpoint."""
    url = api_base.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "validation-bench/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach {url}: {e}", file=sys.stderr)
        sys.exit(1)
    model_ids = [m["id"] for m in data.get("data", [])]
    if not model_ids:
        print("Error: no models available at the endpoint.", file=sys.stderr)
        sys.exit(1)
    model_id = model_ids[0]
    print(f"Auto-detected model: {model_id}")
    return model_id


def derive_slug(model: str, reasoning_effort: str | None = None) -> str:
    """Derive a filesystem-friendly slug from a model string (with optional provider prefix).

    Examples:
        anthropic/claude-opus-4-6        -> claude-opus-4.6
        anthropic/claude-sonnet-4-20250514 -> claude-sonnet-4.0
        openai/gpt-5.3-codex             -> gpt-5.3-codex
        openai/gpt-5.3-codex + high      -> gpt-5.3-codex-high
        fireworks_ai/accounts/fireworks/models/glm-5p1 -> fireworks-glm-5p1
        openai/Qwen3.5-122B-A10B-UD-Q6_K_XL-00001-of-00004.gguf -> qwen3.5-122b-a10b-q6_k_xl
    """
    # Hosted-OSS prefixes keep a short provider tag; first-party prefixes drop it
    # (OpenAI/Anthropic identity is implicit in the model name).
    if model.startswith("fireworks_ai/") or model.startswith("fireworks/"):
        rest = model.split("/", 1)[1]
        rest = re.sub(r"^accounts/[^/]+/models/", "", rest)
        name = f"fireworks-{rest}"
    elif "/" in model:
        name = model.split("/", 1)[1]
    else:
        name = model
    # Collapse any remaining slashes.
    name = name.replace("/", "-")

    # Strip GGUF filenames: keep quant level, drop shard suffix and extension
    # e.g. Qwen3.5-122B-A10B-UD-Q6_K_XL-00001-of-00004.gguf -> Qwen3.5-122B-A10B-Q6_K_XL
    # e.g. model-UD-IQ3_XXS-00001-of-00004.gguf -> model-IQ3_XXS
    name = re.sub(r'-UD-([A-Za-z0-9_]+)(-\d+-of-\d+)?\.gguf$', r'-\1', name)
    name = re.sub(r'(-\d+-of-\d+)?\.gguf$', '', name, flags=re.IGNORECASE)

    name = name.lower()

    # Map Claude model IDs to friendly versions
    name = re.sub(r'^claude-(.*)-4-20250514$', r'claude-\1-4.0', name)
    name = re.sub(r'^claude-(.*)-4-0$', r'claude-\1-4.0', name)
    name = re.sub(r'^claude-(.*)-4-6$', r'claude-\1-4.6', name)

    # Append reasoning effort if present
    if reasoning_effort:
        name = f"{name}-{reasoning_effort}"

    return name


def make_attempt_id(task: str, slug: str) -> str:
    """Generate a unique, sortable attempt ID: <task>_<slug>_YYYYMMDD-HHMMSS-<8hex>."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{task}_{slug}_{ts}-{secrets.token_hex(4)}"


def save_attempt_log(attempt_dir: Path, messages: list):
    """Save the full conversation transcript as messages.json."""
    (attempt_dir / "messages.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False)
    )
