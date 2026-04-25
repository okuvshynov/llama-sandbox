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
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# Stamped into every results row + failures row so past runs can be filtered
# by harness version if a scoring/prompt/behavior change later breaks
# comparability. Bump on: prompt template changes, scoring logic changes,
# tool schema changes, sandbox compile flags, or any other change that
# could affect the numbers for a given (task, model, slug) triple.
# Rows without this field predate the scheme — treat as "pre-0.0.2".
VB_VERSION = "0.0.2"


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


@dataclass
class Submission:
    turn: int
    matrix: ConfusionMatrix | None = None  # None for failed submissions
    error: str | None = None  # e.g. "compile_error", "compile_timeout"


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
    """
    language: str
    docker_image: str
    source_filename: str
    prepare_cmd: str | None
    run_cmd: str
    test_timeout_seconds: float = 5.0
    prepare_timeout_seconds: float = 30.0
    extra: dict = field(default_factory=dict)


_TASK_CONFIG_KNOWN = {
    "language", "docker_image", "source_filename", "prepare_cmd", "run_cmd",
    "test_timeout_seconds", "prepare_timeout_seconds",
}


def load_task_config(task_dir: Path) -> TaskConfig:
    """Read task.json from a task directory."""
    config_file = task_dir / "task.json"
    if not config_file.exists():
        raise FileNotFoundError(f"task.json not found in {task_dir}")
    data = json.loads(config_file.read_text())
    extras = {k: v for k, v in data.items() if k not in _TASK_CONFIG_KNOWN}
    return TaskConfig(
        language=data["language"],
        docker_image=data["docker_image"],
        source_filename=data["source_filename"],
        prepare_cmd=data.get("prepare_cmd"),
        run_cmd=data["run_cmd"],
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

    def _exec(self, cmd: list[str], input_data: bytes | None = None,
              timeout: float = 30) -> subprocess.CompletedProcess:
        full_cmd = ["docker", "exec"]
        if input_data is not None:
            full_cmd.append("-i")
        full_cmd.extend([self.container_id] + cmd)
        return subprocess.run(full_cmd, input=input_data,
                              capture_output=True, timeout=timeout)

    def prepare(self, source_code: str) -> tuple[bool, str]:
        """Write source into container and run prepare_cmd. Returns (success, output).

        Equivalent to "compile" for compiled languages (clang++, go build) and to a
        syntax check for interpreters that have one (luac -p, py_compile). When
        prepare_cmd is None, only the file write happens and we report success.
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
            return False, f"Preparation timed out ({self.config.prepare_timeout_seconds:g}s limit)."

        return comp.returncode == 0, comp.stderr.decode()

    def run_input(self, input_data: bytes) -> int:
        """Run config.run_cmd in /work with input via stdin. Returns exit code (-1 on timeout)."""
        try:
            proc = self._exec(
                ["sh", "-c", f"cd /work && {self.config.run_cmd}"],
                input_data=input_data,
                timeout=self.config.test_timeout_seconds,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            return -1


def run_tests(sandbox: Sandbox, tests: list[dict], task_dir: Path) -> tuple[str, ConfusionMatrix]:
    """Run all test cases against the prepared solution in sandbox, return (output_text, matrix)."""
    matrix = ConfusionMatrix()
    lines = []

    for t in tests:
        input_data = (task_dir / t["input_file"]).read_bytes()

        tid = t.get("id", "?")
        label = t["label"]
        expected = t["expected"]

        rc = sandbox.run_input(input_data)

        passed = (expected == "valid" and rc == 0) or (expected == "invalid" and rc != 0)
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
            lines.append(f"FAIL {tid}: {label} (exit={rc}, expected {expected})")

    lines.append(f"{matrix.passed}/{matrix.total} passed")
    return "\n".join(lines), matrix


def handle_submit(source_code: str, tests: list[dict], sandbox: Sandbox, task_dir: Path) -> TestResult:
    compiled, compiler_output = sandbox.prepare(source_code)

    if not compiled:
        return TestResult(
            compiled=False,
            compiler_output=compiler_output,
            test_output="",
            matrix=ConfusionMatrix(),
        )

    test_output, matrix = run_tests(sandbox, tests, task_dir)

    return TestResult(
        compiled=True,
        compiler_output=compiler_output,
        test_output=test_output,
        matrix=matrix,
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
