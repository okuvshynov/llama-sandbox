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
from dataclasses import dataclass
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


DOCKER_IMAGE = "vb-sandbox"
COMPILE_CMD = "clang++ -std=c++17 -O2"


class Sandbox:
    """Docker container sandbox for compiling and running untrusted code."""

    def __init__(self, startup_timeout: float = 600):
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
                 DOCKER_IMAGE, "sleep", "infinity"],
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

    def compile(self, source_code: str) -> tuple[bool, str]:
        """Copy source into container and compile. Returns (success, compiler_output)."""
        # Write source via stdin to avoid mount
        write = self._exec(["sh", "-c", "cat > /work/solution.cpp"],
                           input_data=source_code.encode())
        if write.returncode != 0:
            return False, f"Failed to write source: {write.stderr.decode()}"

        try:
            comp = self._exec(
                ["sh", "-c", f"cd /work && {COMPILE_CMD} -o solution solution.cpp"],
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out (30s limit)."

        return comp.returncode == 0, comp.stderr.decode()

    def run_binary(self, input_data: bytes) -> int:
        """Run /work/solution with input via stdin. Returns exit code (-1 on timeout)."""
        try:
            proc = self._exec(["/work/solution"], input_data=input_data, timeout=5)
            return proc.returncode
        except subprocess.TimeoutExpired:
            return -1


def run_tests(sandbox: Sandbox, tests: list[dict], task_dir: Path) -> tuple[str, ConfusionMatrix]:
    """Run all test cases against the binary in sandbox, return (output_text, matrix)."""
    matrix = ConfusionMatrix()
    lines = []

    for t in tests:
        input_data = (task_dir / t["input_file"]).read_bytes()

        tid = t.get("id", "?")
        label = t["label"]
        expected = t["expected"]

        rc = sandbox.run_binary(input_data)

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
    compiled, compiler_output = sandbox.compile(source_code)

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
