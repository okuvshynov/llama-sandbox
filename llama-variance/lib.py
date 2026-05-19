#!/usr/bin/env python3
"""Sandbox + scoring primitives for llama-variance.

Lifted in spirit from validation-bench's validation_bench_lib but deliberately
trimmed and forked — this project studies sampling variance, not multi-turn
agentic behavior, so the multi-attempt / multi-provider machinery has been
stripped out. The pieces that remain are the docker sandbox, the `valid` /
`invalid` strict-verdict scoring contract, and a single-call task loader
that composes preamble + spec_body the same way validation-bench does.

If validation-bench's scoring contract evolves, this file does NOT
follow automatically — by design. The two projects are allowed to drift.
"""
import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


HERE = Path(__file__).resolve().parent


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
    prepare_seconds: float = 0.0
    tests_seconds: float = 0.0


@dataclass
class TaskConfig:
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
    "spec", "env", "test_timeout_seconds", "prepare_timeout_seconds",
}


SEPARATOR = "\n\n---\n\n"


def load_task(task_name: str) -> tuple[TaskConfig, str, Path]:
    """Load (config, rendered_prompt, tests_root) for a (spec, env) cell.

    `task_name` is a directory name under data/tasks/, e.g. "toml-1.0-cpp17".
    Only the composed shape is supported (task.json carries {spec, env}; env
    meta comes from data/envs/<env>/meta.json; preamble + spec_body compose
    the prompt). tests_root is data/specs/<spec>/.
    """
    task_dir = HERE / "data" / "tasks" / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task dir not found: {task_dir}")
    task_meta = json.loads((task_dir / "task.json").read_text())
    spec_name = task_meta["spec"]
    env_name = task_meta["env"]

    env_meta = json.loads(
        (HERE / "data" / "envs" / env_name / "meta.json").read_text())
    spec_dir = HERE / "data" / "specs" / spec_name
    spec_meta = json.loads((spec_dir / "meta.json").read_text())
    body_path = spec_dir / "spec_body.md"
    spec_body = body_path.read_text() if body_path.exists() else ""

    # Compose TaskConfig from env meta; extras carry forward fields like
    # compile_cmd for {placeholder} substitution in the preamble.
    extras = {
        k: v for k, v in env_meta.items() if k not in _TASK_CONFIG_KNOWN
    }
    config = TaskConfig(
        language=env_meta["language"],
        docker_image=env_meta["docker_image"],
        source_filename=env_meta["source_filename"],
        prepare_cmd=env_meta.get("prepare_cmd"),
        run_cmd=env_meta["run_cmd"],
        spec=spec_name,
        env=env_name,
        test_timeout_seconds=env_meta.get("test_timeout_seconds", 5.0),
        prepare_timeout_seconds=env_meta.get("prepare_timeout_seconds", 30.0),
        extra=extras,
    )

    preamble = (task_dir / "preamble.md").read_text()
    if spec_meta.get("has_spec_body", True) and spec_body:
        prompt = preamble.rstrip("\n") + SEPARATOR + spec_body
    else:
        prompt = preamble

    # {placeholder} substitution from config fields + extras.
    subs = {
        "language": config.language,
        "source_filename": config.source_filename,
        "prepare_cmd": config.prepare_cmd or "",
        "run_cmd": config.run_cmd,
    }
    subs.update(config.extra)
    for k, v in subs.items():
        prompt = prompt.replace(f"{{{k}}}", str(v))

    return config, prompt, spec_dir


def load_tests(tests_file: Path) -> list[dict]:
    tests = []
    with open(tests_file) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


class Sandbox:
    """Docker container sandbox for compile + per-test stdin/stdout runs.

    Mirrors validation-bench's flags (network=none, memory=512m, cpus=1,
    pids-limit=256, read-only with tmpfs /work + /tmp). begin_submission()
    restarts the container on every submission after the first to avoid
    pids-cgroup saturation from any zombies / threads the prior submission
    left behind — same reasoning as validation-bench.
    """

    def __init__(self, config: TaskConfig, startup_timeout: float = 600):
        self.config = config
        self.container_id: str | None = None
        self.startup_timeout = startup_timeout
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
                f"`docker run` did not return within {self.startup_timeout}s")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start sandbox: {result.stderr}")
        self.container_id = result.stdout.strip()

    def stop(self):
        if self.container_id:
            subprocess.run(["docker", "kill", self.container_id],
                           capture_output=True)
            self.container_id = None

    def begin_submission(self):
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

    def prepare(self, source_code: str) -> tuple[bool, str]:
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
            return False, (f"Preparation timed out "
                           f"({self.config.prepare_timeout_seconds:g}s limit).")
        return comp.returncode == 0, (comp.stdout + comp.stderr).decode()

    def run_input(self, input_data: bytes) -> tuple[int, bytes]:
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


def run_tests(sandbox: Sandbox, tests: list[dict],
              tests_root: Path) -> tuple[str, ConfusionMatrix]:
    """Score solution against the corpus. A test passes iff stdout.strip()
    equals the expected verdict literal AND rc == 0. Same contract as
    validation-bench vb_version 0.0.9+."""
    matrix = ConfusionMatrix()
    lines = []
    for t in tests:
        input_data = (tests_root / t["input_file"]).read_bytes()
        expected = t["expected"]
        rc, stdout = sandbox.run_input(input_data)
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
    lines.append(f"{matrix.passed}/{matrix.total} passed")
    return "\n".join(lines), matrix


def handle_submit(source_code: str, tests: list[dict], sandbox: Sandbox,
                  tests_root: Path) -> TestResult:
    sandbox.begin_submission()
    t0 = time.perf_counter()
    compiled, compiler_output = sandbox.prepare(source_code)
    prepare_seconds = time.perf_counter() - t0
    if not compiled:
        return TestResult(
            compiled=False, compiler_output=compiler_output, test_output="",
            matrix=ConfusionMatrix(), prepare_seconds=prepare_seconds,
            tests_seconds=0.0,
        )
    t1 = time.perf_counter()
    test_output, matrix = run_tests(sandbox, tests, tests_root)
    tests_seconds = time.perf_counter() - t1
    return TestResult(
        compiled=True, compiler_output=compiler_output, test_output=test_output,
        matrix=matrix, prepare_seconds=prepare_seconds,
        tests_seconds=tests_seconds,
    )


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
