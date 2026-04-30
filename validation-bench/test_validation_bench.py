"""Tests for validation_bench helper functions and the scoring contract."""

import pytest
import re

from validation_bench_lib import (
    derive_slug, make_attempt_id, InfraFailure,
    _fail_detail, run_tests,
)


@pytest.mark.parametrize("model, effort, expected", [
    # Anthropic — provider prefix dropped (identity implicit in model name)
    ("anthropic/claude-opus-4-6", None, "claude-opus-4.6"),
    ("anthropic/claude-sonnet-4-20250514", None, "claude-sonnet-4.0"),
    ("anthropic/claude-sonnet-4-0", None, "claude-sonnet-4.0"),
    # OpenAI — prefix dropped
    ("openai/gpt-5.3-codex", None, "gpt-5.3-codex"),
    ("openai/gpt-5.3-codex", "high", "gpt-5.3-codex-high"),
    ("openai/gpt-5.3-codex", "low", "gpt-5.3-codex-low"),
    # Hosted OSS — keep short provider tag so we know who served it
    ("fireworks_ai/accounts/fireworks/models/glm-5p1", None, "fireworks-glm-5p1"),
    ("fireworks/accounts/fireworks/models/llama-v3-70b", None, "fireworks-llama-v3-70b"),
    # GGUF filenames — strip quantization/shard suffixes
    ("openai/Qwen3.5-122B-A10B-UD-Q8_K_XL-00001-of-00004.gguf", None, "qwen3.5-122b-a10b-q8_k_xl"),
    ("openai/some-model.gguf", None, "some-model"),
    # Bare model name (local)
    ("openai/qwen2.5-coder-32b", None, "qwen2.5-coder-32b"),
    ("some-model", None, "some-model"),
])
def test_derive_slug(model, effort, expected):
    assert derive_slug(model, effort) == expected


def test_make_attempt_id_format():
    """Attempt IDs follow <task>_<slug>_YYYYMMDD-HHMMSS-<4hex>."""
    aid = make_attempt_id("toml-1.1-cpp", "gpt-5.3-codex")
    assert re.fullmatch(r"toml-1\.1-cpp_gpt-5\.3-codex_\d{8}-\d{6}-[0-9a-f]{8}", aid)


def test_make_attempt_id_unique():
    """Back-to-back IDs differ even within the same second."""
    ids = {make_attempt_id("t", "s") for _ in range(50)}
    assert len(ids) == 50


def test_infra_failure_dataclass():
    """InfraFailure stores error details."""
    f = InfraFailure(
        timestamp="2025-01-01T00:00:00+00:00",
        turn=2,
        error_type="api_error",
        error_message="Connection refused",
    )
    assert f.turn == 2
    assert f.error_type == "api_error"


# ----------------------------------------------------------------------
# Scoring contract (vb_version 0.0.9):
#   passed iff stdout.strip() == expected ('valid' or 'invalid', exact
#   case) AND rc == 0. Anything else fails. _fail_detail composes the
#   feedback line that goes back to the model.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("rc, stdout, verdict, expected_substr", [
    # Got a verdict, clean exit, but it doesn't match the expected label —
    # detail is just the verdict string.
    (0, b"valid",   "valid",   "got 'valid'"),
    (0, b"invalid", "invalid", "got 'invalid'"),
    # Got the right verdict, but rc != 0 — the 0.0.9 tightening.
    (1,   b"valid",   "valid",   "got 'valid' but exited with rc=1"),
    (2,   b"invalid", "invalid", "got 'invalid' but exited with rc=2"),
    (124, b"valid",   "valid",   "got 'valid' but timed out"),
    (-1,  b"valid",   "valid",   "got 'valid' but timed out"),
    (137, b"valid",   "valid",   "got 'valid' but killed by signal 9"),
    (139, b"invalid", "invalid", "got 'invalid' but killed by signal 11"),
    # No verdict matched — diagnose by exit code.
    (124, b"",        None, "timeout (no verdict printed)"),
    (-1,  b"",        None, "timeout (no verdict printed)"),
    (137, b"",        None, "killed by signal 9 (no verdict printed)"),
    (139, b"",        None, "killed by signal 11 (no verdict printed)"),
    (0,   b"",        None, "no output (exit=0)"),
    (1,   b"",        None, "no output (exit=1)"),
    # No verdict, but stdout has unrelated content.
    (0,   b"Valid",   None, "got 'Valid'"),
    (0,   b"VALID",   None, "got 'VALID'"),
    (0,   b"yes",     None, "got 'yes'"),
])
def test_fail_detail(rc, stdout, verdict, expected_substr):
    detail = _fail_detail(rc, stdout, verdict)
    assert expected_substr in detail


def test_fail_detail_truncates_long_output():
    """Long stdout should be previewed, not dumped wholesale."""
    long_text = b"x" * 200
    detail = _fail_detail(0, long_text, None)
    assert len(detail) < 200  # reasonably bounded
    assert "x" in detail


class _FakeSandbox:
    """Sandbox stub for run_tests: returns predetermined (rc, stdout)
    sequentially, one per call, in the order tests are processed."""
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def run_input(self, input_data):
        rc, stdout = self._responses[self._idx]
        self._idx += 1
        return rc, stdout


def _make_tests(tmp_path, expected_seq):
    """Build a tests list (with on-disk input files) of the given length;
    expected_seq is a list of 'valid'/'invalid' labels per test."""
    out = []
    for i, expected in enumerate(expected_seq):
        f = tmp_path / f"test{i}.bin"
        f.write_bytes(b"x")  # content doesn't matter — _FakeSandbox ignores it
        out.append({
            "id": f"t{i}",
            "label": f"label-{i}",
            "expected": expected,
            "input_file": f"test{i}.bin",
        })
    return out


def test_scoring_clean_pass(tmp_path):
    """Right verdict + clean exit → pass."""
    tests = _make_tests(tmp_path, ["valid", "invalid"])
    sb = _FakeSandbox([(0, b"valid"), (0, b"invalid")])
    _, m = run_tests(sb, tests, tmp_path)
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 0, 0, 1)


def test_scoring_wrong_verdict(tmp_path):
    """Clean exit but the verdict is the opposite of expected → fail."""
    tests = _make_tests(tmp_path, ["valid", "invalid"])
    sb = _FakeSandbox([(0, b"invalid"), (0, b"valid")])
    _, m = run_tests(sb, tests, tmp_path)
    assert (m.tp, m.fn, m.fp, m.tn) == (0, 1, 1, 0)


def test_scoring_right_verdict_dirty_exit(tmp_path):
    """Right verdict but rc != 0 → fail (the 0.0.9 tightening)."""
    tests = _make_tests(tmp_path, ["valid", "invalid"])
    sb = _FakeSandbox([
        (1, b"valid"),       # right verdict, exit=1
        (139, b"invalid"),   # right verdict, SIGSEGV
    ])
    _, m = run_tests(sb, tests, tmp_path)
    assert (m.tp, m.fn, m.fp, m.tn) == (0, 1, 1, 0)


def test_scoring_right_verdict_timeout(tmp_path):
    """Right verdict followed by an infinite loop (rc=124) → fail."""
    tests = _make_tests(tmp_path, ["valid"])
    sb = _FakeSandbox([(124, b"valid")])
    _, m = run_tests(sb, tests, tmp_path)
    assert m.fn == 1


def test_scoring_timeout_no_output(tmp_path):
    """Timeout with no stdout → fail regardless of expected label.
    This is the loophole the print-based contract closes — under the old
    exit-code rule, rc != 0 on an 'invalid' test counted as TN."""
    tests = _make_tests(tmp_path, ["valid", "invalid"])
    sb = _FakeSandbox([(124, b""), (124, b"")])
    _, m = run_tests(sb, tests, tmp_path)
    assert m.passed == 0
    assert (m.tp, m.fn, m.fp, m.tn) == (0, 1, 1, 0)


def test_scoring_signal_kill_no_verdict(tmp_path):
    """SIGSEGV with no stdout → fail regardless of expected label."""
    tests = _make_tests(tmp_path, ["invalid"])
    sb = _FakeSandbox([(139, b"")])
    _, m = run_tests(sb, tests, tmp_path)
    assert m.tn == 0
    assert m.fp == 1


def test_scoring_whitespace_tolerated(tmp_path):
    """Surrounding whitespace and trailing newlines are stripped."""
    tests = _make_tests(tmp_path, ["valid", "invalid"])
    sb = _FakeSandbox([(0, b"  valid  \n"), (0, b"\ninvalid\n")])
    _, m = run_tests(sb, tests, tmp_path)
    assert (m.tp, m.tn) == (1, 1)


def test_scoring_case_strict(tmp_path):
    """Case mismatch fails — `Valid` ≠ `valid` under the contract."""
    tests = _make_tests(tmp_path, ["valid"])
    sb = _FakeSandbox([(0, b"Valid")])
    _, m = run_tests(sb, tests, tmp_path)
    assert m.tp == 0
    assert m.fn == 1


def test_scoring_extra_content_fails(tmp_path):
    """Stdout with debug noise alongside the verdict fails — the entire
    stripped stdout must equal `valid` or `invalid`."""
    tests = _make_tests(tmp_path, ["valid"])
    sb = _FakeSandbox([(0, b"debug stuff\nvalid")])
    _, m = run_tests(sb, tests, tmp_path)
    assert m.fn == 1


def test_scoring_empty_stdout_fails(tmp_path):
    """Empty stdout (clean exit) fails: no verdict was printed."""
    tests = _make_tests(tmp_path, ["valid", "invalid"])
    sb = _FakeSandbox([(0, b""), (0, b"")])
    _, m = run_tests(sb, tests, tmp_path)
    assert m.passed == 0


def test_run_tests_fail_lines_carry_detail(tmp_path):
    """run_tests' returned text has FAIL lines with semantic detail —
    that's what gets surfaced to the model in tool feedback."""
    tests = _make_tests(tmp_path, ["valid", "invalid", "valid"])
    sb = _FakeSandbox([
        (0, b"invalid"),       # wrong verdict, clean exit
        (124, b""),            # timeout, no output
        (139, b"valid"),       # right verdict but SIGSEGV
    ])
    output, _ = run_tests(sb, tests, tmp_path)
    assert "got 'invalid'" in output                # tests[0]
    assert "timeout (no verdict printed)" in output  # tests[1]
    assert "got 'valid' but killed by signal 11" in output  # tests[2]
