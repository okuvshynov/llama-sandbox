"""Tests for validation_bench helper functions."""

import pytest
import re

from validation_bench import derive_slug, make_attempt_id, InfraFailure


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
    ("openrouter/z-ai/glm-5.1", None, "openrouter-glm-5.1"),
    ("openrouter/anthropic/claude-opus-4", None, "openrouter-claude-opus-4"),
    # Strip "-latest" suffix
    ("mistral/devstral-latest", None, "devstral"),
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
