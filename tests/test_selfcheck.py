"""Tests for the SubPrompt self-check."""

from types import SimpleNamespace

from pathlib import Path

from selfcheck import (
    check_trust_config,
    is_oauth_provider,
    load_trust_block,
    run_selfcheck,
)


def _text(text):
    return SimpleNamespace(text=text)


class TestRunSelfcheck:
    def test_resolved_probe_is_ok(self):
        llm = SimpleNamespace(complete=lambda *a, **kw: _text("WebSocket proxy"))
        result = run_selfcheck(llm)
        assert result.ok is True
        assert result.term == "WebSocket proxy"
        assert result.latency >= 0.0

    def test_unresolved_probe_not_ok(self):
        llm = SimpleNamespace(complete=lambda *a, **kw: _text("NONE"))
        result = run_selfcheck(llm)
        assert result.ok is False
        assert result.term is None

    def test_raising_llm_not_ok(self):
        def boom(*a, **kw):
            raise RuntimeError("model down")

        result = run_selfcheck(SimpleNamespace(complete=boom))
        assert result.ok is False


class TestCheckTrustConfig:
    def test_provider_override_blocked(self):
        warns = check_trust_config("opencode-go", None, {})
        assert any("allow_provider_override" in w for w in warns)

    def test_model_override_blocked(self):
        warns = check_trust_config(None, "deepseek-v4-flash", {})
        assert any("allow_model_override" in w for w in warns)

    def test_overrides_allowed_no_warnings(self):
        warns = check_trust_config(
            "opencode-go",
            "deepseek-v4-flash",
            {"allow_provider_override": True, "allow_model_override": True},
        )
        assert warns == []

    def test_no_env_no_warnings(self):
        assert check_trust_config(None, None, {}) == []


class TestLoadTrustBlock:
    def test_reads_nested_block(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "plugins:\n"
            "  entries:\n"
            "    subprompt:\n"
            "      llm:\n"
            "        allow_provider_override: true\n"
        )
        block = load_trust_block(cfg)
        assert block == {"allow_provider_override": True}

    def test_missing_file_returns_none(self, tmp_path):
        assert load_trust_block(tmp_path / "nope.yaml") is None

    def test_block_absent_returns_empty(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("plugins:\n  enabled:\n    - subprompt\n")
        assert load_trust_block(cfg) == {}


class TestIsOauthProvider:
    def test_oauth_device_code(self):
        assert is_oauth_provider("oauth_device_code") is True

    def test_oauth_external(self):
        assert is_oauth_provider("oauth_external") is True

    def test_api_key(self):
        assert is_oauth_provider("api_key") is False

    def test_none(self):
        assert is_oauth_provider(None) is False
