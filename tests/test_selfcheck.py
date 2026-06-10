"""Tests for the SubPrompt self-check."""

from types import SimpleNamespace

from selfcheck import check_trust_config, run_selfcheck


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
