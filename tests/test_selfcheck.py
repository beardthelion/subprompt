"""Tests for the SubPrompt self-check."""

from types import SimpleNamespace

from selfcheck import run_selfcheck


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
