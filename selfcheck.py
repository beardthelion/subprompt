"""SubPrompt self-check: end-to-end probe + static config diagnostics.

Pure logic only — no host imports. The CLI wiring (building a PluginLlm) lives
in ``__main__.py``.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple, Optional

from resolver import RESOLVE_TIMEOUT, disambiguate

PROBE = "the nginx thing for websockets"


class SelfCheckResult(NamedTuple):
    term: Optional[str]
    latency: float
    ok: bool


def run_selfcheck(llm: Any, probe: str = PROBE, timeout: float = RESOLVE_TIMEOUT) -> SelfCheckResult:
    """Resolve a canned probe marker against ``llm`` and time it."""
    started = time.monotonic()
    term, _confidence = disambiguate(llm, probe, timeout=timeout)
    latency = time.monotonic() - started
    return SelfCheckResult(term=term, latency=latency, ok=term is not None)
