"""SubPrompt self-check: end-to-end probe + static config diagnostics.

Pure logic only — no host imports. The CLI wiring (building a PluginLlm) lives
in ``__main__.py``.
"""

from __future__ import annotations

import time
from typing import Any, List, NamedTuple, Optional

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


def check_trust_config(
    env_provider: Optional[str],
    env_model: Optional[str],
    trust_block: dict,
) -> List[str]:
    """Warn when an env override is set but the trust block won't honor it.

    ``trust_block`` is ``plugins.entries.subprompt.llm`` (or ``{}``). Returns a
    list of human-readable warnings; empty means no footgun detected.
    """
    warnings: List[str] = []
    if env_provider and not trust_block.get("allow_provider_override"):
        warnings.append(
            "SUBPROMPT_LLM_PROVIDER is set but allow_provider_override is not "
            "enabled in the trust block; the override will be silently ignored."
        )
    if env_model and not trust_block.get("allow_model_override"):
        warnings.append(
            "SUBPROMPT_LLM_MODEL is set but allow_model_override is not enabled "
            "in the trust block; the override will be silently ignored."
        )
    return warnings
