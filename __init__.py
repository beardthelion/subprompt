"""SubPrompt — inline ``{{...}}`` resolution for Hermes Agent.

Resolves the *user's articulation gap*: when a user can't name a thing
precisely, ``{{the nginx thing for websockets}}`` is disambiguated to the
correct term (e.g. ``proxy_pass`` with ``Upgrade``/``Connection`` headers)
*before* the model anchors on the fumbled phrasing.

Integration: the ``pre_llm_call`` hook, which fires inside the agent turn
(off the gateway event loop, post-auth) and appends a fenced clarification
to the user message. See ~/.hermes/plans/subprompt-impl-plan.md.

All resolution logic lives in ``resolver.py`` (unit-tested); this module is
pure wiring.
"""

from __future__ import annotations

import logging

from .resolver import make_callback

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Plugin entrypoint: wire marker resolution onto ``pre_llm_call``."""
    ctx.register_hook("pre_llm_call", make_callback(ctx.llm))
    logger.info("SubPrompt plugin loaded — resolving {{markers}} via pre_llm_call")
