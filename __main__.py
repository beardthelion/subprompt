"""``python -m subprompt selfcheck``: verify this box will resolve markers."""

from __future__ import annotations

import os
import sys

# Run as ``python -m subprompt`` the package dir is not on sys.path, so the
# absolute imports below (and selfcheck's own ``from resolver import``) need it.
# Mirrors what tests/conftest.py does for the test run.
sys.path.insert(0, os.path.dirname(__file__))

from resolver import RESOLVE_TIMEOUT  # noqa: E402
from selfcheck import PROBE, check_trust_config, load_trust_block, run_selfcheck  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "selfcheck":
        print("usage: python -m subprompt selfcheck")
        return 2

    env_provider = os.getenv("SUBPROMPT_LLM_PROVIDER")
    env_model = os.getenv("SUBPROMPT_LLM_MODEL")

    print("SubPrompt self-check")
    print(f"  provider override : {env_provider or '(unset)'}")
    print(f"  model override    : {env_model or '(unset)'}")

    trust = load_trust_block()
    if trust is None:
        print("  trust block       : (host config not readable, static check skipped)")
    else:
        warnings = check_trust_config(env_provider, env_model, trust)
        if warnings:
            print("  trust block       : OVERRIDE WILL BE IGNORED")
            for w in warnings:
                print(f"    ! {w}")
        else:
            print("  trust block       : override allowed")

    try:
        from agent.plugin_llm import PluginLlm
    except Exception as exc:  # host not importable
        print(f"  probe             : cannot run (host not importable: {exc})")
        return 1

    llm = PluginLlm(plugin_id="subprompt")
    result = run_selfcheck(llm, timeout=RESOLVE_TIMEOUT)
    print(f'  probe "{PROBE}"')
    if result.ok:
        print(f'    -> "{result.term}"  ({result.latency:.1f}s)')
        print("  OK")
        return 0
    print(f"    -> unresolved  ({result.latency:.1f}s)")
    print("  FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
