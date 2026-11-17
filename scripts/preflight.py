# -*- coding: utf-8 -*-
"""Pre-launch provider check: one minimal call per unique endpoint in a config.

Quota deaths keep arriving as surprises mid-run (Kimi 403 poisoned 6 records,
then a membership freeze took both keys to 402 overnight). One tiny call per
provider before launching turns "discover at pair 8" into "discover in 30s".

Usage:  python scripts/preflight.py configs/production.yaml
Exit code 0 = every provider answered; 1 = at least one failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examsociety.config import load_config
from examsociety.llm import LLMClient, LLMError, QuotaExhausted, UpstreamUnavailable


def main(config_path: str) -> int:
    cfg = load_config(config_path)
    specs = [
        cfg.roles.topic_mapper, *cfg.roles.proposers, *cfg.roles.validators,
        *cfg.roles.hardeners, *cfg.solvers,
    ]
    # one probe per unique upstream, not per seat
    unique = {}
    for m in specs:
        unique.setdefault((m.api, m.base_url, m.api_key_env, m.model), m)

    client = LLMClient()
    failures = 0
    for (api, base, key_env, model), m in unique.items():
        label = f"{m.id:16s} {api:10s} {model}"
        try:
            client.complete(m.ref(), "You are a health check.",
                            "reply with the single word OK", 400)
            print(f"  OK    {label}")
        except QuotaExhausted as e:
            print(f"  QUOTA {label} -- {str(e)[:110]}")
            failures += 1
        except (UpstreamUnavailable, LLMError) as e:
            print(f"  FAIL  {label} -- {str(e)[:110]}")
            failures += 1
    print()
    if failures:
        print(f"{failures} provider(s) failed -- fix or bench those seats before launching.")
        return 1
    print("all providers healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/production.yaml"))
