# -*- coding: utf-8 -*-
"""Smoke-test the PRODUCTION traj/agent.py rollout() end to end, for real, on
arvo:10400 -- this is the "get agent.py's rollout() running at least once"
item from CLAUDE.md's next-steps list. run_one.py deliberately bypassed
examsociety (and its budget cap) to sidestep dependency issues on 2026-08-27;
this script is the one that actually exercises the loop meant for production.

One LLMClient, one shared call budget, several models run SEQUENTIALLY against
the same task dir -- BudgetExhausted stops the whole smoke run cleanly instead
of one rollout silently burning the rest of the cap.

Non-GPT only (2026-10-08 instruction): kimi-k3 and glm-5.3 through the one
s-flag gateway key as the strong seats. This is a rollout()-works-at-all check,
not the real batch, so it stays to one or two cheap cloud seats.

Usage (cyber conda env, CyberGym server already running):
    set -a && source .env && set +a
    python traj/smoke_rollout.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examsociety.llm import BudgetExhausted, LLMClient, ModelRef  # noqa: E402

from traj.agent import rollout  # noqa: E402
from traj.env import CyberGymEnv  # noqa: E402

TASK_DIR = os.environ.get("CYBERGYM_TASK_DIR", "/root/cybergym-src/cybergym_tmp")
MAX_CALLS = int(os.environ.get("MAX_CALLS", "200"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "15"))

_GW = "https://yyt.s-flag.com/v1"  # s-flag gateway (already purchased)

MODELS = [
    # kimi 400s on temperature != 1 through the gateway; glm-5.3 is an always-on
    # thinking model and needs {"thinking":{"type":"disabled"}} or it returns an
    # empty `content` (see batch_rollout.py for the full probe notes).
    ModelRef(id="kimi-k3", model="kimi-k3", api="openai", base_url=_GW,
             api_key_env="SFLAG_KEY", family="kimi",
             extra_body={"temperature": 1}),
    ModelRef(id="glm-5.3", model="glm-5.3", api="openai", base_url=_GW,
             api_key_env="SFLAG_KEY", family="glm",
             extra_body={"thinking": {"type": "disabled"}}),
    # claude-sonnet-5 (claude-cli) deliberately excluded from traj/ rollouts:
    # the claude-cli protocol runs a full Claude Code CLI session with its own
    # system identity, which correctly treats the harness's wrapped
    # <instructions> prompt as a prompt-injection pattern rather than
    # adopting the agent persona -- see memory/project_agent-rollout-smoke-
    # test.md. Not fixable by rewording; would need a real Anthropic API key
    # on the plain `api: "anthropic"` protocol instead. Set aside for now
    # (2026-09-02 decision) -- revisit only if that key becomes available.
]


def main() -> None:
    llm = LLMClient(max_calls=MAX_CALLS)
    out_dir = Path(__file__).parent / "smoke_out"
    out_dir.mkdir(exist_ok=True)

    wanted = os.environ.get("MODEL_IDS")
    models = [m for m in MODELS if m.id in wanted.split(",")] if wanted else MODELS

    for ref in models:
        print(f"\n=== {ref.id} ({ref.model}) ===", flush=True)

        # The task dir is reused across models sequentially; clear any PoC left
        # by a previous model so verify() can never credit this model with a
        # crash it did not cause.
        poc_path = Path(TASK_DIR) / "poc"
        poc_path.unlink(missing_ok=True)

        env = CyberGymEnv(TASK_DIR)
        try:
            traj = rollout(env, llm, ref, max_steps=MAX_STEPS)
        except BudgetExhausted as e:
            print(f"BUDGET EXHAUSTED, stopping smoke run: {e}")
            break

        out_path = out_dir / f"trajectory_{ref.id}.json"
        out_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        print(f"steps={traj.n_steps} stop={traj.stop_reason} tier={traj.tier} "
              f"success={traj.success} vul={traj.vul_exit_code} fix={traj.fix_exit_code}")
        print(f"detail: {traj.verifier_detail}")
        print(f"-> {out_path}")

    print(f"\ntotal LLM calls used: {llm.calls}/{MAX_CALLS}")


if __name__ == "__main__":
    main()
