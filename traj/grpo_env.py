"""TRL GRPOTrainer environment adapter for CyberGym.

TRL 1.13's GRPOTrainer(environment_factory=...) expects an object whose public
methods become model-callable tools (reflected via type hints + docstring),
plus a `reset(**kwargs)` -- called once per rollout, kwargs coming straight
from the training dataset row -- and an optional `get_reward()` that owns the
reward for that rollout.

Critical constraint found reading grpo_trainer.py directly (2026-10-03): the
trainer POOLS and REUSES environment instances across a batch -- one instance
is `reset()` for task A, used for a whole rollout, then later `reset()` again
for task B. So unlike CyberGymEnv (one instance = one fixed task_dir, set up
in __init__), this adapter must tear down and rebuild its CyberGymEnv on every
reset(), not just once at construction.

This wraps traj.env.CyberGymEnv rather than reimplementing it, so the tool
behavior (read_file/grep/list_dir/write_poc/submit) and the verify() judgment
chain stay byte-for-byte the same object other code already trusts -- this
file only adds the reset/pooling/reward-float layer TRL's interface requires.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from .batch_rollout import gen_task
from .env import CyberGymEnv
from .schemas import TIER_DEGENERATE, TIER_INFRA_ERROR, TIER_ORDER

# Reward scale for GRPO's group-relative advantage. Ordinal, matching
# preference.TIER_ORDER's ranking (solved > crashed_unclear_fix > no_crash >
# degenerate) -- NOT preference.rank_score itself, because rank_score's
# within-tier bonus needs the full Trajectory (step count, which action was
# reached); get_reward() here only has the final Verdict. Coarser is fine for
# a first smoke test: the group-relative normalization only needs tiers to be
# orderable and gapped, not an exact replica of the DPO ranking heuristic.
_TIER_REWARD = {tier: float(len(TIER_ORDER) - i) for i, tier in enumerate(TIER_ORDER)}
# {"solved": 4.0, "crashed_unclear_fix": 3.0, "no_crash": 2.0, "degenerate": 1.0}
# infra_error (verify() couldn't render ANY verdict -- not the model's doing,
# see schemas.TIER_INFRA_ERROR) must NOT fall through .get()'s 0.0 default:
# that would score it below even "wrote nothing" (degenerate=1.0), training
# the model to avoid episodes where the grader broke as if that were a worse
# behavior than giving up. Pin it level with the worst REAL behavioral tier
# instead -- neutral-worst, not worse-than-worst. (2026-10-08, review P1.)
_TIER_REWARD[TIER_INFRA_ERROR] = _TIER_REWARD[TIER_DEGENERATE]

# --- Process reward (ref: XekRung "Agentic RL + 过程奖励") -------------------
# The terminal tier reward is FLAT across a group whenever every rollout lands
# in the same tier (the common case on near-floor tasks: all degenerate, or all
# no_crash). GRPO's group-relative advantage is then ~0 and nothing is learned.
# A small, bounded process bonus gives WITHIN-tier variance so the group has
# signal even when the terminal outcome is identical. Bounded to < 1.0 (the
# tier gap) so it NEVER reorders tiers: a shaped degenerate still ranks below
# a shaped no_crash. Off by default (GRPO_REWARD_MODE=terminal); set
# GRPO_REWARD_MODE=shaped to enable. Delivered through the existing single
# get_reward()->float, so NO GPU-trainer change is needed to use it.
#
# 2026-10-27 redesign (see docs/FINDINGS-2026-10-23-grpo-protocol-fix.md,
# 10-26 section): the original version also rewarded WROTE_POC and EXPLORE --
# i.e. it paid for LOOKING complete (called write_poc, tried several action
# types) with no read on whether the PoC actually did anything. That is
# confirmed to move degenerate->no_crash (reproduced across two configs) but
# never no_crash->solved, because it cannot see real PoC quality -- it is
# rewarding behavioral completeness, not correctness, which is exactly the
# shape of a reward-hacking surface (submit something that LOOKS like a
# finished attempt to collect the bonus, regardless of whether it crashes
# anything). Both components are removed. NONEMPTY is removed for the same
# reason (non-empty garbage still collects it). What remains is only the
# anti-repeat penalty, at half its previous weight -- a pure penalty (this
# bonus is now <= 0), just big enough to discourage looping on one action
# forever (the 86%-of-zero-variance-groups failure mode from 10-26: explores
# but never converges to submit) without paying out for surface completeness.
# The real quality signal stays where it already was and always should be --
# the terminal tier reward -- and the plan for making THAT differentiate more
# often inside a group is Dynamic Sampling (oversample per task, discard
# zero-variance groups), not a richer bonus. That part is GPU-trainer-side
# (grpo_train_v1.py's group construction) and is intentionally NOT done here;
# it is next-up work for when GRPO resumes after the 27B DPO stage, not
# something this bonus function can do on its own.
_REWARD_MODE = os.environ.get("GRPO_REWARD_MODE", "terminal")
_W_REPEAT_PEN = 0.10     # max penalty for repeating the same action


class CyberGymGRPOEnv:
    """One pooled slot. reset(task_id=...) tears down any previous episode's
    CyberGymEnv and builds a fresh one; get_reward() asks it to verify()."""

    def __init__(self, work_root: str = "/data/traj_batch/grpo_work"):
        self._work_root = Path(work_root)
        self._work_root.mkdir(parents=True, exist_ok=True)
        self._env: Optional[CyberGymEnv] = None
        self._work: Optional[Path] = None
        self._done = False
        self._last_verdict_tier: Optional[str] = None
        self._reward_mode = os.environ.get("GRPO_REWARD_MODE", _REWARD_MODE)
        self._reset_progress()
        self.last_reward_breakdown: dict = {}

    def _reset_progress(self) -> None:
        self._sigs: list[str] = []       # action signatures in call order
        self._wrote_poc = False          # write_poc was called
        self._wrote_nonempty = False     # write_poc with non-empty content

    def reset(self, task_id: str) -> str:
        """Start a new episode on `task_id` (e.g. "arvo:3938"). Tears down the
        previous episode's working directory first -- this instance is reused
        across many different tasks over a training run, not one-shot."""
        self._teardown()
        work = self._work_root / f"{task_id.replace(':', '-')}__{uuid.uuid4().hex[:8]}"
        gen_task(task_id, work)
        self._env = CyberGymEnv(str(work))
        self._work = work
        self._done = False
        self._reset_progress()
        return self._env.task().prompt

    def get_reward(self) -> float:
        """Environment-anchored reward: verify() runs the real CyberGym
        docker judge (vul/fix exit codes), never a model self-assessment.

        KNOWN LIMITATION (2026-10-08, round-2 review P1): pinning
        infra_error's reward level with degenerate (see _TIER_REWARD above)
        stops it from scoring WORSE than every real behavioral tier, but it
        still injects a fake reward contrast into a group whose other
        members are genuinely homogeneous -- e.g. three real no_crash results
        plus one infra_error still produces non-zero group advantage where
        there is no real signal to learn from. Fully fixing this needs the
        GPU-side trainer to compute group mean/std from valid rollouts only
        and mask the loss on invalid ones -- that trainer code is not in this
        repo (TRL's GRPOTrainer(environment_factory=...) interface here only
        has a single get_reward() -> float; there is no in-protocol way to
        also say "and this one doesn't count"). reward_is_valid() below is
        the hook a trainer-side change would need; it does nothing on its
        own until something on the GPU side actually reads it.
        """
        assert self._env is not None, "get_reward() called before reset()"
        verdict = self._env.verify()
        self._last_verdict_tier = verdict.tier
        terminal = _TIER_REWARD.get(verdict.tier, 0.0)
        # Always COMPUTE the shaped bonus (when the verdict is real) so it is
        # available as a monitoring metric even when reward_mode=="terminal"
        # -- only whether it's ADDED to the returned training reward depends
        # on the mode. Before this fix, "terminal" mode skipped computing it
        # entirely, so last_reward_breakdown always showed bonus=0.0 and
        # there was nothing to watch (2026-10-20, GRPO reward-design review:
        # train on terminal-only first, keep bonus as a logged diagnostic to
        # ablate later, not blind to it).
        bonus = 0.0
        if verdict.tier != TIER_INFRA_ERROR:
            bonus = self._process_bonus()
        applied = bonus if self._reward_mode == "shaped" else 0.0
        n_actions = len(self._sigs)
        self.last_reward_breakdown = {
            "tier": verdict.tier, "terminal": terminal,
            "bonus": round(bonus, 4), "bonus_applied": round(applied, 4),
            "mode": self._reward_mode,
            # per-component diagnostics (named, not buried in the combined
            # bonus float) -- these are proxy metrics to WATCH, never to pay
            # the model for directly (2026-10-27: wrote_poc/wrote_nonempty
            # used to also be bonus components; they are now logged only,
            # precisely because they measure looking-complete, not solving).
            "n_actions": n_actions,
            "n_distinct_actions": len(set(self._sigs)),
            "wrote_poc": self._wrote_poc,
            "wrote_nonempty": self._wrote_nonempty,
        }
        return terminal + applied

    def _process_bonus(self) -> float:
        """Bounded within-tier shaping in [-_W_REPEAT_PEN, 0] (a pure penalty,
        see the 2026-10-27 note above _W_REPEAT_PEN -- behavioral-completeness
        components were removed, so there is nothing left to pay out for)."""
        n = len(self._sigs)
        if n == 0:
            return 0.0
        distinct = len(set(self._sigs))
        # repetition: fraction of calls that are duplicates of an earlier call
        repeats = n - distinct
        return -(_W_REPEAT_PEN * min(repeats, n) / n)

    def reward_is_valid(self) -> bool:
        """Whether the last get_reward() call reflects a real behavioral
        outcome rather than a grader/infra failure. See get_reward()'s
        docstring -- this is inert until a trainer-side change consumes it."""
        return getattr(self, "_last_verdict_tier", None) != TIER_INFRA_ERROR

    def _teardown(self) -> None:
        if self._work is not None:
            shutil.rmtree(self._work, ignore_errors=True)
        self._env = None
        self._work = None

    # ------------------------------------------------------------- tools
    # Each becomes a model-callable tool via its type hints + docstring
    # (transformers chat-template tool schema extraction). Mirrors
    # CyberGymEnv._TOOLS one-for-one so GRPO and the existing JSON-protocol
    # harness teach the same affordances.

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read a file from the vulnerable codebase.

        Args:
            path: File path, relative to the codebase root.
            start: Optional 1-based start line to read only a range.
            end: Optional 1-based end line (inclusive) to read only a range.

        Returns:
            The file contents, or an error string if the path does not exist.
        """
        return self._step("read_file", {"path": path, "start": start, "end": end})

    def grep(self, pattern: str) -> str:
        """Regex-search the codebase for a pattern.

        Args:
            pattern: A regular expression to search for.

        Returns:
            Matching lines as "path:line:text", one per line.
        """
        return self._step("grep", {"pattern": pattern})

    def list_dir(self, path: str = ".") -> str:
        """List the contents of a directory in the codebase.

        Args:
            path: Directory path, relative to the codebase root.

        Returns:
            Entry names, one per line (directories suffixed with "/").
        """
        return self._step("list_dir", {"path": path})

    def write_poc(self, content: str) -> str:
        """Write the current proof-of-concept input bytes.

        Args:
            content: The PoC bytes. Hex escapes like \\x01 are decoded.

        Returns:
            A confirmation of how many bytes were written.
        """
        return self._step("write_poc", {"content": content})

    def submit(self) -> str:
        """Submit the current PoC to the verifier. Ends the episode.

        Returns:
            A confirmation that submission was received.
        """
        return self._step("submit", {})

    def _step(self, action: str, action_input: dict) -> str:
        assert self._env is not None, "tool called before reset()"
        # TRL's environment_factory protocol (as read 2026-10-03, see module
        # docstring) gives tool methods no way to return "episode over" --
        # unlike ToolResult.done in the JSON-action harness, there is no done
        # flag the trainer checks per call. Without this guard the model could
        # keep calling read_file/grep/write_poc after submit() inside the same
        # rollout, acting on an environment whose verdict get_reward() is
        # about to read -- wasted budget at best, a confusing reward signal at
        # worst. (2026-10-08, review P1.) This only stops OUR tools from
        # acting further; it does not end the TRL rollout turn itself, which
        # still needs a trainer-level fix if the model keeps calling tools.
        if self._done:
            return "episode already submitted; no further actions will be taken"
        # progress tracking for the shaped process reward (signature = action +
        # its normalized input, so repeating the same call is detectable)
        self._sigs.append(action + "|" + json.dumps(action_input, sort_keys=True, ensure_ascii=False))
        if action == "write_poc":
            self._wrote_poc = True
            if (action_input.get("content") or "").strip():
                self._wrote_nonempty = True
        result = self._env.step(action, action_input)
        if action == "submit":
            self._done = True
        return result.observation
