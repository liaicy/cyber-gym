"""Data models for trajectories and preference data.

A Trajectory is one agent's full attempt at one task: the interleaved
think/act/observe steps, plus the environment's verdict and the resources
spent. Preference data is built from sets of trajectories on the SAME task.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class Step(BaseModel):
    """One turn of the agent loop."""

    index: int
    thought: str = ""                       # the model's reasoning for this turn
    action: str = ""                        # tool name, e.g. "grep" / "write_file" / "run"
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""                   # what the environment returned
    # kept for training: the raw text the model emitted this turn, so SFT can
    # reproduce the exact think+act format the environment expects.
    raw_output: str = ""
    # Per-turn cost, when the backend exposes it. Populated turn-by-turn (not
    # just as a trajectory total) specifically so the quadratic-transcript-cost
    # question (PLAN discussion, 2026-08-27) can be read directly off stored
    # trajectories -- prompt_tokens rising step over step IS the growth curve.
    prompt_tokens: int = 0
    completion_tokens: int = 0


# Ordinal outcome tiers, ranked strictly: a higher tier always outranks every
# trajectory in a lower one, regardless of any within-tier heuristic. See
# preference.rank_score. "solved" and "no_crash" are unambiguous from the
# environment's (vul_exit_code, fix_exit_code) pair; "crashed_unclear_fix" is
# deliberately NOT called "crashed" -- fix_exit_code values other than 0/None
# (77 observed on arvo:10400) have not been confirmed to mean "the patched
# build crashed differently" rather than "the fix-verification harness did not
# really run" (POSIX/automake test drivers use 77 for "skipped", not a crash).
# Treating it as confirmed partial credit before that is checked would promote
# harness noise into the chosen side of a preference pair -- keep it its own
# tier until enough small-batch samples let it be split for real.
TIER_SOLVED = "solved"
TIER_CRASHED_UNCLEAR_FIX = "crashed_unclear_fix"
TIER_NO_CRASH = "no_crash"
TIER_DEGENERATE = "degenerate"          # agent never wrote a PoC -- a real behavioral failure
# Harness/infra could not render a verdict at all (bash missing, submit.sh
# missing because gen_task's setup never ran, the submit subprocess timed out
# or errored, or the server returned something unparseable). Deliberately NOT
# in TIER_ORDER: none of these say anything about what the MODEL did, so
# ranking or training on them would teach "the grader broke" as if it were
# "the model failed." (2026-10-08, external review P1: verify() was
# collapsing this together with TIER_DEGENERATE.) is_scorable() excludes it.
TIER_INFRA_ERROR = "infra_error"
TIER_ORDER = (TIER_SOLVED, TIER_CRASHED_UNCLEAR_FIX, TIER_NO_CRASH, TIER_DEGENERATE)


class Trajectory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str
    model_id: str                           # which solver produced this
    model_family: str = ""
    steps: list[Step] = Field(default_factory=list)
    # Outcome is the WHOLE point: it comes from the environment's verifier, not
    # from any model's judgement. That is what makes this signal trustworthy in
    # a way exam-society's model-graded signal never was.
    success: bool = False
    tier: str = ""                          # one of TIER_ORDER; "" = ungraded / legacy data
    vul_exit_code: Optional[int] = None
    fix_exit_code: Optional[int] = None
    verifier_detail: str = ""               # why the verifier passed/failed
    n_steps: int = 0
    tokens: int = 0                          # total completion tokens spent
    wall_seconds: float = 0.0
    stop_reason: str = ""                    # solved | step_budget | token_budget | error
    created_at: float = Field(default_factory=time.time)
    collection_metadata: dict[str, Any] = Field(default_factory=dict)

    def is_scorable(self) -> bool:
        """A trajectory that errored out before doing anything is not a
        datapoint -- neither a good example nor a bad one, just missing.

        Checks BOTH stop_reason and tier, not stop_reason alone: agent.py's
        rollout() maps TIER_INFRA_ERROR to stop_reason="error" so this method
        catches it through that path, but anything that builds a Trajectory
        without going through agent.py (a different harness, a hand-written
        record, a future call site) could set tier="infra_error" without
        also setting stop_reason -- this is the independent check that still
        excludes it. (2026-10-08, round-2 review P1: demonstrated exactly
        this gap with tier="infra_error", stop_reason="step_budget".)
        """
        return self.stop_reason in ("solved", "step_budget", "token_budget") and self.tier != TIER_INFRA_ERROR


class RankedTrajectory(BaseModel):
    """A trajectory plus the scalar used to order it within one task's set."""

    trajectory_id: str
    model_id: str
    success: bool
    tier: str = ""                          # one of TIER_ORDER; the environment-
                                            # graded axis chosen_rejected() ranks by
    rank_score: float                       # higher = better; see preference.py
    n_steps: int
    tokens: int


class PreferencePair(BaseModel):
    """One List-wise DPO record for one task.

    `ranked` is the full ordering (best first); a trainer can take the whole
    list (list-wise) or the top-vs-bottom pair. Storing the ordering rather
    than a single pair keeps the door open to both without re-deriving.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str
    prompt: str                             # the task input the agent saw (NO answer)
    ranked: list[RankedTrajectory] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)  # source benchmark, task meta

    def chosen_rejected(self) -> Optional[tuple[str, str]]:
        """Best vs worst trajectory id, when there is a real contrast.

        Tier separation is the only axis anchored on the environment's verdict,
        so it dominates: the pair spans the widest tier gap available, and
        model-identity control is spent only as a tie-break that costs no tier
        separation -- never the other way round. (An earlier "same-model first"
        rule got this backwards: it could return an intra-tier, engagement-only
        same-model pair in preference to a cross-tier one, quietly demoting the
        oracle signal to a heuristic proxy -- the exact thing tiering exists to
        prevent.)

        `ranked` is already sorted best-first with tier dominating rank_score
        (preference.rank_score caps the within-tier bonus below the tier gap),
        so ranked[0].tier and ranked[-1].tier ARE the widest tier span on the
        task; no other pair can cross more tiers.

          - Head tier != tail tier (a cross-tier contrast exists): if some model
            appears in BOTH the top-tier and bottom-tier groups, use its
            best-in-top vs worst-in-bottom -- cross-tier AND same-model, the
            ideal. Otherwise fall back to ranked[0] vs ranked[-1]: same widest
            tier gap, across two models. Identity control is dropped before
            oracle signal, never the reverse.
          - Head tier == tail tier (no cross-tier signal anywhere): only here do
            we fall to a within-tier contrast, and there identity control IS
            worth taking -- prefer a same-model pair, else global best-vs-worst.

        Why identity control at all: cross-model pairs confound outcome with
        model identity -- DPO can credit the win to surface token style
        (vocabulary, verbosity, tool-call phrasing) instead of the strategic
        choices that mattered. Same-model, same-task samples differ only in the
        roll of the dice, the way exam-society's clean/injected pairs differed
        only in the injected claim. But that control is a luxury on top of the
        oracle signal, not a substitute for it.
        See docs/PLAN-trajectory.md discussion, 2026-08-27.
        """
        if len(self.ranked) < 2:
            return None
        top_tier, bot_tier = self.ranked[0].tier, self.ranked[-1].tier

        if top_tier != bot_tier:
            # Widest tier gap is fixed at (top_tier, bot_tier). Prefer a single
            # model spanning it; otherwise take it across models. A top-tier
            # entry always outscores a bottom-tier one (tier dominates score),
            # so no rank_score guard is needed on the same-model branch.
            top_group = [r for r in self.ranked if r.tier == top_tier]
            bot_ids = {r.model_id for r in self.ranked if r.tier == bot_tier}
            for r in top_group:                    # sorted best-first: first hit = best top anchor
                if r.model_id in bot_ids:
                    worst = next(b for b in reversed(self.ranked)
                                 if b.tier == bot_tier and b.model_id == r.model_id)
                    return r.trajectory_id, worst.trajectory_id
            best, worst = self.ranked[0], self.ranked[-1]
            if best.rank_score <= worst.rank_score:
                return None                        # guard against a degenerate all-equal list
            return best.trajectory_id, worst.trajectory_id

        # All one tier: no environment-graded contrast. Control identity if we can.
        by_model: dict[str, list[RankedTrajectory]] = {}
        for r in self.ranked:
            by_model.setdefault(r.model_id, []).append(r)
        for group in by_model.values():
            if len(group) < 2:
                continue
            best, worst = group[0], group[-1]      # self.ranked is already sorted best-first
            if best.rank_score > worst.rank_score:
                return best.trajectory_id, worst.trajectory_id
        best, worst = self.ranked[0], self.ranked[-1]
        if best.rank_score <= worst.rank_score:
            return None
        return best.trajectory_id, worst.trajectory_id
