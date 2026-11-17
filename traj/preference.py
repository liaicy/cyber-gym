"""Turn per-task trajectory sets into List-wise DPO data.

Meeting decision: strong-model correct answers are `chosen`, weak-model answers
are `reject`. Generalised here to a ranking over ALL trajectories on a task, so
a trainer can use the full list (list-wise) or just top-vs-bottom.

The ranking key is deliberately simple and, critically, its top end is anchored
on the ENVIRONMENT'S verdict, not on any model's opinion -- a trajectory only
outranks another for "success" if the verifier actually passed it.
"""
from __future__ import annotations

from .schemas import PreferencePair, RankedTrajectory, TIER_ORDER, Trajectory

# Tier is the dominant axis -- a fixed gap between tiers that no within-tier
# heuristic bonus below can ever close. Ties directly to schemas.TIER_ORDER;
# missing/legacy tier ("") falls back to the bottom band alongside degenerate.
_TIER_BASE = {tier: 1000.0 * (len(TIER_ORDER) - i) for i, tier in enumerate(TIER_ORDER)}


def rank_score(t: Trajectory) -> float:
    """Higher is better. Tier dominates everything else; within a tier,
    cheaper/successful beats engaged-but-failed beats degenerating into noise.

    Two real-data lessons are encoded here, both found by actually running the
    pipeline rather than reasoning about it offline:

    1. (2026-08-26) The original version scored FAILURES by raw n_steps
       ("longer = more engaged"). On real CyberGym rollouts (arvo:10400) that
       ranking was backwards: a model that explored, wrote two PoC attempts,
       and submitted (7 steps, all valid) scored BELOW a model that explored
       briefly then degenerated into 12 straight unparseable turns (18 steps).
       Raw step count rewards stalling, not effort. Fixed by counting only
       steps with a real parsed action, plus bonuses for reaching write_poc /
       submit.

    2. (2026-08-27) Even fixed, that heuristic was the ENTIRE ranking for any
       trajectory that didn't hit the environment's binary success flag --
       meaning on the vast majority of tasks (CyberGym's frontier success rate
       is <10%), "chosen" was picked by a heuristic proxy, not by the world.
       Worse, a raw environment signal was being thrown away: a vul-crashing
       PoC with an ambiguous fix_exit_code (e.g. 77 on arvo:10400) was scored
       identically to a PoC that never crashed anything. CyberGym's own
       (vul_exit_code, fix_exit_code) pair is already an ordinal, environment-
       graded signal (see schemas.TIER_ORDER / env.Verdict) -- using it here
       shrinks how much of the ranking rests on heuristics instead of the
       verifier, without adding any new machinery.
    """
    tier = t.tier or (TIER_ORDER[0] if t.success else TIER_ORDER[-1])  # legacy data: no tier recorded
    base = _TIER_BASE.get(tier, _TIER_BASE[TIER_ORDER[-1]])
    if tier == TIER_ORDER[0]:                        # solved: cheaper wins
        return base - min(t.tokens, 900) / 1000.0 - t.n_steps * 0.01
    # Every other tier: engagement is a tie-break WITHIN the tier only. Hard-
    # capped below the 1000-point tier gap so it can NEVER let a lower tier
    # outscore a higher one -- chosen_rejected() relies on this being structural
    # (a 990-step rollout must not float a no_crash above a crashed_unclear_fix)
    # rather than merely true for the current step budget.
    valid = [s for s in t.steps if s.action]        # non-empty action = a real
                                                     # tool call, not a parse failure
    reached_write = any(s.action == "write_poc" for s in t.steps)
    reached_submit = any(s.action == "submit" for s in t.steps)
    bonus = len(valid) * 1.0
    if reached_write:
        bonus += 5.0
    if reached_submit:
        bonus += 5.0
    return base + min(bonus, 900.0)


def build_preference(
    task_id: str,
    prompt: str,
    trajectories: list[Trajectory],
    provenance: dict | None = None,
) -> PreferencePair | None:
    """One List-wise DPO record for one task, or None if there is no contrast.

    Requires at least one success OR a real spread of failures; a set where
    every trajectory is an identical instant-failure carries no preference
    signal and is dropped rather than fed as noise.
    """
    scorable = [t for t in trajectories if t.is_scorable()]
    if len(scorable) < 2:
        return None
    ranked = sorted(
        (RankedTrajectory(
            trajectory_id=t.id, model_id=t.model_id, success=t.success,
            # Mirror rank_score's legacy fallback exactly, so the tier stored on
            # the RankedTrajectory agrees with the score it was ranked by.
            tier=t.tier or (TIER_ORDER[0] if t.success else TIER_ORDER[-1]),
            rank_score=rank_score(t), n_steps=t.n_steps, tokens=t.tokens,
        ) for t in scorable),
        key=lambda r: r.rank_score,
        reverse=True,
    )
    if ranked[0].rank_score <= ranked[-1].rank_score:
        return None                                   # flat: no signal
    # Recorded, not enforced. At CyberGym's <10% solve rate most tasks top out
    # below `solved`, so "did this pair cross a tier boundary?" and "what is its
    # ceiling?" are exactly the two ratios the small batch needs to decide --
    # empirically, not by guess -- whether to down-weight or drop the
    # never-crossed-a-tier and capped-at-crashed_unclear_fix pairs at train
    # time. Tagging keeps that call with the trainer instead of silently
    # dropping data here where yield is already scarce.
    prov = dict(provenance or {})
    prov["crosses_tier"] = ranked[0].tier != ranked[-1].tier
    prov["top_tier"] = ranked[0].tier
    return PreferencePair(task_id=task_id, prompt=prompt, ranked=ranked, provenance=prov)
