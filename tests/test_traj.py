# -*- coding: utf-8 -*-
"""Offline tests for the trajectory -> preference pipeline (no API calls).

Run: python tests/test_traj.py

The load-bearing assertion here is that the environment-graded TIER is
inviolable in BOTH places it matters, which are two different claims:
  1. in rank_score  -- a lower tier can never outscore a higher one, whatever
     its engagement bonus (test_tier_dominates_score);
  2. in pair SELECTION -- chosen_rejected() never trades a cross-tier contrast
     for a within-tier one to buy model-identity control
     (test_chosen_rejected_never_trades_tier_gap_for_model_identity).
An earlier suite only checked (1); the fixture below pins (2) so a future edit
to chosen_rejected() can't silently reintroduce the regression.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traj.preference import build_preference, rank_score
from traj.schemas import (
    Step,
    Trajectory,
    TIER_CRASHED_UNCLEAR_FIX,
    TIER_DEGENERATE,
    TIER_NO_CRASH,
    TIER_ORDER,
    TIER_SOLVED,
)


def _traj(model_id, tier, *, actions=(), success=False, tokens=0, n_steps=None):
    """A minimal scorable trajectory. `actions` become real (parsed) steps, so
    the engagement bonus = len(actions) plus write/submit credit."""
    steps = [Step(index=i, action=a) for i, a in enumerate(actions)]
    return Trajectory(
        task_id="t", model_id=model_id, tier=tier, success=success, steps=steps,
        tokens=tokens, n_steps=n_steps if n_steps is not None else len(steps),
        stop_reason="solved" if success else "step_budget",
    )


def _by_id(trajs):
    return {t.id: t for t in trajs}


def test_tier_dominates_score():
    """No engagement bonus can float a lower tier above a higher one, even with
    the bonus maxed out against a bare solve."""
    solved_bare = _traj("A", TIER_SOLVED, success=True, actions=["submit"],
                        tokens=900, n_steps=40)
    busy_fail = _traj("B", TIER_NO_CRASH,
                     actions=["grep"] * 38 + ["write_poc", "submit"])
    assert rank_score(solved_bare) > rank_score(busy_fail)

    # Full ordering across all four tiers, each with identical engagement.
    scored = [
        _traj("m", tr, success=(tr == TIER_SOLVED), actions=["grep", "read_file"])
        for tr in TIER_ORDER
    ]
    scores = [rank_score(t) for t in scored]
    assert scores == sorted(scores, reverse=True), scores


def test_chosen_rejected_never_trades_tier_gap_for_model_identity():
    """The reproducer (2026-08-27). Model A has two crashed_unclear_fix samples
    differing only by engagement; model B has a no_crash sample. The old
    same-model-first rule returned the intra-tier A/A pair; the fix MUST return
    the cross-tier A(crashed) vs B(no_crash) pair -- oracle signal over identity
    control."""
    a_hi = _traj("A", TIER_CRASHED_UNCLEAR_FIX,
                actions=["grep", "read_file", "write_poc", "submit"])
    a_lo = _traj("A", TIER_CRASHED_UNCLEAR_FIX, actions=["grep"])
    b = _traj("B", TIER_NO_CRASH, actions=["grep", "read_file"])

    pref = build_preference("t", "<p>", [a_hi, a_lo, b])
    assert pref is not None
    cr = pref.chosen_rejected()
    assert cr is not None
    chosen, reject = cr
    by = _by_id([a_hi, a_lo, b])
    assert by[chosen].tier == TIER_CRASHED_UNCLEAR_FIX
    assert by[reject].tier == TIER_NO_CRASH          # crossed the tier boundary
    assert by[reject].model_id == "B"                # not the intra-tier A/A pair
    assert pref.provenance["crosses_tier"] is True
    assert pref.provenance["top_tier"] == TIER_CRASHED_UNCLEAR_FIX


def test_same_model_span_taken_when_it_costs_no_tier_gap():
    """When the widest gap (solved..no_crash) is itself achievable within one
    model, identity control is free -- take the same-model pair."""
    a_solved = _traj("A", TIER_SOLVED, success=True, actions=["submit"])
    a_nocrash = _traj("A", TIER_NO_CRASH, actions=["grep"])
    b_nocrash = _traj("B", TIER_NO_CRASH, actions=["grep", "read_file"])

    pref = build_preference("t", "<p>", [a_solved, a_nocrash, b_nocrash])
    chosen, reject = pref.chosen_rejected()
    by = _by_id([a_solved, a_nocrash, b_nocrash])
    assert by[chosen].model_id == "A" and by[chosen].tier == TIER_SOLVED
    assert by[reject].model_id == "A" and by[reject].tier == TIER_NO_CRASH


def test_cross_model_fallback_keeps_tier_gap():
    """No single model spans the tier gap -> keep the gap, drop identity."""
    a = _traj("A", TIER_SOLVED, success=True, actions=["submit"])
    b = _traj("B", TIER_NO_CRASH, actions=["grep"])

    pref = build_preference("t", "<p>", [a, b])
    chosen, reject = pref.chosen_rejected()
    by = _by_id([a, b])
    assert by[chosen].tier == TIER_SOLVED and by[chosen].model_id == "A"
    assert by[reject].tier == TIER_NO_CRASH and by[reject].model_id == "B"


def test_all_one_tier_controls_identity():
    """No cross-tier signal anywhere: only now does identity control kick in,
    and the pair is tagged as not crossing a tier."""
    a_hi = _traj("A", TIER_NO_CRASH, actions=["grep", "read_file", "write_poc"])
    a_lo = _traj("A", TIER_NO_CRASH, actions=["grep"])
    b = _traj("B", TIER_NO_CRASH, actions=["grep", "read_file"])

    pref = build_preference("t", "<p>", [a_hi, a_lo, b])
    chosen, reject = pref.chosen_rejected()
    by = _by_id([a_hi, a_lo, b])
    assert by[chosen].model_id == "A" and by[reject].model_id == "A"
    assert pref.provenance["crosses_tier"] is False
    assert pref.provenance["top_tier"] == TIER_NO_CRASH


def test_legacy_untiered_falls_back_by_success():
    """tier='' (data predating tiering) is scored/ranked by the plain success
    bool: win -> solved band, loss -> degenerate band, ordering preserved."""
    win = _traj("A", "", success=True, actions=["submit"])
    lose = _traj("B", "", success=False, actions=["grep", "write_poc", "submit"])

    pref = build_preference("t", "<p>", [win, lose])
    chosen, _ = pref.chosen_rejected()
    assert _by_id([win, lose])[chosen].success is True
    assert pref.provenance["top_tier"] == TIER_SOLVED
    assert pref.provenance["crosses_tier"] is True


def test_no_contrast_returns_none():
    """A single trajectory, and a flat set (identical instant-fails), both carry
    no preference signal and must be dropped rather than fed as noise."""
    solo = _traj("A", TIER_NO_CRASH, actions=["grep"])
    assert build_preference("t", "<p>", [solo]) is None

    x = _traj("A", TIER_DEGENERATE)      # 0 actions -> bonus 0
    y = _traj("B", TIER_DEGENERATE)      # 0 actions -> bonus 0 -> equal scores
    assert build_preference("t", "<p>", [x, y]) is None


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run()
