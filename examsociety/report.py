from __future__ import annotations

from collections import Counter

import numpy as np

from .psychometrics import analyse
from .schemas import ItemRecord


def validation_agreement(records: list[ItemRecord]) -> dict:
    """How often a second, independent validator overturns the first.

    With `loop.validator_consensus > 1` the gate draws several validators from
    distinct families and stops at the first dissent. Every dissent is therefore
    an item that single-validator gating WOULD have accepted, and the dissent
    rate is a lower bound on the answer-key error rate — a number the
    single-validator design cannot produce at all, because a wrong key that the
    lone validator shares is invisible to it.

    It is only a lower bound: errors both validators make together stay hidden,
    which is what the expert audit in EVALUATION.md is ultimately for.
    """
    seconded = overturned = cross_family = cross_overturned = 0
    for r in records:
        rounds: dict[int, list] = {}
        for v in r.validations:
            rounds.setdefault(v.version if v.version is not None else -1, []).append(v)
        for results in rounds.values():
            if len(results) < 2 or results[0].verdict != "pass":
                continue  # no second opinion was sought, or the first already failed
            for later in results[1:]:
                seconded += 1
                dissent = later.verdict != "pass"
                overturned += dissent
                if later.cross_family:
                    cross_family += 1
                    cross_overturned += dissent
    return {
        "n_second_opinions": seconded,
        "n_overturned": overturned,
        "key_error_lower_bound": (overturned / seconded) if seconded else None,
        "n_cross_family": cross_family,
        "cross_family_overturn_rate": (
            (cross_overturned / cross_family) if cross_family else None
        ),
    }


def run_summary(records: list[ItemRecord]) -> dict:
    from .psychometrics import response_consistency

    statuses = Counter(r.final_status for r in records)
    finals = [r for r in records if r.final_item() is not None]
    misinfo_n = sum(1 for r in finals if r.final_item().misinfo is not None)
    consistency = response_consistency(records)
    revised_n = sum(1 for r in finals if (r.final_version or 0) > 0)
    solve_rates = [
        p.solve_rate
        for r in finals
        for p in r.panel_runs
        if p.version == r.final_version
    ]
    stats, theta, solver_ids = analyse(records)
    flagged = [s for s in stats if s.flag]
    agreement = validation_agreement(records)

    def _nanmean(vals):
        arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
        return float(arr.mean()) if arr.size else None

    return {
        "n_records": len(records),
        "statuses": dict(statuses),
        "n_final_items": len(finals),
        "n_with_misinfo": misinfo_n,
        "n_revised": revised_n,
        "n_scored_items": len(stats),
        "irt_reliable": len(stats) >= 20 and len(solver_ids) >= 10,
        "mean_final_solve_rate": float(np.mean(solve_rates)) if solve_rates else None,
        "mean_point_biserial": _nanmean([s.point_biserial for s in stats]),
        "mean_irt_a": _nanmean([s.irt_a for s in stats]),
        "mean_irt_b": _nanmean([s.irt_b for s in stats]),
        "n_flagged_items": len(flagged),
        "solver_theta": dict(zip(solver_ids, [float(t) for t in theta])) if len(theta) else {},
        "validation": agreement,
        "consistency": consistency,
    }


def format_summary(name: str, s: dict) -> str:
    lines = [f"== {name} =="]
    lines.append(f"records: {s['n_records']}  statuses: {s['statuses']}")
    c = s.get("consistency") or {}
    if c.get("consistency") is not None:
        lines.append(
            f"replicate consistency: {c['consistency']:.0%} over "
            f"{c['n_replicated_cells']} repeated cells "
            f"(low = the bank is unstable, not hard)"
        )
    lines.append(
        f"final items: {s['n_final_items']}  revised: {s['n_revised']}  "
        f"with misinfo: {s['n_with_misinfo']}"
    )
    if s["mean_final_solve_rate"] is not None:
        lines.append(f"mean final solve rate (difficulty, lower=harder): {s['mean_final_solve_rate']:.3f}")
    if s["mean_point_biserial"] is not None:
        b_str = f"{s['mean_irt_b']:.3f}" if s["mean_irt_b"] is not None else "n/a"
        lines.append(
            f"discrimination: point-biserial={s['mean_point_biserial']:.3f}  "
            f"IRT a={s['mean_irt_a']:.3f}  IRT b={b_str}  "
            f"flagged={s['n_flagged_items']}"
        )
    if not s["irt_reliable"] and s["n_scored_items"]:
        lines.append(
            f"  (!) IRT exploratory only: {s['n_scored_items']} items / "
            f"{len(s['solver_theta'])} solvers — need >=20 items & >=10 solvers "
            f"for stable estimates"
        )
    val = s.get("validation") or {}
    if val.get("n_second_opinions"):
        lines.append(
            f"validator consensus: {val['n_overturned']}/{val['n_second_opinions']} "
            f"second opinions overturned the first "
            f"-> key error rate >= {val['key_error_lower_bound']:.1%}"
        )
        if val.get("cross_family_overturn_rate") is not None:
            lines.append(
                f"  cross-family second opinions: {val['n_cross_family']}, "
                f"overturn rate {val['cross_family_overturn_rate']:.1%}"
            )
    if s["solver_theta"]:
        theta_str = "  ".join(f"{k}={v:+.2f}" for k, v in s["solver_theta"].items())
        lines.append(f"solver ability (theta): {theta_str}")
    return "\n".join(lines)


def compare(named_runs: list[tuple[str, list[ItemRecord]]]) -> str:
    blocks = [format_summary(name, run_summary(records)) for name, records in named_runs]
    return "\n\n".join(blocks)
