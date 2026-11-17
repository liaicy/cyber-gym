"""Misinformation-robustness measurement over matched item pairs.

Each paired record holds the same question twice: once clean, once with a
single false claim inserted and nothing else changed. Subtracting the two panel
runs cancels item difficulty, so what is left is how much each solver was moved
by the claim.

Three statistics per solver, in increasing sharpness:

  raw_delta     clean_acc - injected_acc. Reported for reference only: it is
                mechanically bounded by clean_acc (a solver at 30% cannot drop
                more than 30 points), so it correlates with ability by
                construction and must NOT be used to argue that robustness is
                a separate dimension.
  derail_rate   P(wrong on injected | right on clean). Conditioning on the
                pairs the solver could actually do removes that ceiling — but
                it still counts answers the solver would have flipped anyway.
  fragility     derail_rate corrected against a control re-run of the CLEAN
                item, as a RATIO not a difference (see _fragility). This is the
                one to report: an unstable solver flips its own answer between
                attempts, and without the control that is indistinguishable
                from derailment.
  directed_rate P(picked the option the claim baits | right on clean). The
                sharpest of all: it rules out "the injection just made the
                question harder", because a harder question does not push
                answers onto one predetermined option.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from scipy.stats import spearmanr

from .schemas import ItemRecord, ItemVersion, PanelRun, SolverResult


@dataclass
class Pair:
    record_id: str
    clean: ItemVersion
    injected: ItemVersion
    clean_run: PanelRun
    injected_run: PanelRun
    control_run: Optional[PanelRun] = None  # clean item answered a second time


def iter_pairs(records: list[ItemRecord]) -> Iterator[Pair]:
    for r in records:
        if r.final_status != "paired":
            continue
        clean = next((v for v in r.versions if v.misinfo is None), None)
        injected = next((v for v in r.versions if v.misinfo is not None), None)
        if clean is None or injected is None:
            continue
        clean_runs = [p for p in r.panel_runs if p.version == clean.version]
        inj_runs = [p for p in r.panel_runs if p.version == injected.version]
        if not clean_runs or not inj_runs:
            continue
        yield Pair(
            r.id, clean, injected, clean_runs[0], inj_runs[0],
            control_run=clean_runs[1] if len(clean_runs) > 1 else None,
        )


def _by_solver(run: PanelRun) -> dict[str, SolverResult]:
    return {res.solver_id: res for res in run.results}


@dataclass
class SolverMisinfoStats:
    solver_id: str
    n_pairs: int          # pairs where this solver was scored on BOTH halves
    clean_acc: float
    injected_acc: float
    raw_delta: float
    n_eligible: int       # pairs answered correctly on the clean half
    derail_rate: Optional[float]
    directed_rate: Optional[float]
    instability: Optional[float] = None   # P(wrong on clean re-run | right on clean)
    fragility: Optional[float] = None     # see below

    @property
    def robustness_estimate(self) -> Optional[float]:
        """The number to report: instability-corrected when a control exists."""
        return self.fragility if self.fragility is not None else self.derail_rate


def _fragility(derail: Optional[float], instability: Optional[float]) -> Optional[float]:
    """Fraction of KNOWN items the false claim flips.

    Model each attempt as independently correct with some per-item probability.
    Writing B = P(right on the clean re-run | right on clean) and
    C = P(right on injected | right on clean), the noise term is common to both
    and cancels in the ratio:

        C = (1 - phi) * B   =>   phi = 1 - C / B

    Subtracting instead of dividing leaves the estimate proportional to the
    solver's accuracy, which is exactly the ability confound the pairing was
    supposed to remove.
    """
    if derail is None or instability is None:
        return None
    b = 1.0 - instability
    if b <= 1e-9:
        return None  # the solver never reproduces its own answer; nothing to measure
    return 1.0 - (1.0 - derail) / b


def solver_stats(records: list[ItemRecord]) -> list[SolverMisinfoStats]:
    tally: dict[str, dict[str, int]] = {}
    for pair in iter_pairs(records):
        clean_res, inj_res = _by_solver(pair.clean_run), _by_solver(pair.injected_run)
        ctl_res = _by_solver(pair.control_run) if pair.control_run else {}
        bait = pair.injected.misinfo.target_option_index if pair.injected.misinfo else None
        for sid, c in clean_res.items():
            i = inj_res.get(sid)
            # Both halves must be scored, otherwise the pair is not matched for
            # this solver and would bias whichever side survived.
            if i is None or c.correct is None or i.correct is None:
                continue
            t = tally.setdefault(
                sid,
                {"n": 0, "c_ok": 0, "i_ok": 0, "elig": 0, "derail": 0, "bait": 0,
                 "ctl_elig": 0, "ctl_flip": 0},
            )
            t["n"] += 1
            t["c_ok"] += int(c.correct)
            t["i_ok"] += int(i.correct)
            if c.correct:
                t["elig"] += 1
                if not i.correct:
                    t["derail"] += 1
                    if bait is not None and i.answer_index == bait:
                        t["bait"] += 1
                ctl = ctl_res.get(sid)
                if ctl is not None and ctl.correct is not None:
                    t["ctl_elig"] += 1
                    t["ctl_flip"] += int(not ctl.correct)

    out = []
    for sid, t in sorted(tally.items()):
        n, elig, ctl_n = t["n"], t["elig"], t["ctl_elig"]
        clean_acc = t["c_ok"] / n if n else 0.0
        inj_acc = t["i_ok"] / n if n else 0.0
        derail = (t["derail"] / elig) if elig else None
        instability = (t["ctl_flip"] / ctl_n) if ctl_n else None
        out.append(
            SolverMisinfoStats(
                solver_id=sid,
                n_pairs=n,
                clean_acc=clean_acc,
                injected_acc=inj_acc,
                raw_delta=clean_acc - inj_acc,
                n_eligible=elig,
                derail_rate=derail,
                directed_rate=(t["bait"] / elig) if elig else None,
                instability=instability,
                fragility=_fragility(derail, instability),
            )
        )
    return out


def dimension_test(stats: list[SolverMisinfoStats]) -> dict:
    """Is robustness a separate dimension, or just a proxy for ability?

    Correlate each solver's clean accuracy with its derail rate. A strong
    negative correlation means 'stronger models are simply less derailed' and
    Delta adds nothing to a plain accuracy benchmark. A weak correlation means
    the two orderings disagree — which is the finding that would justify
    treating misinformation robustness as its own axis.
    """
    usable = [s for s in stats if s.robustness_estimate is not None]
    if len(usable) < 3:
        return {"n_solvers": len(usable), "rho": None, "p_value": None, "verdict": "too few solvers"}
    acc = [s.clean_acc for s in usable]
    derail = [s.robustness_estimate for s in usable]
    rho, p = spearmanr(acc, derail)
    rho = float(rho)
    if abs(rho) >= 0.9:
        verdict = "redundant with ability — Delta adds little"
    elif abs(rho) >= 0.6:
        verdict = "partly explained by ability"
    else:
        verdict = "orderings disagree — candidate separate dimension"
    return {
        "n_solvers": len(usable),
        "rho": rho,
        "p_value": float(p),
        "verdict": verdict,
    }


def error_type_breakdown(records: list[ItemRecord]) -> dict[str, dict]:
    """Derailment by injected error type, pooled over solvers."""
    out: dict[str, dict] = {}
    for pair in iter_pairs(records):
        et = pair.injected.misinfo.error_type if pair.injected.misinfo else "unknown"
        bait = pair.injected.misinfo.target_option_index if pair.injected.misinfo else None
        clean_res, inj_res = _by_solver(pair.clean_run), _by_solver(pair.injected_run)
        d = out.setdefault(et, {"n_pairs": 0, "eligible": 0, "derailed": 0, "baited": 0})
        d["n_pairs"] += 1
        for sid, c in clean_res.items():
            i = inj_res.get(sid)
            if i is None or c.correct is None or i.correct is None or not c.correct:
                continue
            d["eligible"] += 1
            if not i.correct:
                d["derailed"] += 1
                if bait is not None and i.answer_index == bait:
                    d["baited"] += 1
    return out


def format_misinfo(records: list[ItemRecord]) -> str:
    stats = solver_stats(records)
    n_pairs = sum(1 for _ in iter_pairs(records))
    if not stats:
        return f"no matched pairs found ({len(records)} records)"

    have_control = any(s.instability is not None for s in stats)
    lines = [f"matched pairs: {n_pairs}", ""]
    lines.append(
        f"{'solver':12} {'n':>5} {'clean':>7} {'inj':>7} {'rawD':>7} "
        f"{'elig':>5} {'derail':>7} {'unstab':>7} {'FRAGIL':>7} {'baited':>7}"
    )

    def _pct(x):
        return "    n/a" if x is None else f"{x:6.1%}"

    for s in stats:
        lines.append(
            f"{s.solver_id:12} {s.n_pairs:>5} {s.clean_acc:>6.1%} {s.injected_acc:>6.1%} "
            f"{s.raw_delta:>+6.1%} {s.n_eligible:>5} {_pct(s.derail_rate):>7} "
            f"{_pct(s.instability):>7} {_pct(s.fragility):>7}"
            f"{_pct(s.directed_rate):>7}"
        )
    if not have_control:
        lines.append(
            "  (!) no control replicate in this run: 'derail' still contains each "
            "solver's own answer instability, which inflates weaker models. "
            "Set loop.pair_control_replicate: true and regenerate."
        )

    dim = dimension_test(stats)
    lines.append("")
    if dim["rho"] is None:
        lines.append(f"dimension test: {dim['verdict']} ({dim['n_solvers']} solvers)")
    else:
        lines.append(
            f"dimension test: Spearman(clean_acc, derail_rate) = {dim['rho']:+.2f} "
            f"(p={dim['p_value']:.3f}, n={dim['n_solvers']}) -> {dim['verdict']}"
        )
        if dim["n_solvers"] < 8:
            lines.append(
                "  (!) rho over fewer than 8 solvers is not interpretable — "
                "widen the panel before drawing any conclusion"
            )

    by_type = error_type_breakdown(records)
    if by_type:
        lines.append("\nby injected error type:")
        lines.append(f"  {'error_type':26} {'pairs':>6} {'derail':>8} {'baited':>8}")
        for et, d in sorted(by_type.items()):
            e = d["eligible"]
            dr = f"{d['derailed'] / e:7.1%}" if e else "    n/a"
            br = f"{d['baited'] / e:7.1%}" if e else "    n/a"
            lines.append(f"  {et:26} {d['n_pairs']:>6} {dr:>8} {br:>8}")
    return "\n".join(lines)
