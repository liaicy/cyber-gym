"""Is the measurement contaminated by who wrote the material?

Two leaks, one statistical shape:

  author advantage    Does a solver do better on items its own family wrote?
                      If so, a/theta measure "authored by me", not ability.
  injector interaction  Is a solver less derailed by false claims its own
                      family wrote? If so, FRAGIL is a property of the
                      solver-injector pair, not of the solver.

Both are differential item functioning: same outcome, two groups, and a
nuisance variable (how hard the thing was) that must be conditioned on rather
than averaged over. Mantel-Haenszel is the standard DIF estimator, so that is
what is used here — including the ETS delta scale, which supplies a principled
"how big is too big" threshold instead of an invented one.

The stratifying variable is always leave-one-out: how the OTHER solvers did on
the same item. Using the solver's own result would make the comparison circular.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from scipy.stats import chi2 as _chi2

from .config import guess_family
from .misinfo import iter_pairs
from .schemas import ItemRecord, PanelRun, SolverResult


def family_of(result: SolverResult) -> str:
    return result.family or guess_family(result.model, default=result.model)


# ------------------------------------------------------------ Mantel-Haenszel


@dataclass
class MHResult:
    """Common odds ratio for own-family vs other-family, difficulty held fixed."""

    label: str
    n_own: int
    n_other: int
    own_rate: Optional[float]
    other_rate: Optional[float]
    n_strata: int
    odds_ratio: Optional[float]
    delta: Optional[float]        # ETS scale: -2.35 * ln(OR)
    chi2: Optional[float]
    p_value: Optional[float]
    category: str                 # A negligible | B moderate | C large | insufficient

    @property
    def significant(self) -> bool:
        return self.p_value is not None and self.p_value < 0.05


def mantel_haenszel(tables: list[tuple[int, int, int, int]], label: str) -> MHResult:
    """tables: per stratum (own_hit, own_miss, other_hit, other_miss).

    'hit' is whatever the outcome of interest is — correct, for author
    advantage; derailed, for injector interaction.
    """
    num = den = 0.0
    obs = exp = var = 0.0
    used = 0
    n_own = n_other = own_hit = other_hit = 0
    for a, b, c, d in tables:
        n = a + b + c + d
        n_own += a + b
        n_other += c + d
        own_hit += a
        other_hit += c
        # a stratum with an empty row or column carries no information
        if n == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        used += 1
        num += a * d / n
        den += b * c / n
        obs += a
        exp += (a + b) * (a + c) / n
        var += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1)) if n > 1 else 0.0

    own_rate = own_hit / n_own if n_own else None
    other_rate = other_hit / n_other if n_other else None
    if used == 0 or den == 0 or num == 0 or var <= 0:
        return MHResult(
            label, n_own, n_other, own_rate, other_rate, used,
            None, None, None, None, "insufficient",
        )

    import math

    odds_ratio = num / den
    delta = -2.35 * math.log(odds_ratio)
    stat = (abs(obs - exp) - 0.5) ** 2 / var
    p = float(_chi2.sf(stat, 1))
    # ETS classification: negligible / moderate / large
    size = abs(delta)
    if not (p < 0.05) or size < 1.0:
        category = "A negligible"
    elif size < 1.5:
        category = "B moderate"
    else:
        category = "C large"
    return MHResult(
        label, n_own, n_other, own_rate, other_rate, used,
        float(odds_ratio), float(delta), float(stat), p, category,
    )


def _per_solver_and_pooled(
    tally: dict[str, dict[int, list[int]]]
) -> list[MHResult]:
    """One MH per solver, plus a pooled row stratified by (solver, difficulty).

    The pooled row is not a convenience: with authorship rotating over k
    families only ~1/k of the items are own-family, so a per-solver test is
    badly underpowered — a real effect can show a consistent odds ratio across
    every solver and still miss significance in each one individually. Pooling
    keeps the comparison strictly within-solver (solver is a stratum) while
    recovering the power.
    """
    results = [
        mantel_haenszel([tuple(c) for c in strata.values()], sid)
        for sid, strata in sorted(tally.items())
    ]
    pooled_tables = [
        tuple(cell) for strata in tally.values() for cell in strata.values()
    ]
    if pooled_tables:
        results.append(mantel_haenszel(pooled_tables, "POOLED"))
    return results


# ------------------------------------------------------------ author advantage


def _first_run(record: ItemRecord, version: int) -> Optional[PanelRun]:
    return next((p for p in record.panel_runs if p.version == version), None)


def author_advantage(records: list[ItemRecord]) -> list[MHResult]:
    """Per solver: accuracy on own-family-authored items vs others.

    Uses v0 and v0's first panel run — the item exactly as proposed, before any
    hardening or injection changed its authorship. Stratified by how many of the
    OTHER solvers got it right.
    """
    # solver -> stratum -> [own_hit, own_miss, other_hit, other_miss]
    tally: dict[str, dict[int, list[int]]] = {}
    for r in records:
        v0 = next((v for v in r.versions if v.version == 0), None)
        run = _first_run(r, 0)
        if v0 is None or run is None or not v0.author_family:
            continue
        scored = [x for x in run.results if x.correct is not None]
        if len(scored) < 2:
            continue  # no leave-one-out stratum can be formed
        total = sum(1 for x in scored if x.correct)
        for res in scored:
            stratum = total - int(bool(res.correct))
            cell = tally.setdefault(res.solver_id, {}).setdefault(stratum, [0, 0, 0, 0])
            own = family_of(res) == v0.author_family
            idx = (0 if own else 2) + (0 if res.correct else 1)
            cell[idx] += 1
    return _per_solver_and_pooled(tally)


# -------------------------------------------------------- injector interaction


def injector_interaction(records: list[ItemRecord]) -> list[MHResult]:
    """Per solver: derailment by own-family-written false claims vs others.

    Restricted to pairs the solver answered correctly on the clean half — only
    those can be derailed. Stratified by how many of the OTHER eligible solvers
    were derailed on the same pair, i.e. by how strong the claim was.
    """
    tally: dict[str, dict[int, list[int]]] = {}
    for pair in iter_pairs(records):
        injector = pair.injected.author_family
        if not injector:
            continue
        clean = {x.solver_id: x for x in pair.clean_run.results}
        inj = {x.solver_id: x for x in pair.injected_run.results}
        eligible = [
            (sid, inj[sid])
            for sid, c in clean.items()
            if c.correct and sid in inj and inj[sid].correct is not None
        ]
        if len(eligible) < 2:
            continue
        derailed_total = sum(1 for _, i in eligible if not i.correct)
        for sid, i in eligible:
            stratum = derailed_total - int(not i.correct)
            cell = tally.setdefault(sid, {}).setdefault(stratum, [0, 0, 0, 0])
            own = family_of(i) == injector
            idx = (0 if own else 2) + (0 if not i.correct else 1)  # 'hit' = derailed
            cell[idx] += 1
    return _per_solver_and_pooled(tally)


# ---------------------------------------------------------------------- report


def _fmt(results: list[MHResult], own_label: str, hit_label: str) -> list[str]:
    if not results:
        return ["  (no usable data — needs rotated authorship and >=2 scored solvers/item)"]
    lines = [
        f"  {'solver':14} {'n_own':>6} {'n_oth':>6} {own_label:>9} {'other':>8} "
        f"{'OR':>6} {'delta':>7} {'p':>7}  DIF"
    ]
    for r in results:
        def pct(x):
            return "    n/a" if x is None else f"{x:7.1%}"

        or_s = "   n/a" if r.odds_ratio is None else f"{r.odds_ratio:6.2f}"
        d_s = "    n/a" if r.delta is None else f"{r.delta:+7.2f}"
        p_s = "    n/a" if r.p_value is None else f"{r.p_value:7.3f}"
        lines.append(
            f"  {r.label:14} {r.n_own:>6} {r.n_other:>6} {pct(r.own_rate)} "
            f"{pct(r.other_rate)} {or_s} {d_s} {p_s}  {r.category}"
        )
    per_solver = [r for r in results if r.label != "POOLED"]
    pooled = next((r for r in results if r.label == "POOLED"), None)
    flagged = [r for r in per_solver if r.category.startswith(("B", "C"))]
    lines.append(
        f"  -> {len(flagged)}/{len(per_solver)} individual solvers show "
        f"non-negligible DIF on {hit_label}"
    )
    if pooled is not None:
        verdict = (
            "LEAK DETECTED" if pooled.category.startswith(("B", "C")) else "no leak detected"
        )
        lines.append(f"  -> pooled across solvers: {verdict} ({pooled.category})")
    return lines


def format_leakage(records: list[ItemRecord]) -> str:
    author = author_advantage(records)
    injector = injector_interaction(records)
    out = [
        "Author advantage — is a solver better on items its OWN family wrote?",
        "  (odds ratio > 1 / delta < 0 favours own-family items)",
        *_fmt(author, "own", "authorship"),
        "",
        "Injector interaction — is a solver derailed LESS by its own family's claims?",
        "  (odds ratio < 1 / delta > 0 means own-family claims derail it less)",
        *_fmt(injector, "own", "injector identity"),
        "",
        "Any B/C above means the corresponding measurement is contaminated:",
        "  author advantage  -> exclude the author's family from its own items' panel",
        "                       and recompute a / theta",
        "  injector effect   -> report FRAGIL stratified by injector family, not pooled",
        "ETS scale: |delta| < 1.0 negligible, 1.0-1.5 moderate, >= 1.5 large",
        "(a category is only B/C when p < 0.05 as well)",
    ]
    return "\n".join(out)
