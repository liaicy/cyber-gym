from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .schemas import ItemRecord


@dataclass
class ItemStats:
    item_id: str
    p_value: float          # classical difficulty (proportion correct)
    point_biserial: float   # classical discrimination (item-rest correlation)
    irt_a: float            # 2PL discrimination
    irt_b: float            # 2PL difficulty
    flag: str               # "" | "low_discrimination" | "negative_discrimination"


SCORABLE_STATUSES = (
    "accepted",
    "suspect_too_hard",
    "too_easy",
    "too_hard",
    "paired",
)


def _scored_run(record: ItemRecord) -> "PanelRunLike | None":
    """The panel run that represents this item for psychometric purposes.

    For a paired record that is the CLEAN half: the injected half is a
    deliberately perturbed variant, so its difficulty is a property of the probe
    rather than of the item. For everything else it is the FIRST run at the
    final version — with solve_replicates > 1 there are several, and the first
    is one unbiased measurement rather than a silently-chosen last one.
    """
    if not record.panel_runs:
        return None
    if record.final_status == "paired":
        clean = next((v for v in record.versions if v.misinfo is None), None)
        if clean is None:
            return None
        return next((p for p in record.panel_runs if p.version == clean.version), None)
    return next(
        (p for p in record.panel_runs if p.version == record.final_version),
        record.panel_runs[0],
    )


def response_consistency(records: list[ItemRecord]) -> dict:
    """Agreement between repeated attempts at the same item version.

    Only meaningful with loop.solve_replicates > 1. Low consistency means the
    bank is full of items answered at random rather than items that are hard —
    and the acceptance rule cannot tell those apart on its own, because an
    unstable item lands under the solve-rate ceiling by chance.
    """
    agree = total = 0
    per_item = []
    for r in records:
        by_version: dict[int, list] = {}
        for p in r.panel_runs:
            by_version.setdefault(p.version, []).append(p)
        for runs in by_version.values():
            if len(runs) < 2:
                continue
            answers: dict[str, list] = {}
            for run in runs:
                for res in run.results:
                    if res.correct is not None:
                        answers.setdefault(res.solver_id, []).append(bool(res.correct))
            hits = [len(set(v)) == 1 for v in answers.values() if len(v) > 1]
            if hits:
                agree += sum(hits)
                total += len(hits)
                per_item.append(sum(hits) / len(hits))
    return {
        "n_replicated_cells": total,
        "consistency": (agree / total) if total else None,
        "n_items": len(per_item),
        "mean_item_consistency": float(np.mean(per_item)) if per_item else None,
    }


def response_matrix(
    records: list[ItemRecord], statuses: tuple[str, ...] = SCORABLE_STATUSES
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build items x solvers 0/1 matrix from each record's final panel run.

    Returns (X, item_ids, solver_ids); unscored cells are NaN. Items whose final
    panel produced no scored answer at all are dropped: an all-NaN row carries
    no information but would still be handed a default discrimination by
    fit_2pl and counted as a real item by the reports.
    """
    solver_ids: list[str] = []
    rows, item_ids = [], []
    for r in records:
        if r.final_status not in statuses:
            continue
        final_run = _scored_run(r)
        if final_run is None:
            continue
        row = {
            res.solver_id: res.correct
            for res in final_run.results
            if res.correct is not None
        }
        if not row:
            continue
        for res in final_run.results:
            if res.solver_id not in solver_ids:
                solver_ids.append(res.solver_id)
        rows.append(row)
        item_ids.append(r.id)
    X = np.full((len(rows), len(solver_ids)), np.nan)
    for i, row in enumerate(rows):
        for j, sid in enumerate(solver_ids):
            v = row.get(sid)
            if v is not None:
                X[i, j] = 1.0 if v else 0.0
    return X, item_ids, solver_ids


def classical_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-item p-values and item-rest point-biserial correlations."""
    with np.errstate(all="ignore"):
        row_has_data = (~np.isnan(X)).any(axis=1)
        p = np.full(X.shape[0], np.nan)
        p[row_has_data] = np.nanmean(X[row_has_data], axis=1)
        totals = np.nansum(X, axis=0)
        pb = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            rest = totals - np.nan_to_num(X[i])
            mask = ~np.isnan(X[i])
            x, y = X[i, mask], rest[mask]
            if x.size < 2 or x.std() < 1e-9 or y.std() < 1e-9:
                pb[i] = 0.0
            else:
                pb[i] = float(np.corrcoef(x, y)[0, 1])
    return p, pb


def fit_2pl(
    X: np.ndarray, n_rounds: int = 30, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint MAP estimation of a 2PL model by alternating optimisation.

    Priors: log(a) ~ N(0, 0.5), b ~ N(0, 2), theta ~ N(0, 1).
    Suitable for the small panels this tool produces; treat estimates as
    exploratory until the panel has dozens of solvers.

    Returns (a, b, theta).
    """
    n_items, n_solvers = X.shape
    mask = ~np.isnan(X)
    Xf = np.nan_to_num(X)
    if n_items < 2 or n_solvers < 2:
        # too small to identify a 2PL scale; return neutral placeholders
        return np.zeros(n_items), np.full(n_items, np.nan), np.zeros(n_solvers)

    # init: theta from standardised total scores, b from item p-values
    scores = np.array(
        [np.nanmean(X[:, j]) if mask[:, j].any() else 0.5 for j in range(n_solvers)]
    )
    theta = (scores - scores.mean()) / (scores.std() + 1e-9)
    with np.errstate(all="ignore"):
        p = np.clip(np.nanmean(X, axis=1), 0.02, 0.98)
    b = -np.log(p / (1 - p))
    log_a = np.zeros(n_items)

    def item_nll(params, i, theta_):
        la, bi = params
        a = math.exp(la)
        z = a * (theta_[mask[i]] - bi)
        pr = expit(z)
        pr = np.clip(pr, 1e-9, 1 - 1e-9)
        x = Xf[i, mask[i]]
        ll = np.sum(x * np.log(pr) + (1 - x) * np.log(1 - pr))
        prior = -(la**2) / (2 * 0.5**2) - (bi**2) / (2 * 2.0**2)
        return -(ll + prior)

    def theta_nll(t, j, log_a_, b_):
        a = np.exp(log_a_[mask[:, j]])
        z = a * (t[0] - b_[mask[:, j]])
        pr = np.clip(expit(z), 1e-9, 1 - 1e-9)
        x = Xf[mask[:, j], j]
        ll = np.sum(x * np.log(pr) + (1 - x) * np.log(1 - pr))
        return -(ll - t[0] ** 2 / 2)

    prev = None
    errctx = np.errstate(all="ignore")
    errctx.__enter__()
    for _ in range(n_rounds):
        for i in range(n_items):
            if not mask[i].any():
                continue
            res = minimize(
                item_nll, [log_a[i], b[i]], args=(i, theta), method="L-BFGS-B",
                bounds=[(-2.5, 2.5), (-6, 6)],
            )
            log_a[i], b[i] = res.x
        for j in range(n_solvers):
            if not mask[:, j].any():
                continue
            res = minimize(
                theta_nll, [theta[j]], args=(j, log_a, b), method="L-BFGS-B",
                bounds=[(-4, 4)],
            )
            theta[j] = res.x[0]
        # identify the scale: theta ~ N(0,1)
        mu, sd = theta.mean(), theta.std() + 1e-9
        theta = (theta - mu) / sd
        b = (b - mu) / sd
        log_a = log_a + math.log(sd)
        total = sum(
            item_nll([log_a[i], b[i]], i, theta) for i in range(n_items) if mask[i].any()
        )
        if prev is not None and abs(prev - total) < tol:
            break
        prev = total
    errctx.__exit__(None, None, None)
    return np.exp(log_a), b, theta


def _run_discrimination(run) -> float | None:
    """Item-rest correlation for a single panel run, using solver identity as
    the only ability signal available within one item."""
    scored = [r for r in run.results if r.correct is not None]
    if len(scored) < 3:
        return None
    return float(np.mean([1.0 if r.correct else 0.0 for r in scored]))


def revision_drift(records: list[ItemRecord]) -> dict:
    """Does the adversarial loop buy difficulty by getting more obscure?

    When "test the concept more deeply" runs out, the cheapest remaining way to
    push the solve rate down is to reach for a more obscure fact. That looks
    like success on every difficulty metric while quietly changing what is being
    measured from reasoning to recall.

    The signature is an ASSOCIATION, not a spread. If an item is hardened by
    depth, being able to do the old version should strongly predict being able
    to do the new one — the same solvers stay ahead. If it is hardened by
    obscurity, the new version turns on whether a solver happens to have
    memorised some fact, which has little to do with who could do the old one,
    so that prediction collapses towards zero.

    Measured by pooling every (item, solver) cell across one revision step:

        retention = P(correct after | correct before)
        flip_in   = P(correct after | wrong before)
        carryover = retention - flip_in

    Carryover near 1 means the revision preserved the ability ordering; near 0
    means the new version is uninformative about the old one, i.e. difficulty
    was bought with obscurity.

    Note this deliberately does NOT use the spread of per-solver accuracy: with
    solve_replicates = 1 each solver's accuracy is 0 or 1, so its standard
    deviation is a deterministic function of the mean and carries no information
    beyond difficulty itself.
    """
    steps = []
    cells = {"a": 0, "b": 0, "c": 0, "d": 0}  # before x after contingency
    for r in records:
        by_version: dict[int, list] = {}
        for p in r.panel_runs:
            by_version.setdefault(p.version, []).append(p)
        versions = sorted(by_version)
        acc: dict[int, dict[str, float]] = {}
        for v in versions:
            tally: dict[str, list] = {}
            for run in by_version[v]:
                for res in run.results:
                    if res.correct is not None:
                        tally.setdefault(res.solver_id, []).append(float(res.correct))
            acc[v] = {k: float(np.mean(vals)) for k, vals in tally.items() if vals}
        for a, b in zip(versions, versions[1:]):
            shared = sorted(set(acc[a]) & set(acc[b]))
            if len(shared) < 3:
                continue
            before = np.array([acc[a][s] for s in shared])
            after = np.array([acc[b][s] for s in shared])
            for x, y in zip(before, after):
                # graded accuracies (replicates > 1) contribute fractionally
                cells["a"] += x * y
                cells["b"] += x * (1 - y)
                cells["c"] += (1 - x) * y
                cells["d"] += (1 - x) * (1 - y)
            steps.append(
                {
                    "item": r.id,
                    "from": a,
                    "to": b,
                    "d_difficulty": float(before.mean() - after.mean()),  # + = harder
                    "misinfo": any(
                        v.version == b and v.misinfo is not None for v in r.versions
                    ),
                }
            )
    if not steps:
        return {"n_steps": 0, "verdict": "no revision steps with >=3 shared solvers"}

    could_before = cells["a"] + cells["b"]
    could_not = cells["c"] + cells["d"]
    retention = (cells["a"] / could_before) if could_before else None
    flip_in = (cells["c"] / could_not) if could_not else None
    carryover = (
        retention - flip_in if retention is not None and flip_in is not None else None
    )
    harder = [s for s in steps if s["d_difficulty"] > 0]
    mean_dd = float(np.mean([s["d_difficulty"] for s in steps]))

    if not harder:
        verdict = "no step made the item harder"
    elif carryover is None:
        verdict = "not enough variation before the revision to judge"
    elif carryover < 0.15:
        verdict = "OBSCURITY DRIFT — the revised item no longer tracks who could do the original"
    elif carryover < 0.35:
        verdict = "mixed — some difficulty is coming from obscurity"
    else:
        verdict = "difficulty is tracking ability, not obscurity"
    return {
        "n_steps": len(steps),
        "n_harder": len(harder),
        "mean_d_difficulty": mean_dd,
        "retention": retention,
        "flip_in": flip_in,
        "carryover": carryover,
        "verdict": verdict,
        "steps": steps,
    }


def format_drift(d: dict) -> str:
    if not d.get("n_steps"):
        return d.get("verdict", "no data")
    lines = [
        f"revision steps analysed: {d['n_steps']}  (made harder: {d['n_harder']})",
        f"mean change in difficulty: {d['mean_d_difficulty']:+.3f}  "
        f"(+ = solve rate fell, i.e. harder)",
        f"mean change in solver spread: {d['mean_d_spread']:+.3f}  "
        f"(- = solvers pulled together, i.e. less able to discriminate)",
    ]
    if d["share_flattened"] is not None:
        lines.append(
            f"steps that got harder AND flatter: {d['n_harder_and_flatter']}"
            f"/{d['n_harder']} = {d['share_flattened']:.0%}"
        )
    lines.append(f"-> {d['verdict']}")
    lines.append(
        "\nReading: hardening by DEPTH keeps strong solvers ahead, so spread "
        "survives.\nHardening by OBSCURITY pushes everyone towards chance, so "
        "spread collapses\nwhile the solve rate falls — which every other metric "
        "reports as success."
    )
    return "\n".join(lines)


def analyse(records: list[ItemRecord]) -> tuple[list[ItemStats], np.ndarray, list[str]]:
    X, item_ids, solver_ids = response_matrix(records)
    if X.size == 0:
        return [], np.array([]), []
    p, pb = classical_stats(X)
    a, b, theta = fit_2pl(X)
    stats = []
    for i, iid in enumerate(item_ids):
        flag = ""
        if pb[i] < 0 or a[i] < 0.2:
            flag = "negative_discrimination" if pb[i] < 0 else "low_discrimination"
        stats.append(
            ItemStats(
                item_id=iid,
                p_value=float(p[i]),
                point_biserial=float(pb[i]),
                irt_a=float(a[i]),
                irt_b=float(b[i]),
                flag=flag,
            )
        )
    return stats, theta, solver_ids
