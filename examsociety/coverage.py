"""Knowledge-yield diagnostics: how much distinct material can a domain give?

The generator can only externalise knowledge the proposer already has, so a
domain has a finite pool of examinable concepts and the run saturates once that
pool is sampled. These functions estimate where that ceiling is, using the
concept labels the proposer already emits (ItemVersion.concepts).

The estimators are the standard species-richness ones: treat each distinct
concept as a species and each item as a sample, then ask how many species the
sample missed. Chao1 and Good-Turing both key off singletons/doubletons: if
most concepts are still appearing for the first time, the pool is far from
exhausted; if almost nothing is new, the domain is mined out.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .schemas import ItemRecord, ItemVersion

# Statuses whose item actually made it into the bank. rejected_invalid and
# panel_error carry no final item and are skipped automatically.
_BANKED = ("accepted", "too_easy", "too_hard", "suspect_too_hard", "paired")


def normalise_concept(c: str) -> str:
    c = re.sub(r"\s+", " ", c.strip().lower())
    return c.strip(" .,;:()[]{}。，、（）")


def banked_items(records: list[ItemRecord]) -> list[tuple[ItemRecord, ItemVersion]]:
    out = []
    for r in records:
        item = r.final_item()
        if item is not None and r.final_status in _BANKED:
            out.append((r, item))
    return out


def concept_counts(records: list[ItemRecord]) -> Counter:
    counts: Counter = Counter()
    for _, item in banked_items(records):
        for c in {normalise_concept(c) for c in item.concepts if normalise_concept(c)}:
            counts[c] += 1
    return counts


def accumulation_curve(records: list[ItemRecord]) -> list[tuple[int, int]]:
    """(items generated, distinct concepts seen so far), in generation order."""
    seen: set[str] = set()
    curve = []
    for n, (_, item) in enumerate(banked_items(records), start=1):
        seen.update(
            normalise_concept(c) for c in item.concepts if normalise_concept(c)
        )
        curve.append((n, len(seen)))
    return curve


@dataclass
class Richness:
    s_observed: int          # distinct concepts actually seen
    f1: int                  # concepts seen exactly once (singletons)
    f2: int                  # concepts seen exactly twice (doubletons)
    n_observations: int      # total concept mentions
    s_estimated: float       # Chao1: observed + estimated unseen
    coverage: float          # Good-Turing: fraction of the pool sampled
    saturated: bool          # coverage high enough to call the domain mined out


def richness(counts: Counter, saturation_coverage: float = 0.95) -> Richness:
    """Bias-corrected Chao1 plus Good-Turing sample coverage.

    Chao1 = S_obs + f1(f1-1) / (2(f2+1)). The bias-corrected form is used
    because the classic f1^2/(2 f2) is undefined once no concept has been seen
    exactly twice, which happens on short runs.
    """
    s_obs = len(counts)
    n = sum(counts.values())
    f1 = sum(1 for v in counts.values() if v == 1)
    f2 = sum(1 for v in counts.values() if v == 2)
    s_est = s_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1)) if s_obs else 0.0
    coverage = (1.0 - f1 / n) if n else 0.0
    return Richness(
        s_observed=s_obs,
        f1=f1,
        f2=f2,
        n_observations=n,
        s_estimated=float(s_est),
        coverage=float(coverage),
        saturated=bool(n) and coverage >= saturation_coverage,
    )


# ------------------------------------------------------------------ redundancy


def _bigrams(text: str) -> set[str]:
    """Character bigrams — language-agnostic, so this works on CJK stems too
    (word tokenisation would collapse a whole Chinese clause into one token)."""
    t = re.sub(r"\s+", "", text)
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _jaccard_pairs(
    sets: list[tuple[str, set]], threshold: float, max_pairs: int
) -> tuple[int, list[tuple[str, str, float]]]:
    """All pairs above a Jaccard threshold, via an inverted index.

    The brute-force form is O(n^2) in set intersections, which stops being
    usable in the thousands of items this tool is meant to produce. Any pair
    with Jaccard >= t must share at least one element, so the index over
    elements only ever proposes pairs that could possibly qualify, and the
    size-ratio bound (Jaccard <= min/max) discards most of those before the
    intersection is computed at all.
    """
    index: dict[object, list[int]] = {}
    for i, (_, s) in enumerate(sets):
        for token in s:
            index.setdefault(token, []).append(i)

    candidates: dict[tuple[int, int], int] = {}
    for holders in index.values():
        if len(holders) > 400:
            continue  # a token this common cannot discriminate anything
        for a_pos, i in enumerate(holders):
            for j in holders[a_pos + 1 :]:
                candidates[(j, i) if j < i else (i, j)] = 0

    pairs: list[tuple[str, str, float]] = []
    n_dupes = 0
    for i, j in candidates:
        (id_i, si), (id_j, sj) = sets[i], sets[j]
        lo, hi = (len(si), len(sj)) if len(si) < len(sj) else (len(sj), len(si))
        if not hi or lo / hi < threshold:
            continue
        inter = len(si & sj)
        jac = inter / (len(si) + len(sj) - inter)
        if jac >= threshold:
            n_dupes += 1
            if len(pairs) < max_pairs:
                pairs.append((id_i, id_j, jac))
    return n_dupes, pairs


def near_duplicate_stems(
    records: list[ItemRecord], threshold: float = 0.6, max_pairs: int = 50
) -> tuple[int, list[tuple[str, str, float]]]:
    """Surface-level near-duplicate detection over item stems.

    Catches 'same question, different numbers'. It will NOT catch a genuine
    concept repeat that was reworded — use concept_duplicates for that.
    """
    sets = [(r.id, _bigrams(item.stem)) for r, item in banked_items(records)]
    return _jaccard_pairs([(i, s) for i, s in sets if s], threshold, max_pairs)


def concept_duplicates(
    records: list[ItemRecord], threshold: float = 0.6, max_pairs: int = 50
) -> tuple[int, list[tuple[str, str, float]]]:
    """Items testing the same concepts, however differently they are worded.

    This is the half that stem similarity structurally cannot see: rewriting the
    scenario changes every character bigram while leaving the examined concept
    identical, which is exactly the degeneration a saturating generator falls
    into. Relies on the proposer's own `concepts` labels, so it is only as good
    as those labels — but it is a far better redundancy signal than surface
    text, and it is free.
    """
    sets = []
    for r, item in banked_items(records):
        cs = {normalise_concept(c) for c in item.concepts if normalise_concept(c)}
        if len(cs) >= 2:  # a single shared label is too weak to call a duplicate
            sets.append((r.id, cs))
    return _jaccard_pairs(sets, threshold, max_pairs)


# ---------------------------------------------------------------------- report


@dataclass
class CoverageReport:
    n_items: int
    overall: Richness
    per_subtopic: dict[str, Richness] = field(default_factory=dict)
    curve: list[tuple[int, int]] = field(default_factory=list)
    n_near_duplicate_pairs: int = 0        # same wording
    n_concept_duplicate_pairs: int = 0     # same concepts, any wording
    new_concepts_last_quarter: float = 0.0  # new concepts per item, final 25%


def analyse_coverage(records: list[ItemRecord]) -> CoverageReport:
    banked = banked_items(records)
    curve = accumulation_curve(records)

    by_subtopic: dict[str, list[ItemRecord]] = {}
    for r, _ in banked:
        by_subtopic.setdefault(r.subtopic, []).append(r)

    # Marginal yield over the last quarter of the run: the honest answer to
    # "is it still finding new material or just rephrasing?"
    rate = 0.0
    if len(curve) >= 8:
        cut = (len(curve) * 3) // 4
        span = len(curve) - cut
        rate = (curve[-1][1] - curve[cut - 1][1]) / span if span else 0.0

    n_dupes, _ = near_duplicate_stems(records)
    n_concept_dupes, _ = concept_duplicates(records)
    return CoverageReport(
        n_items=len(banked),
        overall=richness(concept_counts(records)),
        per_subtopic={
            st: richness(concept_counts(rs)) for st, rs in sorted(by_subtopic.items())
        },
        curve=curve,
        n_near_duplicate_pairs=n_dupes,
        n_concept_duplicate_pairs=n_concept_dupes,
        new_concepts_last_quarter=rate,
    )


def format_coverage(rep: CoverageReport) -> str:
    o = rep.overall
    lines = [
        f"banked items: {rep.n_items}",
        f"distinct concepts: {o.s_observed}  (mentions: {o.n_observations}, "
        f"singletons: {o.f1}, doubletons: {o.f2})",
        f"estimated concept pool (Chao1): {o.s_estimated:.0f}  "
        f"-> {max(0.0, o.s_estimated - o.s_observed):.0f} still unseen",
        f"sample coverage (Good-Turing): {o.coverage:.1%}"
        + ("   [SATURATED]" if o.saturated else ""),
        f"new concepts per item, last 25% of run: {rep.new_concepts_last_quarter:.2f}",
        f"near-duplicate stem pairs: {rep.n_near_duplicate_pairs}  "
        f"(surface wording)",
        f"same-concept pairs: {rep.n_concept_duplicate_pairs}  "
        f"(rewritten but testing the same thing)",
    ]
    if rep.per_subtopic:
        lines.append("\nper subtopic:")
        lines.append(f"  {'subtopic':38} {'concepts':>9} {'pool':>7} {'cover':>7}")
        for st, r in rep.per_subtopic.items():
            lines.append(
                f"  {st[:38]:38} {r.s_observed:>9} {r.s_estimated:>7.0f} "
                f"{r.coverage:>6.1%}"
            )
    return "\n".join(lines)
