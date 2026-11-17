from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class Misinfo(BaseModel):
    """Metadata for deliberately injected misleading content.

    The injected claim must be detectably false in principle: an examinee with
    solid domain knowledge should be able to recognise and discard it. This is
    what separates a robustness probe from an unfair trick question.
    """

    error_type: str  # false_premise | distractor_evidence | plausible_misattribution | spurious_constraint
    false_claim: str
    why_detectable: str
    # Which option the false claim steers towards. Lets the analysis separate
    # "got it wrong" from "got derailed onto the bait", which is a far sharper
    # signal and rules out 'the injection just made the item harder'.
    target_option_index: Optional[int] = None


class ItemVersion(BaseModel):
    version: int
    stem: str
    options: list[str]
    answer_index: int
    solution: str
    concepts: list[str] = Field(default_factory=list)
    misinfo: Optional[Misinfo] = None
    revision_reason: Optional[str] = None
    # Which model produced this version: the proposer for v0, the hardener or
    # injector for later ones. Required to test whether a model is advantaged on
    # items it authored, and whether a solver is unusually robust to false
    # claims written by its own family.
    author_id: Optional[str] = None
    author_family: Optional[str] = None


class SolverResult(BaseModel):
    solver_id: str
    model: str
    # Needed by the leakage analyses; falls back to guess_family(model) for
    # records written before this field existed.
    family: Optional[str] = None
    answer_index: Optional[int]  # None = unparsable / error
    correct: Optional[bool]
    reasoning_brief: str = ""
    error: Optional[str] = None


class PanelRun(BaseModel):
    version: int
    results: list[SolverResult]
    # Fraction of SCORED solvers that were correct. Always read together with
    # n_scored: solve_rate=0.0 means "nobody solved it" only when the panel
    # actually reached quorum, otherwise it just means "no usable answers".
    solve_rate: float
    n_scored: Optional[int] = None  # None = record written before this field existed


class ValidationResult(BaseModel):
    verdict: str  # pass | fail
    independent_answer_index: Optional[int] = None
    key_agrees: Optional[bool] = None
    issues: list[str] = Field(default_factory=list)
    # Which ItemVersion this verdict is about. Needed to group the validators
    # of one consensus round together — record.validations is a flat list that
    # spans every version the item went through.
    version: Optional[int] = None
    validator_id: Optional[str] = None
    validator_family: Optional[str] = None
    # False when author and validator share a family, i.e. key_agrees may be
    # inflated by shared blind spots rather than independent confirmation.
    cross_family: Optional[bool] = None


class ItemRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    domain: str
    subtopic: str
    pipeline: str = "society"  # society | baseline | paired | harden-paired
    versions: list[ItemVersion] = Field(default_factory=list)
    validations: list[ValidationResult] = Field(default_factory=list)
    panel_runs: list[PanelRun] = Field(default_factory=list)
    # accepted | too_easy | suspect_too_hard | rejected_invalid | panel_error | error
    # panel_error: too few solvers returned a usable answer to score the item at
    # all. Carries no difficulty information and is excluded from psychometrics.
    final_status: str = "error"
    final_version: Optional[int] = None
    created_at: float = Field(default_factory=time.time)

    def final_item(self) -> Optional[ItemVersion]:
        if self.final_version is None:
            return None
        for v in self.versions:
            if v.version == self.final_version:
                return v
        return None
