from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .llm import ModelRef

# Substring -> family label, for grouping models in the author-advantage and
# injector-interaction analyses. A model's family matters more than its exact
# name: same-family models share training priors, which is exactly the
# confound those analyses are looking for.
#
# Matched longest-first: "doubao-1.5" contains the substring "o1", so a short
# token must never be tested before a longer one that could contain it.
_FAMILIES = tuple(
    sorted(
        (
            "claude", "gpt", "gemini", "deepseek", "qwen", "kimi", "moonshot",
            "glm", "llama", "mistral", "grok", "ernie", "doubao", "step",
            "o1", "o3",
        ),
        key=len,
        reverse=True,
    )
)


def guess_family(model: str, default: str = "") -> str:
    """Infer a vendor family from a model name. Shared with the leakage
    analyses, which have to work on records that only stored the model name."""
    low = model.lower()
    return next((f for f in _FAMILIES if f in low), default)


class ModelSpec(BaseModel):
    """One addressable model. Accepts a bare model name as shorthand."""

    id: str
    model: str
    # Wire protocol, NOT a vendor: DeepSeek / Qwen / Kimi / GLM / Gemini all
    # speak the openai chat-completions protocol, so `api: openai` on a Kimi
    # endpoint means "talk to it in OpenAI's request shape", not "use OpenAI".
    # Vendor identity lives in `family`, which is what the author-advantage
    # analyses group by. Write it as `protocol:` — `api:` and `provider:` are
    # accepted aliases, but the name has already misled a reader once.
    api: str = "anthropic"               # anthropic | openai
    base_url: Optional[str] = None       # None -> provider default / env
    api_key_env: Optional[str] = None
    token_param: str = "max_tokens"      # some endpoints want max_completion_tokens
    family: Optional[str] = None         # overrides the substring guess below
    # Extra request-body fields passed through verbatim, e.g.
    # extra_body: {reasoning_effort: low}. Two entries pointing at the same
    # model with different effort are genuinely different ability levels, which
    # is what IRT and the drift check need and what a single model otherwise
    # cannot provide.
    extra_body: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"id": v, "model": v}
        if isinstance(v, dict):
            # `protocol` is the accurate name; `api`/`provider` are kept working
            # because existing configs use them.
            for alias in ("protocol", "provider"):
                if alias in v and "api" not in v:
                    v = {**v, "api": v[alias]}
            if "model" not in v and "id" in v:
                v = {**v, "model": v["id"]}
            elif "id" not in v and "model" in v:
                v = {**v, "id": v["model"]}
        return v

    @property
    def resolved_family(self) -> str:
        if self.family:
            return self.family
        return guess_family(self.model, default=self.api)

    @property
    def family_is_guessed_from_protocol(self) -> bool:
        """True when no vendor token matched and the family fell back to the
        wire protocol.

        That fallback is always wrong for the leakage analyses: every
        unrecognised OpenAI-protocol model collapses into one fake `openai`
        family, so models from different vendors look related and models from
        the same vendor (one named `kimi-...`, one named `k3-...`) look
        unrelated. It has to be visible rather than silent.
        """
        return not self.family and not guess_family(self.model)

    def ref(self) -> ModelRef:
        return ModelRef(
            id=self.id,
            model=self.model,
            api=self.api,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            token_param=self.token_param,
            extra_body=dict(self.extra_body),
            family=self.resolved_family,
        )



def _default(name: str = "claude-opus-4-8") -> ModelSpec:
    return ModelSpec(id=name, model=name)


class RolesConfig(BaseModel):
    """Role assignments. proposers/validators/hardeners are LISTS: rotating a
    role across model families is what breaks the self-preference leak (the
    author of an item should not be the only judge of it) and it widens the
    concept pool at the same time."""

    topic_mapper: ModelSpec = Field(default_factory=_default)
    proposers: list[ModelSpec] = Field(default_factory=lambda: [_default()])
    validators: list[ModelSpec] = Field(default_factory=lambda: [_default()])
    hardeners: list[ModelSpec] = Field(default_factory=lambda: [_default()])

    @model_validator(mode="before")
    @classmethod
    def _accept_singular(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        for singular, plural in (
            ("proposer", "proposers"),
            ("validator", "validators"),
            ("hardener", "hardeners"),
        ):
            if singular in v:
                # older configs used a single model per role
                v.setdefault(plural, [v[singular]])
                v.pop(singular)
            if plural in v and not isinstance(v[plural], list):
                v[plural] = [v[plural]]
        return v


class LoopConfig(BaseModel):
    # accept an item once its panel solve rate is at or below this value
    target_max_solve_rate: float = 0.5
    max_revisions: int = 3
    # from revision round 1 on, inject misinformation with this probability
    misinfo_probability: float = 0.5
    proposal_retries: int = 2
    subtopics_per_domain: int = 12
    # How many previous stems / concepts FROM THE SAME SUBTOPIC to show the
    # proposer. A global "last N stems" window constrains nothing once subtopics
    # are cycled, because the recent ones are all about something else.
    avoid_window: int = 40
    avoid_concepts_window: int = 80
    # Grow the subtopic pool every N items (0 disables). Without this the run's
    # knowledge ceiling is the size of the pool the topic mapper produced once
    # at startup, no matter how long it runs.
    subtopic_refresh_every: int = 50
    subtopic_refresh_size: int = 8
    # Answer each item this many times per solver. With one attempt there is no
    # way to tell a hard item from an item the panel answers at random, and the
    # acceptance rule selects for the latter. >1 costs k x panel calls.
    solve_replicates: int = 1
    # Accept only if the panel solve rate is INSIDE [min, max]. A ceiling alone
    # pushes the whole bank towards the guessing floor, where item variance and
    # therefore discrimination go to zero. 0.0 keeps the old ceiling-only rule.
    target_min_solve_rate: float = 0.0
    # How many items to generate concurrently. The panel already fans out
    # across seats, but items ran strictly serially, so wall-clock was the sum
    # of every item's chain -- measured 34-50 min per pair, ~11 days for the 400
    # pairs the dimension test needs. Cost is that items in flight cannot see
    # each other's stems/concepts, so the avoid-window goes blind for one batch.
    item_workers: int = 1
    # Stop the run after this many model calls (0 = unlimited).
    max_api_calls: int = 0
    # paired pipeline: extra attempts when the injector breaks the minimal-edit
    # contract (options modified/reordered). Each retry is one call; giving up
    # discards the ~7 calls already spent on the clean half.
    injection_retries: int = 2
    # paired pipeline: re-run the CLEAN half a second time as a control. Without
    # it, "wrong on the injected half after being right on the clean half" also
    # counts every answer the solver would have flipped on its own, which
    # inflates the measured derailment of unstable (usually weaker) models.
    pair_control_replicate: bool = True
    # prefer a validator from a different family than the item's author, so that
    # key_agrees is not inflated by shared blind spots
    cross_family_validation: bool = True
    # How many validators must independently solve the item and land on the key.
    # >1 draws them from distinct families and rejects on the first dissent. The
    # dissent rate is itself the deliverable: it is a lower bound on the answer
    # key error rate, which single-validator gating cannot estimate at all.
    validator_consensus: int = 1


class GenerationConfig(BaseModel):
    options_count: int = 5
    max_tokens: int = 4000
    solver_max_tokens: int = 2000


class Config(BaseModel):
    # Legacy no-op, kept so older configs still load. Protocol is per-model
    # (ModelSpec.api); vendor identity is ModelSpec.family.
    provider: str = "anthropic"
    roles: RolesConfig = Field(default_factory=RolesConfig)
    solvers: list[ModelSpec] = Field(
        default_factory=lambda: [
            ModelSpec(id="opus46", model="claude-opus-4-6"),
            ModelSpec(id="opus47", model="claude-opus-4-7"),
            ModelSpec(id="opus48", model="claude-opus-4-8"),
        ]
    )
    loop: LoopConfig = Field(default_factory=LoopConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


def warn_on_guessed_families(cfg: Config) -> list[str]:
    """Report models whose family fell back to the wire-protocol name.

    `examsociety leakage` groups by family, so a bad label does not fail — it
    quietly answers the wrong question. Two entries from one vendor named
    `k3-256k` and `kimi-for-coding` land in different families; two entries from
    different vendors with unrecognised names land in the same one.
    """
    bad = [
        m.id
        for m in [
            cfg.roles.topic_mapper,
            *cfg.roles.proposers,
            *cfg.roles.validators,
            *cfg.roles.hardeners,
            *cfg.solvers,
        ]
        if m.family_is_guessed_from_protocol
    ]
    if bad:
        print(
            "[config] WARNING: no vendor family could be inferred for "
            + ", ".join(sorted(set(bad)))
            + " — they fell back to the wire-protocol name. `examsociety "
            "leakage` groups by family, so this silently splits or merges "
            "vendors. Add an explicit `family:` to those entries.",
            file=sys.stderr,
        )
    return bad


def load_config(path: Optional[str]) -> Config:
    if path is None:
        return Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate(raw)
    warn_on_guessed_families(cfg)
    return cfg
