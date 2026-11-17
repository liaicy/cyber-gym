from __future__ import annotations

import json
import random
import re
from typing import Optional

from .config import Config
from .llm import LLMClient, LLMError, ModelRef, UpstreamUnavailable
from .schemas import ItemVersion, Misinfo, SolverResult, ValidationResult

# Every type must be a claim ABOUT THE WORLD, never a claim about how to solve
# this item. Measured on the first four pairs: fact-shaped claims derailed
# solvers onto the baited option; claims that spelled out the flawed reasoning
# step made the item EASIER, because naming the trap is how an expert finds it.
MISINFO_TYPES = [
    # the stem asserts a plausible but false background fact
    "false_premise",
    # a fabricated concrete datum -- a value, a named theorem, an experimental
    # result -- that happens to favour a wrong option. A DATUM, not an argument.
    "distractor_evidence",
    # a real concept attributed to the wrong source, regime or condition
    "plausible_misattribution",
    # a false statement about what the system satisfies (a symmetry, a
    # conservation law, a boundary condition) that is not actually true here
    "spurious_constraint",
]

_JSON_ONLY = (
    "Respond with ONLY a single valid JSON object. No markdown fences, no extra "
    "text. Escape any double quotes inside JSON string values. Keep 'solution' "
    "rigorous but compact (under ~250 words)."
)


class TopicMapper:
    SYSTEM = (
        "You are a curriculum architect. Given a subject domain or keyword, map out "
        "the most important, examination-worthy subtopics — the concepts a rigorous "
        "graduate-level exam in this domain must cover. Prefer subtopics where deep "
        "understanding separates experts from novices. Use the same language as the "
        "domain input for subtopic names. " + _JSON_ONLY
    )

    def __init__(self, llm: LLMClient, ref: ModelRef):
        self.llm, self.ref = llm, ref

    def map(self, domain: str, n: int) -> list[str]:
        user = (
            f"Domain: {domain}\n"
            f"Produce exactly {n} distinct subtopics as JSON: "
            '{"subtopics": ["...", "..."]}'
        )
        return self._ask(user)

    def expand(
        self, domain: str, n: int, existing: list[str], covered_concepts: list[str]
    ) -> list[str]:
        """Grow the subtopic pool once the current one is mined out.

        A fixed pool cycled forever caps the run's knowledge yield no matter how
        long it runs: every later item is a rephrasing of material the pool has
        already produced. Expanding against what has actually been examined is
        what turns the ceiling from 'size of the initial pool' into 'size of the
        domain'.
        """
        covered = ", ".join(covered_concepts[:120])
        user = (
            f"Domain: {domain}\n\n"
            "Subtopics already in use:\n- " + "\n- ".join(existing[:60]) + "\n\n"
            f"Concepts already examined:\n{covered}\n\n"
            f"Produce exactly {n} ADDITIONAL subtopics that do NOT overlap with "
            "any of the above — go deeper, sideways, or into adjacent material "
            "the list has missed. Same language as the domain input.\n"
            'JSON: {"subtopics": ["...", "..."]}'
        )
        fresh = self._ask(user)
        seen = {s.strip().lower() for s in existing}
        return [s for s in fresh if s.strip().lower() not in seen]

    def _ask(self, user: str) -> list[str]:
        obj = self.llm.complete_json(self.ref, self.SYSTEM, user)
        subtopics = [str(s).strip() for s in obj.get("subtopics", []) if str(s).strip()]
        if not subtopics:
            raise LLMError("topic mapper returned no subtopics")
        return subtopics


def _item_from_obj(obj: dict, version: int, options_count: int) -> ItemVersion:
    options = [str(o) for o in obj["options"]]
    answer_index = int(obj["answer_index"])
    if not (2 <= len(options) <= max(6, options_count)):
        raise ValueError(f"bad options count: {len(options)}")
    if not (0 <= answer_index < len(options)):
        raise ValueError(f"answer_index {answer_index} out of range")
    misinfo = None
    if obj.get("misinfo"):
        m = obj["misinfo"]
        target = m.get("target_option_index")
        if target is not None:
            target = int(target)
            if not (0 <= target < len(options)) or target == answer_index:
                raise ValueError(
                    f"misinfo target_option_index {target} is not a wrong option"
                )
        misinfo = Misinfo(
            error_type=str(m.get("error_type", "unknown")),
            false_claim=str(m.get("false_claim", "")),
            why_detectable=str(m.get("why_detectable", "")),
            target_option_index=target,
        )
    return ItemVersion(
        version=version,
        stem=str(obj["stem"]),
        options=options,
        answer_index=answer_index,
        solution=str(obj.get("solution", "")),
        concepts=[str(c) for c in obj.get("concepts", [])],
        misinfo=misinfo,
        revision_reason=obj.get("revision_reason"),
    )


class Proposer:
    SYSTEM = (
        "You are an expert exam item writer for graduate-level and professional "
        "examinations. You write multiple-choice questions that test deep conceptual "
        "understanding, not recall. Requirements:\n"
        "- Exactly ONE option is correct; every distractor must be plausible to a "
        "novice but definitively wrong to an expert.\n"
        "- The stem must be self-contained, unambiguous, and well-posed.\n"
        "- Target difficulty: should challenge a strong graduate student in the field.\n"
        "- Write the question in the same language as the domain/subtopic input.\n"
        + _JSON_ONLY
    )

    def __init__(self, llm: LLMClient, ref: ModelRef, cfg: Config):
        self.llm, self.ref, self.cfg = llm, ref, cfg

    def propose(
        self,
        domain: str,
        subtopic: str,
        avoid: list[str],
        avoid_concepts: Optional[list[str]] = None,
    ) -> ItemVersion:
        # The caller decides what to put here — it filters by subtopic, which
        # matters once subtopics are cycled: the globally most recent stems are
        # mostly about something else and constrain nothing.
        avoid_block = ""
        if avoid:
            avoid_block = (
                "\nAvoid overlapping with these already-generated question stems:\n- "
                + "\n- ".join(a[:160] for a in avoid)
            )
        if avoid_concepts:
            avoid_block += (
                "\n\nThese concepts have ALREADY been examined in this subtopic. "
                "Test something genuinely different — do not rephrase them:\n"
                + ", ".join(avoid_concepts)
            )
        n = self.cfg.generation.options_count
        user = (
            f"Domain: {domain}\nSubtopic: {subtopic}{avoid_block}\n\n"
            f"Write one hard multiple-choice question with {n} options.\n"
            "JSON schema: {"
            '"stem": str, "options": [str x'
            f"{n}"
            '], "answer_index": int, "solution": str (full reasoning), '
            '"concepts": [str]}'
        )
        obj = self.llm.complete_json(
            self.ref, self.SYSTEM, user, self.cfg.generation.max_tokens
        )
        return _item_from_obj(obj, version=0, options_count=n)


class Validator:
    """Validity gate: solves the item blind, then critiques well-posedness.

    The key check happens in code: the validator never sees the answer key, so
    key_agrees is an independent expert-solve comparison, not a rubber stamp.
    """

    SYSTEM = (
        "You are a meticulous domain expert reviewing an exam question you did not "
        "write. First solve it yourself from scratch. Then critique it: is it "
        "well-posed, is exactly one option defensible, are there ambiguities or "
        "errors? Note: the stem may deliberately contain misleading background "
        "claims as a robustness probe — that alone is NOT a flaw, as long as the "
        "misleading claim is recognisably false to an expert and one option remains "
        "uniquely correct. " + _JSON_ONLY
    )

    def __init__(self, llm: LLMClient, ref: ModelRef, cfg: Config):
        self.llm, self.ref, self.cfg = llm, ref, cfg

    def validate(self, item: ItemVersion) -> ValidationResult:
        probe_note = ""
        if item.misinfo is not None:
            probe_note = (
                "\nNOTE: this item INTENTIONALLY embeds one misleading claim as a "
                "robustness probe. The presence of a false claim is by design and "
                "must NOT by itself make well_posed false. Set well_posed=false "
                "only if (a) no option remains uniquely correct once the false "
                "claim is discarded, or (b) the false claim could not be "
                "recognised as false by a well-trained expert using standard "
                "domain knowledge.\n"
            )
        user = (
            f"{_render_item(item)}\n{probe_note}\n"
            "JSON schema: {"
            '"answer_index": int (your own independently derived answer), '
            '"well_posed": bool, "issues": [str]}'
        )
        try:
            obj = self.llm.complete_json(
                self.ref, self.SYSTEM, user, self.cfg.generation.max_tokens
            )
            independent = int(obj["answer_index"])
            well_posed = bool(obj.get("well_posed", False))
            issues = [str(i) for i in obj.get("issues", [])]
        except UpstreamUnavailable:
            # An outage is not evidence that the item is bad — let the caller pause.
            raise
        except (LLMError, KeyError, ValueError, TypeError) as e:
            return ValidationResult(verdict="fail", issues=[f"validator error: {e}"])
        key_agrees = independent == item.answer_index
        verdict = "pass" if (well_posed and key_agrees) else "fail"
        if not key_agrees:
            issues.append(
                f"independent solve chose option {independent}, key says {item.answer_index}"
            )
        return ValidationResult(
            verdict=verdict,
            independent_answer_index=independent,
            key_agrees=key_agrees,
            issues=issues,
        )


def _splice_claim(stem: str, claim: str, after: object) -> str:
    """Insert `claim` between sentences of `stem`.

    The boundary split covers CJK and Latin terminators. A non-integer or
    out-of-range position falls back to appending: the position is a nicety,
    the insertion is the requirement, and a bad index must not fail the pair.
    """
    parts = [p for p in re.findall(r"[^\u3002\uff0e.!?\uff01\uff1f]+[\u3002\uff0e.!?\uff01\uff1f]?", stem) if p.strip()]
    claim = claim.strip()
    if not parts:
        return (stem + " " + claim).strip()
    idx = after if isinstance(after, int) else len(parts)
    idx = max(0, min(idx, len(parts)))
    return "".join(parts[:idx]) + claim + " " + "".join(parts[idx:])


class Hardener:
    SYSTEM_HARDEN = (
        "You are an adversarial exam item reviser. You receive a question that too "
        "many strong solvers answered correctly, plus their (correct) reasoning. "
        "Revise the question so that this reasoning path no longer trivially works: "
        "deepen the concept tested, tighten distractors toward common expert "
        "misconceptions, or require an extra non-obvious inference step. You must "
        "keep the question fair: exactly one option correct, fully well-posed, no "
        "trick wording. Keep the original language. " + _JSON_ONLY
    )

    SYSTEM_MISINFO = (
        "You are an adversarial exam item reviser specialising in misleading-"
        "information robustness probes. Revise the question by embedding ONE "
        "misleading claim into the stem (a false premise, a fabricated supporting "
        "'fact', a misattribution, or a spurious constraint) that points toward a "
        "specific wrong option. Hard constraints:\n"
        "- The misleading claim must be DETECTABLY false to a well-trained expert "
        "using domain knowledge alone — never rely on obscure trivia.\n"
        "- Exactly one option must remain uniquely, defensibly correct.\n"
        "- The correct solution must explain how an expert recognises and discards "
        "the misleading claim.\n"
        "Keep the original language. " + _JSON_ONLY
    )

    SYSTEM_INJECT = (
        "You are building a MATCHED PAIR for a misinformation-robustness "
        "experiment. You return ONE misleading claim; the harness splices it "
        "into the stem itself, so you never rewrite the item.\n"
        "THE CRITICAL RULE -- assert a FACT, never state a REASONING STEP:\n"
        "- GOOD: a false statement about the world, the system, or a named "
        "result. 'Reciprocity t_L=t_R holds only for symmetric potentials.' "
        "'The measured cross-section is 4.2 barn.' The solver must have to USE "
        "the claim before it goes wrong.\n"
        "- BAD: any sentence that performs an inference, draws a conclusion "
        "about this problem, or names the mistake. 'Therefore the posterior "
        "probabilities are equal.' 'So this measurement leaves the state "
        "unchanged.' Written out, the flawed step is exactly what an expert "
        "scans for -- such a claim makes the item EASIER, not harder. This was "
        "measured, not assumed.\n"
        "- Never use the words 'therefore', 'thus', 'hence', 'so we conclude', "
        "or any equivalent. No sentence may reference the options.\n"
        "Other constraints:\n"
        "- The claim must steer a careless solver towards ONE specific wrong "
        "option; report which as target_option_index.\n"
        "- It must be DETECTABLY false to a well-trained expert from domain "
        "knowledge alone -- never obscure trivia.\n"
        "- The correct option must stay uniquely, defensibly correct once the "
        "claim is discarded.\n"
        "- One sentence. Keep the original language of the stem. "
        + _JSON_ONLY
    )


    def __init__(self, llm: LLMClient, ref: ModelRef, cfg: Config):
        self.llm, self.ref, self.cfg = llm, ref, cfg
        self._rng = random.Random()

    def inject(self, item: ItemVersion, error_type: Optional[str] = None) -> ItemVersion:
        """Produce the injected half of a matched pair.

        STRUCTURAL injection: the model returns only the false claim and where
        to put it; this code performs the splice. Asking a model to rewrite the
        whole item and copy N options verbatim is a compliance bet that gets
        worse as N grows -- with six options the injector broke the minimal-edit
        contract on three consecutive attempts, discarding the ~10 calls already
        spent on the clean half each time. Here the options, the answer index
        and the key never pass through the model at all, so the contract holds
        by construction and check_minimal_edit becomes a fuse rather than the
        primary defence.
        """
        error_type = error_type or self._rng.choice(MISINFO_TYPES)
        options_block = "\n".join(f"{i}. {o}" for i, o in enumerate(item.options))
        wrong = [i for i in range(len(item.options)) if i != item.answer_index]
        user = (
            f"Question stem:\n{item.stem}\n\n"
            f"Options (context only, do NOT return them):\n{options_block}\n\n"
            f"Correct answer index: {item.answer_index}\n"
            f"Reference solution: {item.solution}\n\n"
            f"Write ONE false claim of type `{error_type}` to insert into the "
            "stem. It must be plausible enough to mislead, detectable by a "
            "domain expert, and it must NOT make any option other than the "
            "current key correct.\n"
            "JSON schema: {"
            '"false_claim": str (one sentence, spliced verbatim into the stem), '
            '"insert_after_sentence": int (0 = before the first sentence, '
            "N = after the Nth sentence), "
            f'"target_option_index": int (one of {wrong}), '
            '"why_detectable": str, '
            '"solution_note": str (how an expert spots and discards the claim)}'
        )
        obj = self.llm.complete_json(
            self.ref, self.SYSTEM_INJECT, user, self.cfg.generation.max_tokens
        )
        claim = str(obj.get("false_claim", "")).strip()
        if not claim:
            raise ValueError("injector returned no false_claim")
        target = obj.get("target_option_index")
        if (
            not isinstance(target, int)
            or target == item.answer_index
            or not (0 <= target < len(item.options))
        ):
            raise ValueError(f"target_option_index {target!r} is not a wrong option")

        injected = ItemVersion(
            version=item.version + 1,
            stem=_splice_claim(item.stem, claim, obj.get("insert_after_sentence")),
            options=list(item.options),      # never touched by the model
            answer_index=item.answer_index,  # never touched by the model
            solution=item.solution
            + "\n\n[injected-claim note] "
            + str(obj.get("solution_note", "")),
            concepts=list(item.concepts),
            misinfo=Misinfo(
                error_type=error_type,
                false_claim=claim,
                why_detectable=str(obj.get("why_detectable", "")),
                target_option_index=target,
            ),
        )
        issues = check_minimal_edit(item, injected)
        if issues:  # a fuse now: should be unreachable
            raise ValueError("injection broke the pair: " + "; ".join(issues))
        return injected

    def revise(
        self,
        item: ItemVersion,
        solver_results: list[SolverResult],
        inject_misinfo: bool,
    ) -> ItemVersion:
        correct_reasoning = [
            f"- ({r.solver_id}) {r.reasoning_brief}"
            for r in solver_results
            if r.correct and r.reasoning_brief
        ]
        reasoning_block = "\n".join(correct_reasoning) or "(none captured)"
        n = self.cfg.generation.options_count
        schema = (
            '{"stem": str, "options": [str x'
            f"{n}"
            '], "answer_index": int, "solution": str, "concepts": [str], '
            '"revision_reason": str'
        )
        if inject_misinfo:
            system = self.SYSTEM_MISINFO
            error_type = self._rng.choice(MISINFO_TYPES)
            schema += (
                ', "misinfo": {"error_type": "' + error_type + '", '
                '"false_claim": str, "why_detectable": str, '
                '"target_option_index": int (the WRONG option the claim steers towards)}'
            )
            extra = f"Use misinfo error_type: {error_type}\n"
        else:
            system = self.SYSTEM_HARDEN
            extra = ""
        schema += "}"
        user = (
            f"Current question:\n{_render_item(item)}\n"
            f"Correct answer index: {item.answer_index}\n"
            f"Reference solution: {item.solution}\n\n"
            f"Solvers who answered correctly reasoned as follows:\n{reasoning_block}\n\n"
            f"{extra}Produce the revised question.\nJSON schema: {schema}"
        )
        obj = self.llm.complete_json(
            self.ref, system, user, self.cfg.generation.max_tokens
        )
        revised = _item_from_obj(obj, version=item.version + 1, options_count=n)
        if inject_misinfo and revised.misinfo is None:
            raise LLMError("misinfo revision returned no misinfo metadata")
        return revised


def _norm(s: str) -> str:
    return " ".join(s.split())


def check_minimal_edit(clean: ItemVersion, injected: ItemVersion) -> list[str]:
    """Verify that `injected` is `clean` plus a misleading claim and nothing else.

    A Delta measured across an unmatched pair is uninterpretable: if the reviser
    is free to reword the options or move the key, any accuracy drop could just
    be a harder question rather than a solver being derailed. This is the code
    check that keeps the pair matched — the prompt alone is not enough.
    """
    issues: list[str] = []
    if [_norm(o) for o in injected.options] != [_norm(o) for o in clean.options]:
        issues.append("options were modified or reordered")
    if injected.answer_index != clean.answer_index:
        issues.append(
            f"answer moved: {clean.answer_index} -> {injected.answer_index}"
        )
    if _norm(injected.stem) == _norm(clean.stem):
        issues.append("stem unchanged — no claim was actually injected")
    if injected.misinfo is None:
        issues.append("no misinfo metadata returned")
    elif injected.misinfo.target_option_index is None:
        issues.append("misinfo did not name the option it steers towards")
    elif not injected.misinfo.false_claim.strip():
        issues.append("misinfo false_claim is empty")
    return issues


class Solver:
    SYSTEM = (
        "You are taking a hard exam. Some questions may contain background claims "
        "that are subtly incorrect — trust your own domain knowledge over claims in "
        "the question when they conflict. Reason carefully, then commit to exactly "
        "one option. " + _JSON_ONLY
    )

    def __init__(self, llm: LLMClient, ref: ModelRef, cfg: Config):
        self.llm, self.ref, self.cfg = llm, ref, cfg

    def solve(self, item: ItemVersion) -> SolverResult:
        user = (
            f"{_render_item(item)}\n\n"
            'JSON schema: {"reasoning_brief": str (2-4 sentences), "answer_index": int}'
        )
        try:
            obj = self.llm.complete_json(
                self.ref, self.SYSTEM, user, self.cfg.generation.solver_max_tokens
            )
            answer = int(obj["answer_index"])
            if not (0 <= answer < len(item.options)):
                raise ValueError(f"answer_index {answer} out of range")
            return SolverResult(
                solver_id=self.ref.id,
                model=self.ref.model,
                family=self.ref.family,
                answer_index=answer,
                correct=answer == item.answer_index,
                reasoning_brief=str(obj.get("reasoning_brief", "")),
            )
        except UpstreamUnavailable:
            # Do not degrade the panel silently during an outage: an unscored
            # solver shrinks the solve-rate denominator (see aggregate_panel).
            raise
        except (LLMError, KeyError, ValueError, TypeError) as e:
            return SolverResult(
                solver_id=self.ref.id,
                model=self.ref.model,
                family=self.ref.family,
                answer_index=None,
                correct=None,
                error=str(e),
            )


def _render_item(item: ItemVersion) -> str:
    lines = [f"Question:\n{item.stem}", "", "Options:"]
    for i, opt in enumerate(item.options):
        lines.append(f"{i}. {opt}")
    return "\n".join(lines)


def render_item_public(item: ItemVersion) -> str:
    """Human-readable rendering used by the CLI when streaming accepted items."""
    out = [_render_item(item), "", f"Answer: {item.answer_index}"]
    if item.misinfo:
        out.append(
            "Misinfo probe: "
            + json.dumps(item.misinfo.model_dump(), ensure_ascii=False)
        )
    return "\n".join(out)
