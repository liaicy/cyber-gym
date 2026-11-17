"""Offline tests (no API calls). Run: python tests/test_offline.py"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examsociety.agents import Solver, Validator, check_minimal_edit
from examsociety.config import Config, ModelSpec
from examsociety.coverage import (
    accumulation_curve,
    concept_counts,
    near_duplicate_stems,
    richness,
)
from examsociety.llm import (
    LLMClient,
    LLMError,
    ModelRef,
    UpstreamUnavailable,
    extract_json,
)
from examsociety.misinfo import dimension_test, solver_stats
from examsociety.psychometrics import analyse, classical_stats, fit_2pl, response_matrix
from examsociety.schemas import (
    ItemRecord,
    ItemVersion,
    Misinfo,
    PanelRun,
    SolverResult,
    ValidationResult,
)
from examsociety.society import aggregate_panel, classify_panel, panel_quorum


def test_extract_json():
    assert extract_json('noise {"a": 1, "b": {"c": [1, 2]}} trailing') == {
        "a": 1,
        "b": {"c": [1, 2]},
    }
    assert extract_json('```json\n{"x": "va\\"l{ue}"}\n```') == {"x": 'va"l{ue}'}
    assert extract_json("{broken") is None
    assert extract_json("no json here") is None
    # first candidate broken, second valid
    assert extract_json('{bad} then {"ok": true}') == {"ok": True}


def test_2pl_recovery():
    rng = np.random.default_rng(7)
    n_items, n_solvers = 60, 40
    true_a = np.exp(rng.normal(0, 0.3, n_items))
    true_b = rng.normal(0, 1.2, n_items)
    true_theta = rng.normal(0, 1, n_solvers)
    z = true_a[:, None] * (true_theta[None, :] - true_b[:, None])
    P = 1 / (1 + np.exp(-z))
    X = (rng.random((n_items, n_solvers)) < P).astype(float)

    a, b, theta = fit_2pl(X)
    r_b = np.corrcoef(b, true_b)[0, 1]
    r_t = np.corrcoef(theta, true_theta)[0, 1]
    assert r_b > 0.85, f"difficulty recovery too weak: r={r_b:.3f}"
    assert r_t > 0.85, f"ability recovery too weak: r={r_t:.3f}"

    p, pb = classical_stats(X)
    assert p.shape == (n_items,)
    # harder items (higher b) should have lower p-values
    assert np.corrcoef(p, true_b)[0, 1] < -0.7


def test_record_roundtrip():
    item = ItemVersion(
        version=0, stem="s", options=["a", "b", "c", "d"], answer_index=2, solution="because"
    )
    rec = ItemRecord(
        domain="d",
        subtopic="t",
        versions=[item],
        panel_runs=[
            PanelRun(
                version=0,
                results=[
                    SolverResult(solver_id="x", model="m", answer_index=2, correct=True)
                ],
                solve_rate=1.0,
            )
        ],
        final_status="too_easy",
        final_version=0,
    )
    restored = ItemRecord.model_validate(rec.model_dump())
    assert restored.final_item().answer_index == 2


def _result(sid, correct):
    """A scored result, or an errored one when correct is None."""
    return SolverResult(
        solver_id=sid,
        model="m",
        answer_index=0 if correct else (1 if correct is False else None),
        correct=correct,
        error=None if correct is not None else "boom",
    )


def test_panel_classification():
    """A failed panel must never be mistaken for a hard item."""
    assert panel_quorum(3) == 2 and panel_quorum(2) == 2 and panel_quorum(1) == 1

    # every solver errored: solve_rate is 0.0 only because there is no data
    dead = aggregate_panel(0, [_result(s, None) for s in "abc"])
    assert dead.solve_rate == 0.0 and dead.n_scored == 0
    assert classify_panel(dead, 3, 0.5) == "panel_error"

    # one survivor answers wrong -> 0/1, must not become suspect_too_hard
    lone_wrong = aggregate_panel(0, [_result("a", False), _result("b", None), _result("c", None)])
    assert lone_wrong.solve_rate == 0.0
    assert classify_panel(lone_wrong, 3, 0.5) == "panel_error"

    # one survivor answers right -> 1/1, must not trigger the hardening loop
    lone_right = aggregate_panel(0, [_result("a", True), _result("b", None), _result("c", None)])
    assert lone_right.solve_rate == 1.0
    assert classify_panel(lone_right, 3, 0.5) == "panel_error"

    # genuine outcomes still classify as before
    quorum_ok = aggregate_panel(0, [_result("a", False), _result("b", False), _result("c", None)])
    assert classify_panel(quorum_ok, 3, 0.5) == "suspect_too_hard"
    half = aggregate_panel(0, [_result("a", True), _result("b", False)])
    assert classify_panel(half, 2, 0.5) == "accepted"
    easy = aggregate_panel(0, [_result(s, True) for s in "abc"])
    assert classify_panel(easy, 3, 0.5) == "too_easy"


def test_unscored_items_excluded_from_psychometrics():
    """An all-errors panel must not reach the IRT fit with a default a=1.0."""

    def rec(rid, corrects):
        return ItemRecord(
            id=rid, domain="d", subtopic="t",
            versions=[ItemVersion(version=0, stem="s", options=["a", "b"], answer_index=0, solution="")],
            panel_runs=[aggregate_panel(0, [_result(f"s{i}", c) for i, c in enumerate(corrects)])],
            final_status="accepted", final_version=0,
        )

    live = [rec(f"ok{i}", [True, False, i % 2 == 0]) for i in range(4)]
    dead = rec("dead", [None, None, None])
    X, item_ids, _ = response_matrix(live + [dead])
    assert "dead" not in item_ids, "all-NaN item leaked into the response matrix"
    assert X.shape[0] == len(live)
    assert not any(s.item_id == "dead" for s in analyse(live + [dead])[0])


class _StubLLM:
    def __init__(self, exc):
        self.exc = exc

    def complete_json(self, *args, **kwargs):
        raise self.exc


REF = ModelRef(id="x", model="m", family="test")


def test_outage_is_not_swallowed():
    """Roles degrade on request-level errors but must let an outage through."""
    cfg = Config()
    item = ItemVersion(version=0, stem="s", options=["a", "b"], answer_index=0, solution="")

    assert Validator(_StubLLM(LLMError("bad json")), REF, cfg).validate(item).verdict == "fail"
    assert Solver(_StubLLM(LLMError("bad json")), REF, cfg).solve(item).correct is None

    for label, call in [
        ("validator", lambda: Validator(_StubLLM(UpstreamUnavailable("503")), REF, cfg).validate(item)),
        ("solver", lambda: Solver(_StubLLM(UpstreamUnavailable("503")), REF, cfg).solve(item)),
    ]:
        try:
            call()
        except UpstreamUnavailable:
            continue
        raise AssertionError(f"{label} swallowed UpstreamUnavailable")


def test_outage_retry_policy():
    """Outages retry indefinitely; request-level errors are bounded."""
    from examsociety.society import ExamSociety

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise UpstreamUnavailable("503")
        return "done"

    # unbound call: the helper does not touch self
    assert ExamSociety._with_outage_retry(None, flaky, "t", wait_s=0) == "done"
    assert len(calls) == 3

    def always_bad():
        calls.append(1)
        raise LLMError("unparseable")

    calls.clear()
    try:
        ExamSociety._with_outage_retry(None, always_bad, "t", wait_s=0, content_retries=2)
    except LLMError:
        pass
    else:
        raise AssertionError("content-level failure retried forever")
    assert len(calls) == 3, f"expected 1 try + 2 retries, got {len(calls)}"


def test_proposer_rotation_is_balanced():
    """Rotating subtopic and proposer with separate counters would confound them."""
    import tempfile

    from examsociety.society import ExamSociety
    from examsociety.store import RunStore

    cfg = Config()
    cfg.roles.proposers = [ModelSpec(id=f"p{i}", model=f"p{i}") for i in range(4)]
    cfg.roles.hardeners = [ModelSpec(id=f"h{i}", model=f"h{i}") for i in range(2)]
    soc = ExamSociety(cfg, RunStore(tempfile.mkdtemp()))
    soc._n_subtopics = 12  # gcd(12, 4) = 4 -> the naive `%` scheme breaks here

    seen: dict[str, set[int]] = {}
    for i in range(48):
        sub, prop, _ = soc.assign(i)
        seen.setdefault(prop.ref.id, set()).add(sub)
    assert len(seen) == 4
    for pid, subs in seen.items():
        assert subs == set(range(12)), f"{pid} was locked onto subtopics {sorted(subs)}"

    # injector cycle must not be tied to the author cycle
    combos = {(soc.assign(i)[1].ref.id, soc.assign(i)[2].ref.id) for i in range(48)}
    assert len(combos) == 8, f"proposer/hardener pairing degenerated: {sorted(combos)}"


def test_cross_family_validator_preference():
    cfg = Config()
    cfg.roles.proposers = [ModelSpec(id="claude-p", model="claude-opus-4-8")]
    cfg.roles.validators = [
        ModelSpec(id="claude-v", model="claude-opus-4-8"),
        ModelSpec(id="gpt-v", model="gpt-x", api="openai", base_url="http://x/v1"),
    ]
    import tempfile

    from examsociety.society import ExamSociety
    from examsociety.store import RunStore

    soc = ExamSociety(cfg, RunStore(tempfile.mkdtemp()))
    author = soc.proposers[0]
    assert author.ref.family == "claude"
    picks = {soc.pick_validators(author.ref.family, 1)[0].ref.id for _ in range(30)}
    assert picks == {"gpt-v"}, f"same-family validator was chosen: {picks}"

    cfg.loop.cross_family_validation = False
    soc2 = ExamSociety(cfg, RunStore(tempfile.mkdtemp()))
    assert {
        soc2.pick_validators(soc2.proposers[0].ref.family, 1)[0].ref.id
        for _ in range(30)
    } == {"claude-v", "gpt-v"}


def test_validator_consensus_pool():
    """Consensus only means anything across families."""
    import tempfile

    from examsociety.society import ExamSociety, consensus_passed
    from examsociety.store import RunStore

    cfg = Config()
    cfg.roles.proposers = [ModelSpec(id="p", model="claude-opus-4-8")]
    cfg.roles.validators = [
        ModelSpec(id="v-claude-a", model="claude-opus-4-8"),
        ModelSpec(id="v-claude-b", model="claude-opus-4-7"),
        ModelSpec(id="v-gpt", model="gpt-x", api="openai", base_url="http://x/v1"),
        ModelSpec(id="v-gemini", model="gemini-x", api="openai", base_url="http://x/v1"),
    ]
    soc = ExamSociety(cfg, RunStore(tempfile.mkdtemp()))

    for _ in range(30):
        picked = soc.pick_validators("claude", 2)
        fams = [v.ref.family for v in picked]
        assert len(picked) == 2
        assert len(set(fams)) == 2, f"same family drawn twice: {fams}"
        assert "claude" not in fams, f"author family drawn: {fams}"

    # only two families exist besides the author -> asking for 3 must still
    # return 3 validators rather than silently weakening the gate
    assert len(soc.pick_validators("claude", 3)) == 3

    ok = [ValidationResult(verdict="pass"), ValidationResult(verdict="pass")]
    mixed = [ValidationResult(verdict="pass"), ValidationResult(verdict="fail")]
    assert consensus_passed(ok) and not consensus_passed(mixed)
    assert not consensus_passed([])


def test_validation_agreement_lower_bound():
    """A dissent is an item single-validator gating would have accepted."""
    from examsociety.report import validation_agreement

    def rec(rounds):
        """rounds: list of (version, [verdicts]) with family alternating."""
        vals = []
        for version, verdicts in rounds:
            for i, verdict in enumerate(verdicts):
                vals.append(
                    ValidationResult(
                        verdict=verdict, version=version,
                        validator_id=f"v{i}", validator_family=f"f{i}",
                        cross_family=i > 0,
                    )
                )
        return ItemRecord(domain="d", subtopic="t", validations=vals)

    stats = validation_agreement([
        rec([(0, ["pass", "pass"])]),          # both agree
        rec([(0, ["pass", "fail"])]),          # overturned
        rec([(0, ["fail"])]),                  # first already failed: no 2nd opinion
        rec([(0, ["pass", "fail"]), (1, ["pass", "pass"])]),  # per-version rounds
    ])
    assert stats["n_second_opinions"] == 4
    assert stats["n_overturned"] == 2
    assert stats["key_error_lower_bound"] == 0.5
    assert stats["n_cross_family"] == 4

    assert validation_agreement([rec([(0, ["pass"])])])["key_error_lower_bound"] is None


def test_acceptance_band():
    """A ceiling alone drives the bank to the guessing floor; a band does not."""
    def panel(rate, n=4):
        hits = round(rate * n)
        return aggregate_panel(0, [_result(f"s{i}", i < hits) for i in range(n)])

    # ceiling-only (old behaviour) still reachable
    assert classify_panel(panel(0.25), 4, 0.5, 0.0) == "accepted"
    # with a floor, the same near-chance item is rejected as too hard
    assert classify_panel(panel(0.25), 4, 0.5, 0.3) == "too_hard"
    assert classify_panel(panel(0.5), 4, 0.5, 0.3) == "accepted"
    assert classify_panel(panel(0.75), 4, 0.5, 0.3) == "too_easy"
    # zero survives as its own status, not as too_hard
    assert classify_panel(panel(0.0), 4, 0.5, 0.3) == "suspect_too_hard"


def test_pool_runs_and_consistency():
    """Repeated measurement is the only thing separating hard from random."""
    from examsociety.psychometrics import response_consistency
    from examsociety.society import pool_runs

    a = aggregate_panel(0, [_result("s1", True), _result("s2", True)])
    b = aggregate_panel(0, [_result("s1", True), _result("s2", False)])
    pooled = pool_runs([a, b])
    assert pooled.n_scored == 4 and pooled.solve_rate == 0.75
    assert pool_runs([a]) is a

    stable = ItemRecord(domain="d", subtopic="t", panel_runs=[a, a],
                        final_status="accepted", final_version=0)
    flaky = ItemRecord(domain="d", subtopic="t", panel_runs=[a, b],
                       final_status="accepted", final_version=0)
    assert response_consistency([stable])["consistency"] == 1.0
    assert response_consistency([flaky])["consistency"] == 0.5
    # a single attempt cannot say anything about stability
    assert response_consistency([
        ItemRecord(domain="d", subtopic="t", panel_runs=[a],
                   final_status="accepted", final_version=0)
    ])["consistency"] is None


def test_revision_drift():
    """Hardening by obscurity flattens the panel; hardening by depth does not."""
    from examsociety.psychometrics import revision_drift

    def rec(rid, v0_rates, v1_rates):
        def run(v, rates):
            return aggregate_panel(v, [
                SolverResult(solver_id=f"s{i}", model="m", answer_index=0 if ok else 1,
                             correct=ok)
                for i, ok in enumerate(rates)
            ])
        return ItemRecord(
            id=rid, domain="d", subtopic="t",
            versions=[ItemVersion(version=v, stem="s", options=list("wx"),
                                  answer_index=0, solution="") for v in (0, 1)],
            panel_runs=[run(0, v0_rates), run(1, v1_rates)],
            final_status="accepted", final_version=1,
        )

    # depth: the solvers who could do v0 are the ones who still can
    depth = [rec(f"d{k}", [True, True, True, False], [True, True, False, False])
             for k in range(6)]
    d = revision_drift(depth)
    assert d["mean_d_difficulty"] > 0, "items should have got harder"
    assert d["carryover"] > 0.5, d
    assert d["verdict"].startswith("difficulty is tracking"), d["verdict"]

    # obscurity: v1 success is spread evenly over solvers regardless of v0,
    # so knowing who could do v0 predicts nothing
    v1_patterns = [
        [True, True, False, False], [True, False, True, False],
        [False, True, True, False], [True, False, False, True],
        [False, True, False, True], [False, False, True, True],
    ]
    obscure = [rec(f"o{k}", [True, True, True, False], pat)
               for k, pat in enumerate(v1_patterns)]
    d = revision_drift(obscure)
    assert d["mean_d_difficulty"] > 0, "items should have got harder"
    assert abs(d["carryover"]) < 0.05, f"carryover should vanish: {d['carryover']}"
    assert "OBSCURITY DRIFT" in d["verdict"], d["verdict"]

    assert revision_drift([])["n_steps"] == 0


def test_concept_duplicates_catch_rewrites():
    """Stem similarity structurally cannot see a reworded same-concept item."""
    from examsociety.coverage import concept_duplicates, near_duplicate_stems

    def rec(rid, stem, concepts):
        r = _pair_record(rid, {"s": (0, 0)}, concepts=concepts)
        r.versions[0].stem = stem
        r.versions[1].stem = stem
        return r

    reworded = [
        rec("a", "一个电子处于三维各向同性谐振子势中，第一激发态简并度是多少",
            ["谐振子简并度", "量子数计数"]),
        rec("b", "考虑各向同性三维简谐势阱里的粒子，求其最低激发能级的态数",
            ["谐振子简并度", "量子数计数"]),
    ]
    assert near_duplicate_stems(reworded)[0] == 0, "wording differs, so stems must not match"
    assert concept_duplicates(reworded)[0] == 1, "same concepts must be caught"


def test_quota_exhausted_is_terminal():
    """A paid-quota rejection must stop the run, not retry like a content error."""
    from examsociety.llm import LLMError, QuotaExhausted, _is_quota_error

    # Kimi's real 403 body, plus other providers' phrasings
    assert _is_quota_error(403, '{"type":"access_terminated_error","message":"usage limit"}')
    assert _is_quota_error(402, "anything")  # 402 is always payment
    assert _is_quota_error(400, "Insufficient Balance")   # deepseek
    assert _is_quota_error(429, "You exceeded your current quota")  # openai-style
    # a plain overload / bad-json 4xx must NOT be misread as quota
    assert not _is_quota_error(400, "invalid request: bad json")
    assert not _is_quota_error(503, "engine overloaded")
    # A SUCCESSFUL response must never be read as a quota failure: a 200 body
    # contains model prose and a "usage" block, and QuotaExhausted is terminal.
    assert not _is_quota_error(200, '{"usage":{"total_tokens":5},"choices":[...]}')
    assert not _is_quota_error(200, "the billing department computes a quota")
    # real deepseek/kimi phrasings still caught
    assert _is_quota_error(403, "usage limit for this billing cycle")


    # it must be terminal: not catchable as an LLMError, so roles can't degrade
    # it into an unscored answer and the outage loop can't retry it forever
    assert not issubclass(QuotaExhausted, LLMError)

    # run() must handle it, or a spent balance kills the process with a raw
    # traceback instead of a resumable message. An earlier patch silently
    # failed to land and this is exactly how that surfaced.
    import inspect

    from examsociety.society import ExamSociety

    handlers = [
        l.strip()
        for l in inspect.getsource(ExamSociety.run).splitlines()
        if l.strip().startswith("except")
    ]
    assert any("QuotaExhausted" in h for h in handlers), handlers
    assert any("BudgetExhausted" in h for h in handlers), handlers



def test_harden_paired_never_injects_while_hardening():
    """The hardening loop's output becomes the CLEAN half of the pair.

    If misinfo leaks in during hardening, the clean half already contains a
    false claim and every Delta computed from that pair is silently meaningless.
    Nothing downstream can detect it, so lock the invariant at the source.
    """
    import inspect

    from examsociety.society import ExamSociety

    src = inspect.getsource(ExamSociety.generate_item)
    guard_start = src.index("inject = (")
    guard = src[guard_start : guard_start + 220]
    assert "harden-paired" in guard, guard
    assert "False" in guard, guard
    # the pairing call for this pipeline must come AFTER the revision loop
    assert src.index("hardened to solve_rate") > guard_start



def test_structural_injection_cannot_break_the_pair():
    """Options must reach the injected half untouched, whatever the model says.

    The model used to rewrite the whole item and copy the options verbatim; with
    six options it violated that contract on three consecutive attempts. Now it
    only returns a claim and a position, so the pair holds by construction --
    even when the model hands back garbage for the position.
    """
    from examsociety.agents import Hardener, _splice_claim, check_minimal_edit

    clean = ItemVersion(
        version=0, stem="First sentence. Second sentence. Third sentence.",
        options=[f"option {i}" for i in range(6)], answer_index=2,
        solution="because", concepts=["c1"],
    )

    class _Stub:
        def __init__(self, obj):
            self.obj = obj

        def complete_json(self, *a, **k):
            return self.obj

    cfg = Config()
    good = {"false_claim": "This claim is false.", "insert_after_sentence": 1,
            "target_option_index": 4, "why_detectable": "w", "solution_note": "n"}
    h = Hardener(_Stub(good), REF, cfg)
    inj = h.inject(clean, error_type="false_premise")
    assert inj.options == clean.options, "options must be carried over verbatim"
    assert inj.answer_index == clean.answer_index
    assert "This claim is false." in inj.stem and inj.stem != clean.stem
    assert not check_minimal_edit(clean, inj)
    assert inj.misinfo.target_option_index == 4

    # a nonsense position must still yield a valid pair, not an exception
    weird = dict(good, insert_after_sentence="not-an-int")
    inj2 = Hardener(_Stub(weird), REF, cfg).inject(clean, error_type="false_premise")
    assert not check_minimal_edit(clean, inj2)

    # but a target pointing at the KEY is a real error and must be rejected
    bad = dict(good, target_option_index=clean.answer_index)
    try:
        Hardener(_Stub(bad), REF, cfg).inject(clean, error_type="false_premise")
    except ValueError:
        pass
    else:
        raise AssertionError("target_option_index == key was accepted")

    # splice fallbacks
    assert _splice_claim("solo", "X", 99).startswith("solo")
    assert _splice_claim("", "X", 0) == "X"


def test_store_survives_concurrent_appends():
    """Items are generated concurrently, so appends collide.

    Without a lock the JSON lines interleave and the file stops parsing --
    silently, and only for the rows written during the collision, which is the
    worst way to lose an overnight run.
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from examsociety.store import RunStore

    with tempfile.TemporaryDirectory() as d:
        store = RunStore(d)

        def write(i):
            store.append(ItemRecord(
                id=f"r{i:04d}", domain="d", subtopic="t" * 200,
                versions=[ItemVersion(version=0, stem="s" * 500, options=["a", "b"],
                                      answer_index=0, solution="x" * 500)],
                final_status="accepted", final_version=0,
            ))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(write, range(120)))

        rows = store.load_all()
        assert len(rows) == 120, f"lost or merged rows: {len(rows)}"
        assert len({r.id for r in rows}) == 120


def test_budget_cap():
    from examsociety.llm import BudgetExhausted, LLMClient

    client = LLMClient(max_calls=3)
    client._backends["openai"] = _openai_backend(
        200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    )
    ref = ModelRef(id="gw", model="m", api="openai", base_url="http://gw.test/v1")
    for _ in range(3):
        assert client.complete(ref, "s", "u", 10) == "hi"
    try:
        client.complete(ref, "s", "u", 10)
    except BudgetExhausted:
        pass
    else:
        raise AssertionError("budget was not enforced")
    # must not be catchable as an LLM failure, or a role would swallow it
    assert not issubclass(BudgetExhausted, LLMError)


def test_paired_records_reach_irt():
    """A paired item's difficulty is the CLEAN half's, not the perturbed one's."""
    from examsociety.psychometrics import _scored_run

    r = _pair_record("p0", {"s1": (0, 1), "s2": (0, 1), "s3": (0, 0)})
    run = _scored_run(r)
    assert run is not None and run.version == 0, "paired IRT must use the clean half"
    X, ids, _ = response_matrix([r])
    assert ids == ["p0"] and X.shape == (1, 3)
    assert X.sum() == 3.0, "clean half had all three correct"


def test_extra_body_and_family_inference():
    """Vendor knobs reach the wire; an unguessable family must not stay silent."""
    from examsociety.config import Config, ModelSpec, warn_on_guessed_families

    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"},
                               "finish_reason": "stop"}]},
        )

    client = LLMClient()
    client._backends["openai"]._clients["http://t/v1"] = httpx.Client(
        base_url="http://t/v1", transport=httpx.MockTransport(handler)
    )
    ref = ModelRef(id="k3", model="k3-256k", api="openai", base_url="http://t/v1",
                   extra_body={"reasoning_effort": "low"})
    client.complete(ref, "s", "u", 100)
    assert seen["reasoning_effort"] == "low", seen
    assert seen["max_tokens"] == 100, "extra_body must not clobber the token param"

    # a dict field on a frozen dataclass must not break hashing
    assert hash(ref) == hash(ref)

    # `k3-256k` carries no vendor token, so the family falls back to the
    # PROTOCOL name — which would silently split one vendor across two families
    unnamed = ModelSpec(id="k3", model="k3-256k", api="openai")
    assert unnamed.resolved_family == "openai"
    assert unnamed.family_is_guessed_from_protocol
    assert warn_on_guessed_families(
        Config.model_validate({"solvers": [unnamed.model_dump()]})
    ) == ["k3"]

    # explicit family wins, and a recognisable name never warns
    pinned = ModelSpec(id="k3", model="k3-256k", api="openai", family="kimi")
    assert pinned.resolved_family == "kimi"
    assert not pinned.family_is_guessed_from_protocol
    assert not ModelSpec(id="k", model="kimi-for-coding",
                         api="openai").family_is_guessed_from_protocol


def test_mantel_haenszel():
    from examsociety.leakage import mantel_haenszel

    # two strata, hand-computed: OR = (10*10/30 + 8*6/20) / (5*5/30 + 2*4/20)
    r = mantel_haenszel([(10, 5, 5, 10), (8, 2, 4, 6)], "t")
    assert abs(r.odds_ratio - 5.733333 / 1.233333) < 1e-6
    assert r.n_own == 25 and r.n_other == 25 and r.n_strata == 2
    assert r.delta < 0  # OR > 1 -> own-group favoured

    # a stratum where everyone succeeded carries no information
    degenerate = mantel_haenszel([(5, 0, 5, 0)], "t")
    assert degenerate.category == "insufficient" and degenerate.odds_ratio is None


def _leaky_records(author_boost, injector_shield, n=500, seed=1):
    """Paired records with a KNOWN own-family advantage planted in them."""
    import random

    from examsociety.society import aggregate_panel

    fams = ["claude", "gpt", "gemini", "deepseek"]
    panel = list(zip([f"{f}-s" for f in fams], fams, [0.75, 0.70, 0.60, 0.55]))
    rng = random.Random(seed)
    recs = []
    for k in range(n):
        author, injector = fams[k % 4], fams[(k // 4) % 4]
        base = rng.uniform(-0.15, 0.15)

        def run(version, injected):
            res = []
            for sid, fam, acc in panel:
                p = acc + base + (author_boost if fam == author else 0.0)
                ok = rng.random() < max(0.05, min(0.95, p))
                if injected and ok:
                    frag = 0.45 - (injector_shield if fam == injector else 0.0)
                    ok = not (rng.random() < frag)
                res.append(SolverResult(solver_id=sid, model=f"{fam}-x", family=fam,
                                        answer_index=0 if ok else 1, correct=ok))
            return aggregate_panel(version, res)

        def version(v, misinfo, author_family):
            return ItemVersion(
                version=v, stem=f"q{k}", options=list("wxyz"), answer_index=0,
                solution="s", concepts=["c"],
                author_id=author_family, author_family=author_family,
                misinfo=Misinfo(error_type="false_premise", false_claim="f",
                                why_detectable="w", target_option_index=1) if misinfo else None,
            )

        recs.append(ItemRecord(
            domain="D", subtopic="S", pipeline="paired",
            versions=[version(0, False, author), version(1, True, injector)],
            panel_runs=[run(0, False), run(0, False), run(1, True)],
            final_status="paired", final_version=1,
        ))
    return recs


def test_leakage_detection():
    """The checks must fire on a planted leak and stay quiet without one."""
    from examsociety.leakage import author_advantage, injector_interaction

    def pooled(results):
        return next(r for r in results if r.label == "POOLED")

    leaky = _leaky_records(author_boost=0.12, injector_shield=0.20)
    a, i = pooled(author_advantage(leaky)), pooled(injector_interaction(leaky))
    assert a.odds_ratio > 1.2 and a.significant, f"author leak missed: {a}"
    assert i.odds_ratio < 0.8 and i.significant, f"injector leak missed: {i}"
    # ETS categories need BOTH p < 0.05 and |delta| >= 1.0, so a small-but-real
    # effect can be significant and still classify A. At 500 pairs both clear it.
    assert a.category.startswith(("B", "C")), f"author effect graded {a.category}"
    assert i.category.startswith(("B", "C")), f"injector effect graded {i.category}"

    clean = _leaky_records(author_boost=0.0, injector_shield=0.0, seed=9)
    a0, i0 = pooled(author_advantage(clean)), pooled(injector_interaction(clean))
    assert a0.category == "A negligible", f"false positive on author: {a0}"
    assert i0.category == "A negligible", f"false positive on injector: {i0}"

    # the pooled row exists precisely because per-solver tests are underpowered
    per_solver = [r for r in injector_interaction(leaky) if r.label != "POOLED"]
    assert len(per_solver) == 4
    assert all(r.odds_ratio < 1 for r in per_solver), "effect direction inconsistent"


def test_family_fallback():
    from examsociety.leakage import family_of

    assert family_of(SolverResult(solver_id="a", model="gpt-4o", answer_index=0, correct=True)) == "gpt"
    assert family_of(SolverResult(solver_id="a", model="x", family="custom",
                                  answer_index=0, correct=True)) == "custom"


def _openai_backend(status, body):
    from examsociety.llm import _OpenAICompatBackend

    b = _OpenAICompatBackend()
    b._clients["http://gw.test/v1"] = httpx.Client(
        base_url="http://gw.test/v1",
        transport=httpx.MockTransport(lambda req: httpx.Response(status, json=body)),
    )
    return b


def test_openai_backend_error_mapping():
    """A second backend is only safe if it maps errors the same way the first does."""
    from examsociety.llm import _Retryable

    ref = ModelRef(id="gw", model="m", api="openai", base_url="http://gw.test/v1")

    def body(content, finish):
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    assert _openai_backend(200, body("hi", "stop")).call(ref, "s", "u", 10) == ("hi", "ok")
    assert _openai_backend(200, body("hi", "length")).call(ref, "s", "u", 10)[1] == "truncated"
    assert _openai_backend(200, body("", "content_filter")).call(ref, "s", "u", 10)[1] == "refusal"

    for status in (429, 500, 502, 503):
        try:
            _openai_backend(status, {}).call(ref, "s", "u", 10)
        except _Retryable:
            continue
        raise AssertionError(f"HTTP {status} should be retryable")

    for status in (400, 401, 404):
        try:
            _openai_backend(status, {}).call(ref, "s", "u", 10)
        except _Retryable:
            raise AssertionError(f"HTTP {status} must not retry forever")
        except LLMError:
            continue
        raise AssertionError(f"HTTP {status} should raise LLMError")


def test_openai_outage_reaches_upstream_unavailable():
    """The outage fix must hold on the new backend, not just on Anthropic."""
    import examsociety.llm as L

    ref = ModelRef(id="gw", model="m", api="openai", base_url="http://gw.test/v1")
    client = L.LLMClient()
    client._backends["openai"] = _openai_backend(503, {})
    real_sleep = L.time.sleep
    L.time.sleep = lambda _s: None
    try:
        client.complete(ref, "s", "u", 10)
    except UpstreamUnavailable:
        pass
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected UpstreamUnavailable, got {type(e).__name__}: {e}")
    else:
        raise AssertionError("sustained 503 did not become UpstreamUnavailable")
    finally:
        L.time.sleep = real_sleep


OPTIONS = ["correct", "bait", "other", "another"]


def _version(v, stem, misinfo=False, options=None, answer=0, concepts=()):
    return ItemVersion(
        version=v, stem=stem, options=list(options or OPTIONS), answer_index=answer,
        solution="because", concepts=list(concepts),
        misinfo=Misinfo(
            error_type="false_premise", false_claim="c", why_detectable="w",
            target_option_index=1,
        ) if misinfo else None,
    )


def test_minimal_edit_check():
    """An unmatched pair makes Delta uninterpretable, so it must be rejected."""
    clean = _version(0, "What is X?")
    assert check_minimal_edit(clean, _version(1, "Given the false claim F, what is X?", True)) == []

    def issues(inj):
        return " ".join(check_minimal_edit(clean, inj))

    assert "options" in issues(
        _version(1, "Given F, what is X?", True, options=["correct", "bait", "other", "CHANGED"])
    )
    assert "answer moved" in issues(_version(1, "Given F, what is X?", True, answer=2))
    assert "stem unchanged" in issues(_version(1, "What is X?", True))
    assert "no misinfo metadata" in issues(_version(1, "Given F, what is X?"))
    bare = _version(1, "Given F, what is X?", True)
    bare.misinfo.target_option_index = None
    assert "did not name" in issues(bare)


def _pair_record(rid, per_solver, concepts=()):
    """per_solver: {solver_id: (clean_answer, injected_answer)}; 0 is correct, 1 is the bait."""

    def run(version, pick):
        return aggregate_panel(version, [
            SolverResult(solver_id=sid, model="m", answer_index=pick(a),
                         correct=pick(a) == 0)
            for sid, a in per_solver.items()
        ])

    return ItemRecord(
        id=rid, domain="d", subtopic="t", pipeline="paired",
        versions=[_version(0, f"stem {rid}", concepts=concepts),
                  _version(1, f"stem {rid} plus false claim", True, concepts=concepts)],
        panel_runs=[run(0, lambda a: a[0]), run(1, lambda a: a[1])],
        final_status="paired", final_version=1,
    )


def test_misinfo_pair_stats():
    """Conditional derailment must survive where raw Delta is blind."""
    # strong: derailed on 1 of 10 | fragile: derailed on 5 of 10
    # weak:   only 4 pairs solvable clean, derailed on 2 of those, but happens to
    #         pick up 2 lucky hits elsewhere so its RAW delta is exactly zero.
    specs = []
    for k in range(10):
        specs.append(_pair_record(f"p{k}", {
            "strong":  (0, 1 if k == 0 else 0),
            "fragile": (0, 1 if k < 5 else 0),
            "weak":    (0 if k < 4 else 2,
                        0 if k in (0, 1, 4, 5) else (1 if k in (2, 3) else 2)),
        }))
    st = {s.solver_id: s for s in solver_stats(specs)}

    assert st["strong"].n_pairs == 10 and st["strong"].n_eligible == 10
    assert st["strong"].derail_rate == 0.1 and st["strong"].directed_rate == 0.1
    assert st["fragile"].derail_rate == 0.5 and st["fragile"].directed_rate == 0.5

    w = st["weak"]
    assert w.clean_acc == 0.4 and w.injected_acc == 0.4
    assert w.raw_delta == 0.0, "raw delta should look perfectly robust here"
    assert w.n_eligible == 4 and w.derail_rate == 0.5 and w.directed_rate == 0.5
    # the whole point: raw delta says 'untouched', conditioning says 'half derailed'
    assert w.raw_delta < st["strong"].raw_delta < w.derail_rate


def test_dimension_test():
    """Ability-ordering vs robustness-ordering is the go/no-go signal."""
    from examsociety.misinfo import SolverMisinfoStats

    def s(sid, acc, derail):
        return SolverMisinfoStats(sid, 50, acc, acc - 0.1, 0.1, 40, derail, derail / 2)

    redundant = [s("a", 0.3, 0.60), s("b", 0.5, 0.45), s("c", 0.7, 0.30), s("d", 0.9, 0.15)]
    assert dimension_test(redundant)["rho"] == -1.0
    assert "redundant" in dimension_test(redundant)["verdict"]

    crossed = [s("a", 0.3, 0.30), s("b", 0.5, 0.60), s("c", 0.7, 0.20), s("d", 0.9, 0.50)]
    assert abs(dimension_test(crossed)["rho"]) < 0.6
    assert "separate dimension" in dimension_test(crossed)["verdict"]

    assert dimension_test(redundant[:2])["rho"] is None


def test_coverage_estimators():
    counts = Counter({"a": 1, "b": 1, "c": 2, "d": 3})
    r = richness(counts)
    # Chao1 (bias-corrected) = 4 + f1(f1-1)/(2(f2+1)) = 4 + 2*1/(2*2) = 4.5
    assert (r.s_observed, r.f1, r.f2, r.n_observations) == (4, 2, 1, 7)
    assert abs(r.s_estimated - 4.5) < 1e-9
    assert abs(r.coverage - (1 - 2 / 7)) < 1e-9
    assert not r.saturated

    # a mined-out domain: every concept seen many times, no singletons
    mined = richness(Counter({c: 10 for c in "abcdef"}))
    assert mined.f1 == 0 and mined.coverage == 1.0 and mined.saturated

    recs = [
        _pair_record("r0", {"s": (0, 0)}, concepts=["Bloch theorem", "bloch  theorem "]),
        _pair_record("r1", {"s": (0, 0)}, concepts=["Bloch Theorem.", "Fermi surface"]),
    ]
    assert set(concept_counts(recs)) == {"bloch theorem", "fermi surface"}
    assert concept_counts(recs)["bloch theorem"] == 2, "normalisation should merge variants"
    assert [n for _, n in accumulation_curve(recs)] == [1, 2]


def test_near_duplicate_detection():
    def rec(rid, stem):
        r = _pair_record(rid, {"s": (0, 0)})
        r.versions[0].stem = stem
        r.versions[1].stem = stem + " plus false claim"
        return r

    same = [rec("a", "计算氢原子基态的电离能"), rec("b", "计算氢原子基态的电离能量")]
    n, pairs = near_duplicate_stems(same)
    assert n == 1 and pairs[0][2] > 0.6

    different = [rec("a", "计算氢原子基态的电离能"), rec("b", "推导正则系综的配分函数")]
    assert near_duplicate_stems(different)[0] == 0


if __name__ == "__main__":
    test_extract_json()
    print("extract_json ok")
    test_2pl_recovery()
    print("2PL parameter recovery ok")
    test_record_roundtrip()
    print("record roundtrip ok")
    test_panel_classification()
    print("panel classification ok")
    test_unscored_items_excluded_from_psychometrics()
    print("unscored items excluded from psychometrics ok")
    test_outage_is_not_swallowed()
    print("outage propagation ok")
    test_outage_retry_policy()
    print("outage retry policy ok")
    test_proposer_rotation_is_balanced()
    print("proposer rotation balance ok")
    test_cross_family_validator_preference()
    print("cross-family validator preference ok")
    test_validator_consensus_pool()
    print("validator consensus pool ok")
    test_validation_agreement_lower_bound()
    print("validation agreement lower bound ok")
    test_acceptance_band()
    print("acceptance band ok")
    test_pool_runs_and_consistency()
    print("replicates + consistency ok")
    test_revision_drift()
    print("revision drift ok")
    test_concept_duplicates_catch_rewrites()
    print("concept duplicate detection ok")
    test_budget_cap()
    print("budget cap ok")
    test_store_survives_concurrent_appends()
    print("concurrent store appends ok")
    test_structural_injection_cannot_break_the_pair()
    print("structural injection ok")
    test_harden_paired_never_injects_while_hardening()
    print("harden-paired clean-half invariant ok")
    test_quota_exhausted_is_terminal()
    print("quota exhaustion terminal ok")
    test_paired_records_reach_irt()
    print("paired records in IRT ok")
    test_extra_body_and_family_inference()
    print("extra_body + family inference ok")
    test_mantel_haenszel()
    print("mantel-haenszel ok")
    test_leakage_detection()
    print("leakage detection ok")
    test_family_fallback()
    print("family fallback ok")
    test_openai_backend_error_mapping()
    print("openai backend error mapping ok")
    test_openai_outage_reaches_upstream_unavailable()
    print("openai outage -> UpstreamUnavailable ok")
    test_minimal_edit_check()
    print("minimal-edit pair check ok")
    test_misinfo_pair_stats()
    print("misinfo pair statistics ok")
    test_dimension_test()
    print("dimension test ok")
    test_coverage_estimators()
    print("coverage estimators ok")
    test_near_duplicate_detection()
    print("near-duplicate detection ok")
    print("ALL OFFLINE TESTS PASSED")
