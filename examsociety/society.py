from __future__ import annotations

import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .agents import Hardener, Proposer, Solver, TopicMapper, Validator
from .config import Config
from .llm import (
    BudgetExhausted,
    LLMClient,
    LLMError,
    QuotaExhausted,
    UpstreamUnavailable,
)
from .schemas import (
    ItemRecord,
    ItemVersion,
    PanelRun,
    SolverResult,
    ValidationResult,
)
from .store import RunStore


def _log(msg: str) -> None:
    print(msg, flush=True)


def consensus_passed(results: list[ValidationResult]) -> bool:
    """An item clears the validity gate only if every validator drawn agreed."""
    return bool(results) and all(r.verdict == "pass" for r in results)


# --------------------------------------------------------------- panel scoring
# Pure functions, kept module-level so the classification rules are testable
# without an LLM client.


def panel_quorum(n_solvers: int) -> int:
    """Minimum number of scored solvers for a panel run to mean anything.

    A solve rate computed from one or two survivors is noise, not difficulty:
    it can only take the values 0.0/0.5/1.0 and would drive both the hardening
    loop and the final classification. Require a majority of the panel, and at
    least two solvers whenever the panel has two.
    """
    return min(n_solvers, max(2, (n_solvers + 1) // 2))


def aggregate_panel(version: int, results: list[SolverResult]) -> PanelRun:
    scored = [r for r in results if r.correct is not None]
    solve_rate = (
        sum(1 for r in scored if r.correct) / len(scored) if scored else 0.0
    )
    return PanelRun(
        version=version,
        results=results,
        solve_rate=solve_rate,
        n_scored=len(scored),
    )


def panel_is_scorable(panel: PanelRun, n_solvers: int) -> bool:
    return (panel.n_scored or 0) >= panel_quorum(n_solvers)


def pool_runs(runs: list[PanelRun]) -> PanelRun:
    """Merge replicate runs of one version into a single view for classification.

    Every (solver, attempt) counts as one observation, so an item answered
    correctly on two of four attempts lands at 0.5 rather than being classified
    off whichever single attempt happened to be stored.
    """
    if len(runs) == 1:
        return runs[0]
    merged = [r for run in runs for r in run.results]
    return aggregate_panel(runs[0].version, merged)


def classify_panel(
    panel: PanelRun,
    n_solvers: int,
    target_max_solve_rate: float,
    target_min_solve_rate: float = 0.0,
) -> str:
    """Map a final panel run to an ItemRecord.final_status.

    Acceptance is a BAND, not a ceiling. A ceiling alone drives the whole bank
    towards the guessing floor, where item variance — and therefore
    discrimination — goes to zero, so the bank measures less the "better" it
    scores. target_min_solve_rate = 0.0 restores the old ceiling-only rule.
    """
    if not panel_is_scorable(panel, n_solvers):
        # No usable answers — says nothing about the item's difficulty.
        return "panel_error"
    if panel.solve_rate == 0.0:
        # Nobody solved it; the key was independently verified, but flag for audit.
        return "suspect_too_hard"
    if panel.solve_rate < target_min_solve_rate:
        return "too_hard"
    if panel.solve_rate <= target_max_solve_rate:
        return "accepted"
    return "too_easy"


class ExamSociety:
    """Orchestrates the propose -> validate -> solve-panel -> harden loop."""

    def __init__(self, cfg: Config, store: RunStore, pipeline: str = "society"):
        self.cfg = cfg
        self.store = store
        self.pipeline = pipeline
        llm = LLMClient(max_calls=cfg.loop.max_api_calls)
        self.llm = llm
        self.mapper = TopicMapper(llm, cfg.roles.topic_mapper.ref())
        self.proposers = [Proposer(llm, s.ref(), cfg) for s in cfg.roles.proposers]
        self.validators = [Validator(llm, s.ref(), cfg) for s in cfg.roles.validators]
        self.hardeners = [Hardener(llm, s.ref(), cfg) for s in cfg.roles.hardeners]
        self.solvers = [Solver(llm, s.ref(), cfg) for s in cfg.solvers]
        self._rng = random.Random()

    # ------------------------------------------------------------- rotation

    def assign(self, index: int) -> tuple[int, Proposer, Hardener]:
        """Balanced (subtopic, proposer) assignment plus an injector.

        Cycling subtopic and proposer with separate `% ` counters would lock
        each proposer onto a fixed subset of subtopics whenever the two counts
        share a factor (12 subtopics / 4 proposers => proposer k only ever sees
        subtopics congruent to k mod 4), making family and subtopic effects
        inseparable. Walking the cross product instead is balanced by
        construction. The hardener rotates on its own cycle so that injector
        identity is not tied to author identity either.
        """
        n_sub = max(1, self._n_subtopics)
        n_prop = len(self.proposers)
        cell = index % (n_sub * n_prop)
        return (
            cell % n_sub,
            self.proposers[cell // n_sub],
            self.hardeners[index % len(self.hardeners)],
        )

    def pick_validators(self, author_family: str, n: int = 1) -> list[Validator]:
        """Draw n validators, preferring families other than the author's — and
        other than each other.

        Consensus only means something across families: two same-family
        validators share the blind spot that makes a wrong key look right, so
        they agree for the wrong reason.
        """
        pool = list(self.validators)
        self._rng.shuffle(pool)
        if self.cfg.loop.cross_family_validation:
            # stable sort keeps the shuffle within each group; foreign first
            pool.sort(key=lambda v: v.ref.family == author_family)
        picked, seen_families = [], set()
        for v in pool:
            if v.ref.family not in seen_families:
                picked.append(v)
                seen_families.add(v.ref.family)
            if len(picked) == n:
                return picked
        for v in pool:  # not enough distinct families — top up with repeats
            if v not in picked:
                picked.append(v)
            if len(picked) == n:
                break
        return picked

    def _validate(self, item: ItemVersion, validator: Validator, author_family: str):
        result = validator.validate(item)
        result.version = item.version
        result.validator_id = validator.ref.id
        result.validator_family = validator.ref.family
        result.cross_family = validator.ref.family != author_family
        return result

    def validate_item(
        self, item: ItemVersion, author_family: str
    ) -> list[ValidationResult]:
        """Run the validity gate, optionally as a multi-validator consensus.

        Short-circuits on the first dissent: one independent expert disagreeing
        with the key is already enough to reject, and skipping the rest keeps
        the extra cost proportional to the pass rate rather than to the panel
        size. It also makes the recorded dissents easy to read as "items that
        single-validator gating would have accepted".
        """
        results: list[ValidationResult] = []
        for validator in self.pick_validators(
            author_family, max(1, self.cfg.loop.validator_consensus)
        ):
            result = self._validate(item, validator, author_family)
            results.append(result)
            if result.verdict != "pass":
                break
        return results

    @staticmethod
    def _stamp(item: ItemVersion, author) -> ItemVersion:
        item.author_id = author.ref.id
        item.author_family = author.ref.family
        return item

    # ------------------------------------------------------------------ panel

    def run_panel(self, item: ItemVersion) -> PanelRun:
        with ThreadPoolExecutor(max_workers=max(1, len(self.solvers))) as pool:
            # Solver.solve swallows per-request failures but re-raises
            # UpstreamUnavailable, which propagates out of pool.map here.
            results: list[SolverResult] = list(
                pool.map(lambda s: s.solve(item), self.solvers)
            )
        return aggregate_panel(item.version, results)

    def run_panel_replicated(self, item: ItemVersion) -> list[PanelRun]:
        """Answer the item `solve_replicates` times.

        With a single attempt there is no way to distinguish a hard item from
        one the panel answers at random — and the acceptance rule actively
        selects for the latter, because an unstable item is very likely to land
        under the solve-rate ceiling by chance. Repeated measurement is the only
        thing that separates them; `response_consistency` in psychometrics reads
        these runs back out.
        """
        k = max(1, self.cfg.loop.solve_replicates)
        return [self.run_panel(item) for _ in range(k)]

    # ------------------------------------------------------------- item cycle

    def generate_item(
        self,
        domain: str,
        subtopic: str,
        avoid: list[str],
        proposer: Proposer,
        hardener: Hardener,
        avoid_concepts: Optional[list[str]] = None,
    ) -> ItemRecord:
        record = ItemRecord(domain=domain, subtopic=subtopic, pipeline=self.pipeline)
        loop_cfg = self.cfg.loop
        author_family = proposer.ref.family

        # 1. propose (with retries through the validity gate)
        item = None
        for attempt in range(loop_cfg.proposal_retries + 1):
            try:
                candidate = self._stamp(
                    proposer.propose(domain, subtopic, avoid, avoid_concepts),
                    proposer,
                )
            except UpstreamUnavailable:
                raise  # outage, not a bad proposal — handled by _with_outage_retry
            except (LLMError, ValueError) as e:
                _log(f"    propose attempt {attempt} failed: {e}")
                continue
            record.versions.append(candidate)
            gate = self.validate_item(candidate, author_family)
            record.validations.extend(gate)
            if consensus_passed(gate):
                item = candidate
                break
            _log(
                f"    validity gate rejected v0 ({gate[-1].validator_id}): "
                f"{'; '.join(gate[-1].issues)[:200]}"
            )
            record.versions.pop()  # discard the invalid candidate entirely
        if item is None:
            record.final_status = "rejected_invalid"
            return record

        # 2. panel solve + adversarial revision loop
        runs = self.run_panel_replicated(item)
        record.panel_runs.extend(runs)
        panel = pool_runs(runs)
        if not panel_is_scorable(panel, len(self.solvers)):
            return self._abandon_unscorable(record, panel)
        _log(f"    v{item.version} solve_rate={panel.solve_rate:.2f}")

        if self.pipeline == "paired":
            return self._make_pair(record, item, panel, hardener)

        revisions = 0
        # baseline: single-shot proposer, no adversarial revision loop.
        # harden-paired: run the loop first, then pair on the hardened item.
        max_revisions = 0 if self.pipeline == "baseline" else loop_cfg.max_revisions
        while panel.solve_rate > loop_cfg.target_max_solve_rate and revisions < max_revisions:
            revisions += 1
            # harden-paired MUST NOT inject during hardening: the loop's output
            # becomes the CLEAN half of the pair, and a clean half that already
            # contains a false claim makes the whole Delta uninterpretable.
            inject = (
                False
                if self.pipeline == "harden-paired"
                else self._rng.random() < loop_cfg.misinfo_probability
            )
            try:
                revised = self._stamp(
                    hardener.revise(item, panel.results, inject_misinfo=inject),
                    hardener,
                )
            except UpstreamUnavailable:
                raise  # outage, not a failed revision
            except (LLMError, ValueError) as e:
                _log(f"    revision {revisions} failed: {e}")
                break
            gate = self.validate_item(revised, hardener.ref.family)
            record.validations.extend(gate)
            if not consensus_passed(gate):
                # keep provenance of the failed attempt, stay on current version
                _log(
                    f"    validity gate rejected v{revised.version} "
                    f"({'misinfo' if inject else 'harden'}, {gate[-1].validator_id}): "
                    f"{'; '.join(gate[-1].issues)[:200]}"
                )
                continue
            record.versions.append(revised)
            item = revised
            runs = self.run_panel_replicated(item)
            record.panel_runs.extend(runs)
            panel = pool_runs(runs)
            if not panel_is_scorable(panel, len(self.solvers)):
                return self._abandon_unscorable(record, panel)
            _log(
                f"    v{item.version} ({'misinfo' if inject else 'harden'}) "
                f"solve_rate={panel.solve_rate:.2f}"
            )

        # 3. classify
        record.final_version = item.version
        if self.pipeline == "harden-paired":
            # The loop has pushed the item as far down as it will go; pair on
            # whatever it reached. Pairing an item nobody can solve is pointless
            # (the conditional event "right on the clean half" is empty), so a
            # zero-solve-rate item is abandoned rather than paired.
            if panel.solve_rate == 0.0:
                record.final_version = item.version
                record.final_status = "suspect_too_hard"
                _log("    hardened to 0.00 — no headroom left to measure a drop")
                return record
            _log(f"    hardened to solve_rate={panel.solve_rate:.2f}, pairing now")
            return self._make_pair(record, item, panel, hardener)

        record.final_status = classify_panel(
            panel,
            len(self.solvers),
            loop_cfg.target_max_solve_rate,
            loop_cfg.target_min_solve_rate,
        )
        return record

    def _make_pair(
        self,
        record: ItemRecord,
        clean: ItemVersion,
        clean_panel: PanelRun,
        injector: Hardener,
    ) -> ItemRecord:
        """Paired pipeline: measure the clean item, then the same item plus one
        injected false claim, on the same panel.

        The unit of analysis is the pair, not the item: because the two halves
        differ only by the inserted claim, item difficulty cancels when the two
        panel runs are subtracted. That is what makes Delta interpretable
        without a large panel or a hard item.
        """
        # Control: the same clean item, answered again. The difference between
        # the two clean runs is pure solver instability, which is what has to be
        # subtracted from the clean->injected drop.
        if self.cfg.loop.pair_control_replicate:
            control = self.run_panel(clean)
            record.panel_runs.append(control)
            if not panel_is_scorable(control, len(self.solvers)):
                return self._abandon_unscorable(record, control)
            _log(f"    v{clean.version} control solve_rate={control.solve_rate:.2f}")

        # By this point the pair has already paid for the clean panel and the
        # control run (~7 calls). One more injection attempt costs one call, so
        # retrying a broken injection is strictly cheaper than abandoning the
        # pair — a minimal-edit violation is a sampling failure of the injector,
        # not a property of the item.
        injected = None
        last_err: Exception | None = None
        for attempt in range(1 + self.cfg.loop.injection_retries):
            try:
                injected = self._stamp(injector.inject(clean), injector)
                break
            except UpstreamUnavailable:
                raise
            except (LLMError, ValueError) as e:
                last_err = e
                _log(f"    injection attempt {attempt + 1} failed: {e}")
        if injected is None:
            record.final_version = clean.version
            record.final_status = "pair_failed"
            _log(f"    injection failed after retries: {last_err}")
            return record

        gate = self.validate_item(injected, injector.ref.family)
        record.validations.extend(gate)
        if not consensus_passed(gate):
            record.final_version = clean.version
            record.final_status = "pair_failed"
            _log(
                f"    validity gate rejected injected half ({gate[-1].validator_id}): "
                f"{'; '.join(gate[-1].issues)[:200]}"
            )
            return record

        record.versions.append(injected)
        injected_panel = self.run_panel(injected)
        record.panel_runs.append(injected_panel)
        if not panel_is_scorable(injected_panel, len(self.solvers)):
            return self._abandon_unscorable(record, injected_panel)

        record.final_version = injected.version
        record.final_status = "paired"
        _log(
            f"    pair complete: clean={clean_panel.solve_rate:.2f} -> "
            f"injected={injected_panel.solve_rate:.2f} "
            f"(delta={clean_panel.solve_rate - injected_panel.solve_rate:+.2f}, "
            f"{injected.misinfo.error_type})"
        )
        return record

    def _abandon_unscorable(self, record: ItemRecord, panel: PanelRun) -> ItemRecord:
        """Give up on an item whose panel produced too few usable answers.

        final_version stays None so the record is invisible to final_item() and
        therefore excluded from reports, exports and psychometrics — exactly
        like rejected_invalid. The versions and per-solver errors are kept for
        diagnosis.
        """
        record.final_status = "panel_error"
        _log(
            f"    panel unscorable: only {panel.n_scored}/{len(self.solvers)} "
            f"solvers returned an answer (quorum {panel_quorum(len(self.solvers))})"
        )
        return record

    # ---------------------------------------------------------------- outage

    def _with_outage_retry(
        self, fn, label: str, wait_s: int = 30, content_retries: int = 3
    ):
        """Run fn(); on an upstream outage wait and retry indefinitely.

        The generation loop is meant to run unattended for days — a multi-minute
        upstream outage should pause it, not kill it. Only UpstreamUnavailable
        qualifies: a request-specific LLMError would never be fixed by waiting,
        so it gets a bounded number of retries and is then raised.
        """
        content_failures = 0
        while True:
            try:
                return fn()
            except UpstreamUnavailable as e:
                _log(f"[{label}] upstream unavailable, retrying in {wait_s}s: {str(e)[:160]}")
                time.sleep(wait_s)
            except LLMError as e:
                content_failures += 1
                if content_failures > content_retries:
                    raise
                _log(
                    f"[{label}] failed ({content_failures}/{content_retries}), "
                    f"retrying: {str(e)[:160]}"
                )

    # ------------------------------------------------------------------- run

    def run(self, domain: str, limit: int | None = None) -> None:
        state = self.store.load_state()
        subtopics: list[str] = state.get("subtopics", [])
        if not subtopics or state.get("domain") != domain:
            _log(f"[topic mapper] expanding domain: {domain}")
            subtopics = self._with_outage_retry(
                lambda: self.mapper.map(domain, self.cfg.loop.subtopics_per_domain),
                "topic mapper",
            )
            state = {"domain": domain, "subtopics": subtopics, "cursor": 0}
            self.store.save_state(state)
            _log("[topic mapper] " + " | ".join(subtopics))

        # Per-subtopic history. Keyed by subtopic because the proposer is asked
        # about one subtopic at a time: a global "most recent N stems" window is
        # almost entirely about other subtopics once the pool is cycled, so it
        # constrains nothing.
        history: dict[str, dict] = {}

        def remember(subtopic: str, item: ItemVersion) -> None:
            h = history.setdefault(subtopic, {"stems": [], "concepts": []})
            h["stems"].append(item.stem)
            for c in item.concepts:
                c = c.strip()
                if c and c not in h["concepts"]:
                    h["concepts"].append(c)

        for r in self.store.load_all():
            if (v := r.final_item()) is not None:
                remember(r.subtopic, v)

        self._n_subtopics = len(subtopics)
        cursor = int(state.get("cursor", 0))
        produced = 0
        loop_cfg = self.cfg.loop
        workers = max(1, loop_cfg.item_workers)
        hist_lock = threading.Lock()

        def one_item(index: int, cur: int) -> Optional[ItemRecord]:
            """Generate a single item. Runs on a worker thread."""
            sub_i, proposer, hardener = self.assign(cur)
            subtopic = subtopics[sub_i]
            with hist_lock:
                h = history.get(subtopic, {"stems": [], "concepts": []})
                stems = h["stems"][-loop_cfg.avoid_window :]
                concepts = h["concepts"][-loop_cfg.avoid_concepts_window :]
            _log(
                f"[item {index}] subtopic: {subtopic} "
                f"| proposer: {proposer.ref.id} | hardener: {hardener.ref.id} "
                f"| seen here: {len(stems)} items / {len(concepts)} concepts"
            )
            record = self._with_outage_retry(
                lambda: self.generate_item(
                    domain, subtopic, stems, proposer, hardener,
                    avoid_concepts=concepts,
                ),
                f"item {index}",
            )
            record.pipeline = self.pipeline
            self.store.append(record)
            final = record.final_item()
            if final is not None:
                with hist_lock:
                    remember(subtopic, final)
            _log(
                f"[item {index}] status={record.final_status} "
                f"versions={len(record.versions)} id={record.id}"
            )
            return record

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while limit is None or produced < limit:
                    if (
                        loop_cfg.subtopic_refresh_every
                        and produced
                        and produced % loop_cfg.subtopic_refresh_every == 0
                    ):
                        subtopics = self._grow_subtopics(
                            domain, subtopics, history, state
                        )

                    # Submit one batch, then wait for it. Batching rather than a
                    # rolling window keeps the cursor bookkeeping honest: after a
                    # batch, everything up to `cursor` is genuinely on disk, so a
                    # crash resumes without gaps or repeats.
                    batch = workers if limit is None else min(workers, limit - produced)
                    futures = []
                    for k in range(batch):
                        futures.append(
                            pool.submit(one_item, produced + k + 1, cursor + k)
                        )
                    cursor += batch
                    for fut in futures:
                        fut.result()  # re-raises outage/quota/budget on this thread
                    produced += batch
                    state["cursor"] = cursor
                    self.store.save_state(state)
        except KeyboardInterrupt:
            _log("\n[stopped by user]")
            sys.exit(0)
        except BudgetExhausted as e:
            _log(f"\n[budget] {e}")
            sys.exit(0)
        except QuotaExhausted as e:
            # A spent balance does not heal by waiting, so stop rather than
            # fill the run with guaranteed failures. Everything already
            # written stays intact; re-running the same command resumes from
            # the cursor once the account is topped up.
            _log(
                "\n[quota] a provider's paid balance is exhausted - stopping. "
                "Top up or swap the key, then re-run the same command to "
                "resume from the cursor.\n  " + str(e)
            )
            sys.exit(0)

    def _grow_subtopics(
        self, domain: str, subtopics: list[str], history: dict, state: dict
    ) -> list[str]:
        covered = [c for h in history.values() for c in h["concepts"]]
        try:
            fresh = self._with_outage_retry(
                lambda: self.mapper.expand(
                    domain, self.cfg.loop.subtopic_refresh_size, subtopics, covered
                ),
                "topic mapper (expand)",
            )
        except LLMError as e:
            _log(f"[topic mapper] expansion failed, keeping current pool: {e}")
            return subtopics
        if not fresh:
            _log("[topic mapper] expansion returned nothing new")
            return subtopics
        subtopics = subtopics + fresh
        self._n_subtopics = len(subtopics)
        state["subtopics"] = subtopics
        self.store.save_state(state)
        _log(f"[topic mapper] pool grown to {len(subtopics)}: " + " | ".join(fresh))
        return subtopics
