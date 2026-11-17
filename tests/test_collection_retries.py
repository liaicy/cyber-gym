"""Offline integration contracts for finite per-unit collection retries."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_collect_batch as fixtures
from examsociety.llm import BudgetExhausted, ModelRef
from traj import batch_rollout as batch
from traj.collect_batch import LedgerClient, collect, ensure_manifest
from traj.collection_failures import UnitAttempts
from traj.collection_resume import upgrade_manifest
from traj.schemas import Trajectory


class CollectionRetries(unittest.TestCase):
    def setUp(self):
        fixtures.CollectionContracts.setUp(self)
        self.config.update(models={"qwen3-8b": 1}, workers=1,
                           call_budget=0, max_errors=1, unit_attempts=3,
                           retry_delay_seconds=0)

    load = fixtures.CollectionContracts.load

    def tasks_for(self, *tasks):
        self.tasks.write_text("\n".join(tasks))
        split = json.loads(self.split.read_text())
        split["roles"]["new_train"] = list(tasks)
        self.split.write_text(json.dumps(split))
        for task in tasks:
            source, number = task.split(":")
            directory = Path(self.config["cybergym_src"]) / "cybergym_data/data" / source / number
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("description.txt", "repo-vul.tar.gz"):
                (directory / name).write_text("fixture")

    def write_record(self, task, ref, rep, metadata, *, bad=False, detail=""):
        out = Path(self.config["out_dir"])
        out.mkdir(exist_ok=True)
        path = out / f"{task.replace(':', '-')}__{ref.id}__r{rep}.json"
        record = Trajectory(task_id=task, model_id=ref.id, tier="degenerate",
                            stop_reason="error" if bad else "step_budget",
                            verifier_detail=detail, collection_metadata=metadata)
        path.write_text(record.model_dump_json(indent=2))
        return path

    def run_collector(self, config, models, fake):
        with patch.dict("os.environ", {"SF_KEY": "fixture-private-credential"}), \
             patch.object(batch, "run_unit", side_effect=fake), patch("builtins.print"):
            result = collect(config, models)
        summary = json.loads((Path(config["out_dir"]) / "summary.json").read_text())
        return result, summary

    def state(self):
        return json.loads((Path(self.config["out_dir"]) / "unit_attempts.json").read_text())["units"]

    def test_persistent_bad_unit_does_not_stop_later_tasks_at_max_errors(self):
        self.tasks_for("arvo:123", "arvo:124", "arvo:125")
        config, models = self.load()
        seen = []
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            seen.append(task)
            if task == "arvo:123":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
            self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(seen, ["arvo:123"] * 3 + ["arvo:124", "arvo:125"])
        self.assertEqual(result, 2)
        self.assertEqual(summary["errors_this_invocation"], 3)
        self.assertEqual(summary["failed_skipped_units"], 1)
        self.assertEqual(summary["successful_units"], 2)
        self.assertEqual(summary["unprocessed_units"], 0)
        self.assertTrue(summary["processing_complete"])
        attempt = self.state()["arvo-123__qwen3-8b__r1.json"]["attempts"][0]
        self.assertEqual(attempt["diagnostic"]["category"], "UnicodeDecodeError")
        self.assertTrue(attempt["diagnostic"]["traceback_frames"])
        self.assertEqual(set(attempt["diagnostic"]["traceback_frames"][0]), {"file", "line", "function"})

    def test_transient_exception_recovers_and_retry_delays_are_bounded(self):
        self.config["retry_delay_seconds"] = 5
        config, models = self.load()
        calls = []
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            calls.append(task)
            if len(calls) < 3:
                raise RuntimeError("api_key=fixture-private-credential")
            self.write_record(task, ref, rep, collection_metadata)
            return "done recovered"
        with patch("traj.collect_batch.time.sleep") as sleep:
            result, summary = self.run_collector(config, models, fake)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 10])
        self.assertEqual(result, 0)
        self.assertEqual(summary["reserved_attempts"], 3)
        self.assertEqual(summary["errors_this_invocation"], 2)
        self.assertNotIn("fixture-private-credential", json.dumps(self.state()))
        self.assertTrue(summary["processing_complete"])

    def test_unscorable_empty_response_is_archived_intact_before_success(self):
        config, models = self.load()
        raw = []
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            if not raw:
                p = self.write_record(task, ref, rep, collection_metadata, bad=True,
                                      detail="empty response; api_key=fixture-private-credential")
                raw.append(p.read_bytes())
            else:
                self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 0)
        self.assertEqual(summary["scorable"], 1)
        archived = list((Path(config["out_dir"]) / "failed_attempts").rglob("*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), raw[0])
        state = self.state()["arvo-123__qwen3-8b__r1.json"]
        self.assertEqual(state["attempts_consumed"], 2)
        self.assertIn("[REDACTED]", state["attempts"][0]["diagnostic"]["detail"])
        self.assertNotIn("fixture-private-credential", json.dumps(state))

    def test_legacy_error_counts_once_and_old_metadata_is_preserved(self):
        config, models = self.load()
        out = Path(config["out_dir"])
        out.mkdir()
        digest = ensure_manifest(out, config)
        LedgerClient(0, out / "attempt_ledger.json")._charge(models[0][0])
        p = self.write_record("arvo:123", models[0][0], 1,
                              {"run_fingerprint": digest, "legacy_metadata": "preserve"},
                              bad=True, detail="model empty response")
        raw = p.read_bytes()
        calls = []
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            calls.append(task)
            self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        archived = list((out / "failed_attempts").rglob("*.json"))
        self.assertEqual(archived[0].read_bytes(), raw)
        self.assertEqual(self.state()[p.name]["attempts_consumed"], 2)
        self.assertEqual(summary["reserved_attempts"], 2)

    def test_exhausted_units_stay_exhausted_across_restart(self):
        config, models = self.load()
        calls = []
        def fake(client, task, ref, rep, **kwargs):
            calls.append(task)
            client._charge(ref)
            raise RuntimeError("persistent fixture failure")
        first, first_summary = self.run_collector(config, models, fake)
        second, second_summary = self.run_collector(config, models, fake)
        self.assertEqual((first, second), (2, 2))
        self.assertEqual(len(calls), 3)
        self.assertEqual(second_summary["reserved_attempts"], 3)
        self.assertEqual(second_summary["errors_this_invocation"], 0)
        self.assertTrue(second_summary["processing_complete"])
        self.assertEqual(second_summary["failed_skipped_units"], 1)

    def test_crash_after_attempt_reservation_cannot_reset_ceiling(self):
        config, models = self.load()
        out = Path(config["out_dir"])
        out.mkdir()
        digest = ensure_manifest(out, config)
        state = UnitAttempts(out)
        key = "arvo-123__qwen3-8b__r1.json"
        state.begin(key, ("arvo:123", "qwen3-8b", 1), 3, digest)
        state.begin(key, ("arvo:123", "qwen3-8b", 1), 3, digest)
        calls = []
        def fake(client, task, ref, rep, **kwargs):
            calls.append(task)
            raise RuntimeError("third failure")
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.state()[key]["attempts_consumed"], 3)

    def test_valid_completed_records_are_unchanged_and_consume_no_new_attempt(self):
        config, models = self.load()
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        self.run_collector(config, models, fake)
        p = Path(config["out_dir"]) / "arvo-123__qwen3-8b__r1.json"
        raw = p.read_bytes()
        before = json.dumps(self.state(), sort_keys=True)
        def forbidden(*args, **kwargs):
            self.fail("valid existing record must not invoke run_unit")
        result, summary = self.run_collector(config, models, forbidden)
        self.assertEqual(result, 0)
        self.assertEqual(p.read_bytes(), raw)
        self.assertEqual(json.dumps(self.state(), sort_keys=True), before)
        self.assertEqual(summary["reserved_attempts"], 1)

    def test_unlimited_ledger_persists_and_does_not_reset_counts(self):
        path = self.root / "ledger.json"
        ref = ModelRef(id="fixture", model="fixture", api="openai")
        client = LedgerClient(0, path)
        for _ in range(7):
            client._charge(ref)
        resumed = LedgerClient(0, path)
        for _ in range(5):
            resumed._charge(ref)
        self.assertEqual(resumed.calls, 12)
        self.assertEqual(resumed.by_model[ref.id], 12)

    def test_nonzero_cap_blocks_retry_without_exceeding_global_budget(self):
        self.tasks_for("arvo:123", "arvo:124")
        self.config["call_budget"] = 2
        config, models = self.load()
        calls = []
        def fake(client, task, ref, rep, **kwargs):
            client._charge(ref)
            calls.append(task)
            raise RuntimeError("bad unit")
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(summary["reserved_attempts"], 2)
        self.assertEqual(summary["failed_skipped_units"], 0)
        self.assertEqual(summary["unprocessed_units"], 2)
        self.assertFalse(summary["processing_complete"])
        self.assertTrue(summary["budget_exhausted"])

    def test_budget_exception_is_not_retried_or_classified_as_model_failure(self):
        config, models = self.load()
        calls = []
        def fake(client, task, ref, rep, **kwargs):
            calls.append(task)
            raise BudgetExhausted("fixture budget abort")
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary["errors_this_invocation"], 0)
        self.assertFalse(summary["processing_complete"])
        self.assertEqual(self.state()["arvo-123__qwen3-8b__r1.json"]["status"], "budget_blocked")

    def test_config_accepts_zero_budget_and_limits_attempts_to_three(self):
        config, _ = self.load()
        self.assertEqual(config["call_budget"], 0)
        for invalid in (-1, True, 1.5):
            self.config["call_budget"] = invalid
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                self.load()
        self.config["call_budget"] = 0
        self.config["unit_attempts"] = 4
        with self.assertRaisesRegex(ValueError, "must not exceed 3"):
            self.load()

    def test_audited_upgrade_accepts_old_valid_fingerprint_without_rewriting_record(self):
        self.tasks_for("arvo:123", "arvo:124")
        config, models = self.load()
        old_config = json.loads(json.dumps(config))
        old_config.pop("unit_attempts")
        old_config.pop("retry_delay_seconds")
        old_config["call_budget"] = 18000
        old_config["source_sha256"]["traj/collect_batch.py"] = "previous-source"
        out = Path(config["out_dir"])
        out.mkdir()
        previous = ensure_manifest(out, old_config)
        LedgerClient(18000, out / "attempt_ledger.json")._charge(models[0][0])
        valid = self.write_record("arvo:123", models[0][0], 1,
                                  {"run_fingerprint": previous, "legacy": "keep exact"})
        raw = valid.read_bytes()
        upgrade_manifest(out, config, "Authorized code repair and unlimited budget", allow_budget_change=True)
        seen = []
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            seen.append(task)
            self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 0)
        self.assertEqual(seen, ["arvo:124"])
        self.assertEqual(valid.read_bytes(), raw)
        self.assertNotIn(valid.name, self.state())
        self.assertEqual(summary["reserved_attempts"], 2)

    def test_budget_abort_on_last_attempt_does_not_block_other_units_on_restart(self):
        self.tasks_for("arvo:123", "arvo:124")
        config, models = self.load()
        out = Path(config["out_dir"])
        out.mkdir()
        digest = ensure_manifest(out, config)
        state = UnitAttempts(out)
        key = "arvo-123__qwen3-8b__r1.json"
        for _ in range(3):
            state.begin(key, ("arvo:123", "qwen3-8b", 1), 3, digest)
        state.finish(key, "budget_blocked", {"category": "BudgetExhausted"})
        seen = []
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            seen.append(task)
            self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        result, summary = self.run_collector(config, models, fake)
        self.assertEqual(result, 2)
        self.assertEqual(seen, ["arvo:124"])
        self.assertTrue(summary["processing_complete"])
        self.assertEqual(summary["failed_skipped_units"], 1)
        self.assertEqual(summary["failed_units"][0]["last_status"], "budget_blocked")

    def test_legacy_exception_logs_consume_attempts_once_across_resume(self):
        config, models = self.load()
        old = json.loads(json.dumps(config))
        old["source_sha256"].pop("traj/collection_failures.py")
        old["source_sha256"].pop("traj/collection_resume.py")
        out = Path(config["out_dir"])
        out.mkdir()
        ensure_manifest(out, old)
        LedgerClient(0, out / "attempt_ledger.json")._charge(models[0][0])
        upgrade_manifest(out, config, "Authorized legacy collector repair")
        Path(config["log_path"]).write_text(
            "[09:49:12] unit failed arvo:123/qwen3-8b/r1: UnicodeDecodeError\n"
            "[09:50:00] unit failed arvo:999/qwen3-8b/r1: RuntimeError\n"
        )
        calls = []
        def fake(client, task, ref, rep, **kwargs):
            client._charge(ref)
            calls.append(task)
            raise RuntimeError("persistent fixture failure")
        first, _ = self.run_collector(config, models, fake)
        second, summary = self.run_collector(config, models, fake)
        self.assertEqual((first, second), (2, 2))
        self.assertEqual(len(calls), 2)
        units = self.state()
        self.assertEqual(len(units), 1)
        state = units["arvo-123__qwen3-8b__r1.json"]
        self.assertEqual(state["attempts_consumed"], 3)
        self.assertEqual(state["attempts"][0]["diagnostic"]["category"], "UnicodeDecodeError")
        self.assertEqual(state["attempts"][0]["legacy_log_source"]["line"], 1)
        self.assertEqual(summary["errors_this_invocation"], 0)

    def test_new_run_does_not_import_an_unrelated_reused_log(self):
        config, models = self.load()
        Path(config["log_path"]).write_text(
            "[09:49:12] unit failed arvo:123/qwen3-8b/r1: UnicodeDecodeError\n"
        )
        def fake(client, task, ref, rep, collection_metadata, **kwargs):
            client._charge(ref)
            self.write_record(task, ref, rep, collection_metadata)
            return "done fixture"
        result, _ = self.run_collector(config, models, fake)
        self.assertEqual(result, 0)
        self.assertEqual(self.state()["arvo-123__qwen3-8b__r1.json"]["attempts_consumed"], 1)


if __name__ == "__main__":
    unittest.main()
