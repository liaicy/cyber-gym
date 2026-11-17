"""Collection contract checks; no provider/judge requests or real data edits."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examsociety.llm import BudgetExhausted, ModelRef
from traj.collect_batch import LedgerClient, collect, ensure_manifest, load_config, preflight_seats, units_for
from traj import batch_rollout as batch
from traj.schemas import Trajectory


class CollectionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks.txt"
        self.tasks.write_text("arvo:123\n")
        self.split = self.root / "split.json"
        self.split.write_text(json.dumps({"status": "candidate_not_activated", "roles": {
            "new_train": ["arvo:123"], "new_dev": ["arvo:456"],
            "final_test": ["arvo:789"], "legacy_dev_regression": ["arvo:999"],
        }}))
        source = self.root / "cyber"
        task_dir = source / "cybergym_data/data/arvo/123"
        task_dir.mkdir(parents=True)
        for name in ("description.txt", "repo-vul.tar.gz"):
            (task_dir / name).write_text("fixture")
        self.path = self.root / "config.json"
        self.config = {
            "run_id": "test_collect", "server": "http://127.0.0.1:18667",
            "task_file": str(self.tasks), "split_manifest": str(self.split), "role": "train",
            "models": {"qwen3-8b": 3, "glm-5.3": 1}, "out_dir": str(self.root / "out"),
            "work_dir": str(self.root / "work"), "log_path": str(self.root / "batch.log"),
            "difficulty": "level1", "max_steps": 40, "max_tokens_per_turn": 4000,
            "workers": 2, "max_provider_concurrency": 1, "call_budget": 30, "max_errors": 1,
            "cybergym_src": str(source), "cybergym_python": sys.executable,
            # These contracts mock run_unit and must not hit the live endpoints;
            # the startup health check is exercised separately in PreflightSeats.
            "preflight_probes": 0,
        }

    def load(self):
        self.path.write_text(json.dumps(self.config))
        return load_config(self.path)

    def test_config_resolves_roles_model_flags_and_exact_replicates(self):
        config, models = self.load()
        self.assertEqual(len(units_for(config["task_ids"], models)), 4)
        self.assertEqual(models[0][0].extra_body, {"enable_thinking": False})
        self.assertEqual(models[1][0].extra_body, {"thinking": {"type": "disabled"}})

    def test_task_role_rejects_reserved_test_even_if_train_list_is_corrupted(self):
        self.tasks.write_text("arvo:789\n")
        # Give it local files, so task-role validation is the rejecting boundary.
        directory = Path(self.config["cybergym_src"]) / "cybergym_data/data/arvo/789"
        directory.mkdir(parents=True)
        for name in ("description.txt", "repo-vul.tar.gz"):
            (directory / name).write_text("fixture")
        data = json.loads(self.split.read_text())
        data["roles"]["new_train"].append("arvo:789")
        self.split.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.load()

    def test_unknown_model_cannot_silently_reduce_the_roster(self):
        self.config["models"] = {"typo-model": 1}
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            self.load()

    def test_duplicates_missing_sources_and_secret_config_fields_are_rejected(self):
        self.tasks.write_text("arvo:123 arvo:123")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.load()
        self.tasks.write_text("arvo:123")
        (Path(self.config["cybergym_src"]) / "cybergym_data/data/arvo/123/repo-vul.tar.gz").unlink()
        with self.assertRaisesRegex(ValueError, "task data missing"):
            self.load()
        self.config["api_key"] = "not-a-real-key"
        with self.assertRaisesRegex(ValueError, "Unknown config"):
            self.load()

    def test_manifest_allows_same_resume_and_rejects_model_or_code_change(self):
        config, _ = self.load()
        out = Path(config["out_dir"])
        out.mkdir()
        original = ensure_manifest(out, config)
        self.assertEqual(ensure_manifest(out, config), original)
        config["model_specs"][0]["ref"]["model"] = "different-checkpoint"
        with self.assertRaisesRegex(ValueError, "changed"):
            ensure_manifest(out, config)

    def test_unmanifested_old_outputs_cannot_be_resumed(self):
        config, _ = self.load()
        out = Path(config["out_dir"])
        out.mkdir()
        (out / "old.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "no matching run manifest"):
            ensure_manifest(out, config)

    def test_attempt_cap_is_exact_under_concurrency_and_persists_across_restart(self):
        client = LedgerClient(5, self.root / "attempts.json")
        ref = ModelRef(id="test", model="test", api="openai")
        def charge(_):
            try:
                client._charge(ref)
                return True
            except BudgetExhausted:
                return False
        with ThreadPoolExecutor(max_workers=8) as executor:
            allowed = list(executor.map(charge, range(40)))
        self.assertEqual(sum(allowed), 5)
        self.assertEqual(client.calls, 5)
        resumed = LedgerClient(5, self.root / "attempts.json")
        with self.assertRaises(BudgetExhausted):
            resumed._charge(ref)
        self.assertEqual(resumed.calls, 5)

    def test_run_unit_records_effective_metadata_and_uses_requested_token_budget(self):
        ref = ModelRef(id="mock", model="actual-model", api="openai")
        trajectory = Trajectory(task_id="arvo-123__mock", model_id="mock", tier="no_crash", stop_reason="step_budget")
        with patch.object(batch, "OUT_DIR", self.root / "out"), \
             patch.object(batch, "WORK_DIR", self.root / "work"), \
             patch.object(batch, "SERVER", "http://fixture-judge:18667"), \
             patch.dict("os.environ", {"CYBERGYM_SERVER": "http://wrong-default:8666"}), \
             patch.object(batch, "gen_task"), patch.object(batch, "CyberGymEnv") as environment, \
             patch.object(batch, "rollout", return_value=trajectory) as rollout:
            batch.run_unit(type("Client", (), {"calls": 0})(), "arvo:123", ref, 1,
                           collection_metadata={"run_fingerprint": "fixture", "model": {"model": ref.model}},
                           max_tokens_per_turn=1234)
            rollout.assert_called_once()
            self.assertEqual(rollout.call_args.kwargs["max_tokens_per_turn"], 1234)
            self.assertEqual(environment.call_args.kwargs["server"], "http://fixture-judge:18667")
        saved = Trajectory.model_validate_json((self.root / "out/arvo-123__mock__r1.json").read_text())
        self.assertEqual(saved.collection_metadata["actual_model_id"], "mock")
        self.assertEqual(saved.collection_metadata["model"]["model"], "actual-model")
        self.assertIsNone(saved.collection_metadata["sampling_seed"])

    def test_collector_executes_mock_units_and_resumes_without_additional_attempts(self):
        config, models = self.load()
        def fake_rollout(env, client, ref, **kwargs):
            client._charge(ref)
            return Trajectory(task_id="arvo-123__fixture", model_id=ref.id,
                              tier="degenerate", stop_reason="step_budget")
        with patch.dict("os.environ", {"SF_KEY": "fixture", "SFLAG_KEY": "fixture"}), \
             patch.object(batch, "gen_task"), patch.object(batch, "CyberGymEnv"), \
             patch.object(batch, "rollout", side_effect=fake_rollout), \
             patch("builtins.print"):
            self.assertEqual(collect(config, models), 0)
            self.assertEqual(collect(config, models), 0)
        summary = json.loads((Path(config["out_dir"]) / "summary.json").read_text())
        self.assertEqual(summary["recorded_units"], 4)
        self.assertEqual(summary["reserved_attempts"], 4)
        self.assertEqual(summary["judge_coverage"]["vul_verdict_records"], 0)
        self.assertEqual(summary["judge_coverage"]["fix_result_records"], 0)

    def test_collector_budget_stops_before_all_units_without_fabricating_outputs(self):
        self.config["call_budget"] = 1
        config, models = self.load()
        def fake_rollout(env, client, ref, **kwargs):
            client._charge(ref)
            return Trajectory(task_id="arvo-123__fixture", model_id=ref.id,
                              tier="degenerate", stop_reason="step_budget")
        with patch.dict("os.environ", {"SF_KEY": "fixture", "SFLAG_KEY": "fixture"}), \
             patch.object(batch, "gen_task"), patch.object(batch, "CyberGymEnv"), \
             patch.object(batch, "rollout", side_effect=fake_rollout), \
             patch("builtins.print"):
            self.assertEqual(collect(config, models), 2)
        summary = json.loads((Path(config["out_dir"]) / "summary.json").read_text())
        self.assertEqual(summary["recorded_units"], 1)
        self.assertEqual(summary["missing_units"], 3)
        self.assertEqual(summary["reserved_attempts"], 1)


class PreflightSeats(unittest.TestCase):
    def _models(self, ids):
        return [(ModelRef(id=i, model=i, api="openai", base_url="http://x/v1",
                          api_key_env="X"), 1) for i in ids]

    def test_zero_probes_disables_nothing_and_skips_network(self):
        with patch("traj.collect_batch._probe_once") as probe:
            down = preflight_seats(self._models(["a", "b"]), 0, lambda m: None)
        self.assertEqual(down, {})
        probe.assert_not_called()

    def test_disables_only_persistently_failing_seats(self):
        def probe(ref, timeout=60):
            return (False, "HTTP 503 down") if ref.id == "bad" else (True, "200")
        with patch("traj.collect_batch._probe_once", side_effect=probe):
            down = preflight_seats(self._models(["good", "bad"]), 3, lambda m: None)
        self.assertEqual(set(down), {"bad"})
        self.assertIn("503", down["bad"])

    def test_one_success_keeps_seat_and_stops_early(self):
        calls = {"n": 0}

        def probe(ref, timeout=60):
            calls["n"] += 1
            return (calls["n"] > 1, "200" if calls["n"] > 1 else "HTTP 503")

        with patch("traj.collect_batch._probe_once", side_effect=probe):
            down = preflight_seats(self._models(["flaky"]), 3, lambda m: None)
        self.assertEqual(down, {})
        self.assertEqual(calls["n"], 2)  # stopped on first success, did not spend all 3

    def test_all_seats_down_raises(self):
        with patch("traj.collect_batch._probe_once", return_value=(False, "HTTP 503")):
            with self.assertRaises(ValueError):
                preflight_seats(self._models(["a", "b"]), 2, lambda m: None)


if __name__ == "__main__":
    unittest.main()
