"""Manifest revision contracts; all files are temporary and no API is used."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traj.collection_resume import accepted_fingerprints, fingerprint, upgrade_manifest
from traj import collection_resume


class CollectionResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.config = {
            "source_sha256": {"traj/collect_batch.py": "old-source"},
            "model_specs": [{"model": "Qwen/Qwen3-8B", "replicates": 3}],
            "task_ids": ["arvo:123"], "server": "http://fixture-judge:8666",
            "difficulty": "level1", "max_steps": 40, "max_tokens_per_turn": 4000,
            "role": "train", "call_budget": 18000, "out_dir": str(self.out),
            "work_dir": str(self.out / "work"), "unit_attempts": 1,
            "retry_delay_seconds": 0,
        }
        self.digest = fingerprint(self.config)
        self.path = self.out / "run_manifest.json"
        self.path.write_text(json.dumps({"fingerprint": self.digest, "config": self.config,
                                       "created_at_utc": "2026-10-09T00:00:00+00:00",
                                       "original_note": "preserve unknown metadata"}, indent=2) + "\n")
        self.original = self.path.read_bytes()
        self.ledger = self.out / "attempt_ledger.json"
        self.ledger.write_text('{"reserved_attempts": 1628}\n')
        self.trajectory = self.out / "arvo-123__qwen3-8b__r1.json"
        self.trajectory.write_text(json.dumps({"collection_metadata": {"run_fingerprint": self.digest}}))

    def changed(self, **updates):
        config = copy.deepcopy(self.config)
        config.update(updates)
        return config

    def manifest(self):
        return json.loads(self.path.read_text())

    def test_unchanged_manifest_is_read_only_and_agrees_with_collector_fingerprint(self):
        expected = hashlib.sha256(json.dumps(self.config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(self.digest, expected)
        self.assertEqual(upgrade_manifest(self.out, self.config, None), self.digest)
        self.assertEqual(accepted_fingerprints(self.out), {self.digest})
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse((self.out / "revisions").exists())

    def test_reason_required_and_only_explicit_fields_may_change(self):
        config = self.changed(source_sha256={"traj/collect_batch.py": "new-source"})
        for reason in (None, "", "   "):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "reason"):
                upgrade_manifest(self.out, config, reason)
        self.assertEqual(self.path.read_bytes(), self.original)
        for key, value in {
            "model_specs": [{"model": "different"}], "task_ids": ["arvo:456"],
            "server": "http://different-judge:8666", "difficulty": "level2",
            "max_steps": 100, "max_tokens_per_turn": 8000, "role": "dev",
            "out_dir": str(self.out / "other"), "work_dir": str(self.out / "other-work"),
            "workers": 100, "call_budget": 20000, "unexpected": "not-authorized",
        }.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "not allowed"):
                upgrade_manifest(self.out, self.changed(**{key: value}), "code fix")
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_revisions_archive_exact_bytes_and_preserve_valid_old_metadata(self):
        ledger_raw, trajectory_raw = self.ledger.read_bytes(), self.trajectory.read_bytes()
        config = self.changed(source_sha256={"traj/collect_batch.py": "new-source"},
                              unit_attempts=3, retry_delay_seconds=2)
        new_digest = upgrade_manifest(self.out, config, "Fix response retry strategy")
        current = self.manifest()
        self.assertEqual(new_digest, fingerprint(config))
        self.assertEqual(current["created_at_utc"], "2026-10-09T00:00:00+00:00")
        self.assertEqual(current["original_note"], "preserve unknown metadata")
        revision = current["history"][0]
        self.assertEqual((self.out / revision["archive"]).read_bytes(), self.original)
        self.assertEqual(revision["changed_fields"], ["retry_delay_seconds", "source_sha256", "unit_attempts"])
        self.assertFalse(revision["allow_budget_change"])
        self.assertEqual(accepted_fingerprints(self.out), {self.digest, new_digest})
        self.assertEqual(self.ledger.read_bytes(), ledger_raw)
        self.assertEqual(self.trajectory.read_bytes(), trajectory_raw)
        second = {**config, "unit_attempts": 4}
        second_digest = upgrade_manifest(self.out, second, "Bounded additional unit retries")
        self.assertEqual(len(self.manifest()["history"]), 2)
        self.assertEqual(accepted_fingerprints(self.out), {self.digest, new_digest, second_digest})
        self.assertEqual(upgrade_manifest(self.out, second, None), second_digest)
        self.assertEqual(len(self.manifest()["history"]), 2)

    def test_budget_to_unlimited_needs_explicit_authorization_and_never_resets_ledger(self):
        config = self.changed(call_budget=0)
        with self.assertRaisesRegex(ValueError, "call_budget"):
            upgrade_manifest(self.out, config, "Unrestricted budget requested")
        digest = upgrade_manifest(self.out, config, "User explicitly authorizes unlimited calls", allow_budget_change=True)
        self.assertTrue(self.manifest()["history"][0]["allow_budget_change"])
        self.assertEqual(self.manifest()["history"][0]["changed_fields"], ["call_budget"])
        self.assertEqual(accepted_fingerprints(self.out), {self.digest, digest})
        self.assertEqual(self.ledger.read_text(), '{"reserved_attempts": 1628}\n')
        # Budget authorization cannot authorize a model/task/config change.
        with self.assertRaisesRegex(ValueError, "task_ids"):
            upgrade_manifest(self.out, {**config, "task_ids": ["arvo:456"]}, "budget authorized", allow_budget_change=True)

    def test_invalid_budget_and_retry_values_are_rejected(self):
        for value in (-1, True, "0", 0.5):
            with self.subTest(budget=value), self.assertRaises(ValueError):
                upgrade_manifest(self.out, self.changed(call_budget=value), "authorization", allow_budget_change=True)
        for values in ({"unit_attempts": 0}, {"unit_attempts": True},
                       {"retry_delay_seconds": -1}, {"retry_delay_seconds": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                upgrade_manifest(self.out, self.changed(**values), "retry fix")

    def test_parallel_revisions_do_not_drop_manifest_history(self):
        configs = [self.changed(unit_attempts=2), self.changed(retry_delay_seconds=2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            digests = list(pool.map(
                lambda config: upgrade_manifest(self.out, config, "explicit retry revision"), configs,
            ))
        self.assertEqual(len(self.manifest()["history"]), 2)
        self.assertEqual(accepted_fingerprints(self.out), {self.digest, *digests})

    def test_tampered_current_or_archived_fingerprints_are_rejected(self):
        current = self.manifest()
        current["config"]["max_steps"] = 41
        self.path.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            accepted_fingerprints(self.out)
        self.path.write_bytes(self.original)
        upgrade_manifest(self.out, self.changed(unit_attempts=2), "retry fix")
        archive = self.out / self.manifest()["history"][0]["archive"]
        archived = json.loads(archive.read_text())
        archived["config"]["role"] = "dev"
        archive.write_text(json.dumps(archived))
        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            accepted_fingerprints(self.out)

    def test_removed_budget_authorization_and_history_path_escape_are_rejected(self):
        upgrade_manifest(self.out, self.changed(call_budget=0), "unlimited authorized", allow_budget_change=True)
        current = self.manifest()
        current["history"][0]["allow_budget_change"] = False
        self.path.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, "call_budget"):
            accepted_fingerprints(self.out)
        current["history"][0]["archive"] = "../outside.json"
        self.path.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, "revisions/"):
            accepted_fingerprints(self.out)

    def test_missing_manifest_not_created_and_failed_publish_keeps_current_intact(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "Invalid collection manifest"):
            upgrade_manifest(self.out, self.config, "fix")
        self.assertFalse(self.path.exists())
        self.path.write_bytes(self.original)
        real_replace = collection_resume.os.replace
        def fail_current_replace(source, target):
            if Path(target) == self.path:
                raise OSError("simulated publish failure")
            return real_replace(source, target)
        with patch.object(collection_resume.os, "replace", side_effect=fail_current_replace):
            with self.assertRaisesRegex(OSError, "publish failure"):
                upgrade_manifest(self.out, self.changed(unit_attempts=2), "retry fix")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(accepted_fingerprints(self.out), {self.digest})


if __name__ == "__main__":
    unittest.main()
