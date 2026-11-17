"""Offline safety/reproducibility checks for candidate task preparation."""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traj.prepare_next_batch import prepare_plan, write_plan
from traj import prepare_next_batch


class PrepareNextBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.batch = self.root / "original.txt"
        self.batch.write_text("\n".join(f"arvo:{i}" for i in range(1, 101)) + "\n")
        self.out = self.root / "out"
        self.out.mkdir()
        for i in range(1, 35):
            record = {"task_id": f"arvo-{i}__qwen3-8b__r1__isolated",
                      "model_id": "qwen3-8b", "tier": "no_crash"}
            (self.out / f"arvo-{i}__qwen3-8b__r1.json").write_text(json.dumps(record))
        self.split = self.root / "old_split.json"
        self.split.write_text(json.dumps({
            "eval_holdout_clean_uniform": [f"arvo:{i}" for i in range(901, 905)],
            "train_clean_uniform": ["arvo:800"],
            "train_contaminated_near_distribution": ["oss-fuzz:900"],
        }))
        self.trained = self.root / "train.jsonl"
        self.trained.write_text(json.dumps({"meta": {"task": "arvo:700"}}) + "\n")
        self.kwargs = dict(batch_file=self.batch, observed_out=self.out,
                           prior_split=self.split, training_jsonls=[self.trained])

    def tearDown(self):
        self.tmp.cleanup()

    def test_role_disjointness_freezes_unobserved_test_and_preserves_regression(self):
        plan, audit = prepare_plan(**self.kwargs)
        roles = plan["roles"]
        self.assertEqual(plan["counts"], {
            "observed_train": 34, "new_train": 41, "train_candidates": 75,
            "new_dev": 10, "final_test": 15, "legacy_dev_regression": 4,
        })
        train = set(roles["observed_train"] + roles["new_train"])
        dev, test = (set(roles[g]) for g in ("new_dev", "final_test"))
        self.assertFalse(train & dev or train & test or dev & test)
        self.assertEqual(train | dev | test, set(self.batch.read_text().split()))
        self.assertFalse(set(roles["observed_train"]) & (dev | test))
        self.assertEqual(roles["legacy_dev_regression"], [f"arvo:{i}" for i in range(901, 905)])
        self.assertIsNone(plan["original_sampling_seed"])
        self.assertEqual(audit["batch_file"]["sha256"], hashlib.sha256(self.batch.read_bytes()).hexdigest())

    def test_split_is_repeatable_without_changing_global_rng(self):
        before = random.getstate()
        a, _ = prepare_plan(**self.kwargs, seed=42)
        b, _ = prepare_plan(**self.kwargs, seed=42)
        c, _ = prepare_plan(**self.kwargs, seed=43)
        self.assertEqual(a["roles"], b["roles"])
        self.assertNotEqual(a["roles"]["final_test"], c["roles"]["final_test"])
        self.assertEqual(before, random.getstate())

    def test_duplicate_and_invalid_planned_task_ids_fail(self):
        self.batch.write_text(self.batch.read_text() + "arvo:1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_plan(**self.kwargs)
        self.batch.write_text("../../escape\n")
        with self.assertRaisesRegex(ValueError, "Invalid task"):
            prepare_plan(**self.kwargs)

    def test_known_training_and_old_holdout_overlap_fail(self):
        for task in ("arvo:700", "arvo:901"):
            with self.subTest(task=task):
                self.batch.write_text("\n".join([task] + [f"arvo:{i}" for i in range(2, 101)]))
                with self.assertRaisesRegex(ValueError, "overlap existing train/eval"):
                    prepare_plan(**self.kwargs)

    def test_bad_or_out_of_pool_observation_is_not_silently_ignored(self):
        path = self.out / "arvo-101__qwen3-8b__r1.json"
        path.write_text(json.dumps({"task_id": "arvo-101__x"}))
        with self.assertRaisesRegex(ValueError, "outside the original"):
            prepare_plan(**self.kwargs)
        path.write_text(json.dumps({"task_id": "arvo-1__x"}))
        with self.assertRaisesRegex(ValueError, "filename disagree"):
            prepare_plan(**self.kwargs)
        path.write_text("{")
        with self.assertRaises(ValueError):
            prepare_plan(**self.kwargs)

    def test_writing_preserves_inputs_and_refuses_nonempty_destination(self):
        sources = [self.batch, self.split, self.trained, *self.out.glob("*.json")]
        originals = {p: p.read_bytes() for p in sources}
        plan, audit = prepare_plan(**self.kwargs)
        dest = self.root / "plan"
        write_plan(dest, plan, audit)
        self.assertEqual(len((dest / "tasks_train.txt").read_text().split()), 41)
        self.assertEqual(len((dest / "tasks_dev.txt").read_text().split()), 10)
        collector_tasks = set((dest / "tasks_train.txt").read_text().split())
        collector_tasks.update((dest / "tasks_dev.txt").read_text().split())
        self.assertFalse(collector_tasks & set(plan["roles"]["final_test"]))
        self.assertFalse(collector_tasks & set(plan["roles"]["legacy_dev_regression"]))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            write_plan(dest, plan, audit)
        self.assertEqual({p: p.read_bytes() for p in sources}, originals)
        self.assertEqual(json.loads((dest / "candidates_split.json").read_text())["status"],
                         "candidate_not_activated")

    def test_cli_extra_training_input_cannot_bypass_known_registry(self):
        self.batch.write_text("\n".join(["arvo:700"] + [f"arvo:{i}" for i in range(2, 101)]))
        extra = self.root / "extra.jsonl"
        extra.write_text(json.dumps({"meta": {"task": "arvo:701"}}) + "\n")
        dest = self.root / "not_written"
        argv = ["prepare_next_batch", "--batch-file", str(self.batch), "--observed-out", str(self.out),
                "--prior-split", str(self.split), "--training-jsonl", str(extra), "--dest", str(dest)]
        errors = io.StringIO()
        with patch.object(prepare_next_batch, "DEFAULT_TRAINING_JSONLS", (self.trained,)), \
                patch.object(sys, "argv", argv), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                prepare_next_batch.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("overlap existing train/eval", errors.getvalue())
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
