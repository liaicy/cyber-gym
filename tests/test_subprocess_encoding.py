"""Offline encoding regressions using local source files and byte emitters.

No model, gen_task command, submission script or HTTP judge is executed.
Run: python tests/test_subprocess_encoding.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examsociety.llm import ModelRef
from traj import batch_rollout
from traj.agent import rollout
from traj.env import CyberGymEnv
from traj.schemas import TIER_INFRA_ERROR


_LOCAL_RUN = subprocess.run
_LOCAL_WHICH = shutil.which


def _byte_emitter(stdout: bytes = b"", stderr: bytes = b"", exit_code: int = 0):
    """Replace the requested command, retaining its real decoding options."""
    code = (
        "import sys;"
        f"sys.stdout.buffer.write(bytes({list(stdout)}));"
        f"sys.stderr.buffer.write(bytes({list(stderr)}));"
        f"sys.exit({exit_code})"
    )

    def run(_command, **kwargs):
        return _LOCAL_RUN([sys.executable, "-c", code], **kwargs)

    return run


class SubprocessEncodingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="subprocess_encoding_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "description.txt").write_text("offline fixture", encoding="utf-8")
        # This script is a fixture only; verify's command is always replaced.
        (self.root / "submit.sh").write_text("never execute this fixture\n", encoding="utf-8")
        (self.root / "poc").write_bytes(b"offline fixture")
        self.env = CyberGymEnv(
            str(self.root), server="http://mock-judge.invalid", api_key="MOCK_ONLY"
        )
        self.env._workroot.mkdir()

    def test_real_grep_and_read_file_agree_on_non_utf8_source(self):
        source = b"target invalid byte: \xff\n"
        (self.env._workroot / "bad.c").write_bytes(source)
        read = self.env.step("read_file", {"path": "bad.c"}).observation
        self.assertEqual(read, "target invalid byte: \ufffd\n")
        available = [tool for tool in ("rg", "grep") if _LOCAL_WHICH(tool)]
        if not available:
            self.skipTest("requires a local rg or grep executable")
        for tool in available:
            # GNU grep can classify invalid multibyte text as binary under a
            # UTF-8 locale. C makes its search byte based; Python must still
            # decode the returned bytes with the explicitly selected UTF-8.
            with self.subTest(tool=tool), patch(
                "traj.env.shutil.which",
                side_effect=lambda name: _LOCAL_WHICH(name) if name == tool else None,
            ), patch.dict("os.environ", {"LC_ALL": "C"}):
                observed = self.env.step("grep", {"pattern": "target"}).observation
                self.assertEqual(observed, "bad.c:1:" + read.rstrip("\n"))
        self.assertEqual((self.env._workroot / "bad.c").read_bytes(), source)

    def test_grep_non_utf8_stderr_does_not_discard_valid_matches(self):
        with patch("traj.env.shutil.which", return_value="/mock/tool"), patch(
            "traj.env.subprocess.run",
            side_effect=_byte_emitter(b"bad.c:1:target\n", b"diagnostic \xff"),
        ):
            result = self.env.step("grep", {"pattern": "target"})
        self.assertEqual(result.observation, "bad.c:1:target")

    def test_gen_task_success_tolerates_non_utf8_on_both_streams(self):
        with patch.object(batch_rollout, "CYBERGYM_SRC", str(self.root)), patch(
            "traj.batch_rollout.subprocess.run",
            side_effect=_byte_emitter(b"setup \xff", b"warning \xfe"),
        ):
            batch_rollout.gen_task("arvo:19757", self.root / "unused")
        self.assertFalse((self.root / "unused").exists())

    def test_gen_task_failure_keeps_non_utf8_diagnostic_and_exit_status(self):
        with patch.object(batch_rollout, "CYBERGYM_SRC", str(self.root)), patch(
            "traj.batch_rollout.subprocess.run",
            side_effect=_byte_emitter(b"setup \xff", b"diagnostic \xfe", exit_code=7),
        ), self.assertRaisesRegex(RuntimeError, "gen_task failed: diagnostic \ufffd"):
            batch_rollout.gen_task("arvo:19757", self.root / "unused")

    def test_verify_rejects_invalid_stdout_or_stderr_as_infra(self):
        # Even a complete verdict before the bad byte must not be recovered.
        response = b'{"poc_id":"offline-poc","exit_code":139}\n'
        for stdout, stderr in ((response + b"\xff", b""), (response, b"\xfe")):
            with self.subTest(stream="stdout" if not stderr else "stderr"), patch(
                "traj.env.subprocess.run", side_effect=_byte_emitter(stdout, stderr),
            ), patch("traj.env.shutil.which", return_value="/mock/bash"), patch(
                "httpx.post", side_effect=AssertionError("HTTP is forbidden in this test"),
            ) as post:
                verdict = self.env.verify()
                post.assert_not_called()
            self.assertEqual(verdict.tier, TIER_INFRA_ERROR)
            self.assertFalse(verdict.success)
            self.assertIsNone(verdict.vul_exit_code)
            self.assertIsNone(verdict.fix_exit_code)
            self.assertEqual(verdict.detail, "submit output is not valid UTF-8")
            self.assertIs(self.env._last_verdict, verdict)

    def test_rollout_retains_decode_failure_as_unscorable_trajectory(self):
        ref = ModelRef(id="offline", model="offline", api="openai", family="qwen")
        answer = '{"thought":"finish","action":"submit","action_input":{}}'
        with patch("traj.agent.chat", return_value=(answer, {})), patch(
            "traj.env.subprocess.run", side_effect=_byte_emitter(b"\xff"),
        ), patch("traj.env.shutil.which", return_value="/mock/bash"), patch(
            "httpx.post", side_effect=AssertionError("HTTP is forbidden in this test"),
        ) as post:
            trajectory = rollout(self.env, object(), ref, max_steps=1)
            post.assert_not_called()
        self.assertEqual(trajectory.n_steps, 1)
        self.assertEqual(trajectory.steps[0].action, "submit")
        self.assertEqual(trajectory.tier, TIER_INFRA_ERROR)
        self.assertEqual(trajectory.stop_reason, "error")
        self.assertFalse(trajectory.is_scorable())
        self.assertEqual(trajectory.verifier_detail, "submit output is not valid UTF-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
