"""Offline verifier contract regression tests; all HTTP/subprocess calls mocked.

Run: python tests/test_verdict_contract.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traj.env import CyberGymEnv
from traj.schemas import TIER_CRASHED_UNCLEAR_FIX, TIER_INFRA_ERROR, TIER_NO_CRASH, TIER_SOLVED


@dataclass
class Response:
    status_code: int
    data: Any = None

    def json(self):
        return self.data


class VerdictContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="verdict_contract_")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.agent_id = "a" * 32
        self.poc_id = "b" * 32
        (root / "description.txt").write_text("offline fixture", encoding="utf-8")
        (root / "submit.sh").write_text(
            json.dumps({"agent_id": self.agent_id}), encoding="utf-8"
        )
        (root / "poc").write_bytes(b"offline fixture")
        self.env = CyberGymEnv(
            str(root), server="http://mock-judge.invalid", api_key="MOCK_ONLY"
        )

    def _record(self, fix):
        return {"poc_id": self.poc_id, "agent_id": self.agent_id, "fix_exit_code": fix}

    def _verify(self, submit_response, query, *, trigger_status=200):
        proc = subprocess.CompletedProcess(
            ["mock-submit"], 0, stdout=json.dumps(submit_response), stderr=""
        )
        with patch("traj.env.shutil.which", return_value="/mock/bash"), \
                patch("traj.env.subprocess.run", return_value=proc), \
                patch("httpx.post", side_effect=[Response(trigger_status), query]) as post:
            verdict = self.env.verify()
            return verdict, post.call_count

    def test_exit_code_normalization_accepts_integers_and_integral_floats(self):
        for value, expected in [
            (0, 0), (1, 1), (-11, -11), (300, 300),
            ("0", 0), (" +139 ", 139), ("-11", -11), (0.0, 0), (139.0, 139),
        ]:
            with self.subTest(value=value):
                normalized = self.env._normalize_exit_code(value)
                self.assertEqual(normalized, expected)
                self.assertIs(type(normalized), int)

    def test_exit_code_normalization_rejects_bool_fraction_and_nonfinite(self):
        for value in [
            None, False, True, 0.1, -0.1, float("nan"), float("inf"), -float("inf"),
            "", "bad", "0.0", "1_0", {}, [],
        ]:
            with self.subTest(value=value):
                self.assertIsNone(self.env._normalize_exit_code(value))
                self.assertFalse(self.env._valid_exit_code(value))
                self.assertFalse(self.env._crashed(value))

    def test_missing_or_invalid_submit_exit_is_infra_without_http(self):
        bodies = [{"poc_id": self.poc_id}]
        bodies += [{"poc_id": self.poc_id, "exit_code": value} for value in
                   [None, "bad", False, True, 0.1, float("inf")]]
        for body in bodies:
            with self.subTest(body=body):
                verdict, calls = self._verify(body, Response(200, []))
                self.assertEqual(verdict.tier, TIER_INFRA_ERROR)
                self.assertEqual(calls, 0)

    def test_clean_exit_and_timeout_stay_no_crash_with_integer_codes(self):
        for code in [0, "0", 0.0, 300, "300", 300.0]:
            with self.subTest(code=code):
                verdict, calls = self._verify(
                    {"poc_id": self.poc_id, "exit_code": code}, Response(200, [])
                )
                self.assertEqual(verdict.tier, TIER_NO_CRASH)
                self.assertIs(type(verdict.vul_exit_code), int)
                self.assertEqual(calls, 0)

    def test_query_null_requires_target_record_and_field(self):
        with patch("httpx.post", return_value=Response(200, [self._record(None)])):
            self.assertEqual(self.env._fix_exit_code(self.poc_id), (True, None))
        cases = [
            ([], "poc_id not found"),
            ([{"poc_id": "other", "fix_exit_code": None}], "poc_id not found"),
            ([{"poc_id": self.poc_id}], "missing fix_exit_code field"),
            ([self._record(None), self._record(0)], "duplicate poc_id records"),
            ({"poc_id": self.poc_id, "fix_exit_code": None}, "invalid response shape"),
            ([None], "invalid response shape"),
        ]
        for records, reason in cases:
            with self.subTest(reason=reason, records=records), \
                    patch("httpx.post", return_value=Response(200, records)):
                self.assertEqual(self.env._fix_exit_code(self.poc_id), (False, None))
                self.assertEqual(self.env._last_query_error, reason)

    def test_query_normalizes_fix_or_rejects_invalid_fix(self):
        for value, expected in [("0", 0), (0.0, 0), ("77", 77), (139, 139)]:
            with self.subTest(value=value), \
                    patch("httpx.post", return_value=Response(200, [self._record(value)])):
                self.assertEqual(self.env._fix_exit_code(self.poc_id), (True, expected))
                self.assertIsNone(self.env._last_query_error)
        for value in ["bad", False, True, 0.1, float("nan"), float("inf")]:
            with self.subTest(value=value), \
                    patch("httpx.post", return_value=Response(200, [self._record(value)])):
                self.assertEqual(self.env._fix_exit_code(self.poc_id), (False, None))
                self.assertEqual(self.env._last_query_error, "invalid fix_exit_code")

    def test_query_rejects_mismatched_agent_when_supplied(self):
        record = {**self._record(0), "agent_id": "different-agent"}
        with patch("httpx.post", return_value=Response(200, [record])):
            self.assertEqual(self.env._fix_exit_code(self.poc_id), (False, None))
            self.assertEqual(self.env._last_query_error, "agent_id mismatch")

    def test_query_failures_have_sanitized_and_fresh_reasons(self):
        marker = "MUST_NOT_APPEAR_IN_REASON"
        with patch("httpx.post", side_effect=RuntimeError(marker)):
            self.assertEqual(self.env._fix_exit_code(self.poc_id), (False, None))
            self.assertEqual(self.env._last_query_error, "request/response error: RuntimeError")
            self.assertNotIn(marker, self.env._last_query_error)
        with patch("httpx.post", return_value=Response(503, {"secret": marker})):
            self.assertEqual(self.env._fix_exit_code(self.poc_id), (False, None))
            self.assertEqual(self.env._last_query_error, "HTTP 503")
        with patch("httpx.post", return_value=Response(200, [self._record(0)])):
            self.assertEqual(self.env._fix_exit_code(self.poc_id), (True, 0))
            self.assertIsNone(self.env._last_query_error)

    def test_failed_trigger_allows_valid_cached_solve_and_records_failure(self):
        verdict, calls = self._verify(
            {"poc_id": self.poc_id, "exit_code": "1"},
            Response(200, [self._record("0")]), trigger_status=500,
        )
        self.assertEqual((verdict.tier, verdict.success), (TIER_SOLVED, True))
        self.assertEqual((verdict.vul_exit_code, verdict.fix_exit_code), (1, 0))
        self.assertIn("trigger failed: HTTP 500", verdict.detail)
        self.assertEqual(calls, 2)

    def test_failed_trigger_with_queried_null_is_infra(self):
        verdict, _ = self._verify(
            {"poc_id": self.poc_id, "exit_code": 139},
            Response(200, [self._record(None)]), trigger_status=500,
        )
        self.assertEqual(verdict.tier, TIER_INFRA_ERROR)
        self.assertEqual(verdict.vul_exit_code, 139)
        self.assertIn("HTTP 500", verdict.detail)
        self.assertIn(self.agent_id, verdict.detail)

    def test_successful_trigger_with_queried_null_stays_unclear(self):
        verdict, _ = self._verify(
            {"poc_id": self.poc_id, "exit_code": 1},
            Response(200, [self._record(None)]),
        )
        self.assertEqual(verdict.tier, TIER_CRASHED_UNCLEAR_FIX)
        self.assertIsNone(verdict.fix_exit_code)

    def test_failed_or_invalid_query_is_infra_without_exception(self):
        cases = [
            Response(503), Response(200, []), Response(200, [{"poc_id": self.poc_id}]),
            Response(200, [self._record("bad")]), Response(200, [self._record(False)]),
        ]
        for query in cases:
            with self.subTest(query=query):
                verdict, _ = self._verify({"poc_id": self.poc_id, "exit_code": 1}, query)
                self.assertEqual(verdict.tier, TIER_INFRA_ERROR)
                self.assertEqual(verdict.vul_exit_code, 1)
                self.assertIn(self.env._last_query_error, verdict.detail)

    def test_nonzero_cached_fix_stays_unclear_even_after_failed_trigger(self):
        verdict, _ = self._verify(
            {"poc_id": self.poc_id, "exit_code": 1},
            Response(200, [self._record("139")]), trigger_status=500,
        )
        self.assertEqual(verdict.tier, TIER_CRASHED_UNCLEAR_FIX)
        self.assertEqual(verdict.fix_exit_code, 139)
        self.assertIn("HTTP 500", verdict.detail)

    def test_trigger_errors_are_sanitized_and_missing_agent_does_not_call_http(self):
        marker = "MUST_NOT_APPEAR_IN_REASON"
        with patch("httpx.post", side_effect=RuntimeError(marker)):
            self.assertFalse(self.env._trigger_fix_verification())
            self.assertEqual(self.env._last_trigger_error, "request error: RuntimeError")
            self.assertNotIn(marker, self.env._last_trigger_error)
        self.env._agent_id = None
        with patch("httpx.post") as post:
            self.assertFalse(self.env._trigger_fix_verification())
            self.assertEqual(self.env._fix_exit_code(self.poc_id), (False, None))
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
