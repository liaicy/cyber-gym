"""Persist bounded unit attempts and preserve failed trajectory artifacts."""
from __future__ import annotations

import json
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def scrub(value):
    """Remove credential values from diagnostic text, without dumping locals."""
    text = str(value)
    for name, secret in os.environ.items():
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I) and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|password|access[_-]?token|secret)\s*[\"']?\s*[:=]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
        r"\1[REDACTED]", text,
    )
    return text[:4000]


def exception_diagnostic(error):
    return {
        "category": type(error).__name__,
        "message": scrub(error),
        "traceback_frames": [
            {"file": scrub(frame.filename), "line": frame.lineno, "function": scrub(frame.name)}
            for frame in traceback.extract_tb(error.__traceback__)
        ],
    }


class UnitAttempts:
    """All mutations are atomic and serialized across collector worker threads.

    The collector's output-directory flock excludes a second process. A crash
    after begin() leaves its reservation consumed, so restart never resets the
    per-unit ceiling. Failed JSON artifacts are moved intact, never rewritten.
    """

    def __init__(self, out):
        self.out = Path(out)
        self.path = self.out / "unit_attempts.json"
        self._lock = threading.RLock()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "units": {}}
        if self.data.get("version") != 1 or not isinstance(self.data.get("units"), dict):
            raise ValueError("Invalid persisted unit attempt state")
        for state in self.data["units"].values():
            count = state.get("attempts_consumed")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Invalid persisted unit attempt count")

    def _save(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def _state(self, key, unit=None):
        if key not in self.data["units"]:
            task, model, replicate = unit
            self.data["units"][key] = {
                "task": task, "model": model, "replicate": replicate,
                "attempts_consumed": 0, "status": "pending", "attempts": [],
            }
        return self.data["units"][key]

    def count(self, key):
        with self._lock:
            return self.data["units"].get(key, {}).get("attempts_consumed", 0)

    def status(self, key):
        with self._lock:
            return self.data["units"].get(key, {}).get("status", "pending")

    def begin(self, key, unit, limit, fingerprint):
        with self._lock:
            state = self._state(key, unit)
            if state["attempts_consumed"] >= limit:
                return None
            state["attempts_consumed"] += 1
            number = state["attempts_consumed"]
            state["status"] = "running"
            state["attempts"].append({"number": number, "started_at_utc": now_utc(),
                                       "run_fingerprint": fingerprint, "status": "running"})
            self._save()
            return number

    def finish(self, key, status, diagnostic=None, archive=None):
        with self._lock:
            state = self._state(key)
            state["status"] = status
            attempt = state["attempts"][-1]
            attempt.update(status=status, finished_at_utc=now_utc())
            if diagnostic is not None:
                attempt["diagnostic"] = diagnostic
            if archive is not None:
                attempt["trajectory_archive"] = str(Path(archive).relative_to(self.out))
            self._save()

    def archive(self, path, number):
        path = Path(path)
        if not path.exists():
            return None
        directory = self.out / "failed_attempts" / path.stem
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"attempt-{number:03d}.json"
        suffix = 1
        while destination.exists():
            destination = directory / f"attempt-{number:03d}-artifact-{suffix}.json"
            suffix += 1
        os.replace(path, destination)
        return destination

    def import_failed_record(self, key, unit, path, diagnostic, fingerprint):
        """Count a legacy failed output once; recover a crashed attempt in place."""
        with self._lock:
            state = self._state(key, unit)
            if (not state["attempts_consumed"]
                    or state["attempts"][-1].get("legacy_log_source")):
                self.begin(key, unit, state["attempts_consumed"] + 1, fingerprint)
            archive = self.archive(path, state["attempts_consumed"])
            self.finish(key, "failed", diagnostic, archive)
            return archive

    def import_exception_history(self, key, unit, entries):
        """Import the legacy collector's exact exception lines only once.

        Those attempts produced no output JSON. Their original traceback and
        precise date are unavailable; retain log path/line/time as evidence
        rather than manufacture them. Valid existing records bypass this call.
        """
        with self._lock:
            if key in self.data["units"] or not entries:
                return
            state = self._state(key, unit)
            for entry in entries:
                state["attempts_consumed"] += 1
                state["attempts"].append({
                    "number": state["attempts_consumed"], "status": "failed",
                    "imported_at_utc": now_utc(), "legacy_log_source": entry["source"],
                    "diagnostic": {"category": entry["category"], "traceback_frames": [],
                                   "message": "Legacy log contains the exception class only; original traceback unavailable"},
                })
            state["status"] = "failed"
            self._save()
