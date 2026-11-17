"""Validated rollout collection. --dry-run performs no network calls.

Config fixes the task role, judge, models and budgets. A run manifest prevents
resuming with different code/prompts/models, and a persisted attempt ledger
caps calls across restarts. Credentials stay in environment variables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import urllib.error
import urllib.request

from examsociety.llm import BudgetExhausted, LLMClient
from traj import batch_rollout as batch
from traj.collection_failures import UnitAttempts, exception_diagnostic, scrub
from traj.collection_resume import accepted_fingerprints
from traj.schemas import Trajectory

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "traj/collect_batch.py", "traj/batch_rollout.py", "traj/agent.py",
    "traj/chat.py", "traj/env.py", "traj/schemas.py", "examsociety/llm.py",
    "traj/collection_failures.py",
    "traj/collection_resume.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _probe_once(ref, timeout=60):
    """One lightweight health request straight to the seat's endpoint.

    Bypasses LLMClient on purpose, so health checks never touch the attempt
    ledger or the long 5xx backoff. openai-protocol seats only; any other
    protocol is assumed up (not probed). Returns (ok, detail).
    """
    if ref.api != "openai":
        return True, "skipped (non-openai protocol assumed up)"
    base = (ref.base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    key = os.environ.get(ref.api_key_env or "OPENAI_API_KEY", "")
    payload = {"model": ref.model,
               "messages": [{"role": "user", "content": "ping"}],
               getattr(ref, "token_param", "max_tokens"): 16,
               **ref.extra_body}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True, "200"
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("error", {}).get("message", "")[:120]
        except Exception:
            pass
        return False, f"HTTP {e.code} {detail}".strip()
    except Exception as e:  # noqa: BLE001 -- any transport failure means "down"
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def preflight_seats(models, n_probes, log):
    """Probe every seat at startup; temporarily disable the ones that are down.

    Each seat gets up to n_probes serial probes and is kept on the FIRST
    success (so a flaky-but-alive seat stays). A seat with zero successes is
    "persistently exploding" and is disabled for THIS run only -- there is no
    persisted state, so the next launch re-probes and a recovered seat rejoins
    automatically. Returns {seat_id: last_error} for the disabled seats.
    """
    down = {}
    if n_probes <= 0:
        return down
    for ref, _ in models:
        ok, last = False, "no probe"
        for _ in range(n_probes):
            ok, last = _probe_once(ref)
            last = scrub(last)
            if ok:
                break
        if ok:
            log(f"preflight: {ref.id} ok")
        else:
            down[ref.id] = last
            log(f"preflight: DISABLING {ref.id} this run -- {n_probes} probes failed, last: {last}")
    if down and len(down) == len(models):
        raise ValueError(f"All seats failed preflight; nothing to collect. Last errors: {down}")
    return down


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    allowed = {
        "run_id", "server", "task_file", "split_manifest", "role", "models",
        "out_dir", "work_dir", "log_path", "difficulty", "max_steps",
        "max_tokens_per_turn", "workers", "max_provider_concurrency",
        "call_budget", "max_errors", "cybergym_src", "cybergym_python",
        "preflight_probes", "unit_attempts", "retry_delay_seconds",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    for key in ("task_file", "split_manifest", "out_dir", "work_dir", "log_path",
                "cybergym_src", "cybergym_python"):
        p = Path(config[key])
        config[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    server = urlsplit(config["server"])
    if server.scheme not in ("http", "https") or not server.hostname:
        raise ValueError("server must be an explicit HTTP(S) URL")
    if server.username or server.password or server.query or server.fragment:
        raise ValueError("server URL must not contain credentials/query/fragment")
    if config["difficulty"] not in ("level1", "level2", "level3", "level0"):
        raise ValueError("Unknown difficulty")
    config.setdefault("max_errors", 3)  # legacy field; no longer stops the batch
    config.setdefault("unit_attempts", 3)
    config.setdefault("retry_delay_seconds", 5)
    for key in ("workers", "max_provider_concurrency", "max_errors", "unit_attempts",
                "max_steps", "max_tokens_per_turn"):
        positive_int(config[key], key)
    if config["unit_attempts"] > 3:
        raise ValueError("unit_attempts must not exceed 3")
    nonnegative_int(config["call_budget"], "call_budget")
    nonnegative_int(config["retry_delay_seconds"], "retry_delay_seconds")
    config.setdefault("preflight_probes", 3)
    if (isinstance(config["preflight_probes"], bool)
            or not isinstance(config["preflight_probes"], int)
            or config["preflight_probes"] < 0):
        raise ValueError("preflight_probes must be a non-negative integer (0 disables preflight)")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_id"]):
        raise ValueError("run_id must be a simple versioned name")
    if not Path(config["cybergym_src"]).is_dir() or not Path(config["cybergym_python"]).is_file():
        raise ValueError("CyberGym source or Python path does not exist")
    if config["out_dir"] == config["work_dir"]:
        raise ValueError("Output and working directories must be different")
    task_text = Path(config["task_file"]).read_text()
    tasks = task_text.replace(",", " ").split()
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Task list must be non-empty and unique")
    if any(not re.fullmatch(r"(?:arvo|oss-fuzz):[0-9]+", t) for t in tasks):
        raise ValueError("Invalid task ID")
    missing_data = []
    for task in tasks:
        source, number = task.split(":")
        directory = Path(config["cybergym_src"]) / "cybergym_data/data" / source / number
        required = ["description.txt", "repo-vul.tar.gz"]
        if config["difficulty"] == "level2":
            required.append("error.txt")
        if any(not (directory / name).is_file() for name in required):
            missing_data.append(task)
    if missing_data:
        raise ValueError(f"Required local task data missing: {missing_data}")
    split = json.loads(Path(config["split_manifest"]).read_text())
    role = {"train": "new_train", "dev": "new_dev"}.get(config["role"])
    if role is None:
        raise ValueError("Collection role must be train or dev; final_test stays reserved")
    roles = split["roles"]
    if not set(tasks).issubset(set(roles[role])):
        raise ValueError(f"Task list contains tasks outside split role {role}")
    forbidden = set(roles["final_test"]) | set(roles["legacy_dev_regression"])
    if config["role"] == "train":
        forbidden |= set(roles["new_dev"])
    if set(tasks) & forbidden:
        raise ValueError("Task list overlaps reserved dev/test")
    available = batch.REF_BY_ID
    models = []
    if not isinstance(config["models"], dict) or not config["models"]:
        raise ValueError("models must map known model IDs to replicate counts")
    for model_id, reps in config["models"].items():
        if model_id not in available:
            raise ValueError(f"Unknown model ID: {model_id}")
        positive_int(reps, f"replicates for {model_id}")
        models.append((available[model_id], reps))
    config["task_ids"] = tasks
    config["model_specs"] = [{"ref": asdict(ref), "replicates": reps} for ref, reps in models]
    config["task_file_sha256"] = sha256(config["task_file"])
    config["split_manifest_sha256"] = sha256(config["split_manifest"])
    config["source_sha256"] = {p: sha256(ROOT / p) for p in SOURCE_FILES}
    config["seed_note"] = "API sampling seed is unset unless present in model extra_body"
    config["split_status"] = split["status"]
    return config, models


def fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_manifest(out_dir, config):
    out_dir = Path(out_dir)
    manifest = out_dir / "run_manifest.json"
    digest = fingerprint(config)
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != digest:
            raise ValueError("Run config/code/split changed; use a new output directory")
        return digest
    if any(out_dir.glob("*.json")):
        raise ValueError("Existing trajectories have no matching run manifest; use a new directory")
    atomic_json(manifest, {"fingerprint": digest, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                           "config": config})
    return digest


class LedgerClient(LLMClient):
    """Reserve attempts under a lock before network calls, including retries.

    A crash may leave an unused reservation. This conservatively preserves
    the ceiling; attempts are not a monetary accounting or token budget.
    """

    def __init__(self, max_calls, ledger):
        super().__init__(max_calls=max_calls)
        self.ledger = Path(ledger)
        self._budget_lock = threading.Lock()
        data = json.loads(self.ledger.read_text()) if self.ledger.exists() else {}
        self.calls = data.get("reserved_attempts", 0)
        if isinstance(self.calls, bool) or not isinstance(self.calls, int) or self.calls < 0:
            raise ValueError("Invalid persisted attempt count")
        self.by_model = Counter(data.get("by_model", {}))

    def _charge(self, ref):
        with self._budget_lock:
            if self.max_calls and self.calls >= self.max_calls:
                raise BudgetExhausted(f"reserved attempt budget spent: {self.calls}/{self.max_calls}")
            self.calls += 1
            self.by_model[ref.id] += 1
            atomic_json(self.ledger, {"reserved_attempts": self.calls, "by_model": dict(self.by_model)})


def units_for(tasks, models):
    # Alternate provider seats before extra target replicates.
    return [(task, ref, rep) for task in tasks
            for rep in range(1, max(n for _, n in models) + 1)
            for ref, n in models if rep <= n]


def collect(config, models):
    import fcntl

    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError("Another collector is using this output directory") from e
        digest = ensure_manifest(out, config)
        for ref, _ in models:
            if not os.environ.get(ref.api_key_env or "OPENAI_API_KEY"):
                raise ValueError(f"Missing credential environment variable: {ref.api_key_env or 'OPENAI_API_KEY'}")
        batch.SERVER = config["server"]
        batch.CYBERGYM_SRC = config["cybergym_src"]
        batch.DATA_DIR = str(Path(batch.CYBERGYM_SRC) / "cybergym_data/data")
        batch.MASK_MAP = str(Path(batch.CYBERGYM_SRC) / "mask_map.json")
        batch.GEN_PY = config["cybergym_python"]
        batch.OUT_DIR, batch.WORK_DIR = out, Path(config["work_dir"])
        batch.LOG_PATH = Path(config["log_path"])
        batch.DIFFICULTY, batch.MAX_STEPS = config["difficulty"], config["max_steps"]
        batch.WORK_DIR.mkdir(parents=True, exist_ok=True)
        batch.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Health-check every seat at startup; persistently-exploding ones (e.g.
        # an upstream that is 503ing for this launch) are disabled for THIS run
        # only and re-probed next launch, so a recovered seat rejoins on its own.
        down = preflight_seats(models, config["preflight_probes"], batch.log)
        llm = LedgerClient(config["call_budget"], out / "attempt_ledger.json")
        limits = {(ref.base_url, ref.api_key_env): threading.Semaphore(config["max_provider_concurrency"])
                  for ref, _ in models}
        metadata = {"run_id": config["run_id"], "run_fingerprint": digest,
                    "role": config["role"], "server": config["server"],
                    "difficulty": config["difficulty"], "max_steps": config["max_steps"],
                    "max_tokens_per_turn": config["max_tokens_per_turn"],
                    "split_manifest_sha256": config["split_manifest_sha256"]}
        planned = units_for(config["task_ids"], models)
        allowed_digests = accepted_fingerprints(out)
        state = UnitAttempts(out)
        attempt_limit = config.get("unit_attempts", 3)
        retry_delay = config.get("retry_delay_seconds", 5)
        budget_stop = threading.Event()
        errors_lock = threading.Lock()
        failures = 0

        def output_path(unit):
            task, ref, rep = unit
            return out / f"{task.replace(':', '-')}__{ref.id}__r{rep}.json"

        def identity(unit):
            task, ref, rep = unit
            return task, ref.id, rep

        def invalid_diagnostic(record):
            return {"category": "UnscorableTrajectory", "tier": record.tier,
                    "stop_reason": record.stop_reason, "detail": scrub(record.verifier_detail)}

        def validated_record(path):
            record = Trajectory.model_validate_json(path.read_text())
            if record.collection_metadata.get("run_fingerprint") not in allowed_digests:
                raise ValueError("Existing trajectory metadata does not match this run")
            return record

        # Validate provenance before changing any old artifacts. Failed legacy
        # outputs count once and move intact so run_unit cannot skip them as done.
        existing = {}
        for unit in planned:
            path = output_path(unit)
            if path.exists():
                existing[path.name] = validated_record(path)
                if not llm.ledger.exists():
                    raise ValueError("Existing trajectories have no persisted attempt ledger")
        legacy_errors = {}
        manifest = json.loads((out / "run_manifest.json").read_text())
        # Only bootstrap logs from an audited ancestor lacking the persistent
        # unit state. A fresh run must not inherit failures from a reused log.
        has_legacy_ancestor = any(
            "traj/collection_failures.py" not in
            json.loads((out / entry["archive"]).read_text())["config"].get("source_sha256", {})
            for entry in manifest.get("history", [])
        )
        if has_legacy_ancestor and llm.ledger.exists() and batch.LOG_PATH.exists():
            pattern = re.compile(
                r"^\[([0-9:]{8})\] unit failed ((?:arvo|oss-fuzz):[0-9]+)/"
                r"([A-Za-z0-9_.-]+)/r([0-9]+): ([A-Za-z_][A-Za-z0-9_.]*)$"
            )
            for line_number, line in enumerate(batch.LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                match = pattern.fullmatch(line)
                if match:
                    observed_time, task, model, rep, category = match.groups()
                    legacy_errors.setdefault((task, model, int(rep)), []).append({
                        "category": category,
                        "source": {"file": str(batch.LOG_PATH), "line": line_number,
                                   "observed_time_hms": observed_time},
                    })
        for unit in planned:
            path = output_path(unit)
            record = existing.get(path.name)
            if record is None or not record.is_scorable():
                state.import_exception_history(path.name, identity(unit), legacy_errors.get(identity(unit), []))
            if record is not None and not record.is_scorable():
                state.import_failed_record(path.name, identity(unit), path,
                                           invalid_diagnostic(record),
                                           record.collection_metadata["run_fingerprint"])

        def run(unit):
            nonlocal failures
            task, ref, rep = unit
            path = output_path(unit)
            key = path.name
            if path.exists() and validated_record(path).is_scorable():
                return "success", f"skip (valid) {key}"
            while state.count(key) < attempt_limit:
                if budget_stop.is_set() or (llm.max_calls and llm.calls >= llm.max_calls):
                    budget_stop.set()
                    return "budget", f"budget blocked {key}"
                consumed = state.count(key)
                if consumed:
                    time.sleep(min(10, retry_delay * consumed))
                with limits[(ref.base_url, ref.api_key_env)]:
                    if budget_stop.is_set() or (llm.max_calls and llm.calls >= llm.max_calls):
                        budget_stop.set()
                        return "budget", f"budget blocked {key}"
                    number = state.begin(key, identity(unit), attempt_limit, digest)
                    if number is None:
                        break
                    try:
                        msg = batch.run_unit(llm, task, ref, rep,
                                             collection_metadata={**metadata, "model": asdict(ref)},
                                             max_tokens_per_turn=config["max_tokens_per_turn"])
                        record = validated_record(path)
                        if record.is_scorable():
                            state.finish(key, "success")
                            return "success", msg
                        diagnostic = invalid_diagnostic(record)
                    except BudgetExhausted as error:
                        budget_stop.set()
                        state.finish(key, "budget_blocked", exception_diagnostic(error))
                        return "budget", f"reserved attempt budget exhausted; {key} left incomplete"
                    except Exception as error:
                        diagnostic = exception_diagnostic(error)
                    archive = state.archive(path, number)
                    state.finish(key, "failed", diagnostic, archive)
                    with errors_lock:
                        failures += 1
                    batch.log(f"unit attempt failed {task}/{ref.id}/r{rep} "
                              f"attempt={number}/{attempt_limit}: {diagnostic['category']}")
            return "exhausted", f"skip failed {key}: {state.count(key)} attempts consumed"

        expected = [u for u in planned if u[1].id not in down]
        pending_units = iter(expected)
        # Unit errors never halt unrelated tasks. Only the explicitly configured
        # nonzero global call ceiling stops admission; in-flight work is drained.
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            active = {}
            while True:
                while (not budget_stop.is_set() and len(active) < config["workers"]
                       and (not llm.max_calls or llm.calls < llm.max_calls)):
                    unit = next(pending_units, None)
                    if unit is None:
                        break
                    active[executor.submit(run, unit)] = unit
                if not active:
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    active.pop(future)
                    status, message = future.result()
                    batch.log(message)
                    if status == "budget":
                        budget_stop.set()
        records = []
        for unit in planned:
            path = output_path(unit)
            if path.exists():
                record = validated_record(path)
                if record.is_scorable():
                    records.append(record)
        expected_success = []
        exhausted = []
        unprocessed = []
        for unit in expected:
            path = output_path(unit)
            if path.exists() and validated_record(path).is_scorable():
                expected_success.append(unit)
            elif state.count(path.name) >= attempt_limit:
                exhausted.append(unit)
            else:
                unprocessed.append(unit)
        processing_complete = not unprocessed
        summary = {"run_id": config["run_id"], "time_utc": datetime.now(timezone.utc).isoformat(),
                   "planned_units": len(planned), "expected_units": len(expected),
                   "skipped_preflight_units": len(planned) - len(expected),
                   "preflight_disabled": down,
                   "recorded_units": len(records),
                   "successful_units": len(expected_success), "failed_skipped_units": len(exhausted),
                   "unprocessed_units": len(unprocessed), "processing_complete": processing_complete,
                   "failed_units": [{"task": u[0], "model": u[1].id, "replicate": u[2],
                                     "attempts_consumed": state.count(output_path(u).name),
                                     "last_status": state.status(output_path(u).name)} for u in exhausted],
                   "missing_units": len(expected) - len(expected_success), "reserved_attempts": llm.calls,
                   "call_budget": llm.max_calls, "scorable": len(records),
                   "expected_scorable": len(expected_success),
                   "errors_this_invocation": failures,
                   "budget_exhausted": bool(budget_stop.is_set() or
                       (llm.max_calls and llm.calls >= llm.max_calls and unprocessed)),
                   "tiers": dict(Counter(t.tier for t in records)), "models": dict(Counter(t.model_id for t in records)),
                   "judge_coverage": {
                       "vul_verdict_records": sum(t.vul_exit_code is not None for t in records),
                       "vul_crash_records": sum(t.vul_exit_code is not None and
                                                t.vul_exit_code not in (0, 300) for t in records),
                       "fix_result_records": sum(t.fix_exit_code is not None for t in records),
                   },
                   "per_model": {ref.id: {
                       "recorded": sum(t.model_id == ref.id for t in records),
                       "scorable": sum(t.model_id == ref.id for t in records),
                       "solved": sum(t.model_id == ref.id and t.success for t in records),
                       "failed_skipped": sum(u[1].id == ref.id for u in exhausted),
                       "unprocessed": sum(u[1].id == ref.id for u in unprocessed),
                       "preflight_down": ref.id in down,
                   } for ref, _ in models}}
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if processing_complete and not exhausted else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        config, models = load_config(args.config)
        planned = units_for(config["task_ids"], models)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "network_calls": 0, "fingerprint": fingerprint(config),
                              "planned_units": len(planned), "max_turn_calls_without_retries": len(planned) * config["max_steps"],
                              "config": config}, ensure_ascii=False, indent=2))
            return 0
        return collect(config, models)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
