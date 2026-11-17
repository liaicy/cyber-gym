"""Audited manifest revisions for resuming collection after an explicit fix.

This module does not collect, train, rewrite trajectories, or reset the ledger.
The collector keeps its run lock; a separate revision lock serializes helper
calls. Historical trajectory fingerprints remain valid through verified history.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_REVISION_FIELDS = frozenset({"source_sha256", "unit_attempts", "retry_delay_seconds"})


def _encoded(value: Any) -> bytes:
    # Keep the collector's JSON separators so existing fingerprints agree.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(_encoded(config)).hexdigest()


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
            raise ValueError("manifest/config must be JSON objects")
        digest = fingerprint(manifest["config"])
        if manifest.get("fingerprint") != digest:
            raise ValueError("fingerprint does not match config")
        if not isinstance(manifest.get("created_at_utc"), str) or not manifest["created_at_utc"]:
            raise ValueError("missing original created_at_utc")
        return manifest, raw
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid collection manifest {path}: {exc}") from exc


def _changes(old: dict[str, Any], new: dict[str, Any], allow_budget_change: bool) -> list[str]:
    if not isinstance(allow_budget_change, bool):
        raise ValueError("allow_budget_change must be an explicit boolean")
    # Compare encoded values, not Python equality (True == 1, 1 == 1.0).
    changed = sorted(key for key in old.keys() | new.keys()
                     if key not in old or key not in new or _encoded(old[key]) != _encoded(new[key]))
    allowed = set(ALLOWED_REVISION_FIELDS)
    if allow_budget_change:
        allowed.add("call_budget")
    forbidden = set(changed) - allowed
    if forbidden:
        raise ValueError(f"Run config changes are not allowed for same-directory resume: {sorted(forbidden)}")
    if "call_budget" in changed:
        value = new.get("call_budget")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Revised call_budget must be a nonnegative integer; 0 means unlimited")
    if "unit_attempts" in changed and "unit_attempts" in new:
        value = new["unit_attempts"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("unit_attempts must be a positive integer")
    if "retry_delay_seconds" in changed and "retry_delay_seconds" in new:
        value = new["retry_delay_seconds"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("retry_delay_seconds must be a finite nonnegative number")
    return changed


def _verified_history(out_dir: Path) -> tuple[dict[str, Any], bytes, set[str]]:
    current, current_raw = _read_manifest(out_dir / "run_manifest.json")
    history = current.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Manifest history must be a list")
    archived = []
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise ValueError("Invalid manifest history entry")
        archive = entry.get("archive")
        if not isinstance(archive, str):
            raise ValueError("History is missing the archive path")
        relative = Path(archive)
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "revisions":
            raise ValueError("History archive must be directly inside revisions/")
        path = out_dir / relative
        revisions = out_dir / "revisions"
        if revisions.is_symlink() or path.is_symlink() or path.resolve().parent != revisions.resolve():
            raise ValueError("History archive path escapes revisions/")
        previous, raw = _read_manifest(path)
        if hashlib.sha256(raw).hexdigest() != entry.get("archive_sha256"):
            raise ValueError("Archived manifest SHA256 does not match history")
        if previous["fingerprint"] != entry.get("from_fingerprint"):
            raise ValueError("Archived fingerprint does not match history")
        if previous["created_at_utc"] != current["created_at_utc"]:
            raise ValueError("Revision changed the original creation time")
        if _encoded(previous.get("history", [])) != _encoded(history[:index]):
            raise ValueError("Archived manifest history is inconsistent")
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise ValueError("Revision requires a recorded reason")
        if not isinstance(entry.get("revised_at_utc"), str) or not entry["revised_at_utc"]:
            raise ValueError("Revision is missing its timestamp")
        archived.append(previous)
    for index, (entry, previous) in enumerate(zip(history, archived)):
        following = archived[index + 1] if index + 1 < len(archived) else current
        if entry.get("to_fingerprint") != following["fingerprint"]:
            raise ValueError("Manifest fingerprint chain is inconsistent")
        changed = _changes(previous["config"], following["config"], entry.get("allow_budget_change", False))
        if not changed or changed != entry.get("changed_fields"):
            raise ValueError("Revision changed_fields do not match the archived configs")
    if history:
        last = history[-1]
        if (current.get("revised_at_utc") != last["revised_at_utc"]
                or current.get("revision_reason") != last["reason"]):
            raise ValueError("Current manifest revision metadata is inconsistent")
    return current, current_raw, {current["fingerprint"], *(m["fingerprint"] for m in archived)}


def accepted_fingerprints(out_dir: str | Path) -> set[str]:
    """Validate current and every archived config/fingerprint, then return IDs.

    Unreferenced archive files are never accepted. This permits old valid
    trajectories to retain their original collection metadata.
    """
    return _verified_history(Path(out_dir))[2]


def _atomic_bytes(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_request(current: dict[str, Any], config: dict[str, Any], reason: str | None,
                      allow_budget_change: bool) -> tuple[str, list[str]]:
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    digest = fingerprint(config)
    if digest == current["fingerprint"]:
        return digest, []
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Run config/code/split changed; explicit revision reason or a new directory is required")
    return digest, _changes(current["config"], config, allow_budget_change)


def upgrade_manifest(out_dir: str | Path, config: dict[str, Any], reason: str | None,
                     *, allow_budget_change: bool = False) -> str:
    """Audit an explicitly authorized revision and return the current digest.

    No manifest is created here: use the collector's ensure_manifest for a new
    run. By default only source hashes and unit retry settings may change.
    call_budget may change only with allow_budget_change=True and a reason;
    the attempt ledger is never read, reset or modified by this helper.
    """
    out_dir = Path(out_dir)
    # JSON round-trip takes an independent snapshot of the caller's config.
    config = json.loads(_encoded(config))
    current, _, _ = _verified_history(out_dir)
    digest, changed = _validate_request(current, config, reason, allow_budget_change)
    if not changed:
        return digest
    with (out_dir / "manifest_revision.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current, current_raw, _ = _verified_history(out_dir)
        digest, changed = _validate_request(current, config, reason, allow_budget_change)
        if not changed:
            return digest
        revisions = out_dir / "revisions"
        if revisions.is_symlink():
            raise ValueError("revisions/ must not be a symlink")
        revisions.mkdir(exist_ok=True)
        archive_name = f"{uuid.uuid4().hex}-{current['fingerprint']}.json"
        archive = revisions / archive_name
        # Archive exact bytes before publishing the new pointer. A crash before
        # the final replace leaves the old current manifest intact.
        _atomic_bytes(archive, current_raw)
        revised_at = datetime.now(timezone.utc).isoformat()
        entry = {
            "from_fingerprint": current["fingerprint"],
            "to_fingerprint": digest,
            "archive": f"revisions/{archive_name}",
            "archive_sha256": hashlib.sha256(current_raw).hexdigest(),
            "revised_at_utc": revised_at,
            "reason": reason,
            "changed_fields": changed,
            "allow_budget_change": allow_budget_change,
        }
        updated = {**current, "config": config, "fingerprint": digest,
                   "history": [*current.get("history", []), entry],
                   "revised_at_utc": revised_at, "revision_reason": reason}
        raw = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        _atomic_bytes(out_dir / "run_manifest.json", raw)
        return digest
