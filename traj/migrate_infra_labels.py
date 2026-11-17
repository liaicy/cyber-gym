"""Retroactively relabel OLD trajectory files whose verifier_detail proves
they were an infra/harness failure, not a real model behavioral failure, to
the new TIER_INFRA_ERROR taxonomy (schemas.py, 2026-10-08).

Scope, deliberately narrow (round-2 external review's own caution: "对无法
区分的设 unknown/待审计,不要靠字符串批量猜测无法确认的历史状态"):

Before 2026-10-08, verify() used ONE tier (TIER_DEGENERATE) for four distinct
causes, each with its own fixed message prefix:
  1. "no submit.sh or no poc written"            -- AMBIGUOUS: this exact
     string fired for BOTH "gen_task setup never ran" (infra) and "the model
     never wrote a PoC" (real behavior). Cannot be split from the string
     alone -- LEFT UNTOUCHED.
  2. "bash unavailable (verifier needs..."        -- unambiguous infra
  3. "submit failed to run: ..."                  -- unambiguous infra
  4. "unparseable server response: ..."           -- unambiguous infra
Only 2-4 are migrated here; this is a label correction (matching
reverify_ambiguous.py's precedent), not a re-run -- the recorded steps and
PoC bytes are untouched.

Does NOT touch any already-exported train.jsonl (dpo/sft/grpo) -- those were
built from the OLD labels and need a fresh export run to pick this up; see
CLAUDE.md for which export runs are known to include an affected trajectory.

Usage:
  python -m traj.migrate_infra_labels --root /data/traj_batch [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import TIER_DEGENERATE, TIER_INFRA_ERROR

_UNAMBIGUOUS_INFRA_PREFIXES = (
    "bash unavailable",
    "submit failed to run:",
    "unparseable server response:",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/traj_batch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    n_scanned = n_matched = n_unreadable = 0
    touched: list[str] = []
    for fn in sorted(root.rglob("*.json")):
        try:
            d = json.loads(fn.read_text(encoding="utf-8"))
        except Exception:
            n_unreadable += 1
            continue
        if not isinstance(d, dict) or "tier" not in d:
            continue  # not a Trajectory file (configs, logs, etc. also match *.json)
        n_scanned += 1
        if d.get("tier") != TIER_DEGENERATE:
            continue
        detail = d.get("verifier_detail", "")
        if not any(detail.startswith(p) for p in _UNAMBIGUOUS_INFRA_PREFIXES):
            continue
        n_matched += 1
        touched.append(str(fn))
        d["tier"] = TIER_INFRA_ERROR
        d["stop_reason"] = "error"
        d["verifier_detail"] = f"[migrated 2026-10-08, was degenerate] {detail}"
        if not args.dry_run:
            fn.write_text(json.dumps(d, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    print(f"scanned {n_scanned} trajectory files ({n_unreadable} unreadable/skipped)")
    print(f"migrated {n_matched} degenerate -> infra_error")
    for p in touched[:20]:
        print(f"  {p}")
    if len(touched) > 20:
        print(f"  ... and {len(touched) - 20} more")
    if args.dry_run:
        print("(dry run, no files modified)")


if __name__ == "__main__":
    main()
