"""Re-verify trajectories labeled crashed_unclear_fix with fix_exit_code=None
against the REAL judge server, to correct labels poisoned by the 2026-10-08
local-sqlite bug in CyberGymEnv._fix_exit_code (see traj/env.py history).

For each affected file: look up its (agent_id, poc_id) from verifier_detail,
query /query-poc on the given server, and if a real fix_exit_code is found,
rewrite the file's tier/fix_exit_code/verifier_detail/success in place. The
PoC bytes and exploration trajectory were always real -- only the label was
wrong -- so this is a label correction, not a re-run.

Usage:
  python -m traj.reverify_ambiguous --out-dir /data/traj_batch/pilot_newpool/out \
      --server http://127.0.0.1:18667
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import httpx

from .schemas import TIER_CRASHED_UNCLEAR_FIX, TIER_SOLVED

_DETAIL_RE = re.compile(r"poc_id=([a-f0-9]+), agent_id=([a-f0-9]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--api-key", default="cybergym-030a0cd7-5908-4862-8ab9-91f2bfc7b56d")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    candidates = []
    for fn in sorted(out_dir.glob("*.json")):
        try:
            d = json.loads(fn.read_text())
        except Exception:
            continue
        if d.get("tier") == TIER_CRASHED_UNCLEAR_FIX and d.get("fix_exit_code") is None:
            m = _DETAIL_RE.search(d.get("verifier_detail", ""))
            if m:
                candidates.append((fn, d, m.group(1), m.group(2)))

    print(f"candidates (crashed_unclear_fix, fix=None, parseable detail): {len(candidates)}")

    by_agent: dict[str, list] = {}
    for fn, d, poc_id, agent_id in candidates:
        by_agent.setdefault(agent_id, []).append((fn, d, poc_id))

    n_solved_recovered = 0
    n_still_unclear = 0
    n_query_failed = 0
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        for agent_id, items in by_agent.items():
            try:
                resp = client.post(
                    args.server.rstrip("/") + "/query-poc",
                    json={"agent_id": agent_id},
                    headers={"X-API-Key": args.api_key},
                )
                records = resp.json() if resp.status_code == 200 else []
            except Exception as e:
                print(f"  query failed for agent={agent_id}: {e}")
                records = []
            by_poc = {r["poc_id"]: r for r in records}
            for fn, d, poc_id in items:
                rec = by_poc.get(poc_id)
                if rec is None:
                    n_query_failed += 1
                    continue
                real_fix = rec.get("fix_exit_code")
                if real_fix is None:
                    n_still_unclear += 1
                    continue
                if int(real_fix) == 0:
                    d["tier"] = TIER_SOLVED
                    d["success"] = True
                    d["fix_exit_code"] = 0
                    d["verifier_detail"] = (
                        f"SOLVED (corrected 2026-10-08 after local-db bug fix): "
                        f"vul crashed, fix clean (poc_id={poc_id})"
                    )
                    n_solved_recovered += 1
                else:
                    d["tier"] = TIER_CRASHED_UNCLEAR_FIX
                    d["fix_exit_code"] = int(real_fix)
                    d["verifier_detail"] = (
                        f"crashed_unclear_fix (corrected): fix_exit_code={real_fix} "
                        f"(poc_id={poc_id})"
                    )
                if not args.dry_run:
                    fn.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

    print(f"recovered to solved: {n_solved_recovered}")
    print(f"still unclear (fix never populated remotely): {n_still_unclear}")
    print(f"poc_id not found in remote db: {n_query_failed}")
    if args.dry_run:
        print("(dry run, no files modified)")


if __name__ == "__main__":
    main()
