# -*- coding: utf-8 -*-
"""Load the raw trajectory JSONs run_one.py wrote, convert to traj.schemas
objects, and run them through build_preference() -- the last untested link in
env -> rollout -> preference data.

Also demonstrates the "no silent noise" guard: a single trajectory must NOT
produce a preference pair (there is nothing to contrast against).

Usage: python traj/convert_and_rank.py /root/cybergym-src/cybergym_tmp
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traj.preference import build_preference, rank_score  # noqa: E402
from traj.schemas import Step, Trajectory  # noqa: E402


def load(path: str) -> Trajectory:
    raw = json.load(open(path, encoding="utf-8"))
    steps = [
        Step(index=s["i"], thought=s.get("thought", ""), action=s.get("action", ""),
            action_input=s.get("action_input") or {}, observation=s.get("observation", ""),
            prompt_tokens=s.get("prompt_tokens", 0), completion_tokens=s.get("completion_tokens", 0))
        for s in raw["steps"]
    ]
    # run_one.py's stop values -> schema's is_scorable() vocabulary.
    # "submitted" means the episode completed (right or wrong); is_scorable()
    # only checks that we HAVE a real outcome, not that it was a solve.
    raw_stop = raw["stop_reason"]
    if raw_stop == "model_error":
        stop_reason = "error"
    elif raw["success"]:
        stop_reason = "solved"
    else:
        stop_reason = "step_budget"  # ran out of useful progress, one way or another
    return Trajectory(
        task_id=raw["task_id"], model_id=raw["model"],
        steps=steps, success=raw["success"],
        tier=raw.get("tier", ""),                      # "" for trajectories predating tiering
        vul_exit_code=raw.get("vul_exit_code"), fix_exit_code=raw.get("fix_exit_code"),
        verifier_detail=raw["verdict"],
        n_steps=raw["n_steps"], tokens=raw.get("tokens", 0), wall_seconds=raw["wall_seconds"],
        stop_reason=stop_reason,
    )


def main():
    task_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/cybergym-src/cybergym_tmp"
    paths = sorted(glob.glob(os.path.join(task_dir, "trajectory_*.json")))
    print(f"found {len(paths)} raw trajectories:")
    trajs = []
    for p in paths:
        t = load(p)
        trajs.append(t)
        print(f"  {os.path.basename(p):40s} model={t.model_id:28s} "
              f"steps={t.n_steps:3d} tier={t.tier or '?':20s} scorable={t.is_scorable()} "
              f"tokens={t.tokens:6d} score={rank_score(t):.2f}")

    print()
    print("=== guard check: ONE trajectory must yield no pair (no contrast) ===")
    single = build_preference(trajs[0].task_id, "<prompt>", trajs[:1])
    print("  result:", single, "  (" + ("PASS: correctly None" if single is None else "FAIL: should be None") + ")")

    if len(trajs) < 2:
        print("\nneed >=2 trajectories for the real ranking test; only got", len(trajs))
        return

    print()
    print("=== real ranking: all trajectories on this task ===")
    desc_path = os.path.join(task_dir, "description.txt")
    prompt = open(desc_path, encoding="utf-8").read() if os.path.exists(desc_path) else "<no description>"
    pref = build_preference(trajs[0].task_id, prompt, trajs, provenance={"source": "cybergym"})
    if pref is None:
        print("  build_preference returned None -- no contrast in this set")
        return
    print(f"  PreferencePair id={pref.id} task={pref.task_id}")
    for r in pref.ranked:
        print(f"    {r.model_id:28s} success={r.success!s:5s} score={r.rank_score:7.2f} "
              f"steps={r.n_steps} tokens={r.tokens}")
    cr = pref.chosen_rejected()
    print("  chosen/reject:", cr)
    if cr:
        chosen_id, reject_id = cr
        chosen = next(t for t in trajs if t.id == chosen_id)
        reject = next(t for t in trajs if t.id == reject_id)
        print(f"  chosen  = {chosen.model_id} (score {rank_score(chosen):.2f})")
        print(f"  reject  = {reject.model_id} (score {rank_score(reject):.2f})")


if __name__ == "__main__":
    main()
