"""Export collected rollout trajectories to GRPO training data.

Route 2 of the 2026-09-24 advisor direction: instead of pairwise offline DPO,
give each task's group of same-policy rollouts a group-relative advantage
(standard GRPO normalization) and let a weighted trajectory-level loss push
the policy toward its own better attempts. This is the data-preparation half
only -- it needs no GPU and reuses traj/export_sft.py's message-building and
traj/preference.py's environment-anchored rank_score (same reward the DPO
side already trusts, so both training paths are judged by the same yardstick).
The GPU-side trainer (weighted CE loss over these messages, LoRA on the
target small model) is a separate script -- see traj/scripts/grpo_lora.py.

Reward = preference.rank_score(trajectory): tier dominates (solved >
crashed_unclear_fix > no_crash > degenerate), a within-tier bonus rewards
reaching write_poc/submit over stalling. Advantage is computed PER TASK GROUP
(all rollouts collected for one task_id under one policy), matching GRPO's
"group" = one prompt, K samples -- a task here plays the role of a prompt.

A group with zero reward variance (every rollout landed in the same tier with
the same within-tier bonus, e.g. all no_crash after an identical number of
steps) gives every member advantage 0 -- no gradient, correctly: there is
nothing in that group for the policy to learn to prefer. Such groups are kept
in the output (advantage=0 rows) rather than silently dropped, so the training
script can see coverage was attempted and decide whether to filter them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from traj.agent import _SYSTEM
from traj.batch_rollout import gen_task
from traj.env import CyberGymEnv
from traj.export_sft import canonical, load, real_task, to_messages  # noqa: F401
from traj.preference import rank_score
from traj.schemas import Trajectory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/traj_batch/out",
                    help="dir of per-rollout trajectory JSONs")
    ap.add_argument("--dest", default="/data/traj_batch/grpo",
                    help="where to write train.jsonl")
    ap.add_argument("--min-group-size", type=int, default=4,
                    help="drop task groups with fewer rollouts than this -- "
                         "GRPO's mean/std baseline is unstable on tiny groups")
    ap.add_argument("--model-id", default=None,
                    help="if set, keep only rollouts from this policy's own "
                         "id (t.model_id) -- GRPO advantage must be computed "
                         "within ONE policy's samples, never mixed across "
                         "models, or the group mean stops meaning anything")
    args = ap.parse_args()

    out_dir, dest = Path(args.out_dir), Path(args.dest)

    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    seen = Counter()
    for t in load(out_dir):
        seen["total"] += 1
        if args.model_id and t.model_id != args.model_id:
            continue
        # Same gap export_dpo.py had (fixed 2026-10-08): a trajectory that
        # errored out after one real tool call still has non-empty
        # raw_output, but rank_score() has no "exclude this" path -- it would
        # fall back to the worst-tier base and quietly count as a real low
        # reward inside this task's GRPO group mean/std, not as missing data.
        if not t.is_scorable():
            continue
        if not any(s.raw_output.strip() for s in t.steps):
            continue
        by_task[real_task(t.task_id)].append(t)
        seen["kept"] += 1

    groups = {task: trajs for task, trajs in by_task.items()
              if len(trajs) >= args.min_group_size}
    dropped_small = len(by_task) - len(groups)
    if not groups:
        print(f"no task group reached --min-group-size={args.min_group_size} "
              f"(seen {len(by_task)} tasks total, largest="
              f"{max((len(v) for v in by_task.values()), default=0)})")
        return

    # reconstruct each task's prompt once (mirrors export_sft.py)
    system = None
    prompts: dict[str, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="grpo_export_"))
    try:
        for task in sorted(groups):
            work = tmp / task.replace(":", "-")
            gen_task(task, work)
            env = CyberGymEnv(str(work))
            prompts[task] = env.task().prompt
            if system is None:
                tools = "\n".join(f"- {n}: {d}" for n, d in env.tools().items())
                system = _SYSTEM.format(tools=tools)
            shutil.rmtree(work, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    dest.mkdir(parents=True, exist_ok=True)
    n_written, n_zero_var, tier_counts = 0, 0, Counter()
    with (dest / "train.jsonl").open("w", encoding="utf-8") as f:
        for task, trajs in sorted(groups.items()):
            rewards = [rank_score(t) for t in trajs]
            mean_r = statistics.fmean(rewards)
            std_r = statistics.pstdev(rewards)
            if std_r < 1e-9:
                n_zero_var += 1
            for t, r in zip(trajs, rewards):
                adv = 0.0 if std_r < 1e-9 else (r - mean_r) / std_r
                rec = {
                    "messages": to_messages(t, system, prompts[task]),
                    "reward": r,
                    "advantage": adv,
                    "meta": {"task": task, "model": t.model_id,
                             "tier": t.tier, "traj_id": t.id,
                             "group_size": len(trajs)},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
                tier_counts[t.tier or "unknown"] += 1

    print(f"scanned {seen['total']} trajectories, {seen['kept']} usable")
    print(f"task groups: {len(groups)} kept, {dropped_small} dropped "
          f"(< {args.min_group_size} rollouts)")
    print(f"zero-variance groups (advantage=0 for the whole task): "
          f"{n_zero_var}/{len(groups)}")
    print(f"rows written: {n_written}")
    print(f"tier breakdown: {dict(tier_counts)}")
    print(f"-> {dest}/train.jsonl")


if __name__ == "__main__":
    main()
