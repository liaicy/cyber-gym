"""Export collected rollout trajectories to SFT training data.

Reads the per-rollout JSON files under an out/ dir (each a serialized
schemas.Trajectory) and emits multi-turn chat examples for cold-start SFT of
the target small model.

Format: OpenAI-style {"messages": [...]} JSONL, one line per kept trajectory:
    system  -> the student's system prompt (agent._SYSTEM, with the tool list)
    user    -> "TASK:\n<task prompt>\n"
    assistant / user / assistant / ...  -> the recorded think+act turns
                (assistant = the model's raw JSON output that turn) interleaved
                with the environment's observations (user turns).

Why multi-turn and not the flat "whole transcript in one user prompt" the
current harness uses: this is the format every SFT framework and the GRPO
stage consume natively. It does mean the inference harness (agent.rollout,
single-shot complete()) must move to a real multi-turn chat before eval/GRPO,
or train/inference will be skewed -- that harness change is a known TODO and
is required for GRPO regardless.

Filtering: cold-start wants clean positive traces, so the default keeps only
`solved` trajectories (raise with --min-tier). `degenerate` is never exported.
A per-task cap stops an easy task (arvo:3938 alone has ~24 solves) from
swamping the set, and the train/val split is by TASK so no task leaks across.

Run from /data/projects/exam-society with the conda env; gen_task reads the
local CyberGym data to reconstruct each task's prompt (no API keys needed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from traj.agent import _SYSTEM
from traj.batch_rollout import gen_task
from traj.env import CyberGymEnv
from traj.preference import rank_score
from traj.schemas import TIER_ORDER, Trajectory

TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}  # 0 = best (solved)


def real_task(task_id: str) -> str:
    """arvo-10628__kimi__r1__hash -> arvo:10628 (the CyberGym task id)."""
    head = task_id.split("__")[0]
    for src in ("oss-fuzz", "arvo"):
        if head.startswith(src + "-"):
            return src + ":" + head[len(src) + 1:]
    return head


def load(out_dir: Path):
    for fn in sorted(out_dir.glob("*.json")):
        try:
            yield Trajectory.model_validate_json(fn.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip unreadable {fn.name}: {e}")


def canonical(s) -> str:
    """Clean JSON target rebuilt from the PARSED step fields, not raw_output.

    The raw text strong solvers emitted is often not clean single-JSON (prose or
    a <think> block wrapped around it); ~17% of solved-trajectory turns don't
    parse. Training the 8B student to reproduce that raw text taught it to emit
    unparseable/empty turns and loop. The env already parsed each turn into
    (thought, action, action_input) -- emit THAT canonical form so every target
    is valid and matches exactly what the harness parses back at inference."""
    return json.dumps({"thought": s.thought, "action": s.action,
                       "action_input": s.action_input}, ensure_ascii=False)


def to_messages(t: Trajectory, system: str, task_prompt: str) -> list[dict]:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"TASK:\n{task_prompt}\n"},
    ]
    steps = [s for s in t.steps if s.action]         # keep only real parsed actions
    for i, s in enumerate(steps):
        msgs.append({"role": "assistant", "content": canonical(s)})
        if i < len(steps) - 1:                       # no trailing obs after the last act
            msgs.append({"role": "user", "content": f"OBSERVATION:\n{s.observation}"})
    return msgs


def in_val(task: str, frac: float) -> bool:
    if frac <= 0:
        return False
    h = int(hashlib.sha1(task.encode()).hexdigest(), 16) % 1000
    return h < frac * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/data/traj_batch/out",
                    help="dir of per-rollout trajectory JSONs")
    ap.add_argument("--dest", default="/data/traj_batch/sft",
                    help="where to write train.jsonl / val.jsonl")
    ap.add_argument("--min-tier", default="solved", choices=list(TIER_ORDER),
                    help="lowest tier to include (default: solved only)")
    ap.add_argument("--per-task-cap", type=int, default=8,
                    help="max trajectories kept per task (best by rank_score)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of TASKS held out for validation")
    args = ap.parse_args()

    out_dir, dest = Path(args.out_dir), Path(args.dest)
    max_rank = TIER_RANK[args.min_tier]

    # group kept trajectories by real task
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    seen = Counter()
    for t in load(out_dir):
        seen["total"] += 1
        tier = t.tier or (TIER_ORDER[0] if t.success else TIER_ORDER[-1])
        if TIER_RANK.get(tier, len(TIER_ORDER)) > max_rank:
            continue
        if not any(s.raw_output.strip() for s in t.steps):
            continue
        by_task[real_task(t.task_id)].append(t)
        seen["kept"] += 1

    if not by_task:
        print(f"no trajectories at tier <= {args.min_tier} in {out_dir}")
        return

    # reconstruct each task's prompt once (and grab the static tool list once)
    system = None
    prompts: dict[str, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="sft_export_"))
    try:
        for task in sorted(by_task):
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
    counts = Counter()
    tier_counts: Counter = Counter()
    with (dest / "train.jsonl").open("w", encoding="utf-8") as ftr, \
         (dest / "val.jsonl").open("w", encoding="utf-8") as fva:
        for task, trajs in sorted(by_task.items()):
            trajs.sort(key=rank_score, reverse=True)
            kept = trajs[: args.per_task_cap]
            sink = fva if in_val(task, args.val_frac) else ftr
            split = "val" if sink is fva else "train"
            for t in kept:
                rec = {
                    "messages": to_messages(t, system, prompts[task]),
                    "meta": {"task": task, "model": t.model_id,
                             "tier": t.tier, "traj_id": t.id},
                }
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[split] += 1
                tier_counts[t.tier or "solved"] += 1

    print(f"scanned {seen['total']} trajectories, {seen['kept']} at tier "
          f"<= {args.min_tier}")
    print(f"tasks: {len(by_task)}  (val tasks: "
          f"{sum(1 for k in by_task if in_val(k, args.val_frac))})")
    print(f"examples written: train={counts['train']} val={counts['val']}")
    print(f"tier breakdown: {dict(tier_counts)}")
    print(f"-> {dest}/train.jsonl , {dest}/val.jsonl")


if __name__ == "__main__":
    main()
