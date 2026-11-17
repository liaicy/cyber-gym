"""Export trajectory-level DPO pairs from collected rollouts.

Method A (2026-09, after SFT cold-start was found to mode-collapse the 8B, see
docs/FINDINGS-2026-09-sft-collapse.md): don't imitate teacher tokens; instead
give the base model preference pairs -- a better trajectory (chosen) vs a worse
one (rejected) ON THE SAME TASK -- so DPO teaches "make progress / don't loop"
without copying token style.

Same-model by construction: only trajectories whose id matches --model (default
qwen3-8b, the architecture we serve as the DPO base) are loaded, so chosen and
rejected come from the same policy -- the pair isolates "solved vs not" rather
than "which model's style".

--require-chosen-solved turns on STRICT mode (the main-experiment recipe the
external review asked for): chosen MUST be tier=solved, rejected MUST be a valid
behavioral failure (no_crash/degenerate; crashed_unclear_fix is excluded on both
sides -- it half-crashed, neither a clean success nor a clean failure). rejected
prefers no_crash (wrote a PoC, didn't trigger) over degenerate, and among those
prefers the SHORTEST continuation -- a short rejected is less likely to be
truncated at train time, and a short no_crash is still a valid failure.

--out-dir takes several dirs (merges batches). --split-manifest keeps dev/test
roles OUT and the pool is asserted leak-free before export.

Coarse (trajectory-level) on purpose -- PLAN-trajectory.md section 8. The two
trajectories in a pair diverge, so the only shared prefix is system+TASK;
chosen/rejected are each the full canonical multi-turn continuation. The GPU-side
trainer applies the chat template + the same sliding window inference uses.

Emits JSONL, one line per pair:
  {"prompt": [system, task_user],
   "chosen": [assistant, obs_user, assistant, ...],
   "rejected": [...],
   "meta": {task, chosen_tier, rejected_tier, crosses_tier, chosen_id, ...}}

Run from repo root with the cyber conda env; gen_task rebuilds each task prompt
from local CyberGym data (no API keys needed).
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from traj.agent import _SYSTEM, _fit_messages
from traj.batch_rollout import gen_task
from traj.env import CyberGymEnv
from traj.preference import rank_score
from traj.schemas import (
    TIER_DEGENERATE, TIER_NO_CRASH, TIER_ORDER, TIER_SOLVED, Trajectory,
)

# agent.py _fit_messages' inference char budget (_CTX_HEAD_CAP 1500 + _CTX_TAIL
# 16000 + 200). A CHARACTER budget, NOT a token/context-length limit -- the real
# train-time cutoff is the trainer's token max_len; this only flags how many
# trajectories get trimmed at all. Keep in sync with agent.py.
_CTX_CHARS = 17700
FAILED_TIERS = {TIER_NO_CRASH, TIER_DEGENERATE}
DECISIVE_ACTIONS = {"write_poc", "submit"}


def real_task(task_id: str) -> str:
    head = task_id.split("__")[0]
    for src in ("oss-fuzz", "arvo"):
        if head.startswith(src + "-"):
            return src + ":" + head[len(src) + 1:]
    return head


def canonical(s) -> str:
    """Clean JSON target rebuilt from parsed fields (never raw_output)."""
    return json.dumps({"thought": s.thought, "action": s.action,
                       "action_input": s.action_input}, ensure_ascii=False)


def continuation(t: Trajectory) -> list[dict]:
    """The assistant/observation turns of one trajectory (no system/task)."""
    steps = [s for s in t.steps if s.action]
    msgs: list[dict] = []
    for i, s in enumerate(steps):
        msgs.append({"role": "assistant", "content": canonical(s)})
        if i < len(steps) - 1:
            msgs.append({"role": "user", "content": f"OBSERVATION:\n{s.observation}"})
    return msgs


def load(out_dir: Path, model_sub: str):
    for fn in sorted(out_dir.glob(f"*__{model_sub}__*.json")):
        try:
            yield Trajectory.model_validate_json(fn.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {fn.name}: {e}")


def cont_chars(t: Trajectory) -> int:
    return sum(len(m["content"]) for m in continuation(t))


def _action_of(content: str) -> str:
    try:
        return json.loads(content).get("action", "")
    except Exception:
        return ""


def decisive_survives(prompt: list[dict], chosen: list[dict]) -> bool | None:
    """Char-level proxy for 'does truncation keep chosen's decisive action?'.

    Run the SAME _fit_messages the agent uses at inference over prompt+chosen;
    return whether every write_poc/submit assistant message survives intact.
    None if this chosen recorded no decisive action. This is a CHAR-level check;
    the token-level one must be run GPU-side with the real tokenizer + max_len.
    """
    decisive = [m["content"] for m in chosen
                if m["role"] == "assistant" and _action_of(m["content"]) in DECISIVE_ACTIONS]
    if not decisive:
        return None
    kept = {m["content"] for m in _fit_messages(list(prompt) + list(chosen))}
    return all(d in kept for d in decisive)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", nargs="+", default=["/data/traj_batch/out"],
                    help="one or more rollout output dirs to merge")
    ap.add_argument("--dest", default="/data/traj_batch/dpo")
    ap.add_argument("--model", default="qwen3-8b",
                    help="only pair trajectories from this model id substring")
    ap.add_argument("--pairs-per-task", type=int, default=4,
                    help="max (chosen,rejected) pairs kept per task")
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="min rank_score gap chosen-rejected (non-strict mode only)")
    ap.add_argument("--exclude-tasks", default="arvo:10628,arvo:11007",
                    help="extra held-out tasks to keep OUT of DPO training data")
    ap.add_argument("--split-manifest", default="",
                    help="candidates_split.json; tasks in --exclude-roles are kept OUT")
    ap.add_argument("--exclude-roles", default="new_dev,final_test,legacy_dev_regression",
                    help="split roles to exclude when --split-manifest is given")
    ap.add_argument("--require-chosen-solved", action="store_true",
                    help="STRICT: chosen must be tier=solved, rejected must be no_crash/degenerate")
    ap.add_argument("--cross-model", action="store_true",
                    help="ABLATION: chosen = a NON-target model's solved (teacher), rejected = the "
                         "target model's failed on the same task (loads all models; teaches style, "
                         "see PLAN section 8). Use with --require-chosen-solved.")
    ap.add_argument("--target-model", default="qwen3-8b",
                    help="student/target model id; its failures are the rejected side in --cross-model")
    args = ap.parse_args()

    excl = {t for t in args.exclude_tasks.split(",") if t}
    if args.split_manifest:
        roles = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))["roles"]
        for r in args.exclude_roles.split(","):
            r = r.strip()
            if r:
                excl.update(roles.get(r, []))
        print(f"excluding {len(excl)} tasks (split roles [{args.exclude_roles}] + --exclude-tasks)")

    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    n_skipped_unscorable = 0
    n_loaded = 0
    for out_dir in args.out_dir:
        for t in load(Path(out_dir), "*" if args.cross_model else args.model):
            rt = real_task(t.task_id)
            if rt in excl:
                continue
            # is_scorable() excludes stop_reason=="error" and TIER_INFRA_ERROR --
            # API/infra failures must not become behavioral `rejected` examples.
            if not t.is_scorable():
                n_skipped_unscorable += 1
                continue
            if any(s.action for s in t.steps):
                by_task[rt].append(t)
                n_loaded += 1
    _src = "all-model (cross)" if args.cross_model else repr(args.model)
    print(f"loaded {n_loaded} scorable {_src} trajectories over "
          f"{len(by_task)} tasks from {len(args.out_dir)} dir(s)")
    if n_skipped_unscorable:
        print(f"skipped {n_skipped_unscorable} non-scorable (errored) trajectories")
    leaked = sorted(tk for tk in by_task if tk in excl)
    assert not leaked, f"LEAK: excluded tasks present in pool: {leaked}"

    # rebuild each task prompt once (strict: only tasks that can yield a pair)
    strict = args.require_chosen_solved
    needs_prompt = sorted(by_task)
    if strict:
        needs_prompt = [tk for tk in needs_prompt
                        if any(t.tier == TIER_SOLVED for t in by_task[tk])]
    dest = Path(args.dest); dest.mkdir(parents=True, exist_ok=True)
    system = None
    prompts: dict[str, str] = {}
    tmp = Path(tempfile.mkdtemp(prefix="dpo_export_"))
    try:
        for task in needs_prompt:
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

    def total_chars(task: str, cont: list[dict]) -> int:
        head = len(system) + len(f"TASK:\n{prompts[task]}\n")
        return head + sum(len(m["content"]) for m in cont)

    n_pairs = n_cross = 0
    tier_pairs: dict[str, int] = defaultdict(int)
    tasks_with_pairs: set[str] = set()
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    chosen_decisive: list[bool] = []

    with (dest / "train.jsonl").open("w", encoding="utf-8") as f:
        for task, trajs in sorted(by_task.items()):
            if strict:
                if args.cross_model:
                    # chosen = teacher (non-target) solved; rejected = target's failed
                    solved = [t for t in trajs if t.tier == TIER_SOLVED and t.model_id != args.target_model]
                    failed_pool = [t for t in trajs if t.model_id == args.target_model and t.tier in FAILED_TIERS]
                else:
                    solved = [t for t in trajs if t.tier == TIER_SOLVED]
                    failed_pool = [t for t in trajs if t.tier in FAILED_TIERS]
                # valid-failed (no_crash) first, then SHORTEST continuation first
                # -> less likely to be truncated at train time; short no_crash is
                # still a real "wrote a PoC, didn't crash" failure.
                failed = sorted(failed_pool,
                                key=lambda t: (t.tier != TIER_NO_CRASH, cont_chars(t)))
                # failed outer, solved inner -> rotate solved so one success isn't
                # overused; cap per task.
                picks = [(hi, lo) for lo in failed for hi in solved][: args.pairs_per_task]
            else:
                scored = sorted(trajs, key=rank_score, reverse=True)
                cands = [(rank_score(hi) - rank_score(lo), hi, lo)
                         for hi in scored for lo in scored
                         if rank_score(hi) - rank_score(lo) >= args.min_gap]
                cands.sort(key=lambda x: -x[0])
                picks = [(hi, lo) for _, hi, lo in cands[: args.pairs_per_task]]

            if not picks:
                continue
            prompt_msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"TASK:\n{prompts[task]}\n"},
            ]
            for hi, lo in picks:
                ct = hi.tier or (TIER_ORDER[0] if hi.success else TIER_ORDER[-1])
                rt_ = lo.tier or (TIER_ORDER[0] if lo.success else TIER_ORDER[-1])
                chosen = continuation(hi)
                rejected = continuation(lo)
                if not chosen or not rejected:
                    continue
                rec = {
                    "prompt": prompt_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {"task": task, "chosen_tier": ct, "rejected_tier": rt_,
                             "crosses_tier": ct != rt_,
                             "chosen_id": hi.id, "rejected_id": lo.id,
                             "chosen_model": hi.model_id, "rejected_model": lo.model_id},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
                n_cross += int(ct != rt_)
                tier_pairs[f"{ct}>{rt_}"] += 1
                tasks_with_pairs.add(task)
                chosen_lens.append(total_chars(task, chosen))
                rejected_lens.append(total_chars(task, rejected))
                surv = decisive_survives(prompt_msgs, chosen)
                if surv is not None:
                    chosen_decisive.append(surv)

    print(f"\n{'STRICT ' if strict else ''}pairs written: {n_pairs}  "
          f"over {len(tasks_with_pairs)} tasks  (cross-tier: {n_cross})")
    print(f"tier contrasts: {dict(tier_pairs)}")
    if chosen_lens:
        print("DATA HEALTH (CHARS, not tokens -- real tokenizer must be checked GPU-side):")
        for name, xs in (("chosen  ", chosen_lens), ("rejected", rejected_lens)):
            over = sum(1 for x in xs if x > _CTX_CHARS)
            print(f"  {name}: min={min(xs)} median={int(statistics.median(xs))} "
                  f"max={max(xs)} | >{_CTX_CHARS}(~ctx budget): {over}/{len(xs)}")
        if chosen_decisive:
            kept = sum(chosen_decisive)
            print(f"  chosen decisive action (write_poc/submit) survives CHAR-level "
                  f"_fit_messages: {kept}/{len(chosen_decisive)} "
                  f"(token-level must be re-checked GPU-side)")
        print("  (prompt [system+TASK] is shared by chosen/rejected by construction "
              "-> two sides have an identical prefix.)")
    print(f"-> {dest}/train.jsonl")


if __name__ == "__main__":
    main()
