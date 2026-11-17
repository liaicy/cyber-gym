#!/usr/bin/env python3
"""Failure-sample utilization (ref: Alibaba XekRung article, 2026-09-22).

Turn FAILED rollouts into "failure-reason -> corrected-action" DPO pairs, for
tasks/situations where the model only ever looped or stalled and so has no
self-generated positive to pair against. We target the clearest, most reliable
failure mode -- LOOPING: the agent repeats the same action >= N times, ignoring
observations, and never progresses (the mode-collapse behavior DPO already helps
with). At the point the loop starts we build a single-decision contrastive pair:

  prompt   = [system, TASK] + the real context up to the loop point
  rejected = the stuck/looping action (what the model actually did)
  chosen   = a teacher's corrected next action (one JSON action that progresses)

Output is the same schema export_dpo.py emits, so dpo_lora.py trains on it
unchanged. This runs OFFLINE on the env machine (no GPU, no CyberGym server):
teacher is a gateway model, task text is read from local description.txt.

Run:
  set -a; . ./.env; set +a
  CYBERGYM_SRC=/root/cybergym-src PYTHONPATH=. \
  /data/conda/envs/cybergym-py312/bin/python -m traj.failure_samples \
    --in gpu_eval_pack/out --out /data/traj_batch/failure_pairs.jsonl \
    --model glm-5.3 --loop-thresh 3 --max 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))
sys.modules["anthropic"].Anthropic = object  # type: ignore

from examsociety.llm import LLMClient, LLMError, ModelRef  # noqa: E402
from traj.agent import _SYSTEM, _parse_action  # noqa: E402
from traj.chat import chat  # noqa: E402
from traj.env import CyberGymEnv  # noqa: E402

CYBERGYM_SRC = os.environ.get("CYBERGYM_SRC", "/root/cybergym-src")
DATA_DIR = Path(CYBERGYM_SRC) / "cybergym_data" / "data"
_GW = "https://yyt.s-flag.com/v1"

TOOLS = "\n".join(f"- {n}: {d}" for n, d in CyberGymEnv._TOOLS.items())
SYSTEM = _SYSTEM.format(tools=TOOLS)


def canon(thought: str, action: str, ai: dict) -> str:
    return json.dumps({"thought": thought, "action": action, "action_input": ai},
                      ensure_ascii=False)


def norm_task(task_id: str) -> str:
    """Rollout task_ids carry per-run suffixes (arvo-10013__deepseek__r1__ab12);
    normalize to the bare task (arvo-10013 / oss-fuzz-370689421)."""
    return (task_id or "").split("__")[0]


def task_prompt(task_id: str) -> str | None:
    """task_id is the work-dir form (arvo-509 / oss-fuzz-370689421)."""
    tid = norm_task(task_id)
    typ, num = None, None
    for t in ("oss-fuzz", "arvo"):
        if tid.startswith(t + "-"):
            typ, num = t, tid[len(t) + 1:]
            break
    if typ is None and "-" in tid:
        typ, num = tid.split("-", 1)
    d = DATA_DIR / typ / num / "description.txt"
    if not d.exists():
        return None
    return d.read_text(encoding="utf-8", errors="replace")


def find_loop(steps: list[dict], thresh: int) -> int | None:
    """First index i whose (action, action_input) has already appeared thresh-1
    times among steps[:i] with the same signature -> i is where the loop is
    plainly established. Returns the index of that offending step."""
    seen: dict[str, int] = {}
    for i, s in enumerate(steps):
        act = s.get("action") or ""
        if not act:
            continue
        sig = act + "|" + json.dumps(s.get("action_input") or {}, sort_keys=True, ensure_ascii=False)
        seen[sig] = seen.get(sig, 0) + 1
        if seen[sig] >= thresh:
            return i
    return None


def build_prefix(task_txt: str, steps: list[dict], upto: int) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TASK:\n{task_txt}\n"}]
    for s in steps[:upto]:
        assistant = s.get("raw_output") or canon(s.get("thought", ""), s.get("action", ""), s.get("action_input") or {})
        msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": f"OBSERVATION:\n{s.get('observation', '')}"})
    return msgs


_CORRECT_INSTRUCTION = (
    "You are reviewing a stuck run of the agent above. It has repeated the SAME "
    "action\n  {stuck}\n{count} times, ignoring the observations, and is making "
    "no progress (mode collapse). Diagnose in one short sentence WHY this is "
    "stuck, then give the single BEST next action that actually makes progress "
    "toward locating the vulnerability and writing a crashing PoC. Use ONLY the "
    "tools listed in the system prompt. Reply as JSON only:\n"
    '{{"thought": "<one-sentence failure reason + plan>", "action": "<tool>", '
    '"action_input": {{...}}}}'
)


def teacher_correction(llm: LLMClient, ref: ModelRef, prefix: list[dict],
                       stuck_action: str, count: int) -> tuple[str, str, dict] | None:
    msgs = prefix + [{"role": "user",
                      "content": _CORRECT_INSTRUCTION.format(stuck=stuck_action, count=count)}]
    try:
        out, _ = chat(llm, ref, msgs, max_tokens=1200)
    except LLMError:
        return None
    thought, action, ai = _parse_action(out)
    if not action or action not in CyberGymEnv._TOOLS:
        return None
    return thought, action, ai


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="comma-separated dirs of rollout JSONs")
    ap.add_argument("--out", required=True, help="output DPO-format jsonl")
    ap.add_argument("--exclude-tasks", default="", help="comma list or @file of normalized task ids to skip (keep eval tasks OUT of training data)")
    ap.add_argument("--per-task-cap", type=int, default=0, help="max pairs per normalized task (0 = no cap)")
    ap.add_argument("--model", default="glm-5.3", help="teacher seat id on the gateway")
    ap.add_argument("--model-name", default=None, help="provider model name (default = --model)")
    ap.add_argument("--base-url", default=_GW)
    ap.add_argument("--api-key-env", default="SFLAG_KEY")
    ap.add_argument("--family", default="glm")
    ap.add_argument("--loop-thresh", type=int, default=3, help="repeat count that counts as a loop")
    ap.add_argument("--min-prefix", type=int, default=1, help="require at least this many context steps before the loop point")
    ap.add_argument("--max", type=int, default=0, help="cap number of pairs (0 = all)")
    ap.add_argument("--fail-tiers", default="degenerate,no_crash", help="which tiers to mine")
    args = ap.parse_args()

    # thinking-off per family (teacher reasoning must land in the JSON, see memory)
    extra: dict = {}
    if args.family == "glm":
        extra = {"thinking": {"type": "disabled"}}
    elif args.family == "deepseek":
        extra = {"thinking": {"type": "disabled"}}
    ref = ModelRef(id=args.model, model=args.model_name or args.model, api="openai",
                   base_url=args.base_url, api_key_env=args.api_key_env,
                   family=args.family, extra_body=extra)
    llm = LLMClient(max_calls=0)
    fail_tiers = set(t.strip() for t in args.fail_tiers.split(","))

    # exclusion set (normalized) -- keep eval tasks out of training data
    excl_raw = args.exclude_tasks
    if excl_raw.startswith("@"):
        excl_raw = Path(excl_raw[1:]).read_text().replace("\n", ",")
    exclude = {norm_task(t.strip().replace(":", "-")) for t in excl_raw.split(",") if t.strip()}

    files = []
    for d in args.inp.split(","):
        files += sorted(Path(d.strip()).glob("*.json"))
    out_f = open(args.out, "w", encoding="utf-8")
    n_pairs = n_scanned = n_loop = n_excl = 0
    per_task: dict = {}
    for f in files:
        n_scanned += 1
        rec = json.loads(f.read_text())
        task_id = rec.get("task_id") or ""
        nt = norm_task(task_id)
        if nt in exclude:
            n_excl += 1
            continue
        if args.per_task_cap and per_task.get(nt, 0) >= args.per_task_cap:
            continue
        tier = (rec.get("verdict") or {}).get("tier") or rec.get("tier") or ""
        if tier not in fail_tiers:
            continue
        steps = rec.get("steps") or []
        i = find_loop(steps, args.loop_thresh)
        if i is None or i < args.min_prefix:
            continue
        n_loop += 1
        txt = task_prompt(task_id)
        if not txt:
            print(f"  skip {f.name}: no description for {task_id}", flush=True)
            continue
        stuck = steps[i]
        stuck_action = canon(stuck.get("thought", ""), stuck.get("action", ""), stuck.get("action_input") or {})
        prefix = build_prefix(txt, steps, i)
        corr = teacher_correction(llm, ref, prefix, stuck_action, args.loop_thresh)
        if corr is None:
            print(f"  skip {f.name}: teacher gave no usable action", flush=True)
            continue
        c_thought, c_action, c_ai = corr
        chosen_msg = canon(c_thought, c_action, c_ai)
        out = {
            "prompt": prefix,
            "chosen": [{"role": "assistant", "content": chosen_msg}],
            "rejected": [{"role": "assistant", "content": stuck_action}],
            "meta": {"task": task_id, "failure_type": "loop",
                     "loop_action": stuck.get("action"), "loop_thresh": args.loop_thresh,
                     "teacher": args.model, "source": f.name,
                     "rejected_tier": tier, "synthetic_chosen": True},
        }
        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_f.flush()
        n_pairs += 1
        per_task[nt] = per_task.get(nt, 0) + 1
        print(f"  pair {n_pairs}: {f.name} loop={stuck.get('action')} -> fix={c_action}", flush=True)
        if args.max and n_pairs >= args.max:
            break
    out_f.close()
    print(f"\nscanned={n_scanned} excluded={n_excl} loop_failures={n_loop} "
          f"pairs_written={n_pairs} distinct_tasks={len(per_task)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
