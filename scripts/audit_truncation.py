"""Audit truncation fidelity of a DPO train.jsonl: OLD trainer window vs NEW.

OLD = the window currently live in gpu_trainer_backup/scripts/dpo_lora.py
(`_cap_pair` with `content[:share-40]`), which hard-cuts an assistant message --
it can slice a chosen trajectory's write_poc/submit action JSON in half.
NEW = traj.agent._fit_messages, which preserves the assistant action JSON intact
(replacing it with an omission note only if it alone overflows).

For each chosen, we take the decisive assistant turns (action in write_poc/submit)
and report how many survive VERBATIM after each window -- i.e. were NOT cut.
Char-level (which is what both windows operate on). Run from repo root, cyber env:
  python scripts/audit_truncation.py /data/traj_batch/dpo_strict_2026_10_12/train.jsonl
"""
import json
import sys

from traj.agent import _fit_messages as fit_new

_CTX_HEAD_CAP, _CTX_TAIL = 1500, 16000
DECISIVE = {"write_poc", "submit"}


def _cap_pair_old(pair, budget):
    sizes = [len(m["content"]) for m in pair]
    total = sum(sizes)
    if total <= budget:
        return pair
    out = []
    for m, size in zip(pair, sizes):
        share = max(200, budget * size // total)
        if size > share:
            m = dict(m)
            m["content"] = m["content"][:share - 40] + "\n[truncated: too large for context]"
        out.append(m)
    return out


def fit_old(messages):
    if sum(len(m["content"]) for m in messages) <= _CTX_HEAD_CAP + _CTX_TAIL + 200:
        return messages
    head = [dict(messages[0]), dict(messages[1])]
    if len(head[1]["content"]) > _CTX_HEAD_CAP:
        head[1]["content"] = head[1]["content"][:_CTX_HEAD_CAP] + "\n[task truncated to fit context]"
    body = messages[2:]
    kept, total = [], 0
    for i in range(len(body) - 2, -1, -2):
        pair = [dict(m) for m in body[i:i + 2]]
        if not kept:
            pair = _cap_pair_old(pair, _CTX_TAIL)
            kept, total = pair, sum(len(m["content"]) for m in pair)
            continue
        size = sum(len(m["content"]) for m in pair)
        if total + size > _CTX_TAIL:
            break
        kept = pair + kept
        total += size
    task = dict(head[1])
    task["content"] += "\n[earlier steps omitted to fit the context window]"
    return [head[0], task] + kept


def action_of(content):
    try:
        return json.loads(content).get("action", "")
    except Exception:
        return None  # unparseable


def audit(fit, rows):
    intact = cut = 0
    for r in rows:
        decisive = [m["content"] for m in r["chosen"]
                    if m["role"] == "assistant" and action_of(m["content"]) in DECISIVE]
        if not decisive:
            continue
        kept = {m["content"] for m in fit(list(r["prompt"]) + list(r["chosen"]))}
        if all(d in kept for d in decisive):
            intact += 1
        else:
            cut += 1
    return intact, cut


def main():
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    n = sum(1 for r in rows
            if any(m["role"] == "assistant" and action_of(m["content"]) in DECISIVE
                   for m in r["chosen"]))
    print(f"pairs with a decisive chosen action: {n}/{len(rows)}")
    for name, fit in (("OLD (live trainer dpo_lora)", fit_old),
                      ("NEW (agent._fit_messages)  ", fit_new)):
        i, c = audit(fit, rows)
        print(f"  {name}: decisive action intact {i}, CUT/lost {c}")


if __name__ == "__main__":
    main()
