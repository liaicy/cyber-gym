"""LoRA SFT of Qwen3-8B on exported CyberGym trajectories (train.jsonl from traj/export_sft.py).

Each multi-turn trajectory {"messages": [...]} is split into per-assistant-turn
prompt/completion examples so the loss is on the model's own turns only (TRL's
prompt-completion format does completion-only loss natively; Qwen3's chat
template has no {% generation %} markers so assistant_only_loss can't be used).

Usage (inside venv, 1x A800):
  python sft_lora.py --data data/train.jsonl --out runs/sft_v1 \
      --exclude-tasks arvo:10628,arvo:11007
"""
import argparse, json, random, statistics
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Same sliding window the inference harness applies (traj/agent.py:_fit_messages,
# copied verbatim so train and rollout see identical prompts): keep system +
# TASK and the most recent (assistant, observation) pairs within _CTX_TAIL chars.
#
# 2026-10-08 resync: this copy had drifted from traj/agent.py -- the previous
# version only checked each KEPT PAIR's own size against _CTX_TAIL, which
# does not bound the pair's own total (an 8000+10000 char pair is under a
# 16000 per-message cap but sums to 18000) nor the head (TASK was never
# truncated, so a verbose level2 crash report could pass through whole).
# Re-copied verbatim from traj/agent.py after the round-2 external review
# caught both with concrete counterexamples. Any prior SFT run (sft_v1,
# sft_v2, selfdistill_sft_v1) built its training examples with the OLD
# buggy version -- this fix does not retroactively correct those.
_CTX_HEAD_CAP = 1500
_CTX_TAIL = 16000


def _cap_pair(pair, budget):
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


def _fit_messages(messages):
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
            pair = _cap_pair(pair, _CTX_TAIL)
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


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-8B")
ap.add_argument("--data", default="data/train.jsonl")
ap.add_argument("--out", default="runs/sft_v1")
ap.add_argument("--exclude-tasks", default="", help="comma list of meta.task held out for eval")
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)

tok = AutoTokenizer.from_pretrained(args.model)
excl = {t for t in args.exclude_tasks.split(",") if t}

rows, n_traj, dropped = [], 0, 0
for line in Path(args.data).read_text().splitlines():
    rec = json.loads(line)
    if rec.get("meta", {}).get("task") in excl:
        dropped += 1
        continue
    n_traj += 1
    msgs = rec["messages"]
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(_fit_messages(msgs[:i]), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": prompt, "completion": m["content"] + "<|im_end|>\n",
                     "task": rec["meta"]["task"], "turn": i})

lens = [len(tok(r["prompt"] + r["completion"]).input_ids) for r in rows]
print(f"trajectories used={n_traj} held-out dropped={dropped} examples={len(rows)}")
print(f"tokens/example: mean={statistics.mean(lens):.0f} p50={statistics.median(lens):.0f} "
      f"max={max(lens)} over_max_len={sum(l > args.max_len for l in lens)}")
rows = [r for r, l in zip(rows, lens) if l <= args.max_len]
random.shuffle(rows)
ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

cfg = SFTConfig(
    output_dir=args.out,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
    max_length=args.max_len,
    completion_only_loss=True,
    packing=False,
    report_to=["tensorboard"],
    seed=args.seed,
)
peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=peft_cfg)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(args.out + "/final")
tok.save_pretrained(args.out + "/final")
print("TRAIN_DONE", args.out)
