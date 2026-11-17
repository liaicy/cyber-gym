"""27B DPO-LoRA memory/compat smoke.

Answers the three things CLAUDE.md lists as untested for the 27B pivot:
  1. LoRA training compatibility  -> does model load + adapter attach + a real
     forward/backward step run at all on Qwen3.8-27B?
  2. bf16 memory planning         -> peak CUDA mem per GPU for (a) the production
     single-GPU path and (b) a 2-GPU device_map="auto" shard.
  3. chat-template compatibility  -> do our real DPO pairs render through the 27B
     template (incl. enable_thinking convention) without the jinja raising?

It does NOT train anything useful: a handful of real pairs, 2 optimizer steps,
then it reports peak memory and exits. The LoRA it writes is throwaway.
"""
import argparse, json, sys, traceback
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3.8-27B")
ap.add_argument("--data", default="data/dpo_train.jsonl")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--device-map", default="single", choices=["single", "auto"])
ap.add_argument("--out", default="runs/smoke_dpo_27b")
args = ap.parse_args()

NVIS = torch.cuda.device_count()
print(f"[smoke] mode=device-map:{args.device_map}  visible_gpus={NVIS}  model={args.model}", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)


def render(prompt_msgs, cont_msgs):
    # mirror dpo_lora.py: prompt = system+user with gen prompt; completion = the rest
    try:
        p = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(prompt_msgs + cont_msgs, tokenize=False,
                                       enable_thinking=False)
    except TypeError:
        # template doesn't accept enable_thinking kwarg -> retry without it
        p = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(prompt_msgs + cont_msgs, tokenize=False)
    return p, full[len(p):]


rows, dropped, render_err = [], 0, None
for line in Path(args.data).read_text().splitlines():
    if len(rows) >= args.n:
        break
    try:
        r = json.loads(line)
        p_ch, ch = render(r["prompt"], r["chosen"])
        p_rj, rj = render(r["prompt"], r["rejected"])
    except Exception as e:  # noqa
        render_err = f"{type(e).__name__}: {e}"
        break
    if len(tok(p_ch + ch).input_ids) > args.max_len or len(tok(p_rj + rj).input_ids) > args.max_len:
        dropped += 1
        continue
    rows.append({"prompt": p_ch, "chosen": ch, "rejected": rj})

if render_err:
    print(f"[smoke] TEMPLATE_RENDER_FAILED {render_err}", flush=True)
    sys.exit(3)
print(f"[smoke] TEMPLATE_OK rows_kept={len(rows)} dropped_too_long={dropped}", flush=True)
if not rows:
    print("[smoke] no usable rows under max-len; lengthen --max-len or raise --n", flush=True)
    sys.exit(4)
ds = Dataset.from_list(rows)

# No reset: max_memory_allocated reports peak since process start, which already
# covers load+train. (reset_peak_memory_stats before any CUDA context raises
# "Invalid device argument" on this torch build.)

cfg = DPOConfig(
    output_dir=args.out,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=2,
    learning_rate=5e-6,
    beta=0.1,
    bf16=True,
    gradient_checkpointing=True,
    max_length=args.max_len,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    seed=0,
)
peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")

if args.device_map == "auto":
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    after_load = {d: torch.cuda.memory_allocated(d) / 2**30 for d in range(NVIS)}
    print(f"[smoke] weights loaded (sharded) GB/gpu=" +
          " ".join(f"{d}:{after_load[d]:.1f}" for d in range(NVIS)), flush=True)
    model_arg = model
else:
    model_arg = args.model  # exactly the production call: TRL loads onto cuda:0

try:
    trainer = DPOTrainer(model=model_arg, ref_model=None, args=cfg,
                         train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    ok = True
    err = None
except torch.cuda.OutOfMemoryError as e:
    ok, err = False, f"OOM: {str(e)[:200]}"
except Exception as e:  # noqa
    ok, err = False, f"{type(e).__name__}: {str(e)[:300]}"
    traceback.print_exc()

peak = {d: torch.cuda.max_memory_allocated(d) / 2**30 for d in range(NVIS)}
print("[smoke] RESULT " + json.dumps({
    "mode": args.device_map,
    "visible_gpus": NVIS,
    "train_ok": ok,
    "error": err,
    "peak_gb_per_gpu": {str(d): round(peak[d], 1) for d in range(NVIS)},
    "peak_gb_max": round(max(peak.values()), 1),
    "rows": len(rows),
}), flush=True)
sys.exit(0 if ok else 5)
