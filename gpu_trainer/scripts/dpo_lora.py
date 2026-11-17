"""Trajectory-level DPO of Qwen3-8B on CyberGym preference pairs (export_dpo.py).

Method A: no teacher-token imitation (that mode-collapsed the 8B, see
docs/FINDINGS-2026-09-sft-collapse.md). DPO nudges the BASE model toward the
better trajectory of each (chosen, rejected) pair. LoRA; the reference model is
the same base with the adapter disabled (PEFT), so no second model is loaded.

Prompt/chosen/rejected are rendered here with the chat template and the SAME
sliding window inference uses (traj/agent.py:_fit_messages), so the DPO contrast
lands on the decisive tail: chosen ends in write_poc/submit, rejected in a loop.

Usage (venv, 1x A800):
  python dpo_lora.py --data data/dpo_train.jsonl --out runs/dpo_v1
"""
import argparse, json, statistics, os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer
import trl.trainer.dpo_trainer as _dpo_trainer_mod


def _entropy_from_logits_seqchunked(logits, chunk_size=128):
    """2026-10-30: TRL 1.13 entropy_from_logits does logits.reshape(-1, V) on the
    NON-contiguous slice outputs.logits[..., :-1, :] -> a full fp32 copy of the
    logits (13.65 GiB at chosen 7375 tok x vocab 150k, job 1096799 OOM on g53).
    Same math, chunked along the sequence dim, no copy. Metric only (detached)."""
    out = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)
    for i in range(0, logits.shape[-2], chunk_size):
        lp = torch.log_softmax(logits[..., i:i + chunk_size, :].float(), dim=-1)
        out[..., i:i + chunk_size] = -(lp.exp() * lp).sum(-1)
        del lp
    return out


_dpo_trainer_mod.entropy_from_logits = _entropy_from_logits_seqchunked

# 2026-10-12 REAL resync. The 2026-10-08 note that used to sit here claimed a
# verbatim re-copy from traj/agent.py, but the code stayed the OLD hard-cut
# version: _cap_pair sliced an assistant action JSON via content[:share-40],
# so a long chosen trajectory's write_poc/submit could be cut in half. An audit
# (scripts/audit_truncation.py on dpo_strict_2026_10_12) found 2/24 chosen
# decisive actions cut by the old window, 0/24 by the new one. Below is the
# CURRENT traj/agent.py _cap_pair + _fit_messages (preserves the action JSON).
# render() still calls fit(); fit is aliased to _fit_messages at the end.


# Qwen3-8B (32k) hit HTTP 400 at ~135k input tokens on step 8, reading a few
# large source files. Keep the system + TASK turns plus the most recent
# step turns so a 4-9B target model (the distillation target) can finish an
# episode. This is a character budget, not a tokenizer/context-length guarantee.
# The head allowance contributes to the total cap; it is not a fixed TASK cap:
# keep a complete task/crash report whenever the available space permits it.
_CTX_HEAD_CAP = 1500
_CTX_TAIL = 16000


def _cap_pair(pair: list[dict], budget: int) -> list[dict]:
    """Fit one turn (or a final assistant message) within a total char budget.

    Preserve the assistant's action JSON and spend the remaining space on the
    observation. If the assistant alone is too large, replace it with an
    explicit JSON omission note rather than cutting an action in half. These
    edits affect only the request history, never the stored raw trajectory.
    """
    budget = max(0, budget)
    if sum(len(m["content"]) for m in pair) <= budget:
        return pair
    out = [dict(m) for m in pair]
    remaining = budget
    # Allocate by priority, but retain the original message/role ordering.
    order = sorted(range(len(pair)), key=lambda i: pair[i]["role"] != "assistant")
    for i in order:
        content = pair[i]["content"]
        if len(content) > remaining:
            if pair[i]["role"] == "assistant":
                note = ('{"thought": "[assistant output omitted: too large for context]", '
                        '"action": "", "action_input": {}}')
                if len(note) <= remaining:
                    content = note
                elif len('"[assistant output omitted]"') <= remaining:
                    content = '"[assistant output omitted]"'
                else:
                    # Tiny utility-call budgets cannot contain an explanation.
                    content = '""' if remaining >= 2 else ""
            else:
                note = "\n[observation truncated for context]"
                content = (content[:remaining - len(note)] + note
                           if remaining >= len(note) else note[:remaining])
        out[i]["content"] = content
        remaining -= len(content)
    return out


def _fit_messages(messages: list[dict]) -> list[dict]:
    """Keep system/TASK and recent complete turns within 17,700 content chars.

    Count the head and every truncation/omission note in the same total budget.
    Preserve task text whenever possible, shortening the latest observation
    before its assistant action. Group from the start so a trailing assistant
    remains an assistant, rather than shifting every pair by one message.
    """
    limit = _CTX_HEAD_CAP + _CTX_TAIL + 200
    if (sum(len(m["content"]) for m in messages) <= limit
            and sum(len(m["content"]) for m in messages[2:]) <= _CTX_TAIL):
        return messages
    head = [dict(m) for m in messages[:2]]
    body = messages[2:]
    turns: list[list[dict]] = []
    i = 0
    while i < len(body):
        end = i + 1
        if (body[i]["role"] == "assistant" and end < len(body)
                and body[end]["role"] in ("user", "tool")):
            end += 1
        turns.append(body[i:end])
        i = end

    # Reserve recent feedback and, when possible, the full latest action. The
    # rest belongs to the original system/task, including long level2 reports.
    reserve = 0
    history_note = ""
    if turns:
        latest = turns[-1]
        latest_size = sum(len(m["content"]) for m in latest)
        assistant_size = sum(len(m["content"]) for m in latest if m["role"] == "assistant")
        reserve = min(latest_size, _CTX_HEAD_CAP)
        if assistant_size <= _CTX_TAIL:
            reserve = max(reserve, assistant_size)
        history_note = "\n[history shortened to fit the context window]"

    def trim(content: str, cap: int, note: str) -> str:
        if len(content) <= cap:
            return content
        return content[:cap - len(note)] + note if cap >= len(note) else note[:cap]

    head_budget = limit - reserve
    if len(head) == 1:
        head[0]["content"] = trim(head[0]["content"], head_budget,
                                  "\n[system truncated for context]")
    elif len(head) == 2:
        system, task = head[0]["content"], head[1]["content"]
        if len(system) + len(task) + len(history_note) > head_budget:
            # Usually the system remains intact. A pathological oversized
            # system must also leave space for the task and its notes.
            task_reserve = min(len(task) + len(history_note), _CTX_HEAD_CAP)
            system = trim(system, head_budget - task_reserve,
                          "\n[system truncated for context]")
            task = trim(task, head_budget - len(system) - len(history_note),
                        "\n[task truncated to fit context]")
        head[0]["content"] = system
        head[1]["content"] = task + history_note

    tail_budget = min(_CTX_TAIL, limit - sum(len(m["content"]) for m in head))
    kept: list[dict] = []
    total = 0
    for turn in reversed(turns):
        if not kept:
            kept = _cap_pair(turn, tail_budget)
            total = sum(len(m["content"]) for m in kept)
            continue
        size = sum(len(m["content"]) for m in turn)
        if total + size > tail_budget:
            break
        kept = turn + kept
        total += size
    return head + kept


fit = _fit_messages  # render() calls fit(); keep the name, use the new window


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/Qwen3-8B")
ap.add_argument("--data", default="data/dpo_train.jsonl")
ap.add_argument("--out", default="runs/dpo_v1")
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--max-prompt-len", type=int, default=6144)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ctx-tail", type=int, default=_CTX_TAIL,
                help="char budget of the fit tail window (default %(default)s). 2026-10-30: shrink so "
                     "the longest pair fits the GPU (4xA800 FSDP OOMs in backward at ~7.4k tokens)")
ap.add_argument("--grad-accum", type=int, default=8,
                help="gradient_accumulation_steps (effective batch = n_gpu x this). 2026-11-01: with 35 pairs "
                     "and 4 GPUs the default 8 gives 2 optimizer steps/epoch -> LoRA delta below bf16 resolution")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--sparse-logits", action="store_true",
                help="2026-11-04: only run lm_head on positions whose next token is in the loss mask "
                     "(transformers logits_to_keep). Same math as dense; cuts the vocab x seq_len fp32 logits "
                     "(the memory hog at long windows) to the masked fraction.")
ap.add_argument("--sparse-check", action="store_true",
                help="on the first batch also compute the dense loss (no_grad) and abort if it differs")
ap.add_argument("--assistant-only-mask", action="store_true",
                help="2026-11-03: DPO loss only on assistant-turn tokens (JSON actions). Default TRL masks nothing "
                     "inside the completion, so 72-86%% of the loss fell on environment OBSERVATION text "
                     "(docs/ANALYSIS-2026-11-03-why-27b-dpo-degrades.md). Attention/context is unchanged.")
ap.add_argument("--save-steps", type=int, default=0,
                help="2026-11-02: >0 saves a raw-LoRA checkpoint every N optimizer steps (see _RawLoraSave), "
                     "keeping only the last --save-limit; 0 = no intermediate checkpoints")
ap.add_argument("--save-limit", type=int, default=2)
ap.add_argument("--precompute-ref", action="store_true",
                help="DPOConfig.precompute_ref_log_probs: reference logps computed once before training, "
                     "no ref forward inside the train step (lower peak memory)")
ap.add_argument("--max-steps", type=int, default=-1,
                help="cap optimizer steps (smoke/debug); -1 = use --epochs")
args = ap.parse_args()
_CTX_TAIL = args.ctx_tail
print(f"[dpo] ctx_tail chars={_CTX_TAIL}")

tok = AutoTokenizer.from_pretrained(args.model)


def render(prompt_msgs, cont_msgs):
    """(prompt_str, completion_str) after windowing prompt+continuation together."""
    win = fit(prompt_msgs + cont_msgs)
    prompt_str = tok.apply_chat_template(win[:2], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(win, tokenize=False, enable_thinking=False)
    return prompt_str, full[len(prompt_str):]


rows, dropped = [], 0
for line in Path(args.data).read_text().splitlines():
    r = json.loads(line)
    p_ch, ch = render(r["prompt"], r["chosen"])
    p_rj, rj = render(r["prompt"], r["rejected"])
    # prompt is identical for chosen/rejected (window keeps [:2] verbatim)
    if len(tok(p_ch + ch).input_ids) > args.max_len or len(tok(p_rj + rj).input_ids) > args.max_len:
        dropped += 1
        continue
    rows.append({"prompt": p_ch, "chosen": ch, "rejected": rj})

lens = [len(tok(x["prompt"] + x["chosen"]).input_ids) for x in rows]
print(f"pairs kept={len(rows)} dropped(too long)={dropped}")
print(f"chosen tokens: mean={statistics.mean(lens):.0f} max={max(lens)}")
ds = Dataset.from_list(rows)


# ---- 2026-11-03 assistant-only loss mask -------------------------------------------------
# TRL 1.13 DataCollatorForPreference builds completion_mask = [0]*len(prompt_ids) + [1]*len(chosen_ids)
# and _compute_loss sums per-token logps wherever completion_mask==1 (attention_mask is separate).
# We keep attention_mask intact and zero completion_mask on every token that is not inside an
# assistant turn (role header `<|im_start|>assistant\n` -> 0, content + its `<|im_end|>` -> 1,
# user/OBSERVATION turns -> 0). Mask is derived from token ids, so it is exact w.r.t. tokenization.
from trl.trainer.dpo_trainer import DataCollatorForPreference
_IM_START = tok.convert_tokens_to_ids("<|im_start|>")
_IM_END = tok.convert_tokens_to_ids("<|im_end|>")
_ASSIST_ROLE = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids[1]
_NL = tok("\n", add_special_tokens=False).input_ids[-1]


_THINK_OPEN = tok.convert_tokens_to_ids("<think>")
_THINK_CLOSE = tok.convert_tokens_to_ids("</think>")
_NL_IDS = set(tok("\n", add_special_tokens=False).input_ids + tok("\n\n", add_special_tokens=False).input_ids)


def assistant_mask_ids(ids):
    """1 for tokens the policy generates (assistant content + closing <|im_end|>), else 0.
    The completion string starts INSIDE the first assistant turn (render() puts the generation
    prompt `<|im_start|>assistant\n` into the prompt side), so scanning starts with in_a=True.
    The template's empty `<think>\n\n</think>\n\n` block at the start of an assistant turn is
    part of the generation prompt at inference (enable_thinking=False) -> masked 0 too."""
    mask, i, n, in_a, turn_start = [], 0, len(ids), True, True
    while i < n:
        t = ids[i]
        if t == _IM_START:
            in_a = (i + 1 < n and ids[i + 1] == _ASSIST_ROLE); turn_start = True
            mask.append(0); i += 1                      # <|im_start|>
            if i < n: mask.append(0); i += 1            # role token
            if i < n and ids[i] in _NL_IDS: mask.append(0); i += 1
            continue
        if t == _IM_END:
            mask.append(1 if in_a else 0); in_a = False; turn_start = False; i += 1
            continue
        if in_a and turn_start and t == _THINK_OPEN:   # empty think block: <think> ... </think> + newlines
            while i < n and ids[i] != _THINK_CLOSE:
                mask.append(0); i += 1
            if i < n: mask.append(0); i += 1            # </think>
            while i < n and ids[i] in _NL_IDS: mask.append(0); i += 1
            turn_start = False
            continue
        turn_start = False
        mask.append(1 if in_a else 0); i += 1
    return mask


class AssistantOnlyCollator(DataCollatorForPreference):
    def torch_call(self, examples):
        out = super().torch_call(examples)
        cm = out["completion_mask"]
        n, L = len(examples), cm.shape[1]
        for i, ex in enumerate(examples):
            plen = len(ex["prompt_ids"])
            for row, ids in ((i, ex["chosen_ids"]), (n + i, ex["rejected_ids"])):
                m = torch.tensor(assistant_mask_ids(ids), dtype=cm.dtype)
                k = min(len(m), max(0, L - plen))
                if k > 0:
                    cm[row, plen:plen + k] = cm[row, plen:plen + k] * m[:k]
        out["completion_mask"] = cm
        return out


if args.assistant_only_mask:
    _fr = []
    for x in rows[:len(rows)]:
        for side in ("chosen", "rejected"):
            ids = tok(x[side], add_special_tokens=False).input_ids
            m = assistant_mask_ids(ids); _fr.append(sum(m) / max(1, len(m)))
    _ids = tok(rows[0]["chosen"], add_special_tokens=False).input_ids; _m = assistant_mask_ids(_ids)
    print(f"[dpo] ASSISTANT_ONLY_MASK on: mean assistant-token fraction of completions = {sum(_fr)/len(_fr):.3f} "
          f"(n={len(_fr)}); row0 chosen kept tokens={sum(_m)}/{len(_m)}; kept text starts: "
          f"{tok.decode([t for t, v in zip(_ids, _m) if v][:60])!r}", flush=True)
# -------------------------------------------------------------------------------------------

cfg = DPOConfig(
    output_dir=args.out,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=args.warmup,
    beta=args.beta,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=args.max_len,
    max_steps=args.max_steps,
    precompute_ref_log_probs=args.precompute_ref,
    logging_steps=2,
    save_strategy="steps" if args.save_steps > 0 else "no",
    save_steps=args.save_steps if args.save_steps > 0 else 500,
    save_total_limit=args.save_limit,
    save_only_model=True,  # no 1.2GB optimizer shards per checkpoint
    ddp_timeout=7200,  # 2026-11-05: long-window steps must not trip the 30-min NCCL collective timeout
    report_to=["tensorboard"],
    seed=args.seed,
)
peft_cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")

# ---- 2026-11-04 sparse logits: rewrite TRL's DPOTrainer._compute_loss at runtime -----------------
# Dense path: outputs.logits is (B, L, V) fp32 for every position, then sliced/masked. With the
# assistant-only mask only ~25% of positions carry loss, so we pass the needed positions as
# `logits_to_keep` (transformers computes lm_head only there) and index labels/mask the same way.
# Everything else in the function (loss types, ref forward, metrics) is TRL's own source, unchanged.
if args.sparse_logits:
    import inspect, textwrap, copy
    import trl.trainer.dpo_trainer as _dt
    _src = textwrap.dedent(inspect.getsource(_dt.DPOTrainer._compute_loss))
    _reps = [
        ("    outputs = model(**model_kwargs)\n",
         "    _scm = inputs[\"completion_mask\"][:, 1:]\n"
         "    _keep = torch.nonzero(_scm.any(dim=0)).squeeze(1)\n"
         "    if _keep.numel() == 0:\n"
         "        _keep = torch.tensor([_scm.shape[1] - 1], device=_scm.device)\n"
         "    model_kwargs[\"logits_to_keep\"] = _keep\n"
         "    outputs = _raw_fwd(model, model_kwargs)\n"
         "    outputs.logits = outputs.logits.to(torch.bfloat16)\n"
         "    if not getattr(self, '_sparse_diag_done', False):\n"
         "        self._sparse_diag_done = True\n"
         "        print(f\"[dpo] SPARSE_DIAG rank={__import__('os').environ.get('RANK','0')} seq_len={inputs['input_ids'].shape[1]} keep={_keep.numel()} logits dtype={outputs.logits.dtype} shape={tuple(outputs.logits.shape)} cuda_alloc={torch.cuda.memory_allocated()/2**30:.1f}GiB\", flush=True)\n"),
        ("                ref_outputs = self.model(**ref_model_kwargs)\n",
         "                ref_outputs = _raw_fwd(self.model, ref_model_kwargs)\n"
         "                ref_outputs.logits = ref_outputs.logits.to(torch.bfloat16)\n"),
        ("                ref_outputs = self.ref_model(**ref_model_kwargs)\n",
         "                ref_outputs = _raw_fwd(self.ref_model, ref_model_kwargs)\n"
         "                ref_outputs.logits = ref_outputs.logits.to(torch.bfloat16)\n"),
        ("    per_token_logps = selective_log_softmax(shift_logits, shift_labels)\n",
         "    per_token_logps = _chunked_logps(shift_logits, shift_labels)\n"),
        ("        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)\n",
         "        ref_per_token_logps = _chunked_logps(ref_shift_logits, shift_labels)\n"),
        ("    shift_logits = outputs.logits[..., :-1, :]\n", "    shift_logits = outputs.logits\n"),
        ("    shift_labels = input_ids[..., 1:]\n", "    shift_labels = input_ids[:, _keep + 1]\n"),
        ("    shift_completion_mask = completion_mask[..., 1:]\n", "    shift_completion_mask = completion_mask[:, _keep + 1]\n"),
        ("        ref_shift_logits = ref_outputs.logits[..., :-1, :]\n", "        ref_shift_logits = ref_outputs.logits\n"),
    ]
    for old, new in _reps:
        assert _src.count(old) == 1, f"sparse patch anchor not found exactly once: {old!r}"
        _src = _src.replace(old, new)
    _src = _src.replace("def _compute_loss(", "def _sparse_compute_loss(", 1)
    _ns = vars(_dt)

    def _raw_fwd(m, kw):
        """Call the model forward WITHOUT accelerate's ConvertOutputsToFp32 wrapper (it .float()s the
        whole logits tensor: 2 x keep x 248k vocab x 4B = 18 GiB at an 11k window -> the OOM in job
        1170330). Falls back to the ORIGINAL pre-wrap forward (saved by accelerate as
        `_original_forward` on every model it native_amp-wraps, regardless of which of its two
        internal wrap branches ran) when `m.forward` is not itself a bound method -- the
        `.__func__.model_forward` path silently no-ops in that branch (2026-11-11, job
        1173614/1171737: logits stayed fp32 despite the autocast wrap below, meaning this exact
        fallback was firing)."""
        f = m.forward
        inner = getattr(getattr(f, "__func__", None), "model_forward", None)
        inner_is_bound = False
        if inner is None:
            inner = getattr(m, "_original_forward", None)
            # `_original_forward` is `model.forward` captured BEFORE accelerate's native_amp wrap,
            # i.e. already a bound method (self=m baked in) -- unlike the `.__func__.model_forward`
            # path above, which is an unbound function. Calling it as inner(m, **kw) double-passes
            # `m`, colliding with `input_ids` in kw (job 1197183: "got multiple values for argument
            # 'input_ids'"). Detect boundness instead of assuming a convention.
            inner_is_bound = inner is not None and getattr(inner, "__self__", None) is not None
        if not getattr(m, "_rawfwd_diag_done", False):
            m._rawfwd_diag_done = True
            print(f"[dpo] RAWFWD_DIAG forward_type={type(f).__name__} has_func={hasattr(f,'__func__')} "
                  f"via_model_forward_attr={getattr(getattr(f,'__func__',None),'model_forward',None) is not None} "
                  f"via_original_forward={getattr(m,'_original_forward',None) is not None} "
                  f"resolved_inner={'none(=fp32-wrapped!)' if inner is None else type(inner).__name__}", flush=True)
        # Force bf16 autocast regardless of which forward we ended up with: lm_head output was
        # torch.float32 in jobs 1171737/1173614 (2026-11-05), so the sparse logits and their grad
        # buffer were 2 x keep x 248k x 4B (13 GiB at keep~6.8k) -> OOM at long windows. logp is
        # still computed in fp32 per chunk in _chunked_logps regardless of this forward's output dtype.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if inner is None:
                return f(**kw)
            return inner(**kw) if inner_is_bound else inner(m, **kw)

    def _chunked_logps(logits, labels, chunk=256):
        """fp32 log_softmax+gather over the last dim, in position chunks, recomputed in backward
        (activation checkpoint) so no full-size fp32 logits/softmax is ever materialised or saved."""
        from torch.utils.checkpoint import checkpoint

        def _f(lg, lb):
            return torch.log_softmax(lg.float(), dim=-1).gather(-1, lb.unsqueeze(-1)).squeeze(-1)

        outs = []
        for i in range(0, logits.shape[1], chunk):
            lg, lb = logits[:, i:i + chunk], labels[:, i:i + chunk]
            if torch.is_grad_enabled() and logits.requires_grad:
                outs.append(checkpoint(_f, lg, lb, use_reentrant=False))
            else:
                outs.append(_f(lg, lb))
        return torch.cat(outs, dim=1)

    _ns["_raw_fwd"] = _raw_fwd
    _ns["_chunked_logps"] = _chunked_logps
    exec(compile(_src, "<sparse_compute_loss>", "exec"), _ns)
    _dense_compute_loss = _dt.DPOTrainer._compute_loss
    _sparse_compute_loss = _ns["_sparse_compute_loss"]

    def _checked_compute_loss(self, model, inputs, return_outputs):
        if args.sparse_check and not getattr(self, "_sparse_checked", False):
            self._sparse_checked = True
            snap = copy.deepcopy(self._metrics)
            with torch.no_grad():
                dense = _dense_compute_loss(self, model, inputs, return_outputs)
            d_loss = (dense[0] if isinstance(dense, tuple) else dense).detach().float().item()
            _last = lambda k: (self._metrics["train"][k][-1] if self._metrics["train"].get(k) else None)
            d = {k: _last(k) for k in ("rewards/margins", "logps/chosen", "logps/rejected", "logits/chosen", "entropy")}
            self._metrics = snap
            res = _sparse_compute_loss(self, model, inputs, return_outputs)
            s_loss = (res[0] if isinstance(res, tuple) else res).detach().float().item()
            sdict = {k: _last(k) for k in d}
            # loss/margin are trivially log2/0 on step 1 (policy == ref); logps/chosen & logps/rejected are the
            # real test of the index alignment (sum of per-token logp over the masked positions, ~ -500 here)
            def _close(a, b, rel=1e-2, abs_=0.5):
                return a is None or b is None or abs(a - b) <= max(abs_, rel * abs(b))
            ok = _close(s_loss, d_loss, abs_=1e-3) and all(_close(sdict[k], d[k]) for k in d)
            print(f"[dpo] SPARSE_CHECK_{'OK' if ok else 'FAIL'} loss sparse={s_loss:.6f} dense={d_loss:.6f} | "
                  + " | ".join(f"{k} sparse={sdict[k]} dense={d[k]}" for k in d)
                  + f" | keep={int(inputs['completion_mask'][:, 1:].any(dim=0).sum())}/{inputs['completion_mask'].shape[1]-1}", flush=True)
            if not ok:
                raise RuntimeError("sparse logits loss does not match dense loss")
            return res
        return _sparse_compute_loss(self, model, inputs, return_outputs)

    _dt.DPOTrainer._compute_loss = _checked_compute_loss
    print("[dpo] SPARSE_LOGITS on (DPOTrainer._compute_loss rewritten from TRL source)", flush=True)
# -------------------------------------------------------------------------------------------

_collator = AssistantOnlyCollator(pad_token_id=tok.pad_token_id, max_length=args.max_len,
                                  truncation_mode="keep_start") if args.assistant_only_mask else None
trainer = DPOTrainer(model=args.model, ref_model=None, args=cfg,
                     train_dataset=ds, processing_class=tok, peft_config=peft_cfg,
                     data_collator=_collator)


class _RawLoraSave(TrainerCallback):
    """2026-11-02: Trainer/PEFT checkpoint saving under FSDP writes an EMPTY adapter
    (see 11-01 notes), so on every save we gather the full state dict ourselves
    (collective, all ranks) and write the lora_* tensors raw into the checkpoint dir.
    save_total_limit then rotates these dirs like normal checkpoints."""
    def on_save(self, args_, state, control, **kw):
        import os as _os
        from safetensors.torch import save_file as _sf
        tr = kw.get("model")
        sd = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.accelerator.is_main_process:
            d = _os.path.join(args_.output_dir, f"checkpoint-{state.global_step}")
            _os.makedirs(d, exist_ok=True)
            lora = {k.replace("_fsdp_wrapped_module.", "").replace(".default.", "."): v.detach().to("cpu").contiguous()
                    for k, v in sd.items() if "lora_" in k}
            _sf(lora, _os.path.join(d, "adapter_model.safetensors"), metadata={"format": "pt"})
            trainer.accelerator.unwrap_model(trainer.model).peft_config["default"].save_pretrained(d)
            print(f"[dpo] CKPT_RAW_LORA step={state.global_step} tensors={len(lora)} -> {d}", flush=True)
        del sd
        return control


trainer.add_callback(_RawLoraSave())


class _StepDiag(TrainerCallback):
    """2026-11-05: per-step wall time / GPU peak / host RSS (rank-local) so a hang or a creeping
    allocation at long windows is visible in the log instead of only in a NCCL watchdog dump."""
    def on_step_begin(self, args_, state, control, **kw):
        import time as _t; self._t0 = _t.time()
    def on_step_end(self, args_, state, control, **kw):
        import time as _t, os as _os, resource as _r
        rss = _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 2**20
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        print(f"[dpo] STEP {state.global_step} rank={_os.environ.get('RANK','0')} wall={_t.time()-self._t0:.1f}s "
              f"gpu_peak={peak:.1f}GiB host_maxrss={rss:.1f}GiB {_smaps()}", flush=True)
        return control


def _smaps():
    """2026-11-05: split host RSS into shared file-backed (safetensors mmap page cache, reclaimable,
    counted once per node) vs private/anonymous (real per-rank pressure)."""
    try:
        want = ("Rss", "Pss", "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty", "Anonymous")
        kv = {}
        for line in open("/proc/self/smaps_rollup"):
            k = line.split(":")[0]
            if k in want:
                kv[k] = int(line.split()[1]) / 2**20
        return "smaps(GiB) " + " ".join(f"{k}={v:.1f}" for k, v in kv.items())
    except Exception as e:
        return f"smaps n/a ({e})"


trainer.add_callback(_StepDiag())
print(f"[dpo] HOSTMEM after model load rank={os.environ.get('RANK','0')} {_smaps()}", flush=True)
trainer.train()
if torch.cuda.is_available():
    import os as _os
    print(f"[dpo] PEAK_GB rank={_os.environ.get('RANK', '0')} "
          f"dev={torch.cuda.current_device()} "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}", flush=True)
trainer.save_model(args.out + "/final")
tok.save_pretrained(args.out + "/final")
# 2026-11-01: verify the adapter really landed. FSDP SHARDED_STATE_DICT wrote 0-byte shards
# (jobs 1038103/1097664); FULL_STATE_DICT + PeftModel.save_pretrained wrote a 40-byte empty
# safetensors (job 1140858) with no warning. So: diagnostics + two PEFT-free fallbacks.
import os as _os
from safetensors.torch import save_file as _save_file
_final = args.out + "/final"
_ad = _os.path.join(_final, "adapter_model.safetensors")
_main = trainer.accelerator.is_main_process
def _ok():
    return _os.path.exists(_ad) and _os.path.getsize(_ad) > 1_000_000
def _clean(k):
    return k.replace("_fsdp_wrapped_module.", "").replace(".default.", ".")
def _write(lora, tag):
    if lora:
        _save_file(lora, _ad, metadata={"format": "pt"})
        print(f"[dpo] {tag}: wrote {len(lora)} lora tensors -> {_ad} size={_os.path.getsize(_ad)}", flush=True)
    else:
        print(f"[dpo] {tag}: no lora tensors found", flush=True)
if _main:
    print(f"[dpo] after save_model: adapter_ok={_ok()} type(trainer.model)={type(trainer.model).__name__} "
          f"type(model_wrapped)={type(trainer.model_wrapped).__name__}", flush=True)
# fallback 1: full state dict via accelerate (collective), raw-save lora keys
_sd = trainer.accelerator.get_state_dict(trainer.model)
if _main:
    _keys = list(_sd.keys())
    print(f"[dpo] diag get_state_dict: n_keys={len(_keys)} lora_keys={sum('lora_' in k for k in _keys)} "
          f"sample={_keys[:3]}", flush=True)
    if not _ok():
        _write({_clean(k): v.detach().to('cpu').contiguous() for k, v in _sd.items() if 'lora_' in k}, "fallback1(get_state_dict)")
del _sd
# fallback 2: summon full params on the FSDP module (collective), read named_parameters
from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
_fm = trainer.model if isinstance(trainer.model, _FSDP) else trainer.model_wrapped
if isinstance(_fm, _FSDP):
    with _FSDP.summon_full_params(_fm, writeback=False, rank0_only=True, offload_to_cpu=True):
        if _main:
            _names = [n for n, _ in _fm.named_parameters()]
            print(f"[dpo] diag summon: n_params={len(_names)} lora_params={sum('lora_' in n for n in _names)} "
                  f"sample={_names[:2]}", flush=True)
            if not _ok():
                _write({_clean(n): p.detach().to('cpu').contiguous() for n, p in _fm.named_parameters() if 'lora_' in n}, "fallback2(summon_full_params)")
trainer.accelerator.wait_for_everyone()
if _main:
    print(f"[dpo] ADAPTER_{'OK' if _ok() else 'MISSING'} {_ad} size={_os.path.getsize(_ad) if _os.path.exists(_ad) else 0}", flush=True)
    print("TRAIN_DONE", args.out)
