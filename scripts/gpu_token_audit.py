"""GPU-side token / max_len audit for a DPO train.jsonl (run on the A800 box).

WHY: export_dpo's data-health numbers are CHARACTER counts; the real train-time
truncation is by TOKEN count against the trainer's max_len. The collection box
has no transformers/model, so this runs where the model lives. It measures:
  1. real Qwen token lengths of prompt+chosen and prompt+rejected,
  2. how many pairs exceed candidate max_len values (default 8192/12288/16384)
     and how many pairs have BOTH sides fitting (i.e. no truncation at all),
  3. (--probe-mem) peak GPU memory of a 1-step fwd+bwd at each max_len, a rough
     lower bound on what fits -- real DPO also runs a reference-model forward and
     keeps optimizer state, so the true OOM threshold is LOWER than this probe.

First copy the jsonl over from the collection box, e.g.:
  scp <cloud>:/data/traj_batch/dpo_strict_2026_10_12/train.jsonl ~/

Usage:
  python gpu_token_audit.py --jsonl ~/train.jsonl --model Qwen/Qwen3-8B
  python gpu_token_audit.py --jsonl ~/train.jsonl --probe-mem
"""
import argparse
import json
import statistics


def seq_tokens(tok, msgs):
    # Tokenize the raw concatenated content. Chat-template special tokens are a
    # few per turn (negligible vs content); this reflects true content length,
    # which is what decides whether a sequence exceeds the trainer's max_len.
    text = "".join(m["content"] for m in msgs)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--caps", default="8192,12288,16384")
    ap.add_argument("--probe-mem", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    caps = [int(c) for c in args.caps.split(",")]
    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    ch = [seq_tokens(tok, r["prompt"] + r["chosen"]) for r in rows]
    rj = [seq_tokens(tok, r["prompt"] + r["rejected"]) for r in rows]

    print(f"pairs: {len(rows)}  (model={args.model})")
    for name, xs in (("prompt+chosen  ", ch), ("prompt+rejected", rj)):
        line = f"  {name}: median={int(statistics.median(xs))} max={max(xs)} tok"
        for c in caps:
            line += f" | >{c}: {sum(x > c for x in xs)}/{len(xs)}"
        print(line)
    # DPO truncates BOTH sides to max_len; a pair is untruncated only if both fit.
    for c in caps:
        both = sum(1 for a, b in zip(ch, rj) if a <= c and b <= c)
        print(f"  pairs with BOTH sides <= {c} tok (no truncation): {both}/{len(rows)}")

    if args.probe_mem:
        import torch
        from transformers import AutoModelForCausalLM
        print("\n-- memory probe (1-step fwd+bwd; real DPO adds a reference-model "
              "forward + optimizer state, so true OOM is LOWER) --")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda")
        for c in caps:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ids = torch.randint(0, 1000, (2, c), device="cuda")  # chosen+rejected
                out = model(ids, labels=ids)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
                print(f"  max_len={c}: OK, peak ~{torch.cuda.max_memory_allocated()/1e9:.1f} GB")
            except torch.cuda.OutOfMemoryError:
                print(f"  max_len={c}: OOM")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
