"""login-node: simulate dpo_lora.py's render (fit + chat template) and report,
per jsonl, how many pairs survive --max-len (kept vs dropped) and whether chosen's
decisive action survives the window at token level.

fit is loaded by exec FROM THE LIVE scripts/dpo_lora.py (the fixed version) so
this never drifts from what the trainer actually does. Tokenizer only -- run on
the login node, no salloc.

  cd ~/WORK/exam-society && venv/bin/python gpu_dpo_data_check.py \
      --model models/Qwen3-8B dpo_strict_train.jsonl dpo_cross_train.jsonl
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "scripts", "dpo_lora.py")).read()
_a = _src.index("_CTX_HEAD_CAP")
_b = _src.index("fit = _fit_messages") + len("fit = _fit_messages")
_ns = {}
exec(_src[_a:_b], _ns)
fit = _ns["fit"]

DEC = {"write_poc", "submit"}


def act(c):
    try:
        return json.loads(c).get("action", "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-8B")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def render(prompt, cont):
        win = fit(list(prompt) + list(cont))
        ps = tok.apply_chat_template(win[:2], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(win, tokenize=False, enable_thinking=False)
        return ps, full[len(ps):]

    for f in args.files:
        rows = [json.loads(l) for l in open(f)]
        kept = dropped = dec_ok = dec_tot = 0
        chtok = []
        for r in rows:
            pch, ch = render(r["prompt"], r["chosen"])
            prj, rj = render(r["prompt"], r["rejected"])
            tch = len(tok(pch + ch).input_ids)
            trj = len(tok(prj + rj).input_ids)
            if tch > args.max_len or trj > args.max_len:
                dropped += 1
                continue
            kept += 1
            chtok.append(tch)
            win = fit(list(r["prompt"]) + list(r["chosen"]))
            kepttext = {m["content"] for m in win}
            dec = [m["content"] for m in r["chosen"]
                   if m["role"] == "assistant" and act(m["content"]) in DEC]
            if dec:
                dec_tot += 1
                if all(d in kepttext for d in dec):
                    dec_ok += 1
        mx = max(chtok) if chtok else 0
        print(f"{os.path.basename(f)}: pairs={len(rows)} kept={kept} "
              f"dropped(>{args.max_len}tok)={dropped} | kept chosen tok max={mx} | "
              f"chosen decisive survives fit = {dec_ok}/{dec_tot}")


if __name__ == "__main__":
    main()
