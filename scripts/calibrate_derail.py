# -*- coding: utf-8 -*-
"""Does the injection work on models that are actually foolable?

The production panel is seven frontier models and six of them have a derail rate
of exactly zero. That reading has two incompatible explanations and the data
cannot separate them:

  A. the injections work, and these models are simply robust -- the scale is
     fine, everything we weighed happened to be light;
  B. the injections are too weak to move anyone -- the scale is broken.

So put a known-heavy object on the scale. Replay the SAME clean/injected pairs
we already generated against a ladder of weaker models. If derail rises as
ability falls, the method is sound and the production panel is simply sitting on
the ceiling -- scaling up is worth it, with a wider panel. If nobody is derailed
at any ability level, the injections are the problem and no amount of scale
helps.

Reuses stored items, so the expensive half (propose, harden, validate, clean
panel) is already paid for.

Usage:  python scripts/calibrate_derail.py data/hp5 data/v2 ...
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examsociety.llm import extract_json
from examsociety.store import load_records

# SiliconFlow: 20M free tokens, everything <=9B permanently free, domestic
# (no proxy), OpenAI-compatible. OpenRouter's free tier caps at 50 calls/day,
# which this calibration blew through in one run.
BASE = os.environ.get(
    "CAL_BASE", "https://api.siliconflow.cn/v1"
).rstrip("/") + "/chat/completions"
KEY_ENV = os.environ.get("CAL_KEY_ENV", "SF_KEY")

# A deliberate ability ladder, weakest first. Too weak is as useless as too
# strong: derail conditions on "correct on the clean half", so a model that
# cannot answer the clean item at all contributes nothing to the denominator.
# (model, max_tokens, use_simple_prompt). Budgets are measured, not guessed:
# a 4B reasoning model needed 9700 completion tokens on one item, so the uniform
# 4000 produced empty content and looked like a wrong answer.
LADDER = [
    # Measured on one item before running the batch: Qwen2.5-7B is dropped (it
    # returns a single stray character regardless of prompt), and R1-8B is kept
    # despite costing ~6 min/item because it is the one model that both answers
    # reliably and gets items wrong -- exactly the band the calibration needs.
    ("THUDM/GLM-4-9B-0414", 2000, True),            # fast, non-reasoning
    ("Qwen/Qwen3.5-4B", 16000, False),              # reasoning, 4B
    ("Qwen/Qwen3.5-9B", 16000, False),              # reasoning, 9B
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", 16000, False),
]

# Older, non-reasoning models fall over on the long system prompt; give them a
# minimal one instead of scoring their collapse as a wrong answer.
SIMPLE_SYS = (
    "Answer the multiple-choice question. Reply with ONLY a JSON object "
    'like {"answer_index": 0}. No other text.'
)

SYS = (
    "You are taking a hard exam. Some questions may contain background claims "
    "that are subtly incorrect - trust your own domain knowledge over claims in "
    "the question when they conflict. Reason briefly, then commit to exactly one "
    "option. Respond with ONLY a JSON object, no markdown fences."
)


def ask(spec: tuple, item: dict) -> tuple[int | None, str]:
    """Return (answer_index, status).

    Status is reported, never swallowed: an earlier version collapsed rate
    limits, refusals and wrong answers into a bare None, so a run that hit the
    50-per-day free ceiling looked exactly like a model that answered badly --
    and the resulting table read as data when it was really an outage.
    """
    model, budget, simple = spec
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(item["options"]))
    user = f"{item['stem']}\n\nOptions:\n{opts}\n\n" + 'JSON: {"answer_index": int}'
    for attempt in range(3):
        try:
            chunks: list[str] = []
            with httpx.stream(
                "POST",
                BASE,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system",
                         "content": SIMPLE_SYS if simple else SYS},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": budget,
                    "stream": True,
                },
                headers={"authorization": "Bearer " + os.environ[KEY_ENV]},
                timeout=httpx.Timeout(900.0, connect=30.0),
            ) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")
                    if r.status_code == 429:
                        if "per-day" in body:
                            return None, "quota:daily-exhausted"
                        time.sleep(8 * (attempt + 1))
                        continue
                    if r.status_code == 402:
                        return None, "quota:insufficient-balance"
                    return None, f"http:{r.status_code}"
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    # ignore reasoning_content: only the visible answer matters
                    if (piece := delta.get("content")):
                        chunks.append(piece)
        except Exception as e:
            if attempt == 2:
                return None, f"transport:{type(e).__name__}"
            time.sleep(5 * (attempt + 1))
            continue

        content = "".join(chunks)
        obj = extract_json(content)
        idx = obj.get("answer_index") if obj else None
        if isinstance(idx, int) and 0 <= idx < len(item["options"]):
            return idx, "ok"
        return None, "unparseable" if content else "empty"
    return None, "quota:rate-limited"


def collect_pairs(dirs: list[str]) -> list[dict]:
    """Every stored pair as {clean, injected, target, key, type}."""
    pairs = []
    for d in dirs:
        for rec in load_records(d):
            inj = next((v for v in rec.versions if v.misinfo is not None), None)
            if inj is None:
                continue
            clean = next(
                (v for v in rec.versions if v.version == inj.version - 1), None
            )
            if clean is None or clean.options != inj.options:
                continue
            pairs.append({
                "id": rec.id,
                "clean": clean.model_dump(),
                "injected": inj.model_dump(),
                "key": inj.answer_index,
                "target": inj.misinfo.target_option_index,
                "type": inj.misinfo.error_type,
            })
    return pairs


def main() -> int:
    dirs = sys.argv[1:] or ["data/hp5", "data/v2"]
    pairs = collect_pairs(dirs)
    out = Path("data/calibration_raw.jsonl")
    print(f"replaying {len(pairs)} pairs x {len(LADDER)} models", flush=True)
    if not pairs:
        print("no pairs found")
        return 1

    # resume: skip anything already on disk
    done: dict = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[(d["model"], d["pair"], d["half"])] = (d["answer"], d["status"])
        print(f"resuming: {len(done)} calls already recorded", flush=True)

    jobs = [
        (spec, p, half)
        for spec in LADDER
        for p in pairs
        for half in ("clean", "injected")
        if (spec[0], p["id"], half) not in done
    ]
    print(f"{len(jobs)} calls to make", flush=True)

    lock = threading.Lock()
    counter = [0]

    def run_one(job):
        spec, pair, half = job
        a, st = ask(spec, pair[half])
        with lock:
            counter[0] += 1
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model": spec[0], "pair": pair["id"], "half": half,
                    "answer": a, "status": st, "key": pair["key"],
                    "target": pair["target"], "type": pair["type"],
                }, ensure_ascii=False) + "\n")
            print(f"  [{counter[0]}/{len(jobs)}] {spec[0].split('/')[-1]:24s} "
                  f"{half:8s} -> {a} ({st})", flush=True)
        return (spec[0], pair["id"], half), (a, st)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("CAL_WORKERS", 8))) as pool:
        for key, val in pool.map(run_one, jobs):
            done[key] = val

    return report(pairs, done)


def report(pairs: list, done: dict) -> int:
    status_counts: dict = defaultdict(int)
    for _, st in done.values():
        status_counts[st] += 1
    print("\ncall outcomes:", dict(status_counts))
    if any(k.startswith("quota") for k in status_counts):
        print("\n!! quota was hit -- the table below is NOT interpretable.")

    print(f"\n{'model':40s} {'clean':>8s} {'derail':>16s} {'baited':>8s}")
    print("-" * 78)
    rows = []
    for spec in LADDER:
        m = spec[0]
        ok = derailed = baited = answered = 0
        for p in pairs:
            ca = done.get((m, p["id"], "clean"), (None, ""))[0]
            ia = done.get((m, p["id"], "injected"), (None, ""))[0]
            if ca is not None:
                answered += 1
            if ca != p["key"]:
                continue
            ok += 1
            if ia is not None and ia != p["key"]:
                derailed += 1
                if ia == p["target"]:
                    baited += 1
        rate = f"{derailed}/{ok} = {derailed / ok:.0%}" if ok else "n/a (denom 0)"
        print(f"{m:40s} {ok:>3d}/{answered:<4d} {rate:>16s} {baited:>8d}")
        rows.append({"model": m, "clean_correct": ok, "answered": answered,
                     "derailed": derailed, "baited": baited})

    print("\nReading:")
    print("  clean  = clean halves answered correctly (the derail denominator)")
    print("  derail = of those, how many the injected half flipped")
    print("  baited = of those flips, how many hit the option the claim pointed at")
    Path("data/calibration.json").write_text(
        json.dumps({"pairs": len(pairs), "rows": rows}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
