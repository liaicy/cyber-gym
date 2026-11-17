# -*- coding: utf-8 -*-
"""Evaluate the 8 k-fold LoRA adapters on their held-out dev task, via a vLLM
serve reached through LOCAL_VLLM_URL (ssh -L tunnel to the GPU serve).

Resumable: an existing output json is skipped, so re-running after a tunnel drop
continues where it stopped. LLMClient's own retries ride out brief drops within
a rollout. Base is NOT re-run here -- compare against the qwen3-8b base solved
rate already in the train/augk batches.

  LOCAL_VLLM_URL=http://127.0.0.1:8000/v1 LOCAL_KEY=x \
  CYBERGYM_SERVER=http://127.0.0.1:18667 PYTHONPATH=. \
    /data/conda/envs/cybergym-py312/bin/python traj/eval_kfold.py
"""
import json
import os
import shutil
import uuid
from pathlib import Path

from examsociety.llm import LLMClient, ModelRef
from traj import batch_rollout as batch
from traj.agent import rollout
from traj.env import CyberGymEnv

URL = os.environ["LOCAL_VLLM_URL"].rstrip("/")
K = int(os.environ.get("EVAL_K", "5"))
FOLDS = json.load(open("/data/traj_batch/kfold_2026_10_12/folds.json"))
T = FOLDS["reachable_tasks"]

OUT = Path("/data/traj_batch/kfold_eval_2026_10_12/out")
OUT.mkdir(parents=True, exist_ok=True)
batch.OUT_DIR = OUT
batch.WORK_DIR = Path("/data/traj_batch/kfold_eval_2026_10_12/work")
batch.WORK_DIR.mkdir(parents=True, exist_ok=True)

llm = LLMClient(max_calls=int(os.environ.get("EVAL_BUDGET", "6000")))


def seat(sid):
    return ModelRef(id=sid, model=sid, api="openai", base_url=URL,
                    api_key_env="LOCAL_KEY", family="qwen",
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})


# each fold's two adapters evaluated ONLY on that fold's held-out dev task
units = [(T[i], f"{name}_fold{i}")
         for i in range(len(T)) for name in ("strict", "cross")]

for dev, sid in units:
    ref = seat(sid)
    for r in range(1, K + 1):
        fn = OUT / f"{dev.replace(':', '-')}__{sid}__r{r}.json"
        if fn.exists():
            continue
        w = batch.WORK_DIR / f"{dev.replace(':', '-')}__{sid}__r{r}__{uuid.uuid4().hex[:6]}"
        try:
            batch.gen_task(dev, w)
            env = CyberGymEnv(str(w))
            traj = rollout(env, llm, ref, max_steps=40)
            fn.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
            print(f"done {fn.name} tier={traj.tier} success={traj.success} "
                  f"steps={traj.n_steps}", flush=True)
        except Exception as e:  # noqa: BLE001 -- one unit must not kill the eval
            print(f"ERR {fn.name}: {type(e).__name__}: {str(e)[:90]}", flush=True)
        finally:
            shutil.rmtree(w, ignore_errors=True)

print("KFOLD_EVAL_DONE", flush=True)
