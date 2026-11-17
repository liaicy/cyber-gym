"""Bridge server: runs on this cloud machine (where CyberGym env/docker/verify
actually live) and exposes CyberGymGRPOEnv over HTTP, so the GRPOTrainer
process on the GPU node -- which calls environment methods in-process as part
of its own generation loop -- can drive a real CyberGymGRPOEnv without any
cybergym package, task data, or docker access on the GPU side at all.

Sessions are keyed by session_id (one per pooled environment slot on the GPU
side -- see grpo_env_client.CyberGymGRPOEnvClient). Each session wraps its own
CyberGymGRPOEnv instance, so concurrent rollouts on the GPU don't share state.

Run: /data/conda/envs/cybergym-py312/bin/python -m traj.grpo_bridge_server
[2026-10-20: the module-level `app = FastAPI()` alone never started a server --
`python -m` just imported the module and exited 0 with no listener. This had
never actually been run via this documented command before now; the
__main__ block below is the fix, plus you can always run it the standard
FastAPI way instead: `uvicorn traj.grpo_bridge_server:app --host 0.0.0.0
--port 18666`.]
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .grpo_env import CyberGymGRPOEnv

app = FastAPI()
_sessions: dict[str, CyberGymGRPOEnv] = {}


class ResetReq(BaseModel):
    task_id: str


class ToolReq(BaseModel):
    session_id: str
    args: dict = {}


class SessionReq(BaseModel):
    session_id: str


def _get(session_id: str) -> CyberGymGRPOEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"unknown session_id {session_id}")
    return env


@app.post("/reset")
def reset(req: ResetReq):
    session_id = uuid.uuid4().hex
    env = CyberGymGRPOEnv(work_root="/data/traj_batch/grpo_work")
    prompt = env.reset(task_id=req.task_id)
    _sessions[session_id] = env
    return {"session_id": session_id, "prompt": prompt}


@app.post("/tool/{name}")
def tool(name: str, req: ToolReq):
    env = _get(req.session_id)
    fn = getattr(env, name, None)
    if fn is None or name in ("reset", "get_reward"):
        raise HTTPException(400, f"no such tool: {name}")
    return {"result": fn(**req.args)}


@app.post("/reward")
def reward(req: SessionReq):
    env = _get(req.session_id)
    r = env.get_reward()
    valid = env.reward_is_valid()
    # Free the session's temp dir once the episode is scored -- GRPO calls
    # get_reward() exactly once per completed rollout (grpo_trainer.py).
    env._teardown()
    del _sessions[req.session_id]
    # `valid` is additive -- the existing client reads only "reward" and
    # keeps working unmodified. It exists so a future trainer-side change
    # (group mean/std from valid rollouts only, masked loss on invalid ones --
    # see grpo_env.get_reward()'s docstring) has something to read; nothing
    # here enforces that on its own. (2026-10-08, round-2 review P1.)
    return {"reward": r, "valid": valid}


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(_sessions)}


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("GRPO_BRIDGE_PORT", 18666)))
