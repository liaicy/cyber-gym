# -*- coding: utf-8 -*-
"""Probe whether Claude is reachable *through your plan* — one real call on the
SAME code path the trajectory generator will use (examsociety.llm's Anthropic
backend), so a green probe means the real rollout path is green, not a parallel
httpx reimplementation that might differ on headers/auth.

Two legitimate "through my plan" routes, both already handled by the backend:

  1. Aggregation gateway (Anthropic protocol proxying every family):
       export ANTHROPIC_BASE_URL=https://<your-gateway>/...   # non-anthropic.com
       export ANTHROPIC_API_KEY=<gateway token>
     The backend auto-scrubs SDK-identifying headers for non-anthropic.com hosts.

  2. Claude subscription, via a setup-token (the sanctioned programmatic bridge
     for a Pro/Max plan — NOT the raw Claude Code OAuth token):
       claude setup-token                # mints an sk-ant-oat... token
       export ANTHROPIC_API_KEY=sk-ant-oat-...     # sent as Bearer automatically
     Leave ANTHROPIC_BASE_URL unset (goes to official api.anthropic.com).
     Caveat: this spends your subscription's rate budget and is meant for
     light/interactive use; for a big batch, a real API key or the gateway is
     sturdier.

Usage (in an env with `anthropic` installed):
    CLAUDE_MODEL=claude-opus-5 python traj/claude_probe.py
Env:
    CLAUDE_MODEL   provider-side model name (default: claude-opus-5). Via a
                   gateway this may be named differently — set it to whatever
                   your plan exposes.
    ANTHROPIC_BASE_URL   gateway base url, or unset for official.
    CLAUDE_KEY_ENV       which env var holds the key (default ANTHROPIC_API_KEY).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examsociety.llm import LLMClient, ModelRef, LLMError, QuotaExhausted  # noqa: E402


def main() -> int:
    model = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    key_env = os.environ.get("CLAUDE_KEY_ENV", "ANTHROPIC_API_KEY")
    key = os.environ.get(key_env, "")

    print("=== Claude reachability probe (via examsociety.llm Anthropic path) ===")
    print(f"  model         : {model}")
    print(f"  route         : {'gateway ' + base if base else 'official api.anthropic.com'}")
    print(f"  key env var   : {key_env}")
    if not key:
        print(f"  key           : (未设置) -> export {key_env}=<你的 key>")
        print("\nFAIL: 没有 key,没法测。把 plan 的 key 设成上面那个环境变量再跑。")
        return 2
    auth_mode = "Bearer (OAuth / setup-token)" if key.startswith("sk-ant-oat") else "x-api-key"
    print(f"  key           : SET (len={len(key)}, 前6={key[:6]}..., auth={auth_mode})")

    ref = ModelRef(
        id="claude-probe", model=model, api="anthropic",
        base_url=base or None, api_key_env=key_env, family="claude",
    )
    # 1024 tokens so a thinking-on model (Opus 5 thinks by default) doesn't come
    # back empty from budget starvation and get misread as a failure.
    client = LLMClient()
    t0 = time.time()
    try:
        # complete() returns a plain str; 4th positional is max_tokens. Give a
        # thinking-on model (Opus 5) room so it doesn't come back empty.
        text = client.complete(ref, "Reply with exactly: OK", "ping", 2048)
        stop = "ok"
    except QuotaExhausted as e:
        print(f"\nFAIL(额度): 认证过了,但额度/计费不通 -> {e}")
        print("  这台 plan 有效但没额度(402/403 一类)。等重置,或换一把 key。")
        return 3
    except LLMError as e:
        msg = str(e)
        # complete() raises LLMError on an empty body -- for a thinking model
        # that usually means auth+quota were fine and thinking ate the budget,
        # NOT that the plan is down. Report it as reachable, not a hard fail.
        if "empty response" in msg.lower():
            dt = time.time() - t0
            print(f"\nOK(连通/认证通过): 返回体为空(thinking 吃光 token),但 plan 是通的。({dt:.1f}s)")
            return 0
        print(f"\nFAIL(其他): {msg[:300]}")
        low = msg.lower()
        if "401" in msg or "authentication" in low or "invalid x-api-key" in low:
            print("  -> 401:key 不对,或该走 Bearer 却当成了 x-api-key(setup-token 要 sk-ant-oat 前缀)。")
        elif "404" in msg or "not_found" in low or "model" in low:
            print(f"  -> 404:base_url 路径或模型名不对。你的网关把这个模型叫别的名?改 CLAUDE_MODEL。")
        elif "429" in msg:
            print("  -> 429:通了,但当前被限流。稍后重试。")
        return 1
    except Exception as e:  # anthropic not installed / network / unexpected
        print(f"\nFAIL(环境/网络): {type(e).__name__}: {str(e)[:300]}")
        return 1

    dt = time.time() - t0
    print(f"\nOK: plan 通了。({dt:.1f}s, stop={stop})")
    print(f"  回复片段: {text[:120]!r}")
    if not text.strip():
        print("  (回复为空但调用成功——多半是 thinking 吃光了 1024 token;连通/认证没问题。)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
