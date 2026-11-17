"""Multi-turn chat completion for the rollout loop.

examsociety.llm.complete() is single-shot (a system + a user string). The
rollout loop needs real multi-turn messages so the trajectory it records -- and
the SFT/GRPO data derived from it -- matches how a chat model is actually
prompted at inference, instead of the whole history flattened into one user
turn (which would train the target on a token layout it never sees when run).

examsociety/ is archived (CLAUDE.md: no new features), so instead of adding a
chat() method there this drives the wire call here while reusing that module's
error taxonomy, retry policy, and budget accounting. Every current solver
(gpt/kimi/deepseek/qwen) speaks the OpenAI chat-completions protocol; a non-
openai ref falls back to the flattened single-shot path (none of the current
solvers use it, so no behaviour that relied on it changes).

Returns (text, usage) -- usage is the response's token block, or {} when the
endpoint omits it, so token accounting can never break a rollout that would
otherwise succeed.
"""
from __future__ import annotations

import os
import time

import httpx

from examsociety.llm import (
    LLMClient,
    LLMError,
    ModelRef,
    QuotaExhausted,
    TruncatedError,
    UpstreamUnavailable,
    _is_quota_error,
)

_FINISH = {"length": "truncated", "content_filter": "refusal"}
_clients: dict[str, httpx.Client] = {}


def _client(base_url: str) -> httpx.Client:
    if base_url not in _clients:
        _clients[base_url] = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(600.0, connect=20.0),
            headers={"user-agent": "examsociety/0.1"},
        )
    return _clients[base_url]


def _flatten(messages: list[dict]) -> tuple[str, str]:
    """(system, user) for the single-shot fallback: system verbatim, the rest
    joined into one transcript -- the pre-multi-turn layout, kept only for
    protocols the current solver pool does not use."""
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    rest = messages[1:] if system else messages
    return system, "\n".join(m["content"] for m in rest)


def chat(
    llm: LLMClient, ref: ModelRef, messages: list[dict], max_tokens: int = 4000
) -> tuple[str, dict]:
    if ref.api != "openai":
        system, user = _flatten(messages)
        return llm.complete(ref, system, user, max_tokens), {}

    base = ref.base_url or os.environ.get("OPENAI_BASE_URL")
    if not base:
        raise LLMError(f"{ref.id}: openai-protocol model needs base_url")
    api_key = os.environ.get(ref.api_key_env or "OPENAI_API_KEY", "")
    payload = {
        "model": ref.model,
        "messages": messages,
        ref.token_param: max_tokens,
        **ref.extra_body,
    }
    # Same outer retry with long backoff as examsociety.llm.complete: pooled
    # gateways can be capacity-starved (429/5xx) for tens of seconds. Budget is
    # charged per attempt (retries included) via the same _charge that complete
    # uses, so BudgetExhausted still stops an unattended run here.
    last_err: str | None = None
    for attempt in range(7):
        if attempt:
            print(
                f"    [llm] {ref.id} attempt {attempt} after error: "
                f"{str(last_err)[:120]}",
                flush=True,
            )
            time.sleep(min(20 * attempt, 90))
        llm._charge(ref)
        try:
            r = _client(base).post(
                "/chat/completions", json=payload,
                headers={"authorization": f"Bearer {api_key}"},
            )
        except httpx.TransportError as e:
            last_err = f"{ref.model}: {e}"
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"{ref.model}: HTTP {r.status_code} {r.text[:200]}"
            continue
        if r.status_code >= 400:
            if _is_quota_error(r.status_code, r.text):
                raise QuotaExhausted(f"{ref.id}: HTTP {r.status_code} {r.text[:200]}")
            raise LLMError(f"{ref.model}: HTTP {r.status_code} {r.text[:300]}")
        try:
            body = r.json()
            choice = body["choices"][0]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise LLMError(f"{ref.model}: unexpected response shape: {r.text[:300]}") from e
        stop = _FINISH.get(choice.get("finish_reason") or "", "ok")
        if stop == "refusal":
            raise LLMError(f"{ref.id}: request refused by safety classifiers")
        if stop == "truncated":
            raise TruncatedError(f"{ref.id}: output truncated at {max_tokens} tokens")
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip():
            raise LLMError(f"{ref.id}: empty response (stop={stop})")
        return text, (body.get("usage") or {})
    raise UpstreamUnavailable(f"{ref.id}: retries exhausted: {last_err}")
