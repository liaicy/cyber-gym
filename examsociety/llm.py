from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import anthropic
import httpx


class LLMError(RuntimeError):
    """A call failed for a reason that is specific to this request.

    Retrying the same request is not expected to help — the caller should
    discard the attempt and move on (write a different item, mark the solver
    unscored, ...).
    """


class TruncatedError(LLMError):
    pass


class UpstreamUnavailable(LLMError):
    """The provider/gateway is down, not the request.

    Raised only after the client's own backoff has been exhausted, so it means
    a sustained (many-minute) outage. Callers must NOT swallow this: an
    unattended run should pause and retry rather than record the outage as a
    content failure.
    """


class _Retryable(Exception):
    """Internal: a failure worth another attempt (5xx, rate limit, connection)."""


class BudgetExhausted(RuntimeError):
    """The configured call budget is spent.

    Deliberately NOT an LLMError: it must not be swallowed by a role's
    degrade-and-continue handler or retried by the outage loop. It ends the run.
    """


class QuotaExhausted(RuntimeError):
    """A provider hard-rejected the call because the account's paid quota is
    spent (HTTP 402/403 access_terminated, insufficient balance, ...).

    Like BudgetExhausted and unlike UpstreamUnavailable, waiting does not help:
    the quota refreshes on the billing cycle, not in the next few minutes. NOT
    an LLMError, so a role cannot degrade it into an unscored answer and the
    outage loop cannot retry it forever. It ends the run so the operator can
    swap keys or top up instead of burning the loop on guaranteed failures.
    """


# --------------------------------------------------------------------- models


@dataclass(frozen=True)
class ModelRef:
    """Everything needed to address one model on one endpoint.

    Kept as a plain dataclass so this module has no dependency on the config
    layer; `config.ModelSpec.ref` builds it.
    """

    id: str                          # short label used in records and reports
    model: str                       # provider-side model name
    api: str = "anthropic"           # wire protocol: anthropic | openai
                                     # NOT a vendor — DeepSeek/Qwen/Kimi/GLM/Gemini
                                     # all speak the openai protocol
    base_url: Optional[str] = None   # None -> provider default / env
    api_key_env: Optional[str] = None
    token_param: str = "max_tokens"  # some endpoints want max_completion_tokens
    family: str = ""                 # grouping for author/injector-interaction analysis
    # Vendor-specific request-body fields merged into the payload verbatim, e.g.
    # {"reasoning_effort": "low"} on Kimi K3 or {"thinking": {...}} on Claude.
    # Deliberately generic: these knobs change per vendor and per model
    # generation, and one of them (effort) is the only way to build a real
    # ability gradient out of a single model.
    extra_body: dict = field(default_factory=dict, compare=False)

    def __str__(self) -> str:
        return self.id


# ------------------------------------------------------------------- backends


def _needs_header_scrub(base_url: str) -> bool:
    """Some third-party gateways block requests carrying SDK-identifying
    headers (User-Agent 'Anthropic/Python', x-stainless-*). Scrub them when
    talking to anything other than the official API."""
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return not host.endswith("anthropic.com")


def _scrub_headers(request) -> None:
    request.headers["user-agent"] = "examsociety/0.1"
    for k in [k for k in request.headers if k.lower().startswith("x-stainless")]:
        del request.headers[k]


class _AnthropicBackend:
    def __init__(self) -> None:
        self._clients: dict[tuple[Optional[str], Optional[str]], Any] = {}

    def _client(self, ref: ModelRef):
        base = ref.base_url or os.environ.get("ANTHROPIC_BASE_URL", "") or None
        key = os.environ.get(ref.api_key_env) if ref.api_key_env else None
        cache_key = (base, ref.api_key_env)
        if cache_key not in self._clients:
            http_client = None
            if _needs_header_scrub(base or ""):
                http_client = anthropic.DefaultHttpxClient(
                    event_hooks={"request": [_scrub_headers]}
                )
            kwargs: dict[str, Any] = {"max_retries": 4, "http_client": http_client}
            if base:
                kwargs["base_url"] = base
            if key:
                # Subscription OAuth tokens (claude setup-token) authenticate
                # with `Authorization: Bearer`, not `x-api-key` — passing one as
                # api_key fails with 401. The prefix is the documented marker.
                if key.startswith("sk-ant-oat"):
                    kwargs["auth_token"] = key
                else:
                    kwargs["api_key"] = key
            # With no explicit key the SDK falls back to ANTHROPIC_API_KEY /
            # ANTHROPIC_AUTH_TOKEN from the environment on its own.
            self._clients[cache_key] = anthropic.Anthropic(**kwargs)
        return self._clients[cache_key]

    def call(self, ref: ModelRef, system: str, user: str, max_tokens: int):
        # extra_body has to work on both protocols or a config knob silently
        # does nothing on one of them. The SDK has its own passthrough for this,
        # so vendor fields (thinking, reasoning_effort, ...) reach the wire
        # without this module needing to know what they mean.
        extra = {"extra_body": dict(ref.extra_body)} if ref.extra_body else {}
        try:
            r = self._client(ref).messages.create(
                model=ref.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **extra,
            )
        except (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        ) as e:
            raise _Retryable(str(e)) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise _Retryable(str(e)) from e
            if _is_quota_error(e.status_code, str(e)):
                raise QuotaExhausted(f"{ref.model}: {e}") from e
            raise LLMError(f"{ref.model}: {e}") from e
        except anthropic.APIError as e:
            raise LLMError(f"{ref.model}: {e}") from e
        text = "".join(b.text for b in r.content if b.type == "text")
        stop = {"max_tokens": "truncated", "refusal": "refusal"}.get(
            r.stop_reason or "", "ok"
        )
        return text, stop


def _is_quota_error(status: int, text: str) -> bool:
    low = (text or "").lower()
    # Phrases, not bare words: "quota" and "billing" alone match ordinary
    # prose and error text that has nothing to do with a spent balance.
    markers = (
        "access_terminated", "insufficient_quota", "insufficient balance",
        "insufficient_user_quota", "insufficient user quota",
        "usage limit", "quota exceeded", "exceeded your current quota",
        "out of credit", "arrears", "billing cycle", "payment required",
    )
    # 402 is always payment; 403 only when the body names a quota/billing cause
    if status == 402:
        return True
    return any(m in low for m in markers)


class _OpenAICompatBackend:
    """Chat-completions backend for every OpenAI-compatible endpoint.

    Covers GPT, Gemini (OpenAI-compat endpoint), DeepSeek, Qwen, Kimi, GLM and
    most aggregating gateways. Implemented on raw httpx rather than the openai
    SDK: no extra dependency, and full control over headers, which is what
    gateways tend to be picky about.
    """

    _FINISH = {"length": "truncated", "content_filter": "refusal"}

    def __init__(self) -> None:
        self._clients: dict[str, httpx.Client] = {}

    def _client(self, base_url: str) -> httpx.Client:
        if base_url not in self._clients:
            self._clients[base_url] = httpx.Client(
                base_url=base_url.rstrip("/"),
                timeout=httpx.Timeout(600.0, connect=20.0),
                headers={"user-agent": "examsociety/0.1"},
            )
        return self._clients[base_url]

    def call(self, ref: ModelRef, system: str, user: str, max_tokens: int):
        base = ref.base_url or os.environ.get("OPENAI_BASE_URL")
        if not base:
            raise LLMError(f"{ref.id}: openai-protocol model needs base_url")
        api_key = os.environ.get(ref.api_key_env or "OPENAI_API_KEY", "")
        payload = {
            "model": ref.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            ref.token_param: max_tokens,
            **ref.extra_body,
        }
        try:
            r = self._client(base).post(
                "/chat/completions",
                json=payload,
                headers={"authorization": f"Bearer {api_key}"},
            )
        except httpx.TransportError as e:
            raise _Retryable(f"{ref.model}: {e}") from e
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"{ref.model}: HTTP {r.status_code} {r.text[:200]}")
        if r.status_code >= 400:
            # Quota check ONLY inside the error branch. A 200 body carries
            # model-generated prose plus a "usage" block, so scanning every
            # response for words like "quota"/"billing" false-positives on
            # success — and QuotaExhausted is terminal, so one false positive
            # kills an entire overnight run.
            if _is_quota_error(r.status_code, r.text):
                raise QuotaExhausted(f"{ref.id}: HTTP {r.status_code} {r.text[:200]}")
            raise LLMError(f"{ref.model}: HTTP {r.status_code} {r.text[:300]}")
        try:
            choice = r.json()["choices"][0]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise LLMError(f"{ref.model}: unexpected response shape: {r.text[:300]}") from e
        text = (choice.get("message") or {}).get("content") or ""
        return text, self._FINISH.get(choice.get("finish_reason") or "", "ok")


def _find_claude_cli() -> str:
    """Locate the Claude Code CLI, newest first.

    Pinning a version in config breaks every time the IDE extension updates
    (2.1.234 -> 2.1.235 silently killed a launch mid-session). Resolve it at
    call time instead; PATH still wins if the user installed it standalone.
    """
    from glob import glob
    from shutil import which

    if (found := which("claude")):
        return found
    pattern = os.path.expanduser(
        r"~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe"
    )
    candidates = sorted(
        glob(pattern),
        key=lambda p: [int(x) for x in re.findall(r"\d+", p.split("claude-code-")[-1])[:3]],
    )
    return candidates[-1] if candidates else "claude"


class _ClaudeCliBackend:
    """Drive the local Claude Code CLI in headless print mode (`claude -p`).

    Why this exists: subscription OAuth tokens are hard-gated to haiku on the
    direct API (429 on every other model regardless of quota), but the same
    subscription reaches every plan model through the official CLI headless
    mode — which is the documented automation path, not a workaround. The CLI
    authenticates from its own local credential store, so no key material
    passes through this process at all.

    Trade-offs, measured: ~5s process startup per call, ~15s total for a short
    completion, and the CLI's own system context (~35k mostly-cached tokens)
    charged to the subscription window on every call. Use for a small number
    of panel seats, not for high-volume roles.
    """

    def call(self, ref: ModelRef, system: str, user: str, max_tokens: int):
        exe = ref.base_url or os.environ.get("CLAUDE_CLI_PATH") or _find_claude_cli()
        # -p mode has no separate system-prompt channel worth relying on
        # across CLI versions; embed it in the prompt instead.
        prompt = f"<instructions>\n{system}\n</instructions>\n\n{user}"
        cmd = [exe, "-p", prompt, "--model", ref.model, "--output-format", "json"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=600,
            )
        except subprocess.TimeoutExpired as e:
            raise _Retryable(f"{ref.id}: cli timeout: {e}") from e
        except OSError as e:
            raise LLMError(f"{ref.id}: cannot launch claude cli at {exe!r}: {e}") from e
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:300]
            low = err.lower()
            if any(t in low for t in ("overloaded", "rate limit", "429", "529", "econnreset")):
                raise _Retryable(f"{ref.id}: {err}")
            if _is_quota_error(0, err):
                raise QuotaExhausted(f"{ref.id}: {err}")
            raise LLMError(f"{ref.id}: cli exit {proc.returncode}: {err}")
        try:
            obj = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            raise LLMError(f"{ref.id}: unparseable cli output: {proc.stdout[:200]}") from e
        if obj.get("is_error"):
            raise LLMError(f"{ref.id}: cli error: {str(obj.get('result'))[:200]}")
        return str(obj.get("result") or ""), "ok"


class LLMClient:
    """Dispatches calls to the right backend for each model.

    Backends normalise their own errors into the same three outcomes
    (_Retryable / TruncatedError / LLMError) so that the outage handling in
    `complete` — and therefore UpstreamUnavailable — behaves identically no
    matter which provider a solver sits on.
    """

    def __init__(self, max_calls: int = 0) -> None:
        self._backends = {
            "anthropic": _AnthropicBackend(),
            "openai": _OpenAICompatBackend(),
            "claude-cli": _ClaudeCliBackend(),
        }
        self.max_calls = max_calls  # 0 = unlimited
        self.calls = 0

    def _charge(self, ref: ModelRef) -> None:
        """Count one upstream call and stop the run once the budget is spent.

        Charged per attempt, retries included — that is what actually costs
        money. An unattended run has no other stopping condition.
        """
        self.calls += 1
        if self.max_calls and self.calls > self.max_calls:
            raise BudgetExhausted(
                f"call budget spent: {self.calls - 1}/{self.max_calls} "
                f"(stopped before calling {ref.id})"
            )

    def complete(
        self, ref: ModelRef, system: str, user: str, max_tokens: int = 4000
    ) -> str:
        backend = self._backends.get(ref.api)
        if backend is None:
            raise LLMError(f"{ref.id}: unknown api protocol {ref.api!r}")
        # Outer retry with long backoff: pooled gateways can be capacity-starved
        # (503) for tens of seconds, beyond any SDK's fast retry window.
        last_err: Exception | None = None
        for attempt in range(7):
            if attempt:
                print(
                    f"    [llm] {ref.id} attempt {attempt} after error: "
                    f"{str(last_err)[:120]}",
                    flush=True,
                )
                time.sleep(min(20 * attempt, 90))
            try:
                self._charge(ref)
                text, stop = backend.call(ref, system, user, max_tokens)
                break
            except _Retryable as e:
                last_err = e
        else:
            raise UpstreamUnavailable(
                f"{ref.id}: retries exhausted: {last_err}"
            ) from last_err
        if stop == "refusal":
            raise LLMError(f"{ref.id}: request refused by safety classifiers")
        if stop == "truncated":
            raise TruncatedError(f"{ref.id}: output truncated at {max_tokens} tokens")
        if not text.strip():
            raise LLMError(f"{ref.id}: empty response (stop={stop})")
        return text

    def complete_json(
        self,
        ref: ModelRef,
        system: str,
        user: str,
        max_tokens: int = 4000,
        repair_rounds: int = 1,
    ) -> dict[str, Any]:
        """Request a JSON object, with one repair round-trip on parse failure.

        On truncation, retries once with a doubled budget.
        """
        try:
            text = self.complete(ref, system, user, max_tokens)
        except TruncatedError:
            text = self.complete(ref, system, user, max_tokens * 2)
        obj = extract_json(text)
        if obj is not None:
            return obj
        for _ in range(repair_rounds):
            text = self.complete(
                ref,
                system,
                user
                + "\n\nYour previous output could not be parsed as JSON. "
                "Respond again with ONLY a single valid JSON object, no markdown fences, "
                "no commentary.",
                max_tokens,
            )
            obj = extract_json(text)
            if obj is not None:
                return obj
        raise LLMError(f"{ref.id}: could not obtain valid JSON. Last output:\n{text[:500]}")


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the first balanced top-level JSON object from model output."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        # strict=False tolerates literal newlines/tabs inside
                        # strings, which models frequently emit in long solutions
                        obj = json.loads(candidate, strict=False)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
