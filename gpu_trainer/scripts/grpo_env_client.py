"""GPU-side thin client for GRPOTrainer(environment_factory=...).

Runs IN the training process on the GPU node, where TRL calls these methods
directly as tools during generation. Does no CyberGym work itself -- every
call is an HTTP round-trip to traj.grpo_bridge_server, which runs back on the
cloud machine where the real CyberGymEnv, task data, and docker judge live.

2026-10-23: public surface is the original FIVE tools (read_file/grep/
list_dir/write_poc/submit) PLUS a sixth, `act(action, action_input)` --
see the long comment above `act` for the two-step history of why both
exist. Short version: an act-only version was tried first (to match the
flat JSON-action format traj/agent.py's rollout() trained this checkpoint
on, instead of TRL's Hermes-style one-tool-per-method reflection), but a
same-day validation run showed the model calling `write_poc`/`submit`
directly as Hermes tool names (its DPO-trained association) rather than
wrapping them in `act(action=...)` -- every one of those calls then failed
with "Tool not found" because only `act` was registered. Restoring the
five as public methods lets the model's actual calling habit land
correctly; `act` stays registered too, at no cost, for the rare sample
that uses it instead. All six still talk to the same bridge-server
endpoints (traj/grpo_bridge_server.py's /tool/{name}), so grpo_env.
CyberGymGRPOEnv on the server side needs no change either way.

No dependency on anything in this repo's traj/env.py or cybergym package --
only `requests`, so it can be dropped alone onto the GPU venv. (`_first_int`
below is a small copy of env.py's helper of the same name, not an import,
to keep that true.)
"""
from __future__ import annotations

import sys
import time

import requests

DEFAULT_BASE_URL = "http://localhost:18666"


def _first_int(d: dict, *keys: str) -> int | None:
    """First key present in d that parses as an int. Models spell the same
    argument several ways (start / start_line / from); accept all of them
    rather than silently ignoring an unrecognised one."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except (TypeError, ValueError):
                continue
    return None

# 2026-10-21: four GRPO smoke-test attempts in a row died from the reverse
# SSH tunnel (.17 -> login node -> compute node) dropping or hanging mid-call
# -- ConnectionResetError, a ReadTimeout, a hung-but-alive tunnel process that
# silently stopped forwarding data for several minutes, and (attempt 4, after
# a FIXED-COUNT retry was added) a fast-failing "connection refused" loop
# that gave up faster than the tunnel watchdog could rebuild the forward. The
# tunnel keeps getting patched (autossh, a node-tracking watchdog with an
# active health probe) but a SHARED, unprivileged login node is not going to
# be rock solid for a sustained multi-step session; the client needs to
# tolerate a transient blip instead of crashing the whole trainer.
#
# Attempt 4's failure specifically showed why a FIXED retry COUNT is wrong:
# a "connection refused" error is near-free to retry (fails in milliseconds),
# so capping at N attempts wastes almost none of the real timeout budget and
# gives up far too early relative to how long a tunnel rebuild actually
# takes (tens of seconds); a genuine HANG (no response within the full
# per-call timeout) is the opposite -- each attempt alone already burns the
# whole timeout, so the SAME fixed count there would make the worst case
# balloon (N x reward_timeout could be well over an hour). Retrying on a time
# BUDGET instead of a count naturally gives the cheap-failure case many fast
# retries and the expensive-failure case few, within the same wall-clock cap.
_RETRY_BUDGET_MULTIPLIER = 1.5  # total retry wall-clock <= timeout * this
_RETRY_BUDGET_FLOOR = 90.0      # ...but never less than this (fast-fail case
                                 # still needs several tries even when timeout
                                 # itself is small, e.g. the 120s tool calls)
_MAX_BACKOFF_DELAY = 20.0       # cap the exponential backoff per-step so a
                                 # cheap-failure loop keeps retrying densely
                                 # instead of sleeping away its own budget
_BASE_DELAY = 2.0


class RewardUnavailableError(RuntimeError):
    """get_reward() could not get a real, verified answer from the bridge.

    2026-10-23: this used to be swallowed -- an unrecoverable network failure
    or an ambiguous 404 (see get_reward below) returned a hardcoded fallback
    float instead of raising, on the theory that a broken judge shouldn't be
    trained as worse-than-worst. The user explicitly rejected that trade:
    a made-up reward is contaminated training data, full stop, and this
    project's stated principle is "void this round or pause training rather
    than let that happen" -- a crashed step (visible, stops the run) is
    strictly preferable to a silent wrong number baked into a gradient
    update that nobody can later tell apart from a real signal. So both
    paths below now raise this instead of returning a float; the training
    script is expected to let it propagate and fail the run (no try/except
    GRPOTrainer.train() wrapper that would paper back over this).
    """


def _post_with_retry(url: str, json_body: dict, timeout: float, what: str):
    """POST with exponential-backoff retry, budgeted on wall-clock time
    rather than a fixed attempt count (see rationale above).

    Does NOT retry on an HTTP error status (4xx/5xx) from a response that
    actually arrived -- that is a real answer, not a transient drop, and the
    caller (get_reward especially) needs to see the real status code to tell
    "genuinely failed" apart from "succeeded but the response got lost, retry
    hit a used-up session". Only requests.exceptions.RequestException
    (connection reset, timeout, DNS, etc. -- no response at all) triggers a
    retry.
    """
    budget = max(timeout * _RETRY_BUDGET_MULTIPLIER, _RETRY_BUDGET_FLOOR)
    deadline = time.monotonic() + budget
    last_exc = None
    attempt = 0
    while True:
        attempt += 1
        attempt_start = time.monotonic()
        try:
            return requests.post(url, json=json_body, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_exc = e
            attempt_elapsed = time.monotonic() - attempt_start
            remaining = deadline - time.monotonic()
            # A slow failure (this attempt itself burned a big chunk of its
            # own `timeout`, i.e. a genuine hang, not a fast refusal/reset)
            # should not start another attempt unless there is enough budget
            # left to plausibly absorb a FULL further timeout -- otherwise a
            # persistently-hanging path would keep attempting right up to
            # (and past) the deadline, on top of the time already spent,
            # instead of bailing out close to the intended budget.
            was_slow = attempt_elapsed > timeout * 0.5
            if remaining <= 0 or (was_slow and remaining < timeout):
                print(f"[grpo_env_client] {what} attempt {attempt} failed after "
                      f"{attempt_elapsed:.0f}s ({type(e).__name__}: {e}), "
                      f"{remaining:.0f}s of retry budget left"
                      f"{' (not enough for another slow attempt)' if was_slow else ''}, "
                      f"giving up", file=sys.stderr, flush=True)
                break
            delay = min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_BACKOFF_DELAY, remaining)
            print(f"[grpo_env_client] {what} attempt {attempt} failed after "
                  f"{attempt_elapsed:.0f}s ({type(e).__name__}: {e}), retrying in "
                  f"{delay:.0f}s ({remaining:.0f}s of retry budget left)",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
    raise last_exc


class CyberGymGRPOEnvClient:
    # `timeout` governs the fast calls (read_file/grep/list_dir/write_poc/
    # submit -- step() for these returns in-process with no subprocess).
    # reset and reward run far longer on the bridge server and must each
    # cover their full worst-case SEQUENTIAL chain, not just the single
    # longest sub-call, or the client times out while the server keeps
    # running as an orphan with no result ever delivered:
    #   reset  -> gen_task's own subprocess.run(..., timeout=300)            -- +margin for
    #             tarball extraction / HTTP around it, not 300 flat (no slack at all)
    #   reward -> verify()'s submit subprocess (600) THEN, only on a crash,
    #             _trigger_fix_verification (120) THEN _fix_exit_code's query
    #             (30) -- these run ONE AFTER ANOTHER inside a single /reward
    #             call, so 650 (< 600+120+30=750) could still fire before the
    #             server finishes. (2026-10-08, round-2 review P2: the first
    #             version of this fix picked a number above the single
    #             biggest sub-timeout, not above their sum.)
    # 2026-10-21: reward_timeout raised 900 -> 1500 as a defensive margin
    # (some pooled tasks are large C/C++ projects -- mupdf/ffmpeg-class --
    # whose docker build+verify cycle may legitimately run long). Debug
    # stderr lines added at construction and before each reset/reward call,
    # printing the timeout actually in effect, after a prior smoke-test
    # crash's exception message showed a value (420.0) that didn't match any
    # override found in the call chain -- not yet reproduced with this
    # instrumentation live, so left in rather than removed.
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        reset_timeout: float = 420.0,
        reward_timeout: float = 1500.0,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._reset_timeout = reset_timeout
        self._reward_timeout = reward_timeout
        self._session_id: str | None = None
        self.last_reward_valid: bool | None = None
        print(f"[grpo_env_client] constructed: base={self._base} "
              f"timeout={self._timeout} reset_timeout={self._reset_timeout} "
              f"reward_timeout={self._reward_timeout}", file=sys.stderr, flush=True)

    def reset(self, task_id: str, **_ignored) -> str:
        """Start a new episode on `task_id` (e.g. "arvo:3938")."""
        print(f"[grpo_env_client] POST /reset task_id={task_id} "
              f"timeout={self._reset_timeout}", file=sys.stderr, flush=True)
        # /reset is idempotent to retry: each call makes a brand-new session
        # server-side regardless of whether an earlier attempt's response was
        # lost in transit, so a retry just costs one orphaned working dir on
        # the bridge (cleaned up at its /reward teardown or next restart),
        # never a wrong-session bug.
        r = _post_with_retry(f"{self._base}/reset", {"task_id": task_id},
                             self._reset_timeout, "reset")
        r.raise_for_status()
        data = r.json()
        self._session_id = data["session_id"]
        return data["prompt"]

    def get_reward(self) -> float:
        """Environment-anchored reward: verify() runs the real CyberGym
        docker judge (vul/fix exit codes), never a model self-assessment."""
        assert self._session_id is not None, "get_reward() called before reset()"
        print(f"[grpo_env_client] POST /reward session_id={self._session_id} "
              f"timeout={self._reward_timeout}", file=sys.stderr, flush=True)
        # /reward is NOT idempotent: the bridge server deletes the session
        # right after computing the reward (traj/grpo_bridge_server.py), so
        # if the first attempt actually succeeded server-side and only the
        # RESPONSE got lost on a flaky tunnel, a retry with the same
        # session_id hits a 404 "unknown session_id" -- that is this
        # project's exact "request went through fine, response never came
        # back" tunnel failure mode (2026-10-20/21 smoke tests), not a fresh
        # failure. 2026-10-23: both this case and a fully unrecoverable
        # network failure used to be treated as "ambiguous-but-probably-
        # fine" and papered over with a hardcoded fallback float. Per the
        # user's explicit instruction, there is no such thing as a safe
        # fallback reward here -- we either have a real, verified number
        # from the judge or we don't, and if we don't, the honest move is
        # to raise and let the whole run fail loudly rather than train on a
        # guess. (see RewardUnavailableError's docstring above.)
        try:
            r = _post_with_retry(f"{self._base}/reward", {"session_id": self._session_id},
                                 self._reward_timeout, "reward")
        except requests.exceptions.RequestException as e:
            msg = (f"/reward unrecoverable after retries for session_id="
                   f"{self._session_id} -- no real reward obtained, this "
                   f"episode's data is VOID (not trainable, not a fallback "
                   f"value). Session may still have been scored server-side; "
                   f"the response just never arrived.")
            print(f"[grpo_env_client] {msg}", file=sys.stderr, flush=True)
            self.last_reward_valid = False
            raise RewardUnavailableError(msg) from e
        if r.status_code == 404:
            msg = (f"/reward got 404 for session_id={self._session_id} -- "
                   f"almost certainly means an earlier retry attempt already "
                   f"scored and deleted this session and we lost that "
                   f"response. No real reward obtained, this episode's data "
                   f"is VOID.")
            print(f"[grpo_env_client] {msg}", file=sys.stderr, flush=True)
            self.last_reward_valid = False
            raise RewardUnavailableError(msg)
        r.raise_for_status()
        data = r.json()
        # Still returns only the float -- TRL's interface here is
        # get_reward() -> float with no in-protocol "and mask this one" slot.
        # last_reward_valid is set for a trainer-side change to read
        # separately; nothing currently consumes it. See
        # grpo_env.CyberGymGRPOEnv.get_reward()'s docstring for why this is a
        # partial, not complete, mitigation. (2026-10-08, round-2 review P1.)
        self.last_reward_valid = data.get("valid")
        return data["reward"]

    # ------------------------------------------------------------- tools
    #
    # 2026-10-23: these five used to each be their OWN public method, which
    # TRL's environment_factory reflects into FIVE separate Hermes-style
    # tool schemas (inspect.getmembers() over every public, non-underscore
    # method -- see trl/trainer/grpo_trainer.py's _sync_tool_dicts /
    # _env_tools construction). That is a completely different interface
    # from what this checkpoint was actually DPO-trained on: traj/agent.py's
    # rollout() never used tool-calling at all -- it prompts for a flat JSON
    # reply `{"thought": ..., "action": "<tool>", "action_input": {...}}`
    # and dispatches on the `action` string itself (see env.py's
    # CyberGymEnv.step()). A 2026-10-23 40-step validation run (0 infra
    # failures, 20 distinct tasks, raised max_completion_length to rule out
    # a budget cutoff) still came back 100% degenerate/zero-reward-variance;
    # inspecting completions showed the model endlessly exploring (one
    # sample ran all the way to the max_tool_calling_iterations=32 ceiling)
    # and NEVER once calling anything resembling write_poc/submit. The model
    # is out-of-distribution for Hermes tool_calls -- it was never trained
    # to decide "when to submit" in that shape.
    #
    # Fix (low-risk version -- the alternative was rewriting TRL's
    # _tool_call_loop to parse free-text JSON directly, which is a much
    # bigger, riskier change deferred for now): collapse all five into ONE
    # public tool, `act`, whose `action`/`action_input` arguments are
    # exactly the action/action_input vocabulary the DPO data was trained
    # on. The model still has to wrap it in a Hermes tool_call to invoke it
    # (TRL's machinery requires that), but the SEMANTIC content of the call
    # -- which action, which arguments -- is the one thing the checkpoint
    # actually has training signal for. The five implementations below are
    # unchanged and are now private (leading underscore -> excluded from
    # TRL's tool reflection) helpers that `act` dispatches to.

    def act(self, action: str, action_input: dict | None = None) -> str:
        """Take one action against the vulnerable codebase or the verifier.

        Args:
            action: One of "read_file", "grep", "list_dir", "write_poc",
                "submit".
            action_input: Arguments for the chosen action:
                read_file: {"path": str, "start": int (optional, 1-based),
                    "end": int (optional, inclusive)}
                grep: {"pattern": str} -- a regular expression.
                list_dir: {"path": str} (defaults to ".").
                write_poc: {"content": str} -- the PoC bytes; hex escapes
                    like \\x01 are decoded.
                submit: {} -- submits the current PoC to the verifier and
                    ends the episode.

        Returns:
            The observation text for the action taken, or an error string
            if action or action_input was invalid.
        """
        action_input = action_input or {}
        if action == "read_file":
            return self.read_file(
                path=str(action_input.get("path", "")),
                start=_first_int(action_input, "start", "start_line", "from"),
                end=_first_int(action_input, "end", "end_line", "to"),
            )
        if action == "grep":
            return self.grep(pattern=str(action_input.get("pattern", "")))
        if action == "list_dir":
            return self.list_dir(path=str(action_input.get("path", ".")))
        if action == "write_poc":
            return self.write_poc(content=str(action_input.get("content", "")))
        if action == "submit":
            return self.submit()
        return f"unknown action: {action}"

    # 2026-10-23, SAME DAY follow-up: the `act`-only version above was
    # deployed and smoke-tested (job 742500, 15 steps). It measurably
    # changed model behavior for the better -- `submit` appeared in a
    # completion for the first time ever in this project (18 times) and
    # `write_poc` 553 times -- but EVERY one of those calls failed with
    # "Tool <name> not found", because the model, given a Hermes tool_call
    # shell to fill in, overwhelmingly put the ACTION NAME it already knows
    # from DPO training (write_poc, submit, read_file, grep, list_dir) into
    # the Hermes "name" field directly, instead of wrapping it as
    # `act(action=<name>, ...)` the way this file originally expected. Of
    # ~1740 tool_call name occurrences across the run, only a handful (if
    # any) were literally "act" -- the model's association from training is
    # "write_poc IS a tool", not "write_poc is an argument value to act".
    # So: the five are restored as public methods (TRL reflects them back
    # into five separate tool schemas, as before 2026-10-23's first attempt)
    # so the model's own preferred calling shape actually lands somewhere.
    # `act` is kept too, at zero cost, as a fallback for the rare sample
    # that does use it.

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read a file from the vulnerable codebase.

        Args:
            path: File path, relative to the codebase root.
            start: Optional 1-based start line to read only a range.
            end: Optional 1-based end line (inclusive) to read only a range.

        Returns:
            The file contents, or an error string if the path does not exist.
        """
        return self._tool("read_file", {"path": path, "start": start, "end": end})

    def grep(self, pattern: str) -> str:
        """Regex-search the codebase for a pattern.

        Args:
            pattern: A regular expression to search for.

        Returns:
            Matching lines as "path:line:text", one per line.
        """
        return self._tool("grep", {"pattern": pattern})

    def list_dir(self, path: str = ".") -> str:
        """List the contents of a directory in the codebase.

        Args:
            path: Directory path, relative to the codebase root.

        Returns:
            Entry names, one per line (directories suffixed with "/").
        """
        return self._tool("list_dir", {"path": path})

    def write_poc(self, content: str) -> str:
        """Write the current proof-of-concept input bytes.

        Args:
            content: The PoC bytes. Hex escapes like \\x01 are decoded.

        Returns:
            A confirmation of how many bytes were written.
        """
        return self._tool("write_poc", {"content": content})

    def submit(self) -> str:
        """Submit the current PoC to the verifier. Ends the episode.

        Returns:
            A confirmation that submission was received.
        """
        return self._tool("submit", {})

    def _tool(self, name: str, args: dict) -> str:
        assert self._session_id is not None, "tool called before reset()"
        # The five tool calls (read_file/grep/list_dir/write_poc/submit) are
        # all safe to retry: read-only ones have no side effect to double up,
        # write_poc overwrites (idempotent), submit just sets a done flag
        # server-side (idempotent).
        r = _post_with_retry(f"{self._base}/tool/{name}",
                             {"session_id": self._session_id, "args": args},
                             self._timeout, f"tool/{name}")
        r.raise_for_status()
        return r.json()["result"]
