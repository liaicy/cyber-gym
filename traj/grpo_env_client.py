"""GPU-side thin client for GRPOTrainer(environment_factory=...).

Runs IN the training process on the GPU node, where TRL calls these methods
directly as tools during generation. Does no CyberGym work itself -- every
call is an HTTP round-trip to traj.grpo_bridge_server, which runs back on the
cloud machine where the real CyberGymEnv, task data, and docker judge live.

Method names, signatures, and docstrings are a byte-for-byte copy of
CyberGymGRPOEnv's tool methods (grpo_env.py) -- transformers' get_json_schema
reflects on these to build the tool schema the model sees, so this class and
the real one must present an identical surface or the model would be trained
against a different tool contract than whatever actually runs eval-side.

No dependency on anything in this repo's traj/env.py or cybergym package --
only `requests`, so it can be dropped alone onto the GPU venv.
"""
from __future__ import annotations

import requests

DEFAULT_BASE_URL = "http://localhost:18666"


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
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        reset_timeout: float = 420.0,
        reward_timeout: float = 900.0,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._reset_timeout = reset_timeout
        self._reward_timeout = reward_timeout
        self._session_id: str | None = None
        self.last_reward_valid: bool | None = None

    def reset(self, task_id: str, **_ignored) -> str:
        """Start a new episode on `task_id` (e.g. "arvo:3938")."""
        r = requests.post(f"{self._base}/reset", json={"task_id": task_id},
                          timeout=self._reset_timeout)
        r.raise_for_status()
        data = r.json()
        self._session_id = data["session_id"]
        return data["prompt"]

    def get_reward(self) -> float:
        """Environment-anchored reward: verify() runs the real CyberGym
        docker judge (vul/fix exit codes), never a model self-assessment."""
        assert self._session_id is not None, "get_reward() called before reset()"
        r = requests.post(f"{self._base}/reward", json={"session_id": self._session_id},
                          timeout=self._reward_timeout)
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
    # Docstrings copied verbatim from grpo_env.CyberGymGRPOEnv -- keep in
    # sync by hand; there are only five.

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
        r = requests.post(f"{self._base}/tool/{name}",
                          json={"session_id": self._session_id, "args": args},
                          timeout=self._timeout)
        r.raise_for_status()
        return r.json()["result"]
