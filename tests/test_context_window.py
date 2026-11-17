"""Offline regressions for rollout history budgets and action/observation roles."""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traj.agent import _cap_pair, _fit_messages


def message(role, content, **extra):
    return {"role": role, "content": content, **extra}


def action(size=0):
    return json.dumps({"thought": "a" * size, "action": "read_file",
                       "action_input": {"path": "src/input.c"}})


def assert_window(window):
    head_size = sum(len(m["content"]) for m in window[:2])
    tail_size = sum(len(m["content"]) for m in window[2:])
    assert head_size + tail_size <= 17700
    assert tail_size <= min(16000, 17700 - head_size)
    assert [m["role"] for m in window[:2]] == ["system", "user"]
    for i, m in enumerate(window[2:]):
        assert m["role"] == ("assistant" if i % 2 == 0 else "user")
        if m["role"] == "assistant":
            json.loads(m["content"])


@pytest.mark.parametrize("system_size,task_size,assistant_size,observation_size", [
    (1100, 1500, 8000, 10000),
    (1100, 1500, 180, 100000),
    (100, 100, 20000, 20000),
    (100, 100, 100, 50000),
    (50000, 20000, 100, 1000),
])
def test_real_budget_counterexamples(system_size, task_size, assistant_size, observation_size):
    original = [message("system", "s" * system_size), message("user", "t" * task_size),
                message("assistant", action(assistant_size)),
                message("user", "o" * observation_size)]
    snapshot = copy.deepcopy(original)
    window = _fit_messages(original)
    assert_window(window)
    assert original == snapshot


def test_clip_observation_preserves_full_action_json():
    assistant = action(8000)
    pair = [message("assistant", assistant), message("user", "o" * 100000)]
    capped = _cap_pair(pair, 14000)
    assert capped[0]["content"] == assistant
    assert "observation truncated" in capped[1]["content"]
    assert sum(len(m["content"]) for m in capped) <= 14000


def test_uneven_pair_has_no_minimum_share_overflow():
    assistant = action(100)
    capped = _cap_pair([message("assistant", assistant),
                        message("user", "o" * 100000)], 16000)
    assert capped[0]["content"] == assistant
    assert sum(len(m["content"]) for m in capped) <= 16000


def test_tail_cap_is_enforced_even_when_small_head_leaves_more_space():
    original = [message("system", "system"), message("user", "task"),
                message("assistant", action()), message("user", "o" * 16500)]
    assert sum(len(m["content"]) for m in original) < 17700
    window = _fit_messages(original)
    assert_window(window)
    assert window[2]["content"] == original[2]["content"]
    assert "observation truncated" in window[3]["content"]


def test_oversized_assistant_is_explicitly_omitted_not_broken_json():
    pair = [message("assistant", action(50000)), message("user", "OBSERVATION: result")]
    snapshot = copy.deepcopy(pair)
    capped = _cap_pair(pair, 16000)
    note = json.loads(capped[0]["content"])
    assert "assistant output omitted" in note["thought"]
    assert capped[1]["content"] == pair[1]["content"]
    assert pair == snapshot
    assert sum(len(m["content"]) for m in capped) <= 16000


def test_long_level2_report_survives_when_head_space_is_available():
    task = "TASK:\nDescription\n--- crash report ---\n" + "report\n" * 900
    original = [message("system", "s" * 1100), message("user", task),
                message("assistant", action()), message("user", "source\n" * 10000)]
    window = _fit_messages(original)
    assert_window(window)
    assert window[0]["content"] == original[0]["content"]
    assert window[1]["content"].startswith(task)
    assert "task truncated" not in window[1]["content"]
    assert window[2]["content"] == original[2]["content"]


def test_large_action_survives_by_shortening_oversized_task():
    assistant = action(4000)
    original = [message("system", "s" * 1100), message("user", "task\n" * 4000),
                message("assistant", assistant), message("user", "o" * 100000)]
    window = _fit_messages(original)
    assert_window(window)
    assert window[0]["content"] == original[0]["content"]
    assert window[2]["content"] == assistant
    assert "task truncated" in window[1]["content"]


@pytest.mark.parametrize("trailing_assistant", [False, True])
def test_recent_turns_keep_role_alignment_and_final_assistant(trailing_assistant):
    original = [message("system", "s" * 1100), message("user", "task")]
    for i in range(8):
        original.extend([message("assistant", action(200), turn=i),
                         message("user", "o" * 7000, turn=i)])
    if trailing_assistant:
        original.append(message("assistant", action(), turn=8))
    snapshot = copy.deepcopy(original)
    window = _fit_messages(original)
    assert_window(window)
    assert window[-1] == original[-1]
    assert window[2]["role"] == "assistant"
    assert window[2]["turn"] > 0
    assert original == snapshot


@pytest.mark.parametrize("original", [
    [],
    [message("system", "small")],
    [message("system", "s"), message("user", "task")],
    [message("system", "s"), message("user", "task"), message("assistant", action())],
])
def test_small_histories_are_unchanged(original):
    assert _fit_messages(original) == original


def test_oversized_head_only_is_bounded_without_index_errors():
    single = [message("system", "s" * 50000)]
    assert sum(len(m["content"]) for m in _fit_messages(single)) <= 17700
    original = [message("system", "s" * 1100), message("user", "task\n" * 10000)]
    snapshot = copy.deepcopy(original)
    window = _fit_messages(original)
    assert_window(window)
    assert window[0]["content"] == original[0]["content"]
    assert "task truncated" in window[1]["content"]
    assert original == snapshot


@pytest.mark.parametrize("budget", [0, 1, 20, 200, 16000])
def test_pair_small_budgets_and_empty_input(budget):
    assert _cap_pair([], budget) == []
    pair = [message("assistant", action()), message("user", "o" * 20000)]
    capped = _cap_pair(pair, budget)
    assert [m["role"] for m in capped] == ["assistant", "user"]
    assert sum(len(m["content"]) for m in capped) <= budget


def test_deterministic_varied_history_shapes_remain_bounded_and_immutable():
    rng = random.Random(9)
    for _ in range(100):
        original = [message("system", "系" * rng.choice([0, 1100, 20000])),
                    message("user", "任务" * rng.choice([0, 100, 5000, 30000]))]
        for i in range(rng.randrange(1, 6)):
            original.extend([message("assistant", action(rng.choice([0, 180, 8000, 50000])), turn=i),
                             message("user", "源" * rng.choice([0, 500, 10000, 100000]), turn=i)])
        if rng.choice([False, True]):
            original.append(message("assistant", action(), turn="final"))
        snapshot = copy.deepcopy(original)
        window = _fit_messages(original)
        assert_window(window)
        assert window[-1]["role"] == original[-1]["role"]
        assert original == snapshot
