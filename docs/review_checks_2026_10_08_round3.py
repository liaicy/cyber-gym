"""Read-only, deterministic checks for the third repair review.

Run from any directory with the cybergym Python environment. Prints JSON;
does not invoke a model, judge, GPU, tokenizer download, or training script.
Trainer functions are extracted by AST to avoid executing top-level training.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traj.agent import _fit_messages  # noqa: E402


def extract_functions(relative, names, extra=None):
    tree = ast.parse((ROOT / relative).read_text())
    wanted = set(names)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in functions} != wanted:
        raise RuntimeError(f"Functions changed in {relative}; review this checker")
    scope = {"_CTX_HEAD_CAP": 1500, "_CTX_TAIL": 16000, **(extra or {})}
    exec(compile(ast.Module(body=functions, type_ignores=[]), relative, "exec"), scope)
    return scope


def messages(system_size, task_size, assistant_size, observation_size):
    return [
        {"role": role, "content": char * size}
        for role, char, size in [
            ("system", "S", system_size), ("user", "T", task_size),
            ("assistant", "A", assistant_size), ("user", "O", observation_size),
        ]
    ]


def size_of(win):
    return {
        "total_chars": sum(len(m["content"]) for m in win),
        "tail_chars": sum(len(m["content"]) for m in win[2:]),
        "message_chars": [len(m["content"]) for m in win],
        "roles": [m["role"] for m in win],
    }


class ToyTokenizer:
    """An explicitly synthetic role template, NOT the Qwen tokenizer.

    Checks the renderer's assumed string-prefix invariant only. Exact Qwen
    token positions, loss masks, and training outcomes remain unverified.
    """

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        out = "".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n" for m in messages)
        return out + ("<assistant>\n" if add_generation_prompt else "")


def main():
    sft_path = "gpu_trainer_backup/scripts/sft_lora.py"
    dpo_path = "gpu_trainer_backup/scripts/dpo_lora.py"
    sft = extract_functions(sft_path, ["_cap_pair", "_fit_messages"])
    dpo = extract_functions(dpo_path, ["_cap_pair", "fit", "render"], {"tok": ToyTokenizer()})
    fitters = {"rollout": _fit_messages, "sft_backup": sft["_fit_messages"], "dpo_backup": dpo["fit"]}
    cases = {
        "original_8000_plus_10000": messages(100, 100, 8000, 10000),
        "original_20000_plus_20000": messages(100, 100, 20000, 20000),
        "single_50000_observation": messages(100, 100, 100, 50000),
        "task_50000": messages(100, 50000, 100, 100),
        "head_plus_full_tail": messages(1100, 1500, 8000, 10000),
        "uneven_pair": messages(1100, 1500, 180, 100000),
    }
    context_results = {}
    for name, original in cases.items():
        wins = {label: fit(original) for label, fit in fitters.items()}
        context_results[name] = {
            "outputs": {label: size_of(win) for label, win in wins.items()},
            "three_copies_identical": len({json.dumps(w, sort_keys=True) for w in wins.values()}) == 1,
        }

    data_path = Path("/data/traj_batch/dpo_v2/train.jsonl")
    counts = collections.Counter()
    examples = []
    for line_number, line in enumerate(data_path.read_text().splitlines(), 1):
        row = json.loads(line)
        wins = {side: dpo["fit"](row["prompt"] + row[side]) for side in ("chosen", "rejected")}
        counts["rows"] += 1
        if wins["chosen"][:2] != wins["rejected"][:2]:
            counts["unequal_prompt_pairs"] += 1
            if len(examples) < 5:
                examples.append({
                    "line": line_number, "task": row["meta"]["task"],
                    "chosen_head_chars": [len(m["content"]) for m in wins["chosen"][:2]],
                    "rejected_head_chars": [len(m["content"]) for m in wins["rejected"][:2]],
                })
        for side, win in wins.items():
            counts[side + "_clipped"] += win != row["prompt"] + row[side]
            counts[side + "_starts_user"] += len(win) > 2 and win[2]["role"] == "user"
            counts[side + "_empty_continuation"] += len(win) <= 2
            counts[side + "_over_17700_chars"] += sum(len(m["content"]) for m in win) > 17700

    toy_prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    toy_cont = [
        {"role": "assistant", "content": "a" * 10000},
        {"role": "user", "content": "o" * 10000},
        {"role": "assistant", "content": "last"},
    ]
    toy_win = dpo["fit"](toy_prompt + toy_cont)
    toy_rendered_prompt, toy_rendered_cont = dpo["render"](toy_prompt, toy_cont)
    toy_full = dpo["tok"].apply_chat_template(toy_win)

    source_paths = [
        "traj/agent.py", "traj/env.py", "traj/schemas.py", "traj/grpo_env.py",
        "traj/grpo_bridge_server.py", "traj/grpo_env_client.py", "traj/migrate_infra_labels.py",
        sft_path, dpo_path, "gpu_trainer_backup/scripts/grpo_train_v1.py",
        "gpu_trainer_backup/scripts/grpo_env_client.py",
        "docs/review_checks_2026_10_08_round3.py",
    ]
    print(json.dumps({
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "verification_mode": "local deterministic read-only checks; no network/model/judge/training",
        "random_seed": None,
        "seed_note": "No stochastic operations",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "context_config_chars": {"head_cap": 1500, "tail": 16000, "fast_path_threshold": 17700},
        "context_cases": context_results,
        "dpo_data": {
            "path": str(data_path), "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "counts_with_current_backup_fit": dict(counts), "prompt_mismatch_examples": examples,
            "scope": "Current preparation functions on existing JSONL; not historical GPU tensors or checkpoint attribution",
        },
        "toy_template_counterexample": {
            "template": "Synthetic ToyTokenizer only; NOT Qwen",
            "fitted_roles": [m["role"] for m in toy_win],
            "full_has_prompt_prefix": toy_full.startswith(toy_rendered_prompt),
            "completion_start": toy_rendered_cont[:100],
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
