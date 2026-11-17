"""Prepare a candidate task split without collecting rollouts or training.

Only the standard library is imported. All source files remain read-only;
the destination is a new directory, and the official split is never updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_PATTERN = re.compile(r"(?:arvo|oss-fuzz):[1-9][0-9]*\Z")
OUTPUT_PATTERN = re.compile(
    r"(?P<source>arvo|oss-fuzz)-(?P<number>[1-9][0-9]*)__.+__r[1-9][0-9]*\.json\Z"
)
DEFAULT_BATCH = Path("/tmp/batch100.txt")
DEFAULT_OBSERVED = Path("/data/traj_batch/pilot_newpool/out")
DEFAULT_SPLIT = Path("/data/traj_batch/splits/task_splits.json")
# An explicit, finite registry: never recursively search work trees or secrets.
DEFAULT_TRAINING_JSONLS = tuple(
    Path("/data/traj_batch") / directory / "train.jsonl"
    for directory in (
        "sft", "sft_v2", "selfdistill_sft_v1", "dpo", "dpo_v2",
        "dpo_r2", "dpo_r3", "dpo_r4", "dpo_r5", "dpo_r6", "dpo_r7", "dpo_r8",
    )
)


def _task_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not TASK_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid task ID in {context}: {value!r}")
    return value


def _unique_tasks(values: Any, context: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"Expected a task list in {context}")
    tasks = [_task_id(value, context) for value in values]
    duplicates = sorted(task for task, n in Counter(tasks).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate task IDs in {context}: {duplicates}")
    return tasks


def _read(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = path.read_bytes()
    return data, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _trajectory_task(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid trajectory task_id in {context}")
    head = value.split("__", 1)[0]
    for source in ("arvo", "oss-fuzz"):
        if head.startswith(source + "-"):
            head = source + ":" + head[len(source) + 1:]
            break
    return _task_id(head, context)


def prepare_plan(
    *, batch_file: Path, observed_out: Path, prior_split: Path,
    training_jsonls: list[Path] | None = None, seed: int = 42,
    test_count: int = 15, dev_count: int = 10, expected_task_count: int = 100,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a candidate split and input audit; do not write any files."""
    if min(test_count, dev_count) < 1 or expected_task_count < 1:
        raise ValueError("test_count, dev_count and expected_task_count must be positive")
    batch_bytes, batch_info = _read(batch_file)
    planned = _unique_tasks(batch_bytes.decode("utf-8").split(), str(batch_file))
    if len(planned) != expected_task_count:
        raise ValueError(f"Expected {expected_task_count} planned tasks, got {len(planned)}")
    planned_set = set(planned)

    split_bytes, split_info = _read(prior_split)
    prior = json.loads(split_bytes)
    prior_groups = {}
    for key in (
        "eval_holdout_clean_uniform", "train_clean_uniform",
        "train_contaminated_near_distribution",
    ):
        prior_groups[key] = _unique_tasks(prior[key], f"{prior_split}:{key}")
    all_prior = [task for tasks in prior_groups.values() for task in tasks]
    if len(set(all_prior)) != len(all_prior):
        raise ValueError("Existing split has task overlap between train/eval groups")
    forbidden = set(all_prior)
    training_audit = []
    missing_default_paths = []
    sources = DEFAULT_TRAINING_JSONLS if training_jsonls is None else training_jsonls
    for path in sources:
        if not path.exists() and training_jsonls is None:
            missing_default_paths.append(str(path))
            continue
        raw, info = _read(path)
        tasks = set()
        records = 0
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            meta = record.get("meta") or {}
            task = meta.get("task") or record.get("task_id") or record.get("task")
            tasks.add(_task_id(task, f"{path}:{line_number}"))
            records += 1
        forbidden.update(tasks)
        training_audit.append({**info, "records": records, "tasks": sorted(tasks)})
    overlap = sorted(planned_set & forbidden)
    if overlap:
        raise ValueError(f"Planned tasks overlap existing train/eval task IDs: {overlap}")

    if not observed_out.is_dir():
        raise ValueError(f"Observed output directory does not exist: {observed_out}")
    observed_tasks = set()
    observed_files = []
    model_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for path in sorted(observed_out.glob("*.json")):
        match = OUTPUT_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected trajectory filename: {path.name}")
        raw, info = _read(path)
        record = json.loads(raw)
        task = _trajectory_task(record.get("task_id"), str(path))
        filename_task = match["source"] + ":" + match["number"]
        if task != filename_task:
            raise ValueError(f"Trajectory task_id and filename disagree: {path}")
        if task not in planned_set:
            raise ValueError(f"Observed trajectory is outside the original task pool: {task}")
        observed_tasks.add(task)
        observed_files.append({**info, "task": task})
        model_counts[str(record.get("model_id", "unknown"))] += 1
        tier_counts[str(record.get("tier", "unknown"))] += 1
    untouched = sorted(planned_set - observed_tasks)
    if len(untouched) <= test_count + dev_count:
        raise ValueError("Not enough untouched tasks for test, dev and nonempty new train")
    # This seed governs only the split of the untouched pool. The original
    # task sampler is unknown and must not be retrospectively labelled random.
    shuffled = untouched[:]
    random.Random(seed).shuffle(shuffled)
    test = sorted(shuffled[:test_count])
    dev = sorted(shuffled[test_count:test_count + dev_count])
    train_new = sorted(shuffled[test_count + dev_count:])
    observed_train = sorted(observed_tasks)
    roles = {
        "observed_train": observed_train,
        "new_train": train_new,
        "new_dev": dev,
        "final_test": test,
        "legacy_dev_regression": sorted(prior_groups["eval_holdout_clean_uniform"]),
    }
    counts = {key: len(tasks) for key, tasks in roles.items()}
    counts["train_candidates"] = len(observed_train) + len(train_new)
    plan = {
        "schema_version": 1,
        "status": "candidate_not_activated",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": seed,
        "split_method": "shuffle sorted untouched task IDs with Python random.Random",
        "original_sampling_seed": None,
        "original_sampling_method": "unknown; original task file preserved by SHA256",
        "source_task_count": len(planned),
        "observed_trajectory_count": len(observed_files),
        "untouched_task_count": len(untouched),
        "counts": counts,
        "roles": roles,
        "collection_scope": {
            "train_task_file": "tasks_train.txt",
            "dev_task_file": "tasks_dev.txt",
            "train_tasks_to_collect": len(train_new),
            "dev_collection": "separate file; not activated by this preparation",
            "final_test_collection": "not activated; no collector task file generated",
            "old_regression_collection": "not activated; no collector task file generated",
        },
        "limitations": [
            "No rollout collection, API call or training is performed by this tool.",
            "This candidate does not modify or activate the official split.",
            "Already observed tasks are train candidates, not evidence of previous training.",
            "Original 100-task selection method/seed is unknown; representativeness is unverified.",
            "Task-ID disjointness is checked; project/CVE-family overlap is not checked.",
            "Exclusion coverage is limited to the prior split and the listed training files.",
            "Task inclusion is independent of solved labels; no reachable-only screening is applied.",
        ],
    }
    provenance = {
        "batch_file": batch_info,
        "prior_split": split_info,
        "training_sources": training_audit,
        "missing_default_training_paths": missing_default_paths,
        "known_excluded_task_ids": sorted(forbidden),
        "observed_directory": str(observed_out.resolve()),
        "observed_files": observed_files,
        "observed_model_counts": dict(sorted(model_counts.items())),
        "observed_tier_counts": dict(sorted(tier_counts.items())),
    }
    return plan, provenance


def write_plan(destination: Path, plan: dict[str, Any], provenance: dict[str, Any]) -> None:
    """Publish into a new/empty directory; never overwrite a populated plan."""
    if destination.is_symlink():
        raise ValueError(f"Destination must not be a symlink: {destination}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Destination already exists and is nonempty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".next_batch_plan_", dir=destination.parent))
    try:
        for name, payload in (("candidates_split.json", plan), ("provenance.json", provenance)):
            (stage / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
        for name, role in (("tasks_train.txt", "new_train"), ("tasks_dev.txt", "new_dev")):
            (stage / name).write_text("\n".join(plan["roles"][role]) + "\n", encoding="utf-8")
        counts = plan["counts"]
        (stage / "README.md").write_text(
            "# 下一批候选任务划分\n\n"
            "这是一份可复核的候选清单，未启动采集或训练，未修改正式 split。\n\n"
            f"原清单 {plan['source_task_count']} 题；已采 {counts['observed_train']} 题"
            f" / {plan['observed_trajectory_count']} 条轨迹，归入 train 候选。"
            f"其余 {plan['untouched_task_count']} 题按 split seed={plan['split_seed']} 分配："
            f"新增 train {counts['new_train']}、dev {counts['new_dev']}、final test {counts['final_test']}。\n\n"
            f"- tasks_train.txt 仅包含本轮新增的 {counts['new_train']} 个 train 任务。\n"
            "- tasks_dev.txt 是独立 dev 清单，本工具不启动其采集。\n"
            "- final test 和旧四题 regression dev 仅保存在 candidates_split.json，"
            "没有对应采集 task 文件。\n"
            "- provenance.json 保留原清单、旧 split、有限训练文件和逐条已采文件的 SHA256。\n\n"
            "原 100 题采样 seed/方法未知；seed 只描述本次未采题的划分。"
            "已检查已知训练/评测 task ID 重叠，项目与 CVE family 关系尚未核查。"
            "按任务状态划分不依赖成功标签，不能称已构建可达 slice。"
            "后续采集和 exporter 必须显式读取候选角色；准备清单本身不构成激活。\n",
            encoding="utf-8",
        )
        # rename replaces an empty directory on POSIX, but refuses a nonempty
        # destination even if another process created files after our check.
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-file", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--observed-out", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--prior-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--training-jsonl", action="append", type=Path,
                        help="Additional training file; repeat to extend the finite default registry")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-count", type=int, default=15)
    parser.add_argument("--dev-count", type=int, default=10)
    parser.add_argument("--expected-task-count", type=int, default=100)
    parser.add_argument("--dest", required=True, type=Path)
    args = parser.parse_args()
    try:
        resolved_dest = args.dest.resolve()
        observed = args.observed_out.resolve()
        if resolved_dest == observed or observed in resolved_dest.parents:
            raise ValueError("Destination must be outside the original observed output directory")
        plan, provenance = prepare_plan(
            batch_file=args.batch_file, observed_out=args.observed_out,
            prior_split=args.prior_split,
            # Additional CLI inputs cannot bypass the known training registry.
            training_jsonls=(None if args.training_jsonl is None else list(dict.fromkeys(
                [path for path in DEFAULT_TRAINING_JSONLS if path.exists()]
                + args.training_jsonl
            ))),
            seed=args.seed, test_count=args.test_count, dev_count=args.dev_count,
            expected_task_count=args.expected_task_count,
        )
        write_plan(args.dest, plan, provenance)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Cannot prepare candidate plan: {exc}\n")
    print(json.dumps({"destination": str(args.dest.resolve()), "counts": plan["counts"],
                      "status": plan["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
