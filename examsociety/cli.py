from __future__ import annotations

import argparse
import json

from .config import load_config
from .store import RunStore, load_records


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="examsociety",
        description="Multi-agent adversarial exam-item generation society",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="generate items (runs until Ctrl-C unless --limit)")
    p_run.add_argument("--domain", required=True, help="subject domain or keyword")
    p_run.add_argument("--out", required=True, help="output run directory")
    p_run.add_argument("--config", default=None, help="YAML config path")
    p_run.add_argument("--limit", type=int, default=None, help="stop after N items")
    p_run.add_argument(
        "--pipeline", choices=["society", "baseline", "paired", "harden-paired"],
        default="society",
        help=(
            "'baseline' skips the adversarial revision loop; "
            "'paired' skips it too and instead emits clean/injected matched pairs "
            "for misinformation-robustness measurement"
        ),
    )

    p_irt = sub.add_parser("irt", help="psychometric analysis of a run")
    p_irt.add_argument("run", help="run directory or items.jsonl path")

    p_cov = sub.add_parser(
        "coverage", help="knowledge yield: concept saturation and redundancy"
    )
    p_cov.add_argument("run", help="run directory or items.jsonl path")

    p_leak = sub.add_parser(
        "leakage",
        help="author-advantage and injector-interaction checks (needs rotated roles)",
    )
    p_leak.add_argument("run", help="run directory or items.jsonl path")

    p_mis = sub.add_parser(
        "misinfo", help="misinformation robustness over matched pairs (--pipeline paired)"
    )
    p_mis.add_argument("run", help="run directory or items.jsonl path")

    p_drift = sub.add_parser(
        "drift", help="is the hardening loop buying difficulty with obscurity?"
    )
    p_drift.add_argument("run", help="run directory or items.jsonl path")

    p_rep = sub.add_parser("report", help="compare one or more runs")
    p_rep.add_argument("runs", nargs="+", help="run directories (optionally name=path)")

    p_exp = sub.add_parser("export", help="export accepted items as clean JSONL")
    p_exp.add_argument("run", help="run directory or items.jsonl path")
    p_exp.add_argument(
        "--statuses", default="accepted",
        help=("comma-separated final_status values to export, or 'all'. The "
              "default keeps only 'accepted', which discards the medium-difficulty "
              "items that carry most of a bank's measurement precision"),
    )
    p_exp.add_argument("--min-solve-rate", type=float, default=None)
    p_exp.add_argument("--max-solve-rate", type=float, default=None)

    args = parser.parse_args()

    if args.command == "run":
        from .society import ExamSociety

        cfg = load_config(args.config)
        store = RunStore(args.out)
        society = ExamSociety(cfg, store, pipeline=args.pipeline)
        society.run(args.domain, limit=args.limit)

    elif args.command == "irt":
        from .psychometrics import analyse

        records = load_records(args.run)
        stats, theta, solver_ids = analyse(records)
        if not stats:
            print("no scorable items found")
            return
        if len(stats) < 20 or len(solver_ids) < 10:
            print(
                f"(!) exploratory only: {len(stats)} items / {len(solver_ids)} "
                f"solvers — 2PL needs >=20 items & >=10 solvers for stable estimates\n"
            )

        def _f(x):
            return "   nan" if x != x else f"{x:6.2f}"

        print(f"{'item':14} {'p':>6} {'r_pb':>7} {'a':>6} {'b':>7}  flag")
        for s in stats:
            print(
                f"{s.item_id:14} {_f(s.p_value)} {_f(s.point_biserial):>7} "
                f"{_f(s.irt_a)} {_f(s.irt_b):>7}  {s.flag}"
            )
        print("\nsolver ability (theta):")
        for sid, t in zip(solver_ids, theta):
            print(f"  {sid:12} {t:+.2f}")

    elif args.command == "coverage":
        from .coverage import analyse_coverage, format_coverage

        print(format_coverage(analyse_coverage(load_records(args.run))))

    elif args.command == "leakage":
        from .leakage import format_leakage

        print(format_leakage(load_records(args.run)))

    elif args.command == "misinfo":
        from .misinfo import format_misinfo

        print(format_misinfo(load_records(args.run)))

    elif args.command == "drift":
        from .psychometrics import format_drift, revision_drift

        print(format_drift(revision_drift(load_records(args.run))))

    elif args.command == "report":
        from .report import compare

        named = []
        for spec in args.runs:
            name, _, path = spec.rpartition("=")
            if not name:
                name = path
            named.append((name, load_records(path)))
        print(compare(named))

    elif args.command == "export":
        from .psychometrics import _scored_run

        records = load_records(args.run)
        wanted = (
            None if args.statuses == "all"
            else {s.strip() for s in args.statuses.split(",") if s.strip()}
        )
        for r in records:
            item = r.final_item()
            if item is None or (wanted is not None and r.final_status not in wanted):
                continue
            run = _scored_run(r)
            solve_rate = run.solve_rate if run is not None else None
            if solve_rate is not None:
                if args.min_solve_rate is not None and solve_rate < args.min_solve_rate:
                    continue
                if args.max_solve_rate is not None and solve_rate > args.max_solve_rate:
                    continue
            out = {
                "id": r.id,
                "domain": r.domain,
                "subtopic": r.subtopic,
                "stem": item.stem,
                "options": item.options,
                "answer_index": item.answer_index,
                "solution": item.solution,
                "concepts": item.concepts,
                "misinfo": item.misinfo.model_dump() if item.misinfo else None,
                "revisions": item.version,
                "final_status": r.final_status,
                "solve_rate": solve_rate,
                "author_family": item.author_family,
            }
            print(json.dumps(out, ensure_ascii=False))
