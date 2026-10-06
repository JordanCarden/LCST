#!/usr/bin/env python3
"""Run or compile the four-block LCST feature-importance analysis."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lcst_pipeline.feature_importance import (  # noqa: E402
    compile_outputs,
    run_shard,
)
from lcst_pipeline.report_evaluation import MODEL_DEFINITIONS, STATIC_SEEDS  # noqa: E402


def _worker(arguments: tuple[str, int, bool]) -> str:
    model_id, seed, force = arguments
    return str(run_shard(model_id, seed, force=force))


def _models(value: str | None) -> list[str]:
    return list(MODEL_DEFINITIONS) if value is None else [value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one model/seed shard or all models for a seed")
    run_parser.add_argument("--seed", type=int, required=True, choices=STATIC_SEEDS)
    run_parser.add_argument("--model", choices=tuple(MODEL_DEFINITIONS))
    run_parser.add_argument("--force", action="store_true", help="Replace an existing valid shard")

    subparsers.add_parser("compile", help="Validate all 40 shards and compile final artifacts")

    all_parser = subparsers.add_parser("all", help="Run missing shards and compile final artifacts")
    all_parser.add_argument("--jobs", type=int, default=1, help="Parallel model/seed workers")
    all_parser.add_argument("--force", action="store_true", help="Replace existing valid shards")

    args = parser.parse_args()
    if args.command == "run":
        for model_id in _models(args.model):
            run_shard(model_id, args.seed, force=args.force)
        return
    if args.command == "compile":
        compile_outputs()
        return

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    work = [
        (model_id, seed, args.force)
        for model_id in MODEL_DEFINITIONS
        for seed in STATIC_SEEDS
    ]
    if args.jobs == 1:
        for arguments in work:
            _worker(arguments)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as executor:
            for path in executor.map(_worker, work):
                print(f"Completed {path}", flush=True)
    compile_outputs()


if __name__ == "__main__":
    main()
