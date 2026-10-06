#!/usr/bin/env python3
"""Train maintained models or run the complete report evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lcst_pipeline.modeling import train_all
from lcst_pipeline.schema import MASTER_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=MASTER_PATH)
    parser.add_argument(
        "--report-evaluation",
        action="store_true",
        help="Run the retained eight-model report evaluation instead of production training",
    )
    parser.add_argument(
        "--phase",
        choices=("static", "static-summary", "adaptation", "compile", "all"),
        default="all",
        help="Report-evaluation phase; used only with --report-evaluation",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="Adaptation repeat (0-9); required for the adaptation phase",
    )
    parser.add_argument(
        "--previous-results-dir",
        type=Path,
        help="Optional previous evaluation directory for corrected-vs-previous tables",
    )
    args = parser.parse_args()
    if args.report_evaluation:
        if args.data != MASTER_PATH:
            parser.error("--data applies only to production model training")
        if args.phase == "adaptation" and args.repeat is None:
            parser.error("--repeat is required for the adaptation phase")
        if args.repeat is not None and args.phase != "adaptation":
            parser.error("--repeat applies only to the adaptation phase")
        from lcst_pipeline.report_evaluation import run_report_evaluation

        run_report_evaluation(args.phase, args.repeat, args.previous_results_dir)
        print("Completed report evaluation phase:", args.phase)
        return
    if (
        args.phase != "all"
        or args.repeat is not None
        or args.previous_results_dir is not None
    ):
        parser.error(
            "--phase, --repeat, and --previous-results-dir require --report-evaluation"
        )
    paths = train_all(args.data)
    print("Trained eight models and wrote two metric tables:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
