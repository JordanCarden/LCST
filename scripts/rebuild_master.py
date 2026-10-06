#!/usr/bin/env python3
"""Rebuild the observed and polymer-target LCST measurement tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lcst_pipeline.ingest import rebuild_masters
from lcst_pipeline.schema import CONFIG_PATH, OBSERVED_MASTER_PATH, POLYMER_MASTER_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--observed-output", type=Path, default=OBSERVED_MASTER_PATH)
    parser.add_argument("--polymer-output", type=Path, default=POLYMER_MASTER_PATH)
    args = parser.parse_args()
    paths = rebuild_masters(args.observed_output, args.polymer_output, args.config)
    print(f"Observed: 1,255 measurements to {paths['observed']}")
    print("Observed outcomes: 927 LCST, 56 UCST, 272 NONE")
    print(f"Polymer: 1,255 measurements to {paths['polymer']}")
    print("Polymer outcomes: 913 LCST, 43 UCST, 299 NONE")


if __name__ == "__main__":
    main()
