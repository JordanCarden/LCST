from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "lcst.json"
OBSERVED_MASTER_PATH = REPO_ROOT / "data" / "processed" / "lcst_master_observed.csv"
POLYMER_MASTER_PATH = REPO_ROOT / "data" / "processed" / "lcst_master_polymer.csv"
# Modeling follows the experimentally observed target by default.
MASTER_PATH = OBSERVED_MASTER_PATH
HEATING_RATE_SHEET = "LCST - Different HeatingRate"

MASTER_COLUMNS = [
    "source_workbook",
    "source_sheet",
    "polymer_name",
    "polymer_mw_kda",
    "polymer_functionalization_percent",
    "polymer_concentration_mg_ml",
    "buffer",
    "heating_rate_c_per_min",
    "additive_name",
    "additive_concentration_mM",
    "salt_name",
    "salt_concentration_mM",
    "transition_type",
    "transition_temperature_c",
]

FORMULATION_COLUMNS = [
    "polymer_name",
    "polymer_mw_kda",
    "polymer_functionalization_percent",
    "polymer_concentration_mg_ml",
    "buffer",
    "additive_name",
    "additive_concentration_mM",
    "salt_name",
    "salt_concentration_mM",
]

PREDICTION_INPUT_COLUMNS = [
    "polymer_name",
    "polymer_mw_kda",
    "polymer_functionalization_percent",
    "polymer_concentration_mg_ml",
    "buffer",
    "heating_rate_c_per_min",
    "additive_name",
    "additive_concentration_mM",
    "salt_name",
    "salt_concentration_mM",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
