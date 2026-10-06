from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# v2 includes the sourced descriptor definitions (active additive moiety and
# nominal-chain F127 estimates); v1 artifacts must not be mixed with this table.
# The later mixed-source HLB change has its own lookup version and renamed
# descriptor column; the two-slot encoding is unchanged.
FEATURE_ENCODING_VERSION = "zero_preserving_log1p_v2"
# Fixed physical reference concentrations; these are not fitted to the dataset.
CONCENTRATION_REFERENCES = {"polymer_mg_ml": 1.0, "additive_mM": 1.0, "salt_mM": 1.0}

DESCRIPTOR_23_COLUMNS = [
    "dex_f_percent",
    "dex_mw_kda",
    "dex_conc_log1p_mg_ml",
    "additive_conc_log1p_mM",
    "additive_mw_g_mol",
    "additive_logp",
    "additive_tpsa_a2",
    "additive_hbd",
    "additive_hba",
    "additive_formal_charge",
    "additive_carbons",
    "additive_hlb",
    "additive_class_anionic",
    "additive_class_cationic",
    "additive_class_zwitterionic",
    "additive_class_nonionic",
    "salt_conc_log1p_mM",
    "cation_charge",
    "anion_charge",
    "cation_hofmeister_index",
    "anion_hofmeister_index",
    "ionic_strength_m",
    "hofmeister_delta",
]

SLOT1_PCA_COLUMNS = [f"slot1_molformer_pca16_{index:02d}" for index in range(16)]
SLOT2_PCA_COLUMNS = [f"slot2_molformer_pca16_{index:02d}" for index in range(16)]
TWO_SLOT_37_COLUMNS = [
    "dex_f_percent",
    "dex_mw_kda",
    "dex_conc_log1p_mg_ml",
    "slot1_additive_conc_log1p_mM",
    *SLOT1_PCA_COLUMNS,
    "slot2_salt_conc_log1p_mM",
    *SLOT2_PCA_COLUMNS,
]
BUFFER_COLUMN = "buffer_di_water"


def concentration_log1p(value: Any, reference: float = 1.0) -> float:
    """log10(1 + c/c0): preserve zero and missingness, reject invalid amounts."""
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Concentration reference must be positive and finite")
    if pd.isna(value) or value == "":
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid concentration: {value!r}")
    if math.isnan(number):
        return np.nan
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Concentration must be nonnegative and finite: {value!r}")
    return math.log1p(number / reference) / math.log(10.0)


def _number(value: Any, *, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def ingredient_present(name: Any, concentration: Any, group: str) -> bool:
    """Use identity to distinguish an unknown amount from an absent ingredient."""
    text = str(name).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        raise ValueError(f"Missing {group} identity; use 'No {group}' for absence")
    amount = concentration_log1p(concentration)  # validate before any zeroing
    if text == f"No {group}":
        if np.isfinite(amount) and amount > 0:
            raise ValueError(f"No {group} has a positive concentration")
        return False
    if amount == 0:
        raise ValueError(f"Present {group} {text!r} has zero concentration; label it 'No {group}'")
    return True


def _feature_frame(records, columns, presence) -> pd.DataFrame:
    result = pd.DataFrame.from_records(records, columns=columns).astype(float)
    # Bookkeeping only, never model input columns. Retained by iloc/column subsets
    # so hierarchical analyses can impute using present-ingredient donors too.
    result.attrs["ingredient_presence"] = {
        group: dict(enumerate(values)) for group, values in presence.items()
    }
    return result


def build_descriptor23(
    conditions: pd.DataFrame,
    chemical_metadata: dict[str, Any],
) -> pd.DataFrame:
    additive_descriptors = chemical_metadata["additive_descriptors"]
    salt_descriptors = chemical_metadata["salt_descriptors"]
    rows: list[dict[str, float]] = []
    presence = {group: [] for group in ("polymer", "additive", "salt")}
    for row in conditions.itertuples(index=False):
        additive_name = str(row.additive_name)
        additive_mm = _number(row.additive_concentration_mM)
        salt_name = str(row.salt_name)
        salt_mm = _number(row.salt_concentration_mM)
        active = {
            "polymer": ingredient_present(row.polymer_name, row.polymer_concentration_mg_ml, "polymer"),
            "additive": ingredient_present(additive_name, row.additive_concentration_mM, "additive"),
            "salt": ingredient_present(salt_name, row.salt_concentration_mM, "salt"),
        }
        for group, present in active.items():
            presence[group].append(present)
        record = {column: 0.0 for column in DESCRIPTOR_23_COLUMNS}
        if active["polymer"]:
            record.update({
                "dex_f_percent": _number(row.polymer_functionalization_percent),
                "dex_mw_kda": _number(row.polymer_mw_kda),
                "dex_conc_log1p_mg_ml": concentration_log1p(row.polymer_concentration_mg_ml, CONCENTRATION_REFERENCES["polymer_mg_ml"]),
            })
        if active["additive"]:
            record.update({column: np.nan for column in DESCRIPTOR_23_COLUMNS[3:16]})
            record["additive_conc_log1p_mM"] = concentration_log1p(additive_mm, CONCENTRATION_REFERENCES["additive_mM"])
            descriptor = additive_descriptors.get(additive_name)
            if descriptor is None:
                raise ValueError(f"No descriptor metadata for additive {additive_name!r}")
            for column, value in descriptor.items():
                if column in record:
                    record[column] = np.nan if value is None else float(value)
        if active["salt"]:
            record["salt_conc_log1p_mM"] = concentration_log1p(salt_mm, CONCENTRATION_REFERENCES["salt_mM"])
            descriptor = salt_descriptors.get(salt_name)
            if descriptor is None:
                raise ValueError(f"No descriptor metadata for salt {salt_name!r}")
            cation_charge = float(descriptor["cation_charge"])
            anion_charge = float(descriptor["anion_charge"])
            cation_stoich = float(descriptor["cation_stoich"])
            anion_stoich = float(descriptor["anion_stoich"])
            concentration_m = salt_mm / 1000.0
            record.update(
                {
                    "cation_charge": cation_charge,
                    "anion_charge": anion_charge,
                    "cation_hofmeister_index": descriptor.get("cation_hofmeister_index"),
                    "anion_hofmeister_index": descriptor.get("anion_hofmeister_index"),
                    "ionic_strength_m": 0.5
                    * (cation_stoich * cation_charge**2 + anion_stoich * anion_charge**2)
                    * concentration_m,
                }
            )
            cation_index = descriptor.get("cation_hofmeister_index")
            anion_index = descriptor.get("anion_hofmeister_index")
            record["hofmeister_delta"] = (
                float(anion_index) - float(cation_index)
                if anion_index is not None and cation_index is not None
                else np.nan
            )
        rows.append(record)
    return _feature_frame(rows, DESCRIPTOR_23_COLUMNS, presence)


def build_two_slot37(
    conditions: pd.DataFrame,
    chemical_metadata: dict[str, Any],
) -> pd.DataFrame:
    pca = chemical_metadata["molformer_pca16"]
    rows: list[dict[str, float]] = []
    presence = {group: [] for group in ("polymer", "additive", "salt")}
    for row in conditions.itertuples(index=False):
        additive_name = str(row.additive_name)
        additive_mm = _number(row.additive_concentration_mM)
        salt_name = str(row.salt_name)
        salt_mm = _number(row.salt_concentration_mM)
        active = {
            "polymer": ingredient_present(row.polymer_name, row.polymer_concentration_mg_ml, "polymer"),
            "additive": ingredient_present(additive_name, row.additive_concentration_mM, "additive"),
            "salt": ingredient_present(salt_name, row.salt_concentration_mM, "salt"),
        }
        for group, present in active.items():
            presence[group].append(present)
        record = {column: 0.0 for column in TWO_SLOT_37_COLUMNS}
        if active["polymer"]:
            record["dex_f_percent"] = _number(row.polymer_functionalization_percent)
            record["dex_mw_kda"] = _number(row.polymer_mw_kda)
            record["dex_conc_log1p_mg_ml"] = concentration_log1p(row.polymer_concentration_mg_ml, CONCENTRATION_REFERENCES["polymer_mg_ml"])
        if active["additive"]:
            vector = pca.get(additive_name)
            if vector is None:
                raise ValueError(f"No two-slot metadata for additive {additive_name!r}")
            record["slot1_additive_conc_log1p_mM"] = concentration_log1p(additive_mm, CONCENTRATION_REFERENCES["additive_mM"])
            record.update(dict(zip(SLOT1_PCA_COLUMNS, map(float, vector))))
        if active["salt"]:
            vector = pca.get(salt_name)
            if vector is None:
                raise ValueError(f"No two-slot metadata for salt {salt_name!r}")
            record["slot2_salt_conc_log1p_mM"] = concentration_log1p(salt_mm, CONCENTRATION_REFERENCES["salt_mM"])
            record.update(dict(zip(SLOT2_PCA_COLUMNS, map(float, vector))))
        rows.append(record)
    return _feature_frame(rows, TWO_SLOT_37_COLUMNS, presence)


def add_buffer_feature(features: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    result[BUFFER_COLUMN] = (
        conditions["buffer"].fillna("").astype(str).str.strip().str.lower().eq("di water").astype(float).to_numpy()
    )
    return result
