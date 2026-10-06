"""Hierarchical polymer-feature importance for the retained LCST pipelines."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lcst-polymer-hierarchical-mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost

from . import feature_importance as four_block
from . import report_evaluation as report
from .modeling import CLASS_LABELS, TARGET_COLUMNS
from .schema import MASTER_PATH


OUTPUT_DIR = report.REPO_ROOT / "outputs" / "polymer_hierarchical_importance"
SHARD_DIR = OUTPUT_DIR / "shards"
CONTRACT_VERSION = 1

POLYMER_FEATURE_NAMES = ("functionalization", "molecular_weight", "concentration")
POLYMER_FEATURE_COLUMNS = {
    "functionalization": "dex_f_percent",
    "molecular_weight": "dex_mw_kda",
    "concentration": "dex_conc_log1p_mg_ml",
}
POLYMER_RAW_COLUMNS = {
    "functionalization": "polymer_functionalization_percent",
    "molecular_weight": "polymer_mw_kda",
    "concentration": "polymer_concentration_mg_ml",
}
POLYMER_FEATURE_LABELS = {
    "functionalization": "Functionalization",
    "molecular_weight": "Molecular weight",
    "concentration": "Polymer concentration",
}
OUTER_GROUP_NAMES = ("additive", "salt", "buffer")


def _all_subsets(names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset
        for size in range(len(names) + 1)
        for subset in itertools.combinations(names, size)
    )


OUTER_STATES = _all_subsets(OUTER_GROUP_NAMES)
POLYMER_STATES = _all_subsets(POLYMER_FEATURE_NAMES)
HIERARCHICAL_STATES = tuple(
    (outer, polymer) for outer in OUTER_STATES for polymer in POLYMER_STATES
)


def _subset_id(subset: Sequence[str]) -> str:
    return "empty" if not subset else "+".join(subset)


def state_id(outer: Sequence[str], polymer: Sequence[str]) -> str:
    return f"o-{_subset_id(outer)}__p-{_subset_id(polymer)}"


STATE_IDS = tuple(state_id(*state) for state in HIERARCHICAL_STATES)
STATE_BY_ID = dict(zip(STATE_IDS, HIERARCHICAL_STATES))
FULL_POLYMER_STATE = POLYMER_STATES[-1]
FULL_OUTER_STATE = OUTER_STATES[-1]
NULL_STATE_ID = state_id((), ())
FULL_STATE_ID = state_id(FULL_OUTER_STATE, FULL_POLYMER_STATE)
BOUNDARY_STATES = tuple(
    (outer, polymer)
    for outer in OUTER_STATES
    for polymer in ((), FULL_POLYMER_STATE)
)
BOUNDARY_STATE_IDS = tuple(state_id(*state) for state in BOUNDARY_STATES)
INTERMEDIATE_STATES = tuple(
    state for state in HIERARCHICAL_STATES if state not in BOUNDARY_STATES
)
INTERMEDIATE_STATE_IDS = tuple(state_id(*state) for state in INTERMEDIATE_STATES)

PREDICTION_COLUMNS = (
    "task",
    "model_id",
    "feature_set",
    "model_family",
    "seed",
    "fold",
    "state_id",
    "outer_coalition_id",
    "polymer_coalition_id",
    "outer_groups",
    "polymer_features",
    "n_outer_groups",
    "n_polymer_features",
    "reused_boundary",
    "condition_id",
    "polymer_name",
    "dexma_present",
    "selected_epoch",
    "target_c",
    "prediction_c",
    "target_lcst",
    "prediction_lcst",
    "target_ucst",
    "prediction_ucst",
    "target_none",
    "prediction_none",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def shard_path(model_id: str, seed: int) -> Path:
    return SHARD_DIR / f"{model_id}__seed_{seed}.csv"


def shard_manifest_path(model_id: str, seed: int) -> Path:
    return SHARD_DIR / f"{model_id}__seed_{seed}.manifest.json"


def expected_shards() -> tuple[Path, ...]:
    return tuple(
        shard_path(model_id, seed)
        for model_id in report.MODEL_DEFINITIONS
        for seed in report.STATIC_SEEDS
    )


def _ordered_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    state_order = {identifier: index for index, identifier in enumerate(STATE_IDS)}
    ordered = frame.assign(
        _state_order=frame["state_id"].map(state_order),
        _condition_order=frame.groupby("state_id", sort=False).cumcount(),
    ).sort_values(["_state_order", "_condition_order"], kind="stable")
    return ordered.drop(columns=["_state_order", "_condition_order"]).reset_index(drop=True)


def _boundary_source_coalition(
    outer: tuple[str, ...], polymer: tuple[str, ...]
) -> tuple[str, ...]:
    if polymer not in ((), FULL_POLYMER_STATE):
        raise ValueError("Only all-or-none polymer states map to four-block coalitions")
    included = set(outer)
    if polymer:
        included.add("polymer")
    return tuple(group for group in four_block.GROUP_NAMES if group in included)


def state_columns(
    feature_set: str,
    outer: Sequence[str],
    polymer: Sequence[str],
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    groups = four_block.feature_groups(feature_set)
    wanted = {POLYMER_FEATURE_COLUMNS[name] for name in polymer}
    for group in outer:
        wanted.update(groups[group])
    return tuple(column for column in feature_columns if column in wanted)


def validate_hierarchy_contract(feature_set: str, feature_columns: Sequence[str]) -> None:
    if len(OUTER_STATES) != 8 or len(POLYMER_STATES) != 8:
        raise AssertionError("The hierarchy must contain eight outer and eight polymer states")
    if len(HIERARCHICAL_STATES) != 64 or len(INTERMEDIATE_STATES) != 48:
        raise AssertionError("The hierarchy must contain 64 states and 48 intermediate states")
    if len(STATE_IDS) != len(set(STATE_IDS)):
        raise AssertionError("Hierarchical state IDs are not unique")
    groups = four_block.feature_groups(feature_set)
    polymer_columns = tuple(POLYMER_FEATURE_COLUMNS.values())
    if polymer_columns != groups["polymer"]:
        raise AssertionError(f"Polymer columns changed for {feature_set}")
    outer_columns = tuple(
        column for group in OUTER_GROUP_NAMES for column in groups[group]
    )
    if set(polymer_columns) & set(outer_columns):
        raise AssertionError("Polymer subfeatures overlap the outer blocks")
    if tuple(polymer_columns + outer_columns) != tuple(feature_columns):
        raise AssertionError(f"Hierarchy does not preserve feature order for {feature_set}")
    for outer, polymer in HIERARCHICAL_STATES:
        columns = state_columns(feature_set, outer, polymer, feature_columns)
        expected = {
            *(POLYMER_FEATURE_COLUMNS[name] for name in polymer),
            *(column for group in outer for column in groups[group]),
        }
        if set(columns) != expected or len(columns) != len(expected):
            raise AssertionError(f"Incorrect columns for {state_id(outer, polymer)}")


def _outer_weight(outer: Sequence[str]) -> float:
    size = len(outer)
    return math.factorial(size) * math.factorial(3 - size) / math.factorial(4)


def _inner_weight(polymer: Sequence[str]) -> float:
    size = len(polymer)
    return math.factorial(size) * math.factorial(2 - size) / math.factorial(3)


def hierarchical_values(
    value_by_state: Mapping[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray | float],
) -> dict[str, np.ndarray | float]:
    """Return exact Owen values for the three features inside the polymer block."""

    if set(value_by_state) != set(HIERARCHICAL_STATES):
        raise ValueError("Hierarchical allocation requires all 64 states")
    output: dict[str, np.ndarray | float] = {}
    for feature in POLYMER_FEATURE_NAMES:
        total: np.ndarray | float | None = None
        for outer in OUTER_STATES:
            outer_weight = _outer_weight(outer)
            for polymer in POLYMER_STATES:
                if feature in polymer:
                    continue
                expanded = tuple(
                    name
                    for name in POLYMER_FEATURE_NAMES
                    if name in set(polymer) | {feature}
                )
                increment = outer_weight * _inner_weight(polymer) * (
                    value_by_state[(outer, expanded)]
                    - value_by_state[(outer, polymer)]
                )
                total = increment if total is None else total + increment
        if total is None:
            raise AssertionError(f"No hierarchical value calculated for {feature}")
        output[feature] = total
    return output


def parent_polymer_value(
    value_by_state: Mapping[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray | float],
) -> np.ndarray | float:
    total: np.ndarray | float | None = None
    for outer in OUTER_STATES:
        increment = _outer_weight(outer) * (
            value_by_state[(outer, FULL_POLYMER_STATE)]
            - value_by_state[(outer, ())]
        )
        total = increment if total is None else total + increment
    if total is None:
        raise AssertionError("No parent polymer value calculated")
    return total


def _self_check_values(
    function: Any,
) -> dict[tuple[tuple[str, ...], tuple[str, ...]], float]:
    return {
        state: float(function(set(state[0]), set(state[1])))
        for state in HIERARCHICAL_STATES
    }


def self_check() -> dict[str, bool]:
    """Run mathematical and feature-contract checks without fitting models."""

    for feature_set in ("descriptor24", "two_slot38"):
        columns = (
            [*four_block.DESCRIPTOR_23_COLUMNS, four_block.BUFFER_COLUMN]
            if feature_set == "descriptor24"
            else [*four_block.TWO_SLOT_37_COLUMNS, four_block.BUFFER_COLUMN]
        )
        validate_hierarchy_contract(feature_set, columns)
    if not np.isclose(sum(_outer_weight(state) for state in OUTER_STATES), 1.0):
        raise AssertionError("Outer Shapley weights do not sum to one")
    for feature in POLYMER_FEATURE_NAMES:
        if not np.isclose(
            sum(
                _inner_weight(state)
                for state in POLYMER_STATES
                if feature not in state
            ),
            1.0,
        ):
            raise AssertionError("Inner Shapley weights do not sum to one")

    additive = _self_check_values(
        lambda _outer, polymer: sum(
            {"functionalization": 1.0, "molecular_weight": 2.0, "concentration": 3.0}[name]
            for name in polymer
        )
    )
    observed = hierarchical_values(additive)
    expected = {"functionalization": 1.0, "molecular_weight": 2.0, "concentration": 3.0}
    if any(not np.isclose(observed[name], expected[name]) for name in expected):
        raise AssertionError("Additive toy allocation failed")

    interaction = _self_check_values(
        lambda _outer, polymer: 6.0 if set(POLYMER_FEATURE_NAMES) <= polymer else 0.0
    )
    observed = hierarchical_values(interaction)
    if any(not np.isclose(observed[name], 2.0) for name in POLYMER_FEATURE_NAMES):
        raise AssertionError("Shared interaction toy allocation failed")

    redundancy = _self_check_values(
        lambda _outer, polymer: 1.0 if polymer else 0.0
    )
    observed = hierarchical_values(redundancy)
    if any(not np.isclose(observed[name], 1.0 / 3.0) for name in POLYMER_FEATURE_NAMES):
        raise AssertionError("Redundancy toy allocation failed")

    negative = _self_check_values(
        lambda _outer, polymer: sum(
            {"functionalization": -2.0, "molecular_weight": 1.0, "concentration": 0.5}[name]
            for name in polymer
        )
    )
    observed = hierarchical_values(negative)
    expected = {"functionalization": -2.0, "molecular_weight": 1.0, "concentration": 0.5}
    if any(not np.isclose(observed[name], expected[name]) for name in expected):
        raise AssertionError("Negative toy allocation failed")

    outer_interaction = _self_check_values(
        lambda outer, polymer: 4.0
        if "additive" in outer and "functionalization" in polymer
        else 0.0
    )
    observed = hierarchical_values(outer_interaction)
    if not np.isclose(observed["functionalization"], 2.0):
        raise AssertionError("Outer interaction allocation failed")
    if not np.isclose(
        sum(float(observed[name]) for name in POLYMER_FEATURE_NAMES),
        float(parent_polymer_value(outer_interaction)),
    ):
        raise AssertionError("Hierarchical efficiency failed")
    return {
        "eight_outer_states": True,
        "eight_polymer_states": True,
        "sixty_four_hierarchical_states": True,
        "forty_eight_intermediate_states": True,
        "nonoverlapping_ordered_feature_contract": True,
        "additive_toy": True,
        "interaction_toy": True,
        "redundancy_toy": True,
        "negative_toy": True,
        "outer_interaction_toy": True,
    }


def _state_prediction_frame(
    conditions: pd.DataFrame,
    task: str,
    model_id: str,
    seed: int,
    folds: np.ndarray,
    outer: tuple[str, ...],
    polymer: tuple[str, ...],
    prediction: np.ndarray,
    epochs: np.ndarray,
    *,
    reused_boundary: bool,
) -> pd.DataFrame:
    definition = report.MODEL_DEFINITIONS[model_id]
    frame = pd.DataFrame(
        {
            "task": task,
            "model_id": model_id,
            "feature_set": definition["feature_set"],
            "model_family": definition["family"],
            "seed": seed,
            "fold": folds,
            "state_id": state_id(outer, polymer),
            "outer_coalition_id": _subset_id(outer),
            "polymer_coalition_id": _subset_id(polymer),
            "outer_groups": ";".join(outer),
            "polymer_features": ";".join(polymer),
            "n_outer_groups": len(outer),
            "n_polymer_features": len(polymer),
            "reused_boundary": reused_boundary,
            "condition_id": conditions["condition_id"].astype(str).to_numpy(),
            "polymer_name": conditions["polymer_name"].astype(str).to_numpy(),
            "dexma_present": conditions["polymer_name"].eq("Dex-MA").to_numpy(),
            "selected_epoch": np.asarray(epochs, dtype=float),
        }
    )
    for column in PREDICTION_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    if task == "regression":
        frame["target_c"] = conditions["transition_temperature_c"].to_numpy(dtype=float)
        frame["prediction_c"] = np.asarray(prediction, dtype=float)
    else:
        target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        values = np.asarray(prediction, dtype=float)
        for index, label in enumerate(CLASS_LABELS):
            lower = label.lower()
            frame[f"target_{lower}"] = target[:, index]
            frame[f"prediction_{lower}"] = values[:, index]
    return frame[list(PREDICTION_COLUMNS)]


def _source_shard(
    model_id: str,
    seed: int,
    conditions: pd.DataFrame,
    features: pd.DataFrame,
    fold_assignments: np.ndarray,
) -> pd.DataFrame:
    path = four_block.shard_path(model_id, seed)
    manifest_path = four_block.shard_manifest_path(model_id, seed)
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Completed four-block shard is required before hierarchy fitting: {path}"
        )
    frame = pd.read_csv(path)
    four_block.validate_shard(frame, model_id, seed, conditions)
    expected_manifest = four_block._expected_shard_manifest(
        model_id, seed, conditions, features, fold_assignments
    )
    four_block._validate_shard_manifest(manifest_path, expected_manifest)
    return frame


def _boundary_frames(
    source: pd.DataFrame,
    conditions: pd.DataFrame,
    task: str,
    model_id: str,
    seed: int,
    folds: np.ndarray,
) -> pd.DataFrame:
    condition_ids = conditions["condition_id"].astype(str).tolist()
    rows: list[pd.DataFrame] = []
    for outer, polymer in BOUNDARY_STATES:
        coalition = _boundary_source_coalition(outer, polymer)
        identifier = four_block.coalition_id(coalition)
        subset = source.loc[source["coalition_id"].eq(identifier)].copy()
        subset["condition_id"] = subset["condition_id"].astype(str)
        subset = subset.set_index("condition_id").loc[condition_ids].reset_index()
        prediction = (
            subset["prediction_c"].to_numpy(dtype=float)
            if task == "regression"
            else subset[
                [f"prediction_{label.lower()}" for label in CLASS_LABELS]
            ].to_numpy(dtype=float)
        )
        rows.append(
            _state_prediction_frame(
                conditions,
                task,
                model_id,
                seed,
                folds,
                outer,
                polymer,
                prediction,
                subset["selected_epoch"].to_numpy(dtype=float),
                reused_boundary=True,
            )
        )
    return _ordered_state_frame(pd.concat(rows, ignore_index=True))


def _shard_manifest_base(
    model_id: str,
    seed: int,
    conditions: pd.DataFrame,
    features: pd.DataFrame,
    fold_assignments: np.ndarray,
) -> dict[str, Any]:
    source_path = four_block.shard_path(model_id, seed)
    source_manifest = four_block.shard_manifest_path(model_id, seed)
    definition = report.MODEL_DEFINITIONS[model_id]
    target_columns = (
        ["condition_id", "transition_temperature_c"]
        if definition["task"] == "regression"
        else ["condition_id", *TARGET_COLUMNS]
    )
    payload = {
        "contract_version": CONTRACT_VERSION,
        "model_id": model_id,
        "model_definition": definition,
        "model_protocol": four_block._model_protocol(model_id, seed),
        "seed": seed,
        "master_sha256": _sha256(MASTER_PATH),
        "condition_id_sha256": four_block._frame_sha256(conditions, ["condition_id"]),
        "target_sha256": four_block._frame_sha256(conditions, target_columns),
        "fold_assignments_sha256": hashlib.sha256(
            np.asarray(fold_assignments, dtype=np.int64).tobytes()
        ).hexdigest(),
        "feature_matrix_sha256": four_block._frame_sha256(
            features, list(features.columns)
        ),
        "polymer_features": POLYMER_FEATURE_COLUMNS,
        "outer_groups": {
            group: list(columns)
            for group, columns in four_block.feature_groups(
                definition["feature_set"]
            ).items()
            if group in OUTER_GROUP_NAMES
        },
        "state_ids": list(STATE_IDS),
        "boundary_state_ids": list(BOUNDARY_STATE_IDS),
        "intermediate_state_ids": list(INTERMEDIATE_STATE_IDS),
        "source_four_block_shard": str(source_path),
        "source_four_block_shard_sha256": _sha256(source_path),
        "source_four_block_manifest": str(source_manifest),
        "source_four_block_manifest_sha256": _sha256(source_manifest),
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def _write_shard_manifest(
    path: Path, base: Mapping[str, Any], completed_state_ids: Sequence[str]
) -> None:
    payload = dict(base)
    payload["completed_state_ids"] = list(completed_state_ids)
    payload["complete"] = set(completed_state_ids) == set(STATE_IDS)
    _atomic_text(json.dumps(payload, indent=2) + "\n", path)


def _validate_shard_manifest(
    path: Path, base: Mapping[str, Any], completed_state_ids: Sequence[str]
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing hierarchical shard manifest: {path}")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(base)
    expected["completed_state_ids"] = list(completed_state_ids)
    expected["complete"] = set(completed_state_ids) == set(STATE_IDS)
    if observed != expected:
        raise AssertionError(f"Stale or incompatible hierarchical shard manifest: {path}")


def validate_shard(
    frame: pd.DataFrame,
    model_id: str,
    seed: int,
    conditions: pd.DataFrame,
    *,
    require_complete: bool,
) -> tuple[str, ...]:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise AssertionError("Hierarchical shard columns changed")
    if frame.empty:
        raise AssertionError("Hierarchical shard is empty")
    definition = report.MODEL_DEFINITIONS[model_id]
    task = definition["task"]
    if frame["model_id"].nunique() != 1 or frame["model_id"].iloc[0] != model_id:
        raise AssertionError("Hierarchical shard model ID is invalid")
    if frame["seed"].nunique() != 1 or int(frame["seed"].iloc[0]) != seed:
        raise AssertionError("Hierarchical shard seed is invalid")
    if frame["task"].nunique() != 1 or frame["task"].iloc[0] != task:
        raise AssertionError("Hierarchical shard task is invalid")
    identifiers = tuple(
        identifier for identifier in STATE_IDS if identifier in set(frame["state_id"])
    )
    if set(frame["state_id"]) - set(STATE_IDS):
        raise AssertionError("Hierarchical shard contains an unknown state")
    if require_complete and set(identifiers) != set(STATE_IDS):
        raise AssertionError(
            f"Hierarchical shard has {len(identifiers)} rather than 64 states"
        )
    expected_ids = conditions["condition_id"].astype(str).tolist()
    _, expected_folds = four_block._folds(conditions, task, seed)
    if frame[["state_id", "condition_id"]].duplicated().any():
        raise AssertionError("Hierarchical shard contains duplicate predictions")
    for identifier in identifiers:
        outer, polymer = STATE_BY_ID[identifier]
        subset = frame.loc[frame["state_id"].eq(identifier)].copy()
        if len(subset) != len(conditions):
            raise AssertionError(f"State {identifier} is incomplete")
        if not subset["outer_coalition_id"].eq(_subset_id(outer)).all():
            raise AssertionError(f"Incorrect outer ID for {identifier}")
        if not subset["polymer_coalition_id"].eq(_subset_id(polymer)).all():
            raise AssertionError(f"Incorrect polymer ID for {identifier}")
        if not subset["n_outer_groups"].eq(len(outer)).all():
            raise AssertionError(f"Incorrect outer size for {identifier}")
        if not subset["n_polymer_features"].eq(len(polymer)).all():
            raise AssertionError(f"Incorrect polymer size for {identifier}")
        if not subset["reused_boundary"].eq(identifier in BOUNDARY_STATE_IDS).all():
            raise AssertionError(f"Incorrect boundary flag for {identifier}")
        subset["condition_id"] = subset["condition_id"].astype(str)
        if set(subset["condition_id"]) != set(expected_ids):
            raise AssertionError(f"State {identifier} has different condition IDs")
        subset = subset.set_index("condition_id").loc[expected_ids]
        if not np.array_equal(subset["fold"].to_numpy(dtype=int), expected_folds):
            raise AssertionError(f"State {identifier} has different fold assignments")
        if task == "regression":
            target = conditions["transition_temperature_c"].to_numpy(dtype=float)
            if not np.allclose(subset["target_c"].to_numpy(dtype=float), target):
                raise AssertionError(f"State {identifier} has different targets")
        else:
            target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
            observed = subset[
                [f"target_{label.lower()}" for label in CLASS_LABELS]
            ].to_numpy(dtype=float)
            if not np.allclose(observed, target):
                raise AssertionError(f"State {identifier} has different targets")
    if task == "regression":
        if not np.isfinite(frame["prediction_c"]).all():
            raise AssertionError("Hierarchical regression shard has non-finite predictions")
    else:
        probabilities = frame[
            [f"prediction_{label.lower()}" for label in CLASS_LABELS]
        ].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-7
        ):
            raise AssertionError("Hierarchical classification probabilities are invalid")
    return identifiers


def _fit_state(
    model_id: str,
    seed: int,
    conditions: pd.DataFrame,
    features: pd.DataFrame,
    split_rows: Sequence[tuple[int, np.ndarray, np.ndarray]],
    fold_assignments: np.ndarray,
    outer: tuple[str, ...],
    polymer: tuple[str, ...],
) -> pd.DataFrame:
    definition = report.MODEL_DEFINITIONS[model_id]
    task = definition["task"]
    columns = state_columns(
        definition["feature_set"], outer, polymer, features.columns
    )
    if not columns:
        raise AssertionError("The null state must be reused rather than fitted")
    output_shape = (len(conditions), 3) if task == "classification" else (len(conditions),)
    prediction = np.full(output_shape, np.nan, dtype=float)
    epochs = np.full(len(conditions), np.nan, dtype=float)
    for fold, train_index, test_index in split_rows:
        fitted, epoch = report.fit_model(
            task,
            model_id,
            features.iloc[train_index].loc[:, list(columns)],
            conditions.iloc[train_index],
            features.iloc[test_index].loc[:, list(columns)],
            seed,
        )
        prediction[test_index] = fitted
        epochs[test_index] = epoch
    return _state_prediction_frame(
        conditions,
        task,
        model_id,
        seed,
        fold_assignments,
        outer,
        polymer,
        prediction,
        epochs,
        reused_boundary=False,
    )


def run_shard(
    model_id: str,
    seed: int,
    *,
    state: str | None = None,
    force: bool = False,
) -> Path:
    """Fit missing hierarchical states for one model and outer-CV seed."""

    self_check()
    if model_id not in report.MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model ID: {model_id}")
    if seed not in report.STATIC_SEEDS:
        raise ValueError(f"Seed must be one of {report.STATIC_SEEDS}")
    if state is not None and state not in STATE_BY_ID:
        raise ValueError(f"Unknown hierarchical state: {state}")
    definition = report.MODEL_DEFINITIONS[model_id]
    task = definition["task"]
    conditions, tables, _ = four_block._task_inputs(task)
    features = tables[definition["feature_set"]]
    validate_hierarchy_contract(definition["feature_set"], features.columns)
    split_rows, fold_assignments = four_block._folds(conditions, task, seed)
    source = _source_shard(model_id, seed, conditions, features, fold_assignments)
    boundaries = _boundary_frames(
        source, conditions, task, model_id, seed, fold_assignments
    )
    base_manifest = _shard_manifest_base(
        model_id, seed, conditions, features, fold_assignments
    )
    path = shard_path(model_id, seed)
    manifest_path = shard_manifest_path(model_id, seed)

    existing: pd.DataFrame
    if path.exists() and not (force and state is None):
        existing = pd.read_csv(path)
        completed = validate_shard(
            existing, model_id, seed, conditions, require_complete=False
        )
        _validate_shard_manifest(manifest_path, base_manifest, completed)
        if force and state is not None:
            existing = existing.loc[~existing["state_id"].eq(state)].copy()
            if state in BOUNDARY_STATE_IDS:
                replacement = boundaries.loc[boundaries["state_id"].eq(state)]
                existing = pd.concat([existing, replacement], ignore_index=True)
    else:
        existing = boundaries.copy()
        completed = validate_shard(
            existing, model_id, seed, conditions, require_complete=False
        )
        _atomic_csv(existing, path)
        _write_shard_manifest(manifest_path, base_manifest, completed)

    requested = [state] if state is not None else list(INTERMEDIATE_STATE_IDS)
    started = time.monotonic()
    torch.set_num_threads(1)
    for identifier in requested:
        if identifier in BOUNDARY_STATE_IDS:
            continue
        if identifier in set(existing["state_id"]) and not force:
            continue
        outer, polymer = STATE_BY_ID[identifier]
        fitted = _fit_state(
            model_id,
            seed,
            conditions,
            features,
            split_rows,
            fold_assignments,
            outer,
            polymer,
        )
        existing = existing.loc[~existing["state_id"].eq(identifier)]
        existing = _ordered_state_frame(pd.concat([existing, fitted], ignore_index=True))
        completed = validate_shard(
            existing, model_id, seed, conditions, require_complete=False
        )
        _atomic_csv(existing, path)
        _write_shard_manifest(manifest_path, base_manifest, completed)
        print(
            f"Checkpointed {path.name}: {len(completed)}/64 states",
            flush=True,
        )
    completed = validate_shard(
        existing, model_id, seed, conditions, require_complete=False
    )
    elapsed = time.monotonic() - started
    print(
        f"Hierarchical shard {path.name}: {len(completed)}/64 states, "
        f"this run {elapsed:.1f} seconds",
        flush=True,
    )
    return path


def _loss_rows(frame: pd.DataFrame, *, average_seeds: bool) -> pd.DataFrame:
    task = str(frame["task"].iloc[0])
    prediction_columns = (
        ["prediction_c"]
        if task == "regression"
        else [f"prediction_{label.lower()}" for label in CLASS_LABELS]
    )
    target_columns = (
        ["target_c"]
        if task == "regression"
        else [f"target_{label.lower()}" for label in CLASS_LABELS]
    )
    group_columns = ["state_id", "condition_id", "polymer_name", "dexma_present"]
    if not average_seeds:
        group_columns.insert(0, "seed")
    aggregation = {column: "mean" for column in prediction_columns}
    aggregation.update({column: "first" for column in target_columns})
    values = frame.groupby(group_columns, sort=False, as_index=False).agg(aggregation)
    if task == "regression":
        values["loss"] = np.abs(values["target_c"] - values["prediction_c"])
    else:
        target = values[target_columns].to_numpy(dtype=float)
        predicted = np.clip(values[prediction_columns].to_numpy(dtype=float), 1e-15, 1.0)
        predicted /= predicted.sum(axis=1, keepdims=True)
        values["loss"] = -np.sum(
            np.where(target > 0, target * np.log(predicted), 0.0), axis=1
        )
    return values


def _population_loss_rows(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary: list[pd.DataFrame] = []
    by_seed: list[pd.DataFrame] = []
    for model_id, definition in report.MODEL_DEFINITIONS.items():
        model = predictions.loc[predictions["model_id"].eq(model_id)]
        populations = [("all", model)]
        if definition["task"] == "classification":
            populations.append(("dexma_only", model.loc[model["dexma_present"]]))
        for population, subset in populations:
            averaged = _loss_rows(subset, average_seeds=True)
            averaged.insert(0, "scoring_population", population)
            averaged.insert(0, "model_family", definition["family"])
            averaged.insert(0, "feature_set", definition["feature_set"])
            averaged.insert(0, "model_id", model_id)
            averaged.insert(0, "task", definition["task"])
            primary.append(averaged)
            seeded = _loss_rows(subset, average_seeds=False)
            seeded.insert(0, "scoring_population", population)
            seeded.insert(0, "model_family", definition["family"])
            seeded.insert(0, "feature_set", definition["feature_set"])
            seeded.insert(0, "model_id", model_id)
            seeded.insert(0, "task", definition["task"])
            by_seed.append(seeded)
    return pd.concat(primary, ignore_index=True), pd.concat(by_seed, ignore_index=True)


def _state_loss_summary(losses: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "task",
        "model_id",
        "feature_set",
        "model_family",
        "scoring_population",
        "state_id",
    ]
    if "seed" in losses.columns:
        group_columns.insert(5, "seed")
    summary = losses.groupby(group_columns, sort=False, as_index=False).agg(
        loss=("loss", "mean"), n_conditions=("condition_id", "nunique")
    )
    summary["outer_coalition_id"] = summary["state_id"].map(
        lambda identifier: _subset_id(STATE_BY_ID[identifier][0])
    )
    summary["polymer_coalition_id"] = summary["state_id"].map(
        lambda identifier: _subset_id(STATE_BY_ID[identifier][1])
    )
    lookup_columns = [
        column for column in group_columns if column != "state_id"
    ]
    null_lookup = summary.loc[summary["state_id"].eq(NULL_STATE_ID)].set_index(
        lookup_columns
    )["loss"]
    summary["null_loss"] = [
        null_lookup.loc[tuple(row[column] for column in lookup_columns)]
        for _, row in summary.iterrows()
    ]
    summary["predictive_value"] = summary["null_loss"] - summary["loss"]
    return summary


def _state_values_from_pivot(pivot: pd.DataFrame) -> dict[
    tuple[tuple[str, ...], tuple[str, ...]], np.ndarray
]:
    null = pivot[NULL_STATE_ID].to_numpy(dtype=float)
    return {
        state: null - pivot[state_id(*state)].to_numpy(dtype=float)
        for state in HIERARCHICAL_STATES
    }


def _importance_tables(
    primary_losses: pd.DataFrame,
    seed_losses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    current_condition_path = (
        four_block.OUTPUT_DIR / "condition_group_contributions.csv"
    )
    current_seed_path = four_block.OUTPUT_DIR / "shapley_importance_by_seed.csv"
    if not current_condition_path.exists() or not current_seed_path.exists():
        raise FileNotFoundError("Compiled four-block importance tables are required")
    current_condition = pd.read_csv(current_condition_path)
    current_condition = current_condition.loc[current_condition["group"].eq("polymer")]
    current_seed = pd.read_csv(current_seed_path)
    current_seed = current_seed.loc[current_seed["group"].eq("polymer")]

    importance_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    keys = ["task", "model_id", "feature_set", "model_family", "scoring_population"]
    max_parent_error = 0.0
    max_bootstrap_error = 0.0
    for values, frame in primary_losses.groupby(keys, sort=False):
        task, model_id, feature_set, family, population = values
        pivot = frame.pivot(index="condition_id", columns="state_id", values="loss")
        pivot = pivot.reindex(columns=STATE_IDS).sort_index()
        if pivot.isna().any().any():
            raise AssertionError(f"Incomplete hierarchical losses for {model_id}/{population}")
        contributions = hierarchical_values(_state_values_from_pivot(pivot))
        parent = np.asarray(parent_polymer_value(_state_values_from_pivot(pivot)), dtype=float)
        child_sum = np.sum(
            np.vstack([np.asarray(contributions[name], dtype=float) for name in POLYMER_FEATURE_NAMES]),
            axis=0,
        )
        max_parent_error = max(max_parent_error, float(np.max(np.abs(child_sum - parent))))
        expected = current_condition.loc[
            current_condition["model_id"].eq(model_id)
            & current_condition["scoring_population"].eq(population)
        ].copy()
        expected["condition_id"] = expected["condition_id"].astype(str)
        expected = expected.set_index("condition_id").reindex(pivot.index)
        if expected["contribution"].isna().any():
            raise AssertionError(f"Missing four-block parent values for {model_id}/{population}")
        max_parent_error = max(
            max_parent_error,
            float(np.max(np.abs(parent - expected["contribution"].to_numpy(dtype=float)))),
        )
        samples = four_block._bootstrap_indices(task, population, len(pivot))
        parent_bootstrap = parent[samples].mean(axis=1)
        child_bootstrap_sum = np.zeros(len(samples), dtype=float)
        for feature in POLYMER_FEATURE_NAMES:
            contribution = np.asarray(contributions[feature], dtype=float)
            distribution = contribution[samples].mean(axis=1)
            child_bootstrap_sum += distribution
            central = float(contribution.mean())
            importance_rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "model_family": family,
                    "scoring_population": population,
                    "polymer_feature": feature,
                    "importance": central,
                    "ci95_low": float(np.quantile(distribution, 0.025)),
                    "ci95_high": float(np.quantile(distribution, 0.975)),
                    "parent_polymer_importance": float(parent.mean()),
                    "n_conditions": len(pivot),
                }
            )
            condition_rows.extend(
                {
                    "task": task,
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "model_family": family,
                    "scoring_population": population,
                    "condition_id": condition_id,
                    "polymer_feature": feature,
                    "contribution": float(contribution[index]),
                }
                for index, condition_id in enumerate(pivot.index)
            )
        max_bootstrap_error = max(
            max_bootstrap_error,
            float(np.max(np.abs(child_bootstrap_sum - parent_bootstrap))),
        )

    seed_rows: list[dict[str, Any]] = []
    seed_keys = [*keys, "seed"]
    for values, frame in seed_losses.groupby(seed_keys, sort=False):
        task, model_id, feature_set, family, population, seed = values
        losses = frame.groupby("state_id", sort=False)["loss"].mean()
        if set(losses.index) != set(STATE_IDS):
            raise AssertionError(f"Incomplete seed losses for {model_id}/{seed}/{population}")
        null_loss = float(losses.loc[NULL_STATE_ID])
        state_values = {
            state: null_loss - float(losses.loc[state_id(*state)])
            for state in HIERARCHICAL_STATES
        }
        contributions = hierarchical_values(state_values)
        parent = float(parent_polymer_value(state_values))
        expected = current_seed.loc[
            current_seed["model_id"].eq(model_id)
            & current_seed["scoring_population"].eq(population)
            & current_seed["seed"].eq(seed),
            "importance",
        ]
        if len(expected) != 1:
            raise AssertionError(f"Missing seed parent value for {model_id}/{seed}/{population}")
        max_parent_error = max(max_parent_error, abs(parent - float(expected.iloc[0])))
        max_parent_error = max(
            max_parent_error,
            abs(sum(float(contributions[name]) for name in POLYMER_FEATURE_NAMES) - parent),
        )
        for feature in POLYMER_FEATURE_NAMES:
            seed_rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "model_family": family,
                    "scoring_population": population,
                    "seed": int(seed),
                    "polymer_feature": feature,
                    "importance": float(contributions[feature]),
                    "parent_polymer_importance": parent,
                }
            )
    by_seed = pd.DataFrame(seed_rows)
    by_seed["rank"] = by_seed.groupby(
        ["task", "model_id", "scoring_population", "seed"]
    )["importance"].rank(method="dense", ascending=False).astype(int)
    stability = by_seed.groupby(
        [
            "task",
            "model_id",
            "feature_set",
            "model_family",
            "scoring_population",
            "polymer_feature",
        ],
        sort=False,
        as_index=False,
    ).agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        median_rank=("rank", "median"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        top_rank_count=("rank", lambda observed: int((observed == 1).sum())),
    )
    stability["top_rank_fraction"] = stability["top_rank_count"] / len(
        report.STATIC_SEEDS
    )
    return (
        pd.DataFrame(importance_rows),
        by_seed,
        stability,
        pd.DataFrame(condition_rows),
        {
            "maximum_parent_identity_error": max_parent_error,
            "maximum_bootstrap_identity_error": max_bootstrap_error,
        },
    )


def _direct_state_values(pivot: pd.DataFrame) -> dict[
    tuple[tuple[str, ...], tuple[str, ...]], np.ndarray
]:
    return {
        state: pivot[state_id(*state)].to_numpy(dtype=float)
        for state in HIERARCHICAL_STATES
    }


def _feature_value_tables() -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for task in ("regression", "classification"):
        conditions, tables, _ = four_block._task_inputs(task)
        values = conditions[
            ["condition_id", *POLYMER_RAW_COLUMNS.values()]
        ].copy()
        descriptor = tables["descriptor24"]
        for feature, column in POLYMER_FEATURE_COLUMNS.items():
            values[f"encoded_{feature}"] = descriptor[column].to_numpy(dtype=float)
        values["condition_id"] = values["condition_id"].astype(str)
        output[task] = values.set_index("condition_id")
    return output


def _prediction_contributions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    feature_values = _feature_value_tables()
    rows: list[dict[str, Any]] = []
    max_parent_error = 0.0
    max_probability_simplex_error = 0.0
    for model_id, definition in report.MODEL_DEFINITIONS.items():
        task = definition["task"]
        model = predictions.loc[predictions["model_id"].eq(model_id)].copy()
        prediction_columns = (
            ["prediction_c"]
            if task == "regression"
            else [f"prediction_{label.lower()}" for label in CLASS_LABELS]
        )
        target_columns = (
            ["target_c"]
            if task == "regression"
            else [f"target_{label.lower()}" for label in CLASS_LABELS]
        )
        aggregation = {column: "mean" for column in prediction_columns}
        aggregation.update({column: "first" for column in target_columns})
        averaged = model.groupby(
            ["state_id", "condition_id", "polymer_name", "dexma_present"],
            sort=False,
            as_index=False,
        ).agg(aggregation)
        populations = [("all", averaged)]
        if task == "classification":
            populations.append(
                ("dexma_only", averaged.loc[averaged["dexma_present"]])
            )
        for population, subset in populations:
            outputs = ["temperature"] if task == "regression" else [label.lower() for label in CLASS_LABELS]
            class_contributions: dict[str, dict[str, np.ndarray]] = {}
            for output_name in outputs:
                value_column = (
                    "prediction_c" if task == "regression" else f"prediction_{output_name}"
                )
                target_column = "target_c" if task == "regression" else f"target_{output_name}"
                pivot = subset.pivot(
                    index="condition_id", columns="state_id", values=value_column
                ).reindex(columns=STATE_IDS).sort_index()
                targets = subset.groupby("condition_id")[target_column].first().reindex(pivot.index)
                if pivot.isna().any().any() or targets.isna().any():
                    raise AssertionError(
                        f"Incomplete prediction allocation inputs for {model_id}/{population}/{output_name}"
                    )
                state_values = _direct_state_values(pivot)
                contributions = {
                    name: np.asarray(value, dtype=float)
                    for name, value in hierarchical_values(state_values).items()
                }
                class_contributions[output_name] = contributions
                parent = np.asarray(parent_polymer_value(state_values), dtype=float)
                child_sum = np.sum(
                    np.vstack([contributions[name] for name in POLYMER_FEATURE_NAMES]),
                    axis=0,
                )
                max_parent_error = max(
                    max_parent_error,
                    float(np.max(np.abs(child_sum - parent))),
                )
                condition_features = feature_values[task].reindex(pivot.index)
                scale = 1.0 if task == "regression" else 100.0
                unit = "degrees_celsius" if task == "regression" else "percentage_points"
                for feature in POLYMER_FEATURE_NAMES:
                    raw_column = POLYMER_RAW_COLUMNS[feature]
                    encoded_column = f"encoded_{feature}"
                    for index, condition_id in enumerate(pivot.index):
                        raw_value = condition_features.iloc[index][raw_column]
                        encoded_value = condition_features.iloc[index][encoded_column]
                        rows.append(
                            {
                                "task": task,
                                "model_id": model_id,
                                "feature_set": definition["feature_set"],
                                "model_family": definition["family"],
                                "scoring_population": population,
                                "condition_id": condition_id,
                                "output": output_name,
                                "polymer_feature": feature,
                                "contribution": float(contributions[feature][index] * scale),
                                "contribution_unit": unit,
                                "feature_value_raw": (
                                    float(raw_value) if pd.notna(raw_value) else np.nan
                                ),
                                "feature_value_encoded": (
                                    float(encoded_value) if pd.notna(encoded_value) else np.nan
                                ),
                                "feature_missing": bool(pd.isna(encoded_value)),
                                "target": float(targets.iloc[index] * scale),
                                "null_prediction": float(
                                    pivot.iloc[index][NULL_STATE_ID] * scale
                                ),
                                "full_prediction": float(
                                    pivot.iloc[index][FULL_STATE_ID] * scale
                                ),
                            }
                        )
            if task == "classification":
                for feature in POLYMER_FEATURE_NAMES:
                    summed = np.sum(
                        np.vstack(
                            [class_contributions[label.lower()][feature] for label in CLASS_LABELS]
                        ),
                        axis=0,
                    )
                    max_probability_simplex_error = max(
                        max_probability_simplex_error,
                        float(np.max(np.abs(summed))),
                    )
    return pd.DataFrame(rows), {
        "maximum_prediction_parent_identity_error": max_parent_error,
        "maximum_probability_simplex_error": max_probability_simplex_error,
    }


def _support_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for task in ("regression", "classification"):
        conditions, _, _ = four_block._task_inputs(task)
        populations = [("all", conditions)]
        if task == "classification":
            populations.append(
                ("dexma_only", conditions.loc[conditions["polymer_name"].eq("Dex-MA")])
            )
        for population, frame in populations:
            for feature in POLYMER_FEATURE_NAMES:
                column = POLYMER_RAW_COLUMNS[feature]
                observed = pd.to_numeric(frame[column], errors="coerce")
                finite = observed.dropna().to_numpy(dtype=float)
                summary_rows.append(
                    {
                        "task": task,
                        "scoring_population": population,
                        "polymer_feature": feature,
                        "n_conditions": len(frame),
                        "n_observed": int(observed.notna().sum()),
                        "n_missing": int(observed.isna().sum()),
                        "n_unique_including_missing": int(observed.nunique(dropna=False)),
                        "minimum": float(np.min(finite)) if len(finite) else np.nan,
                        "median": float(np.median(finite)) if len(finite) else np.nan,
                        "maximum": float(np.max(finite)) if len(finite) else np.nan,
                    }
                )
            joint_columns = list(POLYMER_RAW_COLUMNS.values())
            joint = (
                frame.groupby(joint_columns, dropna=False, sort=True)
                .size()
                .rename("n_conditions")
                .reset_index()
                .sort_values("n_conditions", ascending=False, kind="stable")
            )
            joint.insert(0, "scoring_population", population)
            joint.insert(0, "task", task)
            joint_rows.extend(joint.to_dict("records"))
            cross = (
                frame.groupby(
                    [
                        POLYMER_RAW_COLUMNS["functionalization"],
                        POLYMER_RAW_COLUMNS["molecular_weight"],
                    ],
                    dropna=False,
                    sort=True,
                )
                .size()
                .rename("n_conditions")
                .reset_index()
            )
            cross.insert(0, "scoring_population", population)
            cross.insert(0, "task", task)
            cross_rows.extend(cross.to_dict("records"))
    return pd.DataFrame(summary_rows), pd.DataFrame(joint_rows), pd.DataFrame(cross_rows)


def _plot_performance(importance: pd.DataFrame) -> None:
    selected = importance.loc[importance["scoring_population"].eq("all")].copy()
    colors = {"XGBoost": "#1f6bc1", "MLP": "#f26f00"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharey=True)
    panel = 0
    for row, task in enumerate(("regression", "classification")):
        row_values = selected.loc[selected["task"].eq(task)]
        finite = row_values[["ci95_low", "ci95_high"]].to_numpy(dtype=float)
        low = min(0.0, float(np.nanmin(finite)))
        high = max(0.0, float(np.nanmax(finite)))
        padding = max((high - low) * 0.10, 0.01)
        for column, feature_set in enumerate(("descriptor24", "two_slot38")):
            ax = axes[row, column]
            subset = row_values.loc[row_values["feature_set"].eq(feature_set)]
            positions = np.arange(len(POLYMER_FEATURE_NAMES), dtype=float)
            for family, offset, marker in (
                ("XGBoost", -0.10, "o"),
                ("MLP", 0.10, "s"),
            ):
                family_values = subset.loc[subset["model_family"].eq(family)].set_index(
                    "polymer_feature"
                )
                values = family_values.loc[list(POLYMER_FEATURE_NAMES), "importance"].to_numpy(dtype=float)
                lows = family_values.loc[list(POLYMER_FEATURE_NAMES), "ci95_low"].to_numpy(dtype=float)
                highs = family_values.loc[list(POLYMER_FEATURE_NAMES), "ci95_high"].to_numpy(dtype=float)
                ax.errorbar(
                    values,
                    positions + offset,
                    xerr=np.vstack([values - lows, highs - values]),
                    fmt=marker,
                    color=colors[family],
                    markersize=5.5,
                    capsize=3,
                    linewidth=1.1,
                    label=family,
                )
            ax.axvline(0.0, color="#555555", linewidth=0.9)
            ax.set_yticks(
                positions,
                [POLYMER_FEATURE_LABELS[name] for name in POLYMER_FEATURE_NAMES],
            )
            ax.grid(axis="x", alpha=0.20)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(low - padding, high + padding)
            if row == 0:
                ax.set_title(
                    "Descriptor representation"
                    if column == 0
                    else "Two-slot representation"
                )
                ax.set_xlabel("Allocated MAE reduction (°C)")
            else:
                ax.set_xlabel("Allocated cross-entropy reduction (nats)")
            ax.text(
                -0.10,
                1.04,
                chr(ord("a") + panel),
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )
            panel += 1
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        OUTPUT_DIR / "polymer_feature_performance.png",
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "LCST hierarchical importance"},
    )
    fig.savefig(
        OUTPUT_DIR / "polymer_feature_performance.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "LCST"},
    )
    plt.close(fig)


def _swarm_offsets(
    values: np.ndarray,
    identifiers: Sequence[str],
    x_limits: tuple[float, float],
    *,
    bins: int = 48,
    maximum: float = 0.28,
) -> np.ndarray:
    low, high = x_limits
    width = max(high - low, np.finfo(float).eps)
    bin_index = np.clip(((values - low) / width * bins).astype(int), 0, bins - 1)
    offsets = np.zeros(len(values), dtype=float)
    for value in np.unique(bin_index):
        members = np.flatnonzero(bin_index == value)
        members = np.asarray(
            sorted(
                members,
                key=lambda index: hashlib.sha256(
                    str(identifiers[index]).encode("utf-8")
                ).hexdigest(),
            ),
            dtype=int,
        )
        count = len(members)
        if count <= 1:
            continue
        if count % 2:
            sequence = [0.0]
            for level in range(1, count // 2 + 1):
                sequence.extend([float(level), float(-level)])
        else:
            sequence = []
            for level in range(count // 2):
                value_level = level + 0.5
                sequence.extend([value_level, -value_level])
        scale = maximum / max(abs(number) for number in sequence)
        offsets[members] = np.asarray(sequence[:count]) * scale
    return offsets


def _feature_colors(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame["feature_value_encoded"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    normalized = np.full(len(values), np.nan, dtype=float)
    if finite.any():
        low = float(np.min(values[finite]))
        high = float(np.max(values[finite]))
        normalized[finite] = (
            0.5 if np.isclose(low, high) else (values[finite] - low) / (high - low)
        )
    return normalized, finite


def _model_plot_order(task: str) -> tuple[str, ...]:
    suffix = "regressor" if task == "regression" else "classifier"
    return (
        f"descriptor24_buffer_{suffix}_xgboost",
        f"two_slot_pca38_buffer_{suffix}_xgboost",
        f"descriptor24_buffer_{suffix}_mlp",
        f"two_slot_pca38_buffer_{suffix}_mlp",
    )


def _model_plot_title(model_id: str) -> str:
    definition = report.MODEL_DEFINITIONS[model_id]
    representation = "Descriptor" if definition["feature_set"] == "descriptor24" else "Two-slot"
    return f"{representation} · {definition['family']}"


def _draw_swarm_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_limits: tuple[float, float],
    *,
    xlabel: str,
) -> None:
    cmap = plt.get_cmap("coolwarm")
    for position, feature in enumerate(POLYMER_FEATURE_NAMES):
        subset = frame.loc[frame["polymer_feature"].eq(feature)].sort_values(
            "condition_id", kind="stable"
        )
        values = subset["contribution"].to_numpy(dtype=float)
        offsets = _swarm_offsets(
            values,
            subset["condition_id"].astype(str).tolist(),
            x_limits,
        )
        normalized, finite = _feature_colors(subset)
        if finite.any():
            ax.scatter(
                values[finite],
                position + offsets[finite],
                c=cmap(normalized[finite]),
                s=9,
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
        if (~finite).any():
            ax.scatter(
                values[~finite],
                position + offsets[~finite],
                color="#8c8c8c",
                s=9,
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
    ax.axvline(0.0, color="#555555", linewidth=0.9)
    ax.set_xlim(*x_limits)
    ax.set_yticks(
        np.arange(len(POLYMER_FEATURE_NAMES)),
        [POLYMER_FEATURE_LABELS[name] for name in POLYMER_FEATURE_NAMES],
    )
    # Use explicit limits instead of invert_yaxis(): these panels share their
    # y-axis, so repeatedly toggling the shared axis reverses the requested
    # feature order whenever the number of panels is even.
    ax.set_ylim(len(POLYMER_FEATURE_NAMES) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _symmetric_limits(values: pd.Series) -> tuple[float, float]:
    maximum = float(np.nanmax(np.abs(values.to_numpy(dtype=float))))
    padded = maximum * 1.06 if maximum > 0 else 1.0
    return -padded, padded


def _add_color_key(
    fig: plt.Figure,
    *,
    include_missing: bool,
    bounds: tuple[float, float, float, float],
) -> None:
    mapper = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="coolwarm")
    mapper.set_array([])
    colorbar_ax = fig.add_axes(bounds)
    colorbar = fig.colorbar(
        mapper,
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks([0.0, 1.0], labels=["Low", "High"])
    colorbar.set_label("Observed feature value within each row")
    if include_missing:
        handle = plt.Line2D(
            [], [], marker="o", linestyle="", color="#8c8c8c", markersize=5, label="Missing / no polymer"
        )
        fig.legend(
            handles=[handle],
            loc="lower right",
            bbox_to_anchor=(0.985, 0.015),
            frameon=False,
        )


def _plot_regression_swarm(prediction_values: pd.DataFrame) -> None:
    frame = prediction_values.loc[
        prediction_values["task"].eq("regression")
        & prediction_values["scoring_population"].eq("all")
        & prediction_values["output"].eq("temperature")
    ]
    limits = _symmetric_limits(frame["contribution"])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharex=True, sharey=True)
    for ax, model_id in zip(axes.flat, _model_plot_order("regression")):
        subset = frame.loc[frame["model_id"].eq(model_id)]
        _draw_swarm_panel(
            ax,
            subset,
            limits,
            xlabel="Contribution to predicted LCST (°C)",
        )
        ax.set_title(_model_plot_title(model_id))
    fig.suptitle("Hierarchical polymer contributions to regression predictions", y=0.99)
    fig.subplots_adjust(
        left=0.15, right=0.985, top=0.89, bottom=0.20, hspace=0.36, wspace=0.04
    )
    _add_color_key(
        fig,
        include_missing=bool(frame["feature_missing"].any()),
        bounds=(0.25, 0.075, 0.50, 0.022),
    )
    fig.savefig(
        OUTPUT_DIR / "polymer_feature_regression_beeswarm.png",
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "LCST hierarchical importance"},
    )
    fig.savefig(
        OUTPUT_DIR / "polymer_feature_regression_beeswarm.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "LCST"},
    )
    plt.close(fig)


def _plot_classification_swarm(
    prediction_values: pd.DataFrame, population: str
) -> None:
    frame = prediction_values.loc[
        prediction_values["task"].eq("classification")
        & prediction_values["scoring_population"].eq(population)
    ]
    limits = _symmetric_limits(frame["contribution"])
    model_order = _model_plot_order("classification")
    output_order = tuple(label.lower() for label in CLASS_LABELS)
    fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.8), sharex=True, sharey=True)
    for row, output_name in enumerate(output_order):
        for column, model_id in enumerate(model_order):
            ax = axes[row, column]
            subset = frame.loc[
                frame["model_id"].eq(model_id) & frame["output"].eq(output_name)
            ]
            _draw_swarm_panel(
                ax,
                subset,
                limits,
                xlabel=(
                    "Contribution to class probability\n(percentage points)"
                    if row == len(output_order) - 1
                    else ""
                ),
            )
            if row == 0:
                ax.set_title(_model_plot_title(model_id))
            if column == 0:
                ax.set_ylabel(f"{output_name.upper()} probability")
    population_label = "all conditions" if population == "all" else "DexMA conditions only"
    fig.suptitle(
        f"Hierarchical polymer contributions to classification probabilities — {population_label}",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.12, right=0.99, top=0.90, bottom=0.16, hspace=0.24, wspace=0.06
    )
    _add_color_key(
        fig,
        include_missing=bool(frame["feature_missing"].any()),
        bounds=(0.27, 0.055, 0.46, 0.018),
    )
    stem = f"polymer_feature_classification_beeswarm_{population}"
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "LCST hierarchical importance"},
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "LCST"},
    )
    plt.close(fig)


def _write_results_notes(
    importance: pd.DataFrame,
    stability: pd.DataFrame,
    joint_support: pd.DataFrame,
) -> None:
    lines = [
        "# Hierarchical polymer feature-importance results",
        "",
        "The three values within each pipeline allocate the previously reported polymer-block contribution.",
        "Performance allocations quantify held-out predictive information; prediction beeswarms describe fitted-model behavior.",
        "",
        "## Performance allocation",
        "",
    ]
    for model_id in report.MODEL_DEFINITIONS:
        for population in (
            ("all", "dexma_only")
            if report.MODEL_DEFINITIONS[model_id]["task"] == "classification"
            else ("all",)
        ):
            subset = importance.loc[
                importance["model_id"].eq(model_id)
                & importance["scoring_population"].eq(population)
            ].sort_values("importance", ascending=False)
            values = ", ".join(
                f"{row.polymer_feature}={row.importance:.4f} "
                f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}]"
                for row in subset.itertuples()
            )
            unit = (
                "°C MAE reduction"
                if subset["task"].iloc[0] == "regression"
                else "nats of cross-entropy reduction"
            )
            lines.append(f"- {model_id}, {population} ({unit}): {values}.")
    lines.extend(["", "## Seed stability", ""])
    for model_id in report.MODEL_DEFINITIONS:
        subset = stability.loc[
            stability["model_id"].eq(model_id)
            & stability["scoring_population"].eq("all")
        ].sort_values("importance_mean", ascending=False)
        ranks = ", ".join(
            f"{row.polymer_feature} rank {row.best_rank}–{row.worst_rank}"
            for row in subset.itertuples()
        )
        lines.append(f"- {model_id}: {ranks}.")
    regression_joint = joint_support.loc[
        joint_support["task"].eq("regression")
        & joint_support["scoring_population"].eq("all")
    ]
    classification_joint = joint_support.loc[
        joint_support["task"].eq("classification")
        & joint_support["scoring_population"].eq("all")
    ]
    regression_top = regression_joint.iloc[0]
    classification_top = classification_joint.iloc[0]
    lines.extend(
        [
            "",
            "## Support and interpretation limits",
            "",
            f"- Regression contains {len(regression_joint)} observed polymer-feature triplets; "
            f"the most common triplet occurs in {int(regression_top.n_conditions)}/{int(regression_joint.n_conditions.sum())} conditions.",
            f"- Classification contains {len(classification_joint)} triplets including the "
            f"{int(classification_top.n_conditions)}-condition dominant triplet and 48 no-polymer controls.",
            "- Functionalization and molecular weight are structurally paired in the represented experiments; the hierarchy allocates shared predictive information but cannot make those variables independently identifiable.",
            "- No-polymer controls have an all-zero polymer block; unknown values in present polymers remain missing and are imputed from present-polymer training medians for MLPs. Interpret all-condition values together with the DexMA-only sensitivity.",
            "- Colored beeswarms show how a fitted pipeline uses observed values. They do not estimate physical effects, mechanisms, intervention responses, or causality.",
            "- Values must remain separate by endpoint, representation, and learner and must not be normalized by feature count.",
            "",
        ]
    )
    _atomic_text("\n".join(lines), OUTPUT_DIR / "results_notes.md")


def _verify_boundary_rows(
    predictions: pd.DataFrame,
) -> float:
    max_error = 0.0
    for model_id, definition in report.MODEL_DEFINITIONS.items():
        task = definition["task"]
        conditions, tables, _ = four_block._task_inputs(task)
        features = tables[definition["feature_set"]]
        for seed in report.STATIC_SEEDS:
            _, folds = four_block._folds(conditions, task, seed)
            source = _source_shard(model_id, seed, conditions, features, folds)
            hierarchical = predictions.loc[
                predictions["model_id"].eq(model_id)
                & predictions["seed"].eq(seed)
            ]
            condition_ids = conditions["condition_id"].astype(str).tolist()
            for outer, polymer in BOUNDARY_STATES:
                identifier = state_id(outer, polymer)
                coalition = four_block.coalition_id(
                    _boundary_source_coalition(outer, polymer)
                )
                left = hierarchical.loc[hierarchical["state_id"].eq(identifier)].copy()
                right = source.loc[source["coalition_id"].eq(coalition)].copy()
                left["condition_id"] = left["condition_id"].astype(str)
                right["condition_id"] = right["condition_id"].astype(str)
                left = left.set_index("condition_id").loc[condition_ids]
                right = right.set_index("condition_id").loc[condition_ids]
                columns = (
                    ["prediction_c"]
                    if task == "regression"
                    else [f"prediction_{label.lower()}" for label in CLASS_LABELS]
                )
                difference = np.abs(
                    left[columns].to_numpy(dtype=float)
                    - right[columns].to_numpy(dtype=float)
                )
                max_error = max(max_error, float(np.max(difference)))
    return max_error


def _verify_full_losses(state_losses: pd.DataFrame) -> float:
    reference = pd.read_csv(four_block.REFERENCE_METRICS)
    maximum = 0.0
    full = state_losses.loc[
        state_losses["scoring_population"].eq("all")
        & state_losses["state_id"].eq(FULL_STATE_ID)
    ]
    for row in full.itertuples():
        metric = "mae" if row.task == "regression" else "cross_entropy"
        expected = reference.loc[
            reference["challenge"].eq("overall_benchmark")
            & reference["holdout"].eq("repeated_10fold")
            & reference["model_id"].eq(row.model_id)
            & reference["task"].eq(row.task)
            & reference["metric"].eq(metric),
            "value",
        ]
        if len(expected) != 1:
            raise AssertionError(f"Missing retained benchmark for {row.model_id}")
        maximum = max(maximum, abs(float(row.loss) - float(expected.iloc[0])))
    return maximum


def compile_outputs() -> dict[str, Path]:
    """Validate complete shards and compile tables, figures, notes, and manifest."""

    self_checks = self_check()
    missing = [path for path in expected_shards() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} hierarchical shards; first missing: {missing[0]}"
        )
    frames: list[pd.DataFrame] = []
    for model_id, definition in report.MODEL_DEFINITIONS.items():
        task = definition["task"]
        conditions, tables, _ = four_block._task_inputs(task)
        features = tables[definition["feature_set"]]
        for seed in report.STATIC_SEEDS:
            frame = pd.read_csv(shard_path(model_id, seed))
            completed = validate_shard(
                frame, model_id, seed, conditions, require_complete=True
            )
            _, folds = four_block._folds(conditions, task, seed)
            base = _shard_manifest_base(model_id, seed, conditions, features, folds)
            _validate_shard_manifest(
                shard_manifest_path(model_id, seed), base, completed
            )
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    _, regression, classification, _, _ = report.load_inputs()
    expected_rows = (
        len(report.REGRESSION_MODELS) * len(report.STATIC_SEEDS) * len(regression) * len(HIERARCHICAL_STATES)
        + len(report.CLASSIFICATION_MODELS) * len(report.STATIC_SEEDS) * len(classification) * len(HIERARCHICAL_STATES)
    )
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Compiled hierarchy has {len(predictions)} rather than {expected_rows} prediction rows"
        )
    _atomic_csv(predictions, OUTPUT_DIR / "state_predictions_long.csv")

    primary_losses, seed_losses = _population_loss_rows(predictions)
    state_losses = _state_loss_summary(primary_losses)
    state_losses_by_seed = _state_loss_summary(seed_losses)
    importance, importance_by_seed, stability, condition_values, identity_checks = (
        _importance_tables(primary_losses, seed_losses)
    )
    prediction_values, prediction_checks = _prediction_contributions(predictions)
    support_summary, joint_support, functionalization_mw_support = _support_tables()

    boundary_error = _verify_boundary_rows(predictions)
    full_loss_error = _verify_full_losses(state_losses)
    dexma_counts = importance.loc[
        importance["scoring_population"].eq("dexma_only"), "n_conditions"
    ]
    if not dexma_counts.eq(classification["polymer_name"].eq("Dex-MA").sum()).all():
        raise AssertionError("DexMA-only hierarchy count differs from the current dataset")
    identity_tolerance = 1e-10
    # Boundary predictions are reused from the four-block CSV shards.  Their
    # class probabilities were serialized independently, so the row sums can
    # differ from one by a few times 1e-8 even though every fitted vector was
    # normalized before it was written.  Keep the algebraic identities at the
    # tighter tolerance and allow only this serialization-scale simplex error.
    probability_tolerance = 1e-7
    numeric_checks = {
        **identity_checks,
        **prediction_checks,
        "maximum_reused_boundary_prediction_error": boundary_error,
        "maximum_full_benchmark_loss_error": full_loss_error,
    }
    failed_checks = {
        name: value
        for name, value in numeric_checks.items()
        if value
        > (
            probability_tolerance
            if name == "maximum_probability_simplex_error"
            else identity_tolerance
        )
    }
    if failed_checks:
        raise AssertionError(f"Hierarchical numerical identity failed: {numeric_checks}")

    tables = {
        "state_losses.csv": state_losses,
        "state_losses_by_seed.csv": state_losses_by_seed,
        "polymer_feature_importance.csv": importance,
        "polymer_feature_importance_by_seed.csv": importance_by_seed,
        "polymer_feature_rank_stability.csv": stability,
        "condition_polymer_feature_contributions.csv": condition_values,
        "polymer_feature_prediction_contributions.csv": prediction_values,
        "polymer_feature_support_summary.csv": support_summary,
        "polymer_feature_joint_support.csv": joint_support,
        "polymer_functionalization_mw_support.csv": functionalization_mw_support,
    }
    for name, frame in tables.items():
        _atomic_csv(frame, OUTPUT_DIR / name)

    _plot_performance(importance)
    _plot_regression_swarm(prediction_values)
    _plot_classification_swarm(prediction_values, "all")
    _plot_classification_swarm(prediction_values, "dexma_only")
    _write_results_notes(importance, stability, joint_support)

    output_names = [
        "state_predictions_long.csv",
        *tables,
        "polymer_feature_performance.png",
        "polymer_feature_performance.pdf",
        "polymer_feature_regression_beeswarm.png",
        "polymer_feature_regression_beeswarm.pdf",
        "polymer_feature_classification_beeswarm_all.png",
        "polymer_feature_classification_beeswarm_all.pdf",
        "polymer_feature_classification_beeswarm_dexma_only.png",
        "polymer_feature_classification_beeswarm_dexma_only.pdf",
        "results_notes.md",
    ]
    output_hashes = {name: _sha256(OUTPUT_DIR / name) for name in output_names}
    manifest = {
        "analysis": "exact_cross_fitted_hierarchical_polymer_owen_allocation",
        "contract_version": CONTRACT_VERSION,
        "master_path": str(MASTER_PATH),
        "master_sha256": _sha256(MASTER_PATH),
        "source_four_block_manifest": str(four_block.OUTPUT_DIR / "manifest.json"),
        "source_four_block_manifest_sha256": _sha256(
            four_block.OUTPUT_DIR / "manifest.json"
        ),
        "regression_conditions": len(regression),
        "classification_conditions": len(classification),
        "dexma_only_classification_conditions": int(classification["polymer_name"].eq("Dex-MA").sum()),
        "static_seeds": list(report.STATIC_SEEDS),
        "folds_per_seed": 10,
        "bootstrap_resamples": four_block.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": four_block.BOOTSTRAP_SEED,
        "primary_aggregation": "average five out-of-fold predictions per condition before loss calculation",
        "losses": {
            "regression": "mean absolute error in degrees Celsius",
            "classification": "natural-log soft-target cross-entropy in nats",
        },
        "polymer_features": POLYMER_FEATURE_COLUMNS,
        "outer_groups": list(OUTER_GROUP_NAMES),
        "outer_states": [list(value) for value in OUTER_STATES],
        "polymer_states": [list(value) for value in POLYMER_STATES],
        "state_ids": list(STATE_IDS),
        "boundary_state_ids": list(BOUNDARY_STATE_IDS),
        "intermediate_state_ids": list(INTERMEDIATE_STATE_IDS),
        "outer_weights": {
            _subset_id(value): _outer_weight(value) for value in OUTER_STATES
        },
        "inner_weights": {
            feature: {
                _subset_id(value): _inner_weight(value)
                for value in POLYMER_STATES
                if feature not in value
            }
            for feature in POLYMER_FEATURE_NAMES
        },
        "model_definitions": report.MODEL_DEFINITIONS,
        "software": {
            "python": report.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
        },
        "checks": {
            **self_checks,
            "all_40_model_seed_shards_present": True,
            "prediction_row_count_matches_dataset": True,
            "sixteen_boundary_states_reused": True,
            "finite_predictions": True,
            "classification_probabilities_sum_to_one": True,
            "three_children_sum_to_parent": True,
            "bootstrap_children_sum_to_parent": True,
            "prediction_children_sum_to_parent": True,
            "classification_contributions_preserve_probability_simplex": True,
            "full_losses_match_report_evaluation": True,
            "dexma_only_scoring_count_matches_dataset": True,
            **numeric_checks,
        },
        "source_shard_sha256": {
            path.name: _sha256(path) for path in expected_shards()
        },
        "source_shard_manifest_sha256": {
            shard_manifest_path(model_id, seed).name: _sha256(
                shard_manifest_path(model_id, seed)
            )
            for model_id in report.MODEL_DEFINITIONS
            for seed in report.STATIC_SEEDS
        },
        "output_sha256": output_hashes,
        "interpretation": (
            "A secondary allocation of the existing polymer-block predictive contribution among "
            "three represented polymer variables. Performance values measure held-out predictive "
            "information; prediction values describe fitted-model behavior. Neither is an independent "
            "physical effect, mechanism, intervention response, or causal estimate."
        ),
    }
    _atomic_text(json.dumps(manifest, indent=2) + "\n", OUTPUT_DIR / "manifest.json")
    print(f"Compiled hierarchical polymer importance under {OUTPUT_DIR}", flush=True)
    return {
        "predictions": OUTPUT_DIR / "state_predictions_long.csv",
        "importance": OUTPUT_DIR / "polymer_feature_importance.csv",
        "prediction_contributions": OUTPUT_DIR
        / "polymer_feature_prediction_contributions.csv",
        "figure": OUTPUT_DIR / "polymer_feature_performance.png",
        "manifest": OUTPUT_DIR / "manifest.json",
        "notes": OUTPUT_DIR / "results_notes.md",
    }
