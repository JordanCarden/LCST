"""Four-block, refitting-based feature-importance analysis for the LCST models."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lcst-feature-importance-mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
from sklearn.model_selection import KFold, StratifiedKFold

from .features import (
    FEATURE_ENCODING_VERSION,
    BUFFER_COLUMN,
    DESCRIPTOR_23_COLUMNS,
    TWO_SLOT_37_COLUMNS,
)
from .modeling import CLASS_LABELS, TARGET_COLUMNS
from .schema import MASTER_PATH
from . import report_evaluation as report


OUTPUT_DIR = report.REPO_ROOT / "outputs" / "feature_importance"
SHARD_DIR = OUTPUT_DIR / "shards"
REFERENCE_PREDICTIONS = report.OUTPUT_DIR / "predictions_long.csv"
REFERENCE_MANIFEST = report.OUTPUT_DIR / "evaluation_manifest.json"
REFERENCE_METRICS = report.OUTPUT_DIR / "metrics_summary.csv"
REFERENCE_SPLIT_INVENTORY = report.OUTPUT_DIR / "split_inventory.csv"
REFERENCE_PCA_ARTIFACT = (
    report.REPO_ROOT / "models" / "two_slot_pca38_buffer_regressor_xgboost.joblib"
)

GROUP_NAMES = ("polymer", "additive", "salt", "buffer")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260909
SHARD_CONTRACT_VERSION = 1

PREDICTION_COLUMNS = (
    "task",
    "model_id",
    "feature_set",
    "model_family",
    "seed",
    "fold",
    "coalition_id",
    "groups",
    "n_groups",
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


def feature_groups(feature_set: str) -> dict[str, tuple[str, ...]]:
    """Return the four scientific feature blocks for one representation."""

    if feature_set == "descriptor24":
        return {
            "polymer": tuple(DESCRIPTOR_23_COLUMNS[:3]),
            "additive": tuple(DESCRIPTOR_23_COLUMNS[3:16]),
            "salt": tuple(DESCRIPTOR_23_COLUMNS[16:23]),
            "buffer": (BUFFER_COLUMN,),
        }
    if feature_set == "two_slot38":
        return {
            "polymer": tuple(TWO_SLOT_37_COLUMNS[:3]),
            "additive": tuple(TWO_SLOT_37_COLUMNS[3:20]),
            "salt": tuple(TWO_SLOT_37_COLUMNS[20:37]),
            "buffer": (BUFFER_COLUMN,),
        }
    raise ValueError(f"Unsupported feature set: {feature_set!r}")


def all_coalitions() -> tuple[tuple[str, ...], ...]:
    """Enumerate the 16 block coalitions in stable order."""

    return tuple(
        coalition
        for size in range(len(GROUP_NAMES) + 1)
        for coalition in itertools.combinations(GROUP_NAMES, size)
    )


COALITIONS = all_coalitions()
COALITION_IDS = tuple("empty" if not value else "+".join(value) for value in COALITIONS)
COALITION_BY_ID = dict(zip(COALITION_IDS, COALITIONS))
FULL_COALITION = COALITIONS[-1]
FULL_COALITION_ID = COALITION_IDS[-1]


def coalition_id(coalition: Sequence[str]) -> str:
    value = tuple(coalition)
    if value not in COALITIONS:
        raise ValueError(f"Invalid or noncanonical coalition: {value}")
    return "empty" if not value else "+".join(value)


def coalition_columns(feature_set: str, coalition: Sequence[str]) -> tuple[str, ...]:
    groups = feature_groups(feature_set)
    return tuple(column for group in coalition for column in groups[group])


def validate_group_contract(feature_set: str, feature_columns: Sequence[str]) -> None:
    groups = feature_groups(feature_set)
    flattened = [column for group in GROUP_NAMES for column in groups[group]]
    if len(flattened) != len(set(flattened)):
        raise AssertionError(f"Feature groups overlap for {feature_set}")
    if flattened != list(feature_columns):
        missing = sorted(set(feature_columns) - set(flattened))
        extra = sorted(set(flattened) - set(feature_columns))
        raise AssertionError(
            f"Feature groups do not exactly cover {feature_set}: missing={missing}, extra={extra}"
        )


def shapley_values(
    value_by_coalition: Mapping[tuple[str, ...], np.ndarray | float],
) -> dict[str, np.ndarray | float]:
    """Calculate exact four-player Shapley values for scalars or aligned arrays."""

    if set(value_by_coalition) != set(COALITIONS):
        raise ValueError("Shapley calculation requires all 16 canonical coalitions")
    n_groups = len(GROUP_NAMES)
    result: dict[str, np.ndarray | float] = {}
    for group in GROUP_NAMES:
        contribution: np.ndarray | float | None = None
        for subset in COALITIONS:
            if group in subset:
                continue
            expanded = tuple(name for name in GROUP_NAMES if name in set(subset) | {group})
            size = len(subset)
            weight = (
                math.factorial(size)
                * math.factorial(n_groups - size - 1)
                / math.factorial(n_groups)
            )
            increment = weight * (
                value_by_coalition[expanded] - value_by_coalition[subset]
            )
            contribution = increment if contribution is None else contribution + increment
        if contribution is None:
            raise AssertionError(f"No Shapley contribution calculated for {group}")
        result[group] = contribution
    return result


def null_prediction(
    task: str, train_conditions: pd.DataFrame, n_test: int
) -> np.ndarray:
    """Return the fold-specific intercept-only prediction."""

    if task == "regression":
        value = float(train_conditions["transition_temperature_c"].median())
        return np.full(n_test, value, dtype=float)
    if task == "classification":
        value = train_conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float).mean(axis=0)
        value = value / value.sum()
        return np.tile(value, (n_test, 1))
    raise ValueError(f"Unsupported task: {task!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash a stable, ordered CSV projection of a condition table."""

    payload = frame.loc[:, list(columns)].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


@lru_cache(maxsize=1)
def _retained_pca_contract() -> dict[str, Any]:
    """Load the exact 18-chemical PCA scores used by the retained benchmark."""

    if not REFERENCE_PCA_ARTIFACT.exists() or not REFERENCE_SPLIT_INVENTORY.exists():
        raise FileNotFoundError("The retained two-slot PCA artifact and inventory are required")
    inventory = pd.read_csv(REFERENCE_SPLIT_INVENTORY)
    overall = inventory.loc[
        inventory["challenge"].eq("overall_benchmark")
        & inventory["holdout"].eq("repeated_10fold")
    ]
    component_hashes = set(overall["pca_components_sha256"].dropna().astype(str))
    if len(component_hashes) != 1:
        raise AssertionError("Retained benchmark inventory has no unique PCA component hash")
    expected_component_hash = component_hashes.pop()

    artifact_paths = tuple(
        sorted((report.REPO_ROOT / "models").glob("two_slot_pca38_buffer_*.joblib"))
    )
    if len(artifact_paths) != 4:
        raise AssertionError("Expected four retained two-slot model artifacts")
    score_hashes: set[str] = set()
    scores: dict[str, Sequence[float]] | None = None
    provenance: dict[str, Any] | None = None
    artifact_hashes: dict[str, str] = {}
    for path in artifact_paths:
        artifact = joblib.load(path)
        if artifact.get("feature_encoding") != FEATURE_ENCODING_VERSION:
            raise ValueError("Retrain retained models with the current feature encoding before feature importance")
        if artifact.get("training_master_sha256") != _sha256(MASTER_PATH):
            raise AssertionError(f"Retained PCA artifact uses another dataset: {path.name}")
        bundle = artifact.get("molformer_pca", {})
        candidate_provenance = bundle.get("provenance", {})
        if candidate_provenance.get("pca_components_sha256") != expected_component_hash:
            raise AssertionError(f"Retained PCA component hash mismatch: {path.name}")
        candidate_scores = artifact.get("chemical_metadata", {}).get("molformer_pca16")
        names = tuple(candidate_provenance.get("fit_chemical_names", ()))
        if not isinstance(candidate_scores, dict) or set(candidate_scores) != set(names):
            raise AssertionError(f"Retained PCA scores are incomplete: {path.name}")
        matrix = np.vstack(
            [np.asarray(candidate_scores[name], dtype=np.float64) for name in names]
        )
        if matrix.shape != (18, 16) or not np.isfinite(matrix).all():
            raise AssertionError(f"Retained PCA score matrix is invalid: {path.name}")
        score_hashes.add(hashlib.sha256(matrix.tobytes()).hexdigest())
        scores = candidate_scores
        provenance = candidate_provenance
        artifact_hashes[path.name] = _sha256(path)
    if len(score_hashes) != 1 or scores is None or provenance is None:
        raise AssertionError("Retained model artifacts do not share one PCA score matrix")
    return {
        "scores": scores,
        "scores_sha256": score_hashes.pop(),
        "provenance": provenance,
        "artifact_sha256": artifact_hashes,
    }


def _task_inputs(
    task: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], report.PCAContext]:
    _, regression, classification, metadata, context = report.load_inputs()
    retained = _retained_pca_contract()
    provenance = retained["provenance"]
    if provenance["structure_registry_sha256"] != context.structure_sha256:
        raise AssertionError("Retained PCA basis uses another chemical structure registry")
    all_chemical_metadata = dict(metadata)
    all_chemical_metadata["molformer_pca16"] = retained["scores"]
    conditions = regression if task == "regression" else classification
    tables = report.feature_tables(conditions, all_chemical_metadata)
    return conditions, tables, context


def _model_protocol(model_id: str, seed: int) -> dict[str, Any]:
    definition = report.MODEL_DEFINITIONS[model_id]
    if definition["family"] == "MLP":
        protocol = (
            report.MLP_REGRESSION_PROTOCOL
            if definition["task"] == "regression"
            else report.MLP_CLASSIFICATION_PROTOCOL
        )
        return {"family": "MLP", "feature_encoding": FEATURE_ENCODING_VERSION, "parameters": asdict(protocol)}
    if definition["task"] == "regression":
        estimator = report.xgb_regressor(seed).named_steps["model"]
        parameters = estimator.get_params(deep=False)
    else:
        parameters = report.SingleThreadSoftTargetXGBClassifier(seed).parameters
    return {"family": "XGBoost", "feature_encoding": FEATURE_ENCODING_VERSION, "parameters": parameters}


def _expected_shard_manifest(
    model_id: str,
    seed: int,
    conditions: pd.DataFrame,
    features: pd.DataFrame,
    fold_assignments: np.ndarray,
) -> dict[str, Any]:
    definition = report.MODEL_DEFINITIONS[model_id]
    target_columns = (
        ["condition_id", "transition_temperature_c"]
        if definition["task"] == "regression"
        else ["condition_id", *TARGET_COLUMNS]
    )
    retained_pca = _retained_pca_contract()
    payload = {
        "contract_version": SHARD_CONTRACT_VERSION,
        "model_id": model_id,
        "model_definition": definition,
        "model_protocol": _model_protocol(model_id, seed),
        "seed": seed,
        "master_sha256": _sha256(MASTER_PATH),
        "reference_predictions_sha256": _sha256(REFERENCE_PREDICTIONS),
        "reference_manifest_sha256": _sha256(REFERENCE_MANIFEST),
        "condition_id_sha256": _frame_sha256(conditions, ["condition_id"]),
        "target_sha256": _frame_sha256(conditions, target_columns),
        "fold_assignments_sha256": hashlib.sha256(
            np.asarray(fold_assignments, dtype=np.int64).tobytes()
        ).hexdigest(),
        "feature_matrix_sha256": _frame_sha256(features, list(features.columns)),
        "feature_groups": {
            group: list(columns)
            for group, columns in feature_groups(definition["feature_set"]).items()
        },
        "coalition_ids": list(COALITION_IDS),
        "pca_components_sha256": retained_pca["provenance"][
            "pca_components_sha256"
        ],
        "pca_scores_sha256": retained_pca["scores_sha256"],
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def _validate_shard_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing shard manifest: {path}")
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise AssertionError(f"Stale or incompatible shard manifest: {path}")


def _folds(
    conditions: pd.DataFrame, task: str, seed: int
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], np.ndarray]:
    strata = (
        conditions["stratification_class"].to_numpy()
        if task == "classification"
        else None
    )
    splitter = (
        KFold(n_splits=10, shuffle=True, random_state=seed)
        if task == "regression"
        else StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    )
    iterator = (
        splitter.split(conditions, strata)
        if strata is not None
        else splitter.split(conditions)
    )
    result: list[tuple[int, np.ndarray, np.ndarray]] = []
    assignments = np.full(len(conditions), -1, dtype=int)
    for fold, (train_index, test_index) in enumerate(iterator):
        assignments[test_index] = fold
        result.append((fold, np.asarray(train_index), np.asarray(test_index)))
    if (assignments < 0).any():
        raise AssertionError("Cross-validation did not assign every condition to a fold")
    return result, assignments


def _validate_reference_contract() -> dict[str, Any]:
    if not REFERENCE_MANIFEST.exists() or not REFERENCE_PREDICTIONS.exists():
        raise FileNotFoundError(
            "The retained report evaluation is required; run its static phase first"
        )
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("master_sha256") != _sha256(MASTER_PATH):
        raise AssertionError("Reference predictions use a different master dataset")
    if tuple(manifest.get("static_seeds", ())) != report.STATIC_SEEDS:
        raise AssertionError("Reference predictions use different static seeds")
    if manifest.get("model_definitions") != report.MODEL_DEFINITIONS:
        raise AssertionError("Reference predictions use different model definitions")
    if manifest.get("pca_policy", {}).get("within_dataset_cross_validation") != (
        "fit on all 18 configured chemicals"
    ):
        raise AssertionError("Reference predictions use a different PCA policy")
    return manifest


def _reference_full_prediction(
    model_id: str,
    seed: int,
    task: str,
    conditions: pd.DataFrame,
    fold_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_reference_contract()
    frame = pd.read_csv(REFERENCE_PREDICTIONS)
    frame = frame.loc[
        frame["challenge"].eq("overall_benchmark")
        & frame["holdout"].eq("repeated_10fold")
        & frame["model_id"].eq(model_id)
        & frame["seed"].eq(seed)
        & frame["task"].eq(task)
    ].copy()
    if len(frame) != len(conditions) or frame["condition_id"].duplicated().any():
        raise AssertionError(
            f"Reference full predictions are incomplete for {model_id}, seed {seed}"
        )
    expected_ids = conditions["condition_id"].astype(str).tolist()
    frame["condition_id"] = frame["condition_id"].astype(str)
    if set(frame["condition_id"]) != set(expected_ids):
        raise AssertionError("Reference full predictions contain different condition IDs")
    frame = frame.set_index("condition_id").loc[expected_ids].reset_index()
    if frame["feature_set"].nunique() != 1 or (
        frame["feature_set"].iloc[0] != report.MODEL_DEFINITIONS[model_id]["feature_set"]
    ):
        raise AssertionError("Reference prediction feature-set label changed")
    if frame["model_family"].nunique() != 1 or (
        frame["model_family"].iloc[0] != report.MODEL_DEFINITIONS[model_id]["family"]
    ):
        raise AssertionError("Reference prediction model-family label changed")
    if task == "regression":
        expected_target = conditions["transition_temperature_c"].to_numpy(dtype=float)
        if not np.allclose(frame["target_c"], expected_target):
            raise AssertionError("Reference regression targets differ from current conditions")
        prediction = frame["prediction_c"].to_numpy(dtype=float)
    else:
        expected_target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        observed_target = frame[
            [f"target_{label.lower()}" for label in CLASS_LABELS]
        ].to_numpy(dtype=float)
        if not np.allclose(observed_target, expected_target):
            raise AssertionError("Reference classification targets differ from current conditions")
        prediction = frame[
            [f"prediction_{label.lower()}" for label in CLASS_LABELS]
        ].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise AssertionError("Reference full predictions contain non-finite values")
    if task == "classification" and not np.allclose(
        prediction.sum(axis=1), 1.0, atol=1e-7
    ):
        raise AssertionError("Reference classification probabilities do not sum to one")
    epochs = frame["selected_epoch"].to_numpy(dtype=float)
    if not np.array_equal(fold_assignments, _folds(conditions, task, seed)[1]):
        raise AssertionError("Reconstructed folds changed unexpectedly")
    return prediction, epochs


def _prediction_frame(
    conditions: pd.DataFrame,
    task: str,
    model_id: str,
    seed: int,
    folds: np.ndarray,
    coalition: tuple[str, ...],
    prediction: np.ndarray,
    epochs: np.ndarray,
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
            "coalition_id": coalition_id(coalition),
            "groups": ";".join(coalition),
            "n_groups": len(coalition),
            "condition_id": conditions["condition_id"].astype(str).to_numpy(),
            "polymer_name": conditions["polymer_name"].astype(str).to_numpy(),
            "dexma_present": conditions["polymer_name"].eq("Dex-MA").to_numpy(),
            "selected_epoch": epochs,
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
            frame[f"target_{label.lower()}"] = target[:, index]
            frame[f"prediction_{label.lower()}"] = values[:, index]
    return frame[list(PREDICTION_COLUMNS)]


def validate_shard(
    frame: pd.DataFrame,
    model_id: str,
    seed: int,
    conditions: pd.DataFrame,
) -> None:
    definition = report.MODEL_DEFINITIONS[model_id]
    task = definition["task"]
    expected_rows = len(conditions) * len(COALITIONS)
    if len(frame) != expected_rows:
        raise AssertionError(
            f"Shard has {len(frame)} rows rather than {expected_rows}: {model_id}/{seed}"
        )
    if set(frame["coalition_id"]) != set(COALITION_IDS):
        raise AssertionError("Shard does not contain exactly the 16 coalitions")
    if frame[["coalition_id", "condition_id"]].duplicated().any():
        raise AssertionError("Shard contains duplicate coalition/condition predictions")
    if frame["model_id"].nunique() != 1 or frame["model_id"].iloc[0] != model_id:
        raise AssertionError("Shard model ID is invalid")
    if frame["seed"].nunique() != 1 or int(frame["seed"].iloc[0]) != seed:
        raise AssertionError("Shard seed is invalid")
    if frame["task"].nunique() != 1 or frame["task"].iloc[0] != task:
        raise AssertionError("Shard task is invalid")
    expected_ids = conditions["condition_id"].astype(str).tolist()
    _, expected_folds = _folds(conditions, task, seed)
    for identifier, coalition in zip(COALITION_IDS, COALITIONS):
        subset = frame.loc[frame["coalition_id"].eq(identifier)].copy()
        if len(subset) != len(conditions):
            raise AssertionError(f"Coalition {identifier} is incomplete")
        if not subset["n_groups"].eq(len(coalition)).all():
            raise AssertionError(f"Coalition size is incorrect for {identifier}")
        subset["condition_id"] = subset["condition_id"].astype(str)
        if set(subset["condition_id"]) != set(expected_ids):
            raise AssertionError(f"Coalition {identifier} has different condition IDs")
        subset = subset.set_index("condition_id").loc[expected_ids]
        if not np.array_equal(subset["fold"].to_numpy(dtype=int), expected_folds):
            raise AssertionError(f"Coalition {identifier} has different fold assignments")
        if task == "regression":
            target = conditions["transition_temperature_c"].to_numpy(dtype=float)
            if not np.allclose(subset["target_c"].to_numpy(dtype=float), target):
                raise AssertionError(f"Coalition {identifier} has different targets")
        else:
            target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
            observed = subset[
                [f"target_{label.lower()}" for label in CLASS_LABELS]
            ].to_numpy(dtype=float)
            if not np.allclose(observed, target):
                raise AssertionError(f"Coalition {identifier} has different targets")
    if task == "regression":
        if not np.isfinite(frame["prediction_c"]).all():
            raise AssertionError("Regression shard contains non-finite predictions")
    else:
        probabilities = frame[
            [f"prediction_{label.lower()}" for label in CLASS_LABELS]
        ].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-7
        ):
            raise AssertionError("Classification shard contains invalid probabilities")


def run_shard(model_id: str, seed: int, *, force: bool = False) -> Path:
    """Run all 16 coalitions for one model and one outer-CV seed."""

    if model_id not in report.MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model ID: {model_id}")
    if seed not in report.STATIC_SEEDS:
        raise ValueError(f"Seed must be one of {report.STATIC_SEEDS}")
    definition = report.MODEL_DEFINITIONS[model_id]
    task = definition["task"]
    conditions, tables, _ = _task_inputs(task)
    path = shard_path(model_id, seed)
    manifest_path = shard_manifest_path(model_id, seed)
    feature_set = definition["feature_set"]
    features = tables[feature_set]
    validate_group_contract(feature_set, features.columns)
    split_rows, fold_assignments = _folds(conditions, task, seed)
    expected_manifest = _expected_shard_manifest(
        model_id, seed, conditions, features, fold_assignments
    )
    if path.exists() and not force:
        try:
            existing = pd.read_csv(path)
            validate_shard(existing, model_id, seed, conditions)
            _validate_shard_manifest(manifest_path, expected_manifest)
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
            print(f"Rebuilding incompatible shard {path.name}: {error}", flush=True)
        else:
            print(f"Valid shard already exists: {path}", flush=True)
            return path

    started = time.monotonic()
    torch.set_num_threads(1)
    predictions: dict[tuple[str, ...], np.ndarray] = {}
    epochs: dict[tuple[str, ...], np.ndarray] = {}
    output_shape = (len(conditions), 3) if task == "classification" else (len(conditions),)

    predictions[()] = np.full(output_shape, np.nan, dtype=float)
    epochs[()] = np.full(len(conditions), np.nan, dtype=float)
    fitted_coalitions = [
        coalition for coalition in COALITIONS if coalition and coalition != FULL_COALITION
    ]
    for coalition in fitted_coalitions:
        predictions[coalition] = np.full(output_shape, np.nan, dtype=float)
        epochs[coalition] = np.full(len(conditions), np.nan, dtype=float)

    for fold, train_index, test_index in split_rows:
        train_conditions = conditions.iloc[train_index]
        predictions[()][test_index] = null_prediction(
            task, train_conditions, len(test_index)
        )
        for coalition in fitted_coalitions:
            columns = coalition_columns(feature_set, coalition)
            prediction, epoch = report.fit_model(
                task,
                model_id,
                features.iloc[train_index].loc[:, list(columns)],
                train_conditions,
                features.iloc[test_index].loc[:, list(columns)],
                seed,
            )
            predictions[coalition][test_index] = prediction
            epochs[coalition][test_index] = epoch
        print(
            f"{model_id} seed={seed}: completed fold {fold + 1}/10",
            flush=True,
        )

    full_prediction, full_epochs = _reference_full_prediction(
        model_id, seed, task, conditions, fold_assignments
    )
    predictions[FULL_COALITION] = full_prediction
    epochs[FULL_COALITION] = full_epochs
    outputs = [
        _prediction_frame(
            conditions,
            task,
            model_id,
            seed,
            fold_assignments,
            coalition,
            predictions[coalition],
            epochs[coalition],
        )
        for coalition in COALITIONS
    ]
    shard = pd.concat(outputs, ignore_index=True)
    validate_shard(shard, model_id, seed, conditions)
    _atomic_csv(shard, path)
    _atomic_text(json.dumps(expected_manifest, indent=2) + "\n", manifest_path)
    elapsed = time.monotonic() - started
    print(f"Wrote {path} in {elapsed:.1f} seconds", flush=True)
    return path


def expected_shards() -> tuple[Path, ...]:
    return tuple(
        shard_path(model_id, seed)
        for model_id in report.MODEL_DEFINITIONS
        for seed in report.STATIC_SEEDS
    )


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
    group_columns = ["coalition_id", "condition_id", "polymer_name", "dexma_present"]
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


def _coalition_loss_summary(losses: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "task",
        "model_id",
        "feature_set",
        "model_family",
        "scoring_population",
        "coalition_id",
    ]
    if "seed" in losses:
        group_columns.insert(5, "seed")
    summary = losses.groupby(group_columns, sort=False, as_index=False).agg(
        loss=("loss", "mean"), n_conditions=("condition_id", "nunique")
    )
    summary["groups"] = summary["coalition_id"].map(
        lambda identifier: ";".join(COALITION_BY_ID[identifier])
    )
    summary["n_groups"] = summary["coalition_id"].map(
        lambda identifier: len(COALITION_BY_ID[identifier])
    )
    null_lookup = summary.loc[summary["coalition_id"].eq("empty")].set_index(
        [column for column in group_columns if column != "coalition_id"]
    )["loss"]
    lookup_columns = [column for column in group_columns if column != "coalition_id"]
    summary["null_loss"] = [
        null_lookup.loc[tuple(row[column] for column in lookup_columns)]
        for _, row in summary.iterrows()
    ]
    summary["predictive_value"] = summary["null_loss"] - summary["loss"]
    return summary


def _bootstrap_indices(task: str, population: str, n_conditions: int) -> np.ndarray:
    offsets = {
        ("regression", "all"): 0,
        ("classification", "all"): 1,
        ("classification", "dexma_only"): 2,
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + offsets[(task, population)])
    return rng.integers(
        0,
        n_conditions,
        size=(BOOTSTRAP_RESAMPLES, n_conditions),
        dtype=np.int32,
    )


def _importance_tables(
    primary_losses: pd.DataFrame, seed_losses: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    logo_rows: list[dict[str, Any]] = []
    keys = ["task", "model_id", "feature_set", "model_family", "scoring_population"]
    for values, frame in primary_losses.groupby(keys, sort=False):
        task, model_id, feature_set, family, population = values
        pivot = frame.pivot(index="condition_id", columns="coalition_id", values="loss")
        pivot = pivot.reindex(columns=COALITION_IDS).sort_index()
        if pivot.isna().any().any():
            raise AssertionError(f"Incomplete coalition losses for {model_id}/{population}")
        null = pivot["empty"].to_numpy(dtype=float)
        coalition_values = {
            coalition: null - pivot[coalition_id(coalition)].to_numpy(dtype=float)
            for coalition in COALITIONS
        }
        contributions = shapley_values(coalition_values)
        samples = _bootstrap_indices(task, population, len(pivot))
        full_loss = float(pivot[FULL_COALITION_ID].mean())
        null_loss = float(pivot["empty"].mean())
        total_gain = null_loss - full_loss
        for group in GROUP_NAMES:
            contribution = np.asarray(contributions[group], dtype=float)
            distribution = contribution[samples].mean(axis=1)
            central = float(contribution.mean())
            importance_rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "model_family": family,
                    "scoring_population": population,
                    "group": group,
                    "importance": central,
                    "ci95_low": float(np.quantile(distribution, 0.025)),
                    "ci95_high": float(np.quantile(distribution, 0.975)),
                    "null_loss": null_loss,
                    "full_loss": full_loss,
                    "total_gain": total_gain,
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
                    "condition_id": condition_id_value,
                    "group": group,
                    "contribution": float(contribution[index]),
                }
                for index, condition_id_value in enumerate(pivot.index)
            )
            without = tuple(name for name in GROUP_NAMES if name != group)
            logo_contribution = (
                pivot[coalition_id(without)].to_numpy(dtype=float)
                - pivot[FULL_COALITION_ID].to_numpy(dtype=float)
            )
            logo_distribution = logo_contribution[samples].mean(axis=1)
            logo_rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "model_family": family,
                    "scoring_population": population,
                    "group": group,
                    "logo_importance": float(logo_contribution.mean()),
                    "ci95_low": float(np.quantile(logo_distribution, 0.025)),
                    "ci95_high": float(np.quantile(logo_distribution, 0.975)),
                    "loss_without_group": float(pivot[coalition_id(without)].mean()),
                    "full_loss": full_loss,
                    "n_conditions": len(pivot),
                }
            )

    seed_rows: list[dict[str, Any]] = []
    seed_keys = [*keys[:5], "seed"]
    for values, frame in seed_losses.groupby(seed_keys, sort=False):
        task, model_id, feature_set, family, population, seed = values
        losses = frame.groupby("coalition_id", sort=False)["loss"].mean()
        if set(losses.index) != set(COALITION_IDS):
            raise AssertionError(f"Incomplete seed losses for {model_id}/{seed}/{population}")
        null_loss = float(losses.loc["empty"])
        values_by_coalition = {
            coalition: null_loss - float(losses.loc[coalition_id(coalition)])
            for coalition in COALITIONS
        }
        contributions = shapley_values(values_by_coalition)
        full_loss = float(losses.loc[FULL_COALITION_ID])
        for group in GROUP_NAMES:
            seed_rows.append(
                {
                    "task": task,
                    "model_id": model_id,
                    "feature_set": feature_set,
                    "model_family": family,
                    "scoring_population": population,
                    "seed": int(seed),
                    "group": group,
                    "importance": float(contributions[group]),
                    "null_loss": null_loss,
                    "full_loss": full_loss,
                    "total_gain": null_loss - full_loss,
                }
            )
    by_seed = pd.DataFrame(seed_rows)
    by_seed["rank"] = by_seed.groupby(
        ["task", "model_id", "scoring_population", "seed"]
    )["importance"].rank(method="dense", ascending=False).astype(int)
    stability = by_seed.groupby(
        ["task", "model_id", "feature_set", "model_family", "scoring_population", "group"],
        sort=False,
        as_index=False,
    ).agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
        median_rank=("rank", "median"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        top_rank_count=("rank", lambda values: int((values == 1).sum())),
    )
    stability["top_rank_fraction"] = stability["top_rank_count"] / len(report.STATIC_SEEDS)
    return (
        pd.DataFrame(importance_rows),
        by_seed,
        stability,
        pd.DataFrame(condition_rows),
        pd.DataFrame(logo_rows),
    )


def _plot_importance(importance: pd.DataFrame) -> None:
    selected = importance.loc[importance["scoring_population"].eq("all")].copy()
    colors = {"XGBoost": "#1f6bc1", "MLP": "#f26f00"}
    labels = {
        "polymer": "Polymer",
        "additive": "Additive",
        "salt": "Salt",
        "buffer": "Buffer",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex="row", sharey=True)
    panel = 0
    for row, task in enumerate(("regression", "classification")):
        row_values = selected.loc[selected["task"].eq(task)]
        finite = row_values[["ci95_low", "ci95_high"]].to_numpy(dtype=float)
        low = min(0.0, float(np.nanmin(finite)))
        high = max(0.0, float(np.nanmax(finite)))
        padding = max((high - low) * 0.10, 0.02)
        for column, feature_set in enumerate(("descriptor24", "two_slot38")):
            ax = axes[row, column]
            subset = row_values.loc[row_values["feature_set"].eq(feature_set)]
            positions = np.arange(len(GROUP_NAMES), dtype=float)
            for family, offset, marker in (
                ("XGBoost", -0.11, "o"),
                ("MLP", 0.11, "s"),
            ):
                family_values = subset.loc[subset["model_family"].eq(family)].set_index("group")
                values = family_values.loc[list(GROUP_NAMES), "importance"].to_numpy(dtype=float)
                lows = family_values.loc[list(GROUP_NAMES), "ci95_low"].to_numpy(dtype=float)
                highs = family_values.loc[list(GROUP_NAMES), "ci95_high"].to_numpy(dtype=float)
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
            ax.set_yticks(positions, [labels[group] for group in GROUP_NAMES])
            ax.grid(axis="x", alpha=0.20)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(low - padding, high + padding)
            if row == 0:
                ax.set_title("Descriptor representation" if column == 0 else "Two-slot representation")
            if row == 1:
                ax.set_xlabel("Allocated cross-entropy reduction (nats)")
            else:
                ax.set_xlabel("Allocated MAE reduction (°C)")
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
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "feature_importance_four_panel.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "feature_importance_four_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_results_notes(
    importance: pd.DataFrame, stability: pd.DataFrame
) -> None:
    lines = [
        "# Four-block feature-importance results",
        "",
        "Values allocate the held-out improvement over a fold-specific intercept-only baseline.",
        "They quantify predictive information available to a specified learner and representation, not causality.",
        "",
        "## Primary all-condition results",
        "",
    ]
    primary = importance.loc[importance["scoring_population"].eq("all")]
    for model_id in report.MODEL_DEFINITIONS:
        subset = primary.loc[primary["model_id"].eq(model_id)].sort_values(
            "importance", ascending=False
        )
        values = ", ".join(
            f"{row.group}={row.importance:.4f} [{row.ci95_low:.4f}, {row.ci95_high:.4f}]"
            for row in subset.itertuples()
        )
        unit = "°C MAE reduction" if subset["task"].iloc[0] == "regression" else "nats of cross-entropy reduction"
        lines.append(f"- {model_id} ({unit}): {values}.")
    lines.extend(
        [
            "",
            "## Seed stability",
            "",
        ]
    )
    for model_id in report.MODEL_DEFINITIONS:
        subset = stability.loc[
            stability["model_id"].eq(model_id)
            & stability["scoring_population"].eq("all")
        ]
        top = subset.loc[subset["top_rank_count"].eq(subset["top_rank_count"].max())]
        labels = ", ".join(
            f"{row.group} ({int(row.top_rank_count)}/5 top ranks)" for row in top.itertuples()
        )
        lines.append(f"- {model_id}: {labels}.")
    lines.extend(
        [
            "",
            "## Classification control sensitivity",
            "",
        ]
    )
    for model_id in report.CLASSIFICATION_MODELS:
        subset = importance.loc[importance["model_id"].eq(model_id)]
        all_polymer = float(
            subset.loc[
                subset["scoring_population"].eq("all") & subset["group"].eq("polymer"),
                "importance",
            ].iloc[0]
        )
        dexma_polymer = float(
            subset.loc[
                subset["scoring_population"].eq("dexma_only") & subset["group"].eq("polymer"),
                "importance",
            ].iloc[0]
        )
        lines.append(
            f"- {model_id}: polymer allocation {all_polymer:.4f} on all conditions and "
            f"{dexma_polymer:.4f} when scoring only DexMA-containing conditions."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary for the paper",
            "",
            "The group-level scope is imposed primarily by the dataset, not by a desire to avoid detail. "
            "Additive molecular weight, carbon count, logP, charge, and related descriptors are fixed lookups "
            "for only six additive identities; most salts are singletons while NaCl dominates; salt concentration, "
            "ionic strength, charge, and Hofmeister descriptors overlap strongly; and individual embedding "
            "coordinates have no standalone chemical meaning. An individual-feature chart could therefore label "
            "carbon count as important when the fitted model was actually recognizing SDS or CTAB.",
            "",
            "Do not describe these allocations as physical effect sizes, directions of effect, mechanisms, causal "
            "effects, or proof of transfer to an unseen chemical. Do not compare numerical magnitudes across the "
            "regression and classification endpoints or average XGBoost and MLP results.",
        ]
    )
    _atomic_text("\n".join(lines) + "\n", OUTPUT_DIR / "results_notes.md")


def _verify_compiled(
    predictions: pd.DataFrame,
    coalition_losses: pd.DataFrame,
    importance: pd.DataFrame,
    by_seed: pd.DataFrame,
) -> dict[str, bool]:
    _, regression, classification, _, _ = report.load_inputs()
    expected_predictions = (
        len(report.REGRESSION_MODELS) * len(report.STATIC_SEEDS) * len(regression) * len(COALITIONS)
        + len(report.CLASSIFICATION_MODELS) * len(report.STATIC_SEEDS) * len(classification) * len(COALITIONS)
    )
    if len(predictions) != expected_predictions:
        raise AssertionError(
            f"Compiled predictions have {len(predictions)} rows, expected {expected_predictions}"
        )
    grouped = coalition_losses.groupby(["model_id", "scoring_population"])["coalition_id"].nunique()
    if not grouped.eq(len(COALITIONS)).all():
        raise AssertionError("A compiled model/population is missing coalition losses")
    sums = importance.groupby(["model_id", "scoring_population"], as_index=False).agg(
        importance_sum=("importance", "sum"), total_gain=("total_gain", "first")
    )
    if not np.allclose(sums["importance_sum"], sums["total_gain"], atol=1e-10):
        raise AssertionError("Shapley contributions do not sum to total predictive gain")
    seed_sums = by_seed.groupby(
        ["model_id", "scoring_population", "seed"], as_index=False
    ).agg(importance_sum=("importance", "sum"), total_gain=("total_gain", "first"))
    if not np.allclose(seed_sums["importance_sum"], seed_sums["total_gain"], atol=1e-10):
        raise AssertionError("Seed-specific Shapley contributions do not sum to total gain")

    reference = pd.read_csv(REFERENCE_METRICS)
    full = coalition_losses.loc[
        coalition_losses["scoring_population"].eq("all")
        & coalition_losses["coalition_id"].eq(FULL_COALITION_ID)
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
        if len(expected) != 1 or not np.isclose(row.loss, float(expected.iloc[0]), atol=1e-12):
            raise AssertionError(f"Full-coalition loss does not reproduce {row.model_id}")
    dexma_counts = importance.loc[
        importance["scoring_population"].eq("dexma_only"), "n_conditions"
    ]
    if not dexma_counts.eq(classification["polymer_name"].eq("Dex-MA").sum()).all():
        raise AssertionError("DexMA-only sensitivity count differs from the current dataset")
    return {
        "all_40_model_seed_shards_present": True,
        "prediction_row_count_matches_dataset": True,
        "all_16_coalitions_present": True,
        "finite_predictions": True,
        "classification_probabilities_sum_to_one": True,
        "shapley_efficiency_identity": True,
        "seed_shapley_efficiency_identity": True,
        "full_losses_match_report_evaluation": True,
        "dexma_only_scoring_count_matches_dataset": True,
    }


def compile_outputs() -> dict[str, Path]:
    """Validate all shards and compile final tables, figure, notes, and manifest."""

    _validate_reference_contract()
    missing = [path for path in expected_shards() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} model/seed shards; first missing shard: {missing[0]}"
        )
    task_cache: dict[str, pd.DataFrame] = {}
    frames: list[pd.DataFrame] = []
    for model_id, definition in report.MODEL_DEFINITIONS.items():
        task = definition["task"]
        if task not in task_cache:
            task_cache[task] = _task_inputs(task)[0]
        _, task_tables, _ = _task_inputs(task)
        for seed in report.STATIC_SEEDS:
            frame = pd.read_csv(shard_path(model_id, seed))
            validate_shard(frame, model_id, seed, task_cache[task])
            features = task_tables[definition["feature_set"]]
            _, fold_assignments = _folds(task_cache[task], task, seed)
            _validate_shard_manifest(
                shard_manifest_path(model_id, seed),
                _expected_shard_manifest(
                    model_id,
                    seed,
                    task_cache[task],
                    features,
                    fold_assignments,
                ),
            )
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    _atomic_csv(predictions, OUTPUT_DIR / "coalition_predictions_long.csv")

    primary_losses, seed_losses = _population_loss_rows(predictions)
    coalition_losses = _coalition_loss_summary(primary_losses)
    coalition_losses_by_seed = _coalition_loss_summary(seed_losses)
    (
        importance,
        importance_by_seed,
        rank_stability,
        condition_contributions,
        logo,
    ) = _importance_tables(primary_losses, seed_losses)
    checks = _verify_compiled(
        predictions, coalition_losses, importance, importance_by_seed
    )

    tables = {
        "coalition_losses.csv": coalition_losses,
        "coalition_losses_by_seed.csv": coalition_losses_by_seed,
        "shapley_importance.csv": importance,
        "shapley_importance_by_seed.csv": importance_by_seed,
        "rank_stability.csv": rank_stability,
        "condition_group_contributions.csv": condition_contributions,
        "logo_importance.csv": logo,
    }
    for name, frame in tables.items():
        _atomic_csv(frame, OUTPUT_DIR / name)
    _plot_importance(importance)
    _write_results_notes(importance, rank_stability)

    _, regression, classification, _, context = report.load_inputs()
    retained_pca = _retained_pca_contract()
    pca_provenance = retained_pca["provenance"]
    output_hashes = {
        name: _sha256(OUTPUT_DIR / name)
        for name in ["coalition_predictions_long.csv", *tables]
    }
    manifest = {
        "analysis": "exact_cross_fitted_refitting_based_four_block_shapley",
        "master_path": str(MASTER_PATH),
        "master_sha256": _sha256(MASTER_PATH),
        "reference_predictions": str(REFERENCE_PREDICTIONS),
        "reference_predictions_sha256": _sha256(REFERENCE_PREDICTIONS),
        "reference_manifest_sha256": _sha256(REFERENCE_MANIFEST),
        "regression_conditions": len(regression),
        "classification_conditions": len(classification),
        "dexma_only_classification_conditions": int(classification["polymer_name"].eq("Dex-MA").sum()),
        "condition_id_sha256": {
            "regression": _frame_sha256(regression, ["condition_id"]),
            "classification": _frame_sha256(classification, ["condition_id"]),
        },
        "target_sha256": {
            "regression": _frame_sha256(
                regression, ["condition_id", "transition_temperature_c"]
            ),
            "classification": _frame_sha256(
                classification, ["condition_id", *TARGET_COLUMNS]
            ),
        },
        "static_seeds": list(report.STATIC_SEEDS),
        "folds_per_seed": 10,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_aggregation": "average five out-of-fold predictions per condition before loss calculation",
        "losses": {
            "regression": "mean absolute error in degrees Celsius",
            "classification": "natural-log soft-target cross-entropy in nats",
        },
        "null_models": {
            "regression": "training-fold median LCST",
            "classification": "training-fold mean soft class distribution",
        },
        "group_order": list(GROUP_NAMES),
        "feature_groups": {
            feature_set: {group: list(columns) for group, columns in feature_groups(feature_set).items()}
            for feature_set in ("descriptor24", "two_slot38")
        },
        "coalitions": [
            {"coalition_id": coalition_id(value), "groups": list(value)}
            for value in COALITIONS
        ],
        "model_definitions": report.MODEL_DEFINITIONS,
        "pca_policy": {
            "source": "retained fitted model artifacts used by the overall benchmark",
            "fit_chemical_count": len(pca_provenance["fit_chemical_names"]),
            "fit_chemical_names": list(pca_provenance["fit_chemical_names"]),
            "components_sha256": pca_provenance["pca_components_sha256"],
            "scores_sha256": retained_pca["scores_sha256"],
            "structure_registry_sha256": context.structure_sha256,
            "molformer_model_id": report.MOLFORMER_MODEL_ID,
            "molformer_snapshot_revision": report.MOLFORMER_SNAPSHOT_REVISION,
            "artifact_sha256": retained_pca["artifact_sha256"],
        },
        "software": {
            "python": report.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
        },
        "checks": checks,
        "shard_sha256": {
            path.name: _sha256(path) for path in expected_shards()
        },
        "shard_manifest_sha256": {
            shard_manifest_path(model_id, seed).name: _sha256(
                shard_manifest_path(model_id, seed)
            )
            for model_id in report.MODEL_DEFINITIONS
            for seed in report.STATIC_SEEDS
        },
        "output_sha256": output_hashes,
        "interpretation": (
            "Predictive information available to the specified learner and representation within "
            "the represented formulation distribution; not an effect direction, molecular mechanism, "
            "unseen-chemical guarantee, or causal estimate."
        ),
    }
    _atomic_text(
        json.dumps(manifest, indent=2) + "\n", OUTPUT_DIR / "manifest.json"
    )
    paths = {
        "predictions": OUTPUT_DIR / "coalition_predictions_long.csv",
        "importance": OUTPUT_DIR / "shapley_importance.csv",
        "figure_png": OUTPUT_DIR / "feature_importance_four_panel.png",
        "figure_pdf": OUTPUT_DIR / "feature_importance_four_panel.pdf",
        "manifest": OUTPUT_DIR / "manifest.json",
        "notes": OUTPUT_DIR / "results_notes.md",
    }
    print(f"Compiled four-block feature importance under {OUTPUT_DIR}", flush=True)
    return paths
