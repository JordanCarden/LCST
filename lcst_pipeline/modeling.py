from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBRegressor

from .features import (
    FEATURE_ENCODING_VERSION,
    CONCENTRATION_REFERENCES,
    add_buffer_feature,
    build_descriptor23,
    build_two_slot37,
)
from .embeddings import (
    MolFormerPCAResult,
    fit_molformer_pca,
    generate_molformer_embeddings,
    structure_registry_sha256,
    validate_structure_registry,
)
from .schema import (
    FORMULATION_COLUMNS,
    HEATING_RATE_SHEET,
    MASTER_PATH,
    OBSERVED_MASTER_PATH,
    POLYMER_MASTER_PATH,
    REPO_ROOT,
    load_config,
)


MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CLASS_LABELS = ("LCST", "UCST", "NONE")
TARGET_COLUMNS = ("target_lcst", "target_ucst", "target_none")

MODEL_DEFINITIONS = {
    "descriptor24_buffer_regressor_xgboost": {
        "task": "regression",
        "feature_set": "descriptor24",
        "kind": "xgboost",
    },
    "descriptor24_buffer_regressor_mlp": {
        "task": "regression",
        "feature_set": "descriptor24",
        "kind": "mlp",
    },
    "two_slot_pca38_buffer_regressor_xgboost": {
        "task": "regression",
        "feature_set": "two_slot38",
        "kind": "xgboost",
    },
    "two_slot_pca38_buffer_regressor_mlp": {
        "task": "regression",
        "feature_set": "two_slot38",
        "kind": "mlp",
    },
    "descriptor24_buffer_classifier_xgboost": {
        "task": "classification",
        "feature_set": "descriptor24",
        "kind": "xgboost",
    },
    "descriptor24_buffer_classifier_mlp": {
        "task": "classification",
        "feature_set": "descriptor24",
        "kind": "mlp",
    },
    "two_slot_pca38_buffer_classifier_xgboost": {
        "task": "classification",
        "feature_set": "two_slot38",
        "kind": "xgboost",
    },
    "two_slot_pca38_buffer_classifier_mlp": {
        "task": "classification",
        "feature_set": "two_slot38",
        "kind": "mlp",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_groups(frame: pd.DataFrame):
    return frame.groupby(FORMULATION_COLUMNS, sort=True, dropna=False)


def build_regression_conditions(master: pd.DataFrame) -> pd.DataFrame:
    eligible = master[
        master["transition_type"].eq("LCST")
        & master["transition_temperature_c"].notna()
        & ~master["source_sheet"].eq(HEATING_RATE_SHEET)
    ].copy()
    records: list[dict[str, Any]] = []
    for key, group in _condition_groups(eligible):
        record = dict(zip(FORMULATION_COLUMNS, key))
        record["n_measurements"] = len(group)
        record["transition_temperature_c"] = float(group["transition_temperature_c"].mean())
        records.append(record)
    return pd.DataFrame.from_records(records)


def _dominant_class(target: np.ndarray) -> str:
    maximum = float(target.max())
    tied = {CLASS_LABELS[index] for index, value in enumerate(target) if np.isclose(value, maximum)}
    for label in ("NONE", "UCST", "LCST"):
        if label in tied:
            return label
    raise AssertionError("Classifier target has no class")


def build_classifier_conditions(master: pd.DataFrame) -> pd.DataFrame:
    eligible = master[
        ~master["source_sheet"].eq(HEATING_RATE_SHEET)
    ].copy()
    records: list[dict[str, Any]] = []
    for key, group in _condition_groups(eligible):
        counts = group["transition_type"].value_counts()
        target = np.asarray([counts.get(label, 0) / len(group) for label in CLASS_LABELS], dtype=float)
        record = dict(zip(FORMULATION_COLUMNS, key))
        record.update(
            {
                "n_measurements": len(group),
                "target_lcst": target[0],
                "target_ucst": target[1],
                "target_none": target[2],
                "stratification_class": _dominant_class(target),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _xgboost_regressor(seed: int) -> XGBRegressor:
    return XGBRegressor(
        random_state=seed,
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=1,
        verbosity=0,
    )


def _regression_pipeline(seed: int) -> Pipeline:
    return Pipeline([("scaler", MaxAbsScaler()), ("model", _xgboost_regressor(seed))])


class SoftTargetXGBClassifier:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.booster: xgb.Booster | None = None
        self.feature_columns: list[str] = []

    @staticmethod
    def softmax(margins: np.ndarray) -> np.ndarray:
        centered = margins - margins.max(axis=1, keepdims=True)
        values = np.exp(centered)
        return values / values.sum(axis=1, keepdims=True)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "seed": self.seed,
            "nthread": 1,
            "verbosity": 0,
        }

    def fit(self, features: pd.DataFrame, targets: np.ndarray) -> "SoftTargetXGBClassifier":
        target = np.asarray(targets, dtype=float)
        self.feature_columns = list(features.columns)
        dtrain = xgb.DMatrix(features, feature_names=self.feature_columns)

        def objective(margins: np.ndarray, _: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
            probabilities = self.softmax(margins)
            gradient = probabilities - target
            hessian = np.maximum(2.0 * probabilities * (1.0 - probabilities), 1e-6)
            return gradient, hessian

        self.booster = xgb.train(self.parameters, dtrain, num_boost_round=500, obj=objective)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Classifier is not fitted")
        if list(features.columns) != self.feature_columns:
            raise ValueError("Prediction columns do not match classifier training columns")
        margins = np.asarray(
            self.booster.predict(xgb.DMatrix(features, feature_names=self.feature_columns), output_margin=True),
            dtype=float,
        ).reshape(len(features), 3)
        return self.softmax(margins)


def _feature_tables(conditions: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, pd.DataFrame]:
    descriptor = add_buffer_feature(build_descriptor23(conditions, metadata), conditions)
    two_slot = add_buffer_feature(build_two_slot37(conditions, metadata), conditions)
    return {"descriptor24": descriptor, "two_slot38": two_slot}


def _regression_metrics(conditions: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .evaluation import fit_predict_mlp_regressor

    target = conditions["transition_temperature_c"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model_id, definition in MODEL_DEFINITIONS.items():
        if definition["task"] != "regression":
            continue
        features = tables[str(definition["feature_set"])]
        repeat_scores: list[dict[str, float]] = []
        for repeat in range(5):
            predictions = np.full(len(conditions), np.nan)
            splitter = KFold(n_splits=10, shuffle=True, random_state=42 + repeat)
            for train, test in splitter.split(features):
                if definition["kind"] == "xgboost":
                    model = _regression_pipeline(42 + repeat)
                    model.fit(features.iloc[train], target[train])
                    predictions[test] = model.predict(features.iloc[test])
                else:
                    predictions[test], _ = fit_predict_mlp_regressor(
                        features.iloc[train],
                        target[train],
                        features.iloc[test],
                        42 + repeat,
                    )
            repeat_scores.append(
                {
                    "mae": float(mean_absolute_error(target, predictions)),
                    "rmse": float(math.sqrt(mean_squared_error(target, predictions))),
                    "r2": float(r2_score(target, predictions)),
                    "pearson_r": float(np.corrcoef(target, predictions)[0, 1]),
                }
            )
        scores = pd.DataFrame(repeat_scores)
        row: dict[str, Any] = {
            "model_id": model_id,
            "feature_set": definition["feature_set"],
            "feature_count": features.shape[1],
            "n_conditions": len(conditions),
            "n_repeats": 5,
        }
        for metric in ("mae", "rmse", "r2", "pearson_r"):
            row[f"{metric}_mean"] = float(scores[metric].mean())
            row[f"{metric}_std"] = float(scores[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)


def _distribution_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(predicted, 1e-15, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    positive = target > 0
    absolute = np.abs(probabilities - target)
    return {
        "cross_entropy": float(-np.sum(np.where(positive, target * np.log(probabilities), 0.0), axis=1).mean()),
        "probability_mae": float(absolute.mean()),
        "total_variation": float((0.5 * absolute.sum(axis=1)).mean()),
        "soft_brier": float(np.sum((probabilities - target) ** 2, axis=1).mean()),
    }


def _classifier_metrics(conditions: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .evaluation import fit_predict_mlp_classifier

    target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    strata = conditions["stratification_class"].to_numpy()
    hard_mask = (target > 0).sum(axis=1) == 1
    rows: list[dict[str, Any]] = []
    for model_id, definition in MODEL_DEFINITIONS.items():
        if definition["task"] != "classification":
            continue
        features = tables[str(definition["feature_set"])]
        repeat_scores: list[dict[str, float]] = []
        for repeat in range(5):
            probabilities = np.full((len(conditions), 3), np.nan)
            splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + repeat)
            for train, test in splitter.split(features, strata):
                if definition["kind"] == "xgboost":
                    classifier = SoftTargetXGBClassifier(42 + repeat).fit(
                        features.iloc[train], target[train]
                    )
                    probabilities[test] = classifier.predict_proba(features.iloc[test])
                else:
                    probabilities[test], _ = fit_predict_mlp_classifier(
                        features.iloc[train],
                        target[train],
                        strata[train],
                        features.iloc[test],
                        42 + repeat,
                    )
            scores = _distribution_metrics(target, probabilities)
            true_hard = target[hard_mask].argmax(axis=1)
            predicted_hard = probabilities[hard_mask].argmax(axis=1)
            scores.update(
                {
                    "accuracy": float(accuracy_score(true_hard, predicted_hard)),
                    "balanced_accuracy": float(balanced_accuracy_score(true_hard, predicted_hard)),
                    "macro_f1": float(f1_score(true_hard, predicted_hard, average="macro", zero_division=0)),
                }
            )
            repeat_scores.append(scores)
        scores = pd.DataFrame(repeat_scores)
        row: dict[str, Any] = {
            "model_id": model_id,
            "feature_set": definition["feature_set"],
            "feature_count": features.shape[1],
            "n_conditions": len(conditions),
            "n_repeats": 5,
        }
        for metric in ("cross_entropy", "probability_mae", "total_variation", "soft_brier", "accuracy", "balanced_accuracy", "macro_f1"):
            row[f"{metric}_mean"] = float(scores[metric].mean())
            row[f"{metric}_std"] = float(scores[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cross_entropy_mean").reset_index(drop=True)


def _artifact(
    model_id: str,
    definition: dict[str, str],
    model: Any,
    feature_columns: list[str],
    metadata: dict[str, Any],
    master_hash: str,
    target_policy: str,
    molformer_pca: MolFormerPCAResult | None,
    selected_epoch: int | None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "task": definition["task"],
        "feature_set": definition["feature_set"],
        "model_family": definition["kind"],
        "feature_columns": feature_columns,
        "feature_encoding": FEATURE_ENCODING_VERSION,
        "concentration_references": CONCENTRATION_REFERENCES,
        "class_labels": list(CLASS_LABELS) if definition["task"] == "classification" else None,
        "model": model,
        "selected_epoch": selected_epoch,
        "chemical_metadata": metadata,
        "molformer_pca": (
            {
                "transformer": molformer_pca.pca,
                "provenance": molformer_pca.provenance,
            }
            if molformer_pca is not None and definition["feature_set"] == "two_slot38"
            else None
        ),
        "training_master_sha256": master_hash,
        "training_rules": {
            "pool_across_workbooks": True,
            "exclude_source_sheet": HEATING_RATE_SHEET,
            "target_policy": target_policy,
        },
    }


def train_all(master_path: Path = MASTER_PATH) -> dict[str, Path]:
    from .evaluation import fit_mlp_classifier, fit_mlp_regressor

    # The report protocol is CPU-only and single-threaded for reproducibility.
    import torch

    torch.set_num_threads(1)

    master = pd.read_csv(master_path, keep_default_na=False, na_values=[""])
    config = load_config()
    configured_metadata = config["chemical_metadata"]
    structures = validate_structure_registry(configured_metadata)
    configured_names = tuple(sorted(structures))
    embeddings = generate_molformer_embeddings(
        structures,
        chemical_names=configured_names,
        device="cpu",
    )
    molformer_pca = fit_molformer_pca(
        embeddings,
        fit_chemical_names=configured_names,
        transform_chemical_names=configured_names,
        structure_sha256=structure_registry_sha256(structures),
    )
    metadata = dict(configured_metadata)
    metadata["molformer_pca16"] = molformer_pca.scores
    regression = build_regression_conditions(master)
    classification = build_classifier_conditions(master)
    regression_tables = _feature_tables(regression, metadata)
    classifier_tables = _feature_tables(classification, metadata)
    regression_metrics = _regression_metrics(regression, regression_tables)
    classifier_metrics = _classifier_metrics(classification, classifier_tables)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    master_hash = sha256_file(master_path)
    resolved_master_path = master_path.resolve()
    if resolved_master_path == OBSERVED_MASTER_PATH.resolve():
        target_policy = "observed_behavior"
    elif resolved_master_path == POLYMER_MASTER_PATH.resolve():
        target_policy = "assumed_polymer_behavior"
    else:
        target_policy = "custom_input"
    artifacts: dict[str, dict[str, Any]] = {}
    for model_id, definition in MODEL_DEFINITIONS.items():
        conditions = regression if definition["task"] == "regression" else classification
        tables = regression_tables if definition["task"] == "regression" else classifier_tables
        features = tables[str(definition["feature_set"])]
        selected_epoch: int | None = None
        if definition["task"] == "regression":
            target = conditions["transition_temperature_c"].to_numpy(dtype=float)
            if definition["kind"] == "xgboost":
                model = _regression_pipeline(42)
                model.fit(features, target)
            else:
                model = fit_mlp_regressor(features, target, 42)
                selected_epoch = model.selected_epoch
        else:
            target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
            if definition["kind"] == "xgboost":
                model = SoftTargetXGBClassifier(42).fit(features, target)
            else:
                model = fit_mlp_classifier(
                    features,
                    target,
                    conditions["stratification_class"].to_numpy(),
                    42,
                )
                selected_epoch = model.selected_epoch
        artifacts[model_id] = _artifact(
            model_id,
            definition,
            model,
            list(features.columns),
            metadata,
            master_hash,
            target_policy,
            molformer_pca,
            selected_epoch,
        )

    regression_path = OUTPUTS_DIR / "regression_metrics.csv"
    classifier_path = OUTPUTS_DIR / "classifier_metrics.csv"
    paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="lcst-training-", dir=REPO_ROOT) as staging:
        staging_dir = Path(staging)
        staged_paths: dict[str, Path] = {}
        for model_id, artifact in artifacts.items():
            staged_path = staging_dir / f"{model_id}.joblib"
            joblib.dump(artifact, staged_path)
            staged_paths[model_id] = staged_path
        staged_regression = staging_dir / regression_path.name
        staged_classifier = staging_dir / classifier_path.name
        regression_metrics.to_csv(staged_regression, index=False)
        classifier_metrics.to_csv(staged_classifier, index=False)

        for model_id, staged_path in staged_paths.items():
            path = MODELS_DIR / staged_path.name
            os.replace(staged_path, path)
            paths[model_id] = path
        os.replace(staged_regression, regression_path)
        os.replace(staged_classifier, classifier_path)

    paths["regression_metrics"] = regression_path
    paths["classifier_metrics"] = classifier_path
    return paths
