"""Fresh eight-model LCST evaluation on the complete current master dataset."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "report_evaluation"
EMBEDDING_CACHE = OUTPUT_DIR / "molformer_embeddings.npz"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lcst-report-evaluation-mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBRegressor

from lcst_pipeline import evaluation as metric_helpers
from lcst_pipeline.evaluation import (
    MLP_CLASSIFICATION_PROTOCOL,
    MLP_REGRESSION_PROTOCOL,
    fit_predict_mlp_classifier,
    fit_predict_mlp_regressor,
)
from lcst_pipeline.features import (
    FEATURE_ENCODING_VERSION,
    CONCENTRATION_REFERENCES,
    BUFFER_COLUMN,
    DESCRIPTOR_23_COLUMNS,
    TWO_SLOT_37_COLUMNS,
    add_buffer_feature,
    build_descriptor23,
    build_two_slot37,
)
from lcst_pipeline.embeddings import (
    MOLFORMER_MODEL_ID,
    MOLFORMER_SNAPSHOT_REVISION,
    MolFormerPCAResult,
    fit_molformer_pca,
    generate_molformer_embeddings,
    structure_registry_sha256,
    validate_structure_registry,
)
from lcst_pipeline.modeling import (
    CLASS_LABELS,
    TARGET_COLUMNS,
    SoftTargetXGBClassifier,
    build_classifier_conditions,
    build_regression_conditions,
)
from lcst_pipeline.schema import FORMULATION_COLUMNS, HEATING_RATE_SHEET, MASTER_PATH, load_config


STATIC_SEEDS = tuple(range(42, 47))
ADAPTATION_REPEATS = tuple(range(10))
BOOTSTRAPS = 2000
NO_ADDITIVE = "No additive"
NO_SALT = "No salt"
# These lists select presentation/adaptation panels, never evaluation coverage.
MAIN_ADDITIVES = ("CHAPS", "CTAB", "SDS")
MAIN_SALTS = ("NaCl", "Na2SO4")
RARE_SALTS = (
    "CaCl2",
    "KCl",
    "MgCl2",
    "Na2HPO4",
    "Na2S2O3",
    "NaBr",
    "NaI",
    "NaNO3",
    "NaSCN",
)
ADAPTATION_DOMAINS = (*MAIN_ADDITIVES, *MAIN_SALTS)
POOLED_HOLDOUTS = {"rare_additive_pool", "all_additives_pooled", "unseen_Hofmeister_panel"}
HOLDOUT_COLUMNS = {
    "unseen_additive": "additive_name",
    "unseen_salt": "salt_name",
    "unseen_molecular_weight": "polymer_mw_kda",
}


MODEL_DEFINITIONS: dict[str, dict[str, str]] = {
    "descriptor24_buffer_regressor_xgboost": {
        "task": "regression",
        "feature_set": "descriptor24",
        "family": "XGBoost",
    },
    "descriptor24_buffer_regressor_mlp": {
        "task": "regression",
        "feature_set": "descriptor24",
        "family": "MLP",
    },
    "two_slot_pca38_buffer_regressor_xgboost": {
        "task": "regression",
        "feature_set": "two_slot38",
        "family": "XGBoost",
    },
    "two_slot_pca38_buffer_regressor_mlp": {
        "task": "regression",
        "feature_set": "two_slot38",
        "family": "MLP",
    },
    "descriptor24_buffer_classifier_xgboost": {
        "task": "classification",
        "feature_set": "descriptor24",
        "family": "XGBoost",
    },
    "descriptor24_buffer_classifier_mlp": {
        "task": "classification",
        "feature_set": "descriptor24",
        "family": "MLP",
    },
    "two_slot_pca38_buffer_classifier_xgboost": {
        "task": "classification",
        "feature_set": "two_slot38",
        "family": "XGBoost",
    },
    "two_slot_pca38_buffer_classifier_mlp": {
        "task": "classification",
        "feature_set": "two_slot38",
        "family": "MLP",
    },
}
REGRESSION_MODELS = tuple(
    model_id for model_id, definition in MODEL_DEFINITIONS.items() if definition["task"] == "regression"
)
CLASSIFICATION_MODELS = tuple(
    model_id for model_id, definition in MODEL_DEFINITIONS.items() if definition["task"] == "classification"
)


@dataclass(frozen=True)
class StaticSplit:
    challenge: str
    holdout: str
    train_regression: pd.DataFrame
    test_regression: pd.DataFrame
    train_classification: pd.DataFrame
    test_classification: pd.DataFrame
    exploratory: bool = False
    notes: str = ""


@dataclass
class PCAContext:
    chemical_names: tuple[str, ...]
    embeddings: dict[str, np.ndarray]
    structure_sha256: str
    results: dict[str, MolFormerPCAResult]
    legacy_reproduction_max_abs_difference: float = np.nan


_INPUT_CACHE: tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    PCAContext,
] | None = None


class SingleThreadSoftTargetXGBClassifier(SoftTargetXGBClassifier):
    @property
    def parameters(self) -> dict[str, Any]:
        parameters = super().parameters
        parameters["nthread"] = 1
        return parameters


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.splitlines()


def feature_tables(conditions: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, pd.DataFrame]:
    descriptor = add_buffer_feature(build_descriptor23(conditions, metadata), conditions)
    two_slot = add_buffer_feature(build_two_slot37(conditions, metadata), conditions)
    expected_descriptor = [*DESCRIPTOR_23_COLUMNS, BUFFER_COLUMN]
    expected_two_slot = [*TWO_SLOT_37_COLUMNS, BUFFER_COLUMN]
    if list(descriptor.columns) != expected_descriptor or list(two_slot.columns) != expected_two_slot:
        raise AssertionError("The 24/38-feature contracts were not preserved")
    return {"descriptor24": descriptor, "two_slot38": two_slot}


def pca_result(context: PCAContext, excluded_chemical: str | None = None) -> MolFormerPCAResult:
    key = excluded_chemical or "all_18"
    if key not in context.results:
        fit_names = tuple(
            name for name in context.chemical_names if name != excluded_chemical
        )
        result = fit_molformer_pca(
            context.embeddings,
            fit_chemical_names=fit_names,
            transform_chemical_names=context.chemical_names,
            structure_sha256=context.structure_sha256,
        )
        expected_count = 17 if excluded_chemical else 18
        if len(result.fit_chemical_names) != expected_count:
            raise AssertionError(f"PCA fit used {len(result.fit_chemical_names)} rather than {expected_count} chemicals")
        if excluded_chemical and excluded_chemical in result.fit_chemical_names:
            raise AssertionError(f"Held-out chemical {excluded_chemical!r} entered PCA fitting")
        context.results[key] = result
    return context.results[key]


def metadata_with_pca(metadata: dict[str, Any], result: MolFormerPCAResult) -> dict[str, Any]:
    updated = dict(metadata)
    updated["molformer_pca16"] = result.scores
    return updated


def pca_result_for_split(split: StaticSplit, context: PCAContext) -> MolFormerPCAResult:
    excluded = split.holdout if split.challenge in {"unseen_additive", "unseen_salt"} else None
    return pca_result(context, excluded)


def pca_inventory_fields(result: MolFormerPCAResult, excluded_chemical: str | None) -> dict[str, Any]:
    return {
        "pca_fit_count": len(result.fit_chemical_names),
        "pca_fit_chemical_names": ";".join(result.fit_chemical_names),
        "pca_excluded_chemical": excluded_chemical or "",
        "pca_components_sha256": result.provenance["pca_components_sha256"],
    }


def evaluation_embeddings(
    structures: dict[str, dict[str, str]],
    chemical_names: tuple[str, ...],
    registry_sha256: str,
) -> dict[str, np.ndarray]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = Path("/tmp") / f"lcst-molformer-{registry_sha256}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not EMBEDDING_CACHE.exists():
            generated = generate_molformer_embeddings(
                structures,
                chemical_names=chemical_names,
                device="cpu",
            )
            matrix = np.vstack([generated[name] for name in chemical_names]).astype(np.float32)
            temporary = OUTPUT_DIR / f".molformer_embeddings.{os.getpid()}.npz"
            np.savez_compressed(
                temporary,
                chemical_names=np.asarray(chemical_names),
                embeddings=matrix,
                structure_registry_sha256=np.asarray(registry_sha256),
                model_id=np.asarray(MOLFORMER_MODEL_ID),
                snapshot_revision=np.asarray(MOLFORMER_SNAPSHOT_REVISION),
            )
            os.replace(temporary, EMBEDDING_CACHE)
        with np.load(EMBEDDING_CACHE, allow_pickle=False) as cached:
            cached_names = tuple(cached["chemical_names"].astype(str).tolist())
            matrix = np.asarray(cached["embeddings"], dtype=np.float32)
            cached_registry_hash = str(cached["structure_registry_sha256"].item())
            cached_model_id = str(cached["model_id"].item())
            cached_revision = str(cached["snapshot_revision"].item())
    if cached_names != chemical_names:
        raise AssertionError("Cached MoLFormer chemical order does not match the configured order")
    if cached_registry_hash != registry_sha256:
        raise AssertionError("Cached MoLFormer structures do not match the configured structures")
    if cached_model_id != MOLFORMER_MODEL_ID or cached_revision != MOLFORMER_SNAPSHOT_REVISION:
        raise AssertionError("Cached MoLFormer encoder identity does not match the configured encoder")
    if matrix.shape[0] != len(chemical_names) or not np.isfinite(matrix).all():
        raise AssertionError(f"Invalid cached MoLFormer embedding matrix: {matrix.shape}")
    return {name: matrix[index] for index, name in enumerate(chemical_names)}


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    PCAContext,
]:
    global _INPUT_CACHE
    if _INPUT_CACHE is not None:
        return _INPUT_CACHE
    master = pd.read_csv(MASTER_PATH, keep_default_na=False, na_values=[""])
    eligible = master.loc[~master["source_sheet"].eq(HEATING_RATE_SHEET)].copy()
    regression = metric_helpers.add_ids(build_regression_conditions(master))
    classification = metric_helpers.add_ids(build_classifier_conditions(master))
    if len(master) != 1255 or len(eligible) != 1143:
        raise AssertionError(f"Unexpected measurement counts: master={len(master)}, eligible={len(eligible)}")
    if len(regression) != 272 or len(classification) != 403:
        raise AssertionError(
            f"Unexpected pooled counts: regression={len(regression)}, classification={len(classification)}"
        )
    controls = classification.loc[~classification["polymer_name"].eq("Dex-MA")]
    control_classes = controls["stratification_class"].value_counts().to_dict()
    if len(controls) != 48 or control_classes != {"NONE": 33, "UCST": 11, "LCST": 4}:
        raise AssertionError(f"Unexpected observed no-DexMA targets: {control_classes}")
    control_measurements = eligible.loc[~eligible["polymer_name"].eq("Dex-MA"), "transition_type"]
    if control_measurements.value_counts().to_dict() != {"NONE": 33, "UCST": 11, "LCST": 4}:
        raise AssertionError("Observed no-DexMA measurements do not match the source labels")
    metadata = load_config()["chemical_metadata"]
    expected_chemicals = tuple(sorted(metadata["molformer_pca16"]))
    structures = validate_structure_registry(metadata, expected_names=expected_chemicals)
    if len(expected_chemicals) != 18:
        raise AssertionError(f"Expected exactly 18 configured chemicals, found {len(expected_chemicals)}")
    registry_hash = structure_registry_sha256(structures)
    embeddings = evaluation_embeddings(
        structures,
        expected_chemicals,
        registry_hash,
    )
    context = PCAContext(
        chemical_names=expected_chemicals,
        embeddings=embeddings,
        structure_sha256=registry_hash,
        results={},
    )
    legacy_result = pca_result(context, "Span-85")
    stored_scores = metadata["molformer_pca16"]
    legacy_difference = max(
        float(
            np.max(
                np.abs(
                    np.asarray(legacy_result.scores[name], dtype=float)
                    - np.asarray(stored_scores[name], dtype=float)
                )
            )
        )
        for name in expected_chemicals
    )
    if legacy_difference > 1e-5:
        raise AssertionError(
            "Regenerated MoLFormer embeddings/PCA do not reproduce the stored legacy vectors; "
            f"maximum absolute difference={legacy_difference:.6g}"
        )
    context.legacy_reproduction_max_abs_difference = legacy_difference
    all_chemical_metadata = metadata_with_pca(metadata, pca_result(context))
    regression_tables = feature_tables(regression, all_chemical_metadata)
    classification_tables = feature_tables(classification, all_chemical_metadata)
    for tables in (regression_tables, classification_tables):
        if tables["descriptor24"].shape[1] != 24 or tables["two_slot38"].shape[1] != 38:
            raise AssertionError("Incorrect feature width")
    _INPUT_CACHE = master, regression, classification, metadata, context
    return _INPUT_CACHE


def xgb_regressor(seed: int) -> Pipeline:
    estimator = XGBRegressor(
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
    return Pipeline([("scaler", MaxAbsScaler()), ("model", estimator)])


def fit_model(
    task: str,
    model_id: str,
    train_features: pd.DataFrame,
    train_conditions: pd.DataFrame,
    test_features: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, float]:
    definition = MODEL_DEFINITIONS[model_id]
    if list(train_features.columns) != list(test_features.columns):
        raise AssertionError(f"Train/test feature mismatch for {model_id}")
    if task == "regression":
        target = train_conditions["transition_temperature_c"].to_numpy(dtype=float)
        if definition["family"] == "XGBoost":
            model = xgb_regressor(seed).fit(train_features, target)
            prediction = np.asarray(model.predict(test_features), dtype=float)
            epoch = np.nan
        else:
            prediction, epoch = fit_predict_mlp_regressor(
                train_features, target, test_features, seed
            )
    else:
        target = train_conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        if definition["family"] == "XGBoost":
            model = SingleThreadSoftTargetXGBClassifier(seed).fit(train_features, target)
            prediction = np.asarray(model.predict_proba(test_features), dtype=float)
            epoch = np.nan
        else:
            prediction, epoch = fit_predict_mlp_classifier(
                train_features,
                target,
                train_conditions["stratification_class"].to_numpy(),
                test_features,
                seed,
            )
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise AssertionError(f"Non-finite predictions from {model_id}")
    if task == "classification" and not np.allclose(prediction.sum(axis=1), 1.0, atol=1e-7):
        raise AssertionError(f"Probabilities do not sum to one for {model_id}")
    return prediction, float(epoch)


def prediction_frame(
    conditions: pd.DataFrame,
    task: str,
    challenge: str,
    holdout: str,
    model_id: str,
    seed: int,
    exploratory: bool,
    notes: str,
    prediction: np.ndarray,
    epochs: np.ndarray | float,
) -> pd.DataFrame:
    frame = metric_helpers.prediction_base(
        conditions, challenge, holdout, task, model_id, seed, exploratory, notes
    )
    definition = MODEL_DEFINITIONS[model_id]
    frame["feature_set"] = definition["feature_set"]
    frame["model_family"] = definition["family"]
    frame["selected_epoch"] = epochs
    if task == "regression":
        frame["target_c"] = conditions["transition_temperature_c"].to_numpy(dtype=float)
        frame["prediction_c"] = prediction
    else:
        target = conditions[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        for index, label in enumerate(CLASS_LABELS):
            frame[f"target_{label.lower()}"] = target[:, index]
            frame[f"prediction_{label.lower()}"] = prediction[:, index]
    return frame


def overall_predictions(
    regression: pd.DataFrame, classification: pd.DataFrame, metadata: dict[str, Any]
) -> list[pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    for task, conditions, models in (
        ("regression", regression, REGRESSION_MODELS),
        ("classification", classification, CLASSIFICATION_MODELS),
    ):
        tables = feature_tables(conditions, metadata)
        strata = conditions["stratification_class"].to_numpy() if task == "classification" else None
        for seed in STATIC_SEEDS:
            splitter = (
                KFold(n_splits=10, shuffle=True, random_state=seed)
                if task == "regression"
                else StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            )
            split_iterator = splitter.split(conditions, strata) if strata is not None else splitter.split(conditions)
            predictions: dict[str, np.ndarray] = {
                model_id: np.full((len(conditions), 3), np.nan)
                if task == "classification"
                else np.full(len(conditions), np.nan)
                for model_id in models
            }
            epochs = {model_id: np.full(len(conditions), np.nan) for model_id in models}
            for train_index, test_index in split_iterator:
                train_conditions = conditions.iloc[train_index]
                for model_id in models:
                    feature_set = MODEL_DEFINITIONS[model_id]["feature_set"]
                    prediction, epoch = fit_model(
                        task,
                        model_id,
                        tables[feature_set].iloc[train_index],
                        train_conditions,
                        tables[feature_set].iloc[test_index],
                        seed,
                    )
                    predictions[model_id][test_index] = prediction
                    epochs[model_id][test_index] = epoch
            for model_id in models:
                outputs.append(
                    prediction_frame(
                        conditions,
                        task,
                        "overall_benchmark",
                        "repeated_10fold",
                        model_id,
                        seed,
                        False,
                        "Five repeated shuffled 10-fold evaluations",
                        predictions[model_id],
                        epochs[model_id],
                    )
                )
    return outputs


def split_from_masks(
    challenge: str,
    holdout: str,
    regression: pd.DataFrame,
    classification: pd.DataFrame,
    regression_mask: pd.Series,
    classification_mask: pd.Series,
    exploratory: bool,
    notes: str,
) -> StaticSplit:
    return StaticSplit(
        challenge=challenge,
        holdout=holdout,
        train_regression=regression.loc[~regression_mask].reset_index(drop=True),
        test_regression=regression.loc[regression_mask].reset_index(drop=True),
        train_classification=classification.loc[~classification_mask].reset_index(drop=True),
        test_classification=classification.loc[classification_mask].reset_index(drop=True),
        exploratory=exploratory,
        notes=notes,
    )


def holdout_targets(
    regression: pd.DataFrame, classification: pd.DataFrame,
) -> list[tuple[str, str, str, str | float]]:
    """Discover every real ingredient identity and polymer MW in eligible data."""
    combined = pd.concat([regression, classification], ignore_index=True)
    targets: list[tuple[str, str, str, str | float]] = []
    for challenge, column, absent in (
        ("unseen_additive", "additive_name", NO_ADDITIVE),
        ("unseen_salt", "salt_name", NO_SALT),
    ):
        names = set(combined[column].dropna()) - {absent, ""}
        targets.extend((challenge, name, column, name) for name in sorted(names))
    present = combined["polymer_name"].notna() & ~combined["polymer_name"].isin({"No polymer", ""})
    weights = pd.to_numeric(combined.loc[present, "polymer_mw_kda"], errors="raise")
    weights = weights.loc[np.isfinite(weights) & weights.gt(0)]
    targets.extend(
        ("unseen_molecular_weight", f"{weight:g}kDa", "polymer_mw_kda", float(weight))
        for weight in sorted(weights.unique())
    )
    return targets


def holdout_mask(conditions: pd.DataFrame, column: str, value: str | float) -> pd.Series:
    mask = conditions[column].eq(value)
    if column == "polymer_mw_kda":
        mask &= conditions["polymer_name"].notna() & ~conditions["polymer_name"].isin({"No polymer", ""})
    return mask


def static_splits(regression: pd.DataFrame, classification: pd.DataFrame) -> list[StaticSplit]:
    splits: list[StaticSplit] = []
    for challenge, holdout, column, value in holdout_targets(regression, classification):
        molecular_weight = challenge == "unseen_molecular_weight"
        exploratory = (
            value == 500.0 if molecular_weight else
            holdout not in (MAIN_ADDITIVES if challenge == "unseen_additive" else MAIN_SALTS)
        )
        notes = (
            f"Every {value:g} kDa condition was removed from training" if molecular_weight else
            f"Every condition containing {holdout} was removed from training"
        )
        splits.append(split_from_masks(
            challenge, holdout, regression, classification,
            holdout_mask(regression, column, value),
            holdout_mask(classification, column, value), exploratory, notes,
        ))
    return splits


def verify_holdout_coverage(
    predictions: pd.DataFrame, regression: pd.DataFrame, classification: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:
    """Require every eligible condition, model and seed for every data-derived holdout."""
    columns = ["challenge", "holdout", "task", "model_id", "seed", "condition_id"]
    expected: set[tuple] = set()
    coverage = {challenge: {"regression": [], "classification": []} for challenge in HOLDOUT_COLUMNS}
    for challenge, holdout, column, value in holdout_targets(regression, classification):
        for task, conditions, models in (
            ("regression", regression, REGRESSION_MODELS),
            ("classification", classification, CLASSIFICATION_MODELS),
        ):
            ids = conditions.loc[holdout_mask(conditions, column, value), "condition_id"]
            if not ids.empty:
                coverage[challenge][task].append(holdout)
            expected.update(
                (challenge, holdout, task, model, seed, condition)
                for model in models for seed in STATIC_SEEDS for condition in ids
            )
    individual = predictions.loc[
        predictions["challenge"].isin(HOLDOUT_COLUMNS)
        & ~predictions["holdout"].isin(POOLED_HOLDOUTS), columns
    ]
    if individual.duplicated().any():
        raise AssertionError("Duplicate individual holdout predictions")
    actual = set(individual.itertuples(index=False, name=None))
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise AssertionError(
            f"Incomplete holdout coverage: {len(missing)} missing, {len(unexpected)} unexpected predictions; "
            f"missing examples={sorted(missing)[:3]}; unexpected examples={sorted(unexpected)[:3]}"
        )
    return coverage


def verify_split(split: StaticSplit) -> None:
    for task, train, test in (
        ("regression", split.train_regression, split.test_regression),
        ("classification", split.train_classification, split.test_classification),
    ):
        if test.empty:
            continue
        metric_helpers.assert_disjoint(train, test, f"{split.challenge}/{split.holdout}/{task}")
        column = HOLDOUT_COLUMNS[split.challenge]
        value = float(split.holdout.removesuffix("kDa")) if column == "polymer_mw_kda" else split.holdout
        if holdout_mask(train, column, value).any():
            raise AssertionError(f"Held-out group remains in {task} training: {split.holdout}")
        if not holdout_mask(test, column, value).all():
            raise AssertionError(f"Test set contains another group: {split.holdout}/{task}")
        if train.empty:
            raise ValueError(f"No training conditions remain for {split.holdout}/{task}")


def inventory_rows(
    splits: list[StaticSplit], context: PCAContext,
    regression: pd.DataFrame, classification: pd.DataFrame,
) -> list[dict[str, Any]]:
    all_chemical_pca = pca_result(context)
    rows: list[dict[str, Any]] = [
        {
            "challenge": "overall_benchmark",
            "holdout": "repeated_10fold",
            "task": "regression",
            "n_train": "10-fold",
            "n_test": len(regression),
            "status": "evaluated",
            "exploratory": False,
            "notes": "Five repeats",
            **pca_inventory_fields(all_chemical_pca, None),
        },
        {
            "challenge": "overall_benchmark",
            "holdout": "repeated_10fold",
            "task": "classification",
            "n_train": "10-fold",
            "n_test": len(classification),
            "status": "evaluated",
            "exploratory": False,
            "notes": "Five stratified repeats",
            **pca_inventory_fields(all_chemical_pca, None),
        },
    ]
    for split in splits:
        result = pca_result_for_split(split, context)
        excluded = split.holdout if split.challenge in {"unseen_additive", "unseen_salt"} else None
        for task, train, test in (
            ("regression", split.train_regression, split.test_regression),
            ("classification", split.train_classification, split.test_classification),
        ):
            row: dict[str, Any] = {
                "challenge": split.challenge,
                "holdout": split.holdout,
                "task": task,
                "n_train": len(train),
                "n_test": len(test),
                "status": "not_applicable" if test.empty else "evaluated",
                "reason": "No eligible conditions for this task" if test.empty else "",
                "exploratory": split.exploratory or len(test) < 10,
                "notes": split.notes,
                **pca_inventory_fields(result, excluded),
            }
            if task == "classification":
                for label in CLASS_LABELS:
                    row[f"n_dominant_{label.lower()}"] = int(test["stratification_class"].eq(label).sum())
            rows.append(row)
    return rows


def challenge_predictions(
    split: StaticSplit,
    metadata: dict[str, Any],
    context: PCAContext,
) -> list[pd.DataFrame]:
    verify_split(split)
    split_metadata = metadata_with_pca(metadata, pca_result_for_split(split, context))
    outputs: list[pd.DataFrame] = []
    for task, train, test, models in (
        ("regression", split.train_regression, split.test_regression, REGRESSION_MODELS),
        (
            "classification",
            split.train_classification,
            split.test_classification,
            CLASSIFICATION_MODELS,
        ),
    ):
        if test.empty:
            continue
        train_tables = feature_tables(train, split_metadata)
        test_tables = feature_tables(test, split_metadata)
        exploratory = split.exploratory or len(test) < 10
        for seed in STATIC_SEEDS:
            for model_id in models:
                feature_set = MODEL_DEFINITIONS[model_id]["feature_set"]
                prediction, epoch = fit_model(
                    task,
                    model_id,
                    train_tables[feature_set],
                    train,
                    test_tables[feature_set],
                    seed,
                )
                outputs.append(
                    prediction_frame(
                        test,
                        task,
                        split.challenge,
                        split.holdout,
                        model_id,
                        seed,
                        exploratory,
                        split.notes,
                        prediction,
                        epoch,
                    )
                )
    return outputs


def add_prediction_pools(predictions: pd.DataFrame) -> pd.DataFrame:
    # Rebuild pools from individual holdouts, never pool an existing pool again.
    predictions = predictions.loc[~predictions["holdout"].isin(POOLED_HOLDOUTS)].copy()
    additives = set(predictions.loc[predictions["challenge"].eq("unseen_additive"), "holdout"])
    pools: list[pd.DataFrame] = []
    definitions = (
        ("unseen_additive", {"Span-85", "Urea"}, "rare_additive_pool", True),
        ("unseen_additive", additives, "all_additives_pooled", False),
        ("unseen_salt", set(RARE_SALTS), "unseen_Hofmeister_panel", True),
    )
    for challenge, members, label, exploratory in definitions:
        selected = predictions.loc[
            predictions["challenge"].eq(challenge) & predictions["holdout"].isin(members)
        ].copy()
        if selected.empty:
            continue
        selected["holdout"] = label
        selected["exploratory"] = exploratory
        selected["notes"] = "Predictions pooled after separate molecule-level holdouts"
        pools.append(selected)
    return pd.concat([predictions, *pools], ignore_index=True, sort=False)


def additive_macro_rows(summary: pd.DataFrame, seed_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[
        summary["challenge"].eq("unseen_additive")
        & ~summary["holdout"].isin(POOLED_HOLDOUTS | {"all_additives_macro"})
    ]
    selected_seeds = seed_metrics.loc[
        seed_metrics["challenge"].eq("unseen_additive")
        & ~seed_metrics["holdout"].isin(POOLED_HOLDOUTS | {"all_additives_macro"})
    ]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260825)
    for (task, model_id, metric), frame in selected.groupby(["task", "model_id", "metric"]):
        members = sorted(selected.loc[selected["task"].eq(task), "holdout"].unique())
        central = frame.set_index("holdout")["value"].reindex(members)
        seed_frame = selected_seeds.loc[
            selected_seeds["task"].eq(task)
            & selected_seeds["model_id"].eq(model_id)
            & selected_seeds["metric"].eq(metric)
        ]
        pivot = seed_frame.pivot_table(index="holdout", columns="seed", values="value", aggfunc="mean").reindex(members)
        if central.isna().any() or pivot.isna().any().any():
            raise AssertionError(f"Incomplete additive macro table for {task}/{model_id}/{metric}")
        per_additive = central.to_numpy(dtype=float)
        bootstrap = np.asarray(
            [rng.choice(per_additive, size=len(per_additive), replace=True).mean() for _ in range(BOOTSTRAPS)]
        )
        per_seed = pivot.mean(axis=0)
        rows.append(
            {
                "challenge": "unseen_additive",
                "holdout": "all_additives_macro",
                "task": task,
                "model_id": model_id,
                "metric": metric,
                "value": float(per_additive.mean()),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "seed_mean": float(per_seed.mean()),
                "seed_std": float(per_seed.std(ddof=1)),
                "n_test": int(frame["n_test"].sum()),
                "exploratory": False,
            }
        )
    return pd.DataFrame(rows)


def write_static_summaries(predictions: pd.DataFrame) -> None:
    summary, seed_metrics = metric_helpers.summarize_metrics(predictions)
    summary = pd.concat(
        [summary, additive_macro_rows(summary, seed_metrics)], ignore_index=True, sort=False
    )
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    seed_metrics.to_csv(OUTPUT_DIR / "metrics_by_seed.csv", index=False)

    differences = metric_helpers.paired_model_differences(predictions)
    differences.to_csv(OUTPUT_DIR / "paired_model_differences.csv", index=False)


def run_static() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, regression, classification, metadata, context = load_inputs()
    splits = static_splits(regression, classification)
    all_chemical_metadata = metadata_with_pca(metadata, pca_result(context))
    frames = overall_predictions(regression, classification, all_chemical_metadata)
    for split in splits:
        frames.extend(challenge_predictions(split, metadata, context))
    predictions = add_prediction_pools(pd.concat(frames, ignore_index=True, sort=False))
    verify_holdout_coverage(predictions, regression, classification)
    predictions.to_csv(OUTPUT_DIR / "predictions_long.csv", index=False)
    pd.DataFrame(inventory_rows(splits, context, regression, classification)).to_csv(
        OUTPUT_DIR / "split_inventory.csv", index=False
    )

    write_static_summaries(predictions)


def adaptation_target(
    regression: pd.DataFrame, classification: pd.DataFrame, domain: str, task: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = regression if task == "regression" else classification
    column = "additive_name" if domain in MAIN_ADDITIVES else "salt_name"
    mask = conditions[column].eq(domain)
    original = conditions.loc[~mask].reset_index(drop=True)
    target = conditions.loc[mask].reset_index(drop=True)
    if original[column].eq(domain).any() or len(target) <= 10:
        raise AssertionError(f"Invalid adaptation split for {domain}/{task}")
    metric_helpers.assert_disjoint(original, target, f"adaptation/{domain}/{task}")
    return original, target


def panel_and_pool(target: pd.DataFrame, task: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(target))
    stratify: np.ndarray | None = None
    if task == "classification":
        candidate = target["stratification_class"].to_numpy()
        counts = pd.Series(candidate).value_counts()
        if len(counts) >= 2 and counts.min() >= 2 and 10 >= len(counts):
            stratify = candidate
    else:
        try:
            candidate = pd.qcut(
                target["transition_temperature_c"], q=5, labels=False, duplicates="drop"
            ).to_numpy()
            counts = pd.Series(candidate).value_counts()
            if len(counts) >= 2 and counts.min() >= 2 and len(target) - 10 >= len(counts):
                stratify = candidate
        except ValueError:
            pass
    pool, panel = train_test_split(indices, test_size=10, random_state=seed, stratify=stratify)
    return np.asarray(pool), np.sort(np.asarray(panel))


def adaptation_seed(domain: str, task: str, repeat: int, base: int) -> int:
    return base + ADAPTATION_DOMAINS.index(domain) * 100 + (0 if task == "regression" else 50) + repeat


def run_adaptation_repeat(repeat: int) -> None:
    if repeat not in ADAPTATION_REPEATS:
        raise ValueError("Repeat must be between 0 and 9")
    _, regression, classification, metadata, context = load_inputs()
    adaptation_pca = pca_result(context)
    adaptation_metadata = metadata_with_pca(metadata, adaptation_pca)
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for domain in ADAPTATION_DOMAINS:
        for task, models in (
            ("regression", REGRESSION_MODELS),
            ("classification", CLASSIFICATION_MODELS),
        ):
            original, target = adaptation_target(regression, classification, domain, task)
            partition_seed = adaptation_seed(domain, task, repeat, 4200)
            acquisition_seed = adaptation_seed(domain, task, repeat, 8400)
            pool_index, panel_index = panel_and_pool(target, task, partition_seed)
            order = np.random.default_rng(acquisition_seed).permutation(len(pool_index))
            panel = target.iloc[panel_index].reset_index(drop=True)
            pool = target.iloc[pool_index].reset_index(drop=True)
            ordered_pool = pool.iloc[order].reset_index(drop=True)
            combined = pd.concat([original, ordered_pool, panel], ignore_index=True)
            if combined["condition_id"].duplicated().any():
                raise AssertionError("Adaptation condition overlap")
            tables = feature_tables(combined, adaptation_metadata)
            original_n = len(original)
            pool_n = len(pool)
            panel_positions = np.arange(original_n + pool_n, len(combined))
            inventory.append(
                {
                    "domain": domain,
                    "domain_type": "additive" if domain in MAIN_ADDITIVES else "salt",
                    "task": task,
                    "repeat": repeat,
                    "partition_seed": partition_seed,
                    "acquisition_seed": acquisition_seed,
                    "n_original_train": original_n,
                    "n_target": len(target),
                    "n_evaluation": len(panel),
                    "kmax": pool_n,
                    "evaluation_ids": ";".join(panel["condition_id"]),
                    "acquisition_order_ids": ";".join(ordered_pool["condition_id"]),
                    **pca_inventory_fields(adaptation_pca, None),
                }
            )
            for k in range(pool_n + 1):
                train_positions = np.arange(original_n + k)
                train_conditions = combined.iloc[train_positions]
                if set(train_conditions["condition_id"]) & set(panel["condition_id"]):
                    raise AssertionError("Adaptation evaluation-panel leakage")
                for model_id in models:
                    definition = MODEL_DEFINITIONS[model_id]
                    started = time.perf_counter()
                    prediction, epoch = fit_model(
                        task,
                        model_id,
                        tables[definition["feature_set"]].iloc[train_positions],
                        train_conditions,
                        tables[definition["feature_set"]].iloc[panel_positions],
                        42 + repeat,
                    )
                    elapsed = time.perf_counter() - started
                    if task == "regression":
                        scores = metric_helpers.regression_scores(
                            panel["transition_temperature_c"].to_numpy(dtype=float), prediction
                        )
                    else:
                        scores = metric_helpers.classifier_scores(
                            panel[list(TARGET_COLUMNS)].to_numpy(dtype=float), prediction
                        )
                    common = {
                        "domain": domain,
                        "domain_type": "additive" if domain in MAIN_ADDITIVES else "salt",
                        "task": task,
                        "repeat": repeat,
                        "k": k,
                        "kmax": pool_n,
                        "n_original_train": original_n,
                        "n_train": len(train_positions),
                        "n_evaluation": len(panel),
                        "model_id": model_id,
                        "feature_set": definition["feature_set"],
                        "model_family": definition["family"],
                        "selected_epoch": epoch,
                        "fit_seconds": elapsed,
                    }
                    rows.extend(
                        {**common, "metric": metric, "value": value}
                        for metric, value in scores.items()
                        if metric != "n_hard"
                    )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"adaptation_metrics_repeat_{repeat}.csv", index=False)
    pd.DataFrame(inventory).to_csv(
        OUTPUT_DIR / f"adaptation_inventory_repeat_{repeat}.csv", index=False
    )


def bootstrap_adaptation(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260825)
    groups = [
        "domain",
        "domain_type",
        "task",
        "k",
        "kmax",
        "model_id",
        "feature_set",
        "model_family",
        "metric",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in metrics.groupby(groups, sort=False):
        values = frame["value"].dropna().to_numpy(dtype=float)
        if len(values):
            samples = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
            mean = float(values.mean())
            low, high = map(float, np.quantile(samples, [0.025, 0.975]))
        else:
            mean = low = high = np.nan
        rows.append(
            {
                **dict(zip(groups, keys)),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n_repeats": len(values),
            }
        )
    return pd.DataFrame(rows)


def primary_adaptation(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.loc[
        ((metrics["task"] == "regression") & (metrics["metric"] == "mae"))
        | ((metrics["task"] == "classification") & (metrics["metric"] == "cross_entropy"))
    ].copy()


def k90_rows(curves: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in curves.groupby(group_columns, sort=False):
        ordered = frame.sort_values("k")
        x = ordered["k"].to_numpy(dtype=float)
        y = ordered["mean"].to_numpy(dtype=float)
        fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(x, y)
        improvement = float(fitted[0] - fitted[-1])
        threshold = float(fitted[-1] + 0.10 * improvement)
        candidates = x[fitted <= threshold + 1e-12] if improvement > 0 else np.asarray([])
        rows.append(
            {
                **dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,))),
                "kmax": int(x.max()),
                "fitted_baseline": float(fitted[0]),
                "fitted_endpoint": float(fitted[-1]),
                "fitted_improvement": improvement,
                "k90": int(candidates.min()) if len(candidates) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_adaptation_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer a complete repeat run, or reuse the verified consolidated results."""
    metric_paths = [OUTPUT_DIR / f"adaptation_metrics_repeat_{repeat}.csv" for repeat in ADAPTATION_REPEATS]
    inventory_paths = [
        OUTPUT_DIR / f"adaptation_inventory_repeat_{repeat}.csv" for repeat in ADAPTATION_REPEATS
    ]
    repeat_paths = (*metric_paths, *inventory_paths)
    if any(path.exists() for path in repeat_paths):
        missing = [str(path) for path in repeat_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete adaptation run; missing repeat outputs: {missing}")
        return (
            pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True),
            pd.concat([pd.read_csv(path) for path in inventory_paths], ignore_index=True),
        )

    combined_paths = (
        OUTPUT_DIR / "adaptation_metrics_long.csv",
        OUTPUT_DIR / "adaptation_inventory.csv",
    )
    manifest_path = OUTPUT_DIR / "evaluation_manifest.json"
    missing = [str(path) for path in (*combined_paths, manifest_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing adaptation outputs: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("master_sha256") != file_sha256(MASTER_PATH):
        raise ValueError("Saved adaptation results do not match the current master data; rerun adaptation.")
    saved_hashes = manifest.get("output_sha256", {})
    frames = []
    for path in combined_paths:
        if saved_hashes.get(path.name) != file_sha256(path):
            raise ValueError(f"Cannot verify saved adaptation output {path.name}; rerun adaptation.")
        frame = pd.read_csv(path)
        if "repeat" not in frame or set(frame["repeat"].unique()) != set(ADAPTATION_REPEATS):
            raise ValueError(f"Saved adaptation output {path.name} does not contain every expected repeat.")
        frames.append(frame)
    return frames[0], frames[1]


def compile_adaptation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, inventory = load_adaptation_outputs()
    metrics.to_csv(OUTPUT_DIR / "adaptation_metrics_long.csv", index=False)
    inventory.to_csv(OUTPUT_DIR / "adaptation_inventory.csv", index=False)
    summary = bootstrap_adaptation(metrics)
    summary.to_csv(OUTPUT_DIR / "adaptation_curves.csv", index=False)

    primary_summary = primary_adaptation(summary)
    model_k90 = k90_rows(
        primary_summary,
        ["domain", "domain_type", "task", "model_id", "feature_set", "model_family", "metric"],
    )
    model_k90.to_csv(OUTPUT_DIR / "adaptation_k90_by_model.csv", index=False)

    per_repeat_average = (
        primary_adaptation(metrics)
        .groupby(["domain", "domain_type", "task", "repeat", "k", "kmax", "metric"], as_index=False)[
            "value"
        ]
        .mean()
    )
    model_average = bootstrap_adaptation(
        per_repeat_average.assign(
            model_id="equal_model_average", feature_set="both", model_family="XGBoost_and_MLP"
        )
    )
    model_average.to_csv(OUTPUT_DIR / "adaptation_model_average_curves.csv", index=False)
    average_k90 = k90_rows(
        model_average,
        ["domain", "domain_type", "task", "model_id", "feature_set", "model_family", "metric"],
    )
    average_k90.to_csv(OUTPUT_DIR / "adaptation_k90_model_average.csv", index=False)
    domain_average = average_k90.groupby(["domain_type", "task"], as_index=False).agg(
        mean_k90=("k90", "mean"), min_k90=("k90", "min"), max_k90=("k90", "max")
    )
    both_tasks = average_k90.pivot_table(
        index=["domain", "domain_type"], columns="task", values="k90", aggfunc="first"
    ).reset_index()
    both_tasks["k90"] = both_tasks[["regression", "classification"]].max(axis=1)
    both_summary = both_tasks.groupby("domain_type", as_index=False).agg(
        mean_k90=("k90", "mean"), min_k90=("k90", "min"), max_k90=("k90", "max")
    )
    both_summary["task"] = "both_tasks"
    domain_average = pd.concat([domain_average, both_summary], ignore_index=True, sort=False)
    domain_average.to_csv(OUTPUT_DIR / "adaptation_k90_domain_average.csv", index=False)
    return summary, model_average, average_k90


def model_label(model_id: str) -> str:
    representation = "Descriptor" if model_id.startswith("descriptor") else "Two-slot"
    family = MODEL_DEFINITIONS[model_id]["family"]
    return f"{representation} {family}"


def display_label(value: str) -> str:
    labels = {
        "Na2SO4": r"Na$_2$SO$_4$",
        "rare_additive_pool": "Rare-additive pool",
        "unseen_Hofmeister_panel": "Hofmeister panel",
        "20kDa": "20 kDa",
        "40kDa": "40 kDa",
        "86kDa": "86 kDa",
        "250kDa": "250 kDa",
        "500kDa": "500 kDa",
    }
    return labels.get(value, value)


def primary_static(summary: pd.DataFrame, challenge: str, task: str) -> pd.DataFrame:
    metric = "mae" if task == "regression" else "cross_entropy"
    return summary.loc[
        summary["challenge"].eq(challenge)
        & summary["task"].eq(task)
        & summary["metric"].eq(metric)
    ].copy()


PLOT_COLORS = {
    ("descriptor24", "XGBoost"): "#1f6bc1",
    ("two_slot38", "XGBoost"): "#078f7f",
    ("descriptor24", "MLP"): "#f26f00",
    ("two_slot38", "MLP"): "#7c22a6",
}


def plot_models(task: str) -> tuple[str, ...]:
    suffix = "regressor" if task == "regression" else "classifier"
    return (
        f"descriptor24_buffer_{suffix}_xgboost",
        f"two_slot_pca38_buffer_{suffix}_xgboost",
        f"descriptor24_buffer_{suffix}_mlp",
        f"two_slot_pca38_buffer_{suffix}_mlp",
    )


def model_color(model_id: str) -> str:
    definition = MODEL_DEFINITIONS[model_id]
    return PLOT_COLORS[(definition["feature_set"], definition["family"])]


def compact_model_label(model_id: str) -> str:
    definition = MODEL_DEFINITIONS[model_id]
    representation = "Desc." if definition["feature_set"] == "descriptor24" else "Two-slot"
    family = "XGB" if definition["family"] == "XGBoost" else "MLP"
    return f"{representation}\n{family}"


def exploratory_label(holdout: str) -> str:
    label = display_label(holdout)
    exploratory = {"Pluronic F127", "rare_additive_pool", "NH4Cl", "unseen_Hofmeister_panel", "500kDa"}
    return f"{label}†" if holdout in exploratory else label


def plot_overall_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    models: tuple[str, ...],
    ylabel: str,
    title: str,
) -> None:
    frame = table.set_index("model_id")
    values = np.asarray([frame.loc[model, "value"] for model in models])
    lows = np.asarray([frame.loc[model, "ci95_low"] for model in models])
    highs = np.asarray([frame.loc[model, "ci95_high"] for model in models])
    errors = np.vstack([values - lows, highs - values])
    positions = np.arange(len(models))
    ax.bar(
        positions,
        values,
        width=0.72,
        color=[model_color(model) for model in models],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.errorbar(positions, values, yerr=errors, fmt="none", color="#222222", capsize=4, linewidth=1.1)
    ax.set_xticks(positions, [compact_model_label(model) for model in models])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(highs) * 1.18)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_forest_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    task: str,
    panel_label: str,
) -> None:
    models = plot_models(task)
    metric = "mae" if task == "regression" else "cross_entropy"
    groups = [
        ("Additives", "unseen_additive", ["CHAPS", "CTAB", "SDS"]),
        ("Salts", "unseen_salt", ["NaCl", "Na2SO4"]),
        (
            "Molecular weight",
            "unseen_molecular_weight",
            ["20kDa", "40kDa", "86kDa", "250kDa", "500kDa"],
        ),
    ]
    rows: list[tuple[str, str, str]] = []
    for group, challenge, holdouts in groups:
        rows.extend((group, challenge, holdout) for holdout in holdouts)
    y_positions = np.arange(len(rows) - 1, -1, -1, dtype=float)
    offsets = np.linspace(0.24, -0.24, len(models))
    x_limit = 22.5 if task == "regression" else 3.65
    table = summary.loc[summary["task"].eq(task) & summary["metric"].eq(metric)].copy()
    for model_id, offset in zip(models, offsets):
        for y, (_, challenge, holdout) in zip(y_positions, rows):
            selected = table.loc[
                table["challenge"].eq(challenge)
                & table["holdout"].eq(holdout)
                & table["model_id"].eq(model_id)
            ]
            if selected.empty:
                continue
            row = selected.iloc[0]
            value = float(row["value"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            color = model_color(model_id)
            if value > x_limit:
                marker_x = x_limit - 0.35
                ax.plot(marker_x, y + offset, marker=">", color=color, markersize=6, clip_on=False)
                ax.text(
                    marker_x - 0.28,
                    y + offset,
                    f"{value:.1f}",
                    color=color,
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="semibold",
                )
                continue
            left = max(0.0, value - low)
            right = min(high, x_limit) - value
            ax.errorbar(
                value,
                y + offset,
                xerr=np.asarray([[left], [max(0.0, right)]]),
                fmt="o",
                color=color,
                markersize=4.5,
                capsize=2.5,
                linewidth=1.0,
                label=model_label(model_id),
            )
    labels = []
    for _, _, holdout in rows:
        label = display_label(holdout)
        if holdout == "500kDa":
            label += "†"
        labels.append(label)
    ax.set_yticks(y_positions)
    if task == "regression":
        ax.set_yticklabels(labels)
    else:
        ax.tick_params(axis="y", labelleft=False)
    for group, _, holdouts in groups:
        first_index = next(index for index, row in enumerate(rows) if row[0] == group)
        y = y_positions[first_index] + 0.55
        ax.text(0.02, y, group, transform=ax.get_yaxis_transform(), color="#666666", fontsize=9)
    boundaries = [len(groups[0][2]), len(groups[0][2]) + len(groups[1][2])]
    for boundary in boundaries:
        y = (y_positions[boundary - 1] + y_positions[boundary]) / 2
        ax.axhline(y, color="#b8b8b8", linewidth=0.8)
    ax.set_xlim(0, x_limit)
    ax.set_ylim(-0.7, len(rows) - 0.05)
    ax.set_xlabel("MAE (°C)" if task == "regression" else "Cross-entropy")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.12, 1.035, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def plot_heatmap_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    holdouts: list[str],
    models: tuple[str, ...],
    title: str,
    colorbar_label: str,
) -> None:
    frame = table.pivot(index="holdout", columns="model_id", values="value")
    values = np.asarray(
        [[frame.loc[holdout, model] if holdout in frame.index and model in frame.columns else np.nan for model in models]
         for holdout in holdouts],
        dtype=float,
    )
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(finite, 90)) if len(finite) else 1.0
    vmax = max(vmax, 1e-9)
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            color = "white" if value >= 0.62 * vmax else "#151515"
            text = f"{value:.2f}" if colorbar_label.startswith("MAE") else f"{value:.3f}"
            ax.text(column, row, text, ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(np.arange(len(models)), [compact_model_label(model) for model in models])
    ax.set_yticks(np.arange(len(holdouts)), [exploratory_label(holdout) for holdout in holdouts])
    ax.set_title(title)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.043, pad=0.025, extend="max")
    colorbar.set_label(colorbar_label)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    fig.tight_layout(rect=tight_rect)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_composition(classification: pd.DataFrame) -> None:
    if len(classification) != 403:
        raise AssertionError(
            f"Expected 403 pooled classification conditions, found {len(classification)}"
        )

    colors = {
        "polymer": "#2a6fbb",
        "additive": "#078f7f",
        "salt": "#e66b00",
    }
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.4, 8.0),
        gridspec_kw={"height_ratios": (1.25, 1.0)},
    )

    molecular_weight_labels = ["20", "40", "86", "250", "500", "No\nDexMA"]
    molecular_weight_counts = [
        int(classification["polymer_mw_kda"].eq(value).sum())
        for value in (20.0, 40.0, 86.0, 250.0, 500.0)
    ]
    molecular_weight_counts.append(int(classification["polymer_mw_kda"].isna().sum()))
    ax = axes[0, 0]
    bars = ax.bar(
        np.arange(len(molecular_weight_labels)),
        molecular_weight_counts,
        color=colors["polymer"],
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xticks(np.arange(len(molecular_weight_labels)), molecular_weight_labels)
    ax.set_xlabel("DexMA molecular weight (kDa)")
    ax.set_ylabel("Formulation conditions")
    ax.set_title("Molecular-weight distribution")
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_ylim(0, max(molecular_weight_counts) * 1.15)

    def categorical_panel(
        panel: plt.Axes,
        series: pd.Series,
        title: str,
        color: str,
        label_map: dict[str, str],
    ) -> None:
        counts = series.value_counts()
        labels = [label_map.get(str(name), str(name)) for name in counts.index]
        positions = np.arange(len(counts))
        bars = panel.barh(
            positions,
            counts.to_numpy(dtype=int),
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        panel.set_yticks(positions, labels)
        panel.invert_yaxis()
        panel.set_xlabel("Formulation conditions")
        panel.set_title(title)
        panel.bar_label(bars, padding=2, fontsize=8)
        panel.set_xlim(0, float(counts.max()) * 1.18)

    categorical_panel(
        axes[0, 1],
        classification["additive_name"],
        "Additive distribution",
        colors["additive"],
        {"No additive": "None"},
    )
    categorical_panel(
        axes[0, 2],
        classification["salt_name"],
        "Salt distribution",
        colors["salt"],
        {
            "No salt": "None",
            "Na2SO4": r"Na$_2$SO$_4$",
            "Na2HPO4": r"Na$_2$HPO$_4$",
            "Na2S2O3": r"Na$_2$S$_2$O$_3$",
            "CaCl2": r"CaCl$_2$",
            "MgCl2": r"MgCl$_2$",
            "NH4Cl": r"NH$_4$Cl",
            "NaNO3": r"NaNO$_3$",
        },
    )

    def concentration_panel(
        panel: plt.Axes,
        series: pd.Series,
        title: str,
        xlabel: str,
        color: str,
    ) -> None:
        values = series.to_numpy(dtype=float)
        positive = values[np.isfinite(values) & (values > 0)]
        zero_or_absent = int(len(values) - len(positive))
        edges = np.geomspace(
            float(positive.min()),
            float(positive.max()) * (1.0 + 1e-9),
            15,
        )
        panel.hist(
            positive,
            bins=edges,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        panel.set_xscale("log")
        panel.set_xlim(edges[0], edges[-1])
        panel.set_xlabel(xlabel)
        panel.set_ylabel("Formulation conditions")
        panel.set_title(title)
        panel.text(
            0.02,
            0.95,
            f"Positive: {len(positive)}\nZero/absent: {zero_or_absent}",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
        )

    concentration_panel(
        axes[1, 0],
        classification["polymer_concentration_mg_ml"],
        "DexMA concentration range",
        r"DexMA concentration (mg mL$^{-1}$)",
        colors["polymer"],
    )
    concentration_panel(
        axes[1, 1],
        classification["additive_concentration_mM"],
        "Additive concentration range",
        "Additive concentration (mM)",
        colors["additive"],
    )
    concentration_panel(
        axes[1, 2],
        classification["salt_concentration_mM"],
        "Salt concentration range",
        "Salt concentration (mM)",
        colors["salt"],
    )

    for panel_label, panel in zip("abcdef", axes.flat):
        panel.text(
            -0.13,
            1.06,
            panel_label,
            transform=panel.transAxes,
            fontsize=12,
            fontweight="bold",
        )
        panel.grid(axis="y", alpha=0.18)
        panel.set_axisbelow(True)
        panel.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "dataset_composition")


def plot_static(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for ax, task, ylabel, panel_label in (
        (axes[0], "regression", "MAE (°C)", "a"),
        (axes[1], "classification", "Cross-entropy", "b"),
    ):
        table = primary_static(summary, "overall_benchmark", task)
        plot_overall_panel(ax, table, plot_models(task), ylabel, "")
        ax.text(-0.12, 1.04, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    save_figure(fig, "overall_benchmark")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.2), sharey=True)
    plot_forest_panel(axes[0], summary, "regression", "a")
    plot_forest_panel(axes[1], summary, "classification", "b")
    handles = [
        plt.Line2D([0], [0], marker="o", color=model_color(model), linewidth=1.0, markersize=5,
                   label=model_label(model))
        for model in plot_models("regression")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=9,
        borderaxespad=0,
    )
    save_figure(fig, "unseen_chemistry", tight_rect=(0.0, 0.085, 1.0, 1.0))


def plot_adaptation(model_average: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    domain_colors = {
        "CHAPS": "#1f6bc1",
        "CTAB": "#078f7f",
        "SDS": "#f26f00",
        "NaCl": "#7c22a6",
        "Na2SO4": "#c62828",
    }
    panels = [
        (axes[0, 0], "additive", "regression", "a", "Unseen additives"),
        (axes[0, 1], "additive", "classification", "b", "Unseen additives"),
        (axes[1, 0], "salt", "regression", "c", "Unseen salts"),
        (axes[1, 1], "salt", "classification", "d", "Unseen salts"),
    ]
    for ax, domain_type, task, panel_label, title in panels:
        selected = model_average.loc[
            model_average["domain_type"].eq(domain_type) & model_average["task"].eq(task)
        ]
        for domain, frame in selected.groupby("domain", sort=False):
            frame = frame.sort_values("k")
            color = domain_colors[domain]
            ax.plot(frame["k"], frame["mean"], label=display_label(domain), color=color, linewidth=2.0)
            ax.fill_between(
                frame["k"], frame["ci95_low"], frame["ci95_high"], color=color, alpha=0.14, linewidth=0
            )
        ax.set_xlabel("Target-domain conditions added (k)")
        ax.set_ylabel("MAE (°C)" if task == "regression" else "Cross-entropy")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    save_figure(fig, "adaptation_curves")


def write_notes(summary: pd.DataFrame, average_k90: pd.DataFrame) -> None:
    lines = [
        "# Fresh LCST evaluation notes",
        "",
        "All values below use 272 regression and 403 classification conditions. No source workbook was held out.",
        "",
        "## Primary static metrics",
        "",
    ]
    selections = (
        ("overall_benchmark", ["repeated_10fold"]),
        ("unseen_additive", ["CHAPS", "CTAB", "SDS", "Pluronic F127", "rare_additive_pool"]),
        ("unseen_salt", ["NaCl", "Na2SO4", "NH4Cl", "unseen_Hofmeister_panel"]),
        ("unseen_molecular_weight", ["20kDa", "40kDa", "86kDa", "250kDa", "500kDa"]),
    )
    for challenge, holdouts in selections:
        lines.append(f"### {challenge.replace('_', ' ').title()}")
        lines.append("")
        for task in ("regression", "classification"):
            table = primary_static(summary, challenge, task)
            for holdout in holdouts:
                for _, row in table.loc[table["holdout"].eq(holdout)].sort_values("value").iterrows():
                    lines.append(
                        f"- {task}; {holdout}; {model_label(row['model_id'])}: "
                        f"{row['value']:.4f} (95% CI {row['ci95_low']:.4f}–{row['ci95_high']:.4f}; n={int(row['n_test'])})"
                    )
        lines.append("")
    lines.extend(["## Random adaptation k90", ""])
    for _, row in average_k90.sort_values(["domain_type", "task", "domain"]).iterrows():
        lines.append(
            f"- {row['domain']} {row['task']}: k90={int(row['k90']) if pd.notna(row['k90']) else 'not reached'} "
            f"of kmax={int(row['kmax'])}."
        )
    lines.extend(
        [
            "",
            "The 500 kDa tests and pooled rare-molecule panels are exploratory.",
            "The 20/40 kDa molecular-weight tests also change functionalization and formulation composition.",
            "Withholding 86 kDa leaves a much smaller and less chemically representative training set.",
        ]
    )
    (OUTPUT_DIR / "results_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(
    predictions: pd.DataFrame,
    static_inventory: pd.DataFrame,
    adaptation_inventory: pd.DataFrame,
) -> dict[str, Any]:
    regression = predictions.loc[predictions["task"].eq("regression")]
    classification = predictions.loc[predictions["task"].eq("classification")]
    probability_columns = [f"prediction_{label.lower()}" for label in CLASS_LABELS]
    if not np.isfinite(regression["prediction_c"]).all():
        raise AssertionError("Static regression output contains non-finite values")
    probabilities = classification[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise AssertionError("Static classification probabilities are invalid")
    if not adaptation_inventory["n_evaluation"].eq(10).all():
        raise AssertionError("An adaptation evaluation panel does not contain 10 conditions")
    chemical_holdout = static_inventory["challenge"].isin({"unseen_additive", "unseen_salt"})
    if not static_inventory.loc[chemical_holdout, "pca_fit_count"].eq(17).all():
        raise AssertionError("An unseen-chemical evaluation did not fit PCA on exactly 17 chemicals")
    if not (
        static_inventory.loc[chemical_holdout, "pca_excluded_chemical"].astype(str)
        == static_inventory.loc[chemical_holdout, "holdout"].astype(str)
    ).all():
        raise AssertionError("An unseen-chemical evaluation excluded the wrong chemical from PCA")
    nonchemical = ~chemical_holdout
    if not static_inventory.loc[nonchemical, "pca_fit_count"].eq(18).all():
        raise AssertionError("Within-dataset or molecular-weight evaluation did not use all 18 chemicals")
    if not adaptation_inventory["pca_fit_count"].eq(18).all():
        raise AssertionError("An adaptation evaluation did not fit PCA on all 18 chemicals")
    hashes_per_repeat = adaptation_inventory.groupby("repeat")["pca_components_sha256"].nunique()
    if not hashes_per_repeat.eq(1).all():
        raise AssertionError("The PCA basis changed within an adaptation repeat")
    expected_names = static_inventory.loc[
        static_inventory["challenge"].eq("overall_benchmark"), "pca_fit_chemical_names"
    ].iloc[0]
    if not adaptation_inventory["pca_fit_chemical_names"].eq(expected_names).all():
        raise AssertionError("Adaptation repeats did not use the same 18-chemical fitting library")
    return {
        "dataset_counts_passed": True,
        "no_formulation_leakage": True,
        "held_out_components_absent": True,
        "feature_contracts": {"descriptor": 24, "two_slot": 38},
        "finite_predictions": True,
        "probabilities_sum_to_one": True,
        "fixed_adaptation_panel_size": 10,
        "random_adaptation_repeats": 10,
        "pca_within_dataset_fit_count": 18,
        "pca_unseen_chemical_fit_count": 17,
        "pca_molecular_weight_fit_count": 18,
        "pca_adaptation_fit_count": 18,
        "pca_adaptation_basis_fixed_within_each_repeat": True,
        "pca_adaptation_fitting_library_fixed_across_repeats": True,
    }


def compile_outputs(previous_results_dir: Path | None = None) -> None:
    predictions_path = OUTPUT_DIR / "predictions_long.csv"
    metrics_path = OUTPUT_DIR / "metrics_summary.csv"
    if not predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run the static phase before compiling")
    summary = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    _, regression_conditions, classification_conditions, _, _ = load_inputs()
    holdout_coverage = verify_holdout_coverage(
        predictions, regression_conditions, classification_conditions,
    )
    adaptation_summary, model_average, average_k90 = compile_adaptation()
    static_inventory = pd.read_csv(OUTPUT_DIR / "split_inventory.csv", keep_default_na=False)
    adaptation_inventory = pd.read_csv(OUTPUT_DIR / "adaptation_inventory.csv")
    checks = verify_outputs(predictions, static_inventory, adaptation_inventory)
    checks["complete_individual_holdout_coverage"] = True
    previous_static_path = (
        previous_results_dir / "metrics_summary.csv"
        if previous_results_dir is not None
        else None
    )
    if previous_static_path is not None and previous_static_path.exists():
        previous_static = pd.read_csv(previous_static_path)
        static_keys = ["challenge", "holdout", "task", "model_id", "metric"]
        static_comparison = previous_static.merge(
            summary,
            on=static_keys,
            how="outer",
            suffixes=("_previous", "_corrected"),
            indicator=True,
        )
        static_comparison["delta_corrected_minus_previous"] = (
            static_comparison["value_corrected"] - static_comparison["value_previous"]
        )
        static_comparison.to_csv(
            OUTPUT_DIR / "old_vs_corrected_static_metrics.csv", index=False
        )
    previous_adaptation_path = (
        previous_results_dir / "adaptation_curves.csv"
        if previous_results_dir is not None
        else None
    )
    if previous_adaptation_path is not None and previous_adaptation_path.exists():
        previous_adaptation = pd.read_csv(previous_adaptation_path)
        adaptation_keys = [
            "domain",
            "domain_type",
            "task",
            "k",
            "kmax",
            "model_id",
            "feature_set",
            "model_family",
            "metric",
        ]
        adaptation_comparison = previous_adaptation.merge(
            adaptation_summary,
            on=adaptation_keys,
            how="outer",
            suffixes=("_previous", "_corrected"),
            indicator=True,
        )
        adaptation_comparison["delta_corrected_minus_previous"] = (
            adaptation_comparison["mean_corrected"] - adaptation_comparison["mean_previous"]
        )
        adaptation_comparison.to_csv(
            OUTPUT_DIR / "old_vs_corrected_adaptation_curves.csv", index=False
        )
    plot_static(summary)
    plot_adaptation(model_average)
    write_notes(summary, average_k90)
    master = pd.read_csv(MASTER_PATH, keep_default_na=False, na_values=[""])
    eligible = master.loc[~master["source_sheet"].eq(HEATING_RATE_SHEET)]
    regression = build_regression_conditions(master)
    classification = build_classifier_conditions(master)
    plot_dataset_composition(classification)
    dominant_classes = classification["stratification_class"].value_counts().to_dict()
    mixed_conditions = int(
        (classification[list(TARGET_COLUMNS)].gt(0).sum(axis=1) > 1).sum()
    )
    manifest = {
        "feature_encoding": FEATURE_ENCODING_VERSION,
        "concentration_references": CONCENTRATION_REFERENCES,
        "master_path": str(MASTER_PATH),
        "master_sha256": file_sha256(MASTER_PATH),
        "master_measurements": len(master),
        "eligible_measurements": len(eligible),
        "regression_conditions": len(regression),
        "classification_conditions": len(classification),
        "classification_dominant_classes": dominant_classes,
        "mixed_classification_conditions": mixed_conditions,
        "no_dexma_controls": int(classification["polymer_name"].ne("Dex-MA").sum()),
        "static_seeds": list(STATIC_SEEDS),
        "evaluated_holdouts": holdout_coverage,
        "adaptation_repeats": list(ADAPTATION_REPEATS),
        "adaptation_panel_size": 10,
        "adaptation_k_rule": "Every integer from 0 through N_target - 10",
        "bootstrap_resamples": BOOTSTRAPS,
        "pca_policy": {
            "within_dataset_cross_validation": "fit on all 18 configured chemicals",
            "unseen_additive_or_salt": "fit on the 17 non-held-out chemicals and transform the holdout",
            "molecular_weight_holdout": "fit on all 18 configured chemicals",
            "random_target_domain_adaptation": "fit on all 18 configured chemicals and keep the basis fixed across k",
            "molformer_model_id": MOLFORMER_MODEL_ID,
            "molformer_snapshot_revision": MOLFORMER_SNAPSHOT_REVISION,
            "structure_registry_sha256": load_inputs()[4].structure_sha256,
            "embedding_dimension": len(next(iter(load_inputs()[4].embeddings.values()))),
            "legacy_17_chemical_reproduction_max_abs_difference": (
                load_inputs()[4].legacy_reproduction_max_abs_difference
            ),
        },
        "model_definitions": MODEL_DEFINITIONS,
        "mlp_regression_protocol": asdict(MLP_REGRESSION_PROTOCOL),
        "mlp_classification_protocol": asdict(MLP_CLASSIFICATION_PROTOCOL),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "checks": checks,
        "output_sha256": {
            name: file_sha256(OUTPUT_DIR / name)
            for name in ("adaptation_metrics_long.csv", "adaptation_inventory.csv")
        },
        "repository_status": git_status(),
    }
    (OUTPUT_DIR / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def run_report_evaluation(
    phase: str = "all",
    repeat: int | None = None,
    previous_results_dir: Path | None = None,
) -> None:
    phases = {"static", "static-summary", "adaptation", "compile", "all"}
    if phase not in phases:
        raise ValueError(f"Unsupported report-evaluation phase: {phase!r}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if phase in {"static", "all"}:
        run_static()
    elif phase == "static-summary":
        write_static_summaries(pd.read_csv(OUTPUT_DIR / "predictions_long.csv"))
    if phase == "adaptation":
        if repeat is None:
            raise ValueError("A repeat is required for the adaptation phase")
        if repeat not in ADAPTATION_REPEATS:
            raise ValueError(f"Adaptation repeat must be one of {ADAPTATION_REPEATS}")
        run_adaptation_repeat(repeat)
    elif phase == "all":
        for repeat in ADAPTATION_REPEATS:
            run_adaptation_repeat(repeat)
    if phase in {"compile", "all"}:
        compile_outputs(previous_results_dir)
