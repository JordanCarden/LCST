#!/usr/bin/env python3
"""Score canonical LCST conditions with one or more maintained models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lcst_pipeline.features import (
    FEATURE_ENCODING_VERSION, add_buffer_feature, build_descriptor23,
    build_two_slot37, ingredient_present,
)
from lcst_pipeline.embeddings import generate_molformer_embeddings, validate_structure_registry
from lcst_pipeline.modeling import CLASS_LABELS, MODEL_DEFINITIONS
from lcst_pipeline.schema import PREDICTION_INPUT_COLUMNS, load_config


def _read_conditions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Input must be a .csv, .xlsx, or .xlsm file")
    missing = [column for column in PREDICTION_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing canonical columns: {', '.join(missing)}")
    conditions = frame[PREDICTION_INPUT_COLUMNS].copy()
    for column in ("polymer_name", "buffer", "additive_name", "salt_name"):
        conditions[column] = conditions[column].fillna("").astype(str).str.strip()
    for column in ("additive_concentration_mM", "salt_concentration_mM"):
        conditions[column] = pd.to_numeric(conditions[column], errors="raise")
    return conditions


def _active_chemical_names(conditions: pd.DataFrame) -> tuple[set[str], set[str]]:
    return tuple(
        {
            str(name).strip()
            for name, amount in zip(conditions[f"{group}_name"], conditions[f"{group}_concentration_mM"])
            if ingredient_present(name, amount, group)
        }
        for group in ("additive", "salt")
    )


def _prediction_metadata(conditions: pd.DataFrame, artifact: dict) -> dict:
    metadata = dict(artifact["chemical_metadata"])
    additive_names, salt_names = _active_chemical_names(conditions)
    current = load_config()["chemical_metadata"]

    if artifact["feature_set"] != "two_slot38":
        # Two-slot models use molecular embeddings, not descriptor-table entries.
        additive_descriptors = dict(metadata["additive_descriptors"])
        salt_descriptors = dict(metadata["salt_descriptors"])
        for name in sorted(additive_names - set(additive_descriptors)):
            if name not in current["additive_descriptors"]:
                raise ValueError(f"No descriptor metadata is registered for new additive {name!r}")
            additive_descriptors[name] = current["additive_descriptors"][name]
        for name in sorted(salt_names - set(salt_descriptors)):
            if name not in current["salt_descriptors"]:
                raise ValueError(f"No descriptor metadata is registered for new salt {name!r}")
            salt_descriptors[name] = current["salt_descriptors"][name]
        metadata["additive_descriptors"] = additive_descriptors
        metadata["salt_descriptors"] = salt_descriptors
        return metadata

    scores = dict(metadata["molformer_pca16"])
    requested = additive_names | salt_names
    unknown = tuple(sorted(requested - set(scores)))
    if not unknown:
        return metadata
    pca_state = artifact.get("molformer_pca")
    if not pca_state or "transformer" not in pca_state:
        raise ValueError(
            "This older two-slot model does not contain its fitted PCA transformation and "
            f"cannot score new chemicals: {list(unknown)}. Rebuild the maintained models first."
        )
    structures = validate_structure_registry(current)
    missing_structures = sorted(set(unknown) - set(structures))
    if missing_structures:
        raise ValueError(f"No molecular input is registered for new chemicals: {missing_structures}")
    embeddings = generate_molformer_embeddings(
        structures,
        chemical_names=unknown,
        device="cpu",
    )
    matrix = np.vstack([embeddings[name] for name in unknown])
    transformed = np.asarray(pca_state["transformer"].transform(matrix), dtype=float)
    if transformed.shape != (len(unknown), 16) or not np.isfinite(transformed).all():
        raise ValueError("The stored PCA transformation produced invalid new-chemical features")
    scores.update({name: transformed[index].tolist() for index, name in enumerate(unknown)})
    metadata["molformer_pca16"] = scores
    return metadata


def _features(conditions: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    if artifact.get("feature_encoding") != FEATURE_ENCODING_VERSION:
        raise ValueError(
            "This model uses the previous feature encoding. Retrain the models "
            "before predicting with the current concentration encoding and chemical lookup."
        )
    feature_set = artifact["feature_set"]
    if feature_set == "descriptor24" and "additive_hlb_davies" in artifact.get("feature_columns", []):
        raise ValueError(
            "This descriptor model uses the previous Davies-only HLB feature. "
            "Retrain the descriptor models with the sourced additive_hlb lookup. "
            "Two-slot models are unaffected by this HLB update."
        )
    metadata = _prediction_metadata(conditions, artifact)
    if feature_set == "descriptor24":
        features = build_descriptor23(conditions, metadata)
    elif feature_set == "two_slot38":
        features = build_two_slot37(conditions, metadata)
    else:
        raise ValueError(f"Unsupported feature set in artifact: {feature_set!r}")
    features = add_buffer_feature(features, conditions)
    expected = artifact["feature_columns"]
    if list(features.columns) != expected:
        raise ValueError(f"Feature contract mismatch for {artifact['model_id']}")
    return features


def _write_predictions(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".xlsx", ".xlsm"}:
        frame.to_excel(path, index=False)
    else:
        raise ValueError("Output must be a .csv, .xlsx, or .xlsm file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical CSV or XLSX conditions")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV or XLSX")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_DEFINITIONS),
        default=list(MODEL_DEFINITIONS),
        help="Models to score; defaults to all eight",
    )
    args = parser.parse_args()

    conditions = _read_conditions(args.input)
    output = conditions.copy()
    for model_id in args.models:
        artifact_path = REPO_ROOT / "models" / f"{model_id}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {artifact_path}")
        artifact = joblib.load(artifact_path)
        if artifact.get("model_id") != model_id:
            raise ValueError(f"Artifact identity mismatch in {artifact_path}")
        features = _features(conditions, artifact)
        model = artifact["model"]
        if artifact["task"] == "regression":
            predictions = np.asarray(model.predict(features), dtype=float)
            output[f"{model_id}_lcst_c"] = predictions
        else:
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            if probabilities.shape != (len(conditions), len(CLASS_LABELS)):
                raise ValueError(f"Unexpected classifier output shape from {model_id}")
            for index, label in enumerate(CLASS_LABELS):
                output[f"{model_id}_prob_{label.lower()}"] = probabilities[:, index]
            output[f"{model_id}_predicted_class"] = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]

    numeric_predictions = output.select_dtypes(include=[np.number]).iloc[:, len(
        conditions.select_dtypes(include=[np.number]).columns
    ):]
    if not np.isfinite(numeric_predictions.to_numpy(dtype=float)).all():
        raise ValueError("A selected model produced a non-finite prediction")
    _write_predictions(output, args.output)
    print(f"Scored {len(output):,} conditions with {len(args.models)} model(s): {args.output}")


if __name__ == "__main__":
    main()
