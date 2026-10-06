"""Shared metrics and neural-network helpers for LCST evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from torch import nn

from .modeling import CLASS_LABELS
from .schema import FORMULATION_COLUMNS
from .preprocessing import PresentGroupMedianImputer


BOOTSTRAPS = 2000


def condition_id(frame: pd.DataFrame) -> pd.Series:
    normalized = frame[FORMULATION_COLUMNS].copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(
            lambda value: (
                "<NA>"
                if pd.isna(value)
                else format(float(value), ".12g")
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        )
    joined = normalized.astype(str).agg("|".join, axis=1)
    return joined.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16])


def add_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["condition_id"] = condition_id(result)
    if result["condition_id"].duplicated().any():
        raise AssertionError("Condition pooling produced duplicate formulation IDs")
    return result


def assert_disjoint(train: pd.DataFrame, test: pd.DataFrame, label: str) -> None:
    overlap = set(train["condition_id"]) & set(test["condition_id"])
    if overlap:
        raise AssertionError(f"{label} has {len(overlap)} overlapping train/test formulations")


def finite_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if (
        len(actual) < 2
        or np.isclose(np.std(actual), 0.0)
        or np.isclose(np.std(predicted), 0.0)
    ):
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def regression_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(np.mean(predicted - actual)),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) >= 2 and not np.isclose(np.std(actual), 0.0)
            else float("nan")
        ),
        "pearson_r": finite_correlation(actual, predicted),
    }


def classifier_scores(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(np.asarray(predicted, dtype=float), 1e-15, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    target = np.asarray(target, dtype=float)
    positive = target > 0
    absolute = np.abs(probabilities - target)
    scores = {
        "cross_entropy": float(
            -np.sum(
                np.where(positive, target * np.log(probabilities), 0.0), axis=1
            ).mean()
        ),
        "probability_mae": float(absolute.mean()),
        "total_variation": float((0.5 * absolute.sum(axis=1)).mean()),
        "soft_brier": float(np.sum((probabilities - target) ** 2, axis=1).mean()),
    }
    hard = (target > 0).sum(axis=1) == 1
    if hard.any():
        true_hard = target[hard].argmax(axis=1)
        predicted_hard = probabilities[hard].argmax(axis=1)
        observed_classes = np.unique(true_hard)
        balanced_accuracy = float(
            np.mean(
                [
                    np.mean(predicted_hard[true_hard == label] == label)
                    for label in observed_classes
                ]
            )
        )
        scores.update(
            {
                "accuracy": float(accuracy_score(true_hard, predicted_hard)),
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": float(
                    f1_score(
                        true_hard,
                        predicted_hard,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "n_hard": float(hard.sum()),
            }
        )
    else:
        scores.update(
            {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "n_hard": 0.0,
            }
        )
    return scores


def prediction_base(
    conditions: pd.DataFrame,
    challenge: str,
    holdout: str,
    task: str,
    model_id: str,
    seed: int,
    exploratory: bool,
    notes: str,
) -> pd.DataFrame:
    columns = ["condition_id", *FORMULATION_COLUMNS]
    result = conditions[columns].copy()
    result.insert(0, "task", task)
    result.insert(0, "seed", seed)
    result.insert(0, "model_id", model_id)
    result.insert(0, "holdout", holdout)
    result.insert(0, "challenge", challenge)
    result["exploratory"] = exploratory
    result["notes"] = notes
    return result


def metric_function(task: str) -> Callable[[np.ndarray, np.ndarray], dict[str, float]]:
    return regression_scores if task == "regression" else classifier_scores


def arrays_from_predictions(
    frame: pd.DataFrame, task: str
) -> tuple[np.ndarray, np.ndarray]:
    if task == "regression":
        return (
            frame["target_c"].to_numpy(dtype=float),
            frame["prediction_c"].to_numpy(dtype=float),
        )
    return (
        frame[[f"target_{label.lower()}" for label in CLASS_LABELS]].to_numpy(
            dtype=float
        ),
        frame[[f"prediction_{label.lower()}" for label in CLASS_LABELS]].to_numpy(
            dtype=float
        ),
    )


def bootstrap_metrics(
    frame: pd.DataFrame, task: str, rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    averaged_columns = (
        ["prediction_c"]
        if task == "regression"
        else [f"prediction_{label.lower()}" for label in CLASS_LABELS]
    )
    target_columns = (
        ["target_c"]
        if task == "regression"
        else [f"target_{label.lower()}" for label in CLASS_LABELS]
    )
    base_columns = ["condition_id", *target_columns]
    averaged = frame.groupby(base_columns, dropna=False, as_index=False)[
        averaged_columns
    ].mean()
    actual, predicted = arrays_from_predictions(averaged, task)
    scorer = metric_function(task)
    distributions: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(averaged), len(averaged))
        scores = scorer(actual[index], predicted[index])
        for metric, value in scores.items():
            if metric == "n_hard" or not np.isfinite(value):
                continue
            distributions.setdefault(metric, []).append(float(value))
    return {
        metric: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for metric, values in distributions.items()
        if values
    }


def summarize_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_columns = ["challenge", "holdout", "task", "model_id"]
    rng = np.random.default_rng(20260821)
    for keys, frame in predictions.groupby(group_columns, sort=True):
        challenge, holdout, task, model_id = keys
        for seed, seed_frame in frame.groupby("seed"):
            actual, predicted = arrays_from_predictions(seed_frame, task)
            for metric, value in metric_function(task)(actual, predicted).items():
                if metric == "n_hard":
                    continue
                seed_rows.append(
                    {
                        "challenge": challenge,
                        "holdout": holdout,
                        "task": task,
                        "model_id": model_id,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                        "n_test": seed_frame["condition_id"].nunique(),
                        "exploratory": bool(seed_frame["exploratory"].iloc[0]),
                    }
                )
        averaged_columns = (
            ["prediction_c"]
            if task == "regression"
            else [f"prediction_{label.lower()}" for label in CLASS_LABELS]
        )
        target_columns = (
            ["target_c"]
            if task == "regression"
            else [f"target_{label.lower()}" for label in CLASS_LABELS]
        )
        averaged = frame.groupby(
            ["condition_id", *target_columns], dropna=False, as_index=False
        )[averaged_columns].mean()
        actual, predicted = arrays_from_predictions(averaged, task)
        central = metric_function(task)(actual, predicted)
        intervals = bootstrap_metrics(frame, task, rng)
        seed_subset = pd.DataFrame(seed_rows)
        seed_subset = seed_subset[
            seed_subset["challenge"].eq(challenge)
            & seed_subset["holdout"].eq(holdout)
            & seed_subset["task"].eq(task)
            & seed_subset["model_id"].eq(model_id)
        ]
        for metric, value in central.items():
            if metric == "n_hard":
                continue
            metric_seed = seed_subset[seed_subset["metric"].eq(metric)]["value"]
            ci_low, ci_high = intervals.get(
                metric, (float("nan"), float("nan"))
            )
            summary_rows.append(
                {
                    "challenge": challenge,
                    "holdout": holdout,
                    "task": task,
                    "model_id": model_id,
                    "metric": metric,
                    "value": value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "seed_mean": float(metric_seed.mean()),
                    "seed_std": float(metric_seed.std(ddof=1)),
                    "n_test": len(averaged),
                    "exploratory": bool(frame["exploratory"].iloc[0]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(seed_rows)


def paired_model_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260823)
    for (challenge, holdout, task), frame in predictions.groupby(
        ["challenge", "holdout", "task"]
    ):
        metric = "mae" if task == "regression" else "cross_entropy"
        models = sorted(frame["model_id"].unique())
        for first_index in range(len(models)):
            for second_index in range(first_index + 1, len(models)):
                first, second = models[first_index], models[second_index]
                first_frame = frame[frame["model_id"].eq(first)]
                second_frame = frame[frame["model_id"].eq(second)]
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
                first_average = first_frame.groupby(
                    ["condition_id", *target_columns], as_index=False
                )[prediction_columns].mean()
                second_average = second_frame.groupby(
                    ["condition_id", *target_columns], as_index=False
                )[prediction_columns].mean()
                merged = first_average.merge(
                    second_average,
                    on=["condition_id", *target_columns],
                    suffixes=("_first", "_second"),
                )
                if merged.empty:
                    continue
                actual = merged[target_columns].to_numpy(dtype=float)
                if task == "regression":
                    actual = actual.ravel()
                    prediction_first = merged["prediction_c_first"].to_numpy(
                        dtype=float
                    )
                    prediction_second = merged["prediction_c_second"].to_numpy(
                        dtype=float
                    )
                else:
                    prediction_first = merged[
                        [
                            f"prediction_{label.lower()}_first"
                            for label in CLASS_LABELS
                        ]
                    ].to_numpy(dtype=float)
                    prediction_second = merged[
                        [
                            f"prediction_{label.lower()}_second"
                            for label in CLASS_LABELS
                        ]
                    ].to_numpy(dtype=float)
                scorer = metric_function(task)
                difference = (
                    scorer(actual, prediction_first)[metric]
                    - scorer(actual, prediction_second)[metric]
                )
                bootstrap = []
                for _ in range(BOOTSTRAPS):
                    index = rng.integers(0, len(merged), len(merged))
                    bootstrap.append(
                        scorer(actual[index], prediction_first[index])[metric]
                        - scorer(actual[index], prediction_second[index])[metric]
                    )
                rows.append(
                    {
                        "challenge": challenge,
                        "holdout": holdout,
                        "task": task,
                        "metric": metric,
                        "model_a": first,
                        "model_b": second,
                        "difference_a_minus_b": difference,
                        "ci95_low": float(np.quantile(bootstrap, 0.025)),
                        "ci95_high": float(np.quantile(bootstrap, 0.975)),
                        "n_test": len(merged),
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class MLPRegressionProtocol:
    hidden_layers: tuple[int, int] = (32, 16)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    validation_fraction: float = 0.15
    maximum_epochs: int = 500
    patience: int = 50
    minimum_delta: float = 1e-4
    loss: str = "SmoothL1Loss on standardized LCST"
    optimizer: str = "AdamW"
    missing_values: str = "Present-group training medians; absent groups stay zero"
    feature_scaling: str = "Training-only MaxAbs scaling without centering"


@dataclass(frozen=True)
class MLPClassificationProtocol:
    hidden_layers: tuple[int, int] = (32, 16)
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    validation_fraction: float = 0.15
    maximum_epochs: int = 1000
    patience: int = 50
    minimum_delta: float = 1e-4
    loss: str = "Soft-target cross-entropy"
    optimizer: str = "AdamW"
    missing_values: str = (
        "Present-group training medians; absent groups stay zero"
    )
    feature_scaling: str = "Training-only MaxAbs scaling without centering"


MLP_REGRESSION_PROTOCOL = MLPRegressionProtocol()
MLP_CLASSIFICATION_PROTOCOL = MLPClassificationProtocol()


class _TinyRegressor(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        protocol = MLP_REGRESSION_PROTOCOL
        self.network = nn.Sequential(
            nn.Linear(input_width, protocol.hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(protocol.dropout),
            nn.Linear(protocol.hidden_layers[0], protocol.hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(protocol.dropout),
            nn.Linear(protocol.hidden_layers[1], 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


class _TinyClassifier(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        protocol = MLP_CLASSIFICATION_PROTOCOL
        self.network = nn.Sequential(
            nn.Linear(input_width, protocol.hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(protocol.dropout),
            nn.Linear(protocol.hidden_layers[0], protocol.hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(protocol.dropout),
            nn.Linear(protocol.hidden_layers[1], 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def _regression_validation_indices(
    target: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    protocol = MLP_REGRESSION_PROTOCOL
    indices = np.arange(len(target))
    stratify = None
    try:
        bins = pd.qcut(target, q=5, duplicates="drop", labels=False)
        counts = pd.Series(bins).value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = np.asarray(bins)
    except ValueError:
        stratify = None
    train, validation = train_test_split(
        indices,
        test_size=protocol.validation_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train), np.asarray(validation)


def _train_regressor(
    features: np.ndarray, target: np.ndarray, epochs: int, seed: int
) -> _TinyRegressor:
    protocol = MLP_REGRESSION_PROTOCOL
    _set_seed(seed)
    model = _TinyRegressor(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    loss_function = nn.SmoothL1Loss()
    feature_tensor = _tensor(features)
    target_tensor = _tensor(target)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(feature_tensor), target_tensor)
        loss.backward()
        optimizer.step()
    return model


def _select_regression_epoch(
    features: pd.DataFrame, target: np.ndarray, seed: int
) -> int:
    protocol = MLP_REGRESSION_PROTOCOL
    fit_index, validation_index = _regression_validation_indices(target, seed)
    imputer = PresentGroupMedianImputer().fit(features.iloc[fit_index])
    x_fit = imputer.transform(features.iloc[fit_index])
    x_validation = imputer.transform(features.iloc[validation_index])
    x_scaler = MaxAbsScaler().fit(x_fit)
    y_mean = float(target[fit_index].mean())
    y_scale = float(target[fit_index].std(ddof=0))
    if np.isclose(y_scale, 0.0):
        y_scale = 1.0
    x_fit = x_scaler.transform(x_fit)
    x_validation = x_scaler.transform(x_validation)
    y_fit = (target[fit_index] - y_mean) / y_scale
    y_validation = (target[validation_index] - y_mean) / y_scale

    _set_seed(seed)
    model = _TinyRegressor(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    loss_function = nn.SmoothL1Loss()
    x_fit_tensor = _tensor(x_fit)
    y_fit_tensor = _tensor(y_fit)
    x_validation_tensor = _tensor(x_validation)
    y_validation_tensor = _tensor(y_validation)
    best_loss = float("inf")
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, protocol.maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x_fit_tensor), y_fit_tensor)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(x_validation_tensor), y_validation_tensor).item()
            )
        if validation_loss < best_loss - protocol.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= protocol.patience:
            break
    return best_epoch


@dataclass
class FittedMLPRegressor:
    scaler: MaxAbsScaler
    imputer: PresentGroupMedianImputer
    target_mean: float
    target_scale: float
    network: _TinyRegressor
    selected_epoch: int
    feature_columns: tuple[str, ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != list(self.feature_columns):
            raise ValueError("Prediction columns do not match MLP training columns")
        values = self.imputer.transform(features)
        self.network.eval()
        with torch.no_grad():
            prediction = (
                self.network(_tensor(self.scaler.transform(values))).numpy()
                * self.target_scale
                + self.target_mean
            )
        if not np.isfinite(prediction).all():
            raise ValueError("MLP produced non-finite predictions")
        return np.asarray(prediction, dtype=float)


def fit_mlp_regressor(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    seed: int,
) -> FittedMLPRegressor:
    best_epoch = _select_regression_epoch(train_features, train_target, seed)
    imputer = PresentGroupMedianImputer().fit(train_features)
    train_array = imputer.transform(train_features)
    x_scaler = MaxAbsScaler().fit(train_array)
    y_mean = float(train_target.mean())
    y_scale = float(train_target.std(ddof=0))
    if np.isclose(y_scale, 0.0):
        y_scale = 1.0
    scaled_train = x_scaler.transform(train_array)
    scaled_target = (train_target - y_mean) / y_scale
    model = _train_regressor(
        scaled_train, scaled_target, best_epoch, seed
    )
    return FittedMLPRegressor(
        scaler=x_scaler,
        imputer=imputer,
        target_mean=y_mean,
        target_scale=y_scale,
        network=model,
        selected_epoch=best_epoch,
        feature_columns=tuple(train_features.columns),
    )


def fit_predict_mlp_regressor(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    test_features: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, int]:
    if list(train_features.columns) != list(test_features.columns):
        raise AssertionError("MLP train/test feature contracts differ")
    fitted = fit_mlp_regressor(train_features, train_target, seed)
    return fitted.predict(test_features), fitted.selected_epoch


def _soft_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _classification_validation_indices(
    strata: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    protocol = MLP_CLASSIFICATION_PROTOCOL
    indices = np.arange(len(strata))
    stratify = None
    counts = pd.Series(strata).value_counts()
    validation_size = int(np.ceil(protocol.validation_fraction * len(strata)))
    if len(counts) >= 2 and counts.min() >= 2 and validation_size >= len(counts):
        stratify = strata
    train, validation = train_test_split(
        indices,
        test_size=protocol.validation_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return np.asarray(train), np.asarray(validation)


def _select_classification_epoch(
    features: pd.DataFrame,
    target: np.ndarray,
    strata: np.ndarray,
    seed: int,
) -> int:
    protocol = MLP_CLASSIFICATION_PROTOCOL
    fit_index, validation_index = _classification_validation_indices(strata, seed)
    imputer = PresentGroupMedianImputer().fit(features.iloc[fit_index])
    fit_values = imputer.transform(features.iloc[fit_index])
    validation_values = imputer.transform(features.iloc[validation_index])
    scaler = MaxAbsScaler().fit(fit_values)
    x_fit = _tensor(scaler.transform(fit_values))
    y_fit = _tensor(target[fit_index])
    x_validation = _tensor(scaler.transform(validation_values))
    y_validation = _tensor(target[validation_index])
    _set_seed(seed)
    model = _TinyClassifier(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, protocol.maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _soft_cross_entropy(model(x_fit), y_fit)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _soft_cross_entropy(model(x_validation), y_validation).item()
            )
        if validation_loss < best_loss - protocol.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= protocol.patience:
            break
    return best_epoch


def _train_classifier(
    features: np.ndarray, target: np.ndarray, epochs: int, seed: int
) -> _TinyClassifier:
    protocol = MLP_CLASSIFICATION_PROTOCOL
    _set_seed(seed)
    model = _TinyClassifier(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    x_train = _tensor(features)
    y_train = _tensor(target)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _soft_cross_entropy(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model


@dataclass
class FittedMLPClassifier:
    scaler: MaxAbsScaler
    imputer: PresentGroupMedianImputer
    network: _TinyClassifier
    selected_epoch: int
    feature_columns: tuple[str, ...]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if list(features.columns) != list(self.feature_columns):
            raise ValueError("Prediction columns do not match MLP training columns")
        values = self.imputer.transform(features)
        self.network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(
                self.network(_tensor(self.scaler.transform(values))), dim=1
            ).numpy()
        if probabilities.shape != (len(values), 3) or not np.isfinite(
            probabilities
        ).all():
            raise ValueError("Classifier MLP produced invalid probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("Classifier MLP probabilities do not sum to one")
        return np.asarray(probabilities, dtype=float)


def fit_mlp_classifier(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    train_strata: np.ndarray,
    seed: int,
) -> FittedMLPClassifier:
    epoch = _select_classification_epoch(
        train_features, train_target, train_strata, seed
    )
    imputer = PresentGroupMedianImputer().fit(train_features)
    train_array = imputer.transform(train_features)
    scaler = MaxAbsScaler().fit(train_array)
    model = _train_classifier(
        scaler.transform(train_array), train_target, epoch, seed
    )
    return FittedMLPClassifier(
        scaler=scaler,
        imputer=imputer,
        network=model,
        selected_epoch=epoch,
        feature_columns=tuple(train_features.columns),
    )


def fit_predict_mlp_classifier(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    train_strata: np.ndarray,
    test_features: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, int]:
    if list(train_features.columns) != list(test_features.columns):
        raise AssertionError("Classifier MLP train/test feature contracts differ")
    fitted = fit_mlp_classifier(train_features, train_target, train_strata, seed)
    return fitted.predict_proba(test_features), fitted.selected_epoch
