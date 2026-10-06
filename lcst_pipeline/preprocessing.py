"""Training-only imputation that never replaces structural ingredient zeros."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import DESCRIPTOR_23_COLUMNS, TWO_SLOT_37_COLUMNS


GROUP_COLUMNS = {
    "polymer": tuple(DESCRIPTOR_23_COLUMNS[:3]),
    "additive": tuple(DESCRIPTOR_23_COLUMNS[3:16] + TWO_SLOT_37_COLUMNS[3:20]),
    "salt": tuple(DESCRIPTOR_23_COLUMNS[16:23] + TWO_SLOT_37_COLUMNS[20:37]),
}


class PresentGroupMedianImputer:
    """Fill only NaNs, using medians from observed, present-group training rows.

    Presence is bookkeeping attached to the feature frame, not a model feature.
    No estimate is invented when a training fold has no eligible donor values.
    """

    def fit(self, features: pd.DataFrame) -> "PresentGroupMedianImputer":
        self.feature_columns_ = tuple(features.columns)
        values = features.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite model features are not supported")
        presence = features.attrs.get("ingredient_presence", {})
        self.statistics_ = np.full(values.shape[1], np.nan)
        for index, column in enumerate(features.columns):
            group = next((g for g, columns in GROUP_COLUMNS.items() if column in columns), None)
            if group in presence:
                mapping = presence[group]
                if not all(key in mapping for key in features.index):
                    raise ValueError("Ingredient-presence metadata is not aligned with the feature rows")
                present = np.array([mapping[key] for key in features.index], dtype=bool)
            elif group is not None:
                raise ValueError(f"Missing ingredient-presence metadata for {group} imputation")
            else:
                present = np.ones(len(features), dtype=bool)
            donors = values[present & np.isfinite(values[:, index]), index]
            if len(donors):
                self.statistics_[index] = np.median(donors)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if tuple(features.columns) != self.feature_columns_:
            raise ValueError("Imputer feature columns differ from its training columns")
        values = features.to_numpy(dtype=float, copy=True)
        if np.isinf(values).any():
            raise ValueError("Infinite model features are not supported")
        missing = np.isnan(values)
        unresolved = missing.any(axis=0) & np.isnan(self.statistics_)
        if unresolved.any():
            columns = [self.feature_columns_[i] for i in np.flatnonzero(unresolved)]
            raise ValueError(
                f"Cannot impute {columns}: this training fold has no observed values "
                "among samples containing the ingredient. Supply the missing metadata."
            )
        return np.where(missing, self.statistics_, values)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        return self.fit(features).transform(features)
