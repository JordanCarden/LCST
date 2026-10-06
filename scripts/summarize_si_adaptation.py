#!/usr/bin/env python3
"""Reproduce SI Tables S5 and S6 from the saved adaptation results.

Uses embedding XGBoost only: regression MAE in degrees Celsius and classification
accuracy in percent, each averaged over ten separately scored repeats. Initial
and final scores are unsmoothed. k90 is the first integer acquisition count
achieving 90% of the isotonic-fitted endpoint improvement (decreasing for MAE,
increasing for accuracy). Blank k90 values mean no fitted improvement; a family
mean is blank if any domain has undefined k90. Each domain has equal weight.

Run from any directory with Python and the repository's requirements installed:
    python scripts/summarize_si_adaptation.py

Writes si_adaptation_endpoints.csv and si_adaptation_family_summary.csv under
outputs/report_evaluation. These are the SI-specific summaries; the existing
adaptation_k90_* tables use other model/metric aggregations. No model fitting,
plotting, report files, or per-repeat intermediate files are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "outputs/report_evaluation"
MODELS = {
    "regression": "two_slot_pca38_buffer_regressor_xgboost",
    "classification": "two_slot_pca38_buffer_classifier_xgboost",
}
METRICS = {"regression": "mae", "classification": "accuracy"}
DOMAINS = {
    "additive": ("CHAPS", "CTAB", "SDS"),
    "salt": ("NaCl", "Na2SO4"),
    "molecular_weight": ("20kDa", "40kDa", "86kDa", "250kDa"),
}
REPEATS = tuple(range(10))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    master_hash = file_sha256(REPO_ROOT / "data/processed/lcst_master_observed.csv")
    curves, metrics = [], []
    for directory in (RESULTS, RESULTS / "mw_adaptation"):
        manifest = json.loads((directory / "evaluation_manifest.json").read_text())
        if manifest["master_sha256"] != master_hash:
            raise ValueError(f"Adaptation results do not match the current master: {directory}")
        metric_path = directory / "adaptation_metrics_long.csv"
        curve_path = directory / "adaptation_curves.csv"
        recorded_hashes = manifest.get("output_sha256", {})
        if recorded_hashes.get(metric_path.name) != file_sha256(metric_path):
            raise ValueError(f"Adaptation metrics checksum mismatch: {metric_path}")
        curve_hash = recorded_hashes.get(curve_path.name)
        if curve_hash is not None and curve_hash != file_sha256(curve_path):
            raise ValueError(f"Adaptation curves checksum mismatch: {curve_path}")
        metrics.append(pd.read_csv(metric_path))
        curves.append(pd.read_csv(curve_path))
    return pd.concat(curves, ignore_index=True), pd.concat(metrics, ignore_index=True)


def summarize(
    curves: pd.DataFrame, metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for task, model_id in MODELS.items():
        for domain_type, domains in DOMAINS.items():
            for domain in domains:
                filters = dict(
                    domain=domain, domain_type=domain_type, task=task,
                    model_id=model_id, metric=METRICS[task],
                )
                selected = []
                for source in (curves, metrics):
                    mask = pd.Series(True, index=source.index)
                    for column, value in filters.items():
                        mask &= source[column].eq(value)
                    selected.append(source.loc[mask])
                saved, runs = selected
                saved = saved.sort_values("k")
                if saved.empty or saved.duplicated("k").any():
                    raise ValueError(f"Missing or duplicate curve: {filters}")
                k = saved["k"].to_numpy()
                if not np.array_equal(k, np.arange(len(k))) or k[-1] <= 0:
                    raise ValueError(f"Incomplete acquisition steps: {filters}")
                if not saved["kmax"].eq(k[-1]).all() or not runs["kmax"].eq(k[-1]).all():
                    raise ValueError(f"Inconsistent acquisition pool: {filters}")
                if runs.duplicated(["repeat", "k"]).any():
                    raise ValueError(f"Duplicate repeat scores: {filters}")
                pivot = runs.pivot(index="repeat", columns="k", values="value")
                if tuple(pivot.index) != REPEATS or not np.array_equal(pivot.columns, k):
                    raise ValueError(f"Incomplete repeated runs: {filters}")
                values = pivot.to_numpy(dtype=float)
                if not np.isfinite(values).all() or not saved["n_repeats"].eq(10).all():
                    raise ValueError(f"Missing repeated-run values: {filters}")
                increasing = task == "classification"
                if (values < 0).any() or (increasing and (values > 1).any()):
                    raise ValueError(f"Scores outside the metric range: {filters}")
                mean = values.mean(axis=0)
                if not np.allclose(mean, saved["mean"], rtol=0, atol=1e-12):
                    raise ValueError(f"Saved means disagree with the repeat scores: {filters}")
                mean = mean * (100.0 if increasing else 1.0)
                fitted = IsotonicRegression(increasing=increasing).fit_transform(k, mean)
                direction = 1 if increasing else -1
                improvement = float(direction * (fitted[-1] - fitted[0]))
                threshold = float(fitted[0] + direction * 0.90 * improvement)
                reached = k[direction * (fitted - threshold) >= -1e-12]
                k90 = int(reached[0]) if improvement > 0 and len(reached) else None
                rows.append({
                    **filters,
                    "units": "percent" if increasing else "degrees Celsius",
                    "kmax": int(k[-1]),
                    "baseline": float(mean[0]), "endpoint": float(mean[-1]),
                    "fitted_baseline": float(fitted[0]),
                    "fitted_endpoint": float(fitted[-1]),
                    "fitted_improvement": improvement, "threshold": threshold,
                    "k90": k90,
                    "k90_pool_percent": 100.0 * k90 / int(k[-1]) if k90 is not None else None,
                })

    families = []
    for domain_type in DOMAINS:
        for task, model_id in MODELS.items():
            group = [row for row in rows if row["domain_type"] == domain_type and row["task"] == task]
            defined = [row["k90"] for row in group if row["k90"] is not None]
            families.append({
                "domain_type": domain_type, "task": task,
                "model_id": model_id, "metric": METRICS[task],
                "n_domains": len(group), "n_defined_k90": len(defined),
                "mean_k90": float(np.mean(defined)) if len(defined) == len(group) else None,
            })
    endpoints = pd.DataFrame(rows)
    endpoints["k90"] = endpoints["k90"].astype("Int64")
    return endpoints, pd.DataFrame(families)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS,
        help="Directory for the two SI summary CSVs (default: outputs/report_evaluation)",
    )
    args = parser.parse_args()
    endpoints, families = summarize(*load_results())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("si_adaptation_endpoints.csv", endpoints),
        ("si_adaptation_family_summary.csv", families),
    ):
        path = args.output_dir / name
        table.to_csv(path, index=False)
        print(f"Wrote {len(table)} rows: {path}")


if __name__ == "__main__":
    main()
