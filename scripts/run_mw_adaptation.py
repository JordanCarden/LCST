#!/usr/bin/env python3
"""Extend the saved adaptation study with molecular-weight targets only.

Uses the existing embedding XGBoost fitting, partitioning, scoring and
bootstrap routines. Chemical-target outputs and production models are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from lcst_pipeline import report_evaluation as ev

OUT = ROOT / "outputs/report_evaluation/mw_adaptation"
WEIGHTS = (20, 40, 86, 250)
REPEATS = tuple(range(10))
MODELS = {
    "regression": "two_slot_pca38_buffer_regressor_xgboost",
    "classification": "two_slot_pca38_buffer_classifier_xgboost",
}


def inputs_fingerprint():
    paths = [
        ev.MASTER_PATH, ROOT / "config/lcst.json", ev.EMBEDDING_CACHE,
        Path(__file__),
        *[ROOT / "lcst_pipeline" / f"{name}.py" for name in (
            "report_evaluation", "evaluation", "modeling", "features",
            "preprocessing", "embeddings", "schema")],
    ]
    return {str(p.relative_to(ROOT)): ev.file_sha256(p) for p in paths}


def existing_results_fingerprint():
    paths = sorted(ev.OUTPUT_DIR.glob("*.csv")) + [ev.OUTPUT_DIR / "evaluation_manifest.json"]
    return {str(p.relative_to(ROOT)): ev.file_sha256(p) for p in paths}


def prepared_inputs():
    if not ev.EMBEDDING_CACHE.exists():
        raise FileNotFoundError("Use the existing verified embedding cache for this extension")
    _, regression, classification, metadata, context = ev.load_inputs()
    pca = ev.pca_result(context)
    manifest = json.loads((ev.OUTPUT_DIR / "evaluation_manifest.json").read_text())
    if ev.file_sha256(ev.MASTER_PATH) != manifest["master_sha256"]:
        raise AssertionError("Current data do not match the existing evaluation")
    return regression, classification, ev.metadata_with_pca(metadata, pca), pca


def check_existing_protocol():
    """Reproduce one saved k=0 score for each task before adding new domains."""
    regression, classification, metadata, pca = prepared_inputs()
    saved = pd.read_csv(ev.OUTPUT_DIR / "adaptation_metrics_long.csv")
    inventory = pd.read_csv(ev.OUTPUT_DIR / "adaptation_inventory.csv")
    checks = {}
    for task, model_id in MODELS.items():
        original, target = ev.adaptation_target(regression, classification, "CHAPS", task)
        pool_idx, panel_idx = ev.panel_and_pool(target, task, ev.adaptation_seed("CHAPS", task, 0, 4200))
        order = np.random.default_rng(ev.adaptation_seed("CHAPS", task, 0, 8400)).permutation(len(pool_idx))
        panel = target.iloc[panel_idx].reset_index(drop=True)
        combined = pd.concat([original, target.iloc[pool_idx].iloc[order], panel], ignore_index=True)
        features = ev.feature_tables(combined, metadata)["two_slot38"]
        pred, _ = ev.fit_model(task, model_id, features.iloc[:len(original)], original,
                               features.iloc[-len(panel):], 42)
        scores = (ev.metric_helpers.regression_scores(panel.transition_temperature_c.to_numpy(), pred)
                  if task == "regression" else
                  ev.metric_helpers.classifier_scores(panel[list(ev.TARGET_COLUMNS)].to_numpy(), pred))
        expected = saved.loc[(saved.domain == "CHAPS") & (saved.task == task)
                             & (saved.repeat == 0) & (saved.k == 0) & (saved.model_id == model_id)]
        for row in expected.itertuples():
            np.testing.assert_allclose(scores[row.metric], row.value, atol=1e-10, rtol=0, equal_nan=True)
        inv = inventory.loc[(inventory.domain == "CHAPS") & (inventory.task == task)
                            & (inventory.repeat == 0)].iloc[0]
        assert inv.evaluation_ids == ";".join(panel.condition_id)
        assert inv.pca_components_sha256 == pca.provenance["pca_components_sha256"]
        checks[task] = {"all_saved_metrics_reproduced": True, "fixed_panel_reproduced": True,
                        "primary_score": scores["mae" if task == "regression" else "accuracy"]}
    return checks


def run_shard(weight: int, repeat: int, fingerprint: dict):
    if inputs_fingerprint() != fingerprint:
        raise AssertionError("Input files changed before the worker started")
    started = time.perf_counter()
    regression, classification, metadata, pca = prepared_inputs()
    rows, inventory = [], []
    # Extend the existing domain seed indexing after the five chemical domains.
    domain_index = len(ev.ADAPTATION_DOMAINS) + WEIGHTS.index(weight)
    for task, model_id in MODELS.items():
        conditions = regression if task == "regression" else classification
        mask = conditions.polymer_name.eq("Dex-MA") & conditions.polymer_mw_kda.eq(weight)
        original = conditions.loc[~mask].reset_index(drop=True)
        target = conditions.loc[mask].reset_index(drop=True)
        assert len(target) > 10
        assert not (original.polymer_name.eq("Dex-MA") & original.polymer_mw_kda.eq(weight)).any()
        assert target.polymer_name.eq("Dex-MA").all()
        offset = domain_index * 100 + (0 if task == "regression" else 50) + repeat
        partition_seed, acquisition_seed = 4200 + offset, 8400 + offset
        pool_idx, panel_idx = ev.panel_and_pool(target, task, partition_seed)
        order = np.random.default_rng(acquisition_seed).permutation(len(pool_idx))
        panel = target.iloc[panel_idx].reset_index(drop=True)
        ordered_pool = target.iloc[pool_idx].iloc[order].reset_index(drop=True)
        combined = pd.concat([original, ordered_pool, panel], ignore_index=True)
        assert not combined.condition_id.duplicated().any()
        assert len(combined) == len(conditions) and len(panel) == 10
        ev.metric_helpers.assert_disjoint(original, target, f"MW/{weight}/{task}")
        features = ev.feature_tables(combined, metadata)["two_slot38"]
        panel_features = features.iloc[-10:]
        n_original, kmax = len(original), len(ordered_pool)
        hard = panel[list(ev.TARGET_COLUMNS)].gt(0).sum(axis=1).eq(1) if task == "classification" else None
        n_scored = int(hard.sum()) if hard is not None else 10
        assert n_scored > 0
        inventory.append({
            "domain": f"{weight}kDa", "domain_type": "molecular_weight", "task": task,
            "repeat": repeat, "partition_seed": partition_seed, "acquisition_seed": acquisition_seed,
            "n_original_train": n_original, "n_target": len(target), "n_evaluation": 10,
            "n_scored": n_scored, "kmax": kmax,
            "evaluation_ids": ";".join(panel.condition_id),
            "acquisition_order_ids": ";".join(ordered_pool.condition_id),
            **ev.pca_inventory_fields(pca, None),
        })
        for k in range(kmax + 1):
            train = combined.iloc[:n_original + k]
            assert not set(train.condition_id) & set(panel.condition_id)
            fit_started = time.perf_counter()
            prediction, epoch = ev.fit_model(task, model_id, features.iloc[:len(train)], train,
                                             panel_features, 42 + repeat)
            elapsed = time.perf_counter() - fit_started
            scores = (ev.metric_helpers.regression_scores(panel.transition_temperature_c.to_numpy(), prediction)
                      if task == "regression" else
                      ev.metric_helpers.classifier_scores(panel[list(ev.TARGET_COLUMNS)].to_numpy(), prediction))
            if task == "classification":
                assert scores["n_hard"] == n_scored
            common = {
                "domain": f"{weight}kDa", "domain_type": "molecular_weight", "task": task,
                "repeat": repeat, "k": k, "kmax": kmax, "n_original_train": n_original,
                "n_train": len(train), "n_evaluation": 10, "n_scored": n_scored,
                "model_id": model_id, "feature_set": "two_slot38", "model_family": "XGBoost",
                "selected_epoch": epoch, "fit_seconds": elapsed,
            }
            rows.extend({**common, "metric": metric, "value": value}
                        for metric, value in scores.items() if metric != "n_hard")
    assert inputs_fingerprint() == fingerprint, "Input files changed during the worker run"
    shard = OUT / "shards" / f"{weight}kDa_repeat_{repeat}"
    pd.DataFrame(rows).to_csv(f"{shard}_metrics.csv", index=False)
    pd.DataFrame(inventory).to_csv(f"{shard}_inventory.csv", index=False)
    return f"{weight} kDa repeat {repeat + 1}/10 finished ({time.perf_counter() - started:.1f} s)"


def compile_outputs(fingerprint, preserved, checks, started):
    metrics = pd.concat([pd.read_csv(OUT / "shards" / f"{w}kDa_repeat_{r}_metrics.csv")
                         for w in WEIGHTS for r in REPEATS], ignore_index=True)
    inventory = pd.concat([pd.read_csv(OUT / "shards" / f"{w}kDa_repeat_{r}_inventory.csv")
                           for w in WEIGHTS for r in REPEATS], ignore_index=True)
    assert len(inventory) == 80
    assert inventory.n_evaluation.eq(10).all() and inventory.n_scored.gt(0).all()
    assert inventory.pca_fit_count.eq(18).all()
    assert inventory.pca_components_sha256.nunique() == 1
    for inv in inventory.itertuples():
        panel = inv.evaluation_ids.split(";")
        pool = inv.acquisition_order_ids.split(";")
        assert len(panel) == len(set(panel)) == 10
        assert len(pool) == len(set(pool)) == inv.kmax and not set(pool) & set(panel)
        frame = metrics[(metrics.domain == inv.domain) & (metrics.task == inv.task)
                        & (metrics.repeat == inv.repeat)]
        for _, values in frame.groupby("metric"):
            assert not values.k.duplicated().any()
            assert values.k.tolist() == list(range(inv.kmax + 1))
        primary = frame[frame.metric == ("mae" if inv.task == "regression" else "accuracy")]
        assert np.isfinite(primary.value).all()
        if inv.task == "classification":
            assert primary.value.between(0,1).all()
    assert inputs_fingerprint() == fingerprint
    assert existing_results_fingerprint() == preserved, "Existing evaluation results changed"
    summary = ev.bootstrap_adaptation(metrics)
    metrics.to_csv(OUT / "adaptation_metrics_long.csv", index=False)
    inventory.to_csv(OUT / "adaptation_inventory.csv", index=False)
    summary.to_csv(OUT / "adaptation_curves.csv", index=False)
    result_paths = [OUT / name for name in ("adaptation_metrics_long.csv", "adaptation_inventory.csv", "adaptation_curves.csv")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "master_sha256": ev.file_sha256(ev.MASTER_PATH), "input_sha256": fingerprint,
        "existing_results_sha256": preserved,
        "output_sha256": {p.name:ev.file_sha256(p) for p in result_paths},
        "models": MODELS, "weights_kda": list(WEIGHTS), "repeats": list(REPEATS),
        "excluded_weights": {"500kDa": "Only six target formulations per task; insufficient for a fixed ten-formulation evaluation panel."},
        "fixed_evaluation_panel_size": 10, "k_rule": "Every integer from zero to N_target minus ten",
        "seed_rule": "Append 20, 40, 86 and 250 kDa after the five existing chemical targets; retain existing base seeds and task/repeat offsets.",
        "classification_scoring": "Accuracy on single-observed-class members of each fixed panel; training and probabilistic scores retain empirical replicate proportions.",
        "pca_policy": "All 18 chemicals, fixed across every acquisition step and repeat, as in the existing adaptation analysis.",
        "bootstrap_resamples": ev.BOOTSTRAPS,
        "checks": {"existing_protocol_reproduction": checks, "fixed_disjoint_panels": True,
                   "complete_acquisition_steps": True, "existing_results_unchanged": True},
    }
    (OUT / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(WEIGHTS)} MW targets, two tasks and ten repeats; saved {OUT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    preserved = existing_results_fingerprint()
    checks = check_existing_protocol()
    print("Existing CHAPS regression/classification baseline metrics and panels reproduced exactly.", flush=True)
    if args.check_only:
        print(json.dumps(checks, indent=2))
        return
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing MW run: {OUT}")
    (OUT / "shards").mkdir(parents=True)
    fingerprint = inputs_fingerprint()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_shard, weight, repeat, fingerprint)
                   for weight in (86, 250, 20, 40) for repeat in REPEATS]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    compile_outputs(fingerprint, preserved, checks, started)


if __name__ == "__main__":
    main()
