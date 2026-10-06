#!/usr/bin/env python3
"""Run the observed-data refresh once, in dependency order, without monitoring."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
FIGURE_DIR = ROOT / "report/figures"
FIGURE_SCRIPTS = (
    "plot_figure1_dataset_composition.py",
    "plot_figure2_prediction_diagnostics.py",
    "plot_figure3_shapley_importance.py",
    "plot_figure4_domain_generalization.py",
    "plot_figure5_adaptation_curves.py",
    "plot_polymer_directional_beeswarm.py",
)
FIGURE_PDFS = (
    "figure1_dataset_composition.pdf",
    "figure2_prediction_diagnostics.pdf",
    "figure3_shapley_importance.pdf",
    "figure4_domain_generalization.pdf",
    "figure5_adaptation_curves.pdf",
    "si_concentration_distributions.pdf",
    "polymer_directional_beeswarm_embedding_xgboost.pdf",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_scripts() -> tuple[Path, ...]:
    """Publication assets are local-only; validate them when present."""
    if not FIGURE_DIR.exists():
        return ()
    paths = tuple(FIGURE_DIR / name for name in FIGURE_SCRIPTS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete local publication scripts: {missing}")
    return paths


def stages(jobs: int, figure_scripts: tuple[Path, ...]) -> list[tuple[str, list[list[str]]]]:
    evaluation = ["scripts/train_models.py", "--report-evaluation"]
    result = [
        ("production_training", [["scripts/train_models.py"]]),
        ("static_evaluation", [[*evaluation, "--phase", "static"]]),
        ("adaptation_evaluation", [
            [*evaluation, "--phase", "adaptation", "--repeat", str(repeat)]
            for repeat in range(10)
        ]),
        ("compile_evaluation", [[*evaluation, "--phase", "compile"]]),
        ("group_feature_importance", [[
            "scripts/run_feature_importance.py", "all", "--force", "--jobs", str(jobs)
        ]]),
        ("polymer_feature_importance", [[
            "scripts/run_polymer_hierarchical_importance.py", "all", "--force", "--jobs", str(jobs)
        ]]),
    ]
    if figure_scripts:
        result.append(("publication_figures", [[str(path)] for path in figure_scripts]))
    return result


def verify_outputs(started: float, *, include_publication_figures: bool) -> None:
    import joblib

    sys.path.insert(0, str(ROOT))
    from lcst_pipeline.features import FEATURE_ENCODING_VERSION
    from lcst_pipeline.modeling import MODEL_DEFINITIONS
    from lcst_pipeline.schema import MASTER_PATH, load_config

    metadata = load_config()["chemical_metadata"]
    master_hash = fingerprint(MASTER_PATH)
    expected = [
        ROOT / "outputs/regression_metrics.csv",
        ROOT / "outputs/classifier_metrics.csv",
    ]
    for model_id in MODEL_DEFINITIONS:
        path = ROOT / "models" / f"{model_id}.joblib"
        artifact = joblib.load(path)
        if artifact.get("feature_encoding") != FEATURE_ENCODING_VERSION:
            raise ValueError(f"Outdated input encoding: {path}")
        if artifact.get("training_master_sha256") != master_hash:
            raise ValueError(f"Wrong training dataset: {path}")
        for table in ("additive_descriptors", "salt_descriptors"):
            if artifact["chemical_metadata"][table] != metadata[table]:
                raise ValueError(f"Outdated chemical lookup: {path}")
        expected.append(path)
    for relative in (
        "outputs/report_evaluation/evaluation_manifest.json",
        "outputs/feature_importance/manifest.json",
        "outputs/polymer_hierarchical_importance/manifest.json",
    ):
        path = ROOT / relative
        manifest = json.loads(path.read_text())
        if manifest.get("master_sha256") != master_hash:
            raise ValueError(f"Wrong analysis dataset: {path}")
        expected.append(path)
        for name, digest in manifest.get("output_sha256", {}).items():
            output = path.parent / name
            if fingerprint(output) != digest:
                raise ValueError(f"Analysis output checksum mismatch: {output}")
            expected.append(output)
    if include_publication_figures:
        expected.extend(FIGURE_DIR / name for name in FIGURE_PDFS)
    for path in expected:
        if not path.is_file() or not path.stat().st_size or path.stat().st_mtime < started:
            raise ValueError(f"Expected freshly generated output: {path}")


def run(run_dir: Path, jobs: int) -> None:
    figure_scripts = publication_scripts()
    if not figure_scripts:
        print("Local publication scripts are absent; refreshing model and analysis outputs only.", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Use a new log directory: {run_dir}")
    lock_dir = ROOT / "outputs/refresh_runs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "active.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = [
            ROOT / "data/processed/lcst_master_observed.csv",
            ROOT / "config/lcst.json",
            ROOT / "config/chemical_descriptor_sources.json",
            *sorted((ROOT / "lcst_pipeline").glob("*.py")),
            *sorted((ROOT / "scripts").glob("*.py")),
            *figure_scripts,
        ]
        hashes = {str(path.relative_to(ROOT)): fingerprint(path) for path in inputs}
        started = time.time()
        status = dict(state="running", started_at=now(), pid=os.getpid(), jobs=jobs,
                      stage="initializing", completed_stages=[], input_sha256=hashes,
                      include_publication_figures=bool(figure_scripts))

        def save_status() -> None:
            status["updated_at"] = now()
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(status_path)

        def unchanged_inputs() -> None:
            changed = [name for name, digest in hashes.items() if fingerprint(ROOT / name) != digest]
            if changed:
                raise ValueError(f"Inputs/code changed during this run: {changed}")

        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1"}

        def execute(item: tuple[str, list[str]]) -> None:
            name, arguments = item
            command = [str(PYTHON), "-u", *arguments]
            with (run_dir / f"{name}.log").open("w") as log:
                log.write(f"Started {now()}\nCommand: {command!r}\n")
                log.flush()
                subprocess.run(command, cwd=ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                log.write(f"\nCompleted {now()}\n")

        save_status()
        try:
            for name, commands in stages(jobs, figure_scripts):
                unchanged_inputs()
                status["stage"] = name
                save_status()
                print(f"{now()} Starting {name}", flush=True)
                work = [(f"{name}_{index:02d}", command) for index, command in enumerate(commands)]
                workers = jobs if name == "adaptation_evaluation" else 1
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for result in executor.map(execute, work):
                        pass
                status["completed_stages"].append(name)
                save_status()
                print(f"{now()} Completed {name}", flush=True)
            unchanged_inputs()
            status["stage"] = "output_verification"
            save_status()
            verify_outputs(started, include_publication_figures=bool(figure_scripts))
            status.update(state="complete", stage="complete", finished_at=now())
            save_status()
            print(f"{now()} Complete", flush=True)
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
            save_status()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.dry_run:
        print(json.dumps(stages(args.jobs, publication_scripts()), indent=2))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    run(args.run_dir.resolve(), args.jobs)


if __name__ == "__main__":
    main()
