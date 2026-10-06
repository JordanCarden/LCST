# LCST: machine learning of Dex-MA phase behavior

Data, trained models, and analysis code accompanying **Machine Learning of Methacrylated Dextran Phase Separation for Prediction, Transferability, and Adaptation**.

This repository supports two tasks for methacrylated dextran (Dex-MA) formulations:

- Predict the lower critical solution temperature (LCST), in °C.
- Predict the probabilities of an LCST transition, an upper critical solution temperature (UCST) transition, or no observed transition (`NONE`).

You can [use the saved models](#use-the-saved-models) without retraining, [find the paper's numerical results](#find-the-papers-results), or [reproduce the analyses](#reproduce-the-analyses) from the supplied measurements. The repository contains the numerical results and their calculation code; the paper and supporting information (SI) provide the manuscript figures and scientific interpretation.

## Set up the environment

The saved models and published results were generated with **Python 3.12.3** and the versions in [requirements.txt](requirements.txt). Use Linux, or Linux through WSL on Windows, for the complete analysis workflow; some analysis scripts use Unix file locking. Computation runs on the CPU and does not require a GPU.

```bash
git clone https://github.com/JordanCarden/LCST.git
cd LCST
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
```

Run the commands below from the repository root with this environment active. The thread settings keep numerical work consistent and prevent each analysis worker from starting additional CPU threads. Reproducing all analyses requires substantially more computation than predicting with the saved models.

### MoLFormer download and cache

Prediction for the chemicals already included in this repository uses the representations stored in the model files: **no MoLFormer download is needed**.

When retraining models, or generating embeddings for newly registered chemicals, the code automatically downloads the required files from `ibm-research/MoLFormer-XL-both-10pct`, pinned to revision `7b12d946c181a37f6012b9dc3b002275de070314`. The first download requires internet access. Later runs reuse the complete local cache without downloading again.

`HF_HOME` or `HF_HUB_CACHE` can select the Hugging Face cache location. For a snapshot already copied to an offline computer, set `LCST_MOLFORMER_SNAPSHOT` to its directory. This override must contain the required files from the pinned revision. The evaluation also includes a [cache of the study's chemical embeddings](outputs/report_evaluation/molformer_embeddings.npz).

## Use the saved models

Save the following as `conditions.csv`. These two example formulations are present in the supplied experimental data.

```csv
polymer_name,polymer_mw_kda,polymer_functionalization_percent,polymer_concentration_mg_ml,buffer,heating_rate_c_per_min,additive_name,additive_concentration_mM,salt_name,salt_concentration_mM
Dex-MA,86,88,10,PBS,1,SDS,0.1,NaCl,1
Dex-MA,86,88,10,PBS,1,SDS,1,NaCl,1
```

Run the embedding-based XGBoost regressor and classifier used for the paper's detailed adaptation analyses:

```bash
python scripts/predict.py \
  --input conditions.csv \
  --output predictions.csv \
  --models two_slot_pca38_buffer_regressor_xgboost two_slot_pca38_buffer_classifier_xgboost
```

To compare **all eight models**, omit `--models`:

```bash
python scripts/predict.py --input conditions.csv --output predictions_all.csv
```

Input and output may also be Excel files (`.xlsx`). No transition labels or measured temperatures are needed for prediction.

### Input columns and units

All ten column headers are required. Supply concentrations in their original units; the code applies the concentration transformations internally.

| Column | What to enter |
| --- | --- |
| `polymer_name` | `Dex-MA`, or `No polymer` for a polymer-free control. |
| `polymer_mw_kda` | Dex-MA molecular weight in kDa, for example `86`. |
| `polymer_functionalization_percent` | Functionalization as a percentage, for example `88`, not `0.88`. |
| `polymer_concentration_mg_ml` | Polymer concentration in mg/mL. |
| `buffer` | `PBS` or `DI Water`; use these labels rather than an arbitrary buffer name. |
| `heating_rate_c_per_min` | Heating rate in °C/min, retained as input metadata. Heating rate is **not a model feature**, so changing it does not change a prediction. |
| `additive_name` | A registered additive name below, or `No additive`. |
| `additive_concentration_mM` | Additive concentration in mM. |
| `salt_name` | A registered salt name below, or `No salt`. |
| `salt_concentration_mM` | Salt concentration in mM. |

Use these exact chemical names:

- **Additives:** `CHAPS`, `CTAB`, `Pluronic F127`, `SDS`, `Span-85`, `Urea`.
- **Salts:** `CaCl2`, `KCl`, `MgCl2`, `NH4Cl`, `Na2HPO4`, `Na2S2O3`, `Na2SO4`, `NaBr`, `NaCl`, `NaI`, `NaNO3`, `NaSCN`.

For an absent ingredient, use the explicit `No polymer`, `No additive`, or `No salt` name and a concentration of `0`. For `No polymer`, also set its molecular weight and functionalization to `0`. A named, present ingredient must have a positive concentration when its concentration is known. A blank numeric value means missing information, not absence; supply measured values wherever available. Each row represents at most one additive and one salt, in addition to the specified buffer.

### Read the predictions

The output preserves the canonical input columns and adds columns prefixed by each selected model's identifier:

| Suffix | Meaning |
| --- | --- |
| `_lcst_c` | Predicted LCST temperature in °C. |
| `_prob_lcst` | Probability of an LCST outcome, from 0 to 1. |
| `_prob_ucst` | Probability of a UCST outcome, from 0 to 1. |
| `_prob_none` | Probability of no observed transition, from 0 to 1. |
| `_predicted_class` | Class with the largest predicted probability. |

The regressor always returns a number and does not decide whether a formulation has an LCST. Interpret it alongside the classifier. It does not predict UCST temperatures. `NONE` means no transition was observed within the experimental measurement window; it does not establish the absence of a transition at every temperature.

### Available models

The [models/](models/) directory contains eight `.joblib` artifacts: two feature representations × two model families × two tasks. `descriptor24` uses 24 physicochemical and formulation features. `two_slot_pca38` uses 38 features, including separate 16-component MoLFormer representations for the additive and salt. Both include a buffer indicator. `mlp` denotes a multilayer perceptron.

| Representation / model | Regression identifier | Classification identifier |
| --- | --- | --- |
| Descriptors / XGBoost | `descriptor24_buffer_regressor_xgboost` | `descriptor24_buffer_classifier_xgboost` |
| Descriptors / MLP | `descriptor24_buffer_regressor_mlp` | `descriptor24_buffer_classifier_mlp` |
| Embeddings / XGBoost | `two_slot_pca38_buffer_regressor_xgboost` | `two_slot_pca38_buffer_classifier_xgboost` |
| Embeddings / MLP | `two_slot_pca38_buffer_regressor_mlp` | `two_slot_pca38_buffer_classifier_mlp` |

These are final models fitted using all eligible study formulations. Predictions on those same formulations are not independent evaluation results; use the held-out predictions and metrics below to assess performance. Keep the repository code available when loading the artifacts, since they reference classes in `lcst_pipeline`.

## Find the paper's results

You can inspect the published CSV tables without running any models.

| Analysis or resource | Files |
| --- | --- |
| Original measurements and processed data | [Raw Excel workbooks](data/raw/) and [observed measurement table](data/processed/lcst_master_observed.csv). |
| Features and targets for reuse | [Descriptor features](outputs/feature_tables/descriptor_features_labels.csv), [embedding features](outputs/feature_tables/embedding_features_labels.csv), and [feature definitions/provenance](outputs/feature_tables/export_manifest.json). |
| Repeated cross-validation and held-out additive, salt, and molecular-weight challenges | [Metrics summary](outputs/report_evaluation/metrics_summary.csv), [metrics for each seed](outputs/report_evaluation/metrics_by_seed.csv), [held-out predictions](outputs/report_evaluation/predictions_long.csv), and [split inventory](outputs/report_evaluation/split_inventory.csv). |
| Additive and salt adaptation | [Adaptation curves](outputs/report_evaluation/adaptation_curves.csv), [scores for each repeat and acquisition count](outputs/report_evaluation/adaptation_metrics_long.csv), and [panel/pool inventory](outputs/report_evaluation/adaptation_inventory.csv). |
| Molecular-weight adaptation | [Curves, repeated scores, and inventory](outputs/report_evaluation/mw_adaptation/). |
| SI Tables S5 and S6: adaptation endpoints and acquisition requirements | [Domain summaries](outputs/report_evaluation/si_adaptation_endpoints.csv) and [family summaries](outputs/report_evaluation/si_adaptation_family_summary.csv). |
| Contributions of polymer, additive, salt, and buffer | [Grouped Shapley importance](outputs/feature_importance/shapley_importance.csv) and [related numerical results](outputs/feature_importance/). |
| Contributions of polymer molecular weight, functionalization, and concentration | [Hierarchical polymer importance](outputs/polymer_hierarchical_importance/polymer_feature_importance.csv) and [related numerical results](outputs/polymer_hierarchical_importance/). |
| Chemical representations and sources | [Configuration](config/lcst.json) and [descriptor source information](config/chemical_descriptor_sources.json). |

**Match the paper's aggregation.** For the static benchmark and holdout results in `metrics_summary.csv`, use `seed_mean`: the mean of scores calculated separately for the five seeds. The `value` column instead scores predictions averaged across seeds and can differ. The root-level `outputs/regression_metrics.csv` and `outputs/classifier_metrics.csv` belong to the production-training workflow; use `outputs/report_evaluation/` for the paper's evaluation.

For SI Tables S5 and S6, use the two **`si_adaptation_*`** tables. They select embedding XGBoost, regression MAE, and classification accuracy. The other `adaptation_k90_*` tables include different model/metric aggregations and should not be substituted for the SI summaries. Accuracy is stored as a fraction in the general evaluation tables and as a percentage in the SI endpoint table.

The exported feature tables use the full 18-chemical PCA basis and contain one row per pooled formulation, before model-specific imputation and scaling. Chemical-holdout evaluations refit PCA without the held-out chemical; use the supplied evaluation code to reproduce that protocol rather than splitting the exported embedding table directly.

## Reproduce the analyses

Use a separate clone for regeneration so the published models and results remain available for comparison. Training and analysis commands write to `models/` and `outputs/`. Keep the supplied embedding cache and model artifacts in place. Use the environment and thread settings above; numerical libraries and hardware can affect floating-point results.

### 1. Rebuild the measurement table

```bash
python scripts/rebuild_master.py
```

This reads the five raw workbooks using [config/lcst.json](config/lcst.json). The paper uses `data/processed/lcst_master_observed.csv`, which preserves the experimentally observed outcome, including polymer-free controls. The command also creates a local alternative polymer-target table; that alternative is not the input used for the paper's analyses.

### 2. Recalculate the SI adaptation summaries from the saved results

This is a short calculation and does not retrain models:

```bash
python scripts/summarize_si_adaptation.py
```

To write a separate copy for comparison:

```bash
python scripts/summarize_si_adaptation.py --output-dir reproduced_si
```

The calculation checks the saved results against their recorded data hashes, averages ten separately scored adaptation repeats, and calculates the first acquisition count reaching 90% of the isotonic-fitted endpoint improvement (`k90`). Family means give equal weight to each domain. The regression family means are **37.3 additive, 16.0 salt, and 12.0 molecular-weight formulations**, rounded as in the SI. A blank `k90` means there was no fitted improvement; a family mean is blank if any member has an undefined `k90`.

### 3. Retrain the eight prediction models

```bash
python scripts/train_models.py
```

This trains on the included observed dataset, replaces the eight files in `models/`, and writes the two production-training metric tables. MoLFormer is downloaded automatically if needed. This command alone does not rerun the paper's transferability, adaptation, or feature-importance analyses.

### 4. Rerun cross-validation, holdouts, and chemical adaptation

```bash
python scripts/train_models.py --report-evaluation --phase all
```

This runs the five-seed static evaluation, all ten additive/salt adaptation repeats, and compilation of the results. It fits many models and can take substantial time. To run the stages individually instead:

```bash
python scripts/train_models.py --report-evaluation --phase static
for repeat in {0..9}; do
  python scripts/train_models.py --report-evaluation --phase adaptation --repeat "$repeat"
done
python scripts/train_models.py --report-evaluation --phase compile
```

### 5. Rerun molecular-weight adaptation

The molecular-weight script refuses to overwrite an existing run. In your reproduction clone, move the supplied molecular-weight results to a new backup location before running it:

```bash
backup_dir=$(mktemp -d ../lcst-mw-results.XXXXXX)
mv outputs/report_evaluation/mw_adaptation "$backup_dir/"
python scripts/run_mw_adaptation.py --jobs 1
python scripts/summarize_si_adaptation.py
```

This evaluates the 20, 40, 86, and 250 kDa targets with embedding XGBoost, then updates the SI summaries using the regenerated chemical and molecular-weight results. The 500 kDa group is included in static holdouts but has too few formulations for the fixed ten-formulation adaptation evaluation panel.

### 6. Rerun feature-importance analyses

Run the grouped analysis before the hierarchical polymer analysis, which reuses its intermediate results:

```bash
python scripts/run_feature_importance.py all --jobs 1
python scripts/run_polymer_hierarchical_importance.py all --jobs 1
```

These commands regenerate the intermediate model/seed results and compile the published importance tables. Increase `--jobs` if your CPU and memory allow multiple workers. Intermediate files and locally generated plots are not needed to use the released prediction models.

## Data and evaluation conventions

The observed master contains **1,255 measurements**. The dedicated heating-rate series contributes 112 measurements and is excluded from modeling, leaving **1,143 eligible measurements**. Replicates are pooled by formulation **before** evaluation splits, yielding **403 classification formulations** and **272 LCST regression formulations**.

Regression targets average the observed LCST temperatures for a formulation. Classification targets retain the proportions of LCST, UCST, and NONE observations across replicates, including mixed outcomes. Hard-label classification accuracy is evaluated on formulations with unanimous observed classes. Polymer-free controls retain their observed outcomes.

The main benchmark uses five repetitions of shuffled ten-fold cross-validation, with seeds 42–46. Transferability analyses hold out additive identities, salt identities, or polymer molecular-weight groups. Adaptation starts without the target domain, adds target-domain formulations to training, and measures performance on a fixed ten-formulation panel in each of ten repeats. The inventory and manifest files record the splits, model settings, and data fingerprints.

The dataset does not uniformly cover all combinations of chemistry and formulation. Use the transferability and adaptation results when judging predictions outside familiar conditions; a model accepting an input does not establish its predictive accuracy in that domain.

## Use your own measurements or chemicals

For retraining on another measurement table, follow the column names and outcome conventions in [lcst_master_observed.csv](data/processed/lcst_master_observed.csv), then run in a separate clone:

```bash
python scripts/train_models.py --data path/to/your_measurements.csv
```

This replaces the saved prediction models. The paper-evaluation commands use the repository's observed master and study-specific domains; `--data` only changes production training.

To score a chemical absent from the released models, register its molecular input under `chemical_metadata.molformer.structures` in [config/lcst.json](config/lcst.json). Embedding models generate its MoLFormer representation and apply the PCA transformation stored in the trained artifact. Descriptor models additionally require the corresponding descriptor entry. Select only the embedding models with `--models` if no descriptor entry is available. New chemistry requires independent validation; registering its representation does not retrain the predictive model.

## Cite the study

If you use these data, models, or analyses, please cite the accompanying paper:

Jordan Carden, Erfan Moaseri, Elizabeth Roberge, Linqing Li, and Yaxin An. **Machine Learning of Methacrylated Dextran Phase Separation for Prediction, Transferability, and Adaptation.**

Record the repository commit used for your work so the exact code, data, and model versions can be identified. Questions about running the code or reproducing a result can be raised through [GitHub Issues](https://github.com/JordanCarden/LCST/issues).
