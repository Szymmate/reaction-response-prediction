# ReAction - Final submission

Read [report.pdf](report.pdf) or [report.md](report.md) for the complete report covering Tasks 1–4, including the supervised-only versus unlabeled-data comparison in Section 2. The supplied brief is preserved in [assignment.md](assignment.md).

**Experiment seed:** `20261001`, defined once in `src/features.py`. Employee partitions, model initialization and resampling use this master seed or documented derived seeds.

The assignment requires five outcome probabilities per employee-situation pair, a documented feature pipeline, a supervised/unlabeled-data comparison, calibration and cohort evaluation, and a deployment memo. The written report covers all four tasks. Code for Tasks 1-2 is in `src/`, with the training strategy and primary comparison in Section 2 of the report and numerical evaluation evidence in `results/`. Eight PNG plots and diagrams illustrate the report.

**Selected model:** temperature-scaled LightGBM using named event context, tenure, job family, region, engagement questionnaire and OCEAN scores. It requires no age, burnout, wellness opt-in, wearable or anonymous context inputs. The artifact is `results/model/model.joblib`; its exact feature contract is in `results/model/model_card.json`.

**Decision:** do not deploy for live intervention decisions. Scores are development estimates. No supplied partition is certified untouched by development, and no independent future cohort has been evaluated. Source timing, outcome definitions, review capacity and governance must be resolved before release.

## Setup

Use Python 3.12. Run all commands from this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch is needed only for the optional neural experiments. Report generation also requires `requirements-report.txt`. The three supplied parquet files are unchanged. Employee partitions were generated using seed `20261001`; data and split hashes are recorded in `results/splits/manifest.json`. Reassigning employees does not create an independent test population.

## Reproduce the main evidence

**Final-model training entry point:** `src/train_model.py`. It compares representations, selects the model and saves the calibrated artifact. `src/compare_training_strategies.py` runs the matched supervised/unlabeled comparison; the archived `train_supervised_control.py` trains only the historical research control.

Choose empty output directories to preserve the supplied results.

```bash
python src/make_splits.py --output reproduced/splits
python src/audit.py --output reproduced/data_audit
python src/train_model.py --output reproduced/model --workers 3
python scripts/selection_audit.py \
  --study reproduced/model \
  --output reproduced/selection_audit \
  --audit-date 2026-10-01
python archive/historical_research/scripts/compare_unlabeled_models.py --output reproduced/unlabeled_comparison.json
python src/compare_training_strategies.py --output reproduced/revision --workers 3
```

`make_splits.py` reproduces the supplied assignments. The training scripts read the supplied `results/splits/` files and verify their hashes and seed before fitting. Partition membership uses a deterministic employee hash; outer and inner folds use shuffled stratified employee groups. Exact seed offsets are in the split manifest.

`train_model.py` compares nine candidate models using five employee-separated outer folds and three inner folds for calibration. It writes probability metrics, paired comparisons, class/cohort diagnostics, calibration bins, model selection, an inference benchmark and the fitted artifact. Selection uses development outcomes; it is not independent confirmation. Timing measurements depend on hardware.

`selection_audit.py` measures feedback coverage, missingness and observable selection. Its weighting results are sensitivity analyses with unverified assumptions. `compare_unlabeled_models.py` recomputes probability comparisons from the included supervised and semi-supervised prediction arrays. The selected-model study and selected-contract follow-up use one outer-fold repeat; the historical unlabeled experiments use ten repeats. Their scores are not one common selection experiment. `compare_training_strategies.py` compares supervised, matched soft-target, pooled soft-target and person-only models on the selected feature contract. Every arm receives identical nested employee folds and its own temperature calibration. The supervised predictions must exactly match the selected-model study.


## Rebuild figures and report

```bash
python -m pip install -r requirements-report.txt
python scripts/report_figures.py
python scripts/build_report.py
```

These commands read the supplied result files, verify a common experiment seed, write seven PNG figures to `figures/` and the historical comparison PNG to `archive/historical_research/figures/`, and build `report.md` and `report.pdf` from `scripts/report_content.py`. The Markdown and PDF reports include all four tasks. Task 2 appears before Task 3, and its primary comparison precedes the historical experiments. Edit `scripts/report_content.py` for durable narrative changes; rebuilding regenerates both report formats. The PDF uses bundled fonts. Result-dependent report values are loaded from CSV/JSON files. Values are rounded only for display; the result files retain full precision. Rendering and visual inspection are still required after editing the narrative.

## Archived research experiments

The abandoned denoising, SCARF and original soft-target branches are preserved in [archive/historical_research/](archive/historical_research/README.md), together with their supervised controls and evidence. The selected-contract comparison in `src/compare_training_strategies.py` remains active. The archive preserves reproducibility without mixing retired entry points into `src/`.

```bash
python archive/historical_research/scripts/train_supervised_control.py --output reproduced/supervised_control
python archive/historical_research/scripts/evaluate_research.py --kind supervised --run reproduced/supervised_control
python -m pip install -r archive/historical_research/requirements.txt
python archive/historical_research/scripts/train_unlabeled_models.py --output reproduced/unlabeled_experiments --workers 3
python archive/historical_research/scripts/evaluate_research.py --kind ssl --run reproduced/unlabeled_experiments
python archive/historical_research/scripts/compare_unlabeled_models.py \
  --supervised-run reproduced/supervised_control \
  --unlabeled-run reproduced/unlabeled_experiments \
  --output reproduced/unlabeled_comparison_from_new_fits.json
```

The newly fitted supervised run corresponds to `archive/historical_research/results/supervised_control_revised/`, which uses a class-frequency probability baseline. `archive/historical_research/results/supervised_control/` is retained as historical evidence; its `dummy` arm is the former deterministic majority classifier with infinite log loss. Do not use that historical arm as a probability benchmark. Raw/calibrated predictions are unchanged by the baseline correction. `archive/historical_research/results/unlabeled_comparison_revised.json` repeats the historical comparisons against the corrected run.

The research experiments compare denoising features, contrastive features and soft pseudo-labels with matched labeled-only controls. These arms use their documented research feature contract; they do not replace the selected artifact. Probability loss drives the screening rule. Separate calibration and independent evaluation are required before adoption. The supplied prediction arrays support recalculation of every reported unlabeled comparison without rerunning neural training.

## Batch inference

```bash
python src/predict.py \
  --model results/model/model.joblib \
  --persons data/persons.parquet \
  --situations data/situations.parquet \
  --output predictions.csv
```

Output contains identifiers and five probabilities in the fixed order **completed_effective**, **completed_ineffective**, **partial**, **dropped_out**, **declined**. Each row sums to one. The pipeline rejects out-of-range schema measurements, non-discrete questionnaire/team values and missing or unknown intervention, domain, topic or season values. Legitimate numeric missingness is imputed; novel job families and regions are encoded as unknown categories and should trigger serving telemetry. The CLI fails the batch on validation errors; the proposed service wrapper must retain the default recommendation and record the failure. This command forecasts outcomes; it does not establish the benefit of switching interventions.

## Evidence map

| Location | Contents |
| --- | --- |
| `data/` | Unmodified supplied parquet files |
| `src/` | Tasks 1-2: data audit, feature pipeline, training, inference and held-out model evaluation |
| `scripts/` | Task 3 representativeness audit and report builders |
| `report.md` / `report.pdf` | Complete report for Tasks 1–4, including the Task 2 training comparison |
| `results/data_audit/` | Distributions, correlations and source integrity |
| `results/model/` | Selected model, baselines, ablations, calibration, cohorts and policy diagnostics |
| `results/selection_audit/` | Coverage, missingness, observation model and weighting sensitivity |
| `archive/historical_research/results/supervised_control/` | Historical supervised comparator, including obsolete deterministic dummy |
| `archive/historical_research/results/supervised_control_revised/` | Refit with class-frequency baseline; same raw/calibrated predictions |
| `results/revision/` | Matched selected-contract comparison, context ablation and cohort uncertainty |
| `archive/historical_research/results/unlabeled_experiments/` | Research arms, matched controls and prediction arrays |
| `archive/historical_research/results/unlabeled_comparison.json` | Paired probability-score comparisons |
| `results/splits/` | Seed-specific employee partitions and data/split hash manifest |
| `figures/` | Report plots and diagrams as PNG |


**LLM assistance:** LLM assistance was used for analysis, code development, debugging, and report drafting.

## Revision decisions and evaluation limitations

The original selected artifact remains fixed after input-validation hardening. Supervised learning is retained because the matched follow-up does not establish improvement over it.

The employee allocation contains 3,318 training, 899 historically named tune, 573 calibration and 797 test labels. Only the training allocation is used for fitting, nested calibration and OOF evaluation. The other 2,269 labels stay excluded to keep comparisons on the existing population, not because those labels are clean independent evidence. This is a deliberate data-efficiency tradeoff during artifact maintenance. An eventual all-label refit must be versioned separately and validated externally; the present scores would not evaluate that new artifact.

Historical fit protocols are preserved, including paths that predate moving report utilities from `src/` to `scripts/`. They are records of earlier source snapshots, not hashes of today's tree. `archive/historical_research/moved_paths.json` records relocated and renamed files. Current code includes input-contract fixes, the corrected baseline and reporting changes since the historical fits. Original data, split files and selected model remain hash-identical. Reproduction may change timestamps, recorded source hashes and hardware timings; prediction arrays and metric values are the numerical targets.

The report now includes per-class recall, age/domain discrimination and calibration, employee-bootstrap loss intervals, raw/scaled uncertainty, interpreted wearable missingness, explicit DPIA determination and drift/feedback monitoring. No independent deployment evaluation has been claimed.

After editing the report source, run the figure and report commands above and inspect every PDF page. Package the whole folder, including data, source code and results. The PDF contains all four tasks. Include `archive/historical_research/` because the report cites its evidence. Exclude `.venv`, `__pycache__`, `reproduced` and existing ZIP delivery copies.
