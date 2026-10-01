# Historical research archive

These are the abandoned denoising-autoencoder, SCARF contrastive, and original soft-pseudo-label experiments discussed in Section 2 of the report. Their matched labeled-only and supervised controls are retained alongside them. None replaces the selected model in `results/model/` at the repository root.

The active selected-contract follow-up stays in `src/compare_training_strategies.py`, with results in `results/revision/`. Its supervised comparison is required evidence for the final training strategy. Feature-ablation arms in `src/train_model.py` also remain active evidence explaining the chosen representation.

## Contents

- `scripts/`: historical training, representations, evaluation and comparison commands.
- `results/`: original and corrected supervised controls, neural/pseudo-label results and comparison summaries.
- `figures/`: historical comparison figure used by Section 2.
- `requirements.txt`: optional research dependencies, including PyTorch.
- `moved_paths.json`: old-to-new locations, plus the historical functions extracted from the active metrics module.

Run all commands from the repository root. Full reproduction commands are in the root README. To recompute comparisons from the saved arrays without neural retraining:

```bash
python archive/historical_research/scripts/compare_unlabeled_models.py \
  --output reproduced/unlabeled_comparison.json
```

Install neural-training dependencies only when rerunning those experiments:

```bash
python -m pip install -r archive/historical_research/requirements.txt
```

The archived results and fit protocols retain their original contents, paths and hashes. Current sources have since changed; `moved_paths.json` records relocated and renamed files. No original experiment is represented as having been fitted by the newly organized source tree.

The report readers still use archived evidence. Include this directory in the submission; removing it requires removing its claims and updating the readers together.
