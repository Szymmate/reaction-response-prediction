"""Compare three methods with matched labeled-only and eligible-unlabeled controls."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import features  # Sets thread limits before importing numerical libraries.
import argparse
import joblib
import json
import numpy as np
import time
from datetime import datetime, timezone
from metrics import metrics
from features import (
    FeaturePipeline,
    SEED,
    clean,
    digest,
    load_development,
    model_inputs,
    targets,
)
from model import ResponseModel, fit_soft_lightgbm, fit_supervised
from representations import (
    AugmentedRepresentation,
    AutoencoderInputs,
    ContrastiveEncoder,
    LearnedEncoder,
    PersonInputs,
)


ARMS = ["supervised", "dae_labeled", "dae_pooled", "scarf_labeled", "scarf_pooled", "soft_labeled", "soft_pooled"]

def fit_partition(root, output, repeat, fold):
    """A fresh process owns each boundary, including its random seeds."""
    started = time.monotonic()
    labeled, pool, folds = load_development(root)
    assignments = folds[f"repeat_{repeat}_fold"].to_numpy()
    fit, valid = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
    excluded = set(labeled.iloc[valid].person_id)
    eligible = pool[~pool.person_id.isin(excluded)].reset_index(drop=True)
    assert set(eligible.person_id).isdisjoint(excluded)
    assert set(labeled.iloc[fit].person_id) <= set(eligible.person_id)
    y = targets(labeled.iloc[fit])
    training = clean(model_inputs(labeled.iloc[fit]))
    validation = clean(model_inputs(labeled.iloc[valid]))
    source = clean(model_inputs(eligible))
    unlabeled = source.loc[eligible.response_category.isna()].reset_index(drop=True)
    # Reuse exactly the same labeled-fit transformation and downstream parameters.
    base = FeaturePipeline().fit(training)
    supervised = fit_supervised(training, y, base)
    predictions = {"supervised": supervised.predict_uncalibrated(validation)}
    training_predictions = {"supervised": supervised.predict_uncalibrated(training)}
    models = {"supervised": supervised}
    histories = {}
    inputs = AutoencoderInputs().fit(training)
    for arm, representation_source in [("dae_labeled", training), ("dae_pooled", source)]:
        # Event-level DAE includes named situation features, unlike static-profile SCARF.
        encoder = LearnedEncoder(inputs, latent=16, seed=SEED).fit(representation_source, updates=2000, batch_size=512)
        features = AugmentedRepresentation(base, encoder)
        model = fit_supervised(training, y, features)
        models[arm] = model
        predictions[arm] = model.predict_uncalibrated(validation)
        training_predictions[arm] = model.predict_uncalibrated(training)
        histories[arm] = encoder.history
    inputs = PersonInputs().fit(training)
    for arm, representation_source in [("scarf_labeled", training), ("scarf_pooled", source)]:
        encoder = ContrastiveEncoder(inputs, SEED).fit(representation_source).checkpoint(200)
        features = AugmentedRepresentation(base, encoder)
        model = fit_supervised(training, y, features)
        models[arm] = model
        predictions[arm] = model.predict_uncalibrated(validation)
        training_predictions[arm] = model.predict_uncalibrated(training)
    # Full probability targets, not argmax labels. Match the custom objective in B/C.
    # A native-LightGBM-vs-custom comparison cannot attribute its difference to U.
    x, xu = base.transform(training), base.transform(unlabeled)
    q = supervised.predict_uncalibrated(unlabeled)
    onehot = np.eye(5)[y]
    for arm, inputs_x, targets_q, weights in [
        ("soft_labeled", x, onehot, np.ones(len(y))),
        ("soft_pooled", np.vstack([x, xu]), np.vstack([onehot, q]),
         np.r_[np.ones(len(y)), np.full(len(xu), .3*len(y)/len(xu))])]:
        estimator = fit_soft_lightgbm(inputs_x, targets_q, weights, y, dict(num_leaves=7, n_estimators=100))
        model = ResponseModel(base, estimator)
        models[arm] = model
        predictions[arm] = model.predict_uncalibrated(validation)
        training_predictions[arm] = model.predict_uncalibrated(training)
    output = Path(output)
    prefix = output / f"r{repeat}_f{fold}"
    np.savez_compressed(str(prefix)+".npz", **predictions, indices=valid)
    diagnostics = []
    for arm in ARMS:
        diagnostics.append(dict(arm=arm, population="fitting", **metrics(y, training_predictions[arm])))
        diagnostics.append(dict(arm=arm, population="outer", **metrics(targets(labeled.iloc[valid]), predictions[arm])))
    metadata = dict(repeat=repeat, fold=fold, seconds=time.monotonic()-started,
                    fit_events=len(fit), valid_events=len(valid), eligible_events=len(eligible),
                    unlabeled_events=len(unlabeled), held_out_people=len(excluded), employee_overlap=0,
                    source_person_hash=digest_ids(source.person_id), fit_person_hash=digest_ids(training.person_id),
                    validation_person_hash=digest_ids(validation.person_id),
                    teacher_temperature=1., pseudo_weight=.3*len(y), histories=histories, diagnostics=diagnostics)
    Path(str(prefix)+".json").write_text(json.dumps(metadata, indent=2))
    # One representative fitted bundle supports serialization/inference review.
    # Every other fold is reproducible from the immutable protocol and assignments.
    if repeat == 0 and fold == 0:
        joblib.dump(models, output / "example_fold_models.joblib", compress=3)
    print(f"three-method replication r{repeat} f{fold}: {metadata['seconds']:.1f}s", flush=True)
    return metadata

def digest_ids(values):
    import hashlib
    return hashlib.sha256("\n".join(sorted(set(values))).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("archive/historical_research/results/unlabeled_experiments"))
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Seed-specific development replication; not independent confirmation",
        arms=ARMS, repeats=10, outer_folds=5, seed=SEED, downstream="Fixed unweighted LightGBM hyperparameters; research feature contract includes wellness",
        selection="No settings/checkpoints selected: all fixed before this replication. Historical exploratory development is acknowledged.",
        dae=dict(latent=16, steps=2000, batch=512, numeric_corruption=.15, categorical_corruption=.1, event_level=True),
        scarf=dict(latent=8, steps=200, batch=128, corruption=.3, temperature=.2, static_unique_people=True),
        soft=dict(teacher="raw supervised LightGBM", teacher_temperature=1., all_eligible_unlabeled=True,
                  total_pseudo_weight="0.3 times true-label weight", controls="Identical custom soft-CE learner B and C"),
        primary_comparisons=["dae_pooled - dae_labeled", "scarf_pooled - scarf_labeled", "soft_pooled - soft_labeled"],
        primary_metric="multiclass log_loss",
        success="Preliminary screen: log-loss delta <= -.002 and 98.333% upper bound < 0 against both matched and ordinary supervised controls; Brier 95% upper delta <= .005. F1 is descriptive. Common calibration, cohort safeguards and independent evaluation are required before adoption.",
        uncertainty="6000 paired employee bootstrap; separate new replication family, no global search correction",
        assigned_reserved_populations_scored=False,
        sources={str(p.relative_to(args.root)): digest(p) for folder in [args.root / "src", args.root / "archive/historical_research/scripts"] for p in folder.glob("*.py")})
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = joblib.Parallel(n_jobs=args.workers, backend="loky")(
        joblib.delayed(fit_partition)(args.root, args.output, r, f) for r in range(10) for f in range(5))
    (args.output / "completed.json").write_text(json.dumps(dict(partitions=len(records), cpu_seconds=sum(r["seconds"] for r in records)), indent=2))

if __name__ == "__main__":
    main()
