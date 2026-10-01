"""Matched calibration and feature-contract follow-up; development evidence only.

Keeps the original population/artifact fixed. Each inner SSL boundary excludes
both outer-held-out and inner-held-out employees, including their unlabeled X.
"""
import features
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import (CLASSES, SEED, FeaturePipeline, digest,
                      load_development, inner_assignments, model_inputs, targets)
from model import ResponseModel, fit_supervised, fit_soft_lightgbm, fit_temperature, temper
from train_model import ARMS, CONTEXT_N, cluster_weights, slice_groups, benchmark
from metrics import metrics, paired_loss_interval

NAMES = ["supervised", "soft_labeled", "soft_pooled", "person_only"]


def fit_boundary(fitting, pool, excluded):
    numeric, categorical, _ = ARMS["personality_no_age"]
    assert not set(fitting.person_id) & excluded
    eligible = pool[~pool.person_id.isin(excluded)]
    unlabeled = model_inputs(eligible[eligible.response_category.isna()])
    assert not set(unlabeled.person_id) & excluded
    xframe, y = model_inputs(fitting), targets(fitting)
    pipeline = FeaturePipeline(numeric, categorical).fit(xframe)
    teacher = fit_supervised(xframe, y, pipeline)
    models = {"supervised": teacher}
    x, xu = pipeline.transform(xframe), pipeline.transform(unlabeled)
    q = teacher.predict_uncalibrated(unlabeled)
    for name, inputs, distributions, weights in [
        ("soft_labeled", x, np.eye(5)[y], np.ones(len(y))),
        ("soft_pooled", np.vstack([x, xu]), np.vstack([np.eye(5)[y], q]),
         np.r_[np.ones(len(y)), np.full(len(xu), .3 * len(y) / len(xu))]),
    ]:
        estimator = fit_soft_lightgbm(inputs, distributions, weights, y,
                                     dict(num_leaves=7, n_estimators=100))
        models[name] = ResponseModel(pipeline, estimator)
    # Same hyperparameters and calibration; only the situational inputs change.
    person_features = FeaturePipeline([c for c in numeric if c not in CONTEXT_N],
                                     ["job_family", "region"]).fit(xframe)
    models["person_only"] = fit_supervised(xframe, y, person_features)
    boundary = dict(fit_events=len(fitting), fit_people=fitting.person_id.nunique(),
                    unlabeled_events=len(unlabeled), excluded_people=len(excluded),
                    employee_overlap=0)
    return models, boundary


def fit_fold(root, fold):
    frame, pool, folds = load_development(root)
    outer = folds.repeat_0_fold.to_numpy()
    fitting = frame[outer != fold].reset_index(drop=True)
    valid = frame[outer == fold]
    excluded = set(valid.person_id)
    inner = inner_assignments(root, fitting, 0, fold)
    inner_p = {name: np.full((len(fitting), 5), np.nan) for name in NAMES}
    boundaries = []
    for k in range(3):
        held_out = fitting[inner == k]
        models, boundary = fit_boundary(fitting[inner != k], pool,
                                       excluded | set(held_out.person_id))
        boundaries.append(dict(inner_fold=k, **boundary))
        for name, model in models.items():
            inner_p[name][inner == k] = model.predict_uncalibrated(model_inputs(held_out))
    models, boundary = fit_boundary(fitting, pool, excluded)
    boundaries.append(dict(inner_fold=-1, **boundary))
    predictions, temperatures = {}, {}
    for name, model in models.items():
        temperatures[name] = fit_temperature(targets(fitting), inner_p[name])
        predictions[name + "_raw"] = model.predict_uncalibrated(model_inputs(valid))
        predictions[name + "_calibrated"] = temper(predictions[name + "_raw"], temperatures[name])
    print(f"Matched selected-contract fold {fold + 1}/5 complete", flush=True)
    return dict(fold=fold, indices=np.flatnonzero(outer == fold), predictions=predictions,
                temperatures=temperatures, boundaries=boundaries)


def loss_interval(y, p, people):
    codes, weights = cluster_weights(people)
    counts = np.bincount(codes, minlength=weights.shape[1])
    loss = -np.log(p[np.arange(len(y)), y])
    sums = np.bincount(codes, weights=loss, minlength=weights.shape[1])
    estimates = (weights @ sums) / (weights @ counts)
    return np.quantile(estimates, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("results/revision"))
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=SEED, created_utc=datetime.now(timezone.utc).isoformat(),
        status="Post-review development follow-up, not independent confirmation",
        arms=NAMES, feature_contract="personality_no_age; person_only removes all event context",
        repeats=1, outer_folds=5, inner_folds=3, pseudo_weight=.3,
        calibration="Each arm fits its own scalar temperature using identical inner employee folds",
        selection="No artifact reselection. Compare soft_pooled with calibrated supervised and matched soft_labeled controls.",
        limitations="Method chosen after earlier research; conditional employee intervals do not cover historical search or refit uncertainty.",
        sources={str(p.relative_to(args.root)): digest(p) for p in (args.root/"src").glob("*.py")})
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    frame, _, _ = load_development(args.root); y = targets(frame)
    parts = joblib.Parallel(n_jobs=args.workers)(joblib.delayed(fit_fold)(args.root, f) for f in range(5))
    arrays = {name + "_" + variant: np.full((len(frame), 5), np.nan)
              for name in NAMES for variant in ["raw", "calibrated"]}
    temperatures, boundaries = [], []
    for part in parts:
        for key, p in part["predictions"].items():
            arrays[key][part["indices"]] = p
        temperatures.extend(dict(fold=part["fold"], arm=a, temperature=t) for a,t in part["temperatures"].items())
        boundaries.extend(dict(outer_fold=part["fold"], **b) for b in part["boundaries"])
    for p in arrays.values():
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1.)
    # An executable bridge to the previously submitted selected-model results.
    saved = np.load(args.root/"results/model/oof.npz")
    np.testing.assert_array_equal(saved["situation_id"], frame.situation_id)
    for variant in ["raw", "calibrated"]:
        np.testing.assert_array_equal(arrays[f"supervised_{variant}"], saved[f"personality_no_age_{variant}"])
    np.savez_compressed(args.output/"oof.npz", **arrays, y=y,
        person_id=frame.person_id.to_numpy(str), situation_id=frame.situation_id.to_numpy(str), class_order=CLASSES)
    pd.DataFrame(temperatures).to_csv(args.output/"temperatures.csv", index=False)
    pd.DataFrame(boundaries).to_csv(args.output/"boundaries.csv", index=False)
    pd.DataFrame([dict(arm=k, **metrics(y,p)) for k,p in arrays.items()]).to_csv(args.output/"metrics.csv", index=False)
    pairs = [("supervised_calibrated", "soft_pooled_calibrated"),
             ("soft_labeled_calibrated", "soft_pooled_calibrated"),
             ("person_only_calibrated", "supervised_calibrated"),
             ("supervised_raw", "supervised_calibrated")]
    comparisons = [dict(before=a, after=b, **paired_loss_interval(y, arrays[a][None], arrays[b][None], frame.person_id))
                   for a,b in pairs]
    (args.output/"comparisons.json").write_text(json.dumps(comparisons, indent=2))
    slices = []
    for name, groups in slice_groups(frame).items():
        if name not in {"age", "context_domain"}:
            continue
        for group in sorted(groups.unique()):
            mask = groups.eq(group).to_numpy()
            p = arrays["supervised_calibrated"][mask]
            lo, hi = loss_interval(y[mask], p, frame.loc[mask,"person_id"])
            slices.append(dict(slice=name, group=group, events=int(mask.sum()),
                people=frame.loc[mask,"person_id"].nunique(), log_loss_lower=lo, log_loss_upper=hi,
                **metrics(y[mask],p)))
    pd.DataFrame(slices).to_csv(args.output/"slices.csv",index=False)
    # Rebenchmark the frozen artifact with the current hardened input contract.
    timings = benchmark(joblib.load(args.root/"results/model/model.joblib"), frame)
    (args.output/"benchmark.json").write_text(json.dumps(timings, indent=2))
    (args.output/"summary.json").write_text(json.dumps(dict(seed=SEED,
        selected_model_predictions_reproduced_exactly=True, artifact_reselected=False,
        calibration_comparable=True, historical_feature_contract="includes age and wellness",
        followup_feature_contract="excludes age and wellness", labeled_events=len(y)), indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
