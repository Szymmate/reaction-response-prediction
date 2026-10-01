"""Train and select the final model using baselines and modality ablations.

Repeated development estimates do not constitute independent confirmation.
"""
import features
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from features import (CLASSES, SEED, QUESTIONS, NUMERIC, CATEGORICAL, FeaturePipeline,
                      digest, inner_assignments, load_development,
                      model_inputs, targets)
from model import MODEL, ResponseModel, fit_temperature, temper
from metrics import metrics, per_class, paired_loss_interval

CONTEXT_N = ["workload_index", "team_size", "manager_support_score"]
CONTEXT_C = ["module_topic", "recommended_intervention", "context_domain", "season"]
DEMOGRAPHICS = CONTEXT_N + ["age", "tenure_months"]
QUESTIONNAIRE = DEMOGRAPHICS + QUESTIONS
PERSONALITY = QUESTIONNAIRE + [f"ocean_{c}" for c in "ENACO"]
FULL_C = CONTEXT_C + ["job_family", "region"]
# Rank is a declared minimization preference, not an estimate of legal sensitivity.
ARMS = {
    "prior": (CONTEXT_N, CONTEXT_C, 0),
    "context": (CONTEXT_N, CONTEXT_C, 1),
    "demographics": (DEMOGRAPHICS, FULL_C, 2),
    "questionnaire": (QUESTIONNAIRE, FULL_C, 3),
    "personality_no_age": ([c for c in PERSONALITY if c != "age"], FULL_C, 4),
    "personality": (PERSONALITY, FULL_C, 5),
    "logistic": (PERSONALITY, FULL_C, 6),
    "wellness": (NUMERIC, CATEGORICAL, 7),
    "anonymous_context": (NUMERIC + [f"ctx_feat_{i:03d}" for i in range(1, 51)], CATEGORICAL, 8),
}
ELIGIBLE = [a for a in ARMS if a not in {"wellness", "anonymous_context"}]
MARGINS = {"log_loss": .01, "brier": .005}


def fit_arm(arm, frame, y):
    numeric, categorical, _ = ARMS[arm]
    pipeline = FeaturePipeline(numeric, categorical).fit(frame)
    x = pipeline.transform(frame)
    if arm == "prior":
        estimator = DummyClassifier(strategy="prior")
    elif arm == "logistic":
        estimator = LogisticRegression(C=.1, max_iter=2000, solver="lbfgs", random_state=SEED)
    else:
        estimator = LGBMClassifier(**MODEL)
    estimator.fit(x, y)
    return ResponseModel(pipeline, estimator)


def inner_predictions(root, arm, frame, fold=None):
    assignment = inner_assignments(root, frame, None if fold is None else 0, fold)
    result = np.full((len(frame), 5), np.nan)
    y = targets(frame)
    for k in range(3):
        fit, valid = assignment != k, assignment == k
        assert set(frame.loc[fit, "person_id"]).isdisjoint(frame.loc[valid, "person_id"])
        model = fit_arm(arm, model_inputs(frame.loc[fit]), y[fit])
        result[valid] = model.predict_uncalibrated(model_inputs(frame.loc[valid]))
    return result


def fit_fold(root, arm, fold):
    frame, _, folds = load_development(root)
    assignment = folds.repeat_0_fold.to_numpy()
    fit, valid = assignment != fold, assignment == fold
    fitting = frame.loc[fit].reset_index(drop=True)
    inner = inner_predictions(root, arm, fitting, fold)
    # A prior is the probability reference; fitting temperature would change its meaning.
    t = 1. if arm == "prior" else fit_temperature(targets(fitting), inner)
    model = fit_arm(arm, model_inputs(fitting), targets(fitting))
    raw = model.predict_uncalibrated(model_inputs(frame.loc[valid]))
    return dict(arm=arm, fold=fold, indices=np.flatnonzero(valid), raw=raw,
                calibrated=temper(raw, t), temperature=t)


def cluster_weights(people, draws=1500):
    codes, unique = pd.factorize(people)
    rng = np.random.default_rng(SEED + 61)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=draws)
    return codes, weights


def reliability_intervals(y, probabilities, people):
    """Employee bootstrap of fixed-bin observed frequencies; unique support is explicit."""
    codes, weights = cluster_weights(people)
    rows = []
    for k in [-1] + list(range(5)):
        confidence = probabilities.max(1) if k == -1 else probabilities[:, k]
        outcome = (probabilities.argmax(1) == y) if k == -1 else (y == k)
        assignment = np.minimum((confidence * 10).astype(int), 9)
        for b in range(10):
            mask = assignment == b
            if not mask.any():
                continue
            counts = np.bincount(codes, weights=mask.astype(float), minlength=weights.shape[1])
            positives = np.bincount(codes, weights=(mask & outcome).astype(float), minlength=weights.shape[1])
            denominator = weights @ counts
            numerator = weights @ positives
            estimate = numerator[denominator > 0] / denominator[denominator > 0]
            rows.append(dict(class_name="top_label" if k == -1 else CLASSES[k], bin=b,
                             events=int(mask.sum()), people=int(np.unique(codes[mask]).size),
                             supported=bool(mask.sum()>=100 and np.unique(codes[mask]).size>=50),
                             predicted=float(confidence[mask].mean()), observed=float(outcome[mask].mean()),
                             lower=float(np.quantile(estimate, .025)), upper=float(np.quantile(estimate, .975))))
    return pd.DataFrame(rows)


def slice_groups(frame):
    age = pd.cut(frame.age, [0, 30, 40, 50, np.inf], right=False,
                 labels=["<30", "30–39", "40–49", "50+"]).astype(object).fillna("missing")
    return {"age": age, "context_domain": frame.context_domain,
            "wellness_optin": frame.wellness_optin.astype(str),
            "intervention": frame.recommended_intervention,
            "age_x_context": age.astype(str) + " / " + frame.context_domain.astype(str)}


def policy_metrics(y, probabilities, threshold=.30):
    """Shadow-only default-intervention review, never alternative ranking.

    A flag means P(completed_effective) < .30. Every other outcome is counted
    as 'not effective completion' for this diagnostic, not as equal business cost.
    """
    flag = probabilities[:, 0] < threshold
    adverse = y != 0
    return dict(events=len(y), flags=int(flag.sum()), flag_rate=float(flag.mean()),
                flagged_not_effective_rate=float(adverse[flag].mean()) if flag.any() else None,
                not_effective_recall=float(flag[adverse].mean()) if adverse.any() else None,
                false_flag_rate=float(flag[~adverse].mean()) if (~adverse).any() else None,
                threshold=threshold)


def evaluate_study(root, output, frame, arrays):
    y = targets(frame)
    rows, slices, classes = [], [], []
    for arm in ARMS:
        for variant in ["raw", "calibrated"]:
            p = arrays[f"{arm}_{variant}"]
            rows.append(dict(arm=arm, variant=variant, **metrics(y, p)))
            classes.append(per_class(y, p).assign(arm=arm, variant=variant))
            if variant == "calibrated":
                for name, values in slice_groups(frame).items():
                    for value in sorted(values.unique()):
                        mask = values.eq(value).to_numpy()
                        slices.append(dict(arm=arm, slice=name, group=value, events=int(mask.sum()),
                                           people=frame.loc[mask, "person_id"].nunique(), **metrics(y[mask], p[mask])))
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(slices).to_csv(output / "slices.csv", index=False)
    pd.concat(classes).to_csv(output / "per_class.csv", index=False)
    calibrated = table[table.variant.eq("calibrated")].set_index("arm")
    best = calibrated.loc[ELIGIBLE].log_loss.idxmin()
    candidates = []
    for arm in ELIGIBLE:
        comparison = paired_loss_interval(y, arrays[f"{best}_calibrated"][None],
                                         arrays[f"{arm}_calibrated"][None], frame.person_id, draws=1500)
        close = all(comparison[m]["ci95"][1] <= margin for m, margin in MARGINS.items())
        candidates.append(dict(arm=arm, within_development_margin=bool(close), comparison=comparison))
    selected = min([r["arm"] for r in candidates if r["within_development_margin"]], key=lambda a: ARMS[a][2])
    selection = dict(selected=selected, best_eligible_log_loss=best,
        research_best_log_loss=calibrated.log_loss.idxmin(), margins=MARGINS, candidates=candidates,
        selection_status="development-only; post-selection scores are optimistic, not a release test",
        excluded_from_artifact_selection={"wellness":"purpose/lawful basis and added-value review unresolved",
            "anonymous_context":"definitions and feature-as-of timing unresolved"},
        explanation="Prefer the lowest declared feature-burden rank whose paired upper interval is within both margins of the best eligible log-loss model. Margins are engineering proposals, not approved risk tolerances.")
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    comparisons = []
    for before, after in [("context", "demographics"), ("demographics", "questionnaire"),
                          ("questionnaire", "personality"), ("personality_no_age", "personality"),
                          ("personality", "wellness"), ("wellness", "anonymous_context"), ("prior", selected)]:
        d = paired_loss_interval(y, arrays[f"{before}_calibrated"][None], arrays[f"{after}_calibrated"][None], frame.person_id, draws=1500)
        comparisons.append(dict(before=before, after=after, comparison=d))
    (output / "ablations.json").write_text(json.dumps(comparisons, indent=2))
    p = arrays[f"{selected}_calibrated"]
    intervals = reliability_intervals(y, p, frame.person_id)
    intervals.to_csv(output / "calibration_bins.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, CLASSES + ["top_label"]):
        d = intervals[intervals.class_name.eq(name)]
        ax.plot([0,1], [0,1], "--", color="gray", lw=1)
        enough=d[d.supported]
        sparse=d[~d.supported]
        ax.errorbar(enough.predicted, enough.observed, yerr=np.maximum(np.vstack([enough.observed-enough.lower,enough.upper-enough.observed]),0),
                    fmt="o-", color="#087f8c", capsize=3)
        ax.scatter(sparse.predicted,sparse.observed,facecolors="none",edgecolors="#8a9a9d",label="Sparse: descriptive only")
        ax.set(xlim=(0,1), ylim=(0,1), title=name, xlabel="Predicted probability", ylabel="Observed fraction (95% interval)")
    axes.flat[0].legend(fontsize=8)
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)
    policy = [dict(slice="all", group="all", **policy_metrics(y,p))]
    slice_intervals = []
    for name, values in slice_groups(frame).items():
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            policy.append(dict(slice=name, group=value, **policy_metrics(y[mask],p[mask])))
            if name in {"age", "context_domain", "wellness_optin"}:
                rel = reliability_intervals(y[mask],p[mask],frame.loc[mask,"person_id"])
                slice_intervals.append(rel.assign(slice=name,group=value))
    pd.DataFrame(policy).to_csv(output / "policy.csv",index=False)
    pd.concat(slice_intervals).to_csv(output / "slice_calibration_bins.csv",index=False)
    near = np.abs(p[:,0]-.30) <= .10
    (output / "policy.json").write_text(json.dumps(dict(threshold=.30,
        threshold_status="Fixed shadow diagnostic; requires prospective validation and cost/capacity approval",
        capacity_limit=.20, observed_flag_rate=policy[0]["flag_rate"], within_capacity=policy[0]["flag_rate"]<=.20,
        near_threshold_events=int(near.sum()), near_threshold_people=frame.loc[near,"person_id"].nunique(),
        alternatives="No alternative ranking or automatic reassignment", approved_for_decisions=False),indent=2))
    return selected


def benchmark(model, frame):
    # Covers in-process schema cleaning, transform and inference; excludes I/O/network.
    rows = []
    for n in [1, 100, 1000]:
        sample = model_inputs(frame.iloc[:n])
        model.predict_proba(sample)
        times=[]
        for _ in range(30):
            start=time.perf_counter(); model.predict_proba(sample); times.append((time.perf_counter()-start)*1000)
        rows.append(dict(rows=n, p50_ms=float(np.median(times)), p95_ms=float(np.quantile(times,.95)), repetitions=30))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,default=Path("results/model"))
    parser.add_argument("--workers",type=int,default=3,choices=range(1,5))
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=SEED,split_manifest_sha256=digest(args.root/"results/splits/manifest.json"),created_utc=datetime.now(timezone.utc).isoformat(),arms={a:dict(numeric=n,categorical=c,rank=r) for a,(n,c,r) in ARMS.items()},
        primary_metric="multiclass log_loss", secondary=["multiclass Brier","classwise calibration","cohort diagnostics"],
        selection_margins=MARGINS, selection_population=ELIGIBLE, model=MODEL,
        evaluation="Seed-specific repeat 0: five employee-disjoint outer folds, three inner calibration folds",
        provenance="Repeated development on supplied data. This protocol is saved before fitting; a new seed does not create independent evidence.",
        known_holdout_exposure="User confirmed some original reserved-partition results were inspected; no historical split is certified untouched.",
        formal_confirmation=False, source_hashes={p.name:digest(p) for p in (args.root/"src").glob("*.py")})
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    frame,_,_=load_development(args.root)
    parts=joblib.Parallel(n_jobs=args.workers)(joblib.delayed(fit_fold)(args.root,arm,f) for arm in ARMS for f in range(5))
    arrays={f"{arm}_{variant}":np.full((len(frame),5),np.nan) for arm in ARMS for variant in ["raw","calibrated"]}
    temperatures=[]
    for part in parts:
        for variant in ["raw","calibrated"]:
            arrays[f"{part['arm']}_{variant}"][part["indices"]]=part[variant]
        temperatures.append({k:part[k] for k in ["arm","fold","temperature"]})
    for p in arrays.values():
        assert np.isfinite(p).all() and (p>=0).all(); np.testing.assert_allclose(p.sum(1),1)
    np.savez_compressed(args.output/"oof.npz",**arrays,y=targets(frame),person_id=frame.person_id.to_numpy(str),situation_id=frame.situation_id.to_numpy(str),class_order=np.array(CLASSES))
    pd.DataFrame(temperatures).to_csv(args.output/"temperatures.csv",index=False)
    selected=evaluate_study(args.root,args.output,frame,arrays)
    inner=inner_predictions(args.root,selected,frame)
    t=1. if selected=="prior" else fit_temperature(targets(frame),inner)
    model=fit_arm(selected,model_inputs(frame),targets(frame)); model.temperature=t
    joblib.dump(model,args.output/"model.joblib",compress=3)
    (args.output/"benchmark.json").write_text(json.dumps(benchmark(model,frame),indent=2))
    lock=dict(arm=selected,model_sha256=digest(args.output/"model.joblib"),temperature=t,class_order=CLASSES,
        seed=SEED, trained_on="seed-specific training partition only", feature_contract={"numeric":ARMS[selected][0],"categorical":ARMS[selected][1]},
        intended_use="Research and approved shadow evaluation of default interventions only",approved_for_live_decisions=False,
        evaluation_required="new people absent from the entire historical dataset, pre-outcome predictions, mature labels, adjudicated outcomes and feature-as-of timestamps",
        protocol_sha256=digest(args.output/"protocol.json"))
    (args.output/"model_card.json").write_text(json.dumps(lock,indent=2))
    print(json.dumps(lock,indent=2))

if __name__=="__main__":
    main()
