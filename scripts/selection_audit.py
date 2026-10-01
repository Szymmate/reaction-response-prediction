"""Quantify feedback selection and missingness; IPW is an assumption-bound sensitivity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features
import argparse
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from features import SEED, NUMERIC, CATEGORICAL, FeaturePipeline, load_all, clean


def weighted_scores(y, p, weights):
    weights=np.asarray(weights,dtype=float)
    losses=-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))
    brier=((p-np.eye(5)[y])**2).sum(1)
    errors=[]
    for k in range(5):
        bins=np.minimum((p[:,k]*10).astype(int),9)
        error=0.
        for b in range(10):
            mask=bins==b
            if mask.any():
                error+=weights[mask].sum()*abs(np.average(p[mask,k]-(y[mask]==k),weights=weights[mask]))
        errors.append(error/weights.sum())
    return dict(log_loss=float(np.average(losses,weights=weights)),
                brier=float(np.average(brier,weights=weights)),classwise_ece=float(np.mean(errors)),
                effective_event_n=float(weights.sum()**2/(weights**2).sum()),
                max_weight=float(weights.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,default=Path("results/selection_audit"))
    parser.add_argument("--study",type=Path,default=Path("results/model"))
    parser.add_argument("--audit-date",default="2026-10-01")
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    people,events,responses,frame=load_all(args.root/"data")
    labeled=frame.response_category.notna().to_numpy()
    frame["feedback_observed"]=labeled
    frame["age_cohort"]=pd.cut(frame.age,[0,30,40,50,np.inf],right=False).astype(str).fillna("missing")
    frame["age_x_context"]=frame.age_cohort+" / "+frame.context_domain
    coverage=[]
    for c in CATEGORICAL+["age_cohort","wellness_optin","age_x_context"]:
        tab=frame.groupby(c,dropna=False)["feedback_observed"].agg(events="size",labels="sum",coverage="mean").reset_index()
        tab["population_share"]=tab.events/len(frame)
        tab["label_share"]=tab.labels/labeled.sum()
        tab["share_gap"]=tab.label_share-tab.population_share
        coverage.append(tab.rename(columns={c:"group"}).assign(slice=c))
    pd.concat(coverage).to_csv(args.output/"coverage.csv",index=False)
    clean_frame=clean(frame)
    shifts=[]
    for c in NUMERIC+[f"ctx_feat_{i:03d}" for i in range(1,51)]:
        a,b=clean_frame.loc[labeled,c],clean_frame.loc[~labeled,c]
        sd=np.sqrt((a.var()+b.var())/2)
        shifts.append(dict(feature=c,labeled_mean=a.mean(),unlabeled_mean=b.mean(),
                           standardized_mean_difference=(a.mean()-b.mean())/sd if sd>0 else 0,
                           labeled_missing=a.isna().mean(),unlabeled_missing=b.isna().mean(),
                           missing_gap=a.isna().mean()-b.isna().mean()))
    pd.DataFrame(shifts).to_csv(args.output/"feature_shift.csv",index=False)
    # Employee-weighted missingness avoids counting an employee six times on average.
    people["age_cohort"]=pd.cut(people.age,[0,30,40,50,np.inf],right=False).astype(str).fillna("missing")
    missing=[]
    cols=[c for c in NUMERIC if c in people]
    for key in ["age_cohort","job_family","region","wellness_optin"]:
        for value,group in people.groupby(key,dropna=False):
            for c in cols:
                missing.append(dict(slice=key,group=str(value),feature=c,people=len(group),missing=group[c].isna().mean()))
    pd.DataFrame(missing).to_csv(args.output/"conditional_missingness.csv",index=False)
    # This classifier predicts observation, never response_category. The full
    # population supplies X and label availability; outcomes from reserved people
    # are not used for response fitting, selection or evaluation.
    propensity=np.full(len(frame),np.nan)
    observation_folds=np.full(len(frame),-1,dtype=int)
    for fold,(fit,valid) in enumerate(GroupKFold(n_splits=5).split(frame,labeled,groups=frame.person_id)):
        assert set(frame.iloc[fit].person_id).isdisjoint(frame.iloc[valid].person_id)
        feature=FeaturePipeline().fit(frame.iloc[fit])
        model=LGBMClassifier(n_estimators=100,num_leaves=7,min_child_samples=200,
                             learning_rate=.05,reg_lambda=10,n_jobs=1,random_state=SEED,
                             deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(feature.transform(frame.iloc[fit]),labeled[fit].astype(int))
        propensity[valid]=model.predict_proba(feature.transform(frame.iloc[valid]))[:,1]
        observation_folds[valid]=fold
        print(f"Observation model fold {fold+1}/5",flush=True)
    pd.DataFrame(dict(situation_id=frame.situation_id,person_id=frame.person_id,fold=observation_folds,
                      observed=labeled,observation_probability=propensity)).to_csv(args.output/"observation_oof.csv",index=False)
    stamp=pd.to_datetime(responses.observed_at)
    summary=dict(seed=SEED,people=len(people),events=len(frame),labels=int(labeled.sum()),coverage=float(labeled.mean()),
        observation_auc=float(roc_auc_score(labeled,propensity)),observation_log_loss=float(log_loss(labeled,propensity)),
        propensity_quantiles=dict(zip(["min","p01","p05","p50","p95","max"],np.quantile(propensity,[0,.01,.05,.5,.95,1]).tolist())),
        population_propensity_below_01=float((propensity<.01).mean()),
        timestamp_min=str(stamp.min()),timestamp_max=str(stamp.max()),audit_date=args.audit_date,
        future_recorded_outcomes=int((stamp>pd.Timestamp(args.audit_date)+pd.Timedelta(days=1)).sum()),
        inference="Differences establish observable selection, not MAR; observation AUC cannot identify outcome-dependent nonresponse.",
        weighting_assumptions=["Outcomes independent of observation conditional on measured X (unverifiable MAR)",
                               "Positivity in target population", "Accurate observation model and stable label definitions"],
        caveats="No event dates: selection is confounded with unknown maturity. Cross-fitted weights are diagnostic, not a population calibration certificate.")
    if (args.study/"selection.json").exists():
        chosen=json.loads((args.study/"selection.json").read_text())["selected"]
        saved=np.load(args.study/"oof.npz");y=saved["y"];p=saved[f"{chosen}_calibrated"]
        lookup=pd.Series(np.arange(len(frame)),index=frame.situation_id)
        index=lookup.loc[saved["situation_id"]].to_numpy()
        probabilities=propensity[index]
        rows=[dict(method="unweighted",**weighted_scores(y,p,np.ones(len(y))))]
        for floor in [.005,.01,.02]:
            rows.append(dict(method=f"IPW_floor_{floor}",**weighted_scores(y,p,1/np.maximum(probabilities,floor))))
        domain=frame.context_domain
        target=domain.value_counts(normalize=True)
        observed=domain.iloc[index].value_counts(normalize=True)
        weights=domain.iloc[index].map(target/observed).to_numpy()
        rows.append(dict(method="domain_poststratification",**weighted_scores(y,p,weights)))
        pd.DataFrame(rows).to_csv(args.output/"weighted_sensitivity.csv",index=False)
        summary["weighted_model"]=chosen
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
