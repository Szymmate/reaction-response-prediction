"""Audit joins, source fields, feature invariants and employee boundaries."""
import features  # Sets thread limits before importing numerical libraries.
import argparse
import json
import numpy as np
import pandas as pd
from features import (
    CATEGORICAL,
    CLASSES,
    FeaturePipeline,
    NUMERIC,
    WEARABLE,
    clean,
    digest,
    inner_assignments,
    load_all,
    load_development,
)
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("results/data_audit"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    people, events, responses, joined = load_all(args.root / "data")
    # Independently recover the two source blocks in event order.
    expected = people.set_index("person_id").loc[events.person_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(joined[expected.columns].reset_index(drop=True), expected)
    label_lookup = responses.set_index("situation_id").response_category
    pd.testing.assert_series_equal(joined.response_category, events.situation_id.map(label_lookup), check_names=False)
    cleaned = clean(joined)
    pd.testing.assert_frame_equal(cleaned, clean(cleaned))
    negative = joined.wearable_active_min_30d.lt(0)
    assert cleaned.loc[negative, "wearable_active_min_30d"].isna().all()
    assert cleaned.loc[negative, "invalid_active_minutes"].eq(1).all()
    assert joined.loc[~joined.wellness_optin, WEARABLE].isna().all().all()
    pairs = []
    for a, b in [(1,19), (2,20), (9,17), (10,18)]:
        x, z = people[f"engagement_q{a:02d}"], people[f"engagement_q{b:02d}"]
        both = x.notna() & z.notna()
        assert x[both].eq(z[both]).all()
        pairs.append(dict(first=a, second=b, both_observed=int(both.sum()), different_missingness=int((x.isna()!=z.isna()).sum())))
    labeled, pool, folds = load_development(args.root)
    boundaries = []
    for r in range(10):
        assignment = folds[f"repeat_{r}_fold"].to_numpy()
        assert folds.groupby("person_id")[f"repeat_{r}_fold"].nunique().max() == 1
        for f in range(5):
            fitting = labeled.loc[assignment!=f].reset_index(drop=True)
            held_out = set(labeled.loc[assignment==f].person_id)
            assert set(fitting.person_id).isdisjoint(held_out)
            inner = inner_assignments(args.root, fitting, r, f)
            for i in [-1,0,1,2]:
                excluded = held_out | (set(fitting.loc[inner==i].person_id) if i>=0 else set())
                eligible = pool[~pool.person_id.isin(excluded)]
                fit_people = set(fitting.loc[inner!=i].person_id) if i>=0 else set(fitting.person_id)
                assert fit_people.isdisjoint(excluded)
                assert set(eligible.person_id).isdisjoint(excluded)
                assert fit_people <= set(eligible.person_id)
                boundaries.append(dict(repeat=r, fold=f, inner=i, excluded_people=len(excluded), eligible_rows=len(eligible), overlap=0))
    final_inner = inner_assignments(args.root, labeled)
    for i in range(3):
        a = set(labeled.loc[final_inner!=i].person_id)
        b = set(labeled.loc[final_inner==i].person_id)
        assert a.isdisjoint(b)
    feature = FeaturePipeline().fit(labeled)
    x = feature.transform(labeled.iloc[:40])
    altered = labeled.iloc[:40].copy()
    altered["person_id"] = "arbitrary"
    altered["situation_id"] = "arbitrary"
    altered["response_category"] = "arbitrary"
    altered["observed_at"] = "arbitrary"
    for name in [c for c in altered if c.startswith("ctx_feat_")]:
        altered[name] = 1e10
    np.testing.assert_array_equal(x, feature.transform(altered))
    np.testing.assert_array_equal(x, feature.transform(labeled.iloc[:40][list(reversed(labeled.columns))]))
    cleaned[NUMERIC].describe().to_csv(args.output / "numeric_summary.csv")
    people.isna().mean().rename("missing_fraction").to_csv(args.output / "profile_missingness.csv")
    people.select_dtypes(include="number").corr().to_csv(args.output / "profile_correlations.csv")
    pd.DataFrame(boundaries).to_csv(args.output / "boundaries.csv", index=False)
    report = dict(people=len(people), events=len(events), labels=len(responses), unknown_outcomes=int(joined.response_category.isna().sum()),
                  observed_class_counts=responses.response_category.value_counts().to_dict(),
                  invalid_active_minutes=int(negative.sum()), invalid_active_people=int(people.wearable_active_min_30d.lt(0).sum()),
                  duplicate_question_pairs=pairs, checked_nested_boundaries=203,
                  feature_columns=NUMERIC+CATEGORICAL, encoded_columns=feature.feature_names, class_order=CLASSES,
                  assigned_reserved_populations_scored=False, passed=True,
                  unresolved=["No event/as-of timestamps: cannot establish temporal validity", "Unknown questionnaire coding and wearable derivations",
                              "Selective feedback: outcome missingness is not identified as random", "No team/manager groups for additional dependence checks"],
                  data_sha256={p.name:digest(p) for p in (args.root / "data").glob("*.parquet")})
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ["people","events","labels","unknown_outcomes","passed"]}))

if __name__ == "__main__":
    main()
