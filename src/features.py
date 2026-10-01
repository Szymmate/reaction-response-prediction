"""Validated source joins, fixed employee splits and shared feature engineering."""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20261001
CLASSES = ["completed_effective", "completed_ineffective", "partial", "dropped_out", "declined"]
CATEGORICAL = ["job_family", "region", "module_topic", "recommended_intervention", "context_domain", "season"]
QUESTIONS = [f"engagement_q{i:02d}" for i in range(1, 25)]
WEARABLE = ["wearable_hrv_mean_30d", "wearable_sleep_var_30d", "wearable_steps_mean_30d", "wearable_active_min_30d"]
PERSON_NUMERIC = ["age", "tenure_months"] + [f"ocean_{c}" for c in "ENACO"] + QUESTIONS + ["burnout_composite", "wellness_optin"] + WEARABLE
NUMERIC = PERSON_NUMERIC + ["workload_index", "team_size", "manager_support_score", "invalid_active_minutes"]
FEATURES = NUMERIC + CATEGORICAL

# Validate only declared inputs, so excluded sensitive fields are never required.
# Bounds come from the supplied schema; missing numeric values remain imputable.
BOUNDS = {"age": (22, 64), "tenure_months": (0, 300),
          "burnout_composite": (0, 1), "wellness_optin": (0, 1),
          "workload_index": (0, 1), "team_size": (1, None),
          "manager_support_score": (1, 5),
          **{q: (1, 5) for q in QUESTIONS}}
EVENT_CATEGORIES = {
    "recommended_intervention": {"peer_workshop", "self_paced_video", "coaching_1on1", "simulation_exercise", "reading_pack"},
    "context_domain": {"formal_training", "on_the_job", "external_event"},
    "module_topic": {"negotiation", "public_speaking", "conflict_resolution", "technical_writing", "leadership"},
    "season": {"Q1", "Q2", "Q3", "Q4"},
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique_key(frame, key):
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"Null or duplicate primary key: {key}")

def join_inputs(persons, situations):
    unique_key(persons, "person_id")
    unique_key(situations, "situation_id")
    if not situations.person_id.isin(persons.person_id).all():
        raise ValueError("Orphan situation person_id")
    overlap = (set(persons) & set(situations)) - {"person_id"}
    if overlap:
        raise ValueError(f"Ambiguous overlapping source columns: {overlap}")
    joined = situations.merge(persons, on="person_id", how="left", validate="many_to_one", sort=False)
    np.testing.assert_array_equal(joined.situation_id, situations.situation_id)
    return joined

def load_all(data_dir):
    data_dir = Path(data_dir)
    people, events, responses = [pd.read_parquet(data_dir / f"{name}.parquet") for name in ("persons", "situations", "responses")]
    unique_key(responses, "situation_id")
    if not responses.situation_id.isin(events.situation_id).all():
        raise ValueError("Orphan response situation_id")
    if not responses.response_category.isin(CLASSES).all():
        raise ValueError("Unknown or missing recorded response class")
    frame = join_inputs(people, events).merge(responses, on="situation_id", how="left", validate="one_to_one", sort=False)
    assert len(frame) == len(events)
    assert frame.response_category.notna().sum() == len(responses)
    return people, events, responses, frame

def person_partition(pid):
    u = int(hashlib.sha256(f"{SEED}:{pid}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if u < .60 else "tune" if u < .75 else "calibration" if u < .85 else "test"

def load_development(root):
    """Fit only the training employees in the seed-specific split manifest."""
    root = Path(root)
    manifest = json.loads((root / "results/splits/manifest.json").read_text())
    if manifest.get("seed") != SEED:
        raise ValueError("Split seed does not match the configured experiment seed")
    for relative, expected in manifest["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Input contract changed: {relative}")
    people = pd.read_parquet(root / "data/persons.parquet")
    splits = pd.read_csv(root / "results/splits/person_splits.csv")
    expected = people[["person_id"]].assign(split=people.person_id.map(person_partition))
    pd.testing.assert_frame_equal(splits.sort_values("person_id").reset_index(drop=True), expected.sort_values("person_id").reset_index(drop=True))
    allowed = set(splits.loc[splits.split.eq("train"), "person_id"])
    people = people[people.person_id.isin(allowed)]
    events = pd.read_parquet(root / "data/situations.parquet")
    events = events[events.person_id.isin(allowed)]
    responses = pd.read_parquet(root / "data/responses.parquet")
    responses = responses[responses.situation_id.isin(events.situation_id)]
    unique_key(responses, "situation_id")
    pool = join_inputs(people, events).merge(responses, on="situation_id", how="left", validate="one_to_one")
    labeled = pool[pool.response_category.notna()].reset_index(drop=True)
    folds = pd.read_csv(root / "results/splits/outer_folds.csv")
    pd.testing.assert_frame_equal(labeled[["situation_id", "person_id", "response_category"]], folds.iloc[:, :3])
    counts = dict(labeled_events=len(labeled), labeled_people=labeled.person_id.nunique(),
                  pool_events=len(pool), unlabeled_events=int(pool.response_category.isna().sum()))
    if counts != manifest["development_counts"]:
        raise ValueError("Development population differs from the split manifest")
    return labeled, pool, folds

def targets(frame):
    y = frame.response_category.map(dict(zip(CLASSES, range(5))))
    if y.isna().any():
        raise ValueError("Unknown outcomes must not enter supervised targets")
    return y.to_numpy(int)

def model_inputs(frame):
    return frame.drop(columns=["response_category", "observed_at"], errors="ignore")

def inner_assignments(root, fit, repeat=None, fold=None):
    name = "final_inner_folds.csv" if repeat is None else "inner_folds.csv"
    table = pd.read_csv(Path(root) / "results/splits" / name)
    if repeat is not None:
        table = table[table.outer_repeat.eq(repeat) & table.outer_fold.eq(fold)]
    unique_key(table, "situation_id")
    assert set(table.situation_id) == set(fit.situation_id)
    table = table.set_index("situation_id").loc[fit.situation_id]
    np.testing.assert_array_equal(table.person_id, fit.person_id)
    assignment = table.inner_fold.to_numpy(int)
    assert set(assignment) == {0, 1, 2}
    assert pd.DataFrame({"person": fit.person_id.to_numpy(), "fold": assignment}).groupby("person").fold.nunique().max() == 1
    return assignment


def clean(frame, numeric=None, categorical=None):
    """Idempotent raw-to-clean contract, shared by training and inference."""
    result = frame.copy()
    active = "wearable_active_min_30d"
    numeric = NUMERIC if numeric is None else list(numeric)
    categorical = CATEGORICAL if categorical is None else list(categorical)
    required = set(numeric + categorical) - {"invalid_active_minutes"}
    if "invalid_active_minutes" in numeric:
        required.add(active)
    missing = required - set(result)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    for name in required - set(categorical):
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        if np.isinf(result[name]).any():
            raise ValueError(f"Infinite value in {name}")
        if name in BOUNDS:
            low, high = BOUNDS[name]
            invalid = result[name].lt(low)
            if high is not None:
                invalid |= result[name].gt(high)
            if invalid.any():
                raise ValueError(f"Out-of-range value in {name}: expected {low}..{high}")
        if name in QUESTIONS + ["team_size", "wellness_optin"]:
            if result[name].dropna().mod(1).ne(0).any():
                raise ValueError(f"Expected discrete values in {name}")
    # Preserve the audit flag on repeated cleaning; a cleaned NaN cannot recreate it.
    if active in required:
        previous = result.get("invalid_active_minutes", pd.Series(0., index=result.index))
        if not previous.isin([0, 1]).all():
            raise ValueError("invalid_active_minutes must contain 0 or 1")
        negative = result[active] < 0
        result["invalid_active_minutes"] = (previous.astype(bool) | negative).astype(float)
        result.loc[negative, active] = np.nan
    # Consistent missing category, instead of mixed None/NaN/string dtypes.
    for name in categorical:
        # A novel job/region is tolerable at prediction time. A missing or unknown
        # intervention/domain/topic/season changes the meaning of the event.
        if name in EVENT_CATEGORIES and not result[name].isin(EVENT_CATEGORIES[name]).all():
            raise ValueError(f"Missing or unknown event category in {name}")
        result[name] = result[name].fillna("__missing__").astype(str)
    return result

class FeaturePipeline:
    """Fitted, explicit feature contract; omitted modalities are not required at serving.

    Defaults preserve historical artifacts. A revision arm supplies its own allowlist;
    identity, outcomes, timestamps and arbitrary extra columns remain excluded.
    """
    def __init__(self, numeric=None, categorical=None):
        self.numeric = list(NUMERIC if numeric is None else numeric)
        self.categorical = list(CATEGORICAL if categorical is None else categorical)

    def columns(self):
        # Older serialized transformers predate configurable feature contracts.
        return getattr(self, "numeric", NUMERIC), getattr(self, "categorical", CATEGORICAL)

    def fit(self, frame):
        numeric, categorical = self.columns()
        self.transformer = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        ], sparse_threshold=0)
        self.transformer.fit(clean(frame, numeric, categorical)[numeric + categorical])
        self.feature_names = self.transformer.get_feature_names_out().tolist()
        return self

    def transform(self, frame):
        numeric, categorical = self.columns()
        return self.transformer.transform(clean(frame, numeric, categorical)[numeric + categorical])
