"""Train the archived supervised research control using fixed employee folds."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import features  # Sets thread limits before importing numerical libraries.
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime, timezone
from features import (
    CLASSES,
    SEED,
    digest,
    inner_assignments,
    load_development,
    model_inputs,
    targets,
)
from model import MODEL, fit_supervised, fit_temperature, temper
from sklearn.dummy import DummyClassifier


def calibration_predictions(root, fitting, repeat=None, fold=None):
    assignment = inner_assignments(root, fitting, repeat, fold)
    predictions = np.full((len(fitting), 5), np.nan)
    y = targets(fitting)
    for k in range(3):
        a, b = np.flatnonzero(assignment != k), np.flatnonzero(assignment == k)
        assert set(fitting.iloc[a].person_id).isdisjoint(fitting.iloc[b].person_id)
        model = fit_supervised(model_inputs(fitting.iloc[a]), y[a])
        predictions[b] = model.predict_uncalibrated(model_inputs(fitting.iloc[b]))
    return predictions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("archive/historical_research/results/supervised_control"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory; existing results are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=MODEL, seed=SEED,
                    method="scalar temperature scaling of log probabilities", bounds=[.25, 4.],
                    selection="Minimize inner OOF log loss; one predeclared method, no outer selection",
                    repeats=10, outer_folds=5, inner_folds=3, class_order=CLASSES,
                    primary_metric="log_loss", secondary=["Brier", "ECE"],
                    limitations="Repeated development, not fresh confirmation; calibration trained on smaller inner models may transfer imperfectly to full refit",
                    assigned_reserved_populations_scored=False,
                    sources={str(p.relative_to(args.root)): digest(p) for folder in [args.root / "src", args.root / "archive/historical_research/scripts"] for p in folder.glob("*.py")})
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    started = time.monotonic()
    labeled, _, folds = load_development(args.root)
    y = targets(labeled)
    raw, calibrated, dummy = [np.full((10, len(y), 5), np.nan) for _ in range(3)]
    records, gaps = [], []
    from metrics import metrics
    for repeat in range(10):
        assignments = folds[f"repeat_{repeat}_fold"].to_numpy()
        for fold in range(5):
            fit, valid = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
            assert set(labeled.iloc[fit].person_id).isdisjoint(labeled.iloc[valid].person_id)
            fitting = labeled.iloc[fit].reset_index(drop=True)
            inner_p = calibration_predictions(args.root, fitting, repeat, fold)
            temperature = fit_temperature(y[fit], inner_p)
            model = fit_supervised(model_inputs(fitting), y[fit])
            p = model.predict_uncalibrated(model_inputs(labeled.iloc[valid]))
            raw[repeat, valid] = p
            calibrated[repeat, valid] = temper(p, temperature)
            baseline = DummyClassifier(strategy="prior").fit(np.zeros((len(fit), 1)), y[fit])
            np.testing.assert_array_equal(baseline.classes_, np.arange(5))
            dummy[repeat, valid] = baseline.predict_proba(np.zeros((len(valid), 1)))
            np.savez_compressed(args.output / f"inner_r{repeat}_f{fold}.npz", probabilities=inner_p, y=y[fit], situation_id=fitting.situation_id.to_numpy(str))
            records.append(dict(repeat=repeat, fold=fold, temperature=temperature, fit_events=len(fit), validation_events=len(valid)))
            for population, truth, prob in [("fitting", y[fit], model.predict_uncalibrated(model_inputs(fitting))), ("outer", y[valid], p)]:
                gaps.append(dict(repeat=repeat, fold=fold, population=population, **metrics(truth, prob)))
        print(f"calibration repeat {repeat+1}/10: raw={metrics(y, raw[repeat])['log_loss']:.6f}; calibrated={metrics(y, calibrated[repeat])['log_loss']:.6f}", flush=True)
    np.testing.assert_array_equal(raw.argmax(2), calibrated.argmax(2))
    final_inner = calibration_predictions(args.root, labeled)
    temperature = fit_temperature(y, final_inner)
    model = fit_supervised(model_inputs(labeled), y)
    model.temperature = temperature
    joblib.dump(model, args.output / "model.joblib", compress=3)
    np.savez_compressed(args.output / "final_inner.npz", probabilities=final_inner, y=y)
    np.savez_compressed(args.output / "oof.npz", raw=raw, calibrated=calibrated, dummy=dummy, y=y,
                        situation_id=labeled.situation_id.to_numpy(str), person_id=labeled.person_id.to_numpy(str), class_order=np.array(CLASSES))
    pd.DataFrame(records).to_csv(args.output / "temperatures.csv", index=False)
    pd.DataFrame(gaps).to_csv(args.output / "overfitting.csv", index=False)
    (args.output / "fit.json").write_text(json.dumps(dict(final_temperature=temperature, seconds=time.monotonic()-started,
        outer_fits=50, inner_fits=153, final_fits=1, artifact_sha256=digest(args.output / "model.joblib")), indent=2))
    print(f"Saved calibrated model, T={temperature:.6f}", flush=True)

if __name__ == "__main__":
    main()
