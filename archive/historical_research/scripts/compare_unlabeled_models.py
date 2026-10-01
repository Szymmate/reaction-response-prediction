"""Recompute probability comparisons from saved, seed-specific development fits."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import features
import argparse
import json
import numpy as np
from metrics import paired_loss_interval
from features import SEED, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,default=Path("archive/historical_research/results/unlabeled_comparison.json"))
    parser.add_argument("--supervised-run",type=Path,default=Path("archive/historical_research/results/supervised_control"))
    parser.add_argument("--unlabeled-run",type=Path,default=Path("archive/historical_research/results/unlabeled_experiments"))
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists")
    semi_run=args.root/args.unlabeled_run
    supervised_run=args.root/args.supervised_run
    for run in [semi_run,supervised_run]:
        if json.loads((run/"protocol.json").read_text())["seed"] != SEED:
            raise ValueError("Comparison inputs must use the configured seed")
    semi=np.load(semi_run/"oof.npz")
    supervised=np.load(supervised_run/"oof.npz")
    for name in ["situation_id","person_id","class_order","y"]:
        np.testing.assert_array_equal(semi[name],supervised[name])
    y=semi["y"];people=semi["person_id"]
    results=[]
    for method in ["dae","scarf","soft"]:
        contrasts={}
        for reference,p in [("matched_labeled_control",semi[f"{method}_labeled"]),
                            ("ordinary_supervised",semi["supervised"]),
                            ("calibrated_supervised",supervised["calibrated"])]:
            contrasts[reference]=paired_loss_interval(y,p,semi[f"{method}_pooled"],people)
        results.append(dict(method=method,contrasts=contrasts))
    report=dict(seed=SEED,status="Repeated development evaluation; no independent confirmation",
        input_sha256=dict(supervised=digest(supervised_run/"oof.npz"),unlabeled=digest(semi_run/"oof.npz")),
        primary_metric="multiclass log_loss",secondary="multiclass Brier; calibration and cohort quality needed before adoption",
        practical_improvement_proposal=.002,
        future_rule="In development: >=.002 mean log-loss improvement, multiplicity-adjusted paired interval below zero, Brier deterioration <=.005, no unacceptable class/cohort calibration loss. Compare against the ordinary calibrated baseline as well as matched controls; then freeze and confirm prospectively.",
        limitations="Research arms were not separately calibrated; conditional intervals are not globally corrected for historical search. No superiority established against a common calibrated comparator. No release decision follows from development estimates.",
        results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
