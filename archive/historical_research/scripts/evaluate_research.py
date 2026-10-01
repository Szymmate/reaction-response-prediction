"""Evaluate archived ten-repeat supervised and unlabeled experiments."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import features
import argparse
import json
import numpy as np
import pandas as pd
from features import SEED, CLASSES, load_development, targets
from metrics import reliability, metrics, per_class, paired_loss_interval


def evaluate_supervised(args):
    saved = np.load(args.run / "oof.npz")
    frame, _, _ = load_development(args.root)
    np.testing.assert_array_equal(saved["situation_id"], frame.situation_id)
    np.testing.assert_array_equal(saved["class_order"], CLASSES)
    y = saved["y"]
    rows, classes, slices, bins = [], [], [], []
    ages = pd.cut(frame.age, [0, 30, 40, 50, np.inf], labels=["<30", "30–39", "40–49", "50+"], right=False).astype(object).fillna("missing")
    for arm in ["raw", "calibrated", "dummy"]:
        for repeat, p in enumerate(saved[arm]):
            rows.append(dict(arm=arm, repeat=repeat, **metrics(y, p)))
            classes.append(per_class(y, p).assign(arm=arm, repeat=repeat))
            for name, values in [("age_cohort", ages), ("context_domain", frame.context_domain)]:
                for value in sorted(values.unique()):
                    mask = values.eq(value).to_numpy()
                    slices.append(dict(arm=arm, repeat=repeat, slice=name, group=value, events=int(mask.sum()), **metrics(y[mask], p[mask])))
        bins.append(reliability(np.tile(y, 10), saved[arm].reshape(-1, 5)).assign(arm=arm))
    table = pd.DataFrame(rows)
    table.to_csv(args.run / "repeat_metrics.csv", index=False)
    table.groupby("arm").mean(numeric_only=True).drop(columns="repeat").to_csv(args.run / "metrics.csv")
    pd.concat(classes).to_csv(args.run / "per_class.csv", index=False)
    pd.DataFrame(slices).to_csv(args.run / "slices.csv", index=False)
    rel = pd.concat(bins)
    rel.to_csv(args.run / "calibration_bins.csv", index=False)
    comparisons = paired_loss_interval(y, saved["raw"], saved["calibrated"], saved["person_id"])
    comparisons["macro_f1"] = dict(delta=0., ci95=[0., 0.], reason="Positive scalar scaling preserves argmax exactly")
    comparisons["caveat"] = "Paired employee bootstrap conditional on saved fits. No fresh confirmation or full refit uncertainty. Log loss is the sole new primary metric."
    (args.run / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, CLASSES + ["top_label"]):
        for arm, color in [("raw", "#78899b"), ("calibrated", "#087f8c")]:
            subset = rel[rel.arm.eq(arm) & rel.class_name.eq(name)]
            ax.plot(subset.predicted, subset.observed, color=color, label=arm)
            ax.scatter(subset.predicted, subset.observed, s=8+60*subset.n/subset.n.max(), color=color)
        ax.plot([0, 1], [0, 1], "--", color="#bbbbbb")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name, xlabel="Mean predicted probability", ylabel="Observed fraction")
    axes.flat[0].legend()
    fig.savefig(args.run / "calibration.png", dpi=180)
    plt.close(fig)
    print(table.groupby("arm").mean(numeric_only=True).to_string())
    print(json.dumps(comparisons, indent=2))


ARMS = ["supervised", "dae_labeled", "dae_pooled", "scarf_labeled", "scarf_pooled", "soft_labeled", "soft_pooled"]

def f1_bootstrap(y, predictions, people):
    """Exact average per-repetition F1 in each paired employee bootstrap draw."""
    codes, unique = pd.factorize(people)
    matrices = []
    for probabilities in predictions:
        counts = np.zeros((len(unique), 10, 5, 3))  # TP, predicted support, true support
        for r, p in enumerate(probabilities):
            pred = p.argmax(1)
            np.add.at(counts[:,r,:,0], (codes, y), (pred==y).astype(float))
            np.add.at(counts[:,r,:,1], (codes, pred), 1.)
            np.add.at(counts[:,r,:,2], (codes, y), 1.)
        matrices.append(counts.reshape(len(unique), -1))
    combined = np.concatenate(matrices, axis=1)
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, 6000, 100):
        w = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=min(100, 6000-start))
        totals = (w @ combined).reshape(len(w), len(predictions), 10, 5, 3)
        f1 = np.divide(2*totals[...,0], totals[...,1]+totals[...,2],
                       out=np.zeros_like(totals[...,0]), where=(totals[...,1]+totals[...,2])>0).mean(axis=(2,3))
        samples.append(f1)
    return np.concatenate(samples)

def evaluate_semisupervised(args):
    if not (args.run / "completed.json").exists():
        raise ValueError("All 50 partitions must finish before aggregate evaluation")
    frame, _, _ = load_development(args.root)
    y = targets(frame)
    # The submission ships one combined OOF file rather than 50 duplicate fold files.
    # New experiment runs can be evaluated directly from their per-fold outputs.
    if (args.run / "r0_f0.npz").exists():
        predictions = {arm: np.full((10, len(y), 5), np.nan) for arm in ARMS}
        metadata = []
        for r in range(10):
            for f in range(5):
                with np.load(args.run / f"r{r}_f{f}.npz") as part:
                    for arm in ARMS:
                        predictions[arm][r, part["indices"]] = part[arm]
                metadata.append(json.loads((args.run / f"r{r}_f{f}.json").read_text()))
        (args.run / "fold_metadata.json").write_text(json.dumps(metadata, indent=2))
    else:
        with np.load(args.run / "oof.npz") as saved:
            np.testing.assert_array_equal(saved["situation_id"], frame.situation_id)
            np.testing.assert_array_equal(saved["person_id"], frame.person_id)
            np.testing.assert_array_equal(saved["class_order"], CLASSES)
            np.testing.assert_array_equal(saved["y"], y)
            predictions = {arm: saved[arm] for arm in ARMS}
        metadata = json.loads((args.run / "fold_metadata.json").read_text())
    assert {(m["repeat"], m["fold"]) for m in metadata} == {(r, f) for r in range(10) for f in range(5)}
    gaps = [dict(repeat=m["repeat"], fold=m["fold"], **row) for m in metadata for row in m["diagnostics"]]
    for p in predictions.values():
        assert p.shape == (10, len(y), 5) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(2), 1.)
    np.savez_compressed(args.run / "oof.npz", **predictions, y=y, situation_id=frame.situation_id.to_numpy(str), person_id=frame.person_id.to_numpy(str), class_order=np.array(CLASSES))
    rows, classes, slices = [], [], []
    age = pd.cut(frame.age,[0,30,40,50,np.inf],labels=["<30","30–39","40–49","50+"],right=False).astype(object).fillna("missing")
    for arm, probabilities in predictions.items():
        for r, p in enumerate(probabilities):
            rows.append(dict(arm=arm,repeat=r,**metrics(y,p)))
            classes.append(per_class(y,p).assign(arm=arm,repeat=r))
            for name, groups in [("age_cohort",age),("context_domain",frame.context_domain)]:
                for value in sorted(groups.unique()):
                    mask = groups.eq(value).to_numpy()
                    slices.append(dict(arm=arm,repeat=r,slice=name,group=value,events=int(mask.sum()),**metrics(y[mask],p[mask])))
    table=pd.DataFrame(rows)
    table.to_csv(args.run / "repeat_metrics.csv",index=False)
    table.groupby("arm").mean(numeric_only=True).drop(columns="repeat").to_csv(args.run / "metrics.csv")
    pd.concat(classes).to_csv(args.run / "per_class.csv",index=False)
    pd.DataFrame(slices).to_csv(args.run / "slices.csv",index=False)
    pd.DataFrame(gaps).to_csv(args.run / "overfitting.csv",index=False)
    boot = f1_bootstrap(y, [predictions[a] for a in ARMS[1:]], frame.person_id)
    results = []
    for i, method in enumerate(["dae","scarf","soft"]):
        b, c = f"{method}_labeled",f"{method}_pooled"
        delta = table[table.arm.eq(c)].macro_f1.to_numpy()-table[table.arm.eq(b)].macro_f1.to_numpy()
        samples = boot[:,2*i+1]-boot[:,2*i]
        interval=np.quantile(samples,[.05/6,1-.05/6]).tolist()
        losses=paired_loss_interval(y,predictions[b],predictions[c],frame.person_id)
        baseline_losses=paired_loss_interval(y,predictions["supervised"],predictions[c],frame.person_id)
        # Probability scores drive selection. F1 remains a discrimination diagnostic.
        # Passing this preliminary screen still needs common calibration, cohort
        # checks and independent prospective confirmation before any deployment.
        passed=bool(all(d["log_loss"]["delta"]<=-.002 and d["log_loss"]["ci98_333"][1]<0
                        and d["brier"]["ci95"][1]<=.005 for d in [losses,baseline_losses]))
        results.append(dict(method=method,delta_macro_f1=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                            ci98_333=interval,positive_repeats=int((delta>0).sum()),
                            passes_probability_screen=passed,deployment_approved=False,
                            losses=losses,ordinary_supervised_comparison=baseline_losses,
                            decision_basis="log loss with Brier guardrail; F1 is secondary; common calibration and prospective confirmation still required"))
    (args.run / "comparisons.json").write_text(json.dumps(results,indent=2))
    print(table.groupby("arm").mean(numeric_only=True).to_string())
    print(json.dumps(results,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--kind", choices=["supervised", "ssl"], default="supervised")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "ssl":
        evaluate_semisupervised(args)
    else:
        evaluate_supervised(args)

if __name__ == "__main__":
    main()
