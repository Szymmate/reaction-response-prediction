"""Metrics, calibration plots, cohort slices and paired employee uncertainty."""
import features  # Sets thread limits before importing numerical libraries.
import numpy as np
import pandas as pd
from features import SEED, CLASSES
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def reliability(y, p):
    rows = []
    for k in range(-1, 5):
        confidence = p.max(1) if k == -1 else p[:, k]
        observed = p.argmax(1) == y if k == -1 else y == k
        bins = np.minimum((confidence * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            if mask.any():
                rows.append(dict(class_name="top_label" if k == -1 else CLASSES[k], bin=b, n=int(mask.sum()),
                                 predicted=float(confidence[mask].mean()), observed=float(observed[mask].mean())))
    return pd.DataFrame(rows)

def metrics(y, p):
    predicted = p.argmax(1)
    rel = reliability(y, p)
    ece = {name: np.average(abs(g.predicted - g.observed), weights=g.n) for name, g in rel.groupby("class_name")}
    # A deterministic most-frequent classifier assigns zero to other true classes.
    actual = p[np.arange(len(y)), y]
    ll = float("inf") if (actual == 0).any() else float(-np.log(actual).mean())
    return dict(accuracy=accuracy_score(y, predicted), macro_f1=f1_score(y, predicted, labels=range(5), average="macro", zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y, predicted), log_loss=ll,
                brier=float(((p - np.eye(5)[y]) ** 2).sum(1).mean()),
                macro_auc=roc_auc_score(y, p, multi_class="ovr", average="macro", labels=range(5)) if len(np.unique(y)) == 5 else np.nan,
                top_ece=ece["top_label"], classwise_ece=float(np.mean([ece[c] for c in CLASSES])))

def per_class(y, p):
    precision, recall, f1, support = precision_recall_fscore_support(y, p.argmax(1), labels=range(5), zero_division=0)
    return pd.DataFrame(dict(class_name=CLASSES, precision=precision, recall=recall, f1=f1, support=support))

def paired_loss_interval(y, before, after, people, draws=6000):
    """Resample employees with shared weights; average losses across repeats first."""
    a = -np.log(np.clip(before[:, np.arange(len(y)), y], 1e-15, 1))
    b = -np.log(np.clip(after[:, np.arange(len(y)), y], 1e-15, 1))
    deltas = {"log_loss": (b-a).mean(0),
              "brier": (((after-np.eye(5)[y])**2).sum(2)-((before-np.eye(5)[y])**2).sum(2)).mean(0)}
    codes, unique = pd.factorize(people)
    counts = np.bincount(codes)
    sums = np.column_stack([np.bincount(codes, weights=d) for d in deltas.values()])
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, draws, 200):
        w = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=min(200, draws-start))
        samples.append((w @ sums)/(w @ counts)[:, None])
    samples = np.concatenate(samples)
    return {key: dict(delta=float(d.mean()), ci95=np.quantile(samples[:, i], [.025, .975]).tolist(),
                     ci98_333=np.quantile(samples[:, i], [.05/6, 1-.05/6]).tolist())
            for i, (key, d) in enumerate(deltas.items())}
