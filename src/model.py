"""Fixed LightGBM, temperature calibration and the optional soft-target objective."""
import features  # Sets thread limits before importing numerical libraries.
import lightgbm as lgb
import numpy as np
from features import CLASSES, FeaturePipeline, SEED, clean
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


MODEL = dict(n_estimators=100, num_leaves=7, min_child_samples=40, learning_rate=.05,
             reg_lambda=10., random_state=SEED, n_jobs=1, objective="multiclass",
             num_class=5, deterministic=True, force_col_wise=True, verbosity=-1)


def temper(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return softmax(np.log(np.clip(probabilities, 1e-15, 1)) / temperature, axis=1)

def fit_temperature(y, probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(y), 5) or not np.isfinite(probabilities).all():
        raise ValueError("Incomplete inner OOF probabilities")
    def loss(log_t):
        p = temper(probabilities, np.exp(log_t))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()
    result = minimize_scalar(loss, bounds=(np.log(.25), np.log(4.)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


class ResponseModel:
    def __init__(self, features, estimator, temperature=1.):
        self.features = features
        self.estimator = estimator
        self.temperature = temperature
        self.classes = list(CLASSES)

    def predict_uncalibrated(self, frame):
        if not np.array_equal(self.estimator.classes_, np.arange(5)):
            raise ValueError("Estimator class-order mismatch")
        x = self.features.transform(frame)
        if hasattr(self.estimator, "booster_"):
            p = self.estimator.booster_.predict(x, num_threads=1)
        else:
            p = self.estimator.predict_proba(x)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
            raise ValueError("Invalid five-class distribution")
        return p

    def predict_proba(self, frame):
        return temper(self.predict_uncalibrated(frame), self.temperature)

def fit_supervised(frame, y, features=None):
    features = FeaturePipeline().fit(frame) if features is None else features
    estimator = LGBMClassifier(**MODEL).fit(features.transform(frame), y)
    return ResponseModel(features, estimator)


def validate_targets(q, weights):
    q = np.asarray(q, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(CLASSES):
        raise ValueError('Expected one five-class target distribution per row')
    if weights.shape != (len(q),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid sample weights')
    if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(axis=1), 1, atol=1e-10):
        raise ValueError('Targets must be finite, nonnegative and sum to one')
    return q, weights


def soft_cross_entropy(logits, q, weights):
    """Unnormalized weighted loss; matches the gradient scale passed to LightGBM."""
    return float(np.sum(weights * (logsumexp(logits, axis=1) - np.sum(q * logits, axis=1))))


class SoftCrossEntropyObjective:
    def __init__(self, q, weights):
        self.q, self.weights = validate_targets(q, weights)

    def __call__(self, raw_scores, train_data=None):
        if raw_scores.shape != self.q.shape:
            raise ValueError(f'Unexpected multiclass score shape: {raw_scores.shape}')
        p = softmax(raw_scores, axis=1)
        grad = self.weights[:, None] * (p - self.q)
        # Exact diagonal entries of the Hessian; off-diagonal class terms are not
        # representable in LightGBM's custom objective API. Floor only for stability.
        hess = self.weights[:, None] * np.maximum(p * (1 - p), 1e-12)
        return grad, hess


class SoftTargetEstimator:
    def __init__(self, booster, initial_logits):
        self.booster = booster
        self.initial_logits = np.asarray(initial_logits, dtype=np.float64)
        self.classes_ = np.arange(len(CLASSES))

    def decision_function(self, x):
        # A Dataset init_score affects training but is not embedded in the saved trees.
        return self.booster.predict(x, raw_score=True, num_threads=1) + self.initial_logits[None, :]

    def predict_proba(self, x):
        p = softmax(self.decision_function(x), axis=1)
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError('Invalid custom-objective probabilities')
        return p


def fit_soft_lightgbm(x, q, weights, labeled_y, parameters):
    q, weights = validate_targets(q, weights)
    if len(x) != len(q):
        raise ValueError('Feature/target row mismatch')
    counts = np.bincount(labeled_y, minlength=len(CLASSES)).astype(float)
    if (counts == 0).any():
        raise ValueError('All real-label training classes are required')
    # Use the same real-label prior for both matched soft-target controls.
    # Do not initialize from artificial labels; that would add a second changed factor.
    initial = np.log(counts / counts.sum())
    initial -= initial.mean()
    dataset = lgb.Dataset(np.asarray(x), label=np.zeros(len(x)),
                          init_score=np.tile(initial, (len(x), 1)), free_raw_data=True)
    objective = SoftCrossEntropyObjective(q, weights)
    params = dict(objective=objective, num_class=len(CLASSES), metric='None',
                  learning_rate=.05, lambda_l2=10., min_data_in_leaf=40,
                  num_leaves=int(parameters['num_leaves']), seed=SEED, num_threads=1,
                  deterministic=True, force_col_wise=True, verbosity=-1,
                  boost_from_average=False)
    booster = lgb.train(params, dataset, num_boost_round=int(parameters['n_estimators']))
    return SoftTargetEstimator(booster, initial)
