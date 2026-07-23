"""
Probability calibration + operating-threshold selection.

The model ranks well (AUC ~0.92-0.94) but its raw P(fake) scores are not
calibrated probabilities, and the naive 0.5 cut gives poor accuracy. This fits a
Platt scaler (logistic on the logit of the score) plus an operating threshold
chosen on the out-of-fold predictions, so `predict` returns a meaningful
probability and a sensible label. Everything is JSON-serializable — fit once on
`oof_probs.npy`, save, load at inference.
"""

import json
import math
from pathlib import Path

import numpy as np


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Calibrator:
    """Platt scaling (a*logit(p)+b) + a chosen decision threshold."""

    def __init__(self, a=1.0, b=0.0, threshold=0.5):
        self.a = float(a)
        self.b = float(b)
        self.threshold = float(threshold)

    def transform(self, probs):
        """Raw P(fake) -> calibrated P(fake)."""
        probs = np.asarray(probs, dtype=np.float64)
        return _sigmoid(self.a * _logit(probs) + self.b)

    def label(self, probs):
        """Raw P(fake) -> 0/1 using the calibrated prob and chosen threshold."""
        return (self.transform(probs) >= self.threshold).astype(int)

    @classmethod
    def fit(cls, probs, labels, objective="f1"):
        """Fit Platt scaling + threshold on out-of-fold (probs, labels).

        objective: 'f1' (default), 'youden' (TPR-FPR), or 'accuracy' — used only
        to pick the threshold; calibration itself is objective-independent.
        """
        from sklearn.linear_model import LogisticRegression

        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        x = _logit(probs).reshape(-1, 1)

        lr = LogisticRegression(C=1e6, solver="lbfgs")  # ~unregularized Platt
        lr.fit(x, labels)
        a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])

        cal = _sigmoid(a * _logit(probs) + b)
        threshold = _best_threshold(cal, labels, objective)
        return cls(a, b, threshold)

    def to_dict(self):
        return {"a": self.a, "b": self.b, "threshold": self.threshold}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["a"], d["b"], d["threshold"])


def _best_threshold(cal_probs, labels, objective="f1"):
    """Sweep candidate thresholds and return the one maximizing the objective."""
    order = np.argsort(cal_probs)
    candidates = np.unique(cal_probs[order])
    # midpoints between consecutive unique probs + the ends
    mids = (candidates[:-1] + candidates[1:]) / 2 if len(candidates) > 1 else candidates
    grid = np.concatenate([[0.0], mids, [1.0]])

    pos = labels == 1
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)

    best_t, best_score = 0.5, -1.0
    for t in grid:
        pred = cal_probs >= t
        tp = int((pred & pos).sum())
        fp = int((pred & neg).sum())
        fn = int((~pred & pos).sum())
        if objective == "youden":
            score = tp / n_pos - fp / n_neg
        elif objective == "accuracy":
            tn = n_neg - fp
            score = (tp + tn) / len(labels)
        else:  # f1
            denom = 2 * tp + fp + fn
            score = (2 * tp / denom) if denom > 0 else 0.0
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t
