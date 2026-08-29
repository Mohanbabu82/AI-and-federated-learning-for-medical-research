"""Standardized classification metrics on raw arrays (no DataLoader dependency), so the same
functions back both src/metrics/evaluation.py (used during training loops) and tests.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def auc_ovr(y_true: np.ndarray, probs: np.ndarray) -> float:
    """One-vs-rest AUC. Binary -> standard AUC on the positive class; multiclass -> macro OVR.
    Returns nan if undefined (e.g. a class absent from y_true)."""
    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                    labels=list(range(n_classes))))
    except ValueError:
        return float("nan")
