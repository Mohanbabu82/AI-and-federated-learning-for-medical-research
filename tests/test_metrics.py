"""Unit tests for src/metrics/ on tiny fake tensors (no pytest dependency -- plain asserts).

Usage:
    python -m tests.test_metrics
"""

from __future__ import annotations

import numpy as np
import torch

from src.metrics.classification import accuracy, auc_ovr, macro_f1
from src.metrics.cka import linear_cka
from src.metrics.communication import (
    cumulative_communication,
    mb_to_target_accuracy,
    round_communication_mb,
    rounds_to_target_accuracy,
)


def test_accuracy():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 0.8
    print("PASS: accuracy")


def test_macro_f1():
    # perfect predictions -> macro F1 = 1.0
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y_true, y_pred) == 1.0

    # all wrong -> macro F1 = 0.0
    y_pred_wrong = np.array([1, 2, 0, 1, 2, 0])
    assert macro_f1(y_true, y_pred_wrong) == 0.0
    print("PASS: macro_f1")


def test_auc_ovr_binary_and_multiclass():
    # binary: perfect separation -> AUC = 1.0
    y_true_bin = np.array([0, 0, 1, 1])
    probs_bin = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert abs(auc_ovr(y_true_bin, probs_bin) - 1.0) < 1e-9

    # multiclass: perfect one-hot confidence -> macro OVR AUC = 1.0
    y_true_mc = np.array([0, 1, 2, 0, 1, 2])
    probs_mc = np.eye(3)[y_true_mc]
    assert abs(auc_ovr(y_true_mc, probs_mc) - 1.0) < 1e-9

    # a class missing from y_true -> undefined -> nan, not a crash
    y_true_missing = np.array([0, 0, 0])
    probs_missing = np.array([[0.6, 0.2, 0.2]] * 3)
    assert np.isnan(auc_ovr(y_true_missing, probs_missing))
    print("PASS: auc_ovr")


def test_communication_accounting():
    assert round_communication_mb(n_participants=3, payload_mb=0.5) == 3.0  # up+down
    assert round_communication_mb(n_participants=3, payload_mb=0.5, download=False) == 1.5

    per_round = [1.0, 2.0, 1.5]
    assert cumulative_communication(per_round) == [1.0, 3.0, 4.5]

    accs = [0.1, 0.4, 0.6, 0.8, 0.9]
    assert rounds_to_target_accuracy(accs, target=0.6) == 3
    assert rounds_to_target_accuracy(accs, target=0.99) is None

    cum = cumulative_communication([2.0, 2.0, 2.0, 2.0, 2.0])
    assert mb_to_target_accuracy(accs, cum, target=0.6) == 6.0
    assert mb_to_target_accuracy(accs, cum, target=0.99) is None
    print("PASS: communication accounting")


def test_linear_cka_identity_and_bounds():
    torch.manual_seed(0)
    X = torch.randn(20, 6)

    # CKA(X, X) == 1.0
    assert abs(linear_cka(X, X) - 1.0) < 1e-5

    # CKA(X, cX) == 1.0 for any nonzero scalar c -- linear CKA is scale-invariant
    assert abs(linear_cka(X, 3.7 * X) - 1.0) < 1e-5

    # CKA(X, XR) == CKA(X, X) for an orthogonal rotation R -- rotation-invariant
    Q, _ = torch.linalg.qr(torch.randn(6, 6))
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-4

    # independent random features -> CKA well below 1 (not a strict bound, but should be small-ish)
    Y = torch.randn(20, 6)
    cka_indep = linear_cka(X, Y)
    assert 0.0 <= cka_indep < 0.9

    print("PASS: linear_cka identity/scale/rotation invariance + bounds")


if __name__ == "__main__":
    test_accuracy()
    test_macro_f1()
    test_auc_ovr_binary_and_multiclass()
    test_communication_accounting()
    test_linear_cka_identity_and_bounds()
    print("All metrics tests passed.")
