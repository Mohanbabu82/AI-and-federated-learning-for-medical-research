"""Shared helpers used by every method's training loop (fedavg/fedprox/fedsaa/local_only/centralized)
so they all emit the exact same results/<run>/metrics.csv schema.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import psutil

from src.metrics.cka import compute_cka_matrix

METRICS_FIELDS = [
    "round", "global_acc", "per_client_acc", "macro_f1", "auc",
    "adapter_MB_round", "cumulative_MB", "wall_clock_sec", "ram_mb",
]


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def run_dir(cfg: dict, method: str, seed: int) -> str:
    run_name = f"{cfg['experiment_name']}_{method}_seed{seed}"
    path = os.path.join(cfg["logging"]["results_dir"], run_name)
    os.makedirs(path, exist_ok=True)
    return path


def write_metrics_csv(csv_path: str, rows: list[dict]) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def log_cka_this_round(client_models: list, probe_x, device, cka_dir: str, round_idx: int) -> float:
    """Computes the pairwise client-CKA matrix on the shared probe batch, saves it to
    <cka_dir>/round_NNN.npy, and returns the mean off-diagonal value (a single collapse score
    for quick printing/comparison -- figure F7, FedSAA vs FedAvg)."""
    os.makedirs(cka_dir, exist_ok=True)
    cka_matrix = compute_cka_matrix(client_models, probe_x, device)
    np.save(os.path.join(cka_dir, f"round_{round_idx:03d}.npy"), cka_matrix)

    n = cka_matrix.shape[0]
    if n < 2:
        return float("nan")
    off_diag = cka_matrix[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off_diag))
