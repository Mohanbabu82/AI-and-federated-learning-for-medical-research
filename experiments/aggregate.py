"""Aggregates per-run results/<run>/metrics.csv files into results/summary/*.csv with mean+-std
across seeds, grouped by method (and, for FedSAA, by ablation hyperparameters too).
"""

from __future__ import annotations

import csv
import os

import numpy as np


def _read_final_row(csv_path: str) -> dict:
    """Reads the last row of a run's metrics.csv (i.e. its final-round result)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    return rows[-1]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _summarize_group(final_rows: list[dict]) -> dict:
    accs = [float(r["global_acc"]) for r in final_rows]
    f1s = [float(r["macro_f1"]) for r in final_rows]
    aucs = [float(r["auc"]) for r in final_rows]
    cum_mb = [float(r["cumulative_MB"]) for r in final_rows]
    ram = [float(r["ram_mb"]) for r in final_rows]

    acc_mean, acc_std = _mean_std(accs)
    f1_mean, f1_std = _mean_std(f1s)
    auc_mean, auc_std = _mean_std(aucs)
    mb_mean, mb_std = _mean_std(cum_mb)
    ram_mean, ram_std = _mean_std(ram)

    return {
        "n_seeds": len(final_rows),
        "global_acc_mean": acc_mean, "global_acc_std": acc_std,
        "macro_f1_mean": f1_mean, "macro_f1_std": f1_std,
        "auc_mean": auc_mean, "auc_std": auc_std,
        "cumulative_MB_mean": mb_mean, "cumulative_MB_std": mb_std,
        "ram_mb_mean": ram_mean, "ram_mb_std": ram_std,
    }


def aggregate_all(records: list[dict], summary_dir: str) -> tuple[str, str]:
    """records: list of {group, method, seed, csv_path, lambda, tau, rank, client_fraction}.
    Writes summary_dir/methods_summary.csv (one row per method, mean+-std over seeds) and
    summary_dir/fedsaa_ablation_summary.csv (one row per FedSAA ablation config).
    Returns (methods_csv_path, ablation_csv_path).
    """
    os.makedirs(summary_dir, exist_ok=True)

    # --- method-level summary: group by `method` only (across the base method sweep) ---
    by_method: dict[str, list[dict]] = {}
    for rec in records:
        if rec["lambda"] is not None:
            continue  # ablation-grid runs are summarized separately below
        by_method.setdefault(rec["method"], []).append(rec)

    methods_csv = os.path.join(summary_dir, "methods_summary.csv")
    method_fields = ["method", "n_seeds", "global_acc_mean", "global_acc_std",
                      "macro_f1_mean", "macro_f1_std", "auc_mean", "auc_std",
                      "cumulative_MB_mean", "cumulative_MB_std", "ram_mb_mean", "ram_mb_std"]
    with open(methods_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=method_fields)
        writer.writeheader()
        for method, recs in sorted(by_method.items()):
            final_rows = [_read_final_row(r["csv_path"]) for r in recs]
            summary = _summarize_group(final_rows)
            writer.writerow({"method": method, **summary})

    # --- FedSAA ablation summary: group by (lambda, tau, rank, client_fraction) ---
    by_ablation: dict[tuple, list[dict]] = {}
    for rec in records:
        if rec["lambda"] is None:
            continue
        key = (rec["lambda"], rec["tau"], rec["rank"], rec["client_fraction"])
        by_ablation.setdefault(key, []).append(rec)

    ablation_csv = os.path.join(summary_dir, "fedsaa_ablation_summary.csv")
    ablation_fields = ["lambda", "tau", "rank", "client_fraction"] + method_fields[1:]
    with open(ablation_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ablation_fields)
        writer.writeheader()
        for key, recs in sorted(by_ablation.items(), key=lambda kv: kv[0]):
            lam, tau, rank, frac = key
            final_rows = [_read_final_row(r["csv_path"]) for r in recs]
            summary = _summarize_group(final_rows)
            writer.writerow({"lambda": lam, "tau": tau, "rank": rank,
                              "client_fraction": frac, **summary})

    return methods_csv, ablation_csv
