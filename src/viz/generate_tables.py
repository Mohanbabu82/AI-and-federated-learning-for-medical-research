"""Generates T1-T4 (CSV + booktabs LaTeX) into paper/tables/ from results/summary/ and the
individual run directories under results/.

Usage:
    python -m src.viz.generate_tables --config configs/local_debug.yaml
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from experiments.validation import assert_full_run
from src.metrics.communication import rounds_to_target_accuracy
from src.models.client_model import build_client_model, param_counts
from src.utils.config import load_config
from src.viz.tables import bold_best, fmt_mean_std, write_booktabs, write_csv

METHODS = ["centralized", "local_only", "fedavg", "fedprox", "fedsaa"]
METHOD_LABEL = {
    "centralized": "Centralized", "local_only": "Local-only", "fedavg": "FedAvg",
    "fedprox": "FedProx", "fedsaa": "FedSAA (ours)",
}


def _run_csv(cfg: dict, method: str, seed: int) -> str:
    run_name = f"{cfg['experiment_name']}_{method}_seed{seed}"
    return os.path.join(cfg["logging"]["results_dir"], run_name, "metrics.csv")


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _method_seed_finals(cfg: dict, method: str) -> list[dict]:
    """Reads the final-round row for each seed's run, refusing (loudly) any run that isn't a
    complete full_run_* result -- see experiments/validation.py."""
    finals = []
    for seed in cfg["seeds"]:
        path = _run_csv(cfg, method, seed)
        if os.path.exists(path):
            assert_full_run(path, cfg["federated"]["rounds"])
            finals.append(_read_rows(path)[-1])
    return finals


def generate_t1_main_results(cfg: dict, out_dir: str):
    """T1: method x {global acc, macro-F1, AUC, worst-client acc, total comm MB}, mean+-std,
    bold best per column."""
    rows_data = {}
    for method in METHODS:
        finals = _method_seed_finals(cfg, method)
        if not finals:
            continue
        acc = [float(r["global_acc"]) for r in finals]
        f1 = [float(r["macro_f1"]) for r in finals]
        auc = [float(r["auc"]) for r in finals]
        worst = [min(float(a) for a in r["per_client_acc"].split(";")) for r in finals]
        mb = [float(r["cumulative_MB"]) for r in finals]
        rows_data[method] = {
            "acc": (np.mean(acc), np.std(acc)), "f1": (np.mean(f1), np.std(f1)),
            "auc": (np.mean(auc), np.std(auc)), "worst": (np.mean(worst), np.std(worst)),
            "mb": (np.mean(mb), np.std(mb)),
        }

    methods = list(rows_data.keys())
    acc_vals = [rows_data[m]["acc"][0] for m in methods]
    acc_stds = [rows_data[m]["acc"][1] for m in methods]
    f1_vals = [rows_data[m]["f1"][0] for m in methods]
    f1_stds = [rows_data[m]["f1"][1] for m in methods]
    auc_vals = [rows_data[m]["auc"][0] for m in methods]
    auc_stds = [rows_data[m]["auc"][1] for m in methods]
    worst_vals = [rows_data[m]["worst"][0] for m in methods]
    worst_stds = [rows_data[m]["worst"][1] for m in methods]
    mb_vals = [rows_data[m]["mb"][0] for m in methods]
    mb_stds = [rows_data[m]["mb"][1] for m in methods]

    acc_fmt = bold_best(acc_vals, higher_is_better=True, stds=acc_stds)
    f1_fmt = bold_best(f1_vals, higher_is_better=True, stds=f1_stds)
    auc_fmt = bold_best(auc_vals, higher_is_better=True, stds=auc_stds)
    worst_fmt = bold_best(worst_vals, higher_is_better=True, stds=worst_stds)
    mb_fmt = bold_best(mb_vals, higher_is_better=False, decimals=2, stds=mb_stds)

    csv_rows = []
    latex_rows = []
    for i, method in enumerate(methods):
        csv_rows.append({
            "method": method,
            "global_acc_mean": acc_vals[i], "global_acc_std": acc_stds[i],
            "macro_f1_mean": f1_vals[i], "macro_f1_std": f1_stds[i],
            "auc_mean": auc_vals[i], "auc_std": auc_stds[i],
            "worst_client_acc_mean": worst_vals[i], "worst_client_acc_std": worst_stds[i],
            "total_comm_MB_mean": mb_vals[i], "total_comm_MB_std": mb_stds[i],
        })
        latex_rows.append([METHOD_LABEL[method], acc_fmt[i], f1_fmt[i], auc_fmt[i],
                            worst_fmt[i], mb_fmt[i]])

    write_csv(csv_rows, list(csv_rows[0].keys()) if csv_rows else [],
              os.path.join(out_dir, "T1_main_results.csv"))
    latex = write_booktabs(
        headers=["Method", "Global Acc.", "Macro-F1", "AUC", "Worst-client Acc.", "Total Comm. (MB)"],
        rows=latex_rows,
        caption="Main results: mean$\\pm$std across seeds. Best value per column in bold. "
                "Centralized/Local-only never communicate over a network (0 MB by construction).",
        label="tab:main_results",
        path=os.path.join(out_dir, "T1_main_results.tex"),
    )
    return latex


def generate_t2_hyperparameters(cfg: dict, out_dir: str):
    """T2: full hyperparameter dump (section, key, value), flattened from the config."""
    def flatten(d: dict, prefix: str = "") -> list[tuple[str, str, str]]:
        rows = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows += flatten(v, full_key)
            else:
                section = full_key.split(".")[0]
                rows.append((section, full_key, str(v)))
        return rows

    flat = flatten(cfg)
    csv_rows = [{"section": s, "key": k, "value": v} for s, k, v in flat]
    write_csv(csv_rows, ["section", "key", "value"], os.path.join(out_dir, "T2_hyperparameters.csv"))

    latex_rows = [[_esc(s), _esc(k), _esc(v)] for s, k, v in flat]
    latex = write_booktabs(
        headers=["Section", "Key", "Value"],
        rows=latex_rows,
        caption=f"Full hyperparameter configuration ({os.path.basename(cfg['experiment_name'])}).",
        label="tab:hyperparameters",
        path=os.path.join(out_dir, "T2_hyperparameters.tex"),
        col_align="lll",
    )
    return latex


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def generate_t3_efficiency(cfg: dict, out_dir: str):
    """T3: trained params, MB/round, rounds-to-target, total MB -- one row per method."""
    # trainable params per client model (LoRA + one classifier head), same backbone/rank for all
    # federated/local/centralized methods here since they share configs/*.yaml's model block
    representative_num_classes = 9  # e.g. PathMNIST; head size varies slightly by specialty
    model = build_client_model(cfg, num_classes=representative_num_classes)
    trainable, total = param_counts(model)

    csv_rows = []
    latex_rows = []
    for method in METHODS:
        finals_by_seed = []
        mb_per_round_last = None
        rounds_curve = []
        for seed in cfg["seeds"]:
            path = _run_csv(cfg, method, seed)
            if not os.path.exists(path):
                continue
            assert_full_run(path, cfg["federated"]["rounds"])
            rows = _read_rows(path)
            finals_by_seed.append(rows[-1])
            if seed == cfg["seeds"][0]:
                rounds_curve = [float(r["global_acc"]) for r in rows]
                mb_per_round_last = float(rows[-1]["adapter_MB_round"])
        if not finals_by_seed:
            continue

        total_mb = np.mean([float(r["cumulative_MB"]) for r in finals_by_seed])
        best_final_acc = max(rounds_curve) if rounds_curve else float("nan")
        target = 0.9 * best_final_acc if rounds_curve else float("nan")
        r2t = rounds_to_target_accuracy(rounds_curve, target) if rounds_curve else None

        csv_rows.append({
            "method": method, "trainable_params": trainable, "total_params": total,
            "adapter_MB_per_round": mb_per_round_last, "rounds_to_90pct_of_own_best": r2t,
            "total_comm_MB": total_mb,
        })
        latex_rows.append([
            METHOD_LABEL[method], f"{trainable:,}",
            f"{mb_per_round_last:.3f}" if mb_per_round_last else "0.000",
            str(r2t) if r2t is not None else "--",
            f"{total_mb:.2f}",
        ])

    write_csv(csv_rows, list(csv_rows[0].keys()) if csv_rows else [],
              os.path.join(out_dir, "T3_efficiency.csv"))
    latex = write_booktabs(
        headers=["Method", "Trainable Params", "MB/Round", "Rounds to 90\\% of own best", "Total Comm. (MB)"],
        rows=latex_rows,
        caption="Efficiency: trainable parameter count (LoRA + one client head), per-round and "
                "total communication, and rounds needed to reach 90\\% of that method's own "
                "best observed accuracy.",
        label="tab:efficiency",
        path=os.path.join(out_dir, "T3_efficiency.tex"),
    )
    return latex


def generate_t4_ablations(summary_dir: str, out_dir: str):
    """T4: FedSAA ablation grid (lambda, tau, rank, client_fraction) -> acc/F1/AUC mean+-std."""
    ablation_csv = os.path.join(summary_dir, "fedsaa_ablation_summary.csv")
    if not os.path.exists(ablation_csv):
        return None
    with open(ablation_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    csv_rows = rows  # already in the right shape; just re-export alongside the LaTeX version
    write_csv(csv_rows, list(rows[0].keys()) if rows else [], os.path.join(out_dir, "T4_ablations.csv"))

    latex_rows = []
    for r in rows:
        latex_rows.append([
            r["lambda"], r["tau"], r["rank"], r["client_fraction"],
            fmt_mean_std(float(r["global_acc_mean"]), float(r["global_acc_std"]), latex=True),
            fmt_mean_std(float(r["macro_f1_mean"]), float(r["macro_f1_std"]), latex=True),
            fmt_mean_std(float(r["auc_mean"]), float(r["auc_std"]), latex=True),
        ])
    latex = write_booktabs(
        headers=[r"$\lambda$", r"$\tau$", "rank $r$", "$C$", "Global Acc.", "Macro-F1", "AUC"],
        rows=latex_rows,
        caption="FedSAA ablation over $\\lambda$, $\\tau$, rank $r$, and client fraction $C$ "
                "(mean$\\pm$std across seeds).",
        label="tab:ablations",
        path=os.path.join(out_dir, "T4_ablations.tex"),
    )
    return latex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="paper/tables")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary_dir = os.path.join(cfg["logging"]["results_dir"], "summary")
    os.makedirs(args.out_dir, exist_ok=True)

    t1 = generate_t1_main_results(cfg, args.out_dir)
    generate_t2_hyperparameters(cfg, args.out_dir)
    generate_t3_efficiency(cfg, args.out_dir)
    generate_t4_ablations(summary_dir, args.out_dir)

    written = sorted(os.listdir(args.out_dir))
    print(f"Wrote {len(written)} files to {args.out_dir}:")
    for w in written:
        print(f"  {os.path.join(args.out_dir, w)}")

    print("\n" + "=" * 90)
    print("T1 main_results.tex:")
    print("=" * 90)
    print(t1)


if __name__ == "__main__":
    main()
