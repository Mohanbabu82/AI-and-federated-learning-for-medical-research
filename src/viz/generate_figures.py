"""Generates F1-F7 (PDF+PNG, 300 dpi) into paper/figures/ from results/summary/ and the
individual run directories under results/, plus a captioned paper/figures/README.md.

Usage:
    python -m src.viz.generate_figures --config configs/local_debug.yaml
"""

from __future__ import annotations

import argparse
import glob
import os

from experiments.validation import DebugRunError, assert_full_run
from src.utils.config import load_config
from src.viz.plots import (
    plot_accuracy_vs_comm,
    plot_cka_collapse,
    plot_convergence,
    plot_lambda_ablation,
    plot_per_specialty_bars,
    plot_rank_ablation,
    plot_similarity_heatmap,
)
from src.viz.style import apply_style, save_fig

CAPTIONS = {
    "F1_convergence": (
        "**F1 -- Convergence.** Global test accuracy vs. federated round for each method. "
        "FedSAA's per-client similarity-attended aggregation is compared against the "
        "Centralized upper bound, Local-only lower bound, and the FedAvg/FedProx baselines."
    ),
    "F2_accuracy_vs_comm": (
        "**F2 -- Accuracy vs. communication cost.** Global test accuracy plotted against "
        "cumulative communication (MB of LoRA adapter payload exchanged). Centralized and "
        "Local-only never communicate over a network, so they are shown as horizontal "
        "reference lines at their final accuracy rather than a curve."
    ),
    "F3_per_specialty_bars": (
        "**F3 -- Per-specialty accuracy.** Final-round test accuracy broken out by specialty "
        "(simulated hospital), one bar group per specialty, one bar per method. Highlights "
        "which methods under-serve which specialties under cross-specialty heterogeneity."
    ),
    "F4_similarity_heatmap": (
        "**F4 -- FedSAA client similarity matrix S.** Cosine similarity between clients' "
        "merged LoRA deltas U_i = B_i @ A_i (mean across LoRA layers, from the final logged "
        "round). This is the raw signal the softmax attention (Step 5 of FedSAA) acts on."
    ),
    "F5_lambda_ablation": (
        "**F5 -- FedSAA lambda ablation.** Global test accuracy (mean +/- std across seeds) as "
        "lambda sweeps from 0 (pure similarity-attended aggregation) to 1 (equivalent to "
        "FedAvg on merged LoRA deltas -- see tests/test_fedsaa.py), at fixed tau/rank/"
        "client_fraction."
    ),
    "F6_rank_ablation": (
        "**F6 -- FedSAA rank ablation.** Global test accuracy (left axis) and cumulative "
        "communication cost (right axis) vs. the LoRA rank r used for both local adapters and "
        "the server's truncated-SVD re-factorization, at fixed lambda/tau/client_fraction. "
        "Shows the accuracy/communication trade-off of the rank choice."
    ),
    "F7_collapse_cka": (
        "**F7 -- Representation collapse (CKA).** Mean pairwise linear CKA between clients' "
        "backbone features on a small fixed probe batch, per round. Higher values indicate "
        "clients' representations have homogenized (collapsed) toward each other; FedSAA is "
        "compared against FedAvg to see whether attention-based aggregation preserves more "
        "cross-specialty representational diversity."
    ),
}


def _run_csv(cfg: dict, method: str, seed: int) -> str:
    run_name = f"{cfg['experiment_name']}_{method}_seed{seed}"
    return os.path.join(cfg["logging"]["results_dir"], run_name, "metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seeds"][0]
    results_dir = cfg["logging"]["results_dir"]
    summary_dir = os.path.join(results_dir, "summary")
    expected_rounds = cfg["federated"]["rounds"]
    apply_style()

    methods = ["centralized", "local_only", "fedavg", "fedprox", "fedsaa"]
    method_csv_paths = {m: _run_csv(cfg, m, seed) for m in methods
                         if os.path.exists(_run_csv(cfg, m, seed))}

    # Never let a debug/smoke-test or incomplete run feed a paper figure -- fail loudly instead.
    for m, path in method_csv_paths.items():
        try:
            assert_full_run(path, expected_rounds)
        except DebugRunError as e:
            raise DebugRunError(f"F1-F3 input rejected for method '{m}': {e}") from None

    written: list[str] = []

    if method_csv_paths:
        fig = plot_convergence(method_csv_paths)
        written += save_fig(fig, args.out_dir, "F1_convergence")

        fig = plot_accuracy_vs_comm(method_csv_paths)
        written += save_fig(fig, args.out_dir, "F2_accuracy_vs_comm")

        fig = plot_per_specialty_bars(method_csv_paths, cfg["data"]["specialties"])
        written += save_fig(fig, args.out_dir, "F3_per_specialty_bars")

    fedsaa_run_dir = f"{cfg['experiment_name']}_fedsaa_seed{seed}"
    sim_files = sorted(glob.glob(os.path.join(results_dir, fedsaa_run_dir,
                                               "similarity", "round_*.npz")))
    if sim_files:
        assert_full_run(os.path.join(results_dir, fedsaa_run_dir, "metrics.csv"), expected_rounds)
        fig = plot_similarity_heatmap(sim_files[-1])
        written += save_fig(fig, args.out_dir, "F4_similarity_heatmap")

    ablation_csv = os.path.join(summary_dir, "fedsaa_ablation_summary.csv")
    if os.path.exists(ablation_csv):
        fig = plot_lambda_ablation(ablation_csv)
        written += save_fig(fig, args.out_dir, "F5_lambda_ablation")

        fig = plot_rank_ablation(ablation_csv)
        written += save_fig(fig, args.out_dir, "F6_rank_ablation")

    cka_dirs = {
        m: os.path.join(results_dir, f"{cfg['experiment_name']}_{m}_seed{seed}", "cka")
        for m in ["fedavg", "fedsaa"]
    }
    cka_dirs = {m: d for m, d in cka_dirs.items() if os.path.isdir(d) and glob.glob(os.path.join(d, "*.npy"))}
    for m, d in cka_dirs.items():
        assert_full_run(os.path.join(os.path.dirname(d), "metrics.csv"), expected_rounds)
    if cka_dirs:
        fig = plot_cka_collapse(cka_dirs)
        written += save_fig(fig, args.out_dir, "F7_collapse_cka")

    readme_path = os.path.join(args.out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Figures\n\n")
        f.write(f"Generated from `{args.config}` (seed={seed}). Each figure is exported as "
                f"both `.pdf` (vector, for the paper) and `.png` (300 dpi, for slides/previews). "
                f"Colors follow the colorblind-safe Okabe-Ito palette, consistent across all figures.\n\n")
        for key, caption in CAPTIONS.items():
            if any(key in w for w in written):
                f.write(f"## {key}\n\n{caption}\n\n")
    written.append(readme_path)

    print(f"Wrote {len(written)} files to {args.out_dir}:")
    for w in sorted(written):
        print(f"  {w}")


if __name__ == "__main__":
    main()
