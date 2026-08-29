"""Prints an estimated total CPU wall-clock time and peak RAM for a full experiment sweep
(method sweep x seeds, + FedSAA one-at-a-time ablation x 1 seed each), broken down per stage,
WITHOUT running anything. Use this before committing to `python -m experiments.runner --config <cfg>`.

Usage:
    python -m experiments.estimate --config configs/full_run.yaml
"""

from __future__ import annotations

import argparse
import copy

from experiments.runner import ALL_METHODS, FULL_ABLATION_GRID, _build_ablation_combos
from experiments.runtime_estimate import estimate_seconds
from src.utils.config import load_config

# Rough peak-RAM-per-run estimates (MB), extrapolated from measured local_debug runs (Steps 5-10:
# 3 clients, 200-830 MB observed) scaled by client count and dataset-without-cap size. This is a
# planning number, not a guarantee -- actual peak depends on OS/Python overhead too.
BASE_RSS_MB = 350          # interpreter + torch + medmnist import overhead
PER_CLIENT_MODEL_MB = 55   # one resnet18+LoRA copy (unshared weights) per client model
PER_CLIENT_DATA_MB = 15    # DataLoader/Dataset object overhead per client (data itself is mmap'd)


def estimate_ram_mb(cfg: dict, method: str) -> float:
    n = cfg["federated"]["num_clients"]
    if method == "centralized":
        # one shared backbone + n heads -- much cheaper than n separate backbones
        return BASE_RSS_MB + PER_CLIENT_MODEL_MB + n * PER_CLIENT_DATA_MB
    return BASE_RSS_MB + n * (PER_CLIENT_MODEL_MB + PER_CLIENT_DATA_MB)


def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    method_seeds = base_cfg["seeds"]           # main results: never below 3 seeds
    ablation_seeds = base_cfg["seeds"][:1]     # one-at-a-time ablation points: 1 seed each
    n_method_seeds = len(method_seeds)
    n_ablation_seeds = len(ablation_seeds)

    print("=" * 90)
    print(f"CPU wall-clock / RAM estimate -- {args.config}  "
          f"(method_seeds={method_seeds}, ablation_seeds={ablation_seeds}, "
          f"rounds={base_cfg['federated']['rounds']}, num_clients={base_cfg['federated']['num_clients']})")
    print("=" * 90)

    # --- Stage 1: method sweep (all 3 seeds) ---
    print(f"\n[Stage 1] Method sweep (Centralized, Local-only, FedAvg, FedProx, FedSAA) x {n_method_seeds} seeds")
    stage1_total = 0.0
    peak_ram_seen = 0.0
    for method in ALL_METHODS:
        per_run_sec = estimate_seconds(base_cfg, method)
        stage_sec = per_run_sec * n_method_seeds
        stage1_total += stage_sec
        ram = estimate_ram_mb(base_cfg, method)
        peak_ram_seen = max(peak_ram_seen, ram)
        print(f"  {method:<13} {n_method_seeds} seed(s) x {fmt_hms(per_run_sec)}/run "
              f"= {fmt_hms(stage_sec)}   (~{ram:.0f} MB peak)")
    print(f"  Stage 1 subtotal: {fmt_hms(stage1_total)}")

    # --- Stage 2: FedSAA one-at-a-time ablation (1 seed each) ---
    default = FULL_ABLATION_GRID["default"]
    axes = FULL_ABLATION_GRID["axes"]
    combos = _build_ablation_combos(FULL_ABLATION_GRID)
    print(f"\n[Stage 2] FedSAA one-at-a-time ablation around default {default}: "
          f"lambda in {axes['lambda']}, tau in {axes['tau']}, rank in {axes['rank']} "
          f"(deduplicated -> {len(combos)} unique configs) x {n_ablation_seeds} seed each")
    stage2_total = 0.0
    for _lam, _tau, _rank, frac in combos:
        cfg = copy.deepcopy(base_cfg)
        cfg["federated"]["client_fraction"] = frac
        per_run_sec = estimate_seconds(cfg, "fedsaa")
        stage2_total += per_run_sec * n_ablation_seeds
    n_runs_stage2 = len(combos) * n_ablation_seeds
    print(f"  {len(combos)} configs x {n_ablation_seeds} seed(s) = {n_runs_stage2} runs")
    print(f"  Stage 2 subtotal: {fmt_hms(stage2_total)}  "
          f"(avg {fmt_hms(stage2_total / n_runs_stage2)}/run)")
    peak_ram_seen = max(peak_ram_seen, estimate_ram_mb(base_cfg, "fedsaa"))

    # --- Totals ---
    grand_total = stage1_total + stage2_total
    print("\n" + "=" * 90)
    print(f"TOTAL estimated CPU wall-clock: {fmt_hms(grand_total)}  (~{grand_total/3600:.1f} hours, "
          f"~{grand_total/3600/24:.2f} days)")
    print(f"Estimated peak RAM (worst single run, sequential execution + freed between runs): "
          f"~{peak_ram_seen:.0f} MB (~{peak_ram_seen/1024:.2f} GB)")
    print("=" * 90)

    if grand_total > 3600 * 6:
        print("\nWARNING: this exceeds ~6 hours on this laptop. Recommended: run in the "
              "background with --resume enabled (checkpointed per method/seed/ablation-config), "
              "or offload to a stronger machine, or reduce rounds/#seeds/ablation grid further.")


if __name__ == "__main__":
    main()
