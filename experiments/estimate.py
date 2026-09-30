"""Prints estimated wall-clock time and peak RAM for a full experiment sweep (method sweep x
seeds, + FedSAA ablation grid), broken down per stage and per dataset preset, WITHOUT running
anything. Use this before committing to `python -m experiments.runner --config <cfg>`.

Usage:
    python -m experiments.estimate --config configs/full_run.yaml
"""

from __future__ import annotations

import argparse
import copy

from experiments.runner import (
    ALL_METHODS,
    ALPHA_AXIS,
    ALPHA_SWEEP_N_SEEDS,
    FULL_ABLATION_GRID,
    _build_ablation_combos,
    select_dataset,
)
from experiments.runtime_estimate import GPU_SPEEDUP_FACTOR, estimate_seconds
from src.utils.config import load_config

# Rough peak-RAM-per-run estimates (MB), extrapolated from measured local_debug runs (Steps 5-10:
# 3 clients, 200-830 MB observed) scaled by client count and dataset-without-cap size. This is a
# planning number, not a guarantee -- actual peak depends on OS/Python overhead too, and is the
# same regardless of device (RAM here means host/process memory, not GPU VRAM).
BASE_RSS_MB = 350          # interpreter + torch + medmnist import overhead
PER_CLIENT_MODEL_MB = 55   # one resnet18+LoRA copy (unshared weights) per client model
PER_CLIENT_DATA_MB = 15    # DataLoader/Dataset object overhead per client (data itself is mmap'd)


def estimate_ram_mb(cfg: dict, method: str) -> float:
    n = cfg["federated"]["num_clients"]
    if method == "centralized":
        return BASE_RSS_MB + PER_CLIENT_MODEL_MB + n * PER_CLIENT_DATA_MB
    return BASE_RSS_MB + n * (PER_CLIENT_MODEL_MB + PER_CLIENT_DATA_MB)


def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def estimate_dataset(cfg: dict, dataset_label: str, device: str, verbose: bool) -> tuple[float, float]:
    """Returns (total_seconds, peak_ram_mb) for the full sweep (methods + ablation) on one
    dataset preset, at the given device ("cpu" or "gpu" -- see runtime_estimate.py's
    GPU_SPEEDUP_FACTOR heuristic disclaimer)."""
    method_seeds = cfg["seeds"]
    ablation_seeds = cfg["seeds"][:1]
    n_method_seeds = len(method_seeds)
    n_ablation_seeds = len(ablation_seeds)
    n_alpha_seeds = min(ALPHA_SWEEP_N_SEEDS, len(cfg["seeds"]))

    if verbose:
        print(f"\n--- Dataset {dataset_label} ({device.upper()}) ---")

    stage1_total, peak_ram = 0.0, 0.0
    for method in ALL_METHODS:
        per_run = estimate_seconds(cfg, method, device=device)
        stage1_total += per_run * n_method_seeds
        peak_ram = max(peak_ram, estimate_ram_mb(cfg, method))
        if verbose:
            print(f"  [Stage 1: methods]  {method:<13} {n_method_seeds} seed(s) x "
                  f"{fmt_hms(per_run)}/run = {fmt_hms(per_run * n_method_seeds)}")

    combos = _build_ablation_combos(FULL_ABLATION_GRID)
    stage2_total = 0.0
    for _lam, _tau, _rank, frac, _alpha in combos:
        combo_cfg = copy.deepcopy(cfg)
        combo_cfg["federated"]["client_fraction"] = frac
        stage2_total += estimate_seconds(combo_cfg, "fedsaa", device=device) * n_ablation_seeds
    if verbose:
        print(f"  [Stage 2: lambda/tau/rank ablation] {len(combos)} configs x {n_ablation_seeds} "
              f"seed = {fmt_hms(stage2_total)}")

    stage3_total = 0.0
    for _alpha in ALPHA_AXIS:
        stage3_total += estimate_seconds(cfg, "fedsaa", device=device) * n_alpha_seeds
    if verbose:
        print(f"  [Stage 3: alpha ablation] {len(ALPHA_AXIS)} configs x {n_alpha_seeds} seeds "
              f"= {fmt_hms(stage3_total)}")

    total = stage1_total + stage2_total + stage3_total
    if verbose:
        print(f"  Dataset {dataset_label} subtotal ({device.upper()}): {fmt_hms(total)}")
    return total, peak_ram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    presets = base_cfg.get("data", {}).get("dataset_presets")
    dataset_labels = list(presets.keys()) if presets else [None]

    print("=" * 90)
    print(f"Runtime estimate -- {args.config}  (seeds={base_cfg['seeds']}, "
          f"rounds={base_cfg['federated']['rounds']}, datasets={dataset_labels}, "
          f"configured device={base_cfg['device']})")
    print("=" * 90)

    cpu_grand_total, gpu_grand_total, peak_ram_seen = 0.0, 0.0, 0.0
    for label in dataset_labels:
        ds_cfg = select_dataset(base_cfg, label) if label else base_cfg
        cpu_total, ram = estimate_dataset(ds_cfg, label or "(single)", "cpu", verbose=True)
        gpu_total, _ram = estimate_dataset(ds_cfg, label or "(single)", "gpu", verbose=False)
        cpu_grand_total += cpu_total
        gpu_grand_total += gpu_total
        peak_ram_seen = max(peak_ram_seen, ram)
        print(f"  Dataset {label or '(single)'}: CPU {fmt_hms(cpu_total)}  |  "
              f"GPU (heuristic, {GPU_SPEEDUP_FACTOR:.0f}x) {fmt_hms(gpu_total)}")

    print("\n" + "=" * 90)
    print(f"TOTAL across {len(dataset_labels)} dataset(s):")
    print(f"  CPU wall-clock: {fmt_hms(cpu_grand_total)}  "
          f"(~{cpu_grand_total/3600:.1f} h, ~{cpu_grand_total/3600/24:.2f} days)")
    print(f"  GPU wall-clock (UNVERIFIED {GPU_SPEEDUP_FACTOR:.0f}x heuristic, no GPU available "
          f"to calibrate in this sandbox -- re-check against a real timed batch before trusting "
          f"it): {fmt_hms(gpu_grand_total)} (~{gpu_grand_total/3600:.1f} h, "
          f"~{gpu_grand_total/3600/24:.2f} days)")
    print(f"  Estimated peak host RAM (not GPU VRAM): ~{peak_ram_seen:.0f} MB "
          f"(~{peak_ram_seen/1024:.2f} GB)")
    print("=" * 90)

    if cpu_grand_total > 3600 * 6:
        print("\nCPU WARNING: exceeds ~6 hours on this laptop. Recommended: run in the "
              "background with --resume enabled, or use the GPU estimate above to plan an "
              "offloaded run (python -m experiments.runner --config <cfg> --dataset <A|B>).")


if __name__ == "__main__":
    main()
