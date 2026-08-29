"""Experiment runner: sweeps all methods x seeds, plus the FedSAA ablation grid.

Usage:
    python -m experiments.runner --config configs/local_debug.yaml --reduced
    python -m experiments.runner --config configs/full_run.yaml --resume

--reduced overrides rounds/seeds/ablation-grid to a tiny smoke-test set so the whole pipeline
(run -> log -> aggregate) can be proven quickly on this laptop before committing to a full sweep.

--resume skips any (method/ablation-config, seed) run already recorded as complete in
results/_runner_checkpoint.json, so an interrupted full_run.yaml sweep (see
`python -m experiments.estimate` for why that matters -- it can take days) can be restarted
without redoing finished work. Progress + ETA are logged to results/runner_progress.log after
every run, and memory is freed (gc.collect()) between runs to keep this 8 GB laptop stable over
a long sequential sweep.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import time

from experiments.aggregate import aggregate_all
from experiments.checkpoint import ProgressTracker, is_complete, mark_complete
from experiments.runtime_estimate import format_estimate
from src.federated.centralized import run_centralized
from src.federated.fedsaa import run_fedsaa
from src.federated.local_only import run_local_only
from src.federated.loop import run_federated
from src.federated.server import fedavg_aggregate
from src.utils.config import load_config
from src.utils.seed import set_seed

ALL_METHODS = ["centralized", "local_only", "fedavg", "fedprox", "fedsaa"]

# One-at-a-time ablation around a fixed default: vary exactly one of lambda/tau/rank while
# holding the other two (and client_fraction) at the default -- publication-defensible budget
# for this 8 GB CPU laptop (see experiments/estimate.py), replacing the full
# 5x3x4x2=120-config grid product used earlier.
FULL_ABLATION_DEFAULT = {"lambda": 0.5, "tau": 0.5, "rank": 8, "client_fraction": 1.0}
FULL_ABLATION_AXES = {
    "lambda": [0, 0.25, 0.5, 0.75, 1],
    "tau": [0.1, 0.5, 1],
    "rank": [2, 4, 8, 16],
}
FULL_ABLATION_GRID = {"_mode": "oat", "default": FULL_ABLATION_DEFAULT, "axes": FULL_ABLATION_AXES}

REDUCED_ABLATION_GRID = {
    "lambda": [0, 0.5, 1],
    "tau": [0.1],
    "rank": [4],
    "client_fraction": [1.0],
}


def _run_method(cfg: dict, seed: int, method: str):
    assert cfg["data"]["num_workers"] == 0, (
        "num_workers must stay 0 on this laptop -- worker subprocesses multiply RAM usage "
        "for no benefit on CPU-only training."
    )
    if method == "centralized":
        return run_centralized(cfg, seed)
    if method == "local_only":
        return run_local_only(cfg, seed)
    if method == "fedavg":
        return run_federated(cfg, seed, aggregate_fn=fedavg_aggregate, method_name="fedavg")
    if method == "fedprox":
        mu = cfg["method"]["fedprox"]["mu"]
        return run_federated(cfg, seed, aggregate_fn=fedavg_aggregate, method_name="fedprox", prox_mu=mu)
    if method == "fedsaa":
        return run_fedsaa(cfg, seed)
    raise ValueError(f"Unknown method '{method}'")


def _oat_combos(default: dict, axes: dict) -> list[tuple]:
    """One-at-a-time combos: the default point once, plus each axis swept with the other two
    held at default -- deduplicated so the shared default point isn't run multiple times."""
    keys = ["lambda", "tau", "rank", "client_fraction"]
    seen: dict[tuple, None] = {}
    seen[tuple(default[k] for k in keys)] = None
    for axis, values in axes.items():
        for v in values:
            combo = dict(default)
            combo[axis] = v
            seen[tuple(combo[k] for k in keys)] = None
    return list(seen.keys())


def _build_ablation_combos(grid: dict) -> list[tuple]:
    if grid.get("_mode") == "oat":
        return _oat_combos(grid["default"], grid["axes"])
    return list(itertools.product(grid["lambda"], grid["tau"], grid["rank"], grid["client_fraction"]))


def plan_sweep(base_cfg: dict, methods: list[str], method_seeds: list[int],
               ablation_seeds: list[int], grid: dict, skip_ablation: bool) -> list[dict]:
    """Builds the full list of planned runs (method sweep + ablation grid) as
    {run_key, kind, method, seed, cfg_overrides} without executing anything -- used both to
    know the total run count up front (for ETA) and to drive execution. Main methods run
    `method_seeds` (never below 3 for main results); ablation points run `ablation_seeds`
    (1 seed) since they're one-at-a-time sensitivity checks, not headline numbers.
    """
    plan = []
    for method in methods:
        for seed in method_seeds:
            plan.append({"run_key": f"{method}:seed{seed}", "kind": "method",
                         "method": method, "seed": seed, "overrides": {}})

    if not skip_ablation:
        for lam, tau, rank, frac in _build_ablation_combos(grid):
            tag = f"fedsaa_l{lam}_t{tau}_r{rank}_c{frac}"
            for seed in ablation_seeds:
                plan.append({
                    "run_key": f"{tag}:seed{seed}", "kind": "ablation", "method": "fedsaa",
                    "seed": seed, "tag": tag,
                    "overrides": {"lambda": lam, "tau": tau, "rank": rank, "client_fraction": frac},
                })
    return plan


def _apply_overrides(cfg: dict, overrides: dict, tag: str | None) -> dict:
    cfg = copy.deepcopy(cfg)
    if overrides:
        cfg["method"]["fedsaa"]["lambda"] = overrides["lambda"]
        cfg["method"]["fedsaa"]["tau"] = overrides["tau"]
        cfg["method"]["fedsaa"]["svd_rank"] = overrides["rank"]
        cfg["model"]["lora"]["r"] = overrides["rank"]
        cfg["federated"]["client_fraction"] = overrides["client_fraction"]
        cfg["experiment_name"] = tag
    return cfg


def run_sweep(base_cfg: dict, plan: list[dict], resume: bool) -> list[dict]:
    results_dir = base_cfg["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    already_done = sum(1 for p in plan if resume and is_complete(results_dir, p["run_key"]))
    tracker = ProgressTracker(results_dir, total_runs=len(plan), already_done=already_done)

    records = []
    for item in plan:
        run_key = item["run_key"]

        if resume and is_complete(results_dir, run_key):
            print(f"[resume] skipping already-complete run: {run_key}")
            tracker.record_run(run_key, elapsed_sec=0.0, skipped=True)
            csv_path = load_checkpoint_csv(results_dir, run_key)
        else:
            cfg = _apply_overrides(base_cfg, item.get("overrides", {}), item.get("tag"))
            set_seed(item["seed"], deterministic=cfg["reproducibility"]["deterministic"])

            print(f"\n=== Running {run_key} ===")
            t0 = time.time()
            csv_path, _rows = _run_method(cfg, item["seed"], item["method"])
            elapsed = time.time() - t0
            print(f"=== Finished {run_key} in {elapsed:.1f}s -> {csv_path} ===")

            mark_complete(results_dir, run_key, csv_path)
            tracker.record_run(run_key, elapsed_sec=elapsed)

        overrides = item.get("overrides", {})
        records.append({
            "group": item.get("tag", item["method"]), "method": item["method"], "seed": item["seed"],
            "csv_path": csv_path,
            "lambda": overrides.get("lambda"), "tau": overrides.get("tau"),
            "rank": overrides.get("rank"), "client_fraction": overrides.get("client_fraction",
                                            base_cfg["federated"]["client_fraction"]),
        })
    return records


def load_checkpoint_csv(results_dir: str, run_key: str) -> str:
    from experiments.checkpoint import load_checkpoint
    return load_checkpoint(results_dir)["completed"][run_key]


def print_all_estimates(base_cfg: dict, methods: list[str], grid: dict, skip_ablation: bool = False) -> None:
    for method in methods:
        print(format_estimate(base_cfg, method, label="method sweep"))
    if skip_ablation:
        return
    for lam, tau, rank, frac in _build_ablation_combos(grid):
        cfg = copy.deepcopy(base_cfg)
        cfg["federated"]["client_fraction"] = frac
        tag = f"fedsaa_l{lam}_t{tau}_r{rank}_c{frac}"
        print(format_estimate(cfg, "fedsaa", label=tag))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reduced", action="store_true",
                         help="Tiny smoke-test sweep: 1 seed, few rounds, small ablation grid.")
    parser.add_argument("--rounds-override", type=int, default=None,
                         help="Force a specific #rounds for every run (useful with --reduced).")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--resume", action="store_true",
                         help="Skip any (method/ablation-config, seed) run already recorded "
                              "complete in results/_runner_checkpoint.json.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    assert base_cfg["data"]["num_workers"] == 0, "num_workers must be 0 on this laptop"

    if args.reduced:
        base_cfg = copy.deepcopy(base_cfg)
        base_cfg["seeds"] = base_cfg["seeds"][:1]
        base_cfg["federated"]["rounds"] = args.rounds_override or 2
        grid = REDUCED_ABLATION_GRID
    else:
        if args.rounds_override:
            base_cfg["federated"]["rounds"] = args.rounds_override
        grid = FULL_ABLATION_GRID

    method_seeds = base_cfg["seeds"]              # main results: never below 3 seeds
    ablation_seeds = base_cfg["seeds"][:1]         # one-at-a-time ablation points: 1 seed each

    print("=" * 90)
    print(f"Experiment sweep -- config={args.config}  reduced={args.reduced}  "
          f"method_seeds={method_seeds}  ablation_seeds={ablation_seeds}  "
          f"rounds={base_cfg['federated']['rounds']}  resume={args.resume}")
    print("=" * 90)

    print_all_estimates(base_cfg, ALL_METHODS, grid, skip_ablation=args.skip_ablation)

    plan = plan_sweep(base_cfg, ALL_METHODS, method_seeds, ablation_seeds, grid, args.skip_ablation)
    print(f"\nTotal planned runs: {len(plan)}")

    records = run_sweep(base_cfg, plan, resume=args.resume)

    summary_dir = os.path.join(base_cfg["logging"]["results_dir"], "summary")
    os.makedirs(summary_dir, exist_ok=True)
    methods_csv, ablation_csv = aggregate_all(records, summary_dir)

    print("=" * 90)
    print(f"Wrote method summary:   {methods_csv}")
    print(f"Wrote ablation summary: {ablation_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
