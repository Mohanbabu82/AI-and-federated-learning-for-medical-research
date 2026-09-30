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

NOTE: the final aggregation step (experiments.aggregate.aggregate_all) refuses to summarize
anything that isn't a full_run_*-prefixed, full-round-count result (experiments/validation.py).
Running with --reduced (local_debug-style smoke tests) will therefore fail loudly at the
aggregation step by design -- that mode is for proving the run/log pipeline works, not for
producing paper-facing summaries/tables/figures. Use --skip-ablation with no --reduced flag if
you just want to smoke-test the method-sweep execution path against configs/full_run.yaml
without waiting for a full run.
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
# holding the other two (and client_fraction, alpha) at the default -- publication-defensible
# budget for this 8 GB CPU laptop (see experiments/estimate.py), replacing the full
# 5x3x4x2=120-config grid product used earlier. These points run at ABLATION_SEEDS (1 seed each)
# since they're one-at-a-time sensitivity checks, not headline numbers.
FULL_ABLATION_DEFAULT = {"lambda": 0.5, "tau": 0.5, "rank": 8, "client_fraction": 1.0, "alpha": 0.3}
FULL_ABLATION_AXES = {
    "lambda": [0, 0.25, 0.5, 0.75, 1],
    "tau": [0.1, 0.5, 1],
    "rank": [2, 4, 8, 16],
}
FULL_ABLATION_GRID = {"_mode": "oat", "default": FULL_ABLATION_DEFAULT, "axes": FULL_ABLATION_AXES}

# Local adapter retention ablation: alpha in {0, 0.1, 0.3, 0.5} at the same lambda/tau/rank/
# client_fraction default. Run at more seeds than the other OAT axes (3, not 1) since retention
# directly affects the paper's headline personalization claim -- see Step 15/16.
ALPHA_AXIS = [0, 0.1, 0.3, 0.5]
ALPHA_SWEEP_N_SEEDS = 3

REDUCED_ABLATION_GRID = {
    "lambda": [0, 0.5, 1],
    "tau": [0.1],
    "rank": [4],
    "client_fraction": [1.0],
    "alpha": [0.3],
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
    """One-at-a-time combos: the default point once, plus each axis swept with the others held
    at default -- deduplicated so the shared default point isn't run multiple times."""
    keys = ["lambda", "tau", "rank", "client_fraction", "alpha"]
    seen: dict[tuple, None] = {}
    seen[tuple(default[k] for k in keys)] = None
    for axis, values in axes.items():
        for v in values:
            combo = dict(default)
            combo[axis] = v
            seen[tuple(combo[k] for k in keys)] = None
    return list(seen.keys())


def _build_ablation_combos(grid: dict) -> list[tuple]:
    """Returns (lambda, tau, rank, client_fraction, alpha) 5-tuples."""
    if grid.get("_mode") == "oat":
        return _oat_combos(grid["default"], grid["axes"])
    alphas = grid.get("alpha", [0.3])
    return list(itertools.product(grid["lambda"], grid["tau"], grid["rank"],
                                   grid["client_fraction"], alphas))


def _build_alpha_ablation_plan(base_cfg: dict, default: dict, alpha_values: list, seeds: list[int]) -> list[dict]:
    """Separate ablation axis: alpha in `alpha_values`, all other fedsaa hyperparams held at
    `default`, each config run across `seeds` (more than the 1-seed lambda/tau/rank OAT points,
    since retention directly affects the paper's headline personalization claim)."""
    plan = []
    for alpha in alpha_values:
        tag = f"{base_cfg['experiment_name']}_fedsaa_alpha{alpha}"
        overrides = {**default, "alpha": alpha}
        for seed in seeds:
            plan.append({
                "run_key": f"{tag}:seed{seed}", "kind": "ablation", "method": "fedsaa",
                "seed": seed, "tag": tag, "overrides": overrides,
            })
    return plan


def plan_sweep(base_cfg: dict, methods: list[str], method_seeds: list[int],
               ablation_seeds: list[int], grid: dict, skip_ablation: bool,
               alpha_axis: list | None = None, alpha_seeds: list[int] | None = None) -> list[dict]:
    """Builds the full list of planned runs (method sweep + ablation grid) as
    {run_key, kind, method, seed, cfg_overrides} without executing anything -- used both to
    know the total run count up front (for ETA) and to drive execution. Main methods run
    `method_seeds` (never below 3 for main results); lambda/tau/rank ablation points run
    `ablation_seeds` (1 seed) since they're one-at-a-time sensitivity checks, not headline
    numbers; the alpha (retention) axis runs `alpha_seeds` (3 seeds) separately.
    """
    plan = []
    for method in methods:
        for seed in method_seeds:
            plan.append({"run_key": f"{method}:seed{seed}", "kind": "method",
                         "method": method, "seed": seed, "overrides": {}})

    if not skip_ablation:
        for lam, tau, rank, frac, alpha in _build_ablation_combos(grid):
            # prefixed with the base experiment_name (e.g. "full_run") so ablation run
            # directories are recognized as full_run_* by experiments.validation, not silently
            # excluded from summaries/tables/figures the way local_debug_*/test_run_* runs are
            tag = f"{base_cfg['experiment_name']}_fedsaa_l{lam}_t{tau}_r{rank}_c{frac}"
            for seed in ablation_seeds:
                plan.append({
                    "run_key": f"{tag}:seed{seed}", "kind": "ablation", "method": "fedsaa",
                    "seed": seed, "tag": tag,
                    "overrides": {"lambda": lam, "tau": tau, "rank": rank,
                                  "client_fraction": frac, "alpha": alpha},
                })

        if alpha_axis and alpha_seeds:
            default = grid["default"] if grid.get("_mode") == "oat" else FULL_ABLATION_DEFAULT
            plan += _build_alpha_ablation_plan(base_cfg, default, alpha_axis, alpha_seeds)

    return plan


def _apply_overrides(cfg: dict, overrides: dict, tag: str | None) -> dict:
    cfg = copy.deepcopy(cfg)
    if overrides:
        cfg["method"]["fedsaa"]["lambda"] = overrides["lambda"]
        cfg["method"]["fedsaa"]["tau"] = overrides["tau"]
        cfg["method"]["fedsaa"]["svd_rank"] = overrides["rank"]
        cfg["model"]["lora"]["r"] = overrides["rank"]
        cfg["federated"]["client_fraction"] = overrides["client_fraction"]
        cfg["method"]["fedsaa"]["alpha"] = overrides["alpha"]
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
            "alpha": overrides.get("alpha"),
        })
    return records


def select_dataset(cfg: dict, dataset_key: str) -> dict:
    """Switches `cfg` to one of `data.dataset_presets` (e.g. "A" or "B"), re-deriving
    data.specialties and federated.num_clients, and tagging experiment_name so Dataset A and B
    runs land in distinctly-named (still full_run_*-prefixed) result directories."""
    presets = cfg.get("data", {}).get("dataset_presets")
    if not presets:
        raise ValueError(f"{cfg.get('experiment_name')}: config has no data.dataset_presets to select from")
    if dataset_key not in presets:
        raise ValueError(f"Unknown --dataset '{dataset_key}', expected one of {list(presets)}")

    cfg = copy.deepcopy(cfg)
    cfg["data"]["active_dataset"] = dataset_key
    cfg["data"]["specialties"] = presets[dataset_key]
    cfg["federated"]["num_clients"] = len(presets[dataset_key])
    cfg["experiment_name"] = f"{cfg['experiment_name']}_{dataset_key}"
    return cfg


def load_checkpoint_csv(results_dir: str, run_key: str) -> str:
    from experiments.checkpoint import load_checkpoint
    return load_checkpoint(results_dir)["completed"][run_key]


def print_all_estimates(base_cfg: dict, methods: list[str], grid: dict, skip_ablation: bool = False,
                         alpha_axis: list | None = None) -> None:
    for method in methods:
        print(format_estimate(base_cfg, method, label="method sweep"))
    if skip_ablation:
        return
    for lam, tau, rank, frac, _alpha in _build_ablation_combos(grid):
        cfg = copy.deepcopy(base_cfg)
        cfg["federated"]["client_fraction"] = frac
        tag = f"{base_cfg['experiment_name']}_fedsaa_l{lam}_t{tau}_r{rank}_c{frac}"
        print(format_estimate(cfg, "fedsaa", label=tag))
    if alpha_axis:
        for alpha in alpha_axis:
            tag = f"{base_cfg['experiment_name']}_fedsaa_alpha{alpha}"
            print(format_estimate(base_cfg, "fedsaa", label=tag))


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
    parser.add_argument("--dataset", default=None,
                         help="Select a data.dataset_presets key (e.g. A or B) when the config "
                              "defines multiple specialty mixes. Runs one dataset per invocation "
                              "-- to cover both, run this command twice with --dataset A and "
                              "--dataset B. Omit for configs with a single fixed specialties list.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    assert base_cfg["data"]["num_workers"] == 0, "num_workers must be 0 on this laptop"

    if args.dataset:
        base_cfg = select_dataset(base_cfg, args.dataset)

    if args.reduced:
        base_cfg = copy.deepcopy(base_cfg)
        base_cfg["seeds"] = base_cfg["seeds"][:1]
        base_cfg["federated"]["rounds"] = args.rounds_override or 2
        grid = REDUCED_ABLATION_GRID
        alpha_axis = None  # skip the 3-seed alpha sweep in smoke-test mode
    else:
        if args.rounds_override:
            base_cfg["federated"]["rounds"] = args.rounds_override
        grid = FULL_ABLATION_GRID
        alpha_axis = ALPHA_AXIS

    method_seeds = base_cfg["seeds"]              # main results: never below 3 seeds
    ablation_seeds = base_cfg["seeds"][:1]         # one-at-a-time lambda/tau/rank points: 1 seed
    alpha_seeds = base_cfg["seeds"][:ALPHA_SWEEP_N_SEEDS] if alpha_axis else []  # alpha axis: 3 seeds

    print("=" * 90)
    print(f"Experiment sweep -- config={args.config}  reduced={args.reduced}  "
          f"method_seeds={method_seeds}  ablation_seeds={ablation_seeds}  "
          f"alpha_axis={alpha_axis}  alpha_seeds={alpha_seeds}  "
          f"rounds={base_cfg['federated']['rounds']}  resume={args.resume}")
    print("=" * 90)

    print_all_estimates(base_cfg, ALL_METHODS, grid, skip_ablation=args.skip_ablation,
                         alpha_axis=alpha_axis)

    plan = plan_sweep(base_cfg, ALL_METHODS, method_seeds, ablation_seeds, grid, args.skip_ablation,
                       alpha_axis=alpha_axis, alpha_seeds=alpha_seeds)
    print(f"\nTotal planned runs: {len(plan)}")

    records = run_sweep(base_cfg, plan, resume=args.resume)

    summary_dir = os.path.join(base_cfg["logging"]["results_dir"], "summary")
    os.makedirs(summary_dir, exist_ok=True)
    # expected_rounds enforces experiments.validation's full_run_* + full-round-count check on
    # every record before it can land in a summary CSV -- see Step 14's DebugRunError guard.
    methods_csv, ablation_csv = aggregate_all(records, summary_dir,
                                               expected_rounds=base_cfg["federated"]["rounds"])

    print("=" * 90)
    print(f"Wrote method summary:   {methods_csv}")
    print(f"Wrote ablation summary: {ablation_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
