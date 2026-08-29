"""CLI entrypoint. Dispatches to a method's run function by name.

Usage:
    python run.py --config configs/local_debug.yaml --method fedsaa
    python run.py --config configs/local_debug.yaml --method fedavg

No training logic lives here or is implemented yet -- this is scaffolding only.
Each method module (added in a later step) will expose a `run(cfg)` function.
"""

from __future__ import annotations

import argparse

from src.federated.centralized import run_centralized
from src.federated.fedsaa import run_fedsaa
from src.federated.local_only import run_local_only
from src.federated.loop import run_federated
from src.federated.server import fedavg_aggregate
from src.utils.config import load_config
from src.utils.seed import set_seed


def _run_fedavg(cfg: dict, seed: int) -> None:
    run_federated(cfg, seed, aggregate_fn=fedavg_aggregate, method_name="fedavg")


def _run_fedprox(cfg: dict, seed: int) -> None:
    mu = cfg["method"]["fedprox"]["mu"]
    run_federated(cfg, seed, aggregate_fn=fedavg_aggregate, method_name="fedprox", prox_mu=mu)


def _run_fedsaa(cfg: dict, seed: int) -> None:
    run_fedsaa(cfg, seed)


def _run_local_only(cfg: dict, seed: int) -> None:
    run_local_only(cfg, seed)


def _run_centralized(cfg: dict, seed: int) -> None:
    run_centralized(cfg, seed)


# Populated as each method is implemented in later steps.
METHOD_DISPATCH = {
    "fedsaa": _run_fedsaa,
    "fedavg": _run_fedavg,
    "fedprox": _run_fedprox,
    "local_only": _run_local_only,
    "centralized": _run_centralized,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated fine-tuning experiment runner")
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/)")
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHOD_DISPATCH.keys()),
        help="Which method to run; overrides method.name in the config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["method"]["name"] = args.method

    print(f"Loaded config: {args.config}")
    print(f"Method:        {args.method}")
    print(f"Seeds:         {cfg['seeds']}")
    print(f"Device:        {cfg['device']}")

    for seed in cfg["seeds"]:
        set_seed(seed, deterministic=cfg["reproducibility"]["deterministic"])

        run_fn = METHOD_DISPATCH[args.method]
        if run_fn is None:
            print(
                f"[seed={seed}] '{args.method}' is not implemented yet -- "
                f"scaffolding only (Step 2). Nothing to run."
            )
            continue

        run_fn(cfg, seed=seed)


if __name__ == "__main__":
    main()
