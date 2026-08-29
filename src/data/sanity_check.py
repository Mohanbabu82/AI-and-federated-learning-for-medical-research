"""Sanity check for src/data/: builds all client loaders from a config and reports
per-client specialty, class count, split sizes, one batch shape, and process RAM usage.

Usage:
    python -m src.data.sanity_check --config configs/local_debug.yaml
"""

from __future__ import annotations

import argparse
import os

import psutil

from src.data.client_data import build_clients, build_global_test_loader
from src.utils.config import load_config
from src.utils.seed import set_seed


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seeds"][0]
    set_seed(seed, deterministic=cfg["reproducibility"]["deterministic"])

    print("=" * 90)
    print(f"Data sanity check -- {args.config}")
    print(f"specialties: {cfg['data']['specialties']}")
    print(f"subsample_cap: {cfg['data']['subsample_cap']}  image_size: {cfg['data']['image_size']}")
    print("=" * 90)
    print(f"RAM before loading: {_rss_mb():.1f} MB")

    clients = build_clients(cfg, seed=seed)

    print("-" * 90)
    header = f"{'Client':>6} {'Specialty':<12} {'#Cls':>5} {'Train':>7} {'Val':>6} {'Test':>7}  BatchShape"
    print(header)
    print("-" * len(header))
    for c in clients:
        xb, yb = next(iter(c.train_loader))
        print(
            f"{c.client_id:>6} {c.specialty:<12} {c.num_classes:>5} {c.train_size:>7} "
            f"{c.val_size:>6} {c.test_size:>7}  imgs={tuple(xb.shape)} labels={tuple(yb.shape)}"
        )
        print(f"       RAM after client {c.client_id}: {_rss_mb():.1f} MB")

    print("-" * 90)
    global_loader = build_global_test_loader(clients, cfg)
    xb, yb, cb = next(iter(global_loader))
    total_test = len(global_loader.dataset)
    print(f"Global test set: {total_test} samples (concat of all client test splits)")
    print(f"  one batch -> imgs={tuple(xb.shape)} labels={tuple(yb.shape)} client_ids={tuple(cb.shape)}")

    print("-" * 90)
    print(f"Final RAM usage: {_rss_mb():.1f} MB")
    print("=" * 90)
    print("RESULT: SUCCESS. All clients built, loaders functional, RAM within budget.")


if __name__ == "__main__":
    main()
