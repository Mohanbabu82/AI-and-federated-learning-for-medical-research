"""Rough CPU wall-clock estimate for a config, so you can judge what to run locally vs offload.

Calibrated from the Step 5/6/7 measurements on this laptop (i5-1235U, CPU-only): ~50s/round for
3 clients x 1 local_epoch x ~56 batches/client (batch_size=32, ~1800 train images/client) ->
roughly 0.3s/batch of LoRA local training (forward+backward on a frozen resnet18 backbone).
"""

from __future__ import annotations

import math

SEC_PER_BATCH = 0.3     # calibrated from local_debug FedAvg runs (Step 5)
EVAL_OVERHEAD_SEC = 3.0  # rough per-round cost of per-client + global evaluation


def _train_size_estimate(cfg: dict) -> int:
    """Approximates each client's train set size without touching the dataset (fast estimate
    for planning). Uses the configured cap when set; otherwise a conservative guess."""
    cap = cfg["data"]["subsample_cap"]
    val_fraction = cfg["data"]["val_fraction"]
    if cap is not None:
        return int(cap * (1 - val_fraction))
    # No cap (full_run.yaml): smallest MedMNIST specialty here is DermaMNIST (~7000 train images);
    # use a representative ~50k average across the 5 specialties as a planning estimate.
    return int(50_000 * (1 - val_fraction))


def estimate_seconds(cfg: dict, method: str) -> float:
    fed_cfg = cfg["federated"]
    rounds = fed_cfg["rounds"]
    num_clients = fed_cfg["num_clients"]
    local_epochs = fed_cfg["local_epochs"]
    batch_size = cfg["data"]["batch_size"]
    train_size = _train_size_estimate(cfg)
    batches_per_client_epoch = math.ceil(train_size / batch_size)

    if method == "local_only":
        participants = num_clients  # every client trains every round, never communicates
    elif method == "centralized":
        # one pass over ALL pooled data per local_epoch, per round
        total_batches = math.ceil((train_size * num_clients) / batch_size) * local_epochs
        return rounds * (total_batches * SEC_PER_BATCH + EVAL_OVERHEAD_SEC)
    else:  # fedavg, fedprox, fedsaa
        participants = max(1, round(fed_cfg["client_fraction"] * num_clients))

    total_batches = participants * local_epochs * batches_per_client_epoch
    return rounds * (total_batches * SEC_PER_BATCH + EVAL_OVERHEAD_SEC)


def format_estimate(cfg: dict, method: str, label: str = "") -> str:
    seconds = estimate_seconds(cfg, method)
    minutes = seconds / 60.0
    tag = f"[{label}] " if label else ""
    if minutes < 3:
        verdict = "OK to run locally"
    elif minutes < 20:
        verdict = "runnable locally, but slow -- consider a coffee break"
    else:
        verdict = "SLOW: consider offloading to a stronger machine / reducing rounds/clients"
    return f"{tag}method={method}: ~{minutes:.1f} min estimated ({seconds:.0f}s) -- {verdict}"
