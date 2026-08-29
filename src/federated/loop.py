"""Full federated training loop: FedAvg baseline (FedSAA and others plug in via `aggregate_fn`).

Round structure:
  1. Server broadcasts the current global adapter state to all clients.
  2. Server samples a fraction C of clients to participate this round.
  3. Each sampled client trains locally for `local_epochs` epochs, returns its adapter state
     (+ #samples used, for sample-weighted aggregation).
  4. Server aggregates the collected adapter states -> new global adapter state.
  5. Every client's model is updated to the new global adapter state (their own head is
     untouched) and evaluated: per-client test set + the global (all-specialty) test set.
  6. One row is appended to results/<run>/metrics.csv.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from src.data.client_data import build_clients, build_global_test_loader
from src.federated.client import local_train
from src.federated.common import log_cka_this_round, rss_mb, run_dir, write_metrics_csv
from src.federated.server import fedavg_aggregate
from src.metrics.cka import build_probe_batch
from src.metrics.evaluation import evaluate_client, evaluate_global
from src.models.client_model import build_client_model
from src.models.lora_model import adapter_payload_mb, get_adapter_state, set_adapter_state


def run_federated(cfg: dict, seed: int, aggregate_fn=fedavg_aggregate, method_name: str = "fedavg",
                   prox_mu: float = 0.0):
    """Runs the full federated loop. `aggregate_fn(adapter_states, sample_counts) -> dict`
    lets Step 6+ swap in FedSAA without touching this orchestration. `prox_mu > 0` turns this
    into FedProx by adding a proximal term to each client's local objective.
    """
    device = torch.device(cfg["device"])
    fed_cfg = cfg["federated"]

    clients_data = build_clients(cfg, seed=seed)
    global_test_loader = build_global_test_loader(clients_data, cfg)

    client_models = [
        build_client_model(cfg, num_classes=cd.num_classes).to(device) for cd in clients_data
    ]

    global_adapter_state = get_adapter_state(client_models[0])
    payload_mb = adapter_payload_mb(global_adapter_state)

    out_dir = run_dir(cfg, method_name, seed)
    csv_path = os.path.join(out_dir, "metrics.csv")
    cka_dir = os.path.join(out_dir, "cka")
    probe_x = build_probe_batch(clients_data, n_per_client=4, seed=seed)
    print(f"Writing metrics to {csv_path}")
    print(f"Writing per-round client-CKA matrices (figure F7) to {cka_dir}")

    rng = np.random.RandomState(seed)
    num_clients = fed_cfg["num_clients"]
    num_sampled = max(1, int(round(fed_cfg["client_fraction"] * num_clients)))

    cumulative_mb = 0.0
    rows = []

    for round_idx in range(1, fed_cfg["rounds"] + 1):
        t0 = time.time()

        # 1. broadcast current global adapter to every client
        for m in client_models:
            set_adapter_state(m, global_adapter_state)

        # 2. sample participating clients
        participant_ids = sorted(rng.choice(num_clients, size=num_sampled, replace=False).tolist())

        # 3. local training on sampled clients
        collected_states, sample_counts = [], []
        for cid in participant_ids:
            adapter_state, n_samples, _ = local_train(
                client_models[cid], clients_data[cid].train_loader, cfg, device,
                prox_mu=prox_mu, global_adapter_state=global_adapter_state,
            )
            collected_states.append(adapter_state)
            sample_counts.append(n_samples)

        # 4. aggregate -> new global adapter state
        global_adapter_state = aggregate_fn(collected_states, sample_counts)

        # 5. broadcast the aggregated state and evaluate
        for m in client_models:
            set_adapter_state(m, global_adapter_state)

        per_client_acc, macro_f1s, aucs = [], [], []
        for cid, cd in enumerate(clients_data):
            res = evaluate_client(client_models[cid], cd.test_loader, device)
            per_client_acc.append(res["acc"])
            macro_f1s.append(res["macro_f1"])
            aucs.append(res["auc"])

        global_res = evaluate_global(client_models, global_test_loader, device)
        mean_cka = log_cka_this_round(client_models, probe_x, device, cka_dir, round_idx)

        # communication accounting: each participant uploads + downloads one adapter payload
        adapter_mb_round = num_sampled * payload_mb * 2
        cumulative_mb += adapter_mb_round
        wall_clock = time.time() - t0

        row = {
            "round": round_idx,
            "global_acc": global_res["acc"],
            "per_client_acc": ";".join(f"{a:.4f}" for a in per_client_acc),
            "macro_f1": np.nanmean(macro_f1s),
            "auc": np.nanmean(aucs),
            "adapter_MB_round": adapter_mb_round,
            "cumulative_MB": cumulative_mb,
            "wall_clock_sec": wall_clock,
            "ram_mb": rss_mb(),
        }
        rows.append(row)
        print(
            f"[round {round_idx}/{fed_cfg['rounds']}] global_acc={row['global_acc']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} auc={row['auc']:.4f} "
            f"comm={adapter_mb_round:.3f}MB (cum={cumulative_mb:.3f}MB) "
            f"time={wall_clock:.1f}s ram={row['ram_mb']:.1f}MB mean_cka={mean_cka:.4f}"
        )

    write_metrics_csv(csv_path, rows)
    return csv_path, rows
