"""Local-only baseline (lower bound): each client trains alone on its own specialty data,
never communicating with anyone. Same per-round metrics.csv schema, with adapter_MB always 0.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from src.data.client_data import build_clients, build_global_test_loader
from src.federated.client import local_train
from src.federated.common import rss_mb, run_dir, write_metrics_csv
from src.metrics.evaluation import evaluate_client, evaluate_global
from src.models.client_model import build_client_model


def run_local_only(cfg: dict, seed: int):
    device = torch.device(cfg["device"])
    fed_cfg = cfg["federated"]

    clients_data = build_clients(cfg, seed=seed)
    global_test_loader = build_global_test_loader(clients_data, cfg)
    client_models = [
        build_client_model(cfg, num_classes=cd.num_classes).to(device) for cd in clients_data
    ]

    csv_path = os.path.join(run_dir(cfg, "local_only", seed), "metrics.csv")
    print(f"Writing metrics to {csv_path}")

    rows = []
    for round_idx in range(1, fed_cfg["rounds"] + 1):
        t0 = time.time()

        # each client trains independently on its own data -- no broadcast, no aggregation
        for cid, cd in enumerate(clients_data):
            local_train(client_models[cid], cd.train_loader, cfg, device)

        per_client_acc, macro_f1s, aucs = [], [], []
        for cid, cd in enumerate(clients_data):
            res = evaluate_client(client_models[cid], cd.test_loader, device)
            per_client_acc.append(res["acc"])
            macro_f1s.append(res["macro_f1"])
            aucs.append(res["auc"])

        global_res = evaluate_global(client_models, global_test_loader, device)
        wall_clock = time.time() - t0

        row = {
            "round": round_idx,
            "global_acc": global_res["acc"],
            "per_client_acc": ";".join(f"{a:.4f}" for a in per_client_acc),
            "macro_f1": np.nanmean(macro_f1s),
            "auc": np.nanmean(aucs),
            "adapter_MB_round": 0.0,   # no communication ever happens in local-only
            "cumulative_MB": 0.0,
            "wall_clock_sec": wall_clock,
            "ram_mb": rss_mb(),
        }
        rows.append(row)
        print(
            f"[round {round_idx}/{fed_cfg['rounds']}] global_acc={row['global_acc']:.4f} "
            f"mean_client_acc={np.mean(per_client_acc):.4f} worst_client_acc={min(per_client_acc):.4f} "
            f"macro_f1={row['macro_f1']:.4f} auc={row['auc']:.4f} time={wall_clock:.1f}s "
            f"ram={row['ram_mb']:.1f}MB"
        )

    write_metrics_csv(csv_path, rows)

    final_per_client = [float(a) for a in rows[-1]["per_client_acc"].split(";")]
    print(f"Local-only final: mean_client_acc={np.mean(final_per_client):.4f} "
          f"worst_client_acc={min(final_per_client):.4f}")

    return csv_path, rows
