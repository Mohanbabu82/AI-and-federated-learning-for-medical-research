"""Centralized baseline (upper bound): all clients' data pooled, ONE shared frozen-backbone+LoRA
model trained jointly (each specialty still uses its own classifier head, since #classes differ,
but the head selection happens within a single training process -- nothing is ever communicated
over a network). Same per-round metrics.csv schema; adapter_MB is always 0 (no federation).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.data.client_data import build_clients, build_global_test_loader
from src.data.specialty_dataset import TaggedConcatDataset
from src.federated.common import rss_mb, run_dir, write_metrics_csv
from src.metrics.evaluation import evaluate_client, evaluate_global
from src.models.client_model import ClientModel, build_shared_lora_backbone


def run_centralized(cfg: dict, seed: int):
    device = torch.device(cfg["device"])
    fed_cfg = cfg["federated"]
    data_cfg = cfg["data"]

    clients_data = build_clients(cfg, seed=seed)
    global_test_loader = build_global_test_loader(clients_data, cfg)

    shared_backbone, feature_dim = build_shared_lora_backbone(cfg)
    shared_backbone.to(device)
    client_models = [
        ClientModel(shared_backbone, feature_dim, cd.num_classes).to(device) for cd in clients_data
    ]

    pooled_train = TaggedConcatDataset([cd.train_loader.dataset for cd in clients_data])
    pooled_loader = torch.utils.data.DataLoader(
        pooled_train, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"],
    )

    trainable_params = [p for p in shared_backbone.parameters() if p.requires_grad]
    for m in client_models:
        trainable_params += list(m.head.parameters())
    optimizer = torch.optim.SGD(
        trainable_params, lr=fed_cfg["lr"], momentum=fed_cfg["momentum"],
        weight_decay=fed_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    csv_path = os.path.join(run_dir(cfg, "centralized", seed), "metrics.csv")
    print(f"Writing metrics to {csv_path}")

    rows = []
    for round_idx in range(1, fed_cfg["rounds"] + 1):
        t0 = time.time()

        for m in client_models:
            m.train()
        for _ in range(fed_cfg["local_epochs"]):
            for x, y, cid in pooled_loader:
                x, y, cid = x.to(device), y.to(device), cid.to(device)
                optimizer.zero_grad()
                features = shared_backbone(x)
                loss = 0.0
                for c in torch.unique(cid).tolist():
                    mask = cid == c
                    logits = client_models[c].head(features[mask])
                    loss = loss + criterion(logits, y[mask]) * mask.sum()
                loss = loss / x.size(0)
                loss.backward()
                optimizer.step()

        per_client_acc, macro_f1s, aucs = [], [], []
        for cid_i, cd in enumerate(clients_data):
            res = evaluate_client(client_models[cid_i], cd.test_loader, device)
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
            "adapter_MB_round": 0.0,   # single co-located model, nothing is ever communicated
            "cumulative_MB": 0.0,
            "wall_clock_sec": wall_clock,
            "ram_mb": rss_mb(),
        }
        rows.append(row)
        print(
            f"[round {round_idx}/{fed_cfg['rounds']}] global_acc={row['global_acc']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} auc={row['auc']:.4f} time={wall_clock:.1f}s "
            f"ram={row['ram_mb']:.1f}MB"
        )

    write_metrics_csv(csv_path, rows)
    return csv_path, rows
