"""Evaluation metrics: per-client accuracy/macro-F1/AUC, and a global-test aggregator that
picks the right client's model (different #classes per specialty) per sample.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics.classification import accuracy, auc_ovr, macro_f1 as macro_f1_fn


@torch.no_grad()
def evaluate_client(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    return {
        "acc": accuracy(y_true, y_pred),
        "macro_f1": macro_f1_fn(y_true, y_pred),
        "auc": auc_ovr(y_true, probs),
    }


@torch.no_grad()
def evaluate_global(client_models: list, global_loader, device) -> dict:
    """Evaluates the global test set (all specialties concatenated), routing each sample to
    its own client's model/head since #classes differ per specialty.
    """
    for m in client_models:
        m.eval()

    correct, total = 0, 0
    for x, y, client_idx in global_loader:
        x = x.to(device)
        # batches can mix clients; score each sample with its own client's head
        for cid in torch.unique(client_idx).tolist():
            mask = client_idx == cid
            logits = client_models[cid](x[mask])
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y[mask]).sum().item()
            total += int(mask.sum().item())

    return {"acc": correct / total if total else float("nan")}
