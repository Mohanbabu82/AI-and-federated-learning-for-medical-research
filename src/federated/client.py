"""Client-side local training: E epochs of LoRA fine-tuning, returns the adapter payload."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lora_model import get_adapter_state


def local_train(model: nn.Module, train_loader, cfg: dict, device: torch.device,
                 prox_mu: float = 0.0, global_adapter_state: dict | None = None):
    """Trains `model` (its LoRA params + local head) for federated.local_epochs epochs.

    If prox_mu > 0 and global_adapter_state is given, adds a FedProx proximal term
    pulling the LoRA params back toward the broadcast global adapter state.

    Returns (adapter_state, n_samples, avg_train_loss).
    """
    fed_cfg = cfg["federated"]
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=fed_cfg["lr"],
        momentum=fed_cfg["momentum"],
        weight_decay=fed_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    if prox_mu > 0 and global_adapter_state is not None:
        global_state_dev = {k: v.to(device) for k, v in global_adapter_state.items()}
        named_params = dict(model.named_parameters())

    total_loss, n_batches, n_samples = 0.0, 0, 0
    for _ in range(fed_cfg["local_epochs"]):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if prox_mu > 0 and global_adapter_state is not None:
                prox_term = sum(
                    (named_params[k] - global_state_dev[k]).pow(2).sum()
                    for k in global_state_dev
                )
                loss = loss + (prox_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_batches += 1
            n_samples += x.size(0)

    avg_loss = total_loss / n_samples if n_samples else float("nan")
    return get_adapter_state(model), n_samples, avg_loss


def local_adapt_steps(model: nn.Module, train_loader, cfg: dict, device: torch.device,
                       num_steps: int) -> float:
    """A brief post-aggregation local fine-tuning pass: `num_steps` SGD mini-batches (not full
    epochs) on `model`'s current (already-personalized) adapter + head, in place. Used after
    FedSAA's server-side aggregation so each client actually adapts to its own data again
    before evaluation -- otherwise "personalization" only happens at the server, never locally.

    Returns the average training loss over the steps actually taken (0 if num_steps <= 0).
    """
    if num_steps <= 0:
        return 0.0

    fed_cfg = cfg["federated"]
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=fed_cfg["lr"],
        momentum=fed_cfg["momentum"],
        weight_decay=fed_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    total_loss, n_steps_taken = 0.0, 0
    while n_steps_taken < num_steps:
        for x, y in train_loader:
            if n_steps_taken >= num_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps_taken += 1

    return total_loss / n_steps_taken if n_steps_taken else 0.0
