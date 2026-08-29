"""Linear CKA (Centered Kernel Alignment) between clients' backbone features, computed on a
small shared probe batch -- used to track representation collapse across rounds (figure F7,
FedSAA vs FedAvg). The probe batch is kept tiny so this stays well within the 8 GB budget.
"""

from __future__ import annotations

import numpy as np
import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two (n_samples, n_features) feature matrices from the SAME n_samples
    probe inputs. 1.0 = identical representations (up to rotation/scale), 0.0 = unrelated.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    xty = X.T @ Y
    numerator = (xty ** 2).sum()

    xtx = X.T @ X
    yty = Y.T @ Y
    denom = torch.sqrt((xtx ** 2).sum()) * torch.sqrt((yty ** 2).sum())

    if denom.item() < 1e-12:
        return float("nan")
    return float((numerator / denom).item())


def build_probe_batch(clients_data, n_per_client: int, seed: int) -> torch.Tensor:
    """Small, fixed probe batch shared across all clients/rounds: a handful of images per
    client's test set, concatenated. Kept tiny (n_clients * n_per_client images) for RAM.
    """
    rng = np.random.RandomState(seed)
    images = []
    for cd in clients_data:
        dataset = cd.test_loader.dataset
        n = len(dataset)
        idx = rng.choice(n, size=min(n_per_client, n), replace=False)
        for i in idx:
            x, _y = dataset[int(i)]
            images.append(x)
    return torch.stack(images, dim=0)


@torch.no_grad()
def compute_cka_matrix(client_models: list, probe_x: torch.Tensor, device) -> np.ndarray:
    """Pairwise linear CKA between every pair of clients' backbone features on the same probe
    batch. Returns an (n_clients, n_clients) matrix; high off-diagonal values indicate the
    clients' representations have collapsed toward each other.
    """
    probe_x = probe_x.to(device)
    features = []
    for m in client_models:
        m.eval()
        features.append(m.backbone(probe_x).cpu())

    n = len(client_models)
    cka = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = linear_cka(features[i], features[j])
            cka[i, j] = cka[j, i] = val
    return cka
