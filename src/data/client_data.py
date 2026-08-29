"""Builds per-client (per-specialty) train/val/test DataLoaders, plus a global test loader.

Client i is assigned specialty cfg["data"]["specialties"][i] -- one specialty = one
simulated hospital (cross-specialty non-IID). Train indices are subsampled (cap) and
memory-mapped at load; val is carved out of the (already-capped) train split.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader, Subset

from src.data.specialty_dataset import SpecialtyDataset, TaggedConcatDataset, num_classes_for


@dataclass
class ClientData:
    client_id: int
    specialty: str
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    val_size: int
    test_size: int


def _split_train_val(dataset: SpecialtyDataset, val_fraction: float, seed: int):
    import numpy as np

    n = len(dataset)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def build_clients(cfg: dict, seed: int) -> list[ClientData]:
    data_cfg = cfg["data"]
    specialties = data_cfg["specialties"]
    image_size = data_cfg["image_size"]
    data_root = data_cfg["data_root"]
    cap = data_cfg["subsample_cap"]
    val_fraction = data_cfg["val_fraction"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]

    clients: list[ClientData] = []
    for client_id, specialty in enumerate(specialties):
        train_full = SpecialtyDataset(
            specialty, split="train", image_size=image_size, data_root=data_root,
            cap=cap, seed=seed + client_id,
        )
        train_subset, val_subset = _split_train_val(train_full, val_fraction, seed=seed + client_id)

        test_ds = SpecialtyDataset(
            specialty, split="test", image_size=image_size, data_root=data_root,
            cap=cap, seed=seed + client_id,
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

        clients.append(ClientData(
            client_id=client_id,
            specialty=specialty,
            num_classes=num_classes_for(specialty),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_size=len(train_subset),
            val_size=len(val_subset),
            test_size=len(test_ds),
        ))

    return clients


def build_global_test_loader(clients: list[ClientData], cfg: dict) -> DataLoader:
    """Concatenates every client's test set into one loader, tagging samples by client index."""
    data_cfg = cfg["data"]
    test_datasets = [c.test_loader.dataset for c in clients]
    global_test = TaggedConcatDataset(test_datasets)
    return DataLoader(
        global_test,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
