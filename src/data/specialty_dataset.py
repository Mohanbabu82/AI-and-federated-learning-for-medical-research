"""Per-specialty MedMNIST loading with memory-frugal subsampling.

Each specialty subset stands in for one simulated hospital. Images are
converted grayscale->3-channel via medmnist's `as_rgb`, and loaded at the
target `image_size` directly (medmnist ships pre-sized 28/64/128/224 npz
files, so no separate PIL resize step is needed).

To respect the 8 GB RAM budget, the underlying .npz arrays are opened with
`mmap_mode='r'` (medmnist reads them straight off disk) and only the
subsampled indices are ever pulled into memory via __getitem__ -- the full
array is never materialized.
"""

from __future__ import annotations

import os

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import Dataset

# Display name (as used in configs/*.yaml) -> medmnist flag
SPECIALTY_TO_FLAG = {
    "PathMNIST": "pathmnist",
    "DermaMNIST": "dermamnist",
    "BloodMNIST": "bloodmnist",
    "OCTMNIST": "octmnist",
    "TissueMNIST": "tissuemnist",
}


def _resolve_flag(specialty: str) -> str:
    if specialty in SPECIALTY_TO_FLAG:
        return SPECIALTY_TO_FLAG[specialty]
    lowered = specialty.lower()
    if lowered in INFO:
        return lowered
    raise ValueError(f"Unknown specialty '{specialty}'. Known: {list(SPECIALTY_TO_FLAG)}")


def num_classes_for(specialty: str) -> int:
    flag = _resolve_flag(specialty)
    return len(INFO[flag]["label"])


def _load_raw(specialty: str, split: str, image_size: int, data_root: str):
    """Open the medmnist split with mmap so the full array is never loaded into RAM."""
    os.makedirs(data_root, exist_ok=True)
    flag = _resolve_flag(specialty)
    DataClass = getattr(medmnist, INFO[flag]["python_class"])
    return DataClass(
        split=split,
        download=True,
        as_rgb=True,
        root=data_root,
        size=image_size,
        mmap_mode="r",
    )


def _subsample_indices(n: int, cap: int | None, seed: int) -> np.ndarray:
    """Deterministically pick <=cap indices out of n, without touching the data itself."""
    if cap is None or cap >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


class SpecialtyDataset(Dataset):
    """A subsampled view over one medmnist split for one specialty/client."""

    def __init__(self, specialty: str, split: str, image_size: int, data_root: str,
                 cap: int | None, seed: int):
        self.specialty = specialty
        self.split = split
        raw = _load_raw(specialty, split, image_size, data_root)
        self.indices = _subsample_indices(len(raw), cap, seed)
        self._raw = raw
        self.num_classes = num_classes_for(specialty)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        raw_idx = int(self.indices[i])
        img, label = self._raw[raw_idx]  # img: HxWx3 uint8 (as_rgb=True), pulled lazily from mmap
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        label_int = int(np.asarray(label).reshape(-1)[0])
        return tensor, label_int


class TaggedConcatDataset(Dataset):
    """Concatenates several SpecialtyDataset test sets, tagging each sample with its client index.

    Needed because specialties have different #classes -- downstream code must know
    which client's (specialty-specific) classifier head to score a sample against.
    """

    def __init__(self, datasets: list[SpecialtyDataset]):
        self.datasets = datasets
        self._cum_lengths = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

    def __getitem__(self, i: int):
        client_idx = int(np.searchsorted(self._cum_lengths, i, side="right"))
        prev = 0 if client_idx == 0 else self._cum_lengths[client_idx - 1]
        local_idx = i - int(prev)
        img, label = self.datasets[client_idx][local_idx]
        return img, label, client_idx
