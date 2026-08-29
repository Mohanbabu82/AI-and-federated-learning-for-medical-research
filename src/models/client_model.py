"""Per-client model: shared frozen backbone + LoRA adapters + a client-owned classifier head.

Classifier heads are never aggregated or communicated -- each client keeps its own, sized to
its specialty's #classes. Only get_adapter_state()/set_adapter_state() payloads travel between
client and server.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbone import build_frozen_backbone
from src.models.lora_model import attach_lora


class ClientModel(nn.Module):
    def __init__(self, backbone_with_lora: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone_with_lora  # frozen weights + trainable LoRA A/B
        self.head = nn.Linear(feature_dim, num_classes)  # client-local, never aggregated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def build_client_model(cfg: dict, num_classes: int) -> ClientModel:
    backbone, feature_dim = build_frozen_backbone(cfg)
    backbone = attach_lora(backbone, cfg["model"])
    return ClientModel(backbone, feature_dim, num_classes)


def build_shared_lora_backbone(cfg: dict) -> tuple[nn.Module, int]:
    """Builds ONE frozen-backbone+LoRA module to be shared across all clients -- used by the
    Centralized baseline, where a single adapter is trained jointly on pooled data (only the
    per-specialty heads differ)."""
    backbone, feature_dim = build_frozen_backbone(cfg)
    backbone = attach_lora(backbone, cfg["model"])
    return backbone, feature_dim


def param_counts(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
