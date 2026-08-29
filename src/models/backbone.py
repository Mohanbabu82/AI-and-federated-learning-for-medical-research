"""Frozen pretrained image backbone (CPU-only)."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

_BUILDERS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT),
}


def build_frozen_backbone(cfg: dict) -> tuple[nn.Module, int]:
    """Builds the configured backbone, strips its classification head, and freezes all weights.

    Returns (backbone, feature_dim) where backbone(x) -> (batch, feature_dim) pooled features.
    """
    model_cfg = cfg["model"]
    name = model_cfg["backbone"]
    if name not in _BUILDERS:
        raise ValueError(f"Unsupported backbone '{name}', expected one of {list(_BUILDERS)}")

    builder, default_weights = _BUILDERS[name]
    weights = default_weights if model_cfg["pretrained"] else None
    backbone = builder(weights=weights)

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()  # drop the ImageNet classifier; heads are attached per-client

    if model_cfg["freeze_backbone"]:
        for p in backbone.parameters():
            p.requires_grad_(False)

    backbone.eval() if model_cfg["freeze_backbone"] else backbone.train()
    return backbone, feature_dim


def resolve_conv2d_target_modules(backbone: nn.Module, blocks: list[str]) -> list[str]:
    """Expands config-level block names (e.g. 'layer1') into the exact dotted names of every
    Conv2d submodule inside them, since LoRA can only attach to individual Conv2d/Linear layers,
    not to a whole nn.Sequential block.

    'conv1' matches the top-level stem conv exactly; 'layerN' matches every Conv2d nested under
    that block (e.g. 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', ...).
    """
    resolved: list[str] = []
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if name in blocks:
            resolved.append(name)
            continue
        for block in blocks:
            if name.startswith(block + "."):
                resolved.append(name)
                break
    if not resolved:
        raise ValueError(f"No Conv2d modules matched target_modules blocks {blocks}")
    return resolved
