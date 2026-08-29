"""LoRA adapter attachment (via peft) and adapter-only state extraction/injection.

Only LoRA A/B factors are ever trained or communicated between client and server --
the frozen backbone weights never move.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models.backbone import resolve_conv2d_target_modules

LORA_KEY_MARKERS = ("lora_A", "lora_B")


def attach_lora(backbone: nn.Module, model_cfg: dict) -> nn.Module:
    """Wraps a frozen backbone with LoRA adapters on its Conv2d layers.

    peft freezes every non-LoRA parameter automatically; only lora_A/lora_B stay trainable.
    """
    lora_cfg = model_cfg["lora"]
    target_modules = resolve_conv2d_target_modules(backbone, lora_cfg["target_modules"])

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(backbone, peft_config)


def is_lora_param(name: str) -> bool:
    return any(marker in name for marker in LORA_KEY_MARKERS)


def get_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extracts ONLY the LoRA A/B factors, detached and moved to CPU -- this is exactly the
    payload that gets communicated between a client and the server each round.
    """
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if is_lora_param(name)
    }


def set_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Injects LoRA A/B factors into `model` in place, leaving everything else untouched."""
    own_params = dict(model.named_parameters())
    missing = [k for k in state if k not in own_params]
    if missing:
        raise KeyError(f"set_adapter_state: keys not found in model: {missing[:5]}...")

    with torch.no_grad():
        for name, tensor in state.items():
            own_params[name].copy_(tensor)


def adapter_payload_mb(state: dict[str, torch.Tensor]) -> float:
    """Size in MB of an adapter state dict, as it would be serialized (float32 elements)."""
    total_elems = sum(t.numel() for t in state.values())
    return total_elems * 4 / (1024 ** 2)  # 4 bytes/float32
