"""Sanity check for src/models/: builds a client model, prints trainable vs total params,
adapter payload size, and verifies get_adapter_state/set_adapter_state round-trip.

Usage:
    python -m src.models.sanity_check --config configs/local_debug.yaml
"""

from __future__ import annotations

import argparse

import torch

from src.models.client_model import build_client_model, param_counts
from src.models.lora_model import adapter_payload_mb, get_adapter_state, set_adapter_state
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-classes", type=int, default=9, help="e.g. PathMNIST=9")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0], deterministic=cfg["reproducibility"]["deterministic"])

    print("=" * 90)
    print(f"Model sanity check -- {args.config}")
    lora_cfg = cfg["model"]["lora"]
    print(f"backbone: {cfg['model']['backbone']}  pretrained: {cfg['model']['pretrained']}  "
          f"freeze_backbone: {cfg['model']['freeze_backbone']}")
    print(f"LoRA: r={lora_cfg['r']} alpha={lora_cfg['alpha']} dropout={lora_cfg['dropout']} "
          f"target_modules(blocks)={lora_cfg['target_modules']}")
    print("=" * 90)

    model = build_client_model(cfg, num_classes=args.num_classes)
    device = torch.device(cfg["device"])
    model.to(device)

    trainable, total = param_counts(model)
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable / total:.3f}%")

    adapter_state = get_adapter_state(model)
    n_lora_tensors = len(adapter_state)
    n_lora_params = sum(t.numel() for t in adapter_state.values())
    print(f"LoRA A/B tensors: {n_lora_tensors}  (params: {n_lora_params:,})")
    print(f"Adapter payload size: {adapter_payload_mb(adapter_state):.4f} MB "
          f"(this is what gets communicated per round, per client)")

    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Client head params (never communicated): {head_params:,} "
          f"(Linear -> {args.num_classes} classes)")

    print("-" * 90)
    print("Forward pass smoke test on CPU...")
    x = torch.randn(4, 3, cfg["data"]["image_size"], cfg["data"]["image_size"], device=device)
    with torch.no_grad():
        out = model(x)
    print(f"  input:  {tuple(x.shape)}")
    print(f"  output: {tuple(out.shape)}  (batch, num_classes={args.num_classes})")

    print("-" * 90)
    print("get_adapter_state / set_adapter_state round-trip check...")
    perturbed = {k: v + 1.0 for k, v in adapter_state.items()}
    set_adapter_state(model, perturbed)
    reloaded = get_adapter_state(model)
    matches = all(torch.allclose(reloaded[k], perturbed[k]) for k in perturbed)
    print(f"  round-trip matches perturbed state: {matches}")
    set_adapter_state(model, adapter_state)  # restore original
    restored = get_adapter_state(model)
    restored_ok = all(torch.allclose(restored[k], adapter_state[k]) for k in adapter_state)
    print(f"  restore to original state:          {restored_ok}")

    print("=" * 90)
    ok = matches and restored_ok and trainable < total and out.shape == (4, args.num_classes)
    print("RESULT:", "SUCCESS." if ok else "FAILURE.")


if __name__ == "__main__":
    main()
