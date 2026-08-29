"""Load and validate the shared YAML config schema used by local_debug.yaml and full_run.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_METHODS = {"fedsaa", "fedavg", "fedprox", "local_only", "centralized"}
VALID_BACKBONES = {"resnet18"}


class ConfigError(ValueError):
    pass


def _require(d: dict, key: str, path: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required config key: '{path}.{key}'")
    return d[key]


def load_config(path: str | Path) -> dict:
    """Load a YAML config and validate it against the shared schema.

    Returns the raw (validated) dict rather than a rigid dataclass so all
    hyperparameters stay easily introspectable/overridable.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for top in ["seeds", "device", "data", "model", "federated", "method", "logging", "reproducibility"]:
        _require(cfg, top, "")

    if not isinstance(cfg["seeds"], list) or not cfg["seeds"]:
        raise ConfigError("'seeds' must be a non-empty list of ints")

    data = cfg["data"]
    for key in ["image_size", "num_workers", "batch_size", "specialties", "data_root", "val_fraction"]:
        _require(data, key, "data")
    if not isinstance(data["specialties"], list) or not data["specialties"]:
        raise ConfigError("'data.specialties' must be a non-empty list")

    model = cfg["model"]
    for key in ["backbone", "pretrained", "freeze_backbone", "lora"]:
        _require(model, key, "model")
    if model["backbone"] not in VALID_BACKBONES:
        raise ConfigError(f"Unsupported backbone '{model['backbone']}', expected one of {VALID_BACKBONES}")
    lora = model["lora"]
    for key in ["r", "alpha", "dropout", "target_modules"]:
        _require(lora, key, "model.lora")

    fed = cfg["federated"]
    for key in ["num_clients", "rounds", "local_epochs", "client_fraction",
                "optimizer", "lr", "momentum", "weight_decay"]:
        _require(fed, key, "federated")
    if not (0.0 < fed["client_fraction"] <= 1.0):
        raise ConfigError("federated.client_fraction must be in (0.0, 1.0]")
    if fed["num_clients"] != len(data["specialties"]):
        raise ConfigError(
            f"federated.num_clients ({fed['num_clients']}) must equal "
            f"len(data.specialties) ({len(data['specialties'])}) -- one specialty per client"
        )

    method = cfg["method"]
    name = _require(method, "name", "method")
    if name not in VALID_METHODS:
        raise ConfigError(f"Unknown method.name '{name}', expected one of {VALID_METHODS}")
    if name == "fedsaa":
        fsaa = _require(method, "fedsaa", "method")
        for key in ["tau", "lambda", "svd_rank"]:
            _require(fsaa, key, "method.fedsaa")
    if name == "fedprox":
        fprox = _require(method, "fedprox", "method")
        _require(fprox, "mu", "method.fedprox")

    logging_cfg = cfg["logging"]
    for key in ["results_dir", "log_every_round", "save_checkpoints"]:
        _require(logging_cfg, key, "logging")

    repro = cfg["reproducibility"]
    for key in ["deterministic", "cudnn_benchmark"]:
        _require(repro, key, "reproducibility")
