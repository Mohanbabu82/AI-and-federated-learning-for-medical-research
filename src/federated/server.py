"""Server-side aggregation. FedAvg here; FedSAA (Step 6+) will provide an alternate aggregator
with the same signature so run.py can dispatch between them."""

from __future__ import annotations

import torch


def fedavg_aggregate(adapter_states: list[dict], sample_counts: list[int]) -> dict:
    """Sample-weighted average of LoRA adapter states across participating clients."""
    total = sum(sample_counts)
    weights = [n / total for n in sample_counts]

    keys = adapter_states[0].keys()
    aggregated = {}
    for key in keys:
        stacked = torch.stack([state[key] * w for state, w in zip(adapter_states, weights)])
        aggregated[key] = stacked.sum(dim=0)
    return aggregated
