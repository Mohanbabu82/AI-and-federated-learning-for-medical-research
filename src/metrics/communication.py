"""Communication accounting for federated rounds: MB/round, cumulative MB, and a helper
to read off communication-efficiency (rounds/MB needed to reach a target accuracy)."""

from __future__ import annotations


def round_communication_mb(n_participants: int, payload_mb: float, upload: bool = True,
                            download: bool = True) -> float:
    """MB moved in one round: each participant uploads and/or downloads one adapter payload."""
    multiplier = int(upload) + int(download)
    return n_participants * payload_mb * multiplier


def cumulative_communication(mb_per_round: list[float]) -> list[float]:
    """Running total of MB communicated up to and including each round."""
    total = 0.0
    cumulative = []
    for mb in mb_per_round:
        total += mb
        cumulative.append(total)
    return cumulative


def rounds_to_target_accuracy(accuracies: list[float], target: float) -> int | None:
    """First 1-indexed round at which accuracy >= target, or None if never reached."""
    for i, acc in enumerate(accuracies, start=1):
        if acc >= target:
            return i
    return None


def mb_to_target_accuracy(accuracies: list[float], cumulative_mb: list[float],
                           target: float) -> float | None:
    """Cumulative MB spent by the first round that reaches `target` accuracy, or None."""
    r = rounds_to_target_accuracy(accuracies, target)
    return cumulative_mb[r - 1] if r is not None else None
