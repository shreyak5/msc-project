"""Shared wandb logging helper for training/pretrain.py and training/stage2.py."""

from __future__ import annotations


def flatten_metrics(d: dict, prefix: str = "") -> dict[str, float]:
    """Recursively flattens a nested metrics dict into wandb's flat key format,
    e.g. {"2d": {"photometric": 0.1}} with prefix="train" ->
    {"train/2d/photometric": 0.1}.

    Drops None values - evaluation/metrics.py's summarize() emits None for
    mean/median/std on an empty (all-invalid) split, which wandb.log rejects."""
    flat: dict[str, float] = {}
    for key, value in d.items():
        flat_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=flat_key))
        elif value is not None:
            flat[flat_key] = value
    return flat