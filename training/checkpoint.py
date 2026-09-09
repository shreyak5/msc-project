from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from training.distributed import unwrap_model


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    modules: dict[str, nn.Module],
    optimizer: Optimizer | None = None,
    num_expression_params: int | None = None,
) -> Path:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"step_{step:08d}.pt"
    checkpoint = {"step": step}
    if num_expression_params is not None:
        checkpoint["num_expression_params"] = num_expression_params
    checkpoint.update({name: unwrap_model(module).state_dict() for name, module in modules.items()})
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    modules: dict[str, nn.Module],
    optimizer: Optimizer | None = None,
    device: str = "cpu",
    expected_num_expression_params: int | None = None,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if expected_num_expression_params is not None:
        checkpoint_dim = checkpoint.get("num_expression_params", 100)
        if checkpoint_dim != expected_num_expression_params:
            raise ValueError(
                f"{path} was saved with num_expression_params={checkpoint_dim}, but this run is "
                f"configured for num_expression_params={expected_num_expression_params} - these are "
                f"incompatible (the expression head's weights are a different shape); use a checkpoint "
                f"trained with matching num_expression_params, or start a fresh run instead of resuming."
            )
    for name, module in modules.items():
        result = module.load_state_dict(checkpoint[name], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"[checkpoint] {name}: architecture mismatch against {path} - "
                f"missing {result.missing_keys}, unexpected {result.unexpected_keys} "
                f"(unmatched params stay at current init)"
            )
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]
