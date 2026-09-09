from __future__ import annotations

import torch


def l2_regularization(params: torch.Tensor) -> torch.Tensor:
    """params: (B, D) -> scalar mean squared value - an L2 penalty pulling params
    toward zero (not toward any other target)."""
    return torch.mean(params**2)


def log_scale_regularization(scale: torch.Tensor, reference: float = 7.0) -> torch.Tensor:
    return torch.mean(torch.log(scale.clamp(min=1e-6) / reference) ** 2)
