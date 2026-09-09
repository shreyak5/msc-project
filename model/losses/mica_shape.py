"""Adapted from SMIRK's MICA.calculate_mica_shape_loss
(github.com/georgeretsi/smirk, MIT License, Copyright (c) 2024 George
Retsinas)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mica_shape_loss(shape_params: torch.Tensor, target_mica_shape: torch.Tensor) -> torch.Tensor:
    """shape_params: (B, 300) our own encoder's predicted FLAME shape.
    target_mica_shape: (B, 300) precomputed/cached MICA shape target (already
    detached, since it never carried gradients to begin with - loaded straight
    from disk) -> scalar MSE loss."""
    return F.mse_loss(shape_params, target_mica_shape)
