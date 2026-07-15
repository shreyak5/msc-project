"""Shared helper for the flag_*_valid-gated loss pattern (landmark_cache.py's
flag_landmarks_fan_valid/flag_landmarks_mp_valid, mica_cache.py's
flag_mica_valid) used throughout the Stage 1/Stage 2 training loops.

Matches SMIRK's own actual gating approach (smirk_trainer.py: predicted/target
landmarks are index-filtered by the valid-samples boolean mask BEFORE being
passed to F.mse_loss, not "compute the loss over the whole batch, then zero the
result if invalid") - a batch can have a MIX of valid and invalid samples, and
computing the loss over the whole batch regardless would let invalid/garbage
rows contaminate the loss value for the samples that were actually valid.
"""

from __future__ import annotations

from typing import Callable

import torch


def gated_loss(loss_fn: Callable[..., torch.Tensor], valid_mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor:
    """loss_fn: any of this project's (predicted, target) -> scalar loss
    functions (fan_boundary_loss, mediapipe_landmark_loss, eye_closure_loss,
    lip_closure_loss, mica_shape_loss all fit this shape). valid_mask: (B,)
    bool. *tensors: each (B, ...) - filtered by valid_mask (matching rows only)
    before being passed to loss_fn. Returns a zero scalar (not NaN) if nothing
    in the batch is valid, so callers can always add this into a running total
    without a separate presence check."""
    if valid_mask.sum() == 0:
        return torch.zeros((), device=tensors[0].device)
    filtered = [t[valid_mask] for t in tensors]
    return loss_fn(*filtered)
