"""MICA shape distillation loss (implementation-plan.md Sec 6: "MICA shape
distillation | L2 between predicted β and MICA's predicted β | pretraining +
recon pass").

Adapted from SMIRK's MICA.calculate_mica_shape_loss (src/models/MICA/mica.py,
MIT License, Copyright (c) 2024 George Retsinas) - see model/mica/mica.py's
docstring for the license chain and non-commercial-research status of the
underlying assets/mica.tar weights.

Unlike SMIRK (which runs MICA live every training step), this project
precomputes and caches MICA's shape predictions ahead of time (Sec 5.3,
dataset_processing/dataloading/mica_cache.py) - so this function never touches
the MICA model at all, just a plain MSE against whatever cached target the
dataset item already carries (item["mica_shape"]). The target should already
be gated by the item's flag_mica_valid before being passed in (skip/exclude
invalid entries at the batch-composition level, same pattern as the landmark
losses' flag_landmarks_*_valid).

Dropped SMIRK's dimension-truncation guard (`if mica_shape.size(-1) > D: ...`):
MICA's regressor output dim and our own FLAME_SHAPE_DIM are both fixed at 300 -
this was only ever live for SMIRK's own configurable (possibly-smaller) n_shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mica_shape_loss(shape_params: torch.Tensor, target_mica_shape: torch.Tensor) -> torch.Tensor:
    """shape_params: (B, 300) our own encoder's predicted FLAME shape.
    target_mica_shape: (B, 300) precomputed/cached MICA shape target (already
    detached, since it never carried gradients to begin with - loaded straight
    from disk) -> scalar MSE loss."""
    return F.mse_loss(shape_params, target_mica_shape)
