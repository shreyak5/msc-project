"""MICA shape distillation loss (implementation-plan.md Sec 6: "MICA shape
distillation | L2 between predicted β and MICA's predicted β | pretraining +
recon pass").

Adapted from SMIRK's MICA.calculate_mica_shape_loss (src/models/MICA/mica.py,
MIT License, Copyright (c) 2024 George Retsinas) as a standalone function rather
than a method on the model class, consuming a separately-constructed
model.mica.mica.MICA instance - matching this project's model/ vs model/losses/
split (model/mica/mica.py holds the pretrained model, this file holds the loss
that consumes its output; see model/mica/mica.py's docstring for the license
chain and non-commercial-research status of the underlying assets/mica.tar
weights).

Dropped SMIRK's dimension-truncation guard (`if mica_shape.size(-1) > D: ...`):
MICA's regressor output dim and our own FLAME_SHAPE_DIM are both fixed at 300 -
this was only ever live for SMIRK's own configurable (possibly-smaller) n_shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.mica.mica import MICA


def mica_shape_loss(shape_params: torch.Tensor, mica: MICA, img_mica: torch.Tensor) -> torch.Tensor:
    """shape_params: (B, 300) our own encoder's predicted FLAME shape. mica: a
    constructed (checkpoint-loaded) MICA instance - held/reused by the caller
    across steps, not constructed per call. img_mica: (B, 3, 112, 112) MICA-
    aligned face crop (see model/mica/mica.py's forward() docstring) -> scalar
    MSE loss. MICA always runs under no_grad: it's a frozen distillation target,
    never itself trained."""
    with torch.no_grad():
        mica_shape = mica(img_mica).detach()
    return F.mse_loss(shape_params, mica_shape)
