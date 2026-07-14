"""MICA (Metrical Identity Conditioned Face reconstruction) shape-only model
(implementation-plan.md Sec 6d-9: MICA shape distillation, Sec 9: "Reuse from SMIRK
repo: ... MICA integration for shape distillation").

Adapted from SMIRK (Retsinas et al., CVPR 2024, https://github.com/georgeretsi/smirk,
src/models/MICA/mica.py, MIT License, Copyright (c) 2024 George Retsinas), itself
adapting Zielon et al.'s MICA (https://github.com/Zielon/MICA). Architecture is a
from-scratch reimplementation (model/mica/arcface.py's Arcface backbone + the
MappingNetwork regressor below), but the actual weights loaded from
assets/mica.tar are MICA's own pretrained checkpoint, under MICA's Max-Planck
non-commercial research license (see https://github.com/Zielon/MICA/blob/master/
LICENSE) - use here is non-commercial academic research, same status as the FLAME
model assets (model/flame/flame.py).

MICA predicts FLAME shape (identity) parameters directly from a single face image,
trained with strong identity supervision (metric face-recognition embeddings) -
used here purely as a frozen distillation target (Sec 6d-9's L2 loss against our
own predicted shape), never fine-tuned. Only the shape regressor is used: MICA's
own expression/pose outputs (it doesn't predict any - it's shape-only by design)
aren't relevant here.

Deviations from SMIRK's mica.py:
- Checkpoint path is a constructor argument (model/constants.py's
  MICA_CHECKPOINT_PATH), not hardcoded, matching this project's convention
  elsewhere (model/flame/flame.py, model/flame/renderer.py).
- MappingNetwork's structure/forward is reproduced unchanged (it's SMIRK/MICA's own
  small novel module, not sourced from insightface) but simplified to drop the
  hidden>5 deep-skip-connection branch: MICA's actual regressor always uses
  hidden=3, so that branch is unreachable dead code here.
- forward() returns a bare (B, 300) tensor, not SMIRK's {'shape_params': ...} dict -
  MICA only ever predicts one thing here (shape), unlike our own FLAME/SViT-Heads
  forward()s which bundle several named outputs; model/losses/mica_shape.py
  consumes it directly.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import constants
from model.mica.arcface import Arcface

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _kaiming_leaky_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")


class MappingNetwork(nn.Module):
    """MICA's own small MLP regressor: 512-dim Arcface embedding -> 300-dim FLAME
    shape params. hidden=3 always (MICA's actual trained configuration) - the
    intermediate skip-connection SMIRK's copy supports for hidden>5 is dropped as
    dead code for that reason."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hidden: int = 3):
        super().__init__()
        self.network = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden)]
        )
        self.output = nn.Linear(hidden_dim, out_dim)
        self.network.apply(_kaiming_leaky_init)
        with torch.no_grad():
            self.output.weight *= 0.25

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.network:
            h = F.leaky_relu(layer(h), negative_slope=0.2)
        return self.output(h)


class MICA(nn.Module):
    def __init__(self, checkpoint_path: str | Path = _REPO_ROOT / constants.MICA_CHECKPOINT_PATH):
        super().__init__()
        self.arcface = Arcface()
        self.regressor = MappingNetwork(
            constants.MICA_ARCFACE_FEATURE_DIM, constants.FLAME_SHAPE_DIM, constants.FLAME_SHAPE_DIM
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.arcface.load_state_dict(checkpoint["arcface"], strict=True)

        mapping_network_keys = {
            key.replace("regressor.", ""): value
            for key, value in checkpoint["flameModel"].items()
            if "network" in key or "output" in key
        }
        self.regressor.load_state_dict(mapping_network_keys, strict=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 112, 112) in [0, 1], RGB, tightly cropped/aligned to
        MICA's own expected face alignment (a different crop convention than the
        main SViT input - preparing that crop is the caller's/data pipeline's
        responsibility, not this module's). Returns (B, 300) predicted FLAME shape
        params."""
        assert images.shape[-2:] == (constants.MICA_IMAGE_SIZE, constants.MICA_IMAGE_SIZE), (
            f"MICA expects a {constants.MICA_IMAGE_SIZE}x{constants.MICA_IMAGE_SIZE} aligned face crop, "
            f"got {tuple(images.shape[-2:])}"
        )
        images = images.sub(0.5).div(0.5)  # [0, 1] -> [-1, 1], Arcface's own expected input range
        images = images[:, [2, 1, 0], :, :]  # RGB -> BGR, insightface's own training convention
        arcface_features = F.normalize(self.arcface(images))
        return self.regressor(arcface_features)
