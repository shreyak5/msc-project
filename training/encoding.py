"""SViT -> ComponentHeads encoding, shared by every training loop (Stage 1's
training/pretrain.py, Stage 2's training/stage2.py) and any pass that needs a
model forward pass to get decoded FLAME/camera parameters from pixels."""

from __future__ import annotations

import torch

from model import constants
from model.encoder import SViT
from model.heads import ComponentHeads


def _split_expression(expression_and_eyelid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        expression_and_eyelid[:, : constants.FLAME_EXPRESSION_DIM],
        expression_and_eyelid[:, constants.FLAME_EXPRESSION_DIM :],
    )


def _split_camera(camera: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return camera[:, constants.CAMERA_SCALE_SLICE], camera[:, constants.CAMERA_ROTATION_SLICE], camera[:, constants.CAMERA_TRANSLATION_SLICE]


def encode(svit: SViT, heads: ComponentHeads, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
    """pixel_values: (B, 3, H, W) -> dict of decoded FLAME/camera parameters,
    split into the pieces FLAME.forward()/project_landmarks actually take
    (expression's trailing eyelid dims and camera's scale/rotation/translation
    slices, per model/constants.py's fixed layout)."""
    features = svit(pixel_values)
    params = heads(features)
    expression, eyelid = _split_expression(params["expression"])
    scale, rotation, translation = _split_camera(params["camera"])
    return {
        "shape": params["shape"],
        "expression": expression,
        "eyelid": eyelid,
        "jaw": params["jaw"],
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }
