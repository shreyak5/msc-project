"""Architecture config for the tokenized ViT encoder (implementation-plan.md Sec 2)."""

from __future__ import annotations

import dataclasses

from model import constants


@dataclasses.dataclass(frozen=True)
class ComponentTokenSpec:
    name: str
    param_dim: int


# FLAME/camera parameter groups per component token (implementation-plan.md Sec 2.2).
# Order is the order the tokens are appended to the SViT input sequence.
COMPONENT_TOKENS: tuple[ComponentTokenSpec, ...] = (
    ComponentTokenSpec("shape", constants.SHAPE_TOKEN_DIM),
    ComponentTokenSpec("expression", constants.EXPRESSION_TOKEN_DIM),
    ComponentTokenSpec("jaw", constants.JAW_TOKEN_DIM),
    ComponentTokenSpec("camera", constants.CAMERA_TOKEN_DIM),
)


@dataclasses.dataclass
class SViTConfig:
    img_size: int = constants.SVIT_IMG_SIZE
    patch_size: int = constants.SVIT_PATCH_SIZE
    embed_dim: int = constants.SVIT_EMBED_DIM
    depth: int = constants.SVIT_DEPTH
    num_heads: int = constants.SVIT_NUM_HEADS
    mlp_ratio: float = constants.SVIT_MLP_RATIO
    qkv_bias: bool = True
