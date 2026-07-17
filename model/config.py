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


@dataclasses.dataclass
class TTConfig:
    dim: int = constants.TT_EMBED_DIM
    depth: int = constants.TT_DEPTH
    num_heads: int = constants.TT_NUM_HEADS
    mlp_ratio: float = constants.TT_MLP_RATIO
    m_values: tuple[float, ...] = constants.TT_BIAS_M_VALUES
    n_values: tuple[float, ...] = constants.TT_BIAS_N_VALUES
    # Centred local-attention window (Sec 4.2): a query frame only attends to frames
    # within window_size // 2 of itself. See model/temporal.py for how this is enforced.
    window_size: int = constants.TT_WINDOW_SIZE


@dataclasses.dataclass
class UNetConfig:
    in_channels: int = constants.UNET_IN_CHANNELS
    out_channels: int = constants.UNET_OUT_CHANNELS
    init_features: int = constants.UNET_INIT_FEATURES
    res_blocks: int = constants.UNET_RES_BLOCKS
