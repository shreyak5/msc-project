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
def get_component_tokens(
    expression_dim: int = constants.FLAME_EXPRESSION_DIM,
) -> tuple[ComponentTokenSpec, ...]:
    """Expression's token width depends on expression_dim (+ NUM_EYELID_PARAMS) -
    a function rather than a fixed tuple so ComponentHeads can size its expression
    head to a configured expression parameter count (e.g. 50 instead of the
    default 100 - see training/config.py's num_expression_params) while every
    other consumer of COMPONENT_TOKENS below (which only needs token names/count,
    not param_dim) keeps using the default-100 module-level tuple unchanged."""
    return (
        ComponentTokenSpec("shape", constants.SHAPE_TOKEN_DIM),
        ComponentTokenSpec("expression", expression_dim + constants.NUM_EYELID_PARAMS),
        ComponentTokenSpec("jaw", constants.JAW_TOKEN_DIM),
        ComponentTokenSpec("camera", constants.CAMERA_TOKEN_DIM),
    )


COMPONENT_TOKENS: tuple[ComponentTokenSpec, ...] = get_component_tokens()


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
    # Of num_heads, num_visibility_heads are dedicated visibility-only heads (no QK,
    # no distance term); the rest are QK+ALiBi heads using alibi_slopes for a
    # distance-only bias. See model/temporal.py.
    num_visibility_heads: int = constants.TT_NUM_VISIBILITY_HEADS
    alibi_slopes: tuple[float, ...] = constants.TT_ALIBI_SLOPES
    # Softmax temperature applied to the visibility-only head's logits before
    # softmax (weight(i,j) = softmax_j(visibility_j / visibility_temperature)).
    # See model/temporal.py._compute_visibility_logits.
    visibility_temperature: float = constants.TT_VISIBILITY_TEMPERATURE
    # Centred local-attention window (Sec 4.2): a query frame only attends to frames
    # within window_size // 2 of itself. See model/temporal.py for how this is enforced.
    window_size: int = constants.TT_WINDOW_SIZE


@dataclasses.dataclass
class SimpleTTConfig:
    """Config for SimpleTemporalTransformer (model/temporal.py): same architecture
    sizing as TTConfig, but no visibility_temperature/num_visibility_heads - every
    head is an ordinary QK+ALiBi head, so alibi_slopes is sized to all num_heads."""

    dim: int = constants.TT_EMBED_DIM
    depth: int = constants.TT_DEPTH
    num_heads: int = constants.TT_NUM_HEADS
    mlp_ratio: float = constants.TT_MLP_RATIO
    alibi_slopes: tuple[float, ...] = constants.SIMPLE_TT_ALIBI_SLOPES
    window_size: int = constants.TT_WINDOW_SIZE


@dataclasses.dataclass
class GatedTTConfig:
    """Config for GatedTemporalTransformer (model/temporal.py): same architecture
    sizing as SimpleTTConfig (all num_heads are QK+ALiBi heads), plus gamma - the
    sharpness exponent for the post-softmax visibility gate applied uniformly to
    every head (gate_j = visibility_j ** gamma, then attention weights are
    renormalized after gating)."""

    dim: int = constants.TT_EMBED_DIM
    depth: int = constants.TT_DEPTH
    num_heads: int = constants.TT_NUM_HEADS
    mlp_ratio: float = constants.TT_MLP_RATIO
    alibi_slopes: tuple[float, ...] = constants.SIMPLE_TT_ALIBI_SLOPES
    window_size: int = constants.TT_WINDOW_SIZE
    gamma: float = constants.TT_GATE_GAMMA


@dataclasses.dataclass
class UNetConfig:
    in_channels: int = constants.UNET_IN_CHANNELS
    out_channels: int = constants.UNET_OUT_CHANNELS
    init_features: int = constants.UNET_INIT_FEATURES
    res_blocks: int = constants.UNET_RES_BLOCKS
