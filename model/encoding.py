from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel

from model import constants
from model.config import COMPONENT_TOKENS
from model.encoder import SViT
from model.heads import ComponentHeads
from model.temporal import SimpleTemporalTransformer, TTModule


def _split_expression(
    expression_and_eyelid: torch.Tensor, expression_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        expression_and_eyelid[..., :expression_dim],
        expression_and_eyelid[..., expression_dim:],
    )


def _split_camera(camera: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return camera[..., constants.CAMERA_SCALE_SLICE], camera[..., constants.CAMERA_ROTATION_SLICE], camera[..., constants.CAMERA_TRANSLATION_SLICE]


def _decode_params(params: dict[str, torch.Tensor], expression_dim: int) -> dict[str, torch.Tensor]:
    expression, eyelid = _split_expression(params["expression"], expression_dim)
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


def _expression_dim(heads: ComponentHeads) -> int:
    """heads.expression_dim, unwrapping a DistributedDataParallel-wrapped heads
    first if needed (DDP forwards only a fixed set of attributes, not custom
    ones like this). Only used for this attribute read - callers still call
    `heads(...)` on the (possibly DDP-wrapped) module directly, so gradient
    sync during training is unaffected."""
    module = heads.module if isinstance(heads, torch.nn.parallel.DistributedDataParallel) else heads
    return module.expression_dim


def encode_image(
    svit: SViT, heads: ComponentHeads, pixel_values: torch.Tensor, svit_chunk_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """pixel_values: (B, 3, H, W) -> dict of decoded FLAME/camera parameters.

    svit_chunk_size: None (default) runs svit on the full batch in one forward
    pass, reproducing prior behavior exactly - see encode_video's identical
    param for the chunking rationale (_encode_svit)."""
    features = _encode_svit(svit, pixel_values, svit_chunk_size)
    params = heads(features)
    return _decode_params(params, _expression_dim(heads))


def _fill_missing_frame_tokens(
    tokens: torch.Tensor, real_frame_mask: torch.Tensor, flag_visibility_valid: torch.Tensor,
) -> torch.Tensor:
    has_valid_tokens = real_frame_mask & flag_visibility_valid
    tokens = tokens.clone()
    batch_size, num_frames = real_frame_mask.shape
    for b in range(batch_size):
        good_indices = [n for n in range(num_frames) if has_valid_tokens[b, n]]
        if not good_indices:
            continue
        for n in range(num_frames):
            if not real_frame_mask[b, n] or has_valid_tokens[b, n]:
                continue
            before = max((g for g in good_indices if g < n), default=None)
            after = min((g for g in good_indices if g > n), default=None)
            if before is not None and after is not None:
                tokens[b, n] = (tokens[b, before] + tokens[b, after]) / 2
            elif before is not None:
                tokens[b, n] = tokens[b, before]
            elif after is not None:
                tokens[b, n] = tokens[b, after]
    return tokens


def _pool_identity(shape: torch.Tensor, real_frame_mask: torch.Tensor) -> torch.Tensor:
    mask = real_frame_mask.unsqueeze(-1).to(shape.dtype)  # (B, N, 1)
    pooled = (shape * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # (B, D)
    return pooled.unsqueeze(1).expand_as(shape)


def _encode_svit(
    svit: SViT, flat_pixel_values: torch.Tensor, chunk_size: int | None,
) -> dict[str, torch.Tensor]:
    """flat_pixel_values: (B*N, 3, H, W). chunk_size=None runs svit in a single
    forward pass (prior behavior). Otherwise loops over dim 0 in chunks of that
    size, concatenating each component's outputs back together - numerically
    identical to the unchunked call since SViT.forward has no cross-sample
    state (see encode_video's svit_chunk_size docstring)."""
    if chunk_size is None:
        return svit(flat_pixel_values)
    chunk_outputs = [
        svit(flat_pixel_values[start : start + chunk_size])
        for start in range(0, flat_pixel_values.shape[0], chunk_size)
    ]
    return {name: torch.cat([chunk[name] for chunk in chunk_outputs], dim=0) for name in chunk_outputs[0]}


def encode_video(
    svit: SViT, tt: TTModule, heads: ComponentHeads,
    clip_pixel_values: torch.Tensor, visibility_scores: torch.Tensor,
    frame_indices: torch.Tensor, flag_visibility_valid: torch.Tensor,
    real_frame_mask: torch.Tensor | None = None,
    pool_identity: bool = False,
    skip_tt: bool = False,
    svit_chunk_size: int | None = None,
) -> dict[str, torch.Tensor]:
    batch_size, num_frames = clip_pixel_values.shape[:2]
    if real_frame_mask is None:
        real_frame_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=clip_pixel_values.device)

    flat_pixel_values = clip_pixel_values.reshape(batch_size * num_frames, *clip_pixel_values.shape[2:])
    component_outputs = _encode_svit(svit, flat_pixel_values, svit_chunk_size)
    component_names = [t.name for t in COMPONENT_TOKENS]
    tokens = torch.stack(
        [component_outputs[name].reshape(batch_size, num_frames, -1) for name in component_names], dim=2
    )  # (B, N, 4, D)

    tokens = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)

    # Unwrap DistributedDataParallel before the isinstance check: under real
    # multi-GPU training, callers pass a DDP-wrapped tt (training/stage2.py's
    # train()), and isinstance(ddp_wrapped_tt, SimpleTemporalTransformer) is
    # always False regardless of what's inside - DDP is its own class, not a
    # subclass of whatever it wraps - so checking the wrapper itself would
    # silently fall through to the wrong branch below.
    if skip_tt:
        refined = tokens
    else:
        tt_module = tt.module if isinstance(tt, DistributedDataParallel) else tt
        if isinstance(tt_module, SimpleTemporalTransformer):
            # SimpleTemporalTransformer has no visibility_scores parameter at
            # all - by design, it has no visibility signal anywhere in it (see
            # model/temporal.py's TTModule docstring) - so it can't take the
            # same call as the other two variants.
            refined = tt(tokens, frame_indices, valid_mask=real_frame_mask)
        else:
            refined = tt(tokens, visibility_scores, frame_indices, valid_mask=real_frame_mask)
    features = {name: refined[:, :, i, :] for i, name in enumerate(component_names)}
    params = heads(features)
    decoded = _decode_params(params, _expression_dim(heads))
    if pool_identity:
        decoded["shape"] = _pool_identity(decoded["shape"], real_frame_mask)
    return decoded
