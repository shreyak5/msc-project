"""SViT -> ComponentHeads (-> TT, for video) encoding pipeline: composes the
separately-defined model pieces (model/encoder.py, model/heads.py, model/
temporal.py) into the decoded FLAME/camera parameters every training pass and
the eventual evaluation pipeline (Sec 8) actually consume. Lives under model/
(not training/) since it's a piece of the inference pipeline itself, reusable
outside training - not training-loop machinery like checkpointing/DDP setup.

encode_image is the single-frame path: SViT -> ComponentHeads directly, no TT.
encode_video (Sec 7 Pass C, TT training) is a genuinely different data flow:
SViT runs per-frame on a clip (flattened to a strict (batch,3,H,W) input for
its patch embedding), its per-frame component-token outputs get stacked into
TemporalTransformer.forward's expected (B, N, 4, D) layout, refined by TT,
unstacked back into a features dict, and only THEN decoded by ComponentHeads
(which - per its own docstring - accepts (B,N,D) inputs directly, no flatten
needed there, unlike SViT's patch embedding)."""

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
    """params: dict[component_name] -> (..., param_dim) ComponentHeads output
    -> dict of decoded FLAME/camera parameters, split into the pieces
    FLAME.forward()/project_landmarks actually take (expression's trailing
    eyelid dims and camera's scale/rotation/translation slices, per
    model/constants.py's fixed layout). Shared by encode_image ((B, D) inputs)
    and encode_video ((B, N, D) inputs) - _split_expression/_split_camera slice
    the LAST dimension (...), so both shapes work unchanged. expression_dim must
    match the ComponentHeads that produced `params` (see encode_image/encode_video,
    which read it off heads.expression_dim rather than assuming the default)."""
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
    """tokens: (B, N, 4, D) pre-TT SViT tokens. real_frame_mask: (B, N) bool -
    True where the frame is real, not tail-padding (TemporalTransformer's own
    valid_mask concept). flag_visibility_valid: (B, N) bool - True where a face
    was actually detected for that (real) frame.

    !real_frame_mask: padded frame - not fixed here (TT discards its output
    regardless, per its own docstring).
    real_frame_mask & flag_visibility_valid: real frame, face visible - has
    valid tokens already, not touched.
    real_frame_mask & !flag_visibility_valid: real frame, face NOT visible
    (crop_cache's own detection failed for this frame, so its pixel_values is a
    black fallback and SViT's output for it is garbage) - this is what gets
    fixed: its tokens are replaced with the average of the nearest
    (real_frame_mask & flag_visibility_valid) frame before and after it
    (one-sided if only one exists on that side, left unchanged if the whole
    clip has none at all) - never searching into padding.

    TT's own output is `residual + delta` (model/temporal.py) - a refinement of
    its input, not a full replacement - so without this, TT would have to
    learn to fully cancel out and replace a degenerate residual for such
    frames, rather than just refining an already-reasonable one. This doesn't
    duplicate visibility_scores' own down-weighting of such frames as
    attention *keys* (model/temporal.py's bias formula depends only on the
    key's own visibility, not the query's - a bad frame's query already
    attends normally to good neighbors via that mechanism) - this instead
    fixes the residual baseline those attention-weighted outputs get added to.

    Implemented as a plain nested loop over (batch, frame): clip length is
    small (TT_WINDOW_SIZE) and so are typical video-category batch sizes, so
    this is negligible next to the actual SViT/TT forward cost - not worth
    hand-vectorizing."""
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
    """shape: (B, N, D) decoded per-frame FLAME shape params -> (B, N, D) with
    every real frame's value replaced by that CLIP's own masked mean (never
    mixed across the batch dimension - each of the B clips is generally a
    different subject/video, so pooling across B would blend different
    people's identities together). Padded frames (real_frame_mask == False)
    are excluded from the mean but still receive the pooled value in the
    output, matching every other per-frame field's convention of returning a
    numerically valid (if meaningless) row at padded positions.

    Unweighted by visibility, deliberately: the working hypothesis (Pass C
    identity-pooling experiment) is that TT's own attention already accounts
    for visibility when refining tokens, so a second visibility re-weighting
    here would be redundant - see stage2-config-reference.md's
    pass_c_identity_pooling entry."""
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
    """clip_pixel_values: (B, N, 3, H, W). visibility_scores/frame_indices/
    flag_visibility_valid: (B, N). real_frame_mask: (B, N) bool, real frame vs.
    tail padding (defaults all-True, matching TemporalTransformer.forward's own
    valid_mask default). Returns dict of decoded FLAME/camera parameters, each
    (B, N, ...) - the per-frame equivalent of encode_image's (B, ...) output.

    pool_identity: False (default) reproduces the original per-frame-shape
    behavior exactly for every existing caller. True replaces the decoded
    `shape` entry with _pool_identity's per-clip masked mean, broadcast back
    to every real frame - see _pool_identity's own docstring.

    skip_tt: False (default) runs tt as normal. True bypasses it entirely
    (refined = tokens) - for a checkpoint whose tt weights didn't load due to
    an architecture mismatch, so its heads see the raw SViT tokens instead of
    a partially-random-init tt's output.

    svit_chunk_size: None (default) runs svit on every frame in one forward
    pass, reproducing prior behavior exactly - every existing training/eval
    caller relies on this. Set to an int to instead loop over the flattened
    (batch_size * num_frames) frames in chunks of that size, capping peak SViT
    activation memory for long clips (inference/demo_videos.py's use case).
    SViT's forward has no cross-sample state (LayerNorm + self-attention only,
    batch_first, no BatchNorm) so chunking along this dim is numerically
    identical to the unchunked call."""
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
