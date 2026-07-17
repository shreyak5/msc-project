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

from model import constants
from model.config import COMPONENT_TOKENS
from model.encoder import SViT
from model.heads import ComponentHeads
from model.temporal import TemporalTransformer


def _split_expression(expression_and_eyelid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        expression_and_eyelid[..., : constants.FLAME_EXPRESSION_DIM],
        expression_and_eyelid[..., constants.FLAME_EXPRESSION_DIM :],
    )


def _split_camera(camera: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return camera[..., constants.CAMERA_SCALE_SLICE], camera[..., constants.CAMERA_ROTATION_SLICE], camera[..., constants.CAMERA_TRANSLATION_SLICE]


def _decode_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """params: dict[component_name] -> (..., param_dim) ComponentHeads output
    -> dict of decoded FLAME/camera parameters, split into the pieces
    FLAME.forward()/project_landmarks actually take (expression's trailing
    eyelid dims and camera's scale/rotation/translation slices, per
    model/constants.py's fixed layout). Shared by encode_image ((B, D) inputs)
    and encode_video ((B, N, D) inputs) - _split_expression/_split_camera slice
    the LAST dimension (...), so both shapes work unchanged."""
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


def encode_image(svit: SViT, heads: ComponentHeads, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
    """pixel_values: (B, 3, H, W) -> dict of decoded FLAME/camera parameters."""
    features = svit(pixel_values)
    params = heads(features)
    return _decode_params(params)


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


def encode_video(
    svit: SViT, tt: TemporalTransformer, heads: ComponentHeads,
    clip_pixel_values: torch.Tensor, visibility_scores: torch.Tensor,
    frame_indices: torch.Tensor, flag_visibility_valid: torch.Tensor,
    real_frame_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """clip_pixel_values: (B, N, 3, H, W). visibility_scores/frame_indices/
    flag_visibility_valid: (B, N). real_frame_mask: (B, N) bool, real frame vs.
    tail padding (defaults all-True, matching TemporalTransformer.forward's own
    valid_mask default). Returns dict of decoded FLAME/camera parameters, each
    (B, N, ...) - the per-frame equivalent of encode_image's (B, ...) output."""
    batch_size, num_frames = clip_pixel_values.shape[:2]
    if real_frame_mask is None:
        real_frame_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=clip_pixel_values.device)

    flat_pixel_values = clip_pixel_values.reshape(batch_size * num_frames, *clip_pixel_values.shape[2:])
    component_outputs = svit(flat_pixel_values)
    component_names = [t.name for t in COMPONENT_TOKENS]
    tokens = torch.stack(
        [component_outputs[name].reshape(batch_size, num_frames, -1) for name in component_names], dim=2
    )  # (B, N, 4, D)

    tokens = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)

    refined = tt(tokens, visibility_scores, frame_indices, valid_mask=real_frame_mask)
    features = {name: refined[:, :, i, :] for i, name in enumerate(component_names)}
    params = heads(features)
    return _decode_params(params)
