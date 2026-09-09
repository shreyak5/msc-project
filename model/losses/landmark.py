"""Base landmark loss adapted from SMIRK (github.com/georgeretsi/smirk, MIT
License, Copyright (c) 2024 George Retsinas). Eye/lip-closure terms adapted
from EMOCA (github.com/radekd91/emoca), under the Max Planck Institute's
non-commercial research license."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEDIAPIPE_EMBEDDING_PATH = _REPO_ROOT / "assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz"

# FAN boundary/jaw-contour landmarks used for the landmark loss (matches SMIRK's own
# [:17] slice and this plan's Sec 5.3 "16 boundary pts" - the boundary contour is
# conventionally the first 17 points, indices 0-16, in the standard iBUG-68 ordering).
NUM_FAN_BOUNDARY_POINTS = 17

# Full standard iBUG-68 FAN point count (jaw + eyebrows + nose + eyes + mouth).
# Used by dev-set periodic eval (training/stage2.py), not the training loss above,
# which stays scoped to NUM_FAN_BOUNDARY_POINTS.
NUM_FAN_TOTAL_POINTS = 68

# Raw MediaPipe Face Mesh (478-point) index groups defining eye/lip opening pairs
# (Danecek et al./EMOCA, see module docstring).
_LEFT_UPPER_EYELID = [398, 384, 385, 386, 387, 388, 466]
_LEFT_LOWER_EYELID = [382, 381, 380, 374, 373, 390, 249]
_RIGHT_UPPER_EYELID = [246, 161, 160, 159, 158, 157, 173]
_RIGHT_LOWER_EYELID = [7, 163, 144, 145, 153, 154, 155]
_UPPER_OUTER_LIP = [185, 40, 39, 37, 0, 267, 269, 270, 409]
_LOWER_OUTER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375]
_UPPER_INNER_LIP = [191, 80, 81, 82, 13, 312, 311, 310, 415]
_LOWER_INNER_LIP = [95, 88, 178, 87, 14, 317, 402, 318, 324]


def _embedded_indices(raw_indices: list[int]) -> torch.Tensor:
    """Converts raw MediaPipe (478-space) indices to their position within the
    curated 105-point embedding order (assets/mediapipe_landmark_embedding/
    ...npz's landmark_indices), which is what FLAME's landmarks_mp and this
    project's precomputed GT MediaPipe landmarks are both ordered by."""
    embedding_indices = np.load(_MEDIAPIPE_EMBEDDING_PATH)["landmark_indices"].tolist()
    return torch.tensor([embedding_indices.index(i) for i in raw_indices], dtype=torch.long)


_UPPER_EYELID_IDX = _embedded_indices(sorted(_LEFT_UPPER_EYELID + _RIGHT_UPPER_EYELID))
_LOWER_EYELID_IDX = _embedded_indices(sorted(_LEFT_LOWER_EYELID + _RIGHT_LOWER_EYELID))
_UPPER_LIP_IDX = _embedded_indices(_UPPER_OUTER_LIP + _UPPER_INNER_LIP)
_LOWER_LIP_IDX = _embedded_indices(_LOWER_OUTER_LIP + _LOWER_INNER_LIP)


def fan_boundary_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """predicted, target: (B, >=17, 2) 2D FAN landmarks -> scalar MSE over the
    boundary/jaw-contour points only (indices [:17], matching SMIRK's own usage).

    mask: (B, >=17) bool, optional - True=keep. None (default) reproduces the
    original plain MSE over all boundary points exactly. When given, per-point
    squared error is masked and averaged over kept points only - see
    _masked_point_mean's docstring for the weighting/0-guard details."""
    pred_b = predicted[:, :NUM_FAN_BOUNDARY_POINTS]
    tgt_b = target[:, :NUM_FAN_BOUNDARY_POINTS]
    if mask is None:
        return F.mse_loss(pred_b, tgt_b)
    mask_b = mask[:, :NUM_FAN_BOUNDARY_POINTS]
    return _masked_point_mean((pred_b - tgt_b) ** 2, mask_b)


def mediapipe_landmark_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """predicted, target: (B, 105, 2) 2D MediaPipe landmarks (curated embedding
    order) -> scalar MSE over all 105 points.

    mask: (B, 105) bool, optional - see fan_boundary_loss's own docstring for the
    masking/weighting convention (identical here, just over all 105 points)."""
    if mask is None:
        return F.mse_loss(predicted, target)
    return _masked_point_mean((predicted - target) ** 2, mask)


def _opening_distance(points: torch.Tensor, upper_idx: torch.Tensor, lower_idx: torch.Tensor) -> torch.Tensor:
    """points: (B, 105, 2) -> (B, num_pairs) per-pair euclidean distance between
    each upper/lower landmark pair. A small epsilon inside the sqrt (not present in
    EMOCA's own version) avoids an undefined/infinite gradient if a pair's distance
    is ever exactly zero - a low-risk defensive addition, not a behavior change in
    any normal case."""
    upper = points[:, upper_idx.to(points.device)]
    lower = points[:, lower_idx.to(points.device)]
    return torch.sqrt(((upper - lower) ** 2).sum(dim=-1) + 1e-12)


def landmark_visibility_mask(face_mask: torch.Tensor, landmarks_norm: torch.Tensor) -> torch.Tensor:
    grid = landmarks_norm.unsqueeze(2)  # (B, N, 1, 2), grid_sample's (B, H_out, W_out, 2)
    sampled = F.grid_sample(
        face_mask.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False,
    )  # (B, 1, N, 1)
    return sampled.squeeze(1).squeeze(-1) > 0.5  # (B, N)


def mouth_point_indices() -> torch.Tensor:
    """Deduped union of _UPPER_LIP_IDX/_LOWER_LIP_IDX (indices into the 105-point
    curated MediaPipe order) - a public accessor for callers outside this module
    (training/stage2.py's optional mouth-region-visibility gate for the vertex-space
    smoothness term) that need the mouth/lip landmark set without reaching into the
    underscore-prefixed internals directly."""
    return torch.cat([_UPPER_LIP_IDX, _LOWER_LIP_IDX]).unique()


def _masked_point_mean(sq_or_abs_err: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """sq_or_abs_err: (B, N, 2) per-point per-coord error. mask: (B, N) bool or None.
    None -> plain mean over every element (the original, unmasked behavior). Given ->
    weighted mean over kept points only ("normalize by kept landmarks, not total", so
    a heavily-occluded frame doesn't silently shrink its own loss contribution just
    because most of its points are zeroed), denominator clamped to a minimum of 1e-8
    to guard the all-occluded 0/0 edge case (mirrors temporal_smoothness.py's
    velocity_penalty denom.clamp pattern) - that case contributes ~0, not NaN."""
    if mask is None:
        return sq_or_abs_err.mean()
    denom = (mask.sum().float() * sq_or_abs_err.shape[-1]).clamp(min=1e-8)
    return (sq_or_abs_err * mask.unsqueeze(-1)).sum() / denom


def _masked_pair_mean(abs_err: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
    """abs_err: (B, num_pairs) per-pair error (already |pred_dist - gt_dist|).
    pair_mask: (B, num_pairs) bool or None. Same masked-weighted-mean/0-guard pattern
    as _masked_point_mean, but for closure losses' per-PAIR (not per-point) unit of
    supervision."""
    if pair_mask is None:
        return abs_err.mean()
    denom = pair_mask.sum().float().clamp(min=1e-8)
    return (abs_err * pair_mask).sum() / denom


def eye_closure_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_dist = _opening_distance(predicted, _UPPER_EYELID_IDX, _LOWER_EYELID_IDX)
    gt_dist = _opening_distance(target, _UPPER_EYELID_IDX, _LOWER_EYELID_IDX)
    abs_err = (pred_dist - gt_dist).abs()
    if mask is None:
        return abs_err.mean()
    pair_mask = (
        mask[:, _UPPER_EYELID_IDX.to(mask.device)] & mask[:, _LOWER_EYELID_IDX.to(mask.device)]
    ).float()
    return _masked_pair_mean(abs_err, pair_mask)


def lip_closure_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """predicted, target: (B, 105, 2) MediaPipe landmarks -> scalar L1 loss between
    predicted and target lip-opening distances (inner + outer lip line).

    mask: (B, 105) bool, optional - same per-pair-reduction convention as
    eye_closure_loss's own `mask` parameter (see its docstring)."""
    pred_dist = _opening_distance(predicted, _UPPER_LIP_IDX, _LOWER_LIP_IDX)
    gt_dist = _opening_distance(target, _UPPER_LIP_IDX, _LOWER_LIP_IDX)
    abs_err = (pred_dist - gt_dist).abs()
    if mask is None:
        return abs_err.mean()
    pair_mask = (
        mask[:, _UPPER_LIP_IDX.to(mask.device)] & mask[:, _LOWER_LIP_IDX.to(mask.device)]
    ).float()
    return _masked_pair_mean(abs_err, pair_mask)
