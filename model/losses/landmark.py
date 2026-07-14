"""Landmark loss + eye/mouth-closure terms (implementation-plan.md Sec 6: "L2 of
projected 3D landmarks vs detected 2D landmarks; includes eye-closure and mouth/
lip-closure terms (SMIRK/DECA-style)").

Base landmark loss (L2/MSE, matching SMIRK's own convention) adapted from SMIRK
(Retsinas et al., CVPR 2024, https://github.com/georgeretsi/smirk,
src/smirk_trainer.py, MIT License, Copyright (c) 2024 George Retsinas) - FAN loss
restricted to the boundary/jaw landmarks only ([:17] of the 68-point set), matching
SMIRK's own usage and this plan's Sec 5.3 scoping of precomputed FAN data to
"16 boundary pts".

Eye-closure and lip-closure terms are NOT present anywhere in SMIRK's own codebase
(verified via grep) - adapted from EMOCA (Danecek et al., CVPR 2022,
https://github.com/radekd91/emoca, release/EMOCA_v2/gdl/layers/losses/
MediaPipeLandmarkLosses.py; Software Copyright License for non-commercial
scientific research purposes, Max Planck Institute for Intelligent Systems - same
license family as model/flame/lbs.py; use here is non-commercial academic
research). These compare the *distance* between paired upper/lower eyelid (or lip)
landmarks, not absolute landmark position - a more direct signal for blink/mouth-
open expressions than position error alone. Matches EMOCA's own L1-of-distance-
difference formulation for these two specific terms (the base landmark loss stays
L2/MSE, per this plan's explicit wording and SMIRK's own convention - only these
closure terms use L1, following EMOCA/DECA's own choice for them specifically).
Not ported: EMOCA's separate "mouth corner distance" (smile-width) term - out of
scope, this plan only calls for eye-closure and mouth/lip-closure.

MediaPipe indices: EMOCA's eye/lip landmark groups are defined in the raw 478-point
MediaPipe Face Mesh index space, but both FLAME's own landmarks_mp output (model/
flame/flame.py) and this project's precomputed GT MediaPipe landmarks (Sec 5.3, not
yet implemented - see contract note below) use a curated 105-point subset, ordered
by assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz's own
landmark_indices array - verified byte-for-byte identical (including order) to
EMOCA's own hardcoded index list, since both are the same canonical FLAME/DECA/
EMOCA-lineage embedding asset. Raw indices are converted to positions within that
105-point array at import time from the two source lists (see _embedded_indices),
rather than hardcoding the already-converted result, so the derivation stays
verifiable rather than being an opaque copied array.

Precomputed GT MediaPipe landmarks contract (Sec 5.3, not yet implemented): expected
in this SAME 105-point curated order (matching landmark_indices / FLAME's
landmarks_mp), not the raw 478-point MediaPipe detector output.
"""

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


def fan_boundary_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """predicted, target: (B, >=17, 2) 2D FAN landmarks -> scalar MSE over the
    boundary/jaw-contour points only (indices [:17], matching SMIRK's own usage)."""
    return F.mse_loss(predicted[:, :NUM_FAN_BOUNDARY_POINTS], target[:, :NUM_FAN_BOUNDARY_POINTS])


def mediapipe_landmark_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """predicted, target: (B, 105, 2) 2D MediaPipe landmarks (curated embedding
    order) -> scalar MSE over all 105 points."""
    return F.mse_loss(predicted, target)


def _opening_distance(points: torch.Tensor, upper_idx: torch.Tensor, lower_idx: torch.Tensor) -> torch.Tensor:
    """points: (B, 105, 2) -> (B, num_pairs) per-pair euclidean distance between
    each upper/lower landmark pair. A small epsilon inside the sqrt (not present in
    EMOCA's own version) avoids an undefined/infinite gradient if a pair's distance
    is ever exactly zero - a low-risk defensive addition, not a behavior change in
    any normal case."""
    upper = points[:, upper_idx.to(points.device)]
    lower = points[:, lower_idx.to(points.device)]
    return torch.sqrt(((upper - lower) ** 2).sum(dim=-1) + 1e-12)


def eye_closure_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """predicted, target: (B, 105, 2) MediaPipe landmarks -> scalar L1 loss between
    predicted and target eyelid-opening distances (not absolute landmark position) -
    a more direct signal for blinks than position error alone."""
    pred_dist = _opening_distance(predicted, _UPPER_EYELID_IDX, _LOWER_EYELID_IDX)
    gt_dist = _opening_distance(target, _UPPER_EYELID_IDX, _LOWER_EYELID_IDX)
    return (pred_dist - gt_dist).abs().mean()


def lip_closure_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """predicted, target: (B, 105, 2) MediaPipe landmarks -> scalar L1 loss between
    predicted and target lip-opening distances (inner + outer lip line)."""
    pred_dist = _opening_distance(predicted, _UPPER_LIP_IDX, _LOWER_LIP_IDX)
    gt_dist = _opening_distance(target, _UPPER_LIP_IDX, _LOWER_LIP_IDX)
    return (pred_dist - gt_dist).abs().mean()
