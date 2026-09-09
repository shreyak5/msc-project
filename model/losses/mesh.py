from __future__ import annotations

import pickle
from pathlib import Path

import torch

from model import constants

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_region_weights(
    flame_masks_path: str | Path = _REPO_ROOT / constants.RENDERER_FLAME_MASKS_PATH,
    num_vertices: int = constants.EXPECTED_NUM_FLAME_VERTICES,
) -> torch.Tensor:
    """Returns (num_vertices,) per-vertex loss weight, per model/constants.py's
    MESH_LOSS_* values. Construction order matters: broader/lower-priority regions
    are applied first, more specific/higher-priority ones after, so eyeballs are
    forced to 0 last regardless of any overlap with face/eye_region."""
    with open(flame_masks_path, "rb") as f:
        flame_masks = pickle.load(f, encoding="latin1")

    weights = torch.full((num_vertices,), constants.MESH_LOSS_DEFAULT_WEIGHT)
    weights[flame_masks["face"]] = constants.MESH_LOSS_FACE_WEIGHT
    weights[flame_masks["boundary"]] = constants.MESH_LOSS_BOUNDARY_WEIGHT
    for region in constants.MESH_LOSS_EXPRESSIVE_REGIONS:
        weights[flame_masks[region]] = constants.MESH_LOSS_EXPRESSIVE_WEIGHT
    for region in constants.MESH_LOSS_EYEBALL_REGIONS:
        weights[flame_masks[region]] = constants.MESH_LOSS_EYEBALL_WEIGHT
    return weights


def build_gated_expressive_region_weights(
    flame_masks_path: str | Path = _REPO_ROOT / constants.RENDERER_FLAME_MASKS_PATH,
    num_vertices: int = constants.EXPECTED_NUM_FLAME_VERTICES,
) -> tuple[torch.Tensor, torch.Tensor]:
    with open(flame_masks_path, "rb") as f:
        flame_masks = pickle.load(f, encoding="latin1")

    base = torch.zeros(num_vertices)
    base[flame_masks["face"]] = 1.0
    base[flame_masks["boundary"]] = 1.0
    for region in constants.MESH_LOSS_EYEBALL_REGIONS:
        base[flame_masks[region]] = 0.0
    for region in constants.MESH_LOSS_EXPRESSIVE_REGIONS:
        base[flame_masks[region]] = 0.0  # applied last: never double count with gated_region_mask below

    gated = torch.zeros(num_vertices)
    for region in constants.MESH_LOSS_EXPRESSIVE_REGIONS:
        gated[flame_masks[region]] = 1.0
    return base, gated


def region_weighted_mesh_loss(
    predicted_vertices: torch.Tensor, gt_vertices: torch.Tensor, vertex_weights: torch.Tensor
) -> torch.Tensor:
    """predicted_vertices, gt_vertices: (B, V, 3). vertex_weights: (V,), e.g. from
    build_region_weights(). Returns a scalar weighted-mean L1 loss over all (B, V, 3)
    elements - normalized by the total weight mass actually used (weights.sum() * B *
    3, since each vertex's weight is applied once per batch item per coordinate), not
    just the raw element count - so the loss's scale doesn't shrink depending on how
    many vertices happen to carry zero weight (a naive unweighted .mean() would be
    diluted by them, since it divides by total element count regardless of weight)."""
    diff = (predicted_vertices - gt_vertices).abs()  # (B, V, 3)
    weights = vertex_weights.to(diff.device).view(1, -1, 1)  # (1, V, 1)
    weighted_diff = diff * weights

    batch_size, _, num_coords = diff.shape
    denom = weights.sum() * batch_size * num_coords
    return weighted_diff.sum() / denom


def vertex_consistency_loss(
    swapped_vertices: torch.Tensor, gt_vertices: torch.Tensor, vertex_weights: torch.Tensor
) -> torch.Tensor:
    return region_weighted_mesh_loss(swapped_vertices, gt_vertices, vertex_weights)
