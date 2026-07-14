"""Mesh (3D) region-weighted vertex loss, and vertex consistency (Lvc, 3D identity
swap) (implementation-plan.md Sec 6). Both are fundamentally the same computation -
weighted L1 between two FLAME vertex sets - so they share one region-weighting
scheme and implementation.

Mesh loss: region-weighted L1 between predicted and GT vertices ("3D batches, both
stages"). Region weights (model/constants.py) are a reasoned default, not sourced
from SMIRK or TokenFace's own code: SMIRK doesn't train with direct 3D mesh
supervision at all (no equivalent to check), and TokenFace's exact per-region scheme
isn't published. Built from FLAME_masks.pkl (the same curated vertex-region
annotation already used by model/flame/renderer.py) - up-weights the most
expressive regions (lips, eyes, nose, forehead), keeps general face skin at
baseline, and zeroes out regions with no fitting mechanism (eyeballs - FLAME's eye
joints are fixed, not predicted, Sec 2) or outside the face-only render region
(neck/ears/scalp, Sec 2.5/9).

Vertex consistency loss (Lvc): TokenFace's Eq. 5, L_vc = sum_b w*|V^(b->a) -
V_gt^b|_1 where V^(b->a) = FLAME(beta_a, psi_b, theta_b) - i.e. take a same-identity
sample b, swap in a *different* same-identity sample a's shape while keeping b's
own expression/jaw, and compare the resulting vertices against b's own GT vertices
(tests that shape estimates are consistent across different photos of the same
person). The `w` in Eq. 5 uses the same per-vertex region weights as the mesh loss
above (confirmed). Performing the actual shape swap (re-running FLAME.forward()
with beta_a substituted in) and finding which samples in a batch share identity are
both caller/data-pipeline concerns (Sec 7, not yet built), not this module's -
vertex_consistency_loss here just takes the already-swapped vertices.
"""

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
    """TokenFace Eq. 5 (Lvc). swapped_vertices: (B, V, 3) - the caller's
    FLAME(beta_a, psi_b, theta_b) output, i.e. sample b's own expression/jaw
    re-rendered with a *different* same-identity sample a's shape. gt_vertices:
    (B, V, 3) - sample b's own GT vertices (not sample a's). vertex_weights: (V,),
    e.g. from build_region_weights() - same weighting as the mesh loss (confirmed
    to reuse it, not a separate scheme). Mathematically identical to
    region_weighted_mesh_loss; kept as its own named function since the two losses
    have different intent (accuracy vs. shape/expression disentanglement) and
    reading `vertex_consistency_loss(...)` in the training loop should say that
    directly rather than reusing region_weighted_mesh_loss's name for two purposes."""
    return region_weighted_mesh_loss(swapped_vertices, gt_vertices, vertex_weights)
