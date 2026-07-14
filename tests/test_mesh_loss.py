import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import constants  # noqa: E402
from model.losses.mesh import build_region_weights, region_weighted_mesh_loss  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_region_weights_shape_and_values():
    weights = build_region_weights()
    assert weights.shape == (constants.EXPECTED_NUM_FLAME_VERTICES,)
    assert set(weights.unique().tolist()) == {
        constants.MESH_LOSS_DEFAULT_WEIGHT,
        constants.MESH_LOSS_FACE_WEIGHT,
        constants.MESH_LOSS_EXPRESSIVE_WEIGHT,
    }


def test_eyeball_vertices_are_zero_weighted_despite_face_overlap():
    """Eyeballs overlap with the broader face region but should end up zero-weighted
    regardless, since MESH_LOSS_EYEBALL_WEIGHT is applied after MESH_LOSS_FACE_WEIGHT
    in build_region_weights' construction order."""
    import pickle

    with open(_REPO_ROOT / constants.RENDERER_FLAME_MASKS_PATH, "rb") as f:
        flame_masks = pickle.load(f, encoding="latin1")

    weights = build_region_weights()
    for region in constants.MESH_LOSS_EYEBALL_REGIONS:
        assert torch.all(weights[flame_masks[region]] == constants.MESH_LOSS_EYEBALL_WEIGHT)


def test_zero_for_identical_vertices():
    weights = build_region_weights()
    verts = torch.rand(2, constants.EXPECTED_NUM_FLAME_VERTICES, 3)
    assert region_weighted_mesh_loss(verts, verts, weights).item() == pytest.approx(0.0, abs=1e-6)


def test_zero_weight_vertices_contribute_nothing():
    """Corrupting only the zero-weight vertices' predictions should not change the
    loss at all."""
    weights = build_region_weights()
    pred = torch.rand(2, constants.EXPECTED_NUM_FLAME_VERTICES, 3)
    gt = torch.rand(2, constants.EXPECTED_NUM_FLAME_VERTICES, 3)

    baseline = region_weighted_mesh_loss(pred, gt, weights)

    pred_corrupted = pred.clone()
    zero_mask = weights == 0
    pred_corrupted[:, zero_mask] += 100.0

    corrupted = region_weighted_mesh_loss(pred_corrupted, gt, weights)
    assert corrupted.item() == pytest.approx(baseline.item(), abs=1e-5)


def test_expressive_region_error_contributes_more_than_face_region_error():
    """The same magnitude of per-vertex error should produce a larger loss when
    applied to expressive (weight=2) vertices than face (weight=1) vertices."""
    weights = build_region_weights()
    expressive_idx = (weights == constants.MESH_LOSS_EXPRESSIVE_WEIGHT).nonzero(as_tuple=True)[0][:5]
    face_idx = (weights == constants.MESH_LOSS_FACE_WEIGHT).nonzero(as_tuple=True)[0][:5]

    gt = torch.zeros(1, constants.EXPECTED_NUM_FLAME_VERTICES, 3)

    pred_expressive_error = gt.clone()
    pred_expressive_error[:, expressive_idx] = 1.0
    loss_expressive = region_weighted_mesh_loss(pred_expressive_error, gt, weights)

    pred_face_error = gt.clone()
    pred_face_error[:, face_idx] = 1.0
    loss_face = region_weighted_mesh_loss(pred_face_error, gt, weights)

    assert loss_expressive.item() > loss_face.item()


def test_gradients_are_exactly_zero_at_zero_weight_vertices():
    weights = build_region_weights()
    pred = torch.rand(2, constants.EXPECTED_NUM_FLAME_VERTICES, 3, requires_grad=True)
    gt = torch.rand(2, constants.EXPECTED_NUM_FLAME_VERTICES, 3)

    region_weighted_mesh_loss(pred, gt, weights).backward()

    zero_mask = weights == 0
    assert torch.all(pred.grad[:, zero_mask] == 0)
    assert torch.any(pred.grad[:, weights > 0] != 0)


def test_loss_scale_independent_of_batch_size():
    """The weighted-mean normalization should give (approximately) the same loss
    value for the same per-vertex error pattern, regardless of batch size."""
    weights = build_region_weights()
    torch.manual_seed(0)
    pred_single = torch.rand(1, constants.EXPECTED_NUM_FLAME_VERTICES, 3)
    gt_single = torch.rand(1, constants.EXPECTED_NUM_FLAME_VERTICES, 3)
    loss_single = region_weighted_mesh_loss(pred_single, gt_single, weights)

    pred_batch = pred_single.repeat(4, 1, 1)
    gt_batch = gt_single.repeat(4, 1, 1)
    loss_batch = region_weighted_mesh_loss(pred_batch, gt_batch, weights)

    assert loss_single.item() == pytest.approx(loss_batch.item(), rel=1e-5)
