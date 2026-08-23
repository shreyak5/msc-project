"""CPU-only tests for occlusion-experiment1.md's Change 1: landmark_visibility_mask
and the masked-loss variants of fan_boundary_loss/mediapipe_landmark_loss/
eye_closure_loss/lip_closure_loss. Kept separate from tests/test_landmark_loss.py,
whose module-wide skipif(not cuda) marker covers tests that need the real
Renderer/FLAME - none of these do (pure synthetic tensors)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses import landmark as landmark_losses  # noqa: E402


def test_landmark_visibility_mask_basic():
    # face_mask: left half visible (1.0), right half occluded/background (0.0).
    face_mask = torch.zeros(1, 8, 8)
    face_mask[:, :, :4] = 1.0
    landmarks_norm = torch.tensor([[[-0.9, 0.0], [0.9, 0.0]]])  # (1, 2, 2): left point, right point

    mask = landmark_losses.landmark_visibility_mask(face_mask, landmarks_norm)
    assert mask.shape == (1, 2)
    assert bool(mask[0, 0])
    assert not bool(mask[0, 1])


def test_landmark_visibility_mask_out_of_bounds_is_occluded():
    face_mask = torch.ones(1, 8, 8)  # entirely visible
    landmarks_norm = torch.tensor([[[1.5, 0.0], [-1.5, 0.0]]])  # both outside [-1, 1]

    mask = landmark_losses.landmark_visibility_mask(face_mask, landmarks_norm)
    assert not mask.any()


def test_mouth_point_indices_matches_lip_index_union():
    expected = torch.cat([landmark_losses._UPPER_LIP_IDX, landmark_losses._LOWER_LIP_IDX]).unique()
    actual = landmark_losses.mouth_point_indices()
    assert torch.equal(actual.sort().values, expected.sort().values)


def _fan_inputs():
    torch.manual_seed(0)
    predicted = torch.randn(1, 20, 2)
    target = torch.randn(1, 20, 2)
    return predicted, target


def _mp_inputs():
    torch.manual_seed(1)
    predicted = torch.randn(1, 105, 2)
    target = torch.randn(1, 105, 2)
    return predicted, target


def test_fan_boundary_loss_mask_all_true_matches_unmasked():
    predicted, target = _fan_inputs()
    mask = torch.ones(1, 20, dtype=torch.bool)
    assert landmark_losses.fan_boundary_loss(predicted, target, mask).item() == pytest.approx(
        landmark_losses.fan_boundary_loss(predicted, target).item(), rel=1e-5,
    )


def test_mediapipe_landmark_loss_mask_all_true_matches_unmasked():
    predicted, target = _mp_inputs()
    mask = torch.ones(1, 105, dtype=torch.bool)
    assert landmark_losses.mediapipe_landmark_loss(predicted, target, mask).item() == pytest.approx(
        landmark_losses.mediapipe_landmark_loss(predicted, target).item(), rel=1e-5,
    )


def test_masked_loss_ignores_occluded_points_contribution():
    predicted, target = _mp_inputs()
    mask = torch.ones(1, 105, dtype=torch.bool)
    mask[:, :10] = False  # first 10 points occluded

    baseline = landmark_losses.mediapipe_landmark_loss(predicted, target, mask)

    corrupted = predicted.clone()
    corrupted[:, :10] += 100.0  # corrupt only the occluded points' predictions
    corrupted_loss = landmark_losses.mediapipe_landmark_loss(corrupted, target, mask)

    assert corrupted_loss.item() == pytest.approx(baseline.item(), rel=1e-5)


def test_masked_loss_all_occluded_returns_zero_not_nan():
    predicted, target = _mp_inputs()
    mask = torch.zeros(1, 105, dtype=torch.bool)

    loss = landmark_losses.mediapipe_landmark_loss(predicted, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fan_boundary_loss_all_occluded_returns_zero_not_nan():
    predicted, target = _fan_inputs()
    mask = torch.zeros(1, 20, dtype=torch.bool)

    loss = landmark_losses.fan_boundary_loss(predicted, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_closure_loss_drops_pair_if_either_landmark_occluded():
    predicted, target = _mp_inputs()
    mask = torch.ones(1, 105, dtype=torch.bool)

    baseline_no_mask = landmark_losses.eye_closure_loss(predicted, target)

    # Occlude only ONE landmark of the FIRST eyelid pair - that pair's term should
    # drop out entirely, changing the loss away from the (still all-points-included
    # but now selectively-averaged) baseline computed over the remaining pairs.
    upper0 = landmark_losses._UPPER_EYELID_IDX[0].item()
    mask[:, upper0] = False

    masked_loss = landmark_losses.eye_closure_loss(predicted, target, mask)

    # Manually recompute the expected value: mean abs error over every pair EXCEPT
    # the one touching upper0.
    pred_dist = landmark_losses._opening_distance(
        predicted, landmark_losses._UPPER_EYELID_IDX, landmark_losses._LOWER_EYELID_IDX
    )
    gt_dist = landmark_losses._opening_distance(
        target, landmark_losses._UPPER_EYELID_IDX, landmark_losses._LOWER_EYELID_IDX
    )
    abs_err = (pred_dist - gt_dist).abs()
    keep = landmark_losses._UPPER_EYELID_IDX != upper0
    expected = abs_err[:, keep].mean()

    assert masked_loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert masked_loss.item() != pytest.approx(baseline_no_mask.item(), rel=1e-5)


def test_lip_closure_loss_drops_pair_if_either_landmark_occluded():
    predicted, target = _mp_inputs()
    mask = torch.ones(1, 105, dtype=torch.bool)
    lower0 = landmark_losses._LOWER_LIP_IDX[0].item()
    mask[:, lower0] = False

    masked_loss = landmark_losses.lip_closure_loss(predicted, target, mask)

    pred_dist = landmark_losses._opening_distance(
        predicted, landmark_losses._UPPER_LIP_IDX, landmark_losses._LOWER_LIP_IDX
    )
    gt_dist = landmark_losses._opening_distance(
        target, landmark_losses._UPPER_LIP_IDX, landmark_losses._LOWER_LIP_IDX
    )
    abs_err = (pred_dist - gt_dist).abs()
    keep = landmark_losses._LOWER_LIP_IDX != lower0
    expected = abs_err[:, keep].mean()

    assert masked_loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_masked_loss_gradients_flow():
    predicted, target = _mp_inputs()
    predicted.requires_grad_(True)
    mask = torch.ones(1, 105, dtype=torch.bool)
    mask[:, :10] = False

    loss = (
        landmark_losses.mediapipe_landmark_loss(predicted, target, mask)
        + landmark_losses.eye_closure_loss(predicted, target, mask)
        + landmark_losses.lip_closure_loss(predicted, target, mask)
    )
    loss.backward()

    assert predicted.grad is not None
    # Gradient must be exactly zero at fully-masked-out points for the base
    # position loss (mediapipe_landmark_loss never touches them at all).
    assert torch.all(predicted.grad[:, :10] == 0)
    assert torch.any(predicted.grad[:, 10:] != 0)
