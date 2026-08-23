"""CPU-only tests for training/stage2.py::compute_temporal_smoothness_losses'
occlusion-experiment1.md Change 2 additions (param_smoothness_enabled,
vertices/base_region_weights/gated_region_mask/gate/expressive_region_smooth_weight/
temporal_vertex_smoothness_weight) - pure tensor logic, no GPU/model dependency
(unlike run_pass_c itself, which needs a real SViT/TT/FLAME/Renderer/UNetGenerator
stack - no automated test exists for that, per the implementation plan)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.temporal_smoothness import velocity_penalty, vertex_velocity_penalty  # noqa: E402
from training.stage2 import compute_temporal_smoothness_losses  # noqa: E402


def _make_encoded(batch_size=1, num_frames=4, seed=0):
    torch.manual_seed(seed)
    return {
        "expression": torch.randn(batch_size, num_frames, 100),
        "eyelid": torch.randn(batch_size, num_frames, 2),
        "jaw": torch.randn(batch_size, num_frames, 3),
        "scale": torch.randn(batch_size, num_frames, 1),
        "rotation": torch.randn(batch_size, num_frames, 3),
        "shape": torch.randn(batch_size, num_frames, 300),
    }


def test_param_smoothness_enabled_false_omits_param_term_and_metrics():
    encoded = _make_encoded()
    real_frame_mask = torch.ones(1, 4, dtype=torch.bool)

    total, metrics = compute_temporal_smoothness_losses(
        encoded, real_frame_mask, param_smoothness_enabled=False,
    )

    assert total.item() == pytest.approx(0.0)
    assert metrics == {}


def test_param_smoothness_enabled_true_matches_manual_sum():
    encoded = _make_encoded()
    real_frame_mask = torch.ones(1, 4, dtype=torch.bool)

    from model import constants

    expr_eyelid = torch.cat([encoded["expression"], encoded["eyelid"]], dim=-1)
    camera_rotation = torch.cat([encoded["scale"], encoded["rotation"]], dim=-1)
    expected = constants.TEMPORAL_VELOCITY_WEIGHT * (
        velocity_penalty(expr_eyelid, real_frame_mask)
        + velocity_penalty(encoded["jaw"], real_frame_mask)
        + velocity_penalty(camera_rotation, real_frame_mask)
        + velocity_penalty(encoded["shape"], real_frame_mask)
    )

    total, metrics = compute_temporal_smoothness_losses(encoded, real_frame_mask)
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)
    assert set(metrics.keys()) == {"vel_expr", "vel_jaw", "vel_camera", "vel_shape"}


def test_temporal_vertex_smoothness_weight_zero_skips_vertex_term_and_metrics():
    encoded = _make_encoded()
    real_frame_mask = torch.ones(1, 4, dtype=torch.bool)

    total_without_vertex, metrics_without_vertex = compute_temporal_smoothness_losses(
        encoded, real_frame_mask, temporal_vertex_smoothness_weight=0.0,
    )
    total_baseline, metrics_baseline = compute_temporal_smoothness_losses(encoded, real_frame_mask)

    assert total_without_vertex.item() == pytest.approx(total_baseline.item(), rel=1e-5)
    assert "vertex_smooth" not in metrics_without_vertex
    assert metrics_without_vertex.keys() == metrics_baseline.keys()


def test_temporal_vertex_smoothness_weight_positive_requires_vertex_args():
    encoded = _make_encoded()
    real_frame_mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(AssertionError):
        compute_temporal_smoothness_losses(
            encoded, real_frame_mask, param_smoothness_enabled=False,
            temporal_vertex_smoothness_weight=0.1,
        )


def test_both_terms_combine_additively():
    encoded = _make_encoded()
    real_frame_mask = torch.ones(1, 4, dtype=torch.bool)

    torch.manual_seed(1)
    vertices = torch.randn(1, 4, 5, 3)
    base_region_weights = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    gated_region_mask = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    gate = torch.rand(1, 3)
    expressive_region_smooth_weight = 3.0
    temporal_vertex_smoothness_weight = 0.2

    total, metrics = compute_temporal_smoothness_losses(
        encoded, real_frame_mask,
        vertices=vertices, base_region_weights=base_region_weights,
        gated_region_mask=gated_region_mask, gate=gate,
        expressive_region_smooth_weight=expressive_region_smooth_weight,
        temporal_vertex_smoothness_weight=temporal_vertex_smoothness_weight,
    )

    from model import constants

    expr_eyelid = torch.cat([encoded["expression"], encoded["eyelid"]], dim=-1)
    camera_rotation = torch.cat([encoded["scale"], encoded["rotation"]], dim=-1)
    expected_param_term = constants.TEMPORAL_VELOCITY_WEIGHT * (
        velocity_penalty(expr_eyelid, real_frame_mask)
        + velocity_penalty(encoded["jaw"], real_frame_mask)
        + velocity_penalty(camera_rotation, real_frame_mask)
        + velocity_penalty(encoded["shape"], real_frame_mask)
    )
    expected_vertex_term = vertex_velocity_penalty(
        vertices, base_region_weights, gated_region_mask, gate, expressive_region_smooth_weight,
        valid_mask=real_frame_mask,
    )
    expected_total = expected_param_term + temporal_vertex_smoothness_weight * expected_vertex_term

    assert total.item() == pytest.approx(expected_total.item(), rel=1e-5)
    assert metrics["vertex_smooth"] == pytest.approx(expected_vertex_term.item(), rel=1e-5)
    assert set(metrics.keys()) == {"vel_expr", "vel_jaw", "vel_camera", "vel_shape", "vertex_smooth"}
