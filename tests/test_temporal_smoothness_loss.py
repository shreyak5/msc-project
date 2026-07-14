import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.temporal_smoothness import acceleration_penalty, velocity_penalty  # noqa: E402


def test_zero_for_constant_sequence():
    params = torch.ones(2, 8, 5) * 3.0
    assert acceleration_penalty(params).item() == pytest.approx(0.0)
    assert velocity_penalty(params).item() == pytest.approx(0.0)


def test_acceleration_zero_but_velocity_nonzero_for_constant_velocity_motion():
    """A steady linear ramp has zero second difference (no jitter) but a nonzero
    first difference (genuine motion) - this is the whole point of using
    acceleration rather than velocity to penalize jitter specifically."""
    B, T, D = 2, 8, 5
    t = torch.arange(T).float().view(1, T, 1).expand(B, T, D)
    linear = t * 0.5

    assert acceleration_penalty(linear).item() == pytest.approx(0.0, abs=1e-6)
    assert velocity_penalty(linear).item() > 0.1


def test_both_nonzero_for_jittery_sequence():
    torch.manual_seed(0)
    jittery = torch.randn(2, 8, 5)
    assert acceleration_penalty(jittery).item() > 0
    assert velocity_penalty(jittery).item() > 0


def test_matches_manual_second_and_first_difference_formula():
    torch.manual_seed(1)
    params = torch.randn(2, 6, 3)

    manual_second_diff = params[:, :-2] - 2 * params[:, 1:-1] + params[:, 2:]
    expected_accel = (manual_second_diff**2).mean()
    assert acceleration_penalty(params).item() == pytest.approx(expected_accel.item(), rel=1e-5)

    manual_first_diff = params[:, :-1] - params[:, 1:]
    expected_vel = (manual_first_diff**2).mean()
    assert velocity_penalty(params).item() == pytest.approx(expected_vel.item(), rel=1e-5)


def test_gradients_flow():
    params = torch.randn(2, 8, 5, requires_grad=True)
    loss = acceleration_penalty(params) + velocity_penalty(params)
    loss.backward()
    assert params.grad is not None
    assert torch.any(params.grad != 0)


def test_minimum_frame_count_edge_cases():
    acceleration_penalty(torch.randn(1, 3, 4))  # minimum for acceleration
    velocity_penalty(torch.randn(1, 2, 4))  # minimum for velocity


def test_asserts_on_too_few_frames():
    with pytest.raises(AssertionError):
        acceleration_penalty(torch.randn(1, 2, 4))
    with pytest.raises(AssertionError):
        velocity_penalty(torch.randn(1, 1, 4))
