import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.temporal_smoothness import velocity_penalty  # noqa: E402


def test_zero_for_constant_sequence():
    params = torch.ones(2, 8, 5) * 3.0
    assert velocity_penalty(params).item() == pytest.approx(0.0)


def test_nonzero_for_jittery_sequence():
    torch.manual_seed(0)
    jittery = torch.randn(2, 8, 5)
    assert velocity_penalty(jittery).item() > 0


def test_matches_manual_first_difference_formula():
    torch.manual_seed(1)
    params = torch.randn(2, 6, 3)

    manual_first_diff = params[:, :-1] - params[:, 1:]
    expected_vel = manual_first_diff.abs().mean()
    assert velocity_penalty(params).item() == pytest.approx(expected_vel.item(), rel=1e-5)


def test_gradients_flow():
    params = torch.randn(2, 8, 5, requires_grad=True)
    loss = velocity_penalty(params)
    loss.backward()
    assert params.grad is not None
    assert torch.any(params.grad != 0)


def test_minimum_frame_count_edge_case():
    velocity_penalty(torch.randn(1, 2, 4))  # minimum for velocity


def test_asserts_on_too_few_frames():
    with pytest.raises(AssertionError):
        velocity_penalty(torch.randn(1, 1, 4))
