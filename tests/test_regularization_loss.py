import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.regularization import l2_regularization, log_scale_regularization  # noqa: E402


def test_zero_params_gives_zero_loss():
    params = torch.zeros(4, 100)
    assert l2_regularization(params).item() == pytest.approx(0.0)


def test_matches_manual_mean_squared_value():
    params = torch.randn(4, 300)
    expected = (params**2).mean()
    assert l2_regularization(params).item() == pytest.approx(expected.item(), rel=1e-5)


def test_gradients_flow_and_push_toward_zero():
    params = torch.full((2, 3), 5.0, requires_grad=True)
    loss = l2_regularization(params)
    loss.backward()

    assert params.grad is not None
    # gradient of mean(x^2) w.r.t. a positive x is positive - pushes x down toward 0
    assert torch.all(params.grad > 0)


def test_log_scale_zero_at_reference():
    scale = torch.full((4, 1), 7.0)
    assert log_scale_regularization(scale, reference=7.0).item() == pytest.approx(0.0, abs=1e-6)


def test_log_scale_symmetric_for_proportional_over_and_under_shoot():
    # scale=3.5 (half of 7) and scale=14 (double 7) are equally "wrong" in
    # relative terms - log_scale_regularization should penalize them equally,
    # unlike plain L2 toward zero which would not.
    half = log_scale_regularization(torch.tensor([3.5]), reference=7.0)
    double = log_scale_regularization(torch.tensor([14.0]), reference=7.0)
    assert half.item() == pytest.approx(double.item(), rel=1e-5)


def test_log_scale_matches_manual_formula():
    scale = torch.tensor([2.7, 4.4, 7.4, 10.0])
    expected = (torch.log(scale / 7.0) ** 2).mean()
    assert log_scale_regularization(scale, reference=7.0).item() == pytest.approx(expected.item(), rel=1e-5)


def test_log_scale_gradient_pushes_low_scale_up():
    scale = torch.tensor([2.7], requires_grad=True)
    loss = log_scale_regularization(scale, reference=7.0)
    loss.backward()

    assert scale.grad is not None
    # below the reference, log(scale/reference) is negative -> gradient of its
    # square w.r.t. scale is negative -> Adam would increase scale, pushing
    # it back up toward the reference, not further toward 0.
    assert scale.grad.item() < 0


def test_log_scale_stays_finite_for_non_positive_scale():
    # scale has no positivity constraint upstream (plain nn.Linear output) -
    # the clamp inside log_scale_regularization must keep this finite rather
    # than propagating NaN from log() of a non-positive value.
    scale = torch.tensor([-1.0, 0.0])
    result = log_scale_regularization(scale, reference=7.0)
    assert torch.isfinite(result)
