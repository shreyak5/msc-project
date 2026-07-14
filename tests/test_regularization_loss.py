import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.regularization import l2_regularization  # noqa: E402


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
