import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.loss_utils import gated_loss  # noqa: E402


def test_zero_when_nothing_valid():
    predicted = torch.randn(4, 5)
    target = torch.randn(4, 5)
    valid_mask = torch.zeros(4, dtype=torch.bool)

    loss = gated_loss(F.mse_loss, valid_mask, predicted, target)
    assert loss.item() == 0.0


def test_matches_manual_loss_over_valid_rows_only():
    predicted = torch.randn(6, 3)
    target = torch.randn(6, 3)
    valid_mask = torch.tensor([True, False, True, False, True, False])

    loss = gated_loss(F.mse_loss, valid_mask, predicted, target)
    expected = F.mse_loss(predicted[valid_mask], target[valid_mask])
    assert loss.item() == pytest.approx(expected.item())


def test_invalid_rows_do_not_contaminate_the_loss():
    """A mix of valid/invalid rows: garbage values in the invalid rows should
    have zero effect on the result, since they're filtered out before the loss
    function ever sees them."""
    predicted = torch.zeros(4, 2)
    target = torch.zeros(4, 2)
    valid_mask = torch.tensor([True, False, True, False])

    # Corrupt only the invalid rows with huge garbage values.
    predicted[~valid_mask] = 1e6
    target[~valid_mask] = -1e6

    loss = gated_loss(F.mse_loss, valid_mask, predicted, target)
    assert loss.item() == pytest.approx(0.0)


def test_gradients_flow_only_through_valid_rows():
    predicted = torch.randn(4, 2, requires_grad=True)
    target = torch.randn(4, 2)
    valid_mask = torch.tensor([True, False, True, False])

    loss = gated_loss(F.mse_loss, valid_mask, predicted, target)
    loss.backward()

    assert predicted.grad is not None
    assert torch.all(predicted.grad[~valid_mask] == 0)
    assert torch.any(predicted.grad[valid_mask] != 0)


def test_all_valid_matches_plain_loss():
    predicted = torch.randn(5, 2)
    target = torch.randn(5, 2)
    valid_mask = torch.ones(5, dtype=torch.bool)

    loss = gated_loss(F.mse_loss, valid_mask, predicted, target)
    expected = F.mse_loss(predicted, target)
    assert loss.item() == pytest.approx(expected.item())
