import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import constants  # noqa: E402
from model.losses.cycle import (  # noqa: E402
    augment_expression_cycle,
    expression_cycle_loss,
    identity_cycle_loss,
    load_expression_templates,
    sample_random_template,
)

NUM_EXP_TEMPLATE_DIMS = constants.EXPRESSION_TEMPLATE_NUM_DIMS


@pytest.fixture(scope="module")
def templates():
    return load_expression_templates()


def test_templates_load_and_have_expected_shape(templates):
    assert len(templates) > 0
    for key, arr in templates.items():
        assert arr.ndim == 2
        assert arr.shape[1] == NUM_EXP_TEMPLATE_DIMS


def test_sample_random_template_shape(templates):
    t = sample_random_template(templates, NUM_EXP_TEMPLATE_DIMS)
    assert t.shape == (NUM_EXP_TEMPLATE_DIMS,)


def test_augment_expression_cycle_output_shapes(templates):
    n = 16
    expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)
    jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM) * 0.1
    eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS)

    aug_expr, aug_jaw, aug_eyelid = augment_expression_cycle(
        expression, jaw, eyelid, templates, num_expression_params=NUM_EXP_TEMPLATE_DIMS
    )
    assert aug_expr.shape == expression.shape
    assert aug_jaw.shape == jaw.shape
    assert aug_eyelid.shape == eyelid.shape


def test_augment_expression_cycle_outputs_are_detached(templates):
    n = 16
    expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM, requires_grad=True)
    jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM, requires_grad=True)
    eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS, requires_grad=True)

    aug_expr, aug_jaw, aug_eyelid = augment_expression_cycle(expression, jaw, eyelid, templates)
    assert not aug_expr.requires_grad
    assert not aug_jaw.requires_grad
    assert not aug_eyelid.requires_grad


def test_augmented_jaw_clamped_to_valid_open_range(templates):
    n = 32
    expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)
    jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM) * 5.0  # deliberately out of range pre-augmentation
    eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS)

    _, aug_jaw, _ = augment_expression_cycle(expression, jaw, eyelid, templates)
    assert aug_jaw[..., 0].min() >= 0.0
    assert aug_jaw[..., 0].max() <= 0.5


def test_augmented_eyelid_clamped_to_unit_range(templates):
    n = 32
    expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)
    jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM) * 0.1
    eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS)

    _, _, aug_eyelid = augment_expression_cycle(expression, jaw, eyelid, templates)
    assert aug_eyelid.min() >= 0.0
    assert aug_eyelid.max() <= 1.0


def test_zero_expression_group_forces_jaw_exactly_zero(templates):
    """One of the 4 groups (the zero-expression group) should have jaw forced to
    exactly 0, overriding the milder co-augmentation applied to every sample."""
    torch.manual_seed(0)
    n = 40  # large enough that the zero-expression quarter (n // 4 = 10) is reliably non-empty
    expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)
    jaw = torch.ones(n, constants.FLAME_JAW_POSE_DIM) * 0.3
    eyelid = torch.ones(n, constants.NUM_EYELID_PARAMS) * 0.5

    _, aug_jaw, _ = augment_expression_cycle(expression, jaw, eyelid, templates)
    num_zero_rows = (aug_jaw == 0.0).all(dim=1).sum().item()
    assert num_zero_rows == n // 4


def test_template_injection_group_zeros_dims_beyond_template_size(templates):
    """The template-injection augmentation should zero (up to small jitter) the
    expression dims beyond what the 50-dim templates actually cover, rather than
    leaving them at their large pre-augmentation values."""
    torch.manual_seed(0)
    n = 40
    expression = torch.ones(n, constants.FLAME_EXPRESSION_DIM) * 10.0  # large, so leakage would be obvious
    jaw = torch.zeros(n, constants.FLAME_JAW_POSE_DIM)
    eyelid = torch.zeros(n, constants.NUM_EYELID_PARAMS)

    aug_expr, _, _ = augment_expression_cycle(expression, jaw, eyelid, templates, num_expression_params=NUM_EXP_TEMPLATE_DIMS)
    # at least one row must have small (jitter-only) magnitude beyond dim 50 - i.e. the
    # template-injection group actually got exercised and correctly zeroed those dims
    tail_max_per_row = aug_expr[:, NUM_EXP_TEMPLATE_DIMS:].abs().max(dim=1).values
    assert (tail_max_per_row < 1.0).any()


def test_expression_cycle_loss_zero_when_recon_matches_aug():
    n = 8
    aug_expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)
    aug_jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM)
    aug_eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS)

    loss = expression_cycle_loss(aug_expression, aug_expression, aug_jaw, aug_jaw, aug_eyelid, aug_eyelid)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_expression_cycle_loss_nonzero_when_recon_differs():
    n = 8
    aug_expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)
    aug_jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM)
    aug_eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS)
    recon_expression = aug_expression + 1.0
    recon_jaw = aug_jaw + 1.0
    recon_eyelid = aug_eyelid + 1.0

    loss = expression_cycle_loss(recon_expression, aug_expression, recon_jaw, aug_jaw, recon_eyelid, aug_eyelid)
    assert loss.item() > 0


def test_expression_cycle_loss_weights_jaw_and_eyelid_more_than_expression():
    """CYCLE_JAW_WEIGHT and CYCLE_EYELID_WEIGHT are 10x CYCLE_EXPRESSION_WEIGHT -
    an equal-magnitude mismatch in jaw/eyelid alone should produce a larger loss
    than the same-magnitude mismatch in expression alone."""
    n = 8
    aug_expression = torch.zeros(n, constants.FLAME_EXPRESSION_DIM)
    aug_jaw = torch.zeros(n, constants.FLAME_JAW_POSE_DIM)
    aug_eyelid = torch.zeros(n, constants.NUM_EYELID_PARAMS)

    expression_only_loss = expression_cycle_loss(
        aug_expression + 1.0, aug_expression, aug_jaw, aug_jaw, aug_eyelid, aug_eyelid
    )
    jaw_only_loss = expression_cycle_loss(aug_expression, aug_expression, aug_jaw + 1.0, aug_jaw, aug_eyelid, aug_eyelid)
    assert jaw_only_loss.item() > expression_only_loss.item()


def test_expression_cycle_loss_gradients_flow_only_to_recon():
    n = 8
    aug_expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM)  # detached target, no grad
    aug_jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM)
    aug_eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS)

    recon_expression = torch.randn(n, constants.FLAME_EXPRESSION_DIM, requires_grad=True)
    recon_jaw = torch.randn(n, constants.FLAME_JAW_POSE_DIM, requires_grad=True)
    recon_eyelid = torch.rand(n, constants.NUM_EYELID_PARAMS, requires_grad=True)

    loss = expression_cycle_loss(recon_expression, aug_expression, recon_jaw, aug_jaw, recon_eyelid, aug_eyelid)
    loss.backward()

    assert recon_expression.grad is not None and torch.any(recon_expression.grad != 0)
    assert recon_jaw.grad is not None and torch.any(recon_jaw.grad != 0)
    assert recon_eyelid.grad is not None and torch.any(recon_eyelid.grad != 0)


def test_identity_cycle_loss_zero_when_matching():
    shape = torch.randn(4, constants.FLAME_SHAPE_DIM)
    assert identity_cycle_loss(shape, shape).item() == pytest.approx(0.0, abs=1e-7)


def test_identity_cycle_loss_nonzero_when_differing_and_gradients_flow():
    original_shape = torch.randn(4, constants.FLAME_SHAPE_DIM)
    recon_shape = torch.randn(4, constants.FLAME_SHAPE_DIM, requires_grad=True)

    loss = identity_cycle_loss(recon_shape, original_shape)
    assert loss.item() > 0
    loss.backward()
    assert recon_shape.grad is not None and torch.any(recon_shape.grad != 0)
