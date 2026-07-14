import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import constants  # noqa: E402
from model.losses.mica_shape import mica_shape_loss  # noqa: E402
from model.mica.mica import MICA  # noqa: E402


@pytest.fixture(scope="module")
def mica():
    return MICA()


def test_checkpoint_loads_with_expected_param_counts(mica):
    assert sum(p.numel() for p in mica.arcface.parameters()) > 0
    assert sum(p.numel() for p in mica.regressor.parameters()) > 0


def test_forward_output_shape(mica):
    n = 4
    img = torch.rand(n, 3, constants.MICA_IMAGE_SIZE, constants.MICA_IMAGE_SIZE)
    out = mica(img)
    assert out.shape == (n, constants.FLAME_SHAPE_DIM)


def test_forward_asserts_on_wrong_input_size(mica):
    with pytest.raises(AssertionError):
        mica(torch.rand(2, 3, 224, 224))


def test_early_arcface_layers_are_frozen(mica):
    frozen_params = (
        list(mica.arcface.conv1.parameters())
        + list(mica.arcface.bn1.parameters())
        + list(mica.arcface.prelu.parameters())
        + list(mica.arcface.layer1.parameters())
        + list(mica.arcface.layer2.parameters())
        + list(mica.arcface.layer3.parameters())
    )
    assert len(frozen_params) > 0
    assert all(not p.requires_grad for p in frozen_params)


def test_late_arcface_layers_are_trainable(mica):
    trainable_params = list(mica.arcface.layer4.parameters()) + list(mica.arcface.fc.parameters())
    assert len(trainable_params) > 0
    assert all(p.requires_grad for p in trainable_params)


def test_features_batchnorm_weight_is_frozen(mica):
    """A behavioral flag from the original arcface.py, not a checkpoint-loaded
    value - must survive the state_dict load."""
    assert not mica.arcface.features.weight.requires_grad


def test_mica_shape_loss_zero_when_shape_matches_mica_output(mica):
    n = 2
    img_mica = torch.rand(n, 3, constants.MICA_IMAGE_SIZE, constants.MICA_IMAGE_SIZE)
    with torch.no_grad():
        target_shape = mica(img_mica)

    loss = mica_shape_loss(target_shape, mica, img_mica)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_mica_shape_loss_nonzero_when_shape_differs(mica):
    n = 2
    img_mica = torch.rand(n, 3, constants.MICA_IMAGE_SIZE, constants.MICA_IMAGE_SIZE)
    shape_params = torch.randn(n, constants.FLAME_SHAPE_DIM)

    loss = mica_shape_loss(shape_params, mica, img_mica)
    assert loss.item() > 0


def test_mica_shape_loss_gradients_flow_to_shape_params_only(mica):
    n = 2
    img_mica = torch.rand(n, 3, constants.MICA_IMAGE_SIZE, constants.MICA_IMAGE_SIZE)
    shape_params = torch.randn(n, constants.FLAME_SHAPE_DIM, requires_grad=True)

    loss = mica_shape_loss(shape_params, mica, img_mica)
    loss.backward()

    assert shape_params.grad is not None
    assert torch.any(shape_params.grad != 0)
    assert all(p.grad is None for p in mica.parameters())
