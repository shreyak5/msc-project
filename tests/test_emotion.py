import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import constants  # noqa: E402
from model.emotion.emotion_net import EmotionNet  # noqa: E402
from model.losses.emotion import emotion_loss  # noqa: E402


@pytest.fixture(scope="module")
def emotion_net():
    return EmotionNet()


def test_checkpoint_loads_with_params(emotion_net):
    assert sum(p.numel() for p in emotion_net.parameters()) > 0


def test_all_params_frozen(emotion_net):
    assert all(not p.requires_grad for p in emotion_net.parameters())


def test_starts_in_eval_mode(emotion_net):
    assert not emotion_net.training


def test_forward_output_shape(emotion_net):
    n = 3
    img = torch.rand(n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE)
    feats = emotion_net(img)
    assert feats.shape == (n, 2048)


def test_forward_asserts_on_wrong_input_size(emotion_net):
    with pytest.raises(AssertionError):
        emotion_net(torch.rand(2, 3, 112, 112))


def test_emotion_loss_zero_for_identical_images(emotion_net):
    n = 2
    img = torch.rand(n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE)
    loss = emotion_loss(img, img, emotion_net)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_emotion_loss_nonzero_for_differing_images(emotion_net):
    n = 2
    reconstructed = torch.rand(n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE)
    target = torch.rand(n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE)
    loss = emotion_loss(reconstructed, target, emotion_net)
    assert loss.item() > 0


def test_gradients_flow_to_reconstructed_but_not_to_emotion_net(emotion_net):
    n = 2
    reconstructed = torch.rand(
        n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE, requires_grad=True
    )
    target = torch.rand(n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE)

    loss = emotion_loss(reconstructed, target, emotion_net)
    loss.backward()

    assert reconstructed.grad is not None
    assert torch.any(reconstructed.grad != 0)
    assert all(p.grad is None for p in emotion_net.parameters())


def test_target_branch_does_not_require_grad_even_if_input_does(emotion_net):
    """Even if the caller accidentally passes a requires_grad target, the target
    branch is computed under no_grad and detached - no gradient should reach it."""
    n = 2
    reconstructed = torch.rand(
        n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE, requires_grad=True
    )
    target = torch.rand(n, 3, constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE, requires_grad=True)

    loss = emotion_loss(reconstructed, target, emotion_net)
    loss.backward()

    assert target.grad is None
