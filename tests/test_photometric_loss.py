import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.photometric import VGGPerceptualLoss, photometric_loss  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="VGG weights/compute assumed GPU in this environment")


def test_photometric_loss_zero_for_identical_images():
    img = torch.rand(2, 3, 224, 224)
    assert photometric_loss(img, img).item() == pytest.approx(0.0, abs=1e-6)


def test_photometric_loss_matches_manual_l1():
    a = torch.rand(2, 3, 224, 224)
    b = torch.rand(2, 3, 224, 224)
    expected = (a - b).abs().mean()
    assert photometric_loss(a, b).item() == pytest.approx(expected.item(), abs=1e-6)


def test_photometric_loss_gradients_flow():
    a = torch.rand(2, 3, 224, 224, requires_grad=True)
    b = torch.rand(2, 3, 224, 224)
    photometric_loss(a, b).backward()
    assert a.grad is not None


def test_vgg_loss_zero_for_identical_images():
    device = "cuda"
    vgg = VGGPerceptualLoss().to(device)
    img = torch.rand(1, 3, 224, 224, device=device)
    assert vgg(img, img).item() == pytest.approx(0.0, abs=1e-4)


def test_vgg_loss_all_params_frozen():
    vgg = VGGPerceptualLoss()
    assert sum(p.numel() for p in vgg.parameters() if p.requires_grad) == 0


def test_vgg_loss_gradients_flow_to_input_not_vgg_weights():
    device = "cuda"
    vgg = VGGPerceptualLoss().to(device)
    recon = torch.rand(1, 3, 224, 224, device=device, requires_grad=True)
    target = torch.rand(1, 3, 224, 224, device=device)

    vgg(recon, target).backward()

    assert recon.grad is not None
    assert all(p.grad is None for p in vgg.parameters())


def test_vgg_loss_handles_non_224_input_via_interpolation():
    device = "cuda"
    vgg = VGGPerceptualLoss().to(device)
    recon = torch.rand(1, 3, 96, 96, device=device)
    target = torch.rand(1, 3, 96, 96, device=device)

    loss = vgg(recon, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0
