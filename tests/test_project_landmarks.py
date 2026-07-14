import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.flame.renderer import project_landmarks  # noqa: E402

# Deliberately no CUDA skip here (unlike test_renderer.py) - project_landmarks is
# pure tensor math, no PyTorch3D rasterizer involved, exactly the point of
# factoring it out of Renderer.forward() in the first place.


def test_output_shape():
    points = torch.randn(2, 10, 3)
    camera = torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1)
    out = project_landmarks(points, camera)
    assert out.shape == (2, 10, 2)


def test_y_sign_is_flipped_to_image_convention():
    """A point above the origin in FLAME's Y-up mesh space (y=+1) should map to a
    negative y in the output (image convention: y increases downward) - and a
    point below the origin (y=-1) should map to a positive output y."""
    points = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])  # (1, 2, 3)
    camera = torch.tensor([[1.0, 0.0, 0.0]])  # scale=1, no translation

    out = project_landmarks(points, camera)
    assert out[0, 0, 1].item() == pytest.approx(-1.0)
    assert out[0, 1, 1].item() == pytest.approx(1.0)


def test_scale_and_translation_are_applied():
    points = torch.tensor([[[1.0, 0.0, 0.0]]])  # (1, 1, 3)
    camera = torch.tensor([[2.0, 0.5, 0.0]])  # scale=2, tx=0.5, ty=0

    out = project_landmarks(points, camera)
    # x: scale * (x + tx) = 2 * (1 + 0.5) = 3.0
    assert out[0, 0, 0].item() == pytest.approx(3.0)


def test_gradients_flow_to_points_and_camera():
    points = torch.randn(2, 5, 3, requires_grad=True)
    camera = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)

    out = project_landmarks(points, camera)
    out.sum().backward()

    assert points.grad is not None
    assert torch.any(points.grad != 0)
    assert camera.grad is not None
    assert torch.any(camera.grad != 0)


def test_matches_renderer_forward_landmark_output():
    """Consistency check against the exact same math Renderer.forward() applies
    to landmarks internally (now delegating to this function) - guards against
    the extraction having silently changed behavior."""
    points = torch.randn(3, 7, 3)
    camera = torch.tensor([[1.5, 0.1, -0.2]]).expand(3, -1)

    from model.flame.renderer import batch_orth_proj
    manual = batch_orth_proj(points.clone(), camera.clone())
    manual[:, :, 1:] = -manual[:, :, 1:]
    manual = manual[..., :2]

    out = project_landmarks(points, camera)
    assert torch.allclose(out, manual)
