import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.flame.flame import FLAME  # noqa: E402
from model.flame.renderer import Renderer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="rasterizer needs a GPU in this environment")


def _neutral_flame_output(flame, batch_size=2, device="cuda"):
    return flame(
        shape_params=torch.zeros(batch_size, 300, device=device),
        expression_params=torch.zeros(batch_size, 100, device=device),
        jaw_params=torch.zeros(batch_size, 3, device=device),
        eyelid_params=torch.zeros(batch_size, 2, device=device),
        global_rotation=torch.zeros(batch_size, 3, device=device),
    )


def test_renderer_output_shapes_and_range():
    device = "cuda"
    flame = FLAME().to(device)
    renderer = Renderer(faces=flame.faces_tensor).to(device)

    batch_size = 2
    flame_out = _neutral_flame_output(flame, batch_size, device)
    cam_params = torch.tensor([[6.0, 0.0, 0.0]], device=device).expand(batch_size, -1)

    out = renderer(
        flame_out["vertices"],
        cam_params,
        landmarks_fan=flame_out["landmarks_fan"],
        landmarks_mp=flame_out["landmarks_mp"],
    )

    assert out["rendered_img"].shape == (batch_size, 3, 224, 224)
    assert torch.all(out["rendered_img"] >= 0) and torch.all(out["rendered_img"] <= 1)
    assert out["transformed_vertices"].shape == (batch_size, 5023, 3)
    assert out["transformed_landmarks_fan"].shape == (batch_size, 68, 2)
    assert out["transformed_landmarks_mp"].shape == (batch_size, 105, 2)


def test_renderer_produces_a_nonempty_face_silhouette():
    """A reasonably-scaled neutral face, centered, should rasterize to a plausible
    fraction of the image being covered - not empty, not the whole frame."""
    device = "cuda"
    flame = FLAME().to(device)
    renderer = Renderer(faces=flame.faces_tensor).to(device)

    flame_out = _neutral_flame_output(flame, batch_size=1, device=device)
    cam_params = torch.tensor([[6.0, 0.0, 0.0]], device=device)

    out = renderer(flame_out["vertices"], cam_params)
    covered_fraction = (out["rendered_img"].abs().sum(1) > 0).float().mean().item()
    assert 0.05 < covered_fraction < 0.9


def test_face_region_mask_reduces_face_count():
    """render_full_head=False (the default, Sec 2.5/9) should rasterize noticeably
    fewer triangles than the full FLAME head topology."""
    device = "cuda"
    flame = FLAME().to(device)
    renderer = Renderer(faces=flame.faces_tensor, render_full_head=False).to(device)

    assert renderer.faces.shape[1] < flame.faces_tensor.shape[0]


def test_gradients_flow_from_rendered_image_to_flame_params():
    device = "cuda"
    flame = FLAME().to(device)
    renderer = Renderer(faces=flame.faces_tensor).to(device)

    batch_size = 1
    shape_params = torch.zeros(batch_size, 300, device=device, requires_grad=True)
    expression_params = torch.zeros(batch_size, 100, device=device, requires_grad=True)
    jaw_params = torch.zeros(batch_size, 3, device=device, requires_grad=True)
    eyelid_params = torch.zeros(batch_size, 2, device=device, requires_grad=True)
    global_rotation = torch.zeros(batch_size, 3, device=device, requires_grad=True)
    cam_params = torch.tensor([[6.0, 0.0, 0.0]], device=device, requires_grad=True)

    flame_out = flame(shape_params, expression_params, jaw_params, eyelid_params, global_rotation)
    render_out = renderer(flame_out["vertices"], cam_params)

    render_out["rendered_img"].sum().backward()

    assert shape_params.grad is not None
    assert expression_params.grad is not None
    assert jaw_params.grad is not None
    assert global_rotation.grad is not None
    assert cam_params.grad is not None
