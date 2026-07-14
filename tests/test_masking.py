import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import constants  # noqa: E402
from model.flame import masking as masking_utils  # noqa: E402
from model.flame.flame import FLAME  # noqa: E402
from model.flame.renderer import Renderer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="uses the renderer, which needs a GPU here")


def _render_neutral_face(device="cuda", batch_size=2):
    flame = FLAME().to(device)
    renderer = Renderer(faces=flame.faces_tensor).to(device)
    flame_out = flame(
        torch.zeros(batch_size, 300, device=device),
        torch.zeros(batch_size, 100, device=device),
        torch.zeros(batch_size, 3, device=device),
        torch.zeros(batch_size, 2, device=device),
        torch.zeros(batch_size, 3, device=device),
    )
    cam_params = torch.tensor([[6.0, 0.0, 0.0]], device=device).expand(batch_size, -1)
    render_out = renderer(flame_out["vertices"], cam_params)
    return flame, render_out


def test_load_probabilities_shape_and_known_regions():
    probs = masking_utils.load_probabilities_per_flame_triangle()
    assert probs.shape == (constants.NUM_FLAME_FACES,)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    # not every triangle is eligible - zero-weight regions (neck/ears/eyeballs) exist
    assert (probs == 0).any()
    assert (probs == 1).any()


def test_mesh_based_mask_uniform_faces_shape_and_bounds():
    device = "cuda"
    flame, render_out = _render_neutral_face(device)
    face_probs = masking_utils.load_probabilities_per_flame_triangle().to(device)

    npoints, coords = masking_utils.mesh_based_mask_uniform_faces(
        render_out["transformed_vertices"], flame.faces_tensor, face_probs
    )

    expected_num_points = int(constants.MASK_RATIO * constants.RENDERER_IMAGE_SIZE**2)
    assert npoints.shape == (2, expected_num_points, 3)
    assert torch.all(npoints[..., 0] >= 0) and torch.all(npoints[..., 0] < constants.RENDERER_IMAGE_SIZE)
    assert torch.all(npoints[..., 1] >= 0) and torch.all(npoints[..., 1] < constants.RENDERER_IMAGE_SIZE)
    assert set(coords.keys()) == {"sampled_faces_indices", "barycentric_coords"}


def test_mesh_based_mask_reuses_coords_deterministically():
    device = "cuda"
    flame, render_out = _render_neutral_face(device)
    face_probs = masking_utils.load_probabilities_per_flame_triangle().to(device)

    npoints_a, coords = masking_utils.mesh_based_mask_uniform_faces(
        render_out["transformed_vertices"], flame.faces_tensor, face_probs
    )
    npoints_b, _ = masking_utils.mesh_based_mask_uniform_faces(
        render_out["transformed_vertices"], flame.faces_tensor, face_probs, coords=coords
    )

    assert torch.equal(npoints_a, npoints_b)


def test_transfer_pixels_extracts_exact_values_when_points_match():
    device = "cuda"
    img = torch.rand(2, 3, 224, 224, device=device)
    points = torch.randint(0, 224, (2, 50, 2), device=device)

    extra_points = masking_utils.transfer_pixels(img, points, points)

    for b in range(2):
        for i in range(50):
            x, y = points[b, i, 0].item(), points[b, i, 1].item()
            assert torch.equal(extra_points[b, :, y, x], img[b, :, y, x])


def test_masking_keeps_background_blacks_out_interior_except_extra_points():
    device = "cuda"
    img = torch.rand(1, 3, 64, 64, device=device)
    # mask=1 (kept) everywhere except a black-out region in the middle
    mask = torch.ones(1, 1, 64, 64, device=device)
    mask[:, :, 20:44, 20:44] = 0

    extra_points = torch.zeros_like(img)
    extra_points[:, :, 30, 30] = torch.tensor([0.5, 0.5, 0.5], device=device)

    out = masking_utils.masking(img, mask, extra_points, wr=0, extra_noise=False, random_mask=0)

    # far corner (background, mask=1, away from the eroded boundary) should be kept
    assert torch.allclose(out[:, :, 5, 5], img[:, :, 5, 5])
    # deep inside the blackout region (not the extra_points location) should be zero
    assert torch.allclose(out[:, :, 25, 25], torch.zeros(3, device=device))
    # the extra_points location should show through
    assert torch.allclose(out[:, :, 30, 30], torch.tensor([0.5, 0.5, 0.5], device=device))
