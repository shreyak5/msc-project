import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.flame.flame import FLAME  # noqa: E402
from model.losses.mesh import build_region_weights, vertex_consistency_loss  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="uses FLAME's full model, GPU in this environment")


@pytest.fixture(scope="module")
def flame_and_weights():
    device = "cuda"
    flame = FLAME().to(device)
    weights = build_region_weights().to(device)
    return flame, weights, device


def test_zero_loss_when_shapes_already_match(flame_and_weights):
    """If the two same-identity samples' shape estimates are already identical, the
    shape swap is a no-op and Lvc should be exactly zero."""
    flame, weights, device = flame_and_weights
    shape_b = torch.randn(1, 300, device=device) * 0.1
    expr_b = torch.randn(1, 100, device=device) * 0.3
    jaw_b = torch.tensor([[0.1, 0.0, 0.0]], device=device)
    eyelid_b = torch.zeros(1, 2, device=device)
    rot_b = torch.zeros(1, 3, device=device)

    gt_vertices = flame(shape_b, expr_b, jaw_b, eyelid_b, rot_b)["vertices"]
    swapped_vertices = flame(shape_b, expr_b, jaw_b, eyelid_b, rot_b)["vertices"]  # same shape "swapped" in

    loss = vertex_consistency_loss(swapped_vertices, gt_vertices, weights)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_nonzero_loss_when_shapes_differ(flame_and_weights):
    flame, weights, device = flame_and_weights
    torch.manual_seed(0)
    shape_a = torch.randn(1, 300, device=device) * 0.1
    shape_b = torch.randn(1, 300, device=device) * 0.1
    expr_b = torch.randn(1, 100, device=device) * 0.3
    jaw_b = torch.tensor([[0.1, 0.0, 0.0]], device=device)
    eyelid_b = torch.zeros(1, 2, device=device)
    rot_b = torch.zeros(1, 3, device=device)

    gt_vertices = flame(shape_b, expr_b, jaw_b, eyelid_b, rot_b)["vertices"]
    swapped_vertices = flame(shape_a, expr_b, jaw_b, eyelid_b, rot_b)["vertices"]

    loss = vertex_consistency_loss(swapped_vertices, gt_vertices, weights)
    assert loss.item() > 0


def test_uses_same_region_weights_as_mesh_loss():
    """vertex_consistency_loss should be numerically identical to
    region_weighted_mesh_loss given the same inputs - confirms it isn't
    accidentally using a different weighting scheme."""
    from model.losses.mesh import region_weighted_mesh_loss

    weights = build_region_weights()
    pred = torch.rand(2, 5023, 3)
    gt = torch.rand(2, 5023, 3)

    assert vertex_consistency_loss(pred, gt, weights).item() == pytest.approx(
        region_weighted_mesh_loss(pred, gt, weights).item()
    )


def test_gradients_flow_to_the_swapped_in_shape(flame_and_weights):
    flame, weights, device = flame_and_weights
    shape_a = (torch.randn(1, 300, device=device) * 0.1).requires_grad_(True)
    shape_b = torch.randn(1, 300, device=device) * 0.1
    expr_b = torch.randn(1, 100, device=device) * 0.3
    jaw_b = torch.tensor([[0.1, 0.0, 0.0]], device=device)
    eyelid_b = torch.zeros(1, 2, device=device)
    rot_b = torch.zeros(1, 3, device=device)

    gt_vertices = flame(shape_b, expr_b, jaw_b, eyelid_b, rot_b)["vertices"].detach()
    swapped_vertices = flame(shape_a, expr_b, jaw_b, eyelid_b, rot_b)["vertices"]

    loss = vertex_consistency_loss(swapped_vertices, gt_vertices, weights)
    loss.backward()

    assert shape_a.grad is not None
    assert torch.any(shape_a.grad != 0)
