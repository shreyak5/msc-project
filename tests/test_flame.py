import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.flame.flame import FLAME  # noqa: E402


def _zero_params(batch_size=2):
    return {
        "shape_params": torch.zeros(batch_size, 300),
        "expression_params": torch.zeros(batch_size, 100),
        "jaw_params": torch.zeros(batch_size, 3),
        "eyelid_params": torch.zeros(batch_size, 2),
        "global_rotation": torch.zeros(batch_size, 3),
    }


def test_flame_output_shapes():
    flame = FLAME()
    out = flame(**_zero_params(batch_size=2))

    assert out["vertices"].shape == (2, FLAME.NUM_VERTICES, 3)
    assert out["landmarks_fan"].shape == (2, 68, 3)
    assert out["landmarks_mp"].shape[0] == 2
    assert out["landmarks_mp"].shape[2] == 3


def test_flame_gradients_flow_to_all_inputs():
    flame = FLAME()
    params = _zero_params(batch_size=2)
    for tensor in params.values():
        tensor.requires_grad_(True)

    out = flame(**params)
    loss = out["vertices"].sum() + out["landmarks_fan"].sum() + out["landmarks_mp"].sum()
    loss.backward()

    for name, tensor in params.items():
        assert tensor.grad is not None, f"no gradient reached {name}"


def test_zero_params_gives_template_mesh():
    """All-zero shape/expression/jaw/eyelid/rotation should reproduce the raw FLAME
    template exactly - no shape/expression displacement, identity pose transforms,
    zero eyelid offset."""
    flame = FLAME()
    out = flame(**_zero_params(batch_size=2))

    template = flame.v_template.unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(out["vertices"], template, atol=1e-4)


def test_jaw_rotation_changes_vertices():
    flame = FLAME()
    params = _zero_params(batch_size=1)
    out_neutral = flame(**params)

    params["jaw_params"] = torch.tensor([[0.3, 0.0, 0.0]])
    out_open_jaw = flame(**params)

    assert not torch.allclose(out_neutral["vertices"], out_open_jaw["vertices"])


def test_eyelid_params_only_move_a_small_subset_of_vertices():
    flame = FLAME()
    params = _zero_params(batch_size=1)
    out_open = flame(**params)

    params["eyelid_params"] = torch.tensor([[1.0, 1.0]])
    out_closed = flame(**params)

    diff = (out_closed["vertices"] - out_open["vertices"]).norm(dim=-1).squeeze(0)
    moved = (diff > 1e-6).sum().item()
    assert 0 < moved < FLAME.NUM_VERTICES // 4  # eyelids are a small region of the face


def test_faces_tensor_indices_are_valid():
    flame = FLAME()
    assert flame.faces_tensor.min() >= 0
    assert flame.faces_tensor.max() < FLAME.NUM_VERTICES
