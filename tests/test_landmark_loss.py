import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.flame.flame import FLAME  # noqa: E402
from model.flame.renderer import Renderer  # noqa: E402
from model.losses import landmark as landmark_losses  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="uses the renderer, which needs a GPU here")


def _neutral_projected_landmarks(device="cuda", batch_size=2):
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
    render_out = renderer(
        flame_out["vertices"], cam_params, landmarks_fan=flame_out["landmarks_fan"], landmarks_mp=flame_out["landmarks_mp"]
    )
    return render_out["transformed_landmarks_fan"], render_out["transformed_landmarks_mp"]


def test_index_arrays_shapes_and_uniqueness():
    assert landmark_losses._UPPER_EYELID_IDX.shape == landmark_losses._LOWER_EYELID_IDX.shape
    assert landmark_losses._UPPER_LIP_IDX.shape == landmark_losses._LOWER_LIP_IDX.shape
    for idx in (
        landmark_losses._UPPER_EYELID_IDX,
        landmark_losses._LOWER_EYELID_IDX,
        landmark_losses._UPPER_LIP_IDX,
        landmark_losses._LOWER_LIP_IDX,
    ):
        assert idx.min() >= 0
        assert idx.max() < 105
        assert len(idx.unique()) == len(idx)  # no duplicate indices within a group


def test_fan_boundary_loss_zero_for_identical_and_matches_manual_mse():
    fan, _ = _neutral_projected_landmarks()
    assert landmark_losses.fan_boundary_loss(fan, fan).item() == pytest.approx(0.0, abs=1e-6)

    noisy = fan + 0.01
    expected = torch.nn.functional.mse_loss(
        fan[:, : landmark_losses.NUM_FAN_BOUNDARY_POINTS], noisy[:, : landmark_losses.NUM_FAN_BOUNDARY_POINTS]
    )
    assert landmark_losses.fan_boundary_loss(fan, noisy).item() == pytest.approx(expected.item(), rel=1e-5)


def test_mediapipe_landmark_loss_zero_for_identical():
    _, mp = _neutral_projected_landmarks()
    assert landmark_losses.mediapipe_landmark_loss(mp, mp).item() == pytest.approx(0.0, abs=1e-6)


def test_eye_and_lip_closure_loss_zero_for_identical():
    _, mp = _neutral_projected_landmarks()
    assert landmark_losses.eye_closure_loss(mp, mp).item() == pytest.approx(0.0, abs=1e-6)
    assert landmark_losses.lip_closure_loss(mp, mp).item() == pytest.approx(0.0, abs=1e-6)


def test_eye_closure_loss_detects_opening_amount_change_not_just_position():
    """Shifting every landmark by the SAME translation shouldn't change the eye-
    opening distance (and thus the closure loss should stay ~zero), unlike a naive
    position-based loss which would report a large error."""
    _, mp = _neutral_projected_landmarks(batch_size=1)
    shifted = mp + torch.tensor([0.05, 0.03], device=mp.device)

    assert landmark_losses.eye_closure_loss(mp, shifted).item() == pytest.approx(0.0, abs=1e-5)
    assert landmark_losses.lip_closure_loss(mp, shifted).item() == pytest.approx(0.0, abs=1e-5)
    # sanity: the base position loss WOULD be large for this same shifted pair
    assert landmark_losses.mediapipe_landmark_loss(mp, shifted).item() > 1e-4


def test_eye_closure_loss_responds_to_actual_closing():
    """Moving the lower eyelid points up toward the upper ones (simulating closing
    the eyes) should reduce the opening distance and register as nonzero loss
    relative to an unmodified target."""
    _, mp = _neutral_projected_landmarks(batch_size=1)
    closed = mp.clone()
    upper = closed[:, landmark_losses._UPPER_EYELID_IDX]
    lower_idx = landmark_losses._LOWER_EYELID_IDX
    closed[:, lower_idx] = 0.5 * (closed[:, lower_idx] + upper)  # move lower eyelid halfway to upper

    loss = landmark_losses.eye_closure_loss(closed, mp)
    assert loss.item() > 1e-4


def test_gradients_flow_through_renderer_and_flame():
    device = "cuda"
    flame = FLAME().to(device)
    renderer = Renderer(faces=flame.faces_tensor).to(device)

    shape_params = torch.zeros(1, 300, device=device, requires_grad=True)
    flame_out = flame(
        shape_params,
        torch.zeros(1, 100, device=device),
        torch.zeros(1, 3, device=device),
        torch.zeros(1, 2, device=device),
        torch.zeros(1, 3, device=device),
    )
    cam_params = torch.tensor([[6.0, 0.0, 0.0]], device=device)
    render_out = renderer(flame_out["vertices"], cam_params, landmarks_mp=flame_out["landmarks_mp"])

    target = render_out["transformed_landmarks_mp"].detach() + 0.02
    loss = landmark_losses.eye_closure_loss(
        render_out["transformed_landmarks_mp"], target
    ) + landmark_losses.lip_closure_loss(render_out["transformed_landmarks_mp"], target)
    loss.backward()

    assert shape_params.grad is not None
