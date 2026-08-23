import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.losses.temporal_smoothness import compute_vertex_gate, velocity_penalty, vertex_velocity_penalty  # noqa: E402


def test_zero_for_constant_sequence():
    params = torch.ones(2, 8, 5) * 3.0
    assert velocity_penalty(params).item() == pytest.approx(0.0)


def test_nonzero_for_jittery_sequence():
    torch.manual_seed(0)
    jittery = torch.randn(2, 8, 5)
    assert velocity_penalty(jittery).item() > 0


def test_matches_manual_first_difference_formula():
    torch.manual_seed(1)
    params = torch.randn(2, 6, 3)

    manual_first_diff = params[:, :-1] - params[:, 1:]
    expected_vel = manual_first_diff.pow(2).mean()
    assert velocity_penalty(params).item() == pytest.approx(expected_vel.item(), rel=1e-5)


def test_gradients_flow():
    params = torch.randn(2, 8, 5, requires_grad=True)
    loss = velocity_penalty(params)
    loss.backward()
    assert params.grad is not None
    assert torch.any(params.grad != 0)


def test_minimum_frame_count_edge_case():
    velocity_penalty(torch.randn(1, 2, 4))  # minimum for velocity


def test_asserts_on_too_few_frames():
    with pytest.raises(AssertionError):
        velocity_penalty(torch.randn(1, 1, 4))


def test_frame_weight_of_all_ones_matches_unweighted_valid_mask():
    torch.manual_seed(2)
    params = torch.randn(2, 6, 3)
    valid_mask = torch.ones(2, 6, dtype=torch.bool)
    frame_weight = torch.ones(2, 6)

    unweighted = velocity_penalty(params, valid_mask)
    weighted = velocity_penalty(params, valid_mask, frame_weight)
    assert weighted.item() == pytest.approx(unweighted.item(), rel=1e-5)


def test_frame_weight_matches_manual_weighted_formula():
    # Pass C's actual construction: occlusion_loss_weight at occluded positions,
    # 1.0 elsewhere (training/stage2.py's compute_temporal_smoothness_losses).
    torch.manual_seed(3)
    params = torch.randn(1, 6, 4)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)
    frame_weight = torch.tensor([[1.0, 1.0, 3.0, 1.0, 1.0, 1.0]])

    first_diff = params[:, :-1] - params[:, 1:]
    sq_diff = first_diff.pow(2)
    pair_weight = torch.maximum(frame_weight[:, :-1], frame_weight[:, 1:])  # (1, 5)
    expected = (sq_diff * pair_weight.unsqueeze(-1)).sum() / (pair_weight.sum() * params.shape[-1])

    actual = velocity_penalty(params, valid_mask, frame_weight)
    assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


def test_frame_weight_upweight_increases_loss_for_jittery_pair():
    # A frame whose adjacent jumps are large should contribute more to the total
    # once its frame_weight is raised - confirms the weighting actually biases the
    # mean toward that frame's transitions, not just a no-op reshuffle.
    torch.manual_seed(4)
    params = torch.randn(1, 6, 4)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)

    baseline = velocity_penalty(params, valid_mask)
    frame_weight = torch.ones(1, 6)
    frame_weight[0, 2] = 5.0
    upweighted = velocity_penalty(params, valid_mask, frame_weight)

    assert upweighted.item() != pytest.approx(baseline.item(), rel=1e-5)


def test_frame_weight_gradients_flow():
    params = torch.randn(1, 6, 4, requires_grad=True)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)
    frame_weight = torch.tensor([[1.0, 1.0, 2.0, 1.0, 1.0, 1.0]])

    loss = velocity_penalty(params, valid_mask, frame_weight)
    loss.backward()
    assert params.grad is not None
    assert torch.any(params.grad != 0)


# --- vertex_velocity_penalty (occlusion-experiment1.md Change 2) ---


def test_vertex_velocity_zero_for_constant_sequence():
    vertices = torch.ones(2, 4, 5, 3) * 3.0
    base_region_weights = torch.ones(5)
    gated_region_mask = torch.zeros(5)
    gate = torch.zeros(2, 3)
    loss = vertex_velocity_penalty(vertices, base_region_weights, gated_region_mask, gate, 3.0)
    assert loss.item() == pytest.approx(0.0)


def test_vertex_velocity_matches_manual_weighted_formula():
    torch.manual_seed(0)
    batch_size, num_frames, num_vertices = 2, 4, 5
    vertices = torch.randn(batch_size, num_frames, num_vertices, 3)
    base_region_weights = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
    gated_region_mask = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    gate = torch.rand(batch_size, num_frames - 1)
    expressive_region_smooth_weight = 3.5

    sq_diff = (vertices[:, :-1] - vertices[:, 1:]).pow(2)
    gated_effective = 1.0 + (expressive_region_smooth_weight - 1.0) * gate
    manual_weight = (
        base_region_weights.view(1, 1, -1) + gated_region_mask.view(1, 1, -1) * gated_effective.unsqueeze(-1)
    )
    denom = manual_weight.sum() * 3
    expected = (sq_diff * manual_weight.unsqueeze(-1)).sum() / denom

    actual = vertex_velocity_penalty(
        vertices, base_region_weights, gated_region_mask, gate, expressive_region_smooth_weight,
    )
    assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


def test_vertex_velocity_gate_zero_gives_baseline_weight():
    """At gate=0, a gated-region vertex should carry the SAME effective weight
    (1.0) as an ordinary included (base_region_weights=1) vertex - the formula's
    baseline, before any occlusion-driven ramp-up."""
    torch.manual_seed(1)
    vertices = torch.randn(1, 3, 2, 3)
    base_region_weights = torch.tensor([1.0, 0.0])
    gated_region_mask = torch.tensor([0.0, 1.0])
    gate = torch.zeros(1, 2)

    loss = vertex_velocity_penalty(vertices, base_region_weights, gated_region_mask, gate, 5.0)
    sq_diff = (vertices[:, :-1] - vertices[:, 1:]).pow(2)
    expected = sq_diff.mean()  # both vertices weighted 1.0 -> reduces to a plain mean
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_vertex_velocity_gate_one_gives_peak_weight():
    """At gate=1, a gated-region vertex's effective weight should equal
    expressive_region_smooth_weight exactly."""
    torch.manual_seed(2)
    vertices = torch.randn(1, 3, 2, 3)
    base_region_weights = torch.tensor([1.0, 0.0])
    gated_region_mask = torch.tensor([0.0, 1.0])
    gate = torch.ones(1, 2)
    expressive_region_smooth_weight = 4.0

    loss = vertex_velocity_penalty(
        vertices, base_region_weights, gated_region_mask, gate, expressive_region_smooth_weight,
    )
    sq_diff = (vertices[:, :-1] - vertices[:, 1:]).pow(2)
    per_vertex_weight = torch.tensor([1.0, expressive_region_smooth_weight]).view(1, 1, -1)
    weight = per_vertex_weight.expand(1, 2, 2)
    denom = weight.sum() * 3
    expected = (sq_diff * weight.unsqueeze(-1)).sum() / denom
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_vertex_velocity_excluded_region_contributes_nothing():
    torch.manual_seed(3)
    vertices = torch.randn(1, 3, 3, 3)
    base_region_weights = torch.tensor([1.0, 0.0, 0.0])
    gated_region_mask = torch.tensor([0.0, 1.0, 0.0])  # vertex 2: excluded (0 in both)
    gate = torch.rand(1, 2)

    baseline = vertex_velocity_penalty(vertices, base_region_weights, gated_region_mask, gate, 3.0)
    corrupted = vertices.clone()
    corrupted[:, :, 2, :] += 1000.0
    corrupted_loss = vertex_velocity_penalty(corrupted, base_region_weights, gated_region_mask, gate, 3.0)
    assert corrupted_loss.item() == pytest.approx(baseline.item(), rel=1e-5)


def test_vertex_velocity_valid_mask_excludes_padded_pairs():
    torch.manual_seed(4)
    vertices = torch.randn(1, 4, 2, 3)
    base_region_weights = torch.ones(2)
    gated_region_mask = torch.zeros(2)
    gate = torch.zeros(1, 3)
    valid_mask = torch.tensor([[True, True, True, False]])  # last frame is tail-padding

    loss = vertex_velocity_penalty(
        vertices, base_region_weights, gated_region_mask, gate, 3.0, valid_mask=valid_mask,
    )
    expected = vertex_velocity_penalty(
        vertices[:, :3], base_region_weights, gated_region_mask, gate[:, :2], 3.0,
    )
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_vertex_velocity_gradients_flow():
    vertices = torch.randn(1, 3, 4, 3, requires_grad=True)
    base_region_weights = torch.tensor([1.0, 0.0, 0.0, 1.0])
    gated_region_mask = torch.tensor([0.0, 1.0, 0.0, 0.0])
    gate = torch.rand(1, 2)

    loss = vertex_velocity_penalty(vertices, base_region_weights, gated_region_mask, gate, 3.0)
    loss.backward()

    assert vertices.grad is not None
    assert torch.any(vertices.grad != 0)
    # vertex 2 is excluded (0 weight in both tensors) - must get exactly zero grad.
    assert torch.all(vertices.grad[:, :, 2, :] == 0)


def test_vertex_velocity_asserts_on_too_few_frames():
    with pytest.raises(AssertionError):
        vertex_velocity_penalty(
            torch.randn(1, 1, 3, 3), torch.ones(3), torch.zeros(3), torch.zeros(1, 0), 3.0,
        )


# --- compute_vertex_gate (Stage2Config.vertex_gate_mode) ---


def test_min_vis_matches_original_inline_formula():
    torch.manual_seed(0)
    gate_signal = torch.rand(2, 5)
    expected = 1.0 - torch.minimum(gate_signal[:, :-1], gate_signal[:, 1:])
    actual = compute_vertex_gate(gate_signal, mode="min_vis")
    assert torch.allclose(actual, expected)


def test_min_vis_ignores_cap_and_beta():
    torch.manual_seed(1)
    gate_signal = torch.rand(2, 5)
    default = compute_vertex_gate(gate_signal, mode="min_vis")
    other_params = compute_vertex_gate(gate_signal, mode="min_vis", cap=0.5, beta=10.0)
    assert torch.allclose(default, other_params)


def test_delta_vis_zero_diff_gives_zero_gate():
    gate_signal = torch.full((1, 4), 0.7)  # constant visibility, no frame-to-frame change
    gate = compute_vertex_gate(gate_signal, mode="delta_vis", cap=0.1, beta=1.0)
    assert torch.allclose(gate, torch.zeros(1, 3))


def test_delta_vis_saturates_at_cap():
    gate_signal = torch.tensor([[0.0, 0.5, 0.0]])  # diffs of 0.5, well above any reasonable cap
    gate = compute_vertex_gate(gate_signal, mode="delta_vis", cap=0.1, beta=1.0)
    assert torch.allclose(gate, torch.ones(1, 2))


def test_delta_vis_matches_manual_formula_below_cap():
    gate_signal = torch.tensor([[0.50, 0.52]])  # diff = 0.02, half of cap=0.04
    gate = compute_vertex_gate(gate_signal, mode="delta_vis", cap=0.04, beta=1.0)
    assert torch.allclose(gate, torch.tensor([[0.5]]), atol=1e-6)


def test_delta_vis_beta_widens_gap_between_small_and_large_diff():
    # beta > 1 should suppress a small normalized diff much more than a large
    # one - i.e. widen (not narrow) the separation as beta increases, per
    # Stage2Config.vertex_gate_delta_beta's own docstring.
    gate_signal_small_diff = torch.tensor([[0.0, 0.02]])  # norm = 0.2 at cap=0.1
    gate_signal_large_diff = torch.tensor([[0.0, 0.09]])  # norm = 0.9 at cap=0.1

    small_at_beta1 = compute_vertex_gate(gate_signal_small_diff, mode="delta_vis", cap=0.1, beta=1.0)
    large_at_beta1 = compute_vertex_gate(gate_signal_large_diff, mode="delta_vis", cap=0.1, beta=1.0)
    small_at_beta3 = compute_vertex_gate(gate_signal_small_diff, mode="delta_vis", cap=0.1, beta=3.0)
    large_at_beta3 = compute_vertex_gate(gate_signal_large_diff, mode="delta_vis", cap=0.1, beta=3.0)

    gap_at_beta1 = (large_at_beta1 - small_at_beta1).item()
    gap_at_beta3 = (large_at_beta3 - small_at_beta3).item()
    assert gap_at_beta3 > gap_at_beta1


def test_delta_vis_beta_one_is_a_no_op():
    torch.manual_seed(2)
    gate_signal = torch.rand(2, 5)
    norm_only = compute_vertex_gate(gate_signal, mode="delta_vis", cap=0.2, beta=1.0)
    raw = (gate_signal[:, :-1] - gate_signal[:, 1:]).abs()
    expected = (raw / 0.2).clamp(0.0, 1.0)
    assert torch.allclose(norm_only, expected)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        compute_vertex_gate(torch.rand(1, 3), mode="bogus")
