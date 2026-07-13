import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import TTConfig  # noqa: E402
from model.temporal import TemporalTransformer  # noqa: E402


def _make_inputs(batch_size=2, num_frames=11, num_components=4, dim=768, device="cpu"):
    tokens = torch.randn(batch_size, num_frames, num_components, dim, device=device)
    visibility = torch.rand(batch_size, num_frames, device=device)
    frame_indices = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1).float()
    return tokens, visibility, frame_indices


def test_output_shape_matches_input():
    tt = TemporalTransformer()
    tokens, visibility, frame_indices = _make_inputs()
    out = tt(tokens, visibility, frame_indices)
    assert out.shape == tokens.shape


def test_identity_at_init():
    tt = TemporalTransformer()
    tokens, visibility, frame_indices = _make_inputs()
    out = tt(tokens, visibility, frame_indices)
    assert torch.allclose(out, tokens, atol=1e-6)


def test_gradients_flow_after_perturbing_output_proj():
    # output_proj is zero-initialized, so at exact init d(loss)/d(x) = d(loss)/d(delta)
    # * output_proj.weight == 0 regardless of what's upstream - that would make this
    # test trivially "pass" with all-None/all-zero grads even if the wiring were
    # broken. Nudging output_proj.weight off zero first (untracked, via no_grad) lets
    # the test actually distinguish "wiring works" from "wiring is broken".
    tt = TemporalTransformer()
    with torch.no_grad():
        tt.output_proj.weight.add_(torch.randn_like(tt.output_proj.weight) * 0.01)
    tokens, visibility, frame_indices = _make_inputs()
    tokens.requires_grad_(True)

    out = tt(tokens, visibility, frame_indices)
    out.sum().backward()

    assert tt.component_type_embedding.grad is not None
    assert tt.output_proj.weight.grad is not None
    assert tt.blocks[0].attn.in_proj_weight.grad is not None
    assert tokens.grad is not None


def test_m_n_head_grid_matches_config():
    cfg = TTConfig()
    tt = TemporalTransformer(cfg)
    assert tt.m_slopes.shape == (cfg.num_heads,)
    assert tt.n_slopes.shape == (cfg.num_heads,)
    pairs = set(zip(tt.m_slopes.tolist(), tt.n_slopes.tolist()))
    expected = {(m, n) for m in cfg.m_values for n in cfg.n_values}
    assert pairs == expected


def test_padded_frames_do_not_affect_valid_frame_outputs():
    tt = TemporalTransformer()
    with torch.no_grad():
        tt.output_proj.weight.add_(torch.randn_like(tt.output_proj.weight) * 0.02)

    batch_size, num_frames = 1, 6
    tokens, visibility, frame_indices = _make_inputs(batch_size=batch_size, num_frames=num_frames)
    valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool)
    valid_mask[0, 4:] = False

    out_a = tt(tokens, visibility, frame_indices, valid_mask)

    tokens_b = tokens.clone()
    tokens_b[0, 4:] = torch.randn_like(tokens_b[0, 4:]) * 100
    visibility_b = visibility.clone()
    visibility_b[0, 4:] = 999.0

    out_b = tt(tokens_b, visibility_b, frame_indices, valid_mask)

    assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-4)
    assert not torch.isnan(out_a).any()
    assert not torch.isnan(out_b).any()


def test_window_mean_excludes_padded_frames():
    tt = TemporalTransformer()
    num_components = 4
    visibility = torch.tensor([[0.9, 0.1, 5.0, -5.0]])  # last two are "padding" garbage
    frame_indices = torch.arange(4).unsqueeze(0).float()
    valid_mask = torch.tensor([[True, True, False, False]])

    attn_mask = tt._compute_attn_mask(visibility, frame_indices, valid_mask, num_components)
    # mean over only the 2 valid frames (0.9, 0.1) = 0.5
    expected_mean = 0.5
    # bias(i=0, j=0, h=0) = m_0 * (0.9 - 0.5) - n_0 * |0-0| = m_0 * 0.4
    m0 = tt.m_slopes[0].item()
    bias_00 = attn_mask[0, 0, 0].item()
    assert abs(bias_00 - m0 * (0.9 - expected_mean)) < 1e-4


def test_padded_keys_get_large_negative_bias():
    tt = TemporalTransformer()
    num_components = 4
    visibility = torch.rand(1, 4)
    frame_indices = torch.arange(4).unsqueeze(0).float()
    valid_mask = torch.tensor([[True, True, False, False]])

    attn_mask = tt._compute_attn_mask(visibility, frame_indices, valid_mask, num_components)
    # query frame 0 (token 0), key frame 2 (padded, token index 2*4=8)
    assert attn_mask[0, 0, 8].item() <= -1e8
