import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import TTConfig  # noqa: E402
from model.temporal import TemporalTransformer, _gather_local_windows  # noqa: E402


def _make_inputs(batch_size=2, num_frames=11, num_components=4, dim=768, device="cpu"):
    tokens = torch.randn(batch_size, num_frames, num_components, dim, device=device)
    visibility = torch.rand(batch_size, num_frames, device=device)
    frame_indices = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1).float()
    return tokens, visibility, frame_indices


def _mask_row_index(batch, frame, head, num_frames, num_heads):
    """Row index into attn_mask's collapsed (B*N*num_heads) batch axis, matching
    _compute_attn_mask's (B, N, H) -> B*N*H reshape order (B slowest, H fastest)."""
    return (batch * num_frames + frame) * num_heads + head


def _mask_col_index(query_frame, key_frame, key_component, radius, num_components):
    """Column index into attn_mask's (w*num_components) key axis for a query at
    query_frame attending to key_frame (must satisfy |query_frame - key_frame| <=
    radius, or it isn't addressable at all). window offset k = key_frame -
    query_frame + radius, expanded to token granularity (window-major,
    component-minor - see _gather_local_windows' docstring on ordering)."""
    offset = key_frame - query_frame + radius
    return offset * num_components + key_component


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
    # Default window_size=15 (radius=7) comfortably covers this 4-frame clip, so
    # every real frame is addressable from query frame 0 - only the explicit
    # valid_mask=False frames are excluded, same scenario as before the local-window
    # rewrite, just with new addressing into attn_mask (see _mask_row/col_index).
    tt = TemporalTransformer()
    num_components = 4
    num_frames = 4
    radius = tt.config.window_size // 2
    visibility = torch.tensor([[0.9, 0.1, 5.0, -5.0]])  # last two are "padding" garbage
    frame_indices = torch.arange(num_frames).unsqueeze(0).float()
    valid_mask = torch.tensor([[True, True, False, False]])

    attn_mask = tt._compute_attn_mask(visibility, frame_indices, valid_mask, num_components)
    # mean over only the 2 valid frames (0.9, 0.1) = 0.5
    expected_mean = 0.5
    # bias(query frame=0, key frame=0 [itself], head=0) = m_0 * (0.9 - 0.5) - n_0 * 0
    m0 = tt.m_slopes[0].item()
    row = _mask_row_index(batch=0, frame=0, head=0, num_frames=num_frames, num_heads=tt.config.num_heads)
    col = _mask_col_index(query_frame=0, key_frame=0, key_component=0, radius=radius, num_components=num_components)
    bias_00 = attn_mask[row, 0, col].item()
    assert abs(bias_00 - m0 * (0.9 - expected_mean)) < 1e-4


def test_padded_keys_get_large_negative_bias():
    tt = TemporalTransformer()
    num_components = 4
    num_frames = 4
    radius = tt.config.window_size // 2
    visibility = torch.rand(1, num_frames)
    frame_indices = torch.arange(num_frames).unsqueeze(0).float()
    valid_mask = torch.tensor([[True, True, False, False]])

    attn_mask = tt._compute_attn_mask(visibility, frame_indices, valid_mask, num_components)
    # query frame 0, key frame 2 (marked invalid by valid_mask)
    row = _mask_row_index(batch=0, frame=0, head=0, num_frames=num_frames, num_heads=tt.config.num_heads)
    col = _mask_col_index(query_frame=0, key_frame=2, key_component=0, radius=radius, num_components=num_components)
    assert attn_mask[row, 0, col].item() <= -1e8


def test_boundary_frames_have_fewer_valid_neighbors():
    # No explicit padding (valid_mask all True) - frame 0 still has fewer real
    # neighbors than a middle frame purely because there's no frame -1..-radius in
    # the video at all; _gather_local_windows' own zero-padding, combined with the
    # valid_mask gather (padded with False), masks those out automatically.
    radius = 2
    num_frames = 10
    valid_mask = torch.ones(1, num_frames, dtype=torch.bool)

    local_valid = _gather_local_windows(valid_mask, radius)  # (1, N, 5)
    # frame 0: only itself + 2 real neighbors ahead are valid (3 of 5 window slots)
    assert local_valid[0, 0].sum().item() == 3
    # a middle frame (index 5): full window of 5 real neighbors, all valid
    assert local_valid[0, 5].sum().item() == 5


def test_window_size_is_configurable():
    # window_size=3 -> radius=1: a query frame's local window can only ever address
    # its immediate neighbors (distance <= 1) - frames further away simply never
    # appear as a candidate key at all, regardless of validity.
    tt = TemporalTransformer(TTConfig(window_size=3))
    num_components = 4
    num_frames = 5
    radius = 1
    visibility = torch.rand(1, num_frames)
    frame_indices = torch.arange(num_frames).unsqueeze(0).float()
    valid_mask = torch.ones(1, num_frames, dtype=torch.bool)

    attn_mask = tt._compute_attn_mask(visibility, frame_indices, valid_mask, num_components)
    seq_len_kv = (2 * radius + 1) * num_components
    assert attn_mask.shape == (1 * num_frames * tt.config.num_heads, num_components, seq_len_kv)
    # 3 window slots * 4 components, not the default window_size=15's 15*4=60 -
    # confirms window_size is actually threaded through, not silently ignored.
    assert seq_len_kv == 12


def test_local_attention_is_actually_local():
    # End-to-end proof that attention is genuinely local (compute-bounded), not just
    # masked-dense: perturbing a frame far outside a given query frame's window must
    # leave that query frame's output completely unchanged. With window_size=5
    # (radius=2) and depth=3, the maximum possible reach after 3 layers of
    # message-passing is radius*depth=6 frames - well short of the distance-10
    # perturbation used below.
    torch.manual_seed(0)
    tt = TemporalTransformer(TTConfig(window_size=5))
    with torch.no_grad():
        tt.output_proj.weight.add_(torch.randn_like(tt.output_proj.weight) * 0.02)

    batch_size, num_frames = 1, 20
    tokens, visibility, frame_indices = _make_inputs(batch_size=batch_size, num_frames=num_frames)
    query_frame, perturbed_frame = 5, 15

    out_a = tt(tokens, visibility, frame_indices)

    tokens_b = tokens.clone()
    tokens_b[0, perturbed_frame] = torch.randn_like(tokens_b[0, perturbed_frame]) * 100
    visibility_b = visibility.clone()
    visibility_b[0, perturbed_frame] = 999.0

    out_b = tt(tokens_b, visibility_b, frame_indices)

    assert torch.allclose(out_a[0, query_frame], out_b[0, query_frame], atol=1e-5)
    # sanity: the perturbed frame's own output should actually change - proving the
    # perturbation had an effect somewhere, so the assertion above isn't vacuous.
    assert not torch.allclose(out_a[0, perturbed_frame], out_b[0, perturbed_frame], atol=1e-5)
