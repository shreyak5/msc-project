import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.encoding import _encode_svit, _fill_missing_frame_tokens, _pool_identity  # noqa: E402


def _tokens(batch_size=1, num_frames=5, num_components=4, dim=3):
    return torch.arange(batch_size * num_frames * num_components * dim, dtype=torch.float32).reshape(
        batch_size, num_frames, num_components, dim
    )


def test_valid_frames_left_unchanged():
    tokens = _tokens()
    real_frame_mask = torch.ones(1, 5, dtype=torch.bool)
    flag_visibility_valid = torch.ones(1, 5, dtype=torch.bool)

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    assert torch.equal(out, tokens)


def test_interior_missing_frame_averages_both_neighbors():
    tokens = _tokens(num_frames=5)
    real_frame_mask = torch.ones(1, 5, dtype=torch.bool)
    flag_visibility_valid = torch.tensor([[True, True, False, True, True]])

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    expected = (tokens[0, 1] + tokens[0, 3]) / 2
    assert torch.allclose(out[0, 2], expected)
    # untouched frames stay exactly as-is
    assert torch.equal(out[0, 0], tokens[0, 0])
    assert torch.equal(out[0, 1], tokens[0, 1])
    assert torch.equal(out[0, 3], tokens[0, 3])
    assert torch.equal(out[0, 4], tokens[0, 4])


def test_leading_missing_frame_uses_one_sided_neighbor():
    tokens = _tokens(num_frames=5)
    real_frame_mask = torch.ones(1, 5, dtype=torch.bool)
    flag_visibility_valid = torch.tensor([[False, True, True, True, True]])

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    assert torch.equal(out[0, 0], tokens[0, 1])


def test_trailing_missing_frame_uses_one_sided_neighbor():
    tokens = _tokens(num_frames=5)
    real_frame_mask = torch.ones(1, 5, dtype=torch.bool)
    flag_visibility_valid = torch.tensor([[True, True, True, True, False]])

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    assert torch.equal(out[0, 4], tokens[0, 3])


def test_multiple_adjacent_missing_frames_reach_past_each_other():
    # Mirrors what synthetic occlusion can produce (two independently-sampled
    # occluded frames landing next to each other) - the nearest-valid-neighbor
    # search must skip over the other invalid frame, not just look at i-1/i+1.
    tokens = _tokens(num_frames=6)
    real_frame_mask = torch.ones(1, 6, dtype=torch.bool)
    flag_visibility_valid = torch.tensor([[True, False, False, True, True, True]])

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    expected = (tokens[0, 0] + tokens[0, 3]) / 2
    assert torch.allclose(out[0, 1], expected)
    assert torch.allclose(out[0, 2], expected)


def test_padded_frame_not_touched_even_if_flagged_invalid():
    tokens = _tokens(num_frames=5)
    real_frame_mask = torch.tensor([[True, True, True, False, False]])
    flag_visibility_valid = torch.tensor([[True, True, True, False, False]])

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    assert torch.equal(out[0, 3], tokens[0, 3])
    assert torch.equal(out[0, 4], tokens[0, 4])


def test_no_valid_frames_at_all_leaves_tokens_unchanged():
    tokens = _tokens(num_frames=3)
    real_frame_mask = torch.ones(1, 3, dtype=torch.bool)
    flag_visibility_valid = torch.zeros(1, 3, dtype=torch.bool)

    out = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)
    assert torch.equal(out, tokens)


# --- _pool_identity (Stage2Config.pass_c_identity_pooling) ---


def test_pool_identity_output_is_constant_across_real_frames():
    torch.manual_seed(0)
    shape = torch.randn(2, 5, 4)
    real_frame_mask = torch.ones(2, 5, dtype=torch.bool)

    out = _pool_identity(shape, real_frame_mask)
    for b in range(2):
        for n in range(1, 5):
            assert torch.allclose(out[b, n], out[b, 0])


def test_pool_identity_matches_manual_mean():
    torch.manual_seed(1)
    shape = torch.randn(1, 4, 3)
    real_frame_mask = torch.ones(1, 4, dtype=torch.bool)

    out = _pool_identity(shape, real_frame_mask)
    expected = shape[0].mean(dim=0)
    for n in range(4):
        assert torch.allclose(out[0, n], expected, atol=1e-6)


def test_pool_identity_excludes_padded_frames_from_the_mean():
    shape = torch.tensor([[[1.0], [3.0], [999.0]]])  # last frame is tail-padding
    real_frame_mask = torch.tensor([[True, True, False]])

    out = _pool_identity(shape, real_frame_mask)
    expected = torch.tensor(2.0)  # mean(1.0, 3.0), padded 999.0 excluded
    assert torch.allclose(out[0, 0], expected)
    assert torch.allclose(out[0, 1], expected)
    # padded position still gets a numerically valid (if meaningless) value,
    # matching every other per-frame field's convention.
    assert torch.allclose(out[0, 2], expected)


def test_pool_identity_never_mixes_across_batch():
    # Each clip in the batch is generally a different subject - pooling must
    # never blend clip 0's identity into clip 1's, or vice versa.
    shape = torch.zeros(2, 3, 2)
    shape[0] = torch.tensor([[1.0, 1.0]] * 3)
    shape[1] = torch.tensor([[5.0, 5.0]] * 3)
    real_frame_mask = torch.ones(2, 3, dtype=torch.bool)

    out = _pool_identity(shape, real_frame_mask)
    assert torch.allclose(out[0], torch.full((3, 2), 1.0))
    assert torch.allclose(out[1], torch.full((3, 2), 5.0))


def _fake_svit(flat_pixel_values):
    # Deterministic per-frame function of the input (not a constant), so
    # chunking equivalence is actually exercised rather than trivially true.
    per_frame = flat_pixel_values.flatten(start_dim=1).sum(dim=1, keepdim=True)
    return {"shape": per_frame, "expr": per_frame * 2.0}


def test_encode_svit_no_chunking_matches_direct_call():
    flat_pixel_values = torch.arange(5 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 3, 2, 2)
    out = _encode_svit(_fake_svit, flat_pixel_values, chunk_size=None)
    direct = _fake_svit(flat_pixel_values)
    for name in direct:
        assert torch.allclose(out[name], direct[name])


def test_encode_svit_chunked_matches_unchunked():
    flat_pixel_values = torch.arange(7 * 3 * 2 * 2, dtype=torch.float32).reshape(7, 3, 2, 2)
    unchunked = _encode_svit(_fake_svit, flat_pixel_values, chunk_size=None)
    for chunk_size in (1, 2, 3, 7, 100):
        chunked = _encode_svit(_fake_svit, flat_pixel_values, chunk_size=chunk_size)
        for name in unchunked:
            assert torch.allclose(chunked[name], unchunked[name])


def test_encode_image_chunked_matches_unchunked():
    from model import constants
    from model.encoding import encode_image

    expression_dim = 10

    def fake_svit_image(pixel_values):
        per_frame = pixel_values.flatten(start_dim=1).sum(dim=1, keepdim=True)
        return {
            "shape": per_frame,
            "expression": per_frame.expand(-1, expression_dim + 2),
            "jaw": per_frame,
            "camera": per_frame.expand(-1, constants.CAMERA_TOKEN_DIM),
        }

    class FakeHeads:
        def __init__(self):
            self.expression_dim = expression_dim

        def __call__(self, features):
            return features

    pixel_values = torch.arange(6 * 3 * 2 * 2, dtype=torch.float32).reshape(6, 3, 2, 2)
    unchunked = encode_image(fake_svit_image, FakeHeads(), pixel_values, svit_chunk_size=None)
    chunked = encode_image(fake_svit_image, FakeHeads(), pixel_values, svit_chunk_size=4)
    for name in unchunked:
        assert torch.allclose(chunked[name], unchunked[name])
