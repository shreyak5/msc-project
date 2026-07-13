import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.encoder import SViT  # noqa: E402
from model.farl_weights import _interpolate_pos_embed, load_farl_pretrained  # noqa: E402

_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained_weights",
    "farl",
    "FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
)

_EXPECTED_UNINITIALIZED_KEYS = {
    "component_tokens.shape",
    "component_tokens.expression",
    "component_tokens.jaw",
    "component_tokens.camera",
    "heads.shape.linear.weight",
    "heads.shape.linear.bias",
    "heads.expression.linear.weight",
    "heads.expression.linear.bias",
    "heads.jaw.linear.weight",
    "heads.jaw.linear.bias",
    "heads.camera.linear.weight",
    "heads.camera.linear.bias",
}


@pytest.mark.skipif(not os.path.exists(_CHECKPOINT), reason="FaRL checkpoint not downloaded")
def test_load_farl_pretrained_changes_weights_and_still_runs():
    model = SViT()
    before_block_weight = model.blocks[0].attn.in_proj_weight.clone()
    before_pos_embed = model.pos_embed.clone()

    load_farl_pretrained(model, _CHECKPOINT)

    assert not torch.allclose(before_block_weight, model.blocks[0].attn.in_proj_weight)
    assert not torch.allclose(before_pos_embed, model.pos_embed)

    images = torch.randn(2, 3, 224, 224)
    out = model(images)
    loss = sum(v.sum() for v in out.values())
    loss.backward()


@pytest.mark.skipif(not os.path.exists(_CHECKPOINT), reason="FaRL checkpoint not downloaded")
def test_load_farl_pretrained_only_leaves_component_tokens_and_heads_uninitialized():
    model = SViT()

    result = load_farl_pretrained(model, _CHECKPOINT)

    assert result.unexpected_keys == []
    assert set(result.missing_keys) == _EXPECTED_UNINITIALIZED_KEYS


def test_interpolate_pos_embed_is_noop_when_grids_match():
    pos_embed = torch.randn(196, 768)
    out = _interpolate_pos_embed(pos_embed, (14, 14), (14, 14))
    assert torch.equal(pos_embed, out)


def test_interpolate_pos_embed_resamples_to_new_grid_size():
    pos_embed = torch.randn(196, 768)
    out = _interpolate_pos_embed(pos_embed, (14, 14), (16, 16))
    assert out.shape == (256, 768)
