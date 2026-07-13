import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import COMPONENT_TOKENS, SViTConfig  # noqa: E402
from model.encoder import SViT  # noqa: E402


def test_svit_output_is_raw_features_not_decoded_params():
    model = SViT()
    images = torch.randn(2, 3, 224, 224)

    out = model(images)

    assert set(out.keys()) == {spec.name for spec in COMPONENT_TOKENS}
    for name in out:
        assert out[name].shape == (2, SViTConfig().embed_dim)


def test_svit_gradients_reach_tokens_and_pos_embed():
    model = SViT()
    images = torch.randn(2, 3, 224, 224)

    out = model(images)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    assert model.component_tokens["shape"].grad is not None
    assert model.pos_embed.grad is not None
