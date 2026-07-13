import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import COMPONENT_TOKENS  # noqa: E402
from model.encoder import SViT  # noqa: E402


def test_svit_output_shapes_and_eyelid_range():
    model = SViT()
    images = torch.randn(2, 3, 224, 224)

    out = model(images)

    assert set(out.keys()) == {spec.name for spec in COMPONENT_TOKENS}
    for spec in COMPONENT_TOKENS:
        assert out[spec.name].shape == (2, spec.param_dim)

    eyelids = out["expression"][:, -2:]
    assert torch.all(eyelids >= 0) and torch.all(eyelids <= 1)


def test_svit_gradients_reach_tokens_and_pos_embed():
    model = SViT()
    images = torch.randn(2, 3, 224, 224)

    out = model(images)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    assert model.component_tokens["shape"].grad is not None
    assert model.pos_embed.grad is not None
