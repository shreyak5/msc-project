import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import COMPONENT_TOKENS  # noqa: E402
from model.encoder import SViT  # noqa: E402
from model.heads import ComponentHeads  # noqa: E402


def test_heads_output_shapes_and_eyelid_range():
    heads = ComponentHeads()
    features = {spec.name: torch.randn(2, 768) for spec in COMPONENT_TOKENS}

    out = heads(features)

    assert set(out.keys()) == {spec.name for spec in COMPONENT_TOKENS}
    for spec in COMPONENT_TOKENS:
        assert out[spec.name].shape == (2, spec.param_dim)

    eyelids = out["expression"][:, -2:]
    assert torch.all(eyelids >= 0) and torch.all(eyelids <= 1)


def test_heads_gradients_flow():
    heads = ComponentHeads()
    features = {spec.name: torch.randn(2, 768, requires_grad=True) for spec in COMPONENT_TOKENS}

    out = heads(features)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    for feat in features.values():
        assert feat.grad is not None


def test_svit_then_heads_image_path():
    """SViT -> ComponentHeads: the single-image flow (Sec 3)."""
    svit = SViT()
    heads = ComponentHeads()
    images = torch.randn(2, 3, 224, 224)

    features = svit(images)
    params = heads(features)

    assert set(params.keys()) == {spec.name for spec in COMPONENT_TOKENS}
    for spec in COMPONENT_TOKENS:
        assert params[spec.name].shape == (2, spec.param_dim)
