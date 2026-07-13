import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import UNetConfig  # noqa: E402
from model.generator import UNetGenerator  # noqa: E402


def test_unet_matches_smirk_trainer_config():
    cfg = UNetConfig()
    assert cfg.in_channels == 6
    assert cfg.out_channels == 3
    assert cfg.init_features == 32
    assert cfg.res_blocks == 5


def test_unet_output_shape_and_range():
    cfg = UNetConfig()
    gen = UNetGenerator(cfg)
    x = torch.randn(2, cfg.in_channels, 224, 224)

    out = gen(x)

    assert out.shape == (2, cfg.out_channels, 224, 224)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_unet_gradients_flow_through_encoder_bottleneck_and_decoder():
    gen = UNetGenerator()
    x = torch.randn(2, gen.config.in_channels, 224, 224)

    out = gen(x)
    out.sum().backward()

    assert gen.encoder1[0].weight.grad is not None
    assert gen.resnet_blocks[0].conv_block[1].weight.grad is not None
    assert gen.decoder1[0].weight.grad is not None
    assert gen.conv.weight.grad is not None
