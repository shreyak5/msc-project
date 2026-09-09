"""Adapted from SMIRK's SmirkGenerator/ResnetBlock (github.com/georgeretsi/smirk,
MIT License, Copyright (c) 2024 George Retsinas)."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from model.config import UNetConfig


class ResnetBlock(nn.Module):
    """Adapted from SMIRK's ResnetBlock (src/smirk_generator.py): a conv block with
    a skip connection (He et al., https://arxiv.org/abs/1512.03385)."""

    def __init__(self, dim: int, use_dropout: bool = False, use_bias: bool = False):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, use_dropout, use_bias)

    @staticmethod
    def _build_conv_block(dim: int, use_dropout: bool, use_bias: bool) -> nn.Sequential:
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.BatchNorm2d(dim),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class UNetGenerator(nn.Module):
    """Adapted from SMIRK's SmirkGenerator (src/smirk_generator.py): 4-level UNet
    encoder/decoder with skip connections + a ResNet-block bottleneck."""

    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        self.config = config or UNetConfig()
        cfg = self.config
        features = cfg.init_features

        self.encoder1 = self._block(cfg.in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.resnet_blocks = nn.ModuleList(
            [ResnetBlock(features * 16) for _ in range(cfg.res_blocks)]
        )

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, cfg.out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
                    ),
                    (name + "norm1", nn.BatchNorm2d(features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    ),
                    (name + "norm2", nn.BatchNorm2d(features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) - rendered mesh geometry concatenated with the
        sparsely-masked real image (Sec 2.5) -> (B, out_channels, H, W) in [0, 1]."""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        for resnet_block in self.resnet_blocks:
            bottleneck = resnet_block(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
