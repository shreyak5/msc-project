"""Architecture adapted from insightface (deepinsight/insightface, MIT
License, Copyright (c) 2018 Jiankang Deng and Jia Guo) via SMIRK's copy
(github.com/georgeretsi/smirk). The loaded weights (MICA's own) are under a
separate Max-Planck non-commercial research license."""

from __future__ import annotations

import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """IResNet-100 when layers=[3, 13, 30, 3] (Arcface below)."""

    fc_scale = 7 * 7

    def __init__(self, layers: tuple[int, int, int, int], num_features: int = 512, dropout: float = 0.0):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        # Frozen regardless of checkpoint contents (a behavioral flag, not a weight
        # value - unlike the rest of the module's init, this doesn't get overwritten
        # by the state_dict load in mica.py, so it's kept even though the random
        # weight-init calls insightface's own version does here are dropped as dead
        # code (immediately overwritten by that same strict=True load).
        self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)


class Arcface(IResNet):
    """IResNet-100 (layers=[3, 13, 30, 3], insightface's standard "r100" config).
    conv1/bn1/prelu/layer1-3 are frozen (requires_grad=False) even during MICA's
    own training - only layer4/bn2/fc/features were ever fine-tuned; weights for
    all of it (frozen and trainable parts alike) are loaded from assets/mica.tar,
    not from insightface's own pretrained release."""

    def __init__(self):
        super().__init__(layers=(3, 13, 30, 3))
        for layer in (self.conv1, self.bn1, self.prelu, self.layer1, self.layer2, self.layer3):
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 112, 112) -> (B, 512) unnormalized face-recognition embedding."""
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x
