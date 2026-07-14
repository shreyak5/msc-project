"""ResNet-50 (Bottleneck) backbone, EMOCA's "emoca_specific" stride/maxpool
variant, used by model/emotion/emotion_net.py as a frozen emotion-feature
extractor (implementation-plan.md Sec 6d-10: emotion loss).

Architecture code adapted from EMOCA (Daniel et al., github.com/radekd91/emoca,
gdl/layers/losses/EmoNetLoss.py's ResNet) via SMIRK's copy
(src/losses/resnet.py, no explicit license header of its own) - EMOCA's own
backbone architecture in turn traces to the ResNet-50 reimplementation in
cydonia999/VGGFace2-pytorch (MIT License), which EMOCA modified with the
"emoca_specific" stride-placement (stride moved from conv1 to conv2 in each
Bottleneck block - a ResNet-v1.5-style change) and maxpool padding change. As
with model/mica/arcface.py: this architecture code traces to a permissively
licensed reimplementation, but the actual pretrained checkpoint we load
(assets/ResNet50/checkpoints/...) is EMOCA/DECA's own, under the Max-Planck
non-commercial research license explicitly stated in SMIRK's copy of
ExpressionLoss.py (Copyright 2019 Max-Planck-Gesellschaft, deca@tue.mpg.de) - use
here is non-commercial academic research, same status as the FLAME (model/flame/
flame.py) and MICA (model/mica/mica.py) assets.

Simplified from SMIRK's resnet.py:
- Only the emoca_specific=True code path is kept (SMIRK's ExpressionLoss always
  constructs it that way) - the alternate stride/maxpool branch is dropped as
  unreachable.
- Only the "include_top=False" path is kept: no `fc` classification head is
  defined at all (SMIRK's copy still instantiates one as dead weight, deleting
  it from the checkpoint before a strict=False load; we just never define it and
  load strict=True against a checkpoint dict pre-filtered to backbone.* keys -
  see model/emotion/emotion_net.py).
- Dropped the random Conv2d/BatchNorm2d weight-init loop and the standalone
  Caffe-pickle load_state_dict() utility (both were only ever used to initialize/
  load the *pre-EMOCA-finetuning* VGGFace2 backbone - irrelevant once loading
  EMOCA's own already-finetuned final checkpoint directly, as we do here).
"""

from __future__ import annotations

import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class EmotionResNet50(nn.Module):
    """conv1/bn1/relu/maxpool -> layer1-4 -> avgpool, EMOCA's "emoca_specific"
    variant (maxpool padding=1; stride on each Bottleneck's conv2, not conv1).
    No classification head - forward() returns the pooled (B, 2048, 1, 1)
    feature map."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)
