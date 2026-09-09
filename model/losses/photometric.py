"""VGGPerceptualLoss adapted from SMIRK (github.com/georgeretsi/smirk, MIT
License, Copyright (c) 2024 George Retsinas), which wraps torchvision's
pretrained VGG16."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model import constants


def photometric_loss(
    reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (reconstructed - target).abs()
    if mask is None:
        return diff.mean()
    denom = mask.sum().clamp(min=1) * reconstructed.shape[1]
    return (diff * mask).sum() / denom


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG16 features of two images, summed over 4 intermediate
    blocks (layers 0-4, 4-9, 9-16, 16-23 of torchvision's vgg16.features)."""

    def __init__(self):
        super().__init__()
        vgg_features = torchvision.models.vgg16(weights="DEFAULT").features
        block_bounds = [(0, 4), (4, 9), (9, 16), (16, 23)]
        blocks = [vgg_features[start:end].eval() for start, end in block_bounds]
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.blocks = nn.ModuleList(blocks)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """reconstructed, target: (B, 3, H, W) in [0, 1] -> scalar summed L1 feature loss."""
        x = (reconstructed - self.mean) / self.std
        y = (target - self.mean) / self.std

        target_size = (constants.SVIT_IMG_SIZE, constants.SVIT_IMG_SIZE)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, mode="bilinear", size=target_size, align_corners=False)
            y = F.interpolate(y, mode="bilinear", size=target_size, align_corners=False)

        perceptual_loss = torch.zeros((), device=x.device)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            perceptual_loss = perceptual_loss + F.l1_loss(x, y)
        return perceptual_loss
