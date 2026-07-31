"""Photometric + VGG perceptual losses (implementation-plan.md Sec 6, "2D recon
pass"): L1(I', I) and L1 of VGG features of I' vs I, where I is the real photo and
I' is the UNet's reconstruction.

VGGPerceptualLoss is adapted from SMIRK (Retsinas et al., CVPR 2024,
https://github.com/georgeretsi/smirk, src/losses/VGGPerceptualLoss.py, MIT License,
Copyright (c) 2024 George Retsinas), which itself wraps torchvision's ImageNet-
pretrained VGG16 (standard, auto-downloaded torchvision model weights - not a novel
asset requiring separate sourcing, unlike FaRL/MICA/the emotion net).

Deviations from SMIRK:
- SMIRK's forward() starts with `x = x * 0.5 + 0.5`, converting an assumed
  [-1, 1]-range input to [0, 1] before ImageNet normalization ((x - mean) / std).
  This project's images are already in [0, 1] throughout (dataset_processing's
  _crop_to_tensor divides by 255; model/generator.py's UNet output uses sigmoid) -
  so that first conversion step is dropped; the ImageNet normalization itself is
  unchanged (mean/std are VGG's actual expected preprocessing statistics either way).
- SMIRK unconditionally resizes to 224x224 before running VGG, even when the input
  already is that size (our case, given SVIT_IMG_SIZE/RENDERER_IMAGE_SIZE=224) -
  only resize here if the input size actually differs.
- Kept as-is (not averaged) despite only summing over 4 blocks: the plan's starting
  loss weight ("VGG 10", Sec 6) is calibrated against SMIRK's summed loss - averaging
  would silently quarter the effective loss magnitude and break that anchor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model import constants


def photometric_loss(
    reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """reconstructed, target: (B, 3, H, W) in [0, 1] -> scalar L1 loss.

    mask: (B, 1, H, W), optional - restricts the loss to the masked-in (face)
    region. Without it, the mean is dragged down by the background, which the
    UNet reconstructs almost for free (it's handed through unmasked in
    masking()'s own input) - that dilutes the gradient signal on the face
    region, the part the network actually has to learn to fill in from the
    rendered mesh + sparse pixel hints, and was observed to leave the face
    region a flat, textureless blur despite an otherwise-healthy background
    reconstruction."""
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
