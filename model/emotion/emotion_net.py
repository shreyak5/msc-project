from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model import constants
from model.emotion.resnet import EmotionResNet50

_REPO_ROOT = Path(__file__).resolve().parents[2]


class EmotionNet(nn.Module):
    def __init__(self, checkpoint_path: str | Path = _REPO_ROOT / constants.EMOTION_CHECKPOINT_PATH):
        super().__init__()
        self.backbone = EmotionResNet50()

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone_state_dict = {
            key.removeprefix("backbone."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("backbone.") and not key.startswith("backbone.fc.")
        }
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) in [0, 1] -> (B, 2048) emotion feature vector."""
        assert images.shape[-2:] == (constants.EMOTION_IMAGE_SIZE, constants.EMOTION_IMAGE_SIZE), (
            f"EmotionNet expects {constants.EMOTION_IMAGE_SIZE}x{constants.EMOTION_IMAGE_SIZE} input, "
            f"got {tuple(images.shape[-2:])}"
        )
        return self.backbone(images).view(images.shape[0], -1)
