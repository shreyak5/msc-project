from __future__ import annotations

import torch
import torch.nn.functional as F

from model.emotion.emotion_net import EmotionNet


def emotion_loss(reconstructed: torch.Tensor, target: torch.Tensor, emotion_net: EmotionNet) -> torch.Tensor:
    """reconstructed, target: (B, 3, 224, 224) in [0, 1] - the UNet's
    reconstruction I' and the real photo I. emotion_net: a constructed
    (checkpoint-loaded) EmotionNet instance - held/reused by the caller across
    steps, not constructed per call. -> scalar mean-squared feature distance."""
    reconstructed_features = emotion_net(reconstructed)
    with torch.no_grad():
        target_features = emotion_net(target).detach()
    return F.mse_loss(reconstructed_features, target_features)
