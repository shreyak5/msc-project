"""EMOCA's emotion-recognition network (implementation-plan.md Sec 6d-10: emotion
loss; Sec 9: "Reuse from SMIRK repo: ... emotion network").

Adapted from SMIRK (Retsinas et al., CVPR 2024, https://github.com/georgeretsi/smirk,
src/losses/ExpressionLoss.py; that file's own header: "Code borrowed from EMOCA
https://github.com/radekd91/emoca", and explicitly Copyright 2019 Max-Planck-
Gesellschaft (MPG), non-commercial research use only, deca@tue.mpg.de) - see
model/emotion/resnet.py's docstring for the backbone architecture's own license
chain. The checkpoint (assets/ResNet50/checkpoints/deca-epoch=01-val_loss_total/
dataloader_idx_0=1.27607644.ckpt) is MPG's own, same non-commercial-research
status as the FLAME (model/flame/flame.py) and MICA (model/mica/mica.py) assets.

Used purely as a frozen feature extractor to compute L2 emotion-feature distance
between the UNet's reconstruction and the real photo (Sec 6: "Emotion | L2 between
pretrained emotion-net features of I' and I"), never itself trained.

Deviations from SMIRK's ExpressionLoss:
- Only the backbone (feature extractor) is kept - no classification head at all
  (see model/emotion/resnet.py's docstring), so checkpoint loading here filters
  the raw .ckpt's `state_dict` down to `backbone.*` keys (stripping that prefix)
  and drops `backbone.fc.*`/`linear.*` entirely, then loads strict=True - cleaner
  than SMIRK's own delete-then-strict=False approach, since our module simply
  doesn't define those unused parameters to begin with.
- forward() returns the flattened (B, 2048) feature vector directly (SMIRK's
  ExpressionLoss.forward() calls the backbone then flattens with .view() at the
  call site every time - moved into this module since nothing else consumes the
  unflattened shape).
- Only the 'l2' metric is kept (SMIRK's ExpressionLoss also supports 'l1'/'cos'
  and a use_mean toggle; the plan's own spec is exactly "L2 between pretrained
  emotion-net features of I' and I", so the other options aren't used anywhere in
  this project) - see model/losses/emotion.py.
"""

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
