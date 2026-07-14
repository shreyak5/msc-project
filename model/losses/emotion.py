"""Emotion loss (implementation-plan.md Sec 6: "Emotion | L2 between pretrained
emotion-net features of I' and I; UNet frozen for this loss (only the expression
pathway updates)").

Adapted from SMIRK's ExpressionLoss.forward (src/losses/ExpressionLoss.py, "Code
borrowed from EMOCA"; MPG non-commercial research license - see model/emotion/
emotion_net.py's docstring for the full attribution/license chain) as a standalone
function consuming a separately-constructed model.emotion.emotion_net.EmotionNet
instance, matching this project's model/ vs model/losses/ split (same pattern as
model/mica/mica.py + model/losses/mica_shape.py).

Only the 'l2', use_mean=True path of SMIRK's forward() is implemented - the
plan's spec is exactly "L2 ... features", and nothing else in this project uses
the 'l1'/'cos' metrics SMIRK's version also supports.

`target`'s pass through emotion_net is under no_grad (it's real photo data - a
fixed comparison target, same role as .detach() on the target side of every other
loss in this project), but `reconstructed`'s pass is NOT: EmotionNet's own weights
are frozen either way (requires_grad=False, set in EmotionNet.__init__), but this
loss's whole purpose (per the plan: "only the expression pathway updates") is to
backprop a gradient signal THROUGH the frozen network and into whatever produced
`reconstructed` (the UNet output, and transitively the expression encoder
upstream of it) - wrapping that branch in no_grad too would sever that path
entirely and make the loss a no-op for training. "UNet frozen for this loss" is a
separate, training-loop-level concern (which optimizer/parameter group actually
receives this gradient) - not something enforced here.
"""

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
