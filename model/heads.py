"""Per-token MLP heads mapping 768-dim component-token features to FLAME/camera
parameters (implementation-plan.md Sec 2.2, 2.3).

Kept separate from model.encoder.SViT because the same heads are reused by two
different flows (Sec 3):
  - single image:  SViT -> ComponentHeads
  - video/clip:    SViT -> TemporalTransformer -> ComponentHeads
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import constants
from model.config import COMPONENT_TOKENS


class ComponentHead(nn.Module):
    """Single-layer MLP mapping a component token's 768-dim feature to its FLAME/camera
    parameter group. For the expression head, the trailing NUM_EYELID_PARAMS outputs are
    passed through a sigmoid (eyelid blendshapes are in [0, 1]; sigmoid keeps gradients
    alive at the boundary, unlike a hard clamp - implementation-plan.md Sec 2.2)."""

    def __init__(self, embed_dim: int, param_dim: int, name: str):
        super().__init__()
        self.name = name
        self.linear = nn.Linear(embed_dim, param_dim)
        if name == "expression":
            self._eyelid_start = param_dim - constants.NUM_EYELID_PARAMS
        else:
            self._eyelid_start = None

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        out = self.linear(token)
        if self._eyelid_start is not None:
            expr = out[..., : self._eyelid_start]
            eyelids = torch.sigmoid(out[..., self._eyelid_start :])
            out = torch.cat([expr, eyelids], dim=-1)
        return out


class ComponentHeads(nn.Module):
    """One ComponentHead per component token (Sec 2.3), applied to a features dict
    keyed the same way as model.encoder.SViT's output (and, for video, whatever the
    caller reassembles from TemporalTransformer's stacked output into that same
    dict form - see model.config.COMPONENT_TOKENS for the canonical ordering)."""

    def __init__(self, embed_dim: int = constants.SVIT_EMBED_DIM):
        super().__init__()
        self.heads = nn.ModuleDict(
            {t.name: ComponentHead(embed_dim, t.param_dim, t.name) for t in COMPONENT_TOKENS}
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """features: dict[component_name] -> (..., embed_dim) -> dict[component_name] ->
        (..., param_dim) decoded FLAME/camera parameters."""
        return {name: self.heads[name](feat) for name, feat in features.items()}
