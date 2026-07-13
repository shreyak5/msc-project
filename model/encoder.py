"""Spatial ViT (SViT) encoder: TokenFace-style (Zhang et al., ICCV 2023) tokenized
ViT-B/16 with learnable per-parameter-group component tokens appended to the
image patch tokens (implementation-plan.md Sec 2.1-2.3).

Some transformer-block internals below are adapted from FaRL-B's ViT-B/16
(Zheng et al., CVPR 2022, https://github.com/FacePerceiver/FaRL, MIT License,
Copyright (c) Microsoft Corporation) - specifically farl/network/farl/model.py -
rather than a generic timm ViT, so that the official FaRL checkpoint loads with
an exact 1:1 key/shape mapping (see model/farl_weights.py). Each adapted class
is marked below; this file is not vendored/imported from that repo.

Device/parallelism note: this module is plain nn.Module code with no device or
distributed-training assumptions baked in - DistributedDataParallel wrapping and
device placement happen in the training scripts (Sec 7), not here.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from timm.layers import PatchEmbed

from model.config import COMPONENT_TOKENS, SViTConfig


class LayerNormFp32(nn.Module):
    """Adapted from FaRL's LayerNorm (farl/network/farl/model.py, MIT License,
    https://github.com/FacePerceiver/FaRL): always normalizes in float32, matching
    FaRL/CLIP's implementation, so mixed-precision (fp16/bf16) training doesn't
    destabilize normalization statistics."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return (self.weight * x.to(orig_dtype)) + self.bias


class QuickGELU(nn.Module):
    """Adapted from FaRL's QuickGELU (farl/network/farl/model.py, MIT License,
    https://github.com/FacePerceiver/FaRL): x * sigmoid(1.702 * x), FaRL/CLIP's
    GELU approximation, used throughout its MLPs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Adapted from FaRL's ResidualAttentionBlock (farl/network/farl/model.py, MIT
    License, https://github.com/FacePerceiver/FaRL): pre-norm transformer block with
    combined-qkv self-attention (nn.MultiheadAttention) + QuickGELU MLP + fp32
    LayerNorm. Drop-path/attn-mask support (present in FaRL's version) is omitted -
    not used by this encoder."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_1 = LayerNormFp32(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(dim, hidden_dim)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(hidden_dim, dim)),
                ]
            )
        )
        self.ln_2 = LayerNormFp32(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention: query, key, value are all the same normalized tensor.
        normed = self.ln_1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class SViT(nn.Module):
    """ViT-B/16 spatial encoder with 4 learnable component tokens appended to the
    patch token sequence. Only the component tokens' post-final-layer features are
    returned; image patch tokens are discarded (TokenFace design, Sec 2.1).

    Returns raw 768-dim features, not decoded FLAME/camera parameters: the per-token
    MLP heads (model.heads.ComponentHeads) are a separate module, shared by both the
    single-image path (SViT -> Heads) and the video path (SViT -> TT -> Heads, Sec 3),
    so they aren't owned by SViT itself."""

    def __init__(self, config: SViTConfig | None = None):
        super().__init__()
        self.config = config or SViTConfig()
        cfg = self.config

        # bias=False matches FaRL/CLIP's conv1, which has no bias term.
        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=3,
            embed_dim=cfg.embed_dim,
            bias=False,
        )
        num_patches = self.patch_embed.num_patches

        # Learned absolute position embeddings over patch tokens only (initialized from
        # FaRL's patch position embeddings, trainable from there - see farl_weights.py).
        # Component tokens receive no positional embedding: they are order-independent,
        # register-style tokens, not spatial locations.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.embed_dim))

        # Applied to patch tokens only, before concatenation with component tokens -
        # mirrors FaRL's own data flow (ln_pre normalizes the image signal it was
        # pretrained on); component tokens are a TokenFace addition FaRL never saw.
        self.ln_pre = LayerNormFp32(cfg.embed_dim)

        self.component_names: list[str] = [t.name for t in COMPONENT_TOKENS]
        self.component_tokens = nn.ParameterDict(
            {t.name: nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) for t in COMPONENT_TOKENS}
        )

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = LayerNormFp32(cfg.embed_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images: (B, 3, H, W) -> dict[component_name] -> (B, embed_dim) token feature,
        i.e. the component tokens after the final transformer layer, pre-MLP-head."""
        batch_size = images.shape[0]
        patch_tokens = self.patch_embed(images)
        patch_tokens = self.ln_pre(patch_tokens + self.pos_embed)

        component_tokens = torch.cat(
            [self.component_tokens[name].expand(batch_size, -1, -1) for name in self.component_names],
            dim=1,
        )

        x = torch.cat([patch_tokens, component_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        num_component = len(self.component_names)
        component_out = x[:, -num_component:, :]
        return {name: component_out[:, i, :] for i, name in enumerate(self.component_names)}
