"""Temporal Transformer (TT): refines per-frame component tokens across a temporal
window using a fixed (non-learned) visibility-score + distance attention bias in
place of positional embeddings (implementation-plan.md Sec 2.4, Sec 4).

This module is agnostic to how many frames it is given per call (N) - the typical
case is N == the configured window size w (TT_WINDOW_SIZE), but a window near a
video's start/end has fewer real frames (Sec 4.2). Since a batch requires uniform
N across items, shorter windows are expected to be padded (by the windowing/
data-loading logic, Sec 5, not yet built) up to a common N, with `valid_mask`
marking which frame slots are real. Padded frames are masked out as attention
*keys* only (a large negative bias, so they get ~zero attention weight after
softmax) - not as queries: masking both would give a padded query row where every
key is -inf, and softmax over an all -inf row is 0/0 = NaN. Padded-query output
rows are still numerically valid, just meaningless; callers discard them since
they don't correspond to a real frame. Padded frames are also excluded from the
Sec 4.3 window-mean used for score normalization (see _compute_attn_mask).

The caller also supplies each frame's actual temporal index (used only for
pairwise distance). Passing explicit frame_indices rather than assuming a
contiguous arange(N) only actually matters once frames can be dropped/skipped
within a window (distance between two adjacent entries would then be >1 real
timestep) - out of scope for now per Sec 7, but a free thing to support today.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from model import constants
from model.config import TTConfig

_MASK_VALUE = -1e9


class TTBlock(nn.Module):
    """Pre-norm transformer block, trained from scratch (no FaRL-fidelity
    constraint here, unlike model.encoder.SViT) - standard LayerNorm/GELU.
    Two residual connections per block (attention, then MLP) - the usual
    transformer-block pattern, same as model.encoder.ResidualAttentionBlock."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask (float) is added to the pre-softmax QK^T/sqrt(d) logits by
        # nn.MultiheadAttention itself - required for both the ALiBi-style bias and
        # the padding mask below to work (see model/temporal.py module docstring).
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """Sec 2.4: 3-layer, 8-head transformer over the 4 component tokens x N frames
    in a window. Residual delta design (Sec 2.4): the final output projection is
    zero-initialized, so at init TT(tokens) == tokens exactly (identity map); TT
    refines tokens, it does not replace them. This is why forward() returns
    `residual + delta` rather than `delta` alone - only one layer (output_proj) is
    zero-initialized (not every internal block), so identity-at-init depends on
    this outer residual to carry the original tokens through untouched until TT
    has learned something worth adding.

    Component-type embeddings (Sec 4.4): TT has no positional embeddings at all -
    the attention bias (Sec 4.4) varies only by frame index, not by which of the 4
    tokens within a frame is being attended to, so nothing else in the attention
    mechanism distinguishes "this is the jaw token" from "this is the shape token".
    A dedicated learned embedding per component type gives that identity signal
    explicitly, independent of whatever content each SViT head happens to produce."""

    def __init__(self, config: TTConfig | None = None):
        super().__init__()
        self.config = config or TTConfig()
        cfg = self.config

        num_components = constants.NUM_COMPONENT_TOKENS
        self.component_type_embedding = nn.Parameter(torch.empty(num_components, cfg.dim))
        nn.init.trunc_normal_(self.component_type_embedding, std=0.02)

        # Sec 4.4: heads span a grid of (m, n) combinations, not a single m=k*n line.
        m_n_pairs = list(itertools.product(cfg.m_values, cfg.n_values))
        assert len(m_n_pairs) == cfg.num_heads, (
            f"{len(cfg.m_values)} m-values x {len(cfg.n_values)} n-values must equal "
            f"num_heads ({cfg.num_heads})"
        )
        self.register_buffer("m_slopes", torch.tensor([m for m, _ in m_n_pairs], dtype=torch.float32))
        self.register_buffer("n_slopes", torch.tensor([n for _, n in m_n_pairs], dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [TTBlock(cfg.dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

        self.output_proj = nn.Linear(cfg.dim, cfg.dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _compute_attn_mask(
        self,
        visibility_scores: torch.Tensor,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        num_components: int,
    ) -> torch.Tensor:
        """visibility_scores, frame_indices, valid_mask: (B, N) -> attn_mask
        (B * num_heads, N*C, N*C) ready for nn.MultiheadAttention (float mask,
        added to pre-softmax logits)."""
        batch_size, num_frames = visibility_scores.shape

        # Sec 4.3: mean subtraction only (no std normalization), independently per
        # window - each row of the (B, N) batch is one window/clip, so this mean is
        # never pooled across different windows/clips in the batch. Padded frames
        # are excluded from the mean: multiplying by valid_f zeroes their
        # contribution to the sum, and dividing by valid_count (not N) means the
        # mean is over the window's real frames only.
        valid_f = valid_mask.float()
        valid_count = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        window_mean = (visibility_scores * valid_f).sum(dim=-1, keepdim=True) / valid_count
        s_tilde = visibility_scores - window_mean  # (B, N)

        dist = (frame_indices[:, :, None] - frame_indices[:, None, :]).abs().float()  # (B, N, N)

        m = self.m_slopes.view(1, -1, 1, 1)  # (1, H, 1, 1)
        n = self.n_slopes.view(1, -1, 1, 1)
        s_tilde_j = s_tilde[:, None, None, :]  # (B, 1, 1, N), broadcasts over query dim i
        dist_ij = dist[:, None, :, :]  # (B, 1, N, N)

        # bias(i, j, h) = m_h * s_tilde_j - n_h * |i - j| (Sec 4.4)
        frame_bias = m * s_tilde_j - n * dist_ij  # (B, H, N, N)

        # Padded frames are masked out as keys only (see module docstring).
        key_invalid = ~valid_mask[:, None, None, :]  # (B, 1, 1, N)
        frame_bias = frame_bias.masked_fill(key_invalid, _MASK_VALUE)

        # All 4 component tokens of frame j share the same frame-level bias (Sec 4.4).
        token_bias = frame_bias.repeat_interleave(num_components, dim=2).repeat_interleave(
            num_components, dim=3
        )  # (B, H, N*C, N*C)
        num_heads = self.config.num_heads
        seq_len = num_frames * num_components
        return token_bias.reshape(batch_size * num_heads, seq_len, seq_len)

    def forward(
        self,
        tokens: torch.Tensor,
        visibility_scores: torch.Tensor,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tokens: (B, N, 4, D) - SViT-encoded component-token features (pre-MLP-head),
            frame-major order, for N frames of a (possibly padded) window.
        visibility_scores: (B, N) - raw per-frame face-visibility scores (Sec 4.1).
            Values at padded positions (valid_mask == False) are never used.
        frame_indices: (B, N) - each frame's actual temporal index, used only for
            pairwise distance. Values at padded positions are never used.
        valid_mask: (B, N) bool, True where the frame is real. Defaults to all-True
            (no padding) if omitted.
        Returns: (B, N, 4, D) refined tokens (tokens + zero-init-at-init delta).
            Rows at padded positions are numerically valid but meaningless.
        """
        batch_size, num_frames, num_components, dim = tokens.shape
        assert num_components == constants.NUM_COMPONENT_TOKENS
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=tokens.device)
        residual = tokens

        # Flatten (frame, component) into one sequence dim for full joint attention
        # (Sec 2.4: all 4 tokens x N frames attend to each other, not just within a
        # frame); frame-major order (token index = frame_idx*4 + component_idx)
        # matches the bias expansion below. Un-flattened back to (B,N,4,D) at the end.
        x = tokens + self.component_type_embedding[None, None, :, :]
        x = x.reshape(batch_size, num_frames * num_components, dim)

        attn_mask = self._compute_attn_mask(visibility_scores, frame_indices, valid_mask, num_components)

        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)

        delta = self.output_proj(x)
        delta = delta.reshape(batch_size, num_frames, num_components, dim)
        return residual + delta
