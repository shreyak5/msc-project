"""Temporal Transformer (TT): refines per-frame component tokens using true local
(sliding-window) attention with a fixed (non-learned) visibility-score + distance
attention bias in place of positional embeddings (implementation-plan.md Sec 2.4,
Sec 4).

Each query frame attends only to its own centred `window_size`-frame neighborhood
(radius = window_size // 2 on each side, Sec 4.2) - this is enforced as genuine local
attention, not a mask on top of dense attention: `nn.MultiheadAttention` always
computes a full QK^T regardless of any additive mask, so masking alone would still
cost O(N^2) and would not bound compute for long videos. Instead, each query frame's
own (up to) `window_size` neighbor frames are gathered (`_gather_local_windows`) into
a private key/value set, and `nn.MultiheadAttention` is called with a reshaped batch
dimension of B*N (one "batch entry" per query frame, not per clip) - giving O(N*w)
compute/memory, linear in however many frames N this is called with. This means:
- Training: N is whatever clip length the dataloader provides (`max_frames` in
  dataset_processing/config/dataloader.yaml) - a batching decision, unrelated to the
  window itself, and typically larger than window_size.
- Inference: an entire video (any N) can be passed in one call at batch size 1, since
  cost no longer depends on N - avoiding both compute blow-up and the artificial
  clip-boundary "seams" that fixed-size chunking would otherwise introduce (frames
  only lose neighbor access at the true start/end of the video, not at arbitrary
  chunk edges).

Frames without enough real neighbors on one side (near a clip/video boundary, or next
to clip-tail padding) simply get a smaller effective window - those slots are masked
out as attention *keys* only (a large negative bias, so they get ~zero attention
weight after softmax), never as queries: masking every key for a query row would give
an all -inf row, and softmax over an all -inf row is 0/0 = NaN. Padded-query output
rows are still numerically valid, just meaningless; callers discard them since they
don't correspond to a real frame. The window-mean used for score normalization
(Sec 4.3) is likewise computed per query's own local window, excluding invalid slots.

The caller also supplies each frame's actual temporal index (used only for pairwise
distance within a window). Passing explicit frame_indices rather than assuming a
contiguous arange(N) only actually matters once frames can be dropped/skipped within a
window (distance between two adjacent entries would then be >1 real timestep) - out of
scope for now per Sec 7, but a free thing to support today.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from model import constants
from model.config import TTConfig

_MASK_VALUE = -1e9


def _gather_local_windows(x: torch.Tensor, radius: int) -> torch.Tensor:
    """x: (B, N, *rest) -> (B, N, w, *rest), w = 2*radius+1, where
    output[:, i, k] = x[:, i - radius + k] for in-bounds k, and a zero-valued slot
    otherwise. The zero pad value is never used numerically - out-of-bounds window
    slots (video/clip boundary) are indistinguishable, by construction, from
    clip-tail padding once `valid_mask` is gathered through this same function with
    its own pad value forced to False, so both cases are excluded from attention by
    the identical masking mechanism in `_compute_attn_mask`.

    Implementation: zero-pad by `radius` on each side of the frame dimension, then
    slide a width-w window across it one step at a time (`Tensor.unfold`) - this is
    what gives every one of the N frames its own private centred neighborhood without
    a Python-level loop over frames."""
    batch_size, num_frames, *rest = x.shape
    window_size = 2 * radius + 1
    pad = torch.zeros((batch_size, radius, *rest), dtype=x.dtype, device=x.device)
    padded = torch.cat([pad, x, pad], dim=1)  # (B, N + 2*radius, *rest)
    flat = padded.reshape(batch_size, num_frames + 2 * radius, -1)  # (B, N + 2*radius, R)
    windows = flat.unfold(1, window_size, 1)  # (B, N, R, w) - unfold appends the window dim last
    windows = windows.permute(0, 1, 3, 2)  # (B, N, w, R)
    return windows.reshape(batch_size, num_frames, window_size, *rest)


class TTBlock(nn.Module):
    """Pre-norm transformer block, trained from scratch (no FaRL-fidelity
    constraint here, unlike model.encoder.SViT) - standard LayerNorm/GELU.
    Two residual connections per block (attention, then MLP) - the usual
    transformer-block pattern, same as model.encoder.ResidualAttentionBlock.

    The attention here is cross-attention from each frame's own tokens (query) to a
    freshly-gathered local window of tokens (key/value) - re-gathered from this
    block's own input every call, since after each block neighboring frames' tokens
    have also been updated (see TemporalTransformer's module docstring for why this
    can't just be computed once up front)."""

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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, radius: int) -> torch.Tensor:
        """x: (B, N, C, D). attn_mask: (B*N*num_heads, C, w*C) - already reshaped to
        match the B*N batching used here (see TemporalTransformer._compute_attn_mask).
        attn_mask (float) is added to the pre-softmax QK^T/sqrt(d) logits by
        nn.MultiheadAttention itself - required for both the ALiBi-style bias and
        the validity mask to work (see model/temporal.py module docstring)."""
        batch_size, num_frames, num_components, dim = x.shape
        normed = self.norm1(x)  # (B, N, C, D)
        kv = _gather_local_windows(normed, radius)  # (B, N, w, C, D)

        # nn.MultiheadAttention doesn't require "batch" to be the real batch size -
        # by using B*N (one entry per query frame) instead of B, this single call
        # computes genuinely local attention: each of the B*N entries only ever sees
        # its own w*C key/value tokens, not all N frames' tokens.
        q = normed.reshape(batch_size * num_frames, num_components, dim)
        kv = kv.reshape(batch_size * num_frames, -1, dim)
        attn_out, _ = self.attn(q, kv, kv, attn_mask=attn_mask, need_weights=False)
        attn_out = attn_out.reshape(batch_size, num_frames, num_components, dim)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    """Sec 2.4: 3-layer, 8-head transformer over the 4 component tokens x N frames,
    with each frame's attention restricted to its own centred window_size-frame
    neighborhood (Sec 4.2) via true local attention (see module docstring). Residual
    delta design (Sec 2.4): the final output projection is zero-initialized, so at
    init TT(tokens) == tokens exactly (identity map); TT refines tokens, it does not
    replace them. This is why forward() returns `residual + delta` rather than
    `delta` alone - only one layer (output_proj) is zero-initialized (not every
    internal block), so identity-at-init depends on this outer residual to carry
    the original tokens through untouched until TT has learned something worth adding.

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
        (B*N*num_heads, C, w*C) ready for nn.MultiheadAttention (float mask, added
        to pre-softmax logits), where w = self.config.window_size and "batch" is
        B*N (one entry per query frame, see TTBlock.forward)."""
        batch_size, num_frames = visibility_scores.shape
        radius = self.config.window_size // 2
        window_size = 2 * radius + 1

        local_vis = _gather_local_windows(visibility_scores, radius)  # (B, N, w)
        local_fidx = _gather_local_windows(frame_indices, radius)  # (B, N, w)
        local_valid = _gather_local_windows(valid_mask, radius)  # (B, N, w) bool

        # Sec 4.3: mean subtraction only (no std normalization), independently per
        # QUERY FRAME's own local window (not the whole clip) - each frame's window
        # is centred on itself, so "the window" is now a per-frame concept. Invalid
        # slots (video/clip boundary or clip-tail padding - both fall out of the
        # same valid_mask gather) are excluded from both the sum and the count.
        local_valid_f = local_valid.float()
        local_count = local_valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        local_mean = (local_vis * local_valid_f).sum(dim=-1, keepdim=True) / local_count
        s_tilde = local_vis - local_mean  # (B, N, w)

        # Distance from each query frame to each of its own gathered neighbors.
        dist = (frame_indices[:, :, None] - local_fidx).abs().float()  # (B, N, w)

        m = self.m_slopes.view(1, -1, 1, 1)  # (1, H, 1, 1)
        n = self.n_slopes.view(1, -1, 1, 1)
        s_tilde_ = s_tilde[:, None, :, :]  # (B, 1, N, w)
        dist_ = dist[:, None, :, :]  # (B, 1, N, w)

        # bias(i, j, h) = m_h * s_tilde_j - n_h * |i - j| (Sec 4.4), j now ranging
        # only over query i's own local window instead of the whole clip.
        frame_bias = m * s_tilde_ - n * dist_  # (B, H, N, w)

        # Invalid window slots (boundary/padding) masked out as keys only (see
        # module docstring).
        key_invalid = ~local_valid[:, None, :, :]  # (B, 1, N, w)
        frame_bias = frame_bias.masked_fill(key_invalid, _MASK_VALUE)

        # All 4 component tokens of a frame share that frame's bias (Sec 4.4) -
        # expand both the query-frame axis and the key-window axis to token
        # granularity.
        token_bias = frame_bias.repeat_interleave(num_components, dim=2)  # (B, H, N*C, w)
        token_bias = token_bias.repeat_interleave(num_components, dim=3)  # (B, H, N*C, w*C)

        num_heads = self.config.num_heads
        seq_len_kv = window_size * num_components
        # Reshape into nn.MultiheadAttention's expected (batch*num_heads, L, S)
        # layout with batch = B*N: split N*C back into (N, C), then move N next to
        # B (ahead of H) before merging - required so this mask's (frame, head)
        # ordering matches how TTBlock.forward reshapes Q/K/V into a B*N batch
        # (frame merged into batch, not into the head axis); a plain reshape here
        # (skipping the permute) would silently pair each frame's query with
        # another frame's bias.
        attn_mask = (
            token_bias.reshape(batch_size, num_heads, num_frames, num_components, seq_len_kv)
            .permute(0, 2, 1, 3, 4)  # (B, N, H, C, w*C)
            .reshape(batch_size * num_frames * num_heads, num_components, seq_len_kv)
        )
        return attn_mask

    def forward(
        self,
        tokens: torch.Tensor,
        visibility_scores: torch.Tensor,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tokens: (B, N, 4, D) - SViT-encoded component-token features (pre-MLP-head),
            for N frames (any N - see module docstring for training vs. inference).
        visibility_scores: (B, N) - raw per-frame face-visibility scores (Sec 4.1).
            Values at invalid positions (valid_mask == False) are never used.
        frame_indices: (B, N) - each frame's actual temporal index, used only for
            pairwise distance within a window. Values at invalid positions are
            never used.
        valid_mask: (B, N) bool, True where the frame is real. Defaults to all-True
            (no padding) if omitted.
        Returns: (B, N, 4, D) refined tokens (tokens + zero-init-at-init delta).
            Rows at invalid positions are numerically valid but meaningless.
        """
        batch_size, num_frames, num_components, dim = tokens.shape
        assert num_components == constants.NUM_COMPONENT_TOKENS
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=tokens.device)
        residual = tokens

        x = tokens + self.component_type_embedding[None, None, :, :]

        attn_mask = self._compute_attn_mask(visibility_scores, frame_indices, valid_mask, num_components)
        radius = self.config.window_size // 2

        for block in self.blocks:
            x = block(x, attn_mask, radius)
        x = self.norm(x)

        delta = self.output_proj(x)
        return residual + delta
