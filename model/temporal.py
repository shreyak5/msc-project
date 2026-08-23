"""Temporal Transformer (TT): refines per-frame component tokens using true local
(sliding-window) attention, with visibility and distance no longer mixed into one
shared per-head bias (implementation-plan.md Sec 2.4, Sec 4). Of TT_NUM_HEADS heads,
TT_NUM_VISIBILITY_HEADS (1) is a dedicated visibility-only head: its attention
weights come purely from each key frame's raw visibility score, scaled by
1/config.visibility_temperature before softmax (no QK content term, no distance
term at all) - a fixed, non-learned attention *pattern*, though the value it
aggregates is still a normal learned V projection. The remaining heads keep
ordinary QK^T content attention plus a fixed (non-learned), distance-only ALiBi bias
in place of positional embeddings - no visibility term. Splitting the two signals
into separate heads (rather than one combined additive bias, as an earlier version
of this module used) means a genuinely occlusion-robust attention pattern doesn't
have to compete with content/distance terms inside the same softmax.

Each query frame attends only to its own centred `window_size`-frame neighborhood
(radius = window_size // 2 on each side, Sec 4.2) - this is enforced as genuine local
attention, not a mask on top of dense attention: computing a full dense QK^T over
every frame and masking out everything but the window afterward would still cost
O(N^2), not bounding compute for long videos. Instead, each query frame's own (up
to) `window_size` neighbor frames are gathered (`_gather_local_windows`) into a
private key/value set, and attention is computed with a reshaped batch dimension of
B*N (one "batch entry" per query frame, not per clip, see `TTBlock.forward`) -
giving O(N*w) compute/memory, linear in however many frames N this is called with.
This means:
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
out as attention *keys* only (a large negative value, `_MASK_VALUE = -1e9`, so they
get ~zero attention weight after softmax), never as queries: masking every key for a
query row would give an all-`_MASK_VALUE` row, which - unlike an all -inf row (0/0 =
NaN) - still softmaxes to a well-defined (uniform) distribution, since `_MASK_VALUE`
is a large finite negative, not literal -inf. Padded-query output rows are still
numerically valid, just meaningless; callers discard them since they don't
correspond to a real frame.

The caller also supplies each frame's actual temporal index (used only for pairwise
distance within a window). Passing explicit frame_indices rather than assuming a
contiguous arange(N) only actually matters once frames can be dropped/skipped within a
window (distance between two adjacent entries would then be >1 real timestep) - out of
scope for now per Sec 7, but a free thing to support today.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import constants
from model.config import GatedTTConfig, SimpleTTConfig, TTConfig

_MASK_VALUE = -1e9
_GATE_EPSILON = 1e-8


def _gather_local_windows(x: torch.Tensor, radius: int) -> torch.Tensor:
    """x: (B, N, *rest) -> (B, N, w, *rest), w = 2*radius+1, where
    output[:, i, k] = x[:, i - radius + k] for in-bounds k, and a zero-valued slot
    otherwise. The zero pad value is never used numerically - out-of-bounds window
    slots (video/clip boundary) are indistinguishable, by construction, from
    clip-tail padding once `valid_mask` is gathered through this same function with
    its own pad value forced to False, so both cases are excluded from attention by
    the identical masking mechanism in `TemporalTransformer._compute_distance_bias`/
    `_compute_visibility_logits`.

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
    can't just be computed once up front).

    Custom (not nn.MultiheadAttention) because heads are no longer uniform: the
    num_alibi_heads QK+ALiBi heads do real content-based QK^T attention, while the
    num_visibility_heads visibility-only head's attention weights come directly from
    a fixed visibility-derived logit, never from Q/K at all - a single
    nn.MultiheadAttention call has no way to give one head a different mechanism
    than the rest. Q/K projections are sized to the alibi heads only (the visibility
    head never needs them); V and the output projection span the full width, so
    every head - alibi or visibility - still aggregates its own learned value
    content, only the attention *weights* differ in how they're derived."""

    def __init__(self, dim: int, num_heads: int, num_visibility_heads: int, mlp_ratio: float):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        assert num_visibility_heads == 1, (
            "exactly one visibility-only head is supported: unlike the QK+ALiBi "
            "heads (which vary via distinct ALiBi slopes), there's no per-head "
            "variation mechanism for the visibility-only mechanism, so more than "
            "one would just be redundant duplicates of each other"
        )
        self.num_heads = num_heads
        self.num_visibility_heads = num_visibility_heads
        self.num_alibi_heads = num_heads - num_visibility_heads
        self.head_dim = dim // num_heads

        alibi_dim = self.num_alibi_heads * self.head_dim
        self.q_proj = nn.Linear(dim, alibi_dim)
        self.k_proj = nn.Linear(dim, alibi_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, distance_bias: torch.Tensor, visibility_logits: torch.Tensor, radius: int,
    ) -> torch.Tensor:
        """x: (B, N, C, D). distance_bias: (B, N, num_alibi_heads, C, w*C) - additive
        ALiBi distance-only bias for the QK+ALiBi heads (added to pre-softmax QK^T/
        sqrt(head_dim) logits), invalid key slots already masked to _MASK_VALUE.
        visibility_logits: (B, N, C, w*C) - temperature-scaled raw-visibility
        attention logits for the single visibility-only head (shared identically
        across query components, since visibility is a per-key-frame quantity, not
        per-token), invalid key slots already masked. Both come from
        TemporalTransformer._compute_distance_bias/_compute_visibility_logits (see
        model/temporal.py module docstring for why invalid slots are masked as keys
        only, never queries)."""
        batch_size, num_frames, num_components, dim = x.shape
        normed = self.norm1(x)  # (B, N, C, D)
        kv = _gather_local_windows(normed, radius)  # (B, N, w, C, D)

        # Merging N into the batch dim (bn = B*N, one entry per query frame) is what
        # makes this genuinely local attention: each of the bn entries only ever
        # sees its own w*C key/value tokens, not all N frames' tokens.
        bn = batch_size * num_frames
        q_flat = normed.reshape(bn, num_components, dim)
        kv_flat = kv.reshape(bn, -1, dim)  # (bn, w*C, D)

        q_a = self.q_proj(q_flat).view(bn, num_components, self.num_alibi_heads, self.head_dim).transpose(1, 2)
        k_a = self.k_proj(kv_flat).view(bn, -1, self.num_alibi_heads, self.head_dim).transpose(1, 2)
        v_all = self.v_proj(kv_flat).view(bn, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = v_all[:, : self.num_alibi_heads]  # (bn, num_alibi_heads, w*C, head_dim)
        v_vis = v_all[:, self.num_alibi_heads :]  # (bn, num_visibility_heads, w*C, head_dim)

        # QK+ALiBi heads: standard scaled-dot-product content attention plus the
        # distance-only bias, softmaxed per head over the window's w*C keys.
        dist_bias_flat = distance_bias.reshape(bn, self.num_alibi_heads, num_components, -1)
        alibi_logits = torch.matmul(q_a, k_a.transpose(-2, -1)) / (self.head_dim**0.5)
        alibi_logits = alibi_logits + dist_bias_flat
        alibi_weights = alibi_logits.softmax(dim=-1)
        alibi_out = torch.matmul(alibi_weights, v_a)  # (bn, num_alibi_heads, C, head_dim)

        # Visibility-only head: no Q/K involved at all - weights come directly from
        # visibility_logits, identical for every query component within a frame.
        vis_logits_flat = visibility_logits.reshape(bn, num_components, -1)  # (bn, C, w*C)
        vis_weights = vis_logits_flat.softmax(dim=-1).unsqueeze(1)  # (bn, 1, C, w*C)
        vis_out = torch.matmul(vis_weights, v_vis)  # (bn, num_visibility_heads, C, head_dim)

        attn_out = torch.cat([alibi_out, vis_out], dim=1)  # (bn, num_heads, C, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(bn, num_components, dim)
        attn_out = self.out_proj(attn_out)
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
    internal block, and not TTBlock's own internal out_proj either), so
    identity-at-init depends on this outer residual to carry the original tokens
    through untouched until TT has learned something worth adding.

    Component-type embeddings (Sec 4.4): TT has no positional embeddings at all -
    neither the distance-only ALiBi bias nor the visibility-only head's logits vary
    by which of the 4 tokens within a frame is being attended to, so nothing else in
    the attention mechanism distinguishes "this is the jaw token" from "this is the
    shape token". A dedicated learned embedding per component type gives that
    identity signal explicitly, independent of whatever content each SViT head
    happens to produce."""

    def __init__(self, config: TTConfig | None = None):
        super().__init__()
        self.config = config or TTConfig()
        cfg = self.config

        num_components = constants.NUM_COMPONENT_TOKENS
        self.component_type_embedding = nn.Parameter(torch.empty(num_components, cfg.dim))
        nn.init.trunc_normal_(self.component_type_embedding, std=0.02)

        self.num_alibi_heads = cfg.num_heads - cfg.num_visibility_heads
        assert len(cfg.alibi_slopes) == self.num_alibi_heads, (
            f"alibi_slopes has {len(cfg.alibi_slopes)} values, must equal "
            f"num_heads - num_visibility_heads ({self.num_alibi_heads})"
        )
        self.register_buffer("alibi_slopes", torch.tensor(cfg.alibi_slopes, dtype=torch.float32))
        assert cfg.visibility_temperature > 0, (
            f"visibility_temperature must be positive, got {cfg.visibility_temperature}"
        )

        self.blocks = nn.ModuleList(
            [TTBlock(cfg.dim, cfg.num_heads, cfg.num_visibility_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

        self.output_proj = nn.Linear(cfg.dim, cfg.dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _compute_distance_bias(
        self,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        num_components: int,
    ) -> torch.Tensor:
        """frame_indices, valid_mask: (B, N) -> distance-only ALiBi bias
        (B, N, num_alibi_heads, C, w*C) for the QK+ALiBi heads, where w =
        self.config.window_size. Invalid window slots (boundary/padding) are
        masked out as keys only, never queries (see module docstring)."""
        batch_size, num_frames = frame_indices.shape
        radius = self.config.window_size // 2
        window_size = 2 * radius + 1

        local_fidx = _gather_local_windows(frame_indices, radius)  # (B, N, w)
        local_valid = _gather_local_windows(valid_mask, radius)  # (B, N, w) bool

        # Distance from each query frame to each of its own gathered neighbors.
        dist = (frame_indices[:, :, None] - local_fidx).abs().float()  # (B, N, w)

        n = self.alibi_slopes.view(1, -1, 1, 1)  # (1, num_alibi_heads, 1, 1)
        dist_ = dist[:, None, :, :]  # (B, 1, N, w)

        # bias(i, j, h) = -n_h * |i - j|, j ranging only over query i's own local
        # window instead of the whole clip.
        frame_bias = -n * dist_  # (B, num_alibi_heads, N, w)

        key_invalid = ~local_valid[:, None, :, :]  # (B, 1, N, w)
        frame_bias = frame_bias.masked_fill(key_invalid, _MASK_VALUE)

        # All 4 component tokens of a frame share that frame's bias - expand both
        # the query-frame axis and the key-window axis to token granularity.
        token_bias = frame_bias.repeat_interleave(num_components, dim=2)  # (B, H, N*C, w)
        token_bias = token_bias.repeat_interleave(num_components, dim=3)  # (B, H, N*C, w*C)

        seq_len_kv = window_size * num_components
        token_bias = token_bias.reshape(batch_size, self.num_alibi_heads, num_frames, num_components, seq_len_kv)
        return token_bias.permute(0, 2, 1, 3, 4)  # (B, N, H, C, w*C)

    def _compute_visibility_logits(
        self,
        visibility_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        num_components: int,
    ) -> torch.Tensor:
        """visibility_scores, valid_mask: (B, N) -> raw-visibility attention logits
        (B, N, C, w*C) for the single visibility-only head - no distance term, no
        mean subtraction (unlike the old combined bias's s_tilde, this head's
        weights ARE the raw per-frame visibility score, not a deviation from it),
        scaled by 1/config.visibility_temperature. Same key-only invalid-slot
        masking as _compute_distance_bias - applied AFTER the temperature scaling
        (not before), so _MASK_VALUE always stays exactly _MASK_VALUE regardless
        of temperature, rather than being scaled into something too weak to
        actually zero out an invalid frame's attention weight."""
        batch_size, num_frames = visibility_scores.shape
        radius = self.config.window_size // 2
        window_size = 2 * radius + 1

        local_vis = _gather_local_windows(visibility_scores, radius)  # (B, N, w)
        local_valid = _gather_local_windows(valid_mask, radius)  # (B, N, w) bool

        vis_logits = local_vis / self.config.visibility_temperature
        key_invalid = ~local_valid  # (B, N, w)
        vis_logits = vis_logits.masked_fill(key_invalid, _MASK_VALUE)  # (B, N, w)

        # All 4 component tokens of both the query frame and each key frame share
        # that key frame's visibility logit (visibility is frame-level, not
        # per-token, exactly like the distance bias above).
        token_logits = vis_logits.repeat_interleave(num_components, dim=1)  # (B, N*C, w)
        token_logits = token_logits.repeat_interleave(num_components, dim=2)  # (B, N*C, w*C)

        seq_len_kv = window_size * num_components
        return token_logits.reshape(batch_size, num_frames, num_components, seq_len_kv)  # (B, N, C, w*C)

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

        distance_bias = self._compute_distance_bias(frame_indices, valid_mask, num_components)
        visibility_logits = self._compute_visibility_logits(visibility_scores, valid_mask, num_components)
        radius = self.config.window_size // 2

        for block in self.blocks:
            x = block(x, distance_bias, visibility_logits, radius)
        x = self.norm(x)

        delta = self.output_proj(x)
        return residual + delta


class SimpleTTBlock(nn.Module):
    """Pre-norm transformer block for SimpleTT (see SimpleTemporalTransformer):
    every head is an ordinary QK+ALiBi head - no visibility-only head, so unlike
    TTBlock there's no need to size Q/K separately from V or split the head
    dimension into alibi/visibility groups. Q/K/V/out projections all span the
    full width and every head goes through the identical QK^T + ALiBi-bias
    mechanism."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, distance_bias: torch.Tensor, radius: int) -> torch.Tensor:
        """x: (B, N, C, D). distance_bias: (B, N, num_heads, C, w*C) - additive
        ALiBi distance-only bias for every head (added to pre-softmax QK^T/
        sqrt(head_dim) logits), invalid key slots already masked to _MASK_VALUE.
        See TTBlock.forward for the shared local-window-gathering mechanics."""
        batch_size, num_frames, num_components, dim = x.shape
        normed = self.norm1(x)  # (B, N, C, D)
        kv = _gather_local_windows(normed, radius)  # (B, N, w, C, D)

        bn = batch_size * num_frames
        q_flat = normed.reshape(bn, num_components, dim)
        kv_flat = kv.reshape(bn, -1, dim)  # (bn, w*C, D)

        q = self.q_proj(q_flat).view(bn, num_components, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_flat).view(bn, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_flat).view(bn, -1, self.num_heads, self.head_dim).transpose(1, 2)

        dist_bias_flat = distance_bias.reshape(bn, self.num_heads, num_components, -1)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        logits = logits + dist_bias_flat
        weights = logits.softmax(dim=-1)
        attn_out = torch.matmul(weights, v)  # (bn, num_heads, C, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(bn, num_components, dim)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.reshape(batch_size, num_frames, num_components, dim)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleTemporalTransformer(nn.Module):
    """Alternate TT design: same architecture as TemporalTransformer (Sec 2.4) -
    windowed local attention, component-type embeddings, zero-init output_proj
    residual-delta - but with no visibility-score input and no visibility-only
    head at all. Every one of config.num_heads heads is an ordinary QK+ALiBi head
    (Sec 4.4's QK+ALiBi mechanism), so temporal order is conveyed solely through
    ALiBi distance bias, with no occlusion-awareness signal whatsoever - a
    simpler baseline to compare the visibility-split design against."""

    def __init__(self, config: SimpleTTConfig | None = None):
        super().__init__()
        self.config = config or SimpleTTConfig()
        cfg = self.config

        num_components = constants.NUM_COMPONENT_TOKENS
        self.component_type_embedding = nn.Parameter(torch.empty(num_components, cfg.dim))
        nn.init.trunc_normal_(self.component_type_embedding, std=0.02)

        assert len(cfg.alibi_slopes) == cfg.num_heads, (
            f"alibi_slopes has {len(cfg.alibi_slopes)} values, must equal num_heads ({cfg.num_heads})"
        )
        self.register_buffer("alibi_slopes", torch.tensor(cfg.alibi_slopes, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [SimpleTTBlock(cfg.dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

        self.output_proj = nn.Linear(cfg.dim, cfg.dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _compute_distance_bias(
        self,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        num_components: int,
    ) -> torch.Tensor:
        """Same as TemporalTransformer._compute_distance_bias, sized to all
        config.num_heads instead of a QK+ALiBi subset - there is no
        visibility-only head here to exclude."""
        batch_size, num_frames = frame_indices.shape
        radius = self.config.window_size // 2
        window_size = 2 * radius + 1

        local_fidx = _gather_local_windows(frame_indices, radius)  # (B, N, w)
        local_valid = _gather_local_windows(valid_mask, radius)  # (B, N, w) bool

        dist = (frame_indices[:, :, None] - local_fidx).abs().float()  # (B, N, w)

        n = self.alibi_slopes.view(1, -1, 1, 1)  # (1, num_heads, 1, 1)
        dist_ = dist[:, None, :, :]  # (B, 1, N, w)

        frame_bias = -n * dist_  # (B, num_heads, N, w)

        key_invalid = ~local_valid[:, None, :, :]  # (B, 1, N, w)
        frame_bias = frame_bias.masked_fill(key_invalid, _MASK_VALUE)

        token_bias = frame_bias.repeat_interleave(num_components, dim=2)  # (B, H, N*C, w)
        token_bias = token_bias.repeat_interleave(num_components, dim=3)  # (B, H, N*C, w*C)

        seq_len_kv = window_size * num_components
        token_bias = token_bias.reshape(batch_size, self.config.num_heads, num_frames, num_components, seq_len_kv)
        return token_bias.permute(0, 2, 1, 3, 4)  # (B, N, H, C, w*C)

    def forward(
        self,
        tokens: torch.Tensor,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tokens: (B, N, 4, D). frame_indices: (B, N) - each frame's actual temporal
        index, used only for pairwise distance within a window. valid_mask: (B, N)
        bool, True where the frame is real; defaults to all-True. No
        visibility_scores argument - this design has no visibility-only head to
        feed. Returns: (B, N, 4, D) refined tokens (tokens + zero-init-at-init
        delta). Rows at invalid positions are numerically valid but meaningless.
        """
        batch_size, num_frames, num_components, dim = tokens.shape
        assert num_components == constants.NUM_COMPONENT_TOKENS
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=tokens.device)
        residual = tokens

        x = tokens + self.component_type_embedding[None, None, :, :]

        distance_bias = self._compute_distance_bias(frame_indices, valid_mask, num_components)
        radius = self.config.window_size // 2

        for block in self.blocks:
            x = block(x, distance_bias, radius)
        x = self.norm(x)

        delta = self.output_proj(x)
        return residual + delta


class GatedTTBlock(nn.Module):
    """Pre-norm transformer block for GatedTT (see GatedTemporalTransformer):
    structurally identical to SimpleTTBlock (every head is an ordinary QK+ALiBi
    head, Q/K/V/out projections all span the full width) - the only difference is
    an extra post-softmax step in forward(): each head's softmax attention
    weights are gated by a per-key visibility factor and renormalized back into a
    valid distribution before being used to aggregate V, rather than going
    straight from softmax to the V matmul."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, distance_bias: torch.Tensor, visibility_gate: torch.Tensor, radius: int,
    ) -> torch.Tensor:
        """x: (B, N, C, D). distance_bias: (B, N, num_heads, C, w*C) - additive
        ALiBi distance-only bias for every head, invalid key slots already masked
        to _MASK_VALUE (same as SimpleTTBlock). visibility_gate: (B, N, C, w*C) -
        bounded [0, 1] multiplicative gate (visibility_j ** gamma), shared
        identically across every head (uniform mechanism) and already zeroed at
        invalid key slots - see GatedTemporalTransformer._compute_visibility_gate."""
        batch_size, num_frames, num_components, dim = x.shape
        normed = self.norm1(x)  # (B, N, C, D)
        kv = _gather_local_windows(normed, radius)  # (B, N, w, C, D)

        bn = batch_size * num_frames
        q_flat = normed.reshape(bn, num_components, dim)
        kv_flat = kv.reshape(bn, -1, dim)  # (bn, w*C, D)

        q = self.q_proj(q_flat).view(bn, num_components, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_flat).view(bn, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_flat).view(bn, -1, self.num_heads, self.head_dim).transpose(1, 2)

        dist_bias_flat = distance_bias.reshape(bn, self.num_heads, num_components, -1)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        logits = logits + dist_bias_flat
        weights = logits.softmax(dim=-1)  # (bn, num_heads, C, w*C)

        # Post-softmax visibility gating + renormalization - the same gate value
        # for every head (broadcast over the head dim), so the redistribution
        # effect is uniform across all 8 heads rather than head-specific.
        gate_flat = visibility_gate.reshape(bn, 1, num_components, -1)  # (bn, 1, C, w*C)
        combined = weights * gate_flat
        weights = combined / (combined.sum(dim=-1, keepdim=True) + _GATE_EPSILON)

        attn_out = torch.matmul(weights, v)  # (bn, num_heads, C, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(bn, num_components, dim)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.reshape(batch_size, num_frames, num_components, dim)

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GatedTemporalTransformer(nn.Module):
    """Third TT design: same architecture as SimpleTemporalTransformer (all
    num_heads heads are ordinary QK+ALiBi heads, no dedicated visibility-only
    head) but visibility is reintroduced as a uniform post-softmax gate applied to
    every head: gate_j = visibility_j ** gamma (raw per-frame score, Sec 4.1),
    multiplied into each head's already-softmaxed attention weights and then
    renormalized back into a valid distribution (GatedTTBlock.forward). Unlike
    TemporalTransformer's dedicated visibility-only head (whose weights come
    purely from visibility, with no content/distance term at all) or
    SimpleTemporalTransformer (no visibility signal whatsoever), this design lets
    visibility reshape every head's existing content+distance attention pattern -
    redistributing weight toward more-visible frames - without competing against
    QK/distance inside the same pre-softmax logits (which would have a sign
    problem: those logits can be negative, so a pre-softmax visibility term could
    weaken rather than strengthen suppression). Same residual-delta / zero-init-
    output_proj / component-type-embedding design as the other two TT variants -
    see TemporalTransformer's docstring."""

    def __init__(self, config: GatedTTConfig | None = None):
        super().__init__()
        self.config = config or GatedTTConfig()
        cfg = self.config

        num_components = constants.NUM_COMPONENT_TOKENS
        self.component_type_embedding = nn.Parameter(torch.empty(num_components, cfg.dim))
        nn.init.trunc_normal_(self.component_type_embedding, std=0.02)

        assert len(cfg.alibi_slopes) == cfg.num_heads, (
            f"alibi_slopes has {len(cfg.alibi_slopes)} values, must equal num_heads ({cfg.num_heads})"
        )
        self.register_buffer("alibi_slopes", torch.tensor(cfg.alibi_slopes, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [GatedTTBlock(cfg.dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)

        self.output_proj = nn.Linear(cfg.dim, cfg.dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _compute_distance_bias(
        self,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        num_components: int,
    ) -> torch.Tensor:
        """Same as SimpleTemporalTransformer._compute_distance_bias - sized to all
        config.num_heads, no visibility-only head to exclude."""
        batch_size, num_frames = frame_indices.shape
        radius = self.config.window_size // 2
        window_size = 2 * radius + 1

        local_fidx = _gather_local_windows(frame_indices, radius)  # (B, N, w)
        local_valid = _gather_local_windows(valid_mask, radius)  # (B, N, w) bool

        dist = (frame_indices[:, :, None] - local_fidx).abs().float()  # (B, N, w)

        n = self.alibi_slopes.view(1, -1, 1, 1)  # (1, num_heads, 1, 1)
        dist_ = dist[:, None, :, :]  # (B, 1, N, w)

        frame_bias = -n * dist_  # (B, num_heads, N, w)

        key_invalid = ~local_valid[:, None, :, :]  # (B, 1, N, w)
        frame_bias = frame_bias.masked_fill(key_invalid, _MASK_VALUE)

        token_bias = frame_bias.repeat_interleave(num_components, dim=2)  # (B, H, N*C, w)
        token_bias = token_bias.repeat_interleave(num_components, dim=3)  # (B, H, N*C, w*C)

        seq_len_kv = window_size * num_components
        token_bias = token_bias.reshape(batch_size, self.config.num_heads, num_frames, num_components, seq_len_kv)
        return token_bias.permute(0, 2, 1, 3, 4)  # (B, N, H, C, w*C)

    def _compute_visibility_gate(
        self,
        visibility_scores: torch.Tensor,
        valid_mask: torch.Tensor,
        num_components: int,
    ) -> torch.Tensor:
        """visibility_scores, valid_mask: (B, N) -> bounded [0, 1] multiplicative
        gate (B, N, C, w*C), shared identically across every head and across the
        query-component axis (visibility is a per-key-frame quantity, not
        per-token - same convention as TemporalTransformer._compute_visibility_
        logits). gate_j = clamp(visibility_j, 0, 1) ** gamma - clamped defensively
        (visibility scores are a [0, 1] ratio by construction, Sec 4.1, but a
        fractional gamma on a negative or >1 base could otherwise produce NaN/
        values outside [0, 1]). Invalid window slots (boundary/padding) are forced
        to exactly 0.0 - not just left to whatever a zero-padded visibility score
        would give (0.0 ** gamma is already 0 for gamma > 0, but this stays
        correct even for gamma == 0, where x ** 0 == 1 would otherwise leak a
        nonzero gate at an invalid slot)."""
        batch_size, num_frames = visibility_scores.shape
        radius = self.config.window_size // 2
        window_size = 2 * radius + 1

        local_vis = _gather_local_windows(visibility_scores, radius)  # (B, N, w)
        local_valid = _gather_local_windows(valid_mask, radius)  # (B, N, w) bool

        gate = local_vis.clamp(0.0, 1.0) ** self.config.gamma
        gate = gate.masked_fill(~local_valid, 0.0)  # (B, N, w)

        token_gate = gate.repeat_interleave(num_components, dim=1)  # (B, N*C, w)
        token_gate = token_gate.repeat_interleave(num_components, dim=2)  # (B, N*C, w*C)

        seq_len_kv = window_size * num_components
        return token_gate.reshape(batch_size, num_frames, num_components, seq_len_kv)  # (B, N, C, w*C)

    def forward(
        self,
        tokens: torch.Tensor,
        visibility_scores: torch.Tensor,
        frame_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tokens: (B, N, 4, D). visibility_scores: (B, N) - raw per-frame
        face-visibility scores (Sec 4.1), used only to build the post-softmax
        gate (never mean-subtracted, never fed to Q/K). frame_indices: (B, N).
        valid_mask: (B, N) bool, defaults to all-True. Returns: (B, N, 4, D)
        refined tokens (tokens + zero-init-at-init delta).
        """
        batch_size, num_frames, num_components, dim = tokens.shape
        assert num_components == constants.NUM_COMPONENT_TOKENS
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=tokens.device)
        residual = tokens

        x = tokens + self.component_type_embedding[None, None, :, :]

        distance_bias = self._compute_distance_bias(frame_indices, valid_mask, num_components)
        visibility_gate = self._compute_visibility_gate(visibility_scores, valid_mask, num_components)
        radius = self.config.window_size // 2

        for block in self.blocks:
            x = block(x, distance_bias, visibility_gate, radius)
        x = self.norm(x)

        delta = self.output_proj(x)
        return residual + delta


# Union of the three TT architectures a caller might be holding. NOT
# interchangeable on a single shared call signature, despite that: unlike
# TemporalTransformer/GatedTemporalTransformer (both forward(tokens,
# visibility_scores, frame_indices, valid_mask=None)), SimpleTemporalTransformer
# has no visibility_scores parameter at all (forward(tokens, frame_indices,
# valid_mask=None)) - by design, since it has no visibility signal anywhere in
# it. Callers that accept any TTModule (e.g. model/encoding.py's encode_video)
# must branch on isinstance(tt, SimpleTemporalTransformer) rather than calling
# uniformly.
TTModule = TemporalTransformer | SimpleTemporalTransformer | GatedTemporalTransformer
