from __future__ import annotations

import torch
import torch.nn as nn

from model import constants
from model.config import GatedTTConfig, SimpleTTConfig, TTConfig

_MASK_VALUE = -1e9
_GATE_EPSILON = 1e-8


def _gather_local_windows(x: torch.Tensor, radius: int) -> torch.Tensor:
    batch_size, num_frames, *rest = x.shape
    window_size = 2 * radius + 1
    pad = torch.zeros((batch_size, radius, *rest), dtype=x.dtype, device=x.device)
    padded = torch.cat([pad, x, pad], dim=1)  # (B, N + 2*radius, *rest)
    flat = padded.reshape(batch_size, num_frames + 2 * radius, -1)  # (B, N + 2*radius, R)
    windows = flat.unfold(1, window_size, 1)  # (B, N, R, w) - unfold appends the window dim last
    windows = windows.permute(0, 1, 3, 2)  # (B, N, w, R)
    return windows.reshape(batch_size, num_frames, window_size, *rest)


class TTBlock(nn.Module):
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
