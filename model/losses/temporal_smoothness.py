from __future__ import annotations

import torch


def velocity_penalty(
    params: torch.Tensor, valid_mask: torch.Tensor | None = None, frame_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    assert params.shape[1] >= 2, "velocity_penalty needs at least 2 frames"
    first_diff = params[:, :-1] - params[:, 1:]  # (B, T-1, D)
    sq_diff = first_diff.pow(2)
    if valid_mask is None:
        return sq_diff.mean()
    term_valid = (valid_mask[:, :-1] & valid_mask[:, 1:]).float()  # (B, T-1)
    if frame_weight is not None:
        term_valid = term_valid * torch.maximum(frame_weight[:, :-1], frame_weight[:, 1:])
    denom = term_valid.sum().clamp(min=1e-8) * sq_diff.shape[-1]
    return (sq_diff * term_valid.unsqueeze(-1)).sum() / denom


def vertex_velocity_penalty(
    vertices: torch.Tensor,
    base_region_weights: torch.Tensor,
    gated_region_mask: torch.Tensor,
    gate: torch.Tensor,
    expressive_region_smooth_weight: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    assert vertices.shape[1] >= 2, "vertex_velocity_penalty needs at least 2 frames"
    sq_diff = (vertices[:, :-1] - vertices[:, 1:]).pow(2)  # (B, T-1, V, 3)
    gated_effective = 1.0 + (expressive_region_smooth_weight - 1.0) * gate  # (B, T-1)
    weight = (
        base_region_weights.view(1, 1, -1)
        + gated_region_mask.view(1, 1, -1) * gated_effective.unsqueeze(-1)
    )  # (B, T-1, V)
    if valid_mask is not None:
        pair_valid = (valid_mask[:, :-1] & valid_mask[:, 1:]).float()
        weight = weight * pair_valid.unsqueeze(-1)
    denom = (weight.sum() * sq_diff.shape[-1]).clamp(min=1e-8)
    return (sq_diff * weight.unsqueeze(-1)).sum() / denom


def compute_vertex_gate(
    gate_signal: torch.Tensor, mode: str = "min_vis", cap: float = 0.1, beta: float = 1.0,
) -> torch.Tensor:
    if mode == "min_vis":
        return 1.0 - torch.minimum(gate_signal[:, :-1], gate_signal[:, 1:])
    if mode == "delta_vis":
        raw = (gate_signal[:, :-1] - gate_signal[:, 1:]).abs()
        norm = (raw / cap).clamp(0.0, 1.0)
        return norm**beta
    raise ValueError(f"unknown vertex_gate_mode: {mode!r} (expected 'min_vis' or 'delta_vis')")