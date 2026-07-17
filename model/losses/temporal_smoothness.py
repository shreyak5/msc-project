"""Temporal smoothness losses (implementation-plan.md Sec 6, "temporal pass"):
acceleration penalty (second difference, permits genuine motion, penalizes only
jitter) applied to expression/eyelid/jaw/camera+rotation parameters, and velocity
penalty (first difference) applied to shape parameters (which should be near-
constant within a video - the same person's face shape doesn't change frame to
frame).

Fully self-contained - not adapted from SMIRK/DECA/TokenFace, since this is a
standard formula given directly and completely in the plan itself.

Padding note: an optional valid_mask (B, T) marks which frames are real vs.
tail-padding (Sec 4.2's TT padding, model/temporal.py's own valid_mask concept -
now a real, current requirement for Sec 7's temporal pass, which does feed
padded clips via VideoFaceDataset's clip mode) - a difference term is only
included in the mean if EVERY frame it depends on is real, so a naive jump
across the real/padded boundary never contaminates the loss. Omitting
valid_mask (the default) preserves the original unpadded-sequence behavior
exactly - every existing caller/test is unaffected.
"""

from __future__ import annotations

import torch


def acceleration_penalty(params: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """params: (B, T, D) parameter sequence over T frames (T >= 3) -> scalar mean
    squared second difference (‖p(t-1) - 2p(t) + p(t+1)‖^2), interior frames only.
    Penalizes only JITTER (a change in velocity), not genuine constant-velocity
    motion (e.g. a steady head turn has zero second difference) - so it permits
    fast motion like mouthings or grammatical head nods/shakes.

    valid_mask: (B, T) bool, optional - see module docstring."""
    assert params.shape[1] >= 3, "acceleration_penalty needs at least 3 frames"
    second_diff = params[:, :-2] - 2 * params[:, 1:-1] + params[:, 2:]  # (B, T-2, D)
    squared = second_diff**2
    if valid_mask is None:
        return squared.mean()
    term_valid = valid_mask[:, :-2] & valid_mask[:, 1:-1] & valid_mask[:, 2:]  # (B, T-2)
    denom = term_valid.sum().clamp(min=1) * squared.shape[-1]
    return (squared * term_valid.unsqueeze(-1)).sum() / denom


def velocity_penalty(params: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """params: (B, T, D) parameter sequence over T frames (T >= 2) -> scalar mean
    squared first difference (‖p(t) - p(t+1)‖^2). Used for shape parameters, which
    should be near-constant within a single video.

    valid_mask: (B, T) bool, optional - see module docstring."""
    assert params.shape[1] >= 2, "velocity_penalty needs at least 2 frames"
    first_diff = params[:, :-1] - params[:, 1:]  # (B, T-1, D)
    squared = first_diff**2
    if valid_mask is None:
        return squared.mean()
    term_valid = valid_mask[:, :-1] & valid_mask[:, 1:]  # (B, T-1)
    denom = term_valid.sum().clamp(min=1) * squared.shape[-1]
    return (squared * term_valid.unsqueeze(-1)).sum() / denom