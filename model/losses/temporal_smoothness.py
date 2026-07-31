"""Temporal smoothness loss (implementation-plan.md Sec 6, "temporal pass"):
a velocity penalty (mean absolute first difference, L1) applied uniformly to
expression/eyelid, jaw, camera+rotation, and shape parameters - discourages
frame-to-frame jumps while tolerating smooth, sustained motion (a steady head
turn or a mouthing has a small, roughly constant first difference, not a
large one).

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


def velocity_penalty(params: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """params: (B, T, D) parameter sequence over T frames (T >= 2) -> scalar mean
    absolute first difference (L1, |p(t) - p(t+1)|). Discourages frame-to-frame
    jumps; used uniformly across expression/eyelid, jaw, camera+rotation, and
    shape parameters.

    valid_mask: (B, T) bool, optional - see module docstring."""
    assert params.shape[1] >= 2, "velocity_penalty needs at least 2 frames"
    first_diff = params[:, :-1] - params[:, 1:]  # (B, T-1, D)
    abs_diff = first_diff.abs()
    if valid_mask is None:
        return abs_diff.mean()
    term_valid = valid_mask[:, :-1] & valid_mask[:, 1:]  # (B, T-1)
    denom = term_valid.sum().clamp(min=1) * abs_diff.shape[-1]
    return (abs_diff * term_valid.unsqueeze(-1)).sum() / denom