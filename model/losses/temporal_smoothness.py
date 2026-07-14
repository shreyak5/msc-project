"""Temporal smoothness losses (implementation-plan.md Sec 6, "temporal pass"):
acceleration penalty (second difference, permits genuine motion, penalizes only
jitter) applied to expression/eyelid/jaw/camera+rotation parameters, and velocity
penalty (first difference) applied to shape parameters (which should be near-
constant within a video - the same person's face shape doesn't change frame to
frame).

Fully self-contained - not adapted from SMIRK/DECA/TokenFace, since this is a
standard formula given directly and completely in the plan itself.

Padding note: these assume a full, unpadded sequence of T real consecutive frames.
If Sec 7's temporal pass ever feeds a window with padded frames at the boundary
(Sec 4.2's TT padding, model/temporal.py's valid_mask), a naive difference across
the real/padded boundary would register a spurious jump - not handled here, since
Sec 7's actual clip-windowing strategy isn't decided yet and may avoid boundary
padding during training entirely (unlike inference, training doesn't necessarily
need to cover every possible clip position). Revisit if/when that turns out to be
needed.
"""

from __future__ import annotations

import torch


def acceleration_penalty(params: torch.Tensor) -> torch.Tensor:
    """params: (B, T, D) parameter sequence over T frames (T >= 3) -> scalar mean
    squared second difference (‖p(t-1) - 2p(t) + p(t+1)‖^2), interior frames only.
    Penalizes only JITTER (a change in velocity), not genuine constant-velocity
    motion (e.g. a steady head turn has zero second difference) - so it permits
    fast motion like mouthings or grammatical head nods/shakes."""
    assert params.shape[1] >= 3, "acceleration_penalty needs at least 3 frames"
    second_diff = params[:, :-2] - 2 * params[:, 1:-1] + params[:, 2:]  # (B, T-2, D)
    return (second_diff**2).mean()


def velocity_penalty(params: torch.Tensor) -> torch.Tensor:
    """params: (B, T, D) parameter sequence over T frames (T >= 2) -> scalar mean
    squared first difference (‖p(t) - p(t+1)‖^2). Used for shape parameters, which
    should be near-constant within a single video."""
    assert params.shape[1] >= 2, "velocity_penalty needs at least 2 frames"
    first_diff = params[:, :-1] - params[:, 1:]  # (B, T-1, D)
    return (first_diff**2).mean()