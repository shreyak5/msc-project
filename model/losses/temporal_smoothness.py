"""Temporal smoothness loss (implementation-plan.md Sec 6, "temporal pass"):
a velocity penalty (mean squared first difference, L2) applied uniformly to
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


def velocity_penalty(
    params: torch.Tensor, valid_mask: torch.Tensor | None = None, frame_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """params: (B, T, D) parameter sequence over T frames (T >= 2) -> scalar mean
    squared first difference (L2, (p(t) - p(t+1))^2). Discourages frame-to-frame
    jumps; used uniformly across expression/eyelid, jaw, camera+rotation, and
    shape parameters.

    valid_mask: (B, T) bool, optional - see module docstring.

    frame_weight: (B, T) float, optional - per-frame continuous weight (e.g.
    Pass C's synthetic-occlusion upweighting, training/stage2.py's
    compute_temporal_smoothness_losses). A term between frames t/t+1 is scaled
    by max(frame_weight[t], frame_weight[t+1]) - either endpoint being
    upweighted is enough, since that's exactly the jump the upweighted frame's
    prediction has to bridge. Requires valid_mask (frame_weight alone, without
    valid_mask, isn't a call pattern this project needs - every real caller
    already has a valid_mask available). Omitting it (the default) reproduces
    the unweighted valid_mask formula exactly."""
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
    """occlusion-experiment1.md's Change 2: an L2 penalty on FLAME vertex first
    differences, region- and occlusion-gate-weighted - the vertex-space counterpart
    of velocity_penalty above (which stays param-space, unmodified).

    vertices: (B, T, V, 3) FLAME vertices per frame (T >= 2).
    base_region_weights, gated_region_mask: (V,) each, from model/losses/mesh.py's
    build_gated_expressive_region_weights().
    gate: (B, T-1) float in [0, 1] - per frame-PAIR occlusion gate, e.g.
    gate(t) = 1 - min(vis(t), vis(t+1)) (caller's responsibility, training/
    stage2.py's run_pass_c) - the SAME gate value scales every gated region for
    that pair (lips and the other expressive regions alike), not a separate gate
    per region.
    expressive_region_smooth_weight: peak weight for gated regions under full
    occlusion (gate=1); baseline 1.0 at gate=0.
    valid_mask: (B, T) bool, optional - same real-frame-vs-padding convention as
    velocity_penalty; a pair is excluded unless both its frames are real.

    Effective per-(pair, vertex) weight =
        base_region_weights + gated_region_mask * (1 + (expressive_region_smooth_weight - 1) * gate)
    - so non-gated included regions (face/boundary) stay flat 1.0 regardless of
    gate, gated regions (lips + eye_region + left/right_eye_region + nose +
    forehead) scale from 1.0 (clean pair) up to expressive_region_smooth_weight
    (fully occluded pair), and excluded regions (eyeballs/neck/ears/scalp) stay 0
    throughout (both weight tensors are 0 there).

    L2 (not L1, matching velocity_penalty's own formula) on (V(t) - V(t+1))^2, mean-
    reduced by total weight mass actually used (weight.sum() * 3 coords), clamped to
    a minimum of 1e-8 to guard the all-excluded/all-padded 0/0 edge case - matches
    region_weighted_mesh_loss's and velocity_penalty's own normalization style."""
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
    """gate_signal: (B, N) per-frame score driving vertex_velocity_penalty's
    gate (run_pass_c's own visibility_for_tt, or a mouth-region-visibility
    signal when Stage2Config.mouth_gate_use_region_visibility is set - see
    run_pass_c's own docstring for which). Returns (B, N-1) in [0, 1], one
    value per consecutive frame pair.

    mode="min_vis" (default): gate = 1 - min(gate_signal[t], gate_signal[t+1])
    - the original formula (occlusion-experiment1.md's Change 2), factored out
    here unchanged. Poorly calibrated against real visibility_ratio data: it
    never sits near 1.0 even on clean frames (median 0.68-0.78 measured across
    csl_daily/how2sign/phoenix2014t), so this gate rarely nears its floor and
    isn't very selective between an ordinary frame and a genuinely occluded
    one. cap/beta are ignored in this mode.

    mode="delta_vis": frame-to-frame CHANGE in gate_signal, not its absolute
    level - targets occlusion onset/offset specifically (where a temporal
    "jump" in the model's prediction is actually likely), rather than
    demanding smoothness throughout a whole occluded stretch. Real per-frame
    diffs on this project's video datasets have a tight, near-zero baseline
    with a genuine heavy tail at real transition events, unlike raw
    visibility's own level. Three steps:
      raw = |gate_signal[t] - gate_signal[t+1]|
      norm = clamp(raw / cap, 0, 1) - cap calibrates "large enough to count as
        a real transition" to this data's own scale (see
        stage2-config-reference.md's vertex_gate_delta_cap entry for the
        measured percentiles behind the 0.1 default).
      gate = norm ** beta - beta > 1 suppresses small/moderate norm values
        much faster than large ones (a value already near 1 barely changes
        under any power), widening the separation between genuine transitions
        (which stay near 1) and ordinary jitter (pushed toward 0) as beta
        increases. beta=1 is a no-op (gate=norm, plain linear)."""
    if mode == "min_vis":
        return 1.0 - torch.minimum(gate_signal[:, :-1], gate_signal[:, 1:])
    if mode == "delta_vis":
        raw = (gate_signal[:, :-1] - gate_signal[:, 1:]).abs()
        norm = (raw / cap).clamp(0.0, 1.0)
        return norm**beta
    raise ValueError(f"unknown vertex_gate_mode: {mode!r} (expected 'min_vis' or 'delta_vis')")