"""Gaussian x visibility-softmax kernel smoothing of raw (no-TT) FLAME params.

A non-learned baseline for TemporalTransformer's own visibility-score-biased
attention (implementation-plan.md Sec 4): each frame's final weight over its
window is a Gaussian-in-distance kernel `k` multiplied by a temperature-softmax
over the window's raw visibility scores `s` (Sec 4.4's visibility-only-head
formula, `softmax_j(visibility_j / T)`, reused verbatim here so the two are
directly comparable), renormalized to sum to 1. Every decoded param (shape,
expression, eyelid, jaw, scale, rotation, translation) is smoothed the same way,
independently per frame - no reliance on any model/torch state beyond the
tensors passed in, so this is pure and independently testable across (r, sigma,
T) without re-running any model forward pass.

Shared by temporal-smoothing-experiments/ (the standalone harness) and
evaluation/methods/ours_method.py's OursKernelSmoothMethod - lives under utils/
(not temporal-smoothing-experiments/, a hyphenated directory name and not a
real Python package) so evaluation/ can import it normally.
"""

from __future__ import annotations

import numpy as np
import torch


def gaussian_kernel_weights(offsets: np.ndarray, sigma: float) -> np.ndarray:
    """offsets: (w,) int, j - i for each key frame j in query frame i's window.
    k_j = exp(-offsets_j^2 / (2*sigma^2)) - unnormalized (smooth_encoded_params
    normalizes k*s jointly, not k on its own)."""
    return np.exp(-(offsets.astype(np.float64) ** 2) / (2.0 * sigma**2))


def visibility_softmax_weights(visibility_window: np.ndarray, temperature: float) -> np.ndarray:
    """visibility_window: (w,) raw (un-normalized) per-frame visibility scores in
    [0, 1] - implementation-plan.md Sec 4.1. s = softmax(visibility_window / T),
    identical formula to TT's visibility-only head (Sec 4.4). Subtracts the
    window max before exponentiating for numerical stability only - softmax is
    shift-invariant, so this doesn't change the result."""
    scaled = visibility_window.astype(np.float64) / temperature
    scaled = scaled - scaled.max()
    weights = np.exp(scaled)
    return weights / weights.sum()


def _window_bounds(frame_index: int, num_frames: int, radius: int) -> tuple[int, int]:
    """[lo, hi] inclusive, truncated (not padded) at the clip's boundaries -
    frames near the start/end simply get a smaller, asymmetric window, the same
    effective behavior TT's own masked boundary windows produce (Sec 4.2)."""
    lo = max(0, frame_index - radius)
    hi = min(num_frames - 1, frame_index + radius)
    return lo, hi


def frame_weights(
    frame_index: int, num_frames: int, visibility_scores: np.ndarray, radius: int, sigma: float, temperature: float,
) -> tuple[int, int, np.ndarray]:
    """Returns (lo, hi, weights) - weights: (hi - lo + 1,) float64, summing to 1,
    for query frame `frame_index`'s window [lo, hi]. weights = normalize(k * s)."""
    lo, hi = _window_bounds(frame_index, num_frames, radius)
    offsets = np.arange(lo, hi + 1) - frame_index
    k = gaussian_kernel_weights(offsets, sigma)
    s = visibility_softmax_weights(visibility_scores[lo : hi + 1], temperature)
    combined = k * s
    return lo, hi, combined / combined.sum()


def smooth_encoded_params(
    encoded: dict[str, torch.Tensor],
    visibility_scores: torch.Tensor | np.ndarray,
    radius: int,
    sigma: float,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """encoded: dict of (N, ...) tensors (e.g. extract.ClipEncoding.encoded, or
    model.encoding.encode_image's raw output). visibility_scores: (N,). Returns
    a same-shape dict where every frame i's entry is replaced by
    frame_weights(i, ...)-weighted average of its window, applied independently
    to every param tensor.

    No special-casing of invalid frames beyond their own (typically near-zero)
    visibility_scores value - mirrors TT itself, which only ever downweights an
    invalid frame via the visibility-only head rather than hard-masking it out
    of the window (model/encoding.py's encode_video passes real_frame_mask, not
    flag_visibility_valid, as TT's own valid_mask)."""
    if isinstance(visibility_scores, torch.Tensor):
        visibility_np = visibility_scores.detach().cpu().numpy()
    else:
        visibility_np = np.asarray(visibility_scores)

    num_frames = visibility_np.shape[0]
    smoothed = {name: torch.empty_like(param) for name, param in encoded.items()}

    for i in range(num_frames):
        lo, hi, weights = frame_weights(i, num_frames, visibility_np, radius, sigma, temperature)
        for name, param in encoded.items():
            weights_t = torch.as_tensor(weights, dtype=param.dtype, device=param.device)
            window = param[lo : hi + 1]  # (w, ...)
            weights_t = weights_t.view((window.shape[0],) + (1,) * (window.dim() - 1))
            smoothed[name][i] = (window * weights_t).sum(dim=0)

    return smoothed

