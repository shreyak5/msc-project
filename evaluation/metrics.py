import numpy as np


def per_frame_euclidean_error(pred_xy, gt_xy):
    """Mean Euclidean pixel distance between predicted and reference 2D landmarks for one frame.

    Returns np.nan if either side is missing (None or contains NaN).
    """
    if pred_xy is None or gt_xy is None:
        return np.nan
    if np.isnan(pred_xy).any() or np.isnan(gt_xy).any():
        return np.nan
    return float(np.linalg.norm(pred_xy - gt_xy, axis=-1).mean())


def per_frame_euclidean_error_masked(pred_xy, gt_xy, visible_mask):
    """Same as per_frame_euclidean_error, but restricted to points where
    visible_mask (N,) bool is True. Returns np.nan if either side is missing,
    if visible_mask is None, or if visible_mask has zero True entries (no
    visible landmarks this frame) - NaN, not 0, consistent with this file's
    "NaN = missing/invalid data" convention (summarize() already excludes NaN
    frames). Unlike training's masked-mean loss, this is a reporting metric,
    not a differentiable objective, so there's no need to clamp a zero
    denominator to avoid a NaN gradient.
    """
    if pred_xy is None or gt_xy is None or visible_mask is None:
        return np.nan
    if np.isnan(pred_xy).any() or np.isnan(gt_xy).any():
        return np.nan
    if not np.any(visible_mask):
        return np.nan
    return float(np.linalg.norm(pred_xy[visible_mask] - gt_xy[visible_mask], axis=-1).mean())


def per_frame_vertex_error(vertices_t, vertices_t1):
    """Mean Euclidean distance between two consecutive frames' 3D mesh vertices.

    Returns np.nan if either side is missing (None or contains NaN).
    """
    if vertices_t is None or vertices_t1 is None:
        return np.nan
    if np.isnan(vertices_t).any() or np.isnan(vertices_t1).any():
        return np.nan
    return float(np.linalg.norm(vertices_t - vertices_t1, axis=-1).mean())


def summarize(errors):
    """Aggregate per-frame errors, ignoring NaN (missing-detection) frames."""
    errors = np.asarray(errors, dtype=np.float64)
    valid = errors[~np.isnan(errors)]
    return {
        'num_frames': int(errors.size),
        'num_valid_frames': int(valid.size),
        'num_invalid_frames': int(errors.size - valid.size),
        'mean': float(valid.mean()) if valid.size else None,
        'median': float(np.median(valid)) if valid.size else None,
        'std': float(valid.std()) if valid.size else None,
    }
