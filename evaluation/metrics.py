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
