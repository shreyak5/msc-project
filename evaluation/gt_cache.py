from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from dataset_processing.dataloading.crop_cache import get_cropped_face
from utils.cache_utils import bucket_container_path, entry_key, read_bucket_entry, sentinel_path, write_bucket_entry
from utils.landmark_utils import run_fan, run_mediapipe


def get_cropped_frame(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    frame_bgr: np.ndarray,
    face_detector: object,
    crop_scale: float,
    crop_size: int,
) -> np.ndarray | None:
    """Cached counterpart to evaluate_clip's own live
    `crop_face(frame, face_detector, scale=crop_scale, image_size=crop_size)[0]`
    call - same crop, same None-on-failure contract, but persisted to
    `cache_root` (pass eval_core.CROP_CACHE_ROOT) across runs."""
    cropped = get_cropped_face(
        cache_root, dataset, sample_id, frame_index,
        lambda: frame_bgr, lambda: face_detector, crop_scale, crop_size,
    )
    # get_cropped_face's own sentinel files - written (if not already present)
    # by the call above, whether this was a hit or a fresh miss - are the only
    # way to distinguish a real crop from its silent black-fallback array, same
    # check dataset_processing/dataloading/landmark_cache.py's own
    # _crop_has_real_face helper makes against the same sentinels.
    if sentinel_path(cache_root, dataset, sample_id, frame_index, "noface").exists():
        return None
    if sentinel_path(cache_root, dataset, sample_id, frame_index, "unreadable").exists():
        return None
    return cropped


def _entry_key(sample_id: str, frame_index: int | None, gt_crop_size: int, landmark_set: str) -> str:
    # gt_crop_size folded into the entry key (not the bucket path/container) -
    # one bucket container per (dataset, sample_id) still covers every
    # gt_crop_size a method might request, rather than fragmenting into
    # separate containers per resolution. landmark_set ('fan'/'mediapipe') keeps
    # the two independently cacheable, matching evaluate_clip's own independent
    # `if 'fan' in landmark_sets` / `if 'mediapipe' in landmark_sets` guards -
    # a single combined entry would risk caching a permanently-missing result
    # for whichever set wasn't enabled on the run that happened to populate it.
    return f"{entry_key(sample_id, frame_index)}__gt{gt_crop_size}__{landmark_set}"


def _read_cached_landmarks(cache_root, dataset, sample_id, frame_index, gt_crop_size, landmark_set, field):
    """Shared read path for get_gt_fan/get_gt_mediapipe below. Returns
    (found: bool, value: np.ndarray | None) - found=False means a genuine miss
    (or a self-healed corrupted entry), for the caller to compute+persist;
    found=True means a cache hit, value already resolved (None if that frame's
    detection was itself invalid)."""
    key = _entry_key(sample_id, frame_index, gt_crop_size, landmark_set)
    container_path = bucket_container_path(cache_root, dataset, sample_id)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            cached = np.load(io.BytesIO(cached_bytes))
            return True, (cached[field] if bool(cached["flag_valid"]) else None)
    except Exception as exc:
        # Self-healing, same as every other cache in this codebase - treat a
        # corrupted container/entry like a miss and recompute+rewrite below.
        print(f"warning: corrupted GT-landmark cache entry {key} in {container_path}, recomputing: {exc}")
    return False, None


def _write_cached_landmarks(cache_root, dataset, sample_id, frame_index, gt_crop_size, landmark_set, field, value):
    key = _entry_key(sample_id, frame_index, gt_crop_size, landmark_set)
    flag_valid = value is not None
    fallback = np.zeros((0, 2), dtype=np.float32) if not flag_valid else None
    buffer = io.BytesIO()
    np.savez(
        buffer,
        **{field: (value if flag_valid else fallback).astype(np.float32), "flag_valid": flag_valid},
    )
    write_bucket_entry(cache_root, dataset, sample_id, key, buffer.getvalue())


def get_gt_fan(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    gt_crop_size: int,
    gt_cropped_image: np.ndarray,
    fan_predictor: object,
    fan_box: np.ndarray,
) -> np.ndarray | None:
    """Returns gt_fan (68,2)|None in raw [0, gt_crop_size] pixel space -
    matching evaluate_clip's own current `run_fan(...)` computation exactly,
    just cached. `gt_cropped_image`/`fan_predictor`/`fan_box` are only used on
    a cache miss - a hit never touches them."""
    found, value = _read_cached_landmarks(cache_root, dataset, sample_id, frame_index, gt_crop_size, "fan", "gt_fan")
    if found:
        return value

    gt_fan, _scores = run_fan(fan_predictor, gt_cropped_image, fan_box)
    _write_cached_landmarks(cache_root, dataset, sample_id, frame_index, gt_crop_size, "fan", "gt_fan", gt_fan)
    return gt_fan


def get_gt_mediapipe(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    gt_crop_size: int,
    gt_cropped_image: np.ndarray,
    mediapipe_detector: object,
    mediapipe_gt_indices: np.ndarray,
) -> np.ndarray | None:
    """Returns gt_mp (105,2)|None in raw [0, gt_crop_size] pixel space -
    matching evaluate_clip's own current
    `run_mediapipe(...)[mediapipe_gt_indices, :2]` computation exactly, just
    cached. `gt_cropped_image`/`mediapipe_detector`/`mediapipe_gt_indices` are
    only used on a cache miss - a hit never touches them."""
    found, value = _read_cached_landmarks(
        cache_root, dataset, sample_id, frame_index, gt_crop_size, "mediapipe", "gt_mp",
    )
    if found:
        return value

    gt_mp_full = run_mediapipe(mediapipe_detector, gt_cropped_image)
    gt_mp = gt_mp_full[mediapipe_gt_indices, :2] if gt_mp_full is not None else None
    _write_cached_landmarks(cache_root, dataset, sample_id, frame_index, gt_crop_size, "mediapipe", "gt_mp", gt_mp)
    return gt_mp

