from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np

from dataset_processing.dataloading.cache_utils import atomic_write_bytes, cache_key_path
from dataset_processing.dataloading.crop_cache import get_cropped_face
from model import constants
from model.losses.landmark import NUM_FAN_BOUNDARY_POINTS
from preprocessing.cropping import get_cropped_face_box
from utils.landmark_utils import run_fan, run_mediapipe

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Same asset model/losses/landmark.py's _embedded_indices reads from - here used
# directly to slice raw 478-point MediaPipe output down to the curated 105-point
# subset (landmark_indices[i] = the raw-478-space index of curated point i).
_CURATED_MEDIAPIPE_INDICES = np.load(_REPO_ROOT / constants.FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH)["landmark_indices"]


def _normalize(points_xy: np.ndarray, image_size: int) -> np.ndarray:
    """[0, image_size] pixel coordinates -> [-1, 1], matching the space
    model/flame/renderer.py's batch_orth_proj projects FLAME's landmarks into
    (see model/losses/landmark.py's module docstring for why)."""
    return points_xy / image_size * 2 - 1


def _crop_has_real_face(crop_cache_root: str | Path, dataset: str, sample_id: str, frame_index: int | None) -> bool:
    """crop_cache.get_cropped_face's return value alone can't distinguish a real
    crop from its silent all-black fallback (no face detected / unreadable
    source) - it has no validity flag in its contract. Its own .noface/
    .unreadable sentinel files (written by that same call, on disk before it
    returns) are the actual ground truth, so check those directly instead -
    otherwise FAN/MediaPipe would run on a blank placeholder and (see run_fan's
    docstring: it has no "no face" failure mode once given a box) silently
    report flag_landmarks_fan_valid=True for a crop with no real face at all."""
    key_path = cache_key_path(Path(crop_cache_root), dataset, sample_id, frame_index)
    return not key_path.with_suffix(".noface").exists() and not key_path.with_suffix(".unreadable").exists()


def get_landmarks(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    get_fan_predictor: Callable[[], object],
    get_mediapipe_detector: Callable[[], object],
    crop_cache_root: str | Path,
    crop_scale: float,
    image_size: int,
    on_error: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray | bool]:
    """Caches GT landmarks for the landmark loss (model/losses/landmark.py):
    FAN's boundary/jaw-contour points ((17, 2), matching NUM_FAN_BOUNDARY_POINTS)
    and MediaPipe's curated ((105, 2)) points, both already normalized to
    [-1, 1] - the same crop-pixel-space FLAME's projected landmarks land in
    (see _normalize's docstring).

    Detection runs on the SAME crop the main model input uses (via
    crop_cache.get_cropped_face, reusing crop_cache_root/crop_scale/image_size -
    a cache hit there is just a cheap disk read, not a re-detection), not a
    separate raw-image-space detection later warped into crop space: since that
    crop is already a fixed, deterministic function of (dataset, sample_id,
    frame_index) - unlike SMIRK's own per-access-randomized crop augmentation -
    there's nothing to warp against, so landmarks computed directly on it are
    already in the right space.

    FAN needs a box but not a detector: preprocessing.cropping.
    get_cropped_face_box(image_size, crop_scale) gives the fixed, analytically-
    derived box the original detected face occupies within any such crop (see
    its own docstring) - no second, redundant detector call. MediaPipe's own
    run_mediapipe never took a detector at all (its own detect() call handles
    that internally, opaquely).

    Returns a dict with `landmarks_fan` (17,2), `flag_landmarks_fan_valid`,
    `landmarks_mp` (105,2), `flag_landmarks_mp_valid` - explicit validity flags
    (not NaN-filled arrays, despite that being evaluation/extract_landmarks.py's
    own convention) to match this project's own established pattern for a
    cached value used as a loss *target* (mica_cache.py's flag_mica_valid) -
    keeps the eventual training loop's gating logic uniform across losses."""
    key_path = cache_key_path(Path(cache_root), dataset, sample_id, frame_index)
    npz_path = key_path.with_suffix(".npz")

    fan_fallback = np.zeros((NUM_FAN_BOUNDARY_POINTS, 2), dtype=np.float32)
    mp_fallback = np.zeros((len(_CURATED_MEDIAPIPE_INDICES), 2), dtype=np.float32)

    if npz_path.exists():
        try:
            cached = np.load(npz_path)
            return {
                "landmarks_fan": cached["landmarks_fan"],
                "flag_landmarks_fan_valid": bool(cached["flag_landmarks_fan_valid"]),
                "landmarks_mp": cached["landmarks_mp"],
                "flag_landmarks_mp_valid": bool(cached["flag_landmarks_mp_valid"]),
            }
        except Exception as exc:
            # Corrupted/truncated cache entry - treat like a cache miss and
            # recompute live below, self-healing rather than crashing (see
            # face_parsing_cache.py's get_face_parsing for the full reasoning).
            print(f"warning: corrupted cache entry {npz_path}, recomputing: {exc}")

    try:
        crop = get_cropped_face(
            crop_cache_root, dataset, sample_id, frame_index,
            load_source_image, get_detector, crop_scale, image_size,
        )
    except Exception as exc:
        if on_error is not None:
            on_error(str(exc))
        return {
            "landmarks_fan": fan_fallback, "flag_landmarks_fan_valid": False,
            "landmarks_mp": mp_fallback, "flag_landmarks_mp_valid": False,
        }

    if not _crop_has_real_face(crop_cache_root, dataset, sample_id, frame_index):
        result = {
            "landmarks_fan": fan_fallback, "flag_landmarks_fan_valid": False,
            "landmarks_mp": mp_fallback, "flag_landmarks_mp_valid": False,
        }
        buffer = io.BytesIO()
        np.savez(buffer, **result)
        atomic_write_bytes(npz_path, buffer.getvalue())
        return result

    fan_box = get_cropped_face_box(image_size=image_size, scale=crop_scale)
    landmarks_fan, _scores = run_fan(get_fan_predictor(), crop, fan_box)
    if landmarks_fan is None:
        landmarks_fan_out, flag_fan_valid = fan_fallback, False
    else:
        landmarks_fan_out = _normalize(landmarks_fan[:NUM_FAN_BOUNDARY_POINTS].astype(np.float32), image_size)
        flag_fan_valid = True

    landmarks_mp_raw = run_mediapipe(get_mediapipe_detector(), crop)
    if landmarks_mp_raw is None:
        landmarks_mp_out, flag_mp_valid = mp_fallback, False
    else:
        curated = landmarks_mp_raw[_CURATED_MEDIAPIPE_INDICES, :2].astype(np.float32)
        landmarks_mp_out = _normalize(curated, image_size)
        flag_mp_valid = True

    buffer = io.BytesIO()
    np.savez(
        buffer,
        landmarks_fan=landmarks_fan_out, flag_landmarks_fan_valid=flag_fan_valid,
        landmarks_mp=landmarks_mp_out, flag_landmarks_mp_valid=flag_mp_valid,
    )
    atomic_write_bytes(npz_path, buffer.getvalue())

    return {
        "landmarks_fan": landmarks_fan_out, "flag_landmarks_fan_valid": flag_fan_valid,
        "landmarks_mp": landmarks_mp_out, "flag_landmarks_mp_valid": flag_mp_valid,
    }
