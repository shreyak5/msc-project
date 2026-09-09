from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from dataset_processing.dataloading.crop_cache import get_cropped_face
from model import constants
from model.losses.landmark import NUM_FAN_BOUNDARY_POINTS, NUM_FAN_TOTAL_POINTS
from preprocessing.cropping import get_cropped_face_box
from utils.cache_utils import bucket_container_path, entry_key, read_bucket_entry, sentinel_path, write_bucket_entry
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
    noface = sentinel_path(crop_cache_root, dataset, sample_id, frame_index, "noface")
    unreadable = sentinel_path(crop_cache_root, dataset, sample_id, frame_index, "unreadable")
    return not noface.exists() and not unreadable.exists()


@dataclass
class LandmarkResult:
    status: Literal["ok", "unreadable"]
    landmarks_fan: np.ndarray | None = None
    flag_landmarks_fan_valid: bool | None = None
    landmarks_mp: np.ndarray | None = None
    flag_landmarks_mp_valid: bool | None = None
    error: str | None = None
    # Only populated when compute_landmarks is called with include_fan_full=True
    # (scripts/patch_landmark_cache_fan_full.py) - the full 68-point FAN set,
    # additive to the 17-point landmarks_fan above (see that script's docstring).
    landmarks_fan_full: np.ndarray | None = None
    flag_landmarks_fan_full_valid: bool | None = None


def compute_landmarks(
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
    include_fan_full: bool = False,
) -> LandmarkResult:
    fan_fallback = np.zeros((NUM_FAN_BOUNDARY_POINTS, 2), dtype=np.float32)
    fan_full_fallback = np.zeros((NUM_FAN_TOTAL_POINTS, 2), dtype=np.float32)
    mp_fallback = np.zeros((len(_CURATED_MEDIAPIPE_INDICES), 2), dtype=np.float32)

    try:
        crop = get_cropped_face(
            crop_cache_root, dataset, sample_id, frame_index,
            load_source_image, get_detector, crop_scale, image_size,
        )
    except Exception as exc:
        return LandmarkResult(status="unreadable", error=str(exc))

    if not _crop_has_real_face(crop_cache_root, dataset, sample_id, frame_index):
        return LandmarkResult(
            status="ok",
            landmarks_fan=fan_fallback, flag_landmarks_fan_valid=False,
            landmarks_mp=mp_fallback, flag_landmarks_mp_valid=False,
            landmarks_fan_full=(fan_full_fallback if include_fan_full else None),
            flag_landmarks_fan_full_valid=(False if include_fan_full else None),
        )

    fan_box = get_cropped_face_box(image_size=image_size, scale=crop_scale)
    landmarks_fan, _scores = run_fan(get_fan_predictor(), crop, fan_box)
    if landmarks_fan is None:
        landmarks_fan_out, flag_fan_valid = fan_fallback, False
        landmarks_fan_full_out, flag_fan_full_valid = fan_full_fallback, False
    else:
        landmarks_fan_out = _normalize(landmarks_fan[:NUM_FAN_BOUNDARY_POINTS].astype(np.float32), image_size)
        flag_fan_valid = True
        landmarks_fan_full_out = _normalize(landmarks_fan[:NUM_FAN_TOTAL_POINTS].astype(np.float32), image_size)
        flag_fan_full_valid = True

    landmarks_mp_raw = run_mediapipe(get_mediapipe_detector(), crop)
    if landmarks_mp_raw is None:
        landmarks_mp_out, flag_mp_valid = mp_fallback, False
    else:
        curated = landmarks_mp_raw[_CURATED_MEDIAPIPE_INDICES, :2].astype(np.float32)
        landmarks_mp_out = _normalize(curated, image_size)
        flag_mp_valid = True

    return LandmarkResult(
        status="ok",
        landmarks_fan=landmarks_fan_out, flag_landmarks_fan_valid=flag_fan_valid,
        landmarks_mp=landmarks_mp_out, flag_landmarks_mp_valid=flag_mp_valid,
        landmarks_fan_full=(landmarks_fan_full_out if include_fan_full else None),
        flag_landmarks_fan_full_valid=(flag_fan_full_valid if include_fan_full else None),
    )


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
    fan_fallback = np.zeros((NUM_FAN_BOUNDARY_POINTS, 2), dtype=np.float32)
    mp_fallback = np.zeros((len(_CURATED_MEDIAPIPE_INDICES), 2), dtype=np.float32)

    container_path = bucket_container_path(cache_root, dataset, sample_id)
    key = entry_key(sample_id, frame_index)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            cached = np.load(io.BytesIO(cached_bytes))
            return {
                "landmarks_fan": cached["landmarks_fan"],
                "flag_landmarks_fan_valid": bool(cached["flag_landmarks_fan_valid"]),
                "landmarks_mp": cached["landmarks_mp"],
                "flag_landmarks_mp_valid": bool(cached["flag_landmarks_mp_valid"]),
            }
    except Exception as exc:
        # Covers both a corrupted whole container and a corrupted individual
        # entry's bytes - treat like a cache miss and recompute live below,
        # self-healing rather than crashing (see face_parsing_cache.py's
        # get_face_parsing for the full reasoning, including the accepted
        # coarser-grained blast radius of one corrupted bucket file).
        print(f"warning: corrupted cache entry {key} in {container_path}, recomputing: {exc}")

    # Only reached on an actual cache miss - FAN/MediaPipe (and, transitively,
    # crop_cache's own detector) are lazily constructed inside compute_landmarks.
    # Deliberately unlocked: two different workers both missing on this bucket
    # at the same time both run this (possibly slow) computation fully in
    # parallel rather than serializing on each other.
    result = compute_landmarks(
        dataset, sample_id, frame_index,
        load_source_image, get_detector, get_fan_predictor, get_mediapipe_detector,
        crop_cache_root, crop_scale, image_size,
    )

    if result.status == "unreadable":
        if on_error is not None:
            on_error(result.error)
        return {
            "landmarks_fan": fan_fallback, "flag_landmarks_fan_valid": False,
            "landmarks_mp": mp_fallback, "flag_landmarks_mp_valid": False,
        }

    result_dict = {
        "landmarks_fan": result.landmarks_fan, "flag_landmarks_fan_valid": result.flag_landmarks_fan_valid,
        "landmarks_mp": result.landmarks_mp, "flag_landmarks_mp_valid": result.flag_landmarks_mp_valid,
    }
    buffer = io.BytesIO()
    np.savez(buffer, **result_dict)
    # Only step that takes the bucket's write lock - a fresh re-read-merge-
    # write under lock, so a second writer arriving right after another one
    # just merges on top of whatever's already persisted, never clobbers it.
    write_bucket_entry(cache_root, dataset, sample_id, key, buffer.getvalue())

    return result_dict


def get_landmarks_fan_full(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
) -> tuple[np.ndarray, bool]:
    fallback = np.zeros((NUM_FAN_TOTAL_POINTS, 2), dtype=np.float32)
    container_path = bucket_container_path(cache_root, dataset, sample_id)
    key = entry_key(sample_id, frame_index)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            cached = np.load(io.BytesIO(cached_bytes))
            if "landmarks_fan_full" in cached.files:
                return cached["landmarks_fan_full"], bool(cached["flag_landmarks_fan_full_valid"])
    except Exception as exc:
        print(f"warning: corrupted cache entry {key} in {container_path} reading landmarks_fan_full: {exc}")
        return fallback, False

    print(f"warning: landmarks_fan_full missing for {dataset}/{key} - run scripts/patch_landmark_cache_fan_full.py")
    return fallback, False
