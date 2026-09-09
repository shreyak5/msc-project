from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from preprocessing.cropping import crop_face_with_landmarks
from utils.cache_utils import (
    atomic_write_bytes,
    bucket_container_path,
    entry_key,
    read_bucket_entry,
    sentinel_path,
    write_bucket_entry,
)


@dataclass
class FaceParsingResult:
    """compute_face_parsing's return value - a tagged outcome rather than a
    bare (mask, ratio) pair, since the caller needs to distinguish "computed
    a real result" from the two negative outcomes (and, for "unreadable",
    the actual error) to know what to persist and which callback to fire."""
    status: Literal["ok", "noface", "unreadable"]
    face_mask: np.ndarray | None = None
    visibility_ratio: float | None = None
    error: str | None = None


def visibility_ratio_from_mask_and_box(face_mask: np.ndarray, box_crop: np.ndarray) -> float:
    """mask_area / detected-box-area, per scripts/visible_face_ratio.py - shared by
    every path that turns an already-computed XSeg mask + box into the ratio
    TemporalTransformer's visibility_scores input needs (this module's own
    compute_face_parsing, and utils/inference_utils.py's uncached batched pool)."""
    box_area = max(0.0, box_crop[2] - box_crop[0]) * max(0.0, box_crop[3] - box_crop[1])
    mask_area = float(np.count_nonzero(face_mask > 0.5))
    return mask_area / box_area if box_area > 0 else 0.0


def _acquire_crop_or_sentinel(
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    crop_scale: float,
    image_size: int,
    precomputed_crop: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | FaceParsingResult:
    if precomputed_crop is not None:
        return precomputed_crop

    try:
        image = load_source_image()
    except Exception as exc:
        return FaceParsingResult(status="unreadable", error=str(exc))
    if image is None:
        # cv2.imread returns None (rather than raising) for a missing/corrupt/
        # zero-byte file - the except above never catches that, so it needs
        # its own check to stay on the "unreadable" degrade path.
        return FaceParsingResult(status="unreadable", error="load_source_image returned None")

    cropped, _tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
        image, get_detector(), scale=crop_scale, image_size=image_size,
    )
    if cropped is None:
        return FaceParsingResult(status="noface")

    return cropped, landmarks_5pt_crop, box_crop


def compute_face_parsing(
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    get_xseg: Callable[[], object],
    crop_scale: float,
    image_size: int,
    precomputed_crop: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> FaceParsingResult:
    acquired = _acquire_crop_or_sentinel(load_source_image, get_detector, crop_scale, image_size, precomputed_crop)
    if isinstance(acquired, FaceParsingResult):
        return acquired
    cropped, landmarks_5pt_crop, box_crop = acquired

    xseg = get_xseg()
    face_mask = xseg.parse(cropped, landmarks=landmarks_5pt_crop).astype(np.float32)
    visibility_ratio = visibility_ratio_from_mask_and_box(face_mask, box_crop)

    return FaceParsingResult(status="ok", face_mask=face_mask, visibility_ratio=visibility_ratio)


def _check_cache(
    cache_root: str | Path, dataset: str, sample_id: str, frame_index: int | None,
) -> FaceParsingResult | None:
    """Sentinel check -> bucket-container read (self-healing on a corrupted
    entry), with no computation. Returns a resolved FaceParsingResult on a hit
    (including the "previously found unreadable/noface" sentinel cases), or None
    on a true cache miss, for get_face_parsing to compute+write."""
    unreadable_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "unreadable")
    if unreadable_path.exists():
        return FaceParsingResult(status="unreadable", error="previously found unreadable")

    noface_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "noface")
    if noface_path.exists():
        return FaceParsingResult(status="noface")

    container_path = bucket_container_path(cache_root, dataset, sample_id)
    key = entry_key(sample_id, frame_index)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            cached = np.load(io.BytesIO(cached_bytes))
            return FaceParsingResult(
                status="ok", face_mask=cached["face_mask"], visibility_ratio=float(cached["visibility_ratio"]),
            )
    except Exception as exc:
        # Covers both a corrupted whole container (read_bucket_entry raises
        # zipfile.BadZipFile) and a corrupted individual entry's bytes
        # (np.load raises) - either way, treat like a cache miss and
        # recompute live, rewriting the bucket - self-healing rather than
        # crashing the whole training job over one bad entry. Note this can
        # only reclaim the one entry being requested right now - any other
        # entries that shared this same corrupted bucket file are lost until
        # each is separately re-requested.
        print(f"warning: corrupted cache entry {key} in {container_path}, recomputing: {exc}")

    return None


def _write_cache_result(
    cache_root: str | Path, dataset: str, sample_id: str, frame_index: int | None, result: FaceParsingResult,
) -> None:
    """Persists a freshly computed FaceParsingResult - sentinel write for
    noface/unreadable, bucket-container write (under the bucket's write lock,
    fresh re-read-merge-write, so a second writer arriving right after another
    one just merges on top rather than clobbering it) for "ok"."""
    if result.status == "unreadable":
        atomic_write_bytes(sentinel_path(cache_root, dataset, sample_id, frame_index, "unreadable"), b"")
        return
    if result.status == "noface":
        atomic_write_bytes(sentinel_path(cache_root, dataset, sample_id, frame_index, "noface"), b"")
        return

    buffer = io.BytesIO()
    np.savez(buffer, face_mask=result.face_mask, visibility_ratio=np.float32(result.visibility_ratio))
    key = entry_key(sample_id, frame_index)
    write_bucket_entry(cache_root, dataset, sample_id, key, buffer.getvalue())


def _resolve_result(
    result: FaceParsingResult,
    mask_fallback: np.ndarray,
    on_noface: Callable[[], None] | None,
    on_error: Callable[[str], None] | None,
) -> tuple[np.ndarray, float, bool]:
    """FaceParsingResult -> (face_mask, visibility_ratio, valid), firing
    on_noface/on_error for the two negative outcomes - the exact translation
    get_face_parsing always applied, whether the result came from a cache hit or
    a fresh compute."""
    if result.status == "unreadable":
        if on_error is not None:
            on_error(result.error)
        return mask_fallback, 0.0, False
    if result.status == "noface":
        if on_noface is not None:
            on_noface()
        return mask_fallback, 0.0, False
    return result.face_mask, result.visibility_ratio, True


def get_face_parsing(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    get_xseg: Callable[[], object],
    crop_scale: float,
    image_size: int,
    on_noface: Callable[[], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    precomputed_crop: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float, bool]:
    mask_fallback = np.zeros((image_size, image_size), dtype=np.float32)

    cached = _check_cache(cache_root, dataset, sample_id, frame_index)
    if cached is not None:
        return _resolve_result(cached, mask_fallback, on_noface, on_error)

    # Only reached on an actual cache miss - XSeg/detector are lazily
    # constructed inside compute_face_parsing, the sole place this cache
    # pays their construction cost.
    result = compute_face_parsing(
        load_source_image, get_detector, get_xseg, crop_scale, image_size, precomputed_crop,
    )
    _write_cache_result(cache_root, dataset, sample_id, frame_index, result)
    return _resolve_result(result, mask_fallback, on_noface, on_error)


