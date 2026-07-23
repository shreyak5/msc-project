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


def compute_face_parsing(
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    get_xseg: Callable[[], object],
    crop_scale: float,
    image_size: int,
    precomputed_crop: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> FaceParsingResult:
    """Pure computation - no cache I/O, no sentinel writes, no callbacks.
    Callers (get_face_parsing below, and prewarm's batched per-bucket loop)
    translate the returned status into sentinel writes / on_noface / on_error
    notifications and, on "ok", a bucket container write.

    Caches XSeg's face-parsing mask and the "visible face region" ratio
    (mask_area / detected-box-area, per scripts/visible_face_ratio.py) together,
    from a single detection + single XSeg call - the mask is model/flame/
    masking.py's missing face-region input (its own docstring flags this gap),
    and the ratio is TemporalTransformer's visibility_scores input (Sec 4.1).
    Both come from the same XSeg output, so computing them separately would
    double the detector+XSeg cost for no benefit.

    Builds its own crop directly (crop_face_with_landmarks, mirroring
    mica_cache.py's own independent-alignment-crop pattern) rather than
    reusing crop_cache's cached crop: RetinaFace inference is deterministic,
    so this produces the pixel-identical crop crop_cache would already have -
    the only cost difference is one extra warp_crop (cheap) versus a cache
    read, and a fresh detection is needed either way for XSeg's landmarks
    (crop_cache doesn't persist those).

    precomputed_crop: optional (cropped_image, landmarks_5pt_crop, box_crop) -
    the exact tuple crop_face_with_landmarks would produce, for a caller that
    already ran its own detection on this frame (e.g. inference building both
    a model-input crop and this XSeg mask from one detection, instead of
    detecting twice). All three fields are required, not just crop+landmarks:
    box_crop must still be the real per-frame detected box, since it's
    visibility_ratio's denominator below - a fixed analytic box would corrupt
    the ratio. Skips load_source_image()/get_detector() entirely when given."""
    if precomputed_crop is not None:
        cropped, landmarks_5pt_crop, box_crop = precomputed_crop
    else:
        try:
            image = load_source_image()
        except Exception as exc:
            return FaceParsingResult(status="unreadable", error=str(exc))

        cropped, _tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
            image, get_detector(), scale=crop_scale, image_size=image_size,
        )

        if cropped is None:
            return FaceParsingResult(status="noface")

    xseg = get_xseg()
    face_mask = xseg.parse(cropped, landmarks=landmarks_5pt_crop).astype(np.float32)

    box_area = max(0.0, box_crop[2] - box_crop[0]) * max(0.0, box_crop[3] - box_crop[1])
    mask_area = float(np.count_nonzero(face_mask > 0.5))
    visibility_ratio = mask_area / box_area if box_area > 0 else 0.0

    return FaceParsingResult(status="ok", face_mask=face_mask, visibility_ratio=visibility_ratio)


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
    """Orchestration layer around compute_face_parsing: sentinel check ->
    bucket-container read (self-healing on a corrupted entry, same as
    before) -> on a true miss, compute (unlocked, so concurrent misses on
    the same bucket compute in parallel) -> on success, take the bucket's
    write lock only for the final persist.

    Like mica_cache.get_mica_shape, returns an explicit `valid` flag rather
    than silently falling back to zeros: visibility_ratio is consumed as a
    TT input (not just a rendering aid), so a missing-face fallback must be
    distinguishable from a real 0.0 ratio, matching flag_mica_valid/
    flag_landmarks_*_valid's established contract."""
    mask_fallback = np.zeros((image_size, image_size), dtype=np.float32)

    unreadable_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "unreadable")
    if unreadable_path.exists():
        if on_error is not None:
            on_error("previously found unreadable")
        return mask_fallback, 0.0, False

    noface_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "noface")
    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return mask_fallback, 0.0, False

    container_path = bucket_container_path(cache_root, dataset, sample_id)
    key = entry_key(sample_id, frame_index)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            cached = np.load(io.BytesIO(cached_bytes))
            return cached["face_mask"], float(cached["visibility_ratio"]), True
    except Exception as exc:
        # Covers both a corrupted whole container (read_bucket_entry raises
        # zipfile.BadZipFile) and a corrupted individual entry's bytes
        # (np.load raises) - either way, treat like a cache miss and
        # recompute live below, rewriting the bucket - self-healing rather
        # than crashing the whole training job over one bad entry. Note this
        # can only reclaim the one entry being requested right now - any
        # other entries that shared this same corrupted bucket file are lost
        # until each is separately re-requested.
        print(f"warning: corrupted cache entry {key} in {container_path}, recomputing: {exc}")

    # Only reached on an actual cache miss - XSeg/detector are lazily
    # constructed inside compute_face_parsing, the sole place this cache
    # pays their construction cost. Deliberately unlocked: two different
    # workers both missing on this bucket at the same time both run this
    # (possibly slow) computation fully in parallel rather than serializing
    # on each other.
    result = compute_face_parsing(
        load_source_image, get_detector, get_xseg, crop_scale, image_size, precomputed_crop,
    )

    if result.status == "unreadable":
        atomic_write_bytes(unreadable_path, b"")
        if on_error is not None:
            on_error(result.error)
        return mask_fallback, 0.0, False

    if result.status == "noface":
        atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return mask_fallback, 0.0, False

    buffer = io.BytesIO()
    np.savez(buffer, face_mask=result.face_mask, visibility_ratio=np.float32(result.visibility_ratio))
    # Only step that takes the bucket's write lock - a fresh re-read-merge-
    # write under lock, so a second writer arriving right after another one
    # just merges on top of whatever's already persisted, never clobbers it.
    write_bucket_entry(cache_root, dataset, sample_id, key, buffer.getvalue())

    return result.face_mask, result.visibility_ratio, True
