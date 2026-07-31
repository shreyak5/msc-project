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


@dataclass
class PendingFaceParsing:
    """A cache miss whose crop/landmarks/box are already known but whose XSeg mask
    hasn't been computed yet. Returned by get_face_parsing_or_defer instead of
    computing synchronously inline (like compute_face_parsing does), so a caller
    that wants to batch many misses into one XSeg call (e.g. on GPU - see
    uniface.parsing.XSeg.parse_batch) can collect these first and finish them all
    at once via resolve_pending_face_parsing."""
    sample_id: str
    frame_index: int | None
    cropped: np.ndarray
    landmarks_5pt_crop: np.ndarray
    box_crop: np.ndarray


def _visibility_ratio(face_mask: np.ndarray, box_crop: np.ndarray) -> float:
    """mask_area / detected-box-area, per scripts/visible_face_ratio.py - shared by
    every path that turns an already-computed XSeg mask + box into the ratio
    TemporalTransformer's visibility_scores input needs."""
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
    """Returns (cropped, landmarks_5pt_crop, box_crop) - either straight from
    `precomputed_crop` or via a fresh detection - or an "unreadable"/"noface"
    FaceParsingResult when that's not possible. Shared by compute_face_parsing
    (which still calls XSeg synchronously afterward) and get_face_parsing_or_defer
    (which defers the XSeg call), so both acquire the crop identically.

    precomputed_crop: optional (cropped_image, landmarks_5pt_crop, box_crop) -
    the exact tuple crop_face_with_landmarks would produce, for a caller that
    already ran its own detection on this frame (e.g. inference building both
    a model-input crop and this XSeg mask from one detection, instead of
    detecting twice). All three fields are required, not just crop+landmarks:
    box_crop must still be the real per-frame detected box, since it's
    visibility_ratio's denominator - a fixed analytic box would corrupt the
    ratio. Skips load_source_image()/get_detector() entirely when given."""
    if precomputed_crop is not None:
        return precomputed_crop

    try:
        image = load_source_image()
    except Exception as exc:
        return FaceParsingResult(status="unreadable", error=str(exc))

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
    (crop_cache doesn't persist those). See _acquire_crop_or_sentinel for the
    crop-acquisition details, shared with get_face_parsing_or_defer below."""
    acquired = _acquire_crop_or_sentinel(load_source_image, get_detector, crop_scale, image_size, precomputed_crop)
    if isinstance(acquired, FaceParsingResult):
        return acquired
    cropped, landmarks_5pt_crop, box_crop = acquired

    xseg = get_xseg()
    face_mask = xseg.parse(cropped, landmarks=landmarks_5pt_crop).astype(np.float32)
    visibility_ratio = _visibility_ratio(face_mask, box_crop)

    return FaceParsingResult(status="ok", face_mask=face_mask, visibility_ratio=visibility_ratio)


def _check_cache(
    cache_root: str | Path, dataset: str, sample_id: str, frame_index: int | None,
) -> FaceParsingResult | None:
    """Sentinel check -> bucket-container read (self-healing on a corrupted
    entry), with no computation. Returns a resolved FaceParsingResult on a hit
    (including the "previously found unreadable/noface" sentinel cases), or None
    on a true cache miss - shared by get_face_parsing (which computes+writes the
    result synchronously on a miss) and get_face_parsing_or_defer (which defers
    the XSeg call, to batch it later)."""
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
    one just merges on top rather than clobbering it) for "ok". Shared by
    get_face_parsing's own compute path and resolve_pending_face_parsing's
    batched one."""
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


def get_face_parsing_or_defer(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    crop_scale: float,
    image_size: int,
    precomputed_crop: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float, bool] | PendingFaceParsing:
    """Same cache-check-then-acquire-crop flow as get_face_parsing, but stops
    short of calling XSeg on a true cache miss: returns a PendingFaceParsing for
    the caller to resolve later (batched, e.g. on GPU - see
    resolve_pending_face_parsing) instead of computing synchronously inline.

    noface/unreadable outcomes - whether from a sentinel hit or a fresh
    detection attempt - are still resolved and cached immediately here, exactly
    like get_face_parsing: only the "found a real crop, need XSeg" case is
    deferred, since that's the only step this exists to batch. Has no
    on_noface/on_error/get_xseg params (unlike get_face_parsing) since its only
    caller, utils/inference_utils.py's frame-processing pool, doesn't use them."""
    mask_fallback = np.zeros((image_size, image_size), dtype=np.float32)

    cached = _check_cache(cache_root, dataset, sample_id, frame_index)
    if cached is not None:
        return _resolve_result(cached, mask_fallback, None, None)

    acquired = _acquire_crop_or_sentinel(load_source_image, get_detector, crop_scale, image_size, precomputed_crop)
    if isinstance(acquired, FaceParsingResult):
        _write_cache_result(cache_root, dataset, sample_id, frame_index, acquired)
        return _resolve_result(acquired, mask_fallback, None, None)

    cropped, landmarks_5pt_crop, box_crop = acquired
    return PendingFaceParsing(sample_id, frame_index, cropped, landmarks_5pt_crop, box_crop)


def resolve_pending_face_parsing(
    cache_root: str | Path,
    dataset: str,
    pending: list[PendingFaceParsing],
    xseg,
) -> dict[tuple[str, int | None], tuple[np.ndarray, float, bool]]:
    """Batched XSeg pass over every deferred item from get_face_parsing_or_defer:
    one xseg.parse_batch(...) call (uniface.parsing.XSeg.parse_batch) instead of
    N separate ones, then per-item visibility_ratio + cache write - the same two
    steps compute_face_parsing/get_face_parsing would have done individually.
    Returns a dict keyed by (sample_id, frame_index) for the caller to merge back
    into its own per-frame results in whatever order it needs."""
    if not pending:
        return {}

    images = [item.cropped for item in pending]
    landmarks_list = [item.landmarks_5pt_crop for item in pending]
    masks = xseg.parse_batch(images, landmarks_list)

    resolved: dict[tuple[str, int | None], tuple[np.ndarray, float, bool]] = {}
    for item, mask in zip(pending, masks):
        mask = mask.astype(np.float32)
        ratio = _visibility_ratio(mask, item.box_crop)
        result = FaceParsingResult(status="ok", face_mask=mask, visibility_ratio=ratio)
        _write_cache_result(cache_root, dataset, item.sample_id, item.frame_index, result)
        resolved[(item.sample_id, item.frame_index)] = (mask, ratio, True)

    return resolved
