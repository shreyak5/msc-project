from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np

from dataset_processing.dataloading.cache_utils import atomic_write_bytes, cache_key_path
from preprocessing.cropping import crop_face_with_landmarks


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
) -> tuple[np.ndarray, float, bool]:
    """Caches XSeg's face-parsing mask and the "visible face region" ratio
    (mask_area / detected-box-area, per scripts/visible_face_ratio.py) together,
    from a single detection + single XSeg call - the mask is model/flame/
    masking.py's missing face-region input (its own docstring flags this gap),
    and the ratio is TemporalTransformer's visibility_scores input (Sec 4.1).
    Both come from the same XSeg output, so caching them separately would
    double the detector+XSeg cost for no benefit.

    Builds its own crop directly (crop_face_with_landmarks, mirroring
    mica_cache.py's own independent-alignment-crop pattern) rather than
    reusing crop_cache's cached crop: RetinaFace inference is deterministic,
    so this produces the pixel-identical crop crop_cache would already have -
    the only cost difference is one extra warp_crop (cheap) versus a cache
    read, and a fresh detection is needed either way for XSeg's landmarks
    (crop_cache doesn't persist those).

    Like mica_cache.get_mica_shape, returns an explicit `valid` flag rather
    than silently falling back to zeros: visibility_ratio is consumed as a
    TT input (not just a rendering aid), so a missing-face fallback must be
    distinguishable from a real 0.0 ratio, matching flag_mica_valid/
    flag_landmarks_*_valid's established contract."""
    key_path = cache_key_path(Path(cache_root), dataset, sample_id, frame_index)
    npz_path = key_path.with_suffix(".npz")
    noface_path = key_path.with_suffix(".noface")
    unreadable_path = key_path.with_suffix(".unreadable")

    mask_fallback = np.zeros((image_size, image_size), dtype=np.float32)

    if unreadable_path.exists():
        if on_error is not None:
            on_error("previously found unreadable")
        return mask_fallback, 0.0, False

    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return mask_fallback, 0.0, False

    if npz_path.exists():
        cached = np.load(npz_path)
        return cached["face_mask"], float(cached["visibility_ratio"]), True

    try:
        image = load_source_image()
    except Exception as exc:
        atomic_write_bytes(unreadable_path, b"")
        if on_error is not None:
            on_error(str(exc))
        return mask_fallback, 0.0, False

    # Only reached on an actual cache miss - XSeg is lazily constructed here,
    # the sole place this cache pays its construction cost.
    cropped, _tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
        image, get_detector(), scale=crop_scale, image_size=image_size,
    )

    if cropped is None:
        atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return mask_fallback, 0.0, False

    xseg = get_xseg()
    face_mask = xseg.parse(cropped, landmarks=landmarks_5pt_crop).astype(np.float32)

    box_area = max(0.0, box_crop[2] - box_crop[0]) * max(0.0, box_crop[3] - box_crop[1])
    mask_area = float(np.count_nonzero(face_mask > 0.5))
    visibility_ratio = mask_area / box_area if box_area > 0 else 0.0

    buffer = io.BytesIO()
    np.savez(buffer, face_mask=face_mask, visibility_ratio=np.float32(visibility_ratio))
    atomic_write_bytes(npz_path, buffer.getvalue())

    return face_mask, visibility_ratio, True
