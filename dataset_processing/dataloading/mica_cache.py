from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from dataset_processing.dataloading.cache_utils import atomic_write_bytes, cache_key_path
from preprocessing.cropping import crop_face_arcface

MICA_SHAPE_DIM = 300


def get_mica_shape(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    get_mica: Callable[[], object],
    image_size: int,
    on_noface: Callable[[], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, bool]:
    """Mirrors crop_cache.py's get_cropped_face, caching MICA's predicted
    (300,) FLAME shape params instead of a cropped image - as a .npy file
    (a plain small float vector, no compression/format story needed, unlike a
    PNG crop). Only the shape vector is cached, not the intermediate
    ArcFace-aligned crop used to produce it: that crop is consumed exactly
    once (by MICA's frozen forward pass) and is fully reproducible from the
    original image + detector, so persisting it would just double this
    cache's disk footprint for something nothing else ever reads.

    Unlike get_cropped_face's silent zero-image fallback (tolerable for a
    model *input*), this returns an explicit `valid` flag: this vector is
    used as a loss *target* (model/losses/mica_shape.py), so a missing-face
    fallback must be distinguishable from a real prediction, not silently
    substituted as if it were one - the caller (datasets.py) is expected to
    propagate this as a flag_mica_valid field, gated the same way SMIRK gates
    its own flag_landmarks_fan."""
    key_path = cache_key_path(Path(cache_root), dataset, sample_id, frame_index)
    npy_path = key_path.with_suffix(".npy")
    noface_path = key_path.with_suffix(".noface")
    unreadable_path = key_path.with_suffix(".unreadable")

    fallback = np.zeros(MICA_SHAPE_DIM, dtype=np.float32)

    if unreadable_path.exists():
        if on_error is not None:
            on_error("previously found unreadable")
        return fallback, False

    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return fallback, False

    if npy_path.exists():
        shape = np.load(npy_path)
        return shape, True

    try:
        image = load_source_image()
    except Exception as exc:
        atomic_write_bytes(unreadable_path, b"")
        if on_error is not None:
            on_error(str(exc))
        return fallback, False

    # Only reached on an actual cache miss - this is the sole place MICA is
    # constructed (lazily, on first real miss in this worker process), so a
    # fully prewarmed cache never pays its construction cost at all.
    aligned_crop, _ = crop_face_arcface(image, get_detector(), image_size=image_size)

    if aligned_crop is None:
        atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return fallback, False

    mica = get_mica()
    device = next(mica.parameters()).device
    crop_tensor = torch.from_numpy(aligned_crop).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        shape = mica(crop_tensor)[0].cpu().numpy().astype(np.float32)

    buffer = io.BytesIO()
    np.save(buffer, shape)
    atomic_write_bytes(npy_path, buffer.getvalue())
    return shape, True
