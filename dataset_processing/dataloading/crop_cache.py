from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from preprocessing.cropping import crop_face
from utils.cache_utils import (
    atomic_write_bytes,
    bucket_container_path,
    entry_key,
    read_bucket_entry,
    sentinel_path,
    write_bucket_entry,
)


@dataclass
class CropResult:
    """compute_cropped_face's return value - a tagged outcome rather than a
    bare crop array, since the caller needs to distinguish "computed a real
    crop" from the two negative outcomes (and, for "unreadable", the actual
    error) to know what to persist and which callback to fire."""
    status: Literal["ok", "noface", "unreadable"]
    crop: np.ndarray | None = None
    error: str | None = None


def compute_cropped_face(
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    scale: float,
    image_size: int,
) -> CropResult:
    try:
        image = load_source_image()
    except Exception as exc:
        return CropResult(status="unreadable", error=str(exc))
    if image is None:
        # cv2.imread returns None (rather than raising) for a missing/corrupt/
        # zero-byte file - the except above never catches that, so it needs
        # its own check to stay on the "unreadable" degrade path.
        return CropResult(status="unreadable", error="load_source_image returned None")

    crop, _ = crop_face(image, get_detector(), scale=scale, image_size=image_size)

    if crop is None:
        return CropResult(status="noface")

    return CropResult(status="ok", crop=crop)


def get_cropped_face(
    cache_root: str | Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    scale: float,
    image_size: int,
    on_noface: Callable[[], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> np.ndarray:
    fallback = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    unreadable_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "unreadable")
    if unreadable_path.exists():
        if on_error is not None:
            on_error("previously found unreadable")
        return fallback

    noface_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "noface")
    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return fallback

    container_path = bucket_container_path(cache_root, dataset, sample_id)
    key = entry_key(sample_id, frame_index)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            crop = cv2.imdecode(np.frombuffer(cached_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                raise ValueError("cv2.imdecode returned None for cached crop bytes")
            return crop
    except Exception as exc:
        # Covers a corrupted whole container, a corrupted individual entry's
        # bytes, and a decode failure on otherwise-intact bytes - all treated
        # like a cache miss and recomputed live below, self-healing rather
        # than crashing (see face_parsing_cache.py's get_face_parsing for the
        # full reasoning, including the accepted coarser-grained blast radius
        # of one corrupted bucket file).
        print(f"warning: corrupted cache entry {key} in {container_path}, recomputing: {exc}")

    # Only reached on an actual cache miss - the detector is lazily
    # constructed inside compute_cropped_face, the sole place this cache
    # pays its construction cost. Deliberately unlocked: two different
    # workers both missing on this bucket at the same time both run this
    # (possibly slow) computation fully in parallel rather than serializing
    # on each other.
    result = compute_cropped_face(load_source_image, get_detector, scale, image_size)

    if result.status == "unreadable":
        atomic_write_bytes(unreadable_path, b"")
        if on_error is not None:
            on_error(result.error)
        return fallback

    if result.status == "noface":
        atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return fallback

    ok, encoded = cv2.imencode(".png", result.crop)
    if not ok:
        raise IOError(f"failed to encode crop for {dataset}/{sample_id}/{frame_index}")
    # Only step that takes the bucket's write lock - a fresh re-read-merge-
    # write under lock, so a second writer arriving right after another one
    # just merges on top of whatever's already persisted, never clobbers it.
    write_bucket_entry(cache_root, dataset, sample_id, key, encoded.tobytes())

    return result.crop