from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from preprocessing.cropping import crop_face_arcface
from utils.cache_utils import (
    atomic_write_bytes,
    bucket_container_path,
    entry_key,
    read_bucket_entry,
    sentinel_path,
    write_bucket_entry,
)

MICA_SHAPE_DIM = 300


@dataclass
class MicaShapeResult:
    """compute_mica_shape's return value - a tagged outcome rather than a
    bare (shape, valid) pair, since the caller needs to distinguish "computed
    a real shape" from the two negative outcomes (and, for "unreadable", the
    actual error) to know what to persist and which callback to fire."""
    status: Literal["ok", "noface", "unreadable"]
    shape: np.ndarray | None = None
    error: str | None = None


def compute_mica_shape(
    load_source_image: Callable[[], np.ndarray],
    get_detector: Callable[[], object],
    get_mica: Callable[[], object],
    image_size: int,
) -> MicaShapeResult:
    """Pure computation - no cache I/O, no sentinel writes, no callbacks.
    Callers (get_mica_shape below, and prewarm's batched per-bucket loop)
    translate the returned status into sentinel writes / on_noface / on_error
    notifications and, on "ok", a bucket container write.

    Mirrors crop_cache.py's compute_cropped_face, computing MICA's predicted
    (300,) FLAME shape params instead of a cropped image. Only the shape
    vector is returned/cached, not the intermediate ArcFace-aligned crop used
    to produce it: that crop is consumed exactly once (by MICA's frozen
    forward pass) and is fully reproducible from the original image +
    detector, so persisting it would just double this cache's disk footprint
    for something nothing else ever reads."""
    try:
        image = load_source_image()
    except Exception as exc:
        return MicaShapeResult(status="unreadable", error=str(exc))

    aligned_crop, _ = crop_face_arcface(image, get_detector(), image_size=image_size)

    if aligned_crop is None:
        return MicaShapeResult(status="noface")

    mica = get_mica()
    device = next(mica.parameters()).device
    crop_tensor = torch.from_numpy(aligned_crop).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        shape = mica(crop_tensor)[0].cpu().numpy().astype(np.float32)

    return MicaShapeResult(status="ok", shape=shape)


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
    """Orchestration layer around compute_mica_shape: sentinel check ->
    bucket-container read (self-healing on a corrupted entry, same as
    before) -> on a true miss, compute (unlocked, so concurrent misses on
    the same bucket compute in parallel) -> on success, take the bucket's
    write lock only for the final persist.

    Unlike get_cropped_face's silent zero-image fallback (tolerable for a
    model *input*), this returns an explicit `valid` flag: this vector is
    used as a loss *target* (model/losses/mica_shape.py), so a missing-face
    fallback must be distinguishable from a real prediction, not silently
    substituted as if it were one - the caller (datasets.py) is expected to
    propagate this as a flag_mica_valid field, gated the same way SMIRK gates
    its own flag_landmarks_fan."""
    fallback = np.zeros(MICA_SHAPE_DIM, dtype=np.float32)

    unreadable_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "unreadable")
    if unreadable_path.exists():
        if on_error is not None:
            on_error("previously found unreadable")
        return fallback, False

    noface_path = sentinel_path(cache_root, dataset, sample_id, frame_index, "noface")
    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return fallback, False

    container_path = bucket_container_path(cache_root, dataset, sample_id)
    key = entry_key(sample_id, frame_index)
    try:
        cached_bytes = read_bucket_entry(container_path, key)
        if cached_bytes is not None:
            return np.load(io.BytesIO(cached_bytes)), True
    except Exception as exc:
        # Covers both a corrupted whole container and a corrupted individual
        # entry's bytes - treat like a cache miss and recompute live below,
        # self-healing rather than crashing (see face_parsing_cache.py's
        # get_face_parsing for the full reasoning, including the accepted
        # coarser-grained blast radius of one corrupted bucket file).
        print(f"warning: corrupted cache entry {key} in {container_path}, recomputing: {exc}")

    # Only reached on an actual cache miss - MICA is lazily constructed
    # inside compute_mica_shape, the sole place this cache pays its
    # construction cost. Deliberately unlocked: two different workers both
    # missing on this bucket at the same time both run this (possibly slow)
    # computation fully in parallel rather than serializing on each other.
    result = compute_mica_shape(load_source_image, get_detector, get_mica, image_size)

    if result.status == "unreadable":
        atomic_write_bytes(unreadable_path, b"")
        if on_error is not None:
            on_error(result.error)
        return fallback, False

    if result.status == "noface":
        atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return fallback, False

    buffer = io.BytesIO()
    np.save(buffer, result.shape)
    # Only step that takes the bucket's write lock - a fresh re-read-merge-
    # write under lock, so a second writer arriving right after another one
    # just merges on top of whatever's already persisted, never clobbers it.
    write_bucket_entry(cache_root, dataset, sample_id, key, buffer.getvalue())

    return result.shape, True
