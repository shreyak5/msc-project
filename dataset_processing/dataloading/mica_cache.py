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
    try:
        image = load_source_image()
    except Exception as exc:
        return MicaShapeResult(status="unreadable", error=str(exc))
    if image is None:
        # cv2.imread returns None (rather than raising) for a missing/corrupt/
        # zero-byte file - the except above never catches that, so it needs
        # its own check to stay on the "unreadable" degrade path.
        return MicaShapeResult(status="unreadable", error="load_source_image returned None")

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
