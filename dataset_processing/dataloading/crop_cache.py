from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from dataset_processing.dataloading.cache_utils import atomic_write_bytes, cache_key_path
from preprocessing.cropping import crop_face


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
    key_path = cache_key_path(Path(cache_root), dataset, sample_id, frame_index)
    png_path = key_path.with_suffix(".png")
    noface_path = key_path.with_suffix(".noface")
    unreadable_path = key_path.with_suffix(".unreadable")

    if unreadable_path.exists():
        if on_error is not None:
            on_error("previously found unreadable")
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    if png_path.exists():
        crop = cv2.imread(str(png_path))
        if crop is not None:
            return crop

    # A source frame that genuinely can't be read (e.g. a corrupted/missing individual
    # frame image) shouldn't crash the whole dataset/training run over one bad file -
    # degrade the same way "no face detected" does, but track it separately (a distinct
    # sentinel + callback) so it stays distinguishable from a legitimate no-face case.
    try:
        image = load_source_image()
    except Exception as exc:
        atomic_write_bytes(unreadable_path, b"")
        if on_error is not None:
            on_error(str(exc))
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Only reached on an actual cache miss - this is the sole place the detector is
    # constructed (lazily, on first real miss in this worker process), so a fully
    # prewarmed cache never pays the detector-construction cost at all.
    crop, _ = crop_face(image, get_detector(), scale=scale, image_size=image_size)

    if crop is None:
        atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise IOError(f"failed to encode crop for {dataset}/{sample_id}/{frame_index}")
    atomic_write_bytes(png_path, encoded.tobytes())
    return crop