from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from preprocessing.cropping import crop_face


def _cache_key_path(cache_root: Path, dataset: str, sample_id: str, frame_index: int | None) -> Path:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    safe_sample_id = sample_id.replace("/", "_")
    suffix = "img" if frame_index is None else f"f{frame_index:06d}"
    filename = f"{safe_sample_id}__{suffix}"
    return cache_root / dataset / digest[:2] / digest[2:4] / filename


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{uuid.uuid4().hex}-{path.name}")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


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
) -> np.ndarray:
    key_path = _cache_key_path(Path(cache_root), dataset, sample_id, frame_index)
    png_path = key_path.with_suffix(".png")
    noface_path = key_path.with_suffix(".noface")

    if noface_path.exists():
        if on_noface is not None:
            on_noface()
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    if png_path.exists():
        crop = cv2.imread(str(png_path))
        if crop is not None:
            return crop

    # Only reached on an actual cache miss - this is the sole place the detector is
    # constructed (lazily, on first real miss in this worker process), so a fully
    # prewarmed cache never pays the detector-construction cost at all.
    image = load_source_image()
    crop, _ = crop_face(image, get_detector(), scale=scale, image_size=image_size)

    if crop is None:
        _atomic_write_bytes(noface_path, b"")
        if on_noface is not None:
            on_noface()
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise IOError(f"failed to encode crop for {dataset}/{sample_id}/{frame_index}")
    _atomic_write_bytes(png_path, encoded.tobytes())
    return crop