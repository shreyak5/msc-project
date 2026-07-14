from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


def cache_key_path(cache_root: Path, dataset: str, sample_id: str, frame_index: int | None) -> Path:
    """Shared by crop_cache.py and mica_cache.py (and any future per-sample
    disk cache) so cache layouts stay consistent and lookups agree on a
    sample's key regardless of which cache is asking."""
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    safe_sample_id = sample_id.replace("/", "_")
    suffix = "img" if frame_index is None else f"f{frame_index:06d}"
    filename = f"{safe_sample_id}__{suffix}"
    return cache_root / dataset / digest[:2] / digest[2:4] / filename


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{uuid.uuid4().hex}-{path.name}")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)
