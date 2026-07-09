from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def _final_path(cache_root: Path, dataset: str) -> Path:
    return cache_root / dataset / "frame_counts.json"


def _shard_path(cache_root: Path, dataset: str, shard_index: int) -> Path:
    return cache_root / dataset / "frame_counts_shards" / f"shard_{shard_index}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{uuid.uuid4().hex}-{path.name}")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def load_frame_counts(cache_root: str | Path, dataset: str) -> dict[str, int] | None:
    """Returns the merged sample_id -> frame_count mapping if it's been built (via the
    prewarm script + merge step), else None. Datasets never prewarmed simply have no such
    file - callers should fall back to probing each row's frame count live in that case."""
    path = _final_path(Path(cache_root), dataset)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_shard_frame_counts(
    cache_root: str | Path, dataset: str, shard_index: int, counts: dict[str, int],
) -> None:
    """Each prewarm shard writes only the counts it computed to its own file - no shard
    ever reads-modifies-writes a file shared with other concurrently-running shards, which
    would otherwise race (a later shard's write clobbering an earlier shard's additions)."""
    _atomic_write_json(_shard_path(Path(cache_root), dataset, shard_index), counts)


def merge_frame_count_shards(cache_root: str | Path, dataset: str, num_shards: int) -> dict[str, int]:
    cache_root = Path(cache_root)
    merged: dict[str, int] = {}
    for shard_index in range(num_shards):
        shard_path = _shard_path(cache_root, dataset, shard_index)
        if not shard_path.exists():
            continue
        with open(shard_path) as f:
            merged.update(json.load(f))
    if merged:
        _atomic_write_json(_final_path(cache_root, dataset), merged)
    return merged
