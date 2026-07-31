"""Repackages an existing one-file-per-frame cache directory (crop_cache,
mica_cache, face_parsing_cache, or landmark_cache) into the new bucket-
container format (see utils/cache_utils.py's module docstring), bucket by
bucket: read every real data file in a b1/b2/ directory, validate it still
decodes correctly, pack the good ones into one b1/b2.zip container, verify
by reopening and re-decoding every packed entry, move any .noface/.unreadable
markers up to their new sentinel filename directly under b1/, then delete the
now-empty b2/ directory.

Pure byte repackaging - never constructs a detector/XSeg/MICA/FAN model,
since every file this script touches already represents a successful (or
sentinel) result from a previous run; nothing here is ever recomputed.

Sharded by literal existing b1 (2-hex-char) directory name, not by hashing -
migration's input is already a fixed set of real directories, so ownership
over them is trivially disjoint without needing shard_of. Anything under
cache_root/dataset/ that isn't shaped like a 2-hex-char directory (e.g. a
prewarm_logs/ directory sitting alongside the bucket dirs) is skipped.

Bucket-at-a-time (write -> verify -> delete), never a global build-
everything-then-delete-everything pass: this bounds the transient file-count
increase during migration itself to about one bucket at a time, which
matters given the project's file quota is already at/over its limit."""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.registry import DEFAULT_DATASETS_YAML, load_datasets_yaml  # noqa: E402
from utils.cache_utils import atomic_write_bucket_entries  # noqa: E402

_HEX_CHARS = set("0123456789abcdef")


def _is_hex_pair_dir(name: str) -> bool:
    return len(name) == 2 and all(c in _HEX_CHARS for c in name)


def _validate_and_decode(data: bytes, extension: str) -> bool:
    """True if `data` still decodes correctly as this cache's payload format.
    Used both to decide whether to include an entry (skip+log corrupted
    bytes rather than bake them into the new container) and, after writing,
    to verify the new container before deleting anything old."""
    try:
        if extension == "png":
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) is not None
        if extension == "npy":
            np.load(io.BytesIO(data))
            return True
        if extension == "npz":
            with np.load(io.BytesIO(data)) as npz:
                for key in npz.files:
                    _ = npz[key]  # force each array to actually be read, not just the header
            return True
    except Exception:
        return False
    raise ValueError(f"unknown extension: {extension!r}")


def migrate_bucket(b1_dir: Path, b2: str, extension: str, dry_run: bool) -> dict[str, int]:
    """Migrates one b1/b2/ directory in place. Returns counts for reporting."""
    b2_dir = b1_dir / b2
    stats = {"migrated": 0, "corrupted_skipped": 0, "sentinels_moved": 0}

    entries: dict[str, bytes] = {}
    source_paths: dict[str, Path] = {}  # entry key -> its old standalone file, for post-verify deletion
    corrupted_paths: list[Path] = []
    sentinel_files: list[Path] = []

    for path in sorted(b2_dir.iterdir()):
        if path.name.startswith(".tmp-"):
            continue  # leftover temp file from an interrupted old-format write - ignore
        if path.suffix == f".{extension}":
            data = path.read_bytes()
            if _validate_and_decode(data, extension):
                entries[path.stem] = data  # path.stem is already exactly this cache's entry_key
                source_paths[path.stem] = path
            else:
                print(f"warning: skipping corrupted entry during migration: {path}")
                corrupted_paths.append(path)
                stats["corrupted_skipped"] += 1
        elif path.suffix in (".noface", ".unreadable"):
            sentinel_files.append(path)
        else:
            print(f"warning: unexpected file in bucket dir, leaving in place: {path}")

    if dry_run:
        print(
            f"[dry-run] {b2_dir}: would migrate {len(entries)} entries, "
            f"skip {stats['corrupted_skipped']} corrupted, move {len(sentinel_files)} sentinels"
        )
        return stats

    container_path = b1_dir / f"{b2}.zip"
    if entries:
        atomic_write_bucket_entries(container_path, entries)

        # Verify by reopening and re-decoding every packed entry - never
        # delete any source file below based on unverified output.
        with zipfile.ZipFile(container_path, "r") as zf:
            for key in entries:
                if not _validate_and_decode(zf.read(key), extension):
                    raise RuntimeError(
                        f"migration verification failed for entry {key!r} in {container_path} - "
                        "leaving source files in place, investigate before rerunning"
                    )
        stats["migrated"] = len(entries)

        # Only now, with the new container written AND verified, remove the
        # old standalone files it replaces.
        for path in source_paths.values():
            path.unlink()

    # Corrupted entries are already fully accounted for (skipped, logged) and
    # have no further use - a future prewarm run will recompute them fresh,
    # the same way a live cache miss already self-heals from corruption.
    for path in corrupted_paths:
        path.unlink()

    for old_sentinel_path in sentinel_files:
        new_path = b1_dir / f"{b2}__{old_sentinel_path.name}"
        os.replace(old_sentinel_path, new_path)
        stats["sentinels_moved"] += 1

    # Only delete the old b2/ dir once its container is written+verified and
    # its sentinels moved out - never before.
    remaining = list(b2_dir.iterdir())
    if remaining:
        print(f"warning: {b2_dir} not empty after migration ({len(remaining)} file(s) left), not deleting")
    else:
        b2_dir.rmdir()

    print(
        f"migrated {b2_dir}: {stats['migrated']} entries, "
        f"{stats['corrupted_skipped']} corrupted skipped, {stats['sentinels_moved']} sentinels moved"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate an existing one-file-per-frame cache directory to the bucket-container format, "
                     "for every dataset in the registry.")
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--datasets_yaml", type=str, default=str(DEFAULT_DATASETS_YAML))
    parser.add_argument("--extension", type=str, required=True, choices=["png", "npy", "npz"],
                         help="crop_cache=png, mica_cache=npy, face_parsing_cache/landmark_cache=npz")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # A misconfigured/empty --cache_root (e.g. an unset shell variable when
    # this is invoked outside its intended srun wrapper) would otherwise fail
    # completely silently below - every dataset directory would just look
    # "not found" and the whole run would quietly report all-zero counts, the
    # exact kind of silent failure this project exists to eliminate. Fail
    # loudly here instead of discovering it from a suspiciously-empty summary.
    cache_root = Path(args.cache_root)
    if not cache_root.is_dir():
        raise SystemExit(
            f"--cache_root {cache_root!r} does not exist or is not a directory - "
            "check it was passed/expanded correctly (e.g. the variable wasn't empty)."
        )

    dataset_names = [entry.name for entry in load_datasets_yaml(args.datasets_yaml)]

    # One global list of (dataset, b1_dir) pairs across every dataset, sharded
    # together rather than per-dataset independently - dataset sizes vary
    # hugely (~500 samples up to ~500,000), so splitting each dataset's
    # buckets separately across shards would leave some shards idle once a
    # small dataset finishes while others are still churning through a much
    # larger one. Positional sharding, not shard_of hashing: this input is
    # already a fixed set of real directories, so ownership is trivially
    # disjoint without recomputing any hash.
    all_b1_dirs: list[tuple[str, Path]] = []
    datasets_found = 0
    for dataset in dataset_names:
        dataset_dir = cache_root / dataset
        if not dataset_dir.is_dir():
            continue  # this cache root has no data for this dataset at all - fine, skip
        datasets_found += 1
        for p in sorted(dataset_dir.iterdir()):
            if p.is_dir() and _is_hex_pair_dir(p.name):
                all_b1_dirs.append((dataset, p))

    if datasets_found == 0:
        print(
            f"warning: {cache_root} exists but none of the {len(dataset_names)} registered "
            "dataset names have a subdirectory under it - double check --cache_root points at "
            "the right cache (nothing below will be migrated)."
        )

    my_b1_dirs = all_b1_dirs[args.shard_index::args.num_shards]

    total = {"migrated": 0, "corrupted_skipped": 0, "sentinels_moved": 0}
    for dataset, b1_dir in my_b1_dirs:
        b2_names = sorted(p.name for p in b1_dir.iterdir() if p.is_dir() and _is_hex_pair_dir(p.name))
        for b2 in b2_names:
            stats = migrate_bucket(b1_dir, b2, args.extension, args.dry_run)
            for k in total:
                total[k] += stats[k]

    print(
        f"DONE! migrated={total['migrated']} corrupted_skipped={total['corrupted_skipped']} "
        f"sentinels_moved={total['sentinels_moved']}"
    )


if __name__ == "__main__":
    main()
