"""Shared bucket-container cache infrastructure for the 4 per-frame disk
caches under dataset_processing/dataloading/ (crop_cache.py, mica_cache.py,
landmark_cache.py, face_parsing_cache.py).

Each cache buckets entries by an MD5 digest of `sample_id` (not frame_index -
every frame of one sample shares one bucket, which is what lets one container
hold an entire video's worth of frames instead of one file per frame). `b1`/
`b2` name the two hex-pair directory levels (byte 0 and byte 1 of the 16-byte
digest, i.e. digest[:2] and digest[2:4]) - there's no "high/low" ordering
significance, just which byte of the digest each one is.

A bucket's successful entries live packed together in one zip container
(b1/b2.zip), read via true single-entry random access (ZipFile.read(name)
seeks straight to that one entry, never touches the rest of the archive) and
written via a full-archive rebuild + atomic os.replace - never an in-place
append, so a concurrent reader always sees either the fully-old or the
fully-new container, never a torn one. `.noface`/`.unreadable` sentinels stay
individual files directly under b1/ (not inside the container), since they're
rare (~0.08% of all entries) and existence-checking them shouldn't require
opening any archive.

bucket_write_lock uses flock (confirmed supported on this project's Lustre
mount) to serialize concurrent read-merge-write of the same container. It's
only ever held around that write - never around the (possibly slow)
computation that produces a new entry - so concurrent misses on the same
bucket still compute fully in parallel and only briefly contend at the cheap,
final persist step."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import uuid
import zipfile
from pathlib import Path
from typing import Iterator, Literal


def hash_digest(sample_id: str) -> str:
    return hashlib.md5(sample_id.encode("utf-8")).hexdigest()


def entry_key(sample_id: str, frame_index: int | None) -> str:
    safe_sample_id = sample_id.replace("/", "_")
    suffix = "img" if frame_index is None else f"f{frame_index:06d}"
    return f"{safe_sample_id}__{suffix}"


def _b1_b2(sample_id: str) -> tuple[str, str]:
    digest = hash_digest(sample_id)
    return digest[:2], digest[2:4]


def bucket_dir(cache_root: Path, dataset: str, sample_id: str) -> Path:
    b1, _ = _b1_b2(sample_id)
    return Path(cache_root) / dataset / b1


def bucket_container_path(cache_root: Path, dataset: str, sample_id: str) -> Path:
    b1, b2 = _b1_b2(sample_id)
    return Path(cache_root) / dataset / b1 / f"{b2}.zip"


def bucket_lock_path(cache_root: Path, dataset: str, sample_id: str) -> Path:
    b1, b2 = _b1_b2(sample_id)
    return Path(cache_root) / dataset / b1 / f"{b2}.lock"


def sentinel_path(
    cache_root: Path,
    dataset: str,
    sample_id: str,
    frame_index: int | None,
    extension: Literal["noface", "unreadable"],
) -> Path:
    b1, b2 = _b1_b2(sample_id)
    key = entry_key(sample_id, frame_index)
    return Path(cache_root) / dataset / b1 / f"{b2}__{key}.{extension}"


def shard_of(sample_id: str, num_shards: int) -> int:
    # hash_digest(...)[:4] is exactly b1+b2 concatenated (2 bytes / 16 bits, 0-65535);
    # int(x, 16) parses that hex string into an actual integer so `% num_shards` works.
    return int(hash_digest(sample_id)[:4], 16) % num_shards


def read_bucket_entry(container_path: Path, key: str) -> bytes | None:
    """None on a genuine miss (no container, or key not in it). A corrupted
    container raises zipfile.BadZipFile - callers keep their existing
    self-heal-and-recompute handling for that, same as today."""
    if not container_path.exists():
        return None
    with zipfile.ZipFile(container_path, "r") as zf:
        try:
            return zf.read(key)
        except KeyError:
            return None


def read_all_bucket_entries(container_path: Path) -> dict[str, bytes]:
    """{} for a not-yet-created bucket. Used by prewarm's per-bucket batch
    flush (to see what's already there before computing what's missing) and
    by write_bucket_entry's locked merge below."""
    if not container_path.exists():
        return {}
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(container_path, "r") as zf:
        for name in zf.namelist():
            entries[name] = zf.read(name)
    return entries


def atomic_write_bucket_entries(container_path: Path, entries: dict[str, bytes]) -> None:
    """Always a fresh archive to a temp path + os.replace - never an in-place
    append, so this is the container-level equivalent of atomic_write_bytes
    below, and gives readers the same torn-write-free guarantee."""
    container_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = container_path.with_name(f".tmp-{uuid.uuid4().hex}-{container_path.name}")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for key, data in entries.items():
            zf.writestr(key, data)
    os.replace(tmp_path, container_path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Unchanged from the pre-existing per-file cache helper - still used
    as-is for sentinel writes, which stay individual small files by design."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{uuid.uuid4().hex}-{path.name}")
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


@contextlib.contextmanager
def bucket_write_lock(cache_root: Path, dataset: str, sample_id: str) -> Iterator[None]:
    lock_path = bucket_lock_path(cache_root, dataset, sample_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def write_bucket_entry(cache_root: Path, dataset: str, sample_id: str, key: str, data: bytes) -> None:
    """Locked read-merge-write of one new entry into sample_id's bucket
    container - the shared final step every get_* function's success path
    uses (see each cache module's compute_*/get_* split). Re-reads the
    container fresh under the lock so a second writer arriving right after a
    first one just merges on top of whatever the first already persisted,
    rather than clobbering it.

    A corrupted pre-existing container (zipfile.BadZipFile) is tolerated here
    rather than propagated: by the time a caller reaches this function it
    already has a freshly-computed, good entry to persist, and this write
    must not be blocked by damage to whatever was there before - this is the
    point where a corrupted bucket actually gets rebuilt. Any other entries
    that container held are lost (the accepted, coarser-grained blast radius
    of bucketing many frames into one file); callers that want visibility
    into that (e.g. prewarm's own logging) should check read_all_bucket_entries
    themselves before calling this, rather than relying on this function to
    report it."""
    container_path = bucket_container_path(cache_root, dataset, sample_id)
    with bucket_write_lock(cache_root, dataset, sample_id):
        try:
            entries = read_all_bucket_entries(container_path)
        except zipfile.BadZipFile:
            entries = {}
        entries[key] = data
        atomic_write_bucket_entries(container_path, entries)


def write_bucket_entries(cache_root: Path, dataset: str, sample_id: str, entries: dict[str, bytes]) -> None:
    """Batched equivalent of write_bucket_entry, for prewarm's per-bucket
    flush: compute everything missing in a bucket first (via compute_*,
    unlocked, potentially many frames), accumulate in memory, then persist
    the whole batch in one locked read-merge-write - not one write per frame.

    `sample_id` only needs to be ANY sample whose bucket this is: prewarm
    groups its owned rows by bucket_container_path before calling this, so
    every key in `entries` and the `sample_id` passed here are already
    guaranteed (by construction) to hash to the same bucket."""
    container_path = bucket_container_path(cache_root, dataset, sample_id)
    with bucket_write_lock(cache_root, dataset, sample_id):
        try:
            existing = read_all_bucket_entries(container_path)
        except zipfile.BadZipFile:
            existing = {}
        existing.update(entries)
        atomic_write_bucket_entries(container_path, existing)
