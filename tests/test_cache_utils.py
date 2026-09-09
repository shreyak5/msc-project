import multiprocessing
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_utils import (  # noqa: E402
    bucket_container_path,
    entry_key,
    read_all_bucket_entries,
    write_bucket_entry,
)


def _write_one_frame(cache_root, dataset, sample_id, frame_index):
    """Top-level (picklable) worker for multiprocessing.Process - writes one
    frame's entry into sample_id's bucket through the real locked write path."""
    key = entry_key(sample_id, frame_index)
    payload = f"frame-{frame_index}-payload".encode() * 1000  # non-trivial size
    write_bucket_entry(cache_root, dataset, sample_id, key, payload)


def test_concurrent_writers_to_the_same_bucket_lose_no_entries(tmp_path):
    """Several real OS processes (not threads - needed to exercise
    fcntl.flock across processes) writing DIFFERENT frames of the SAME
    sample (same bucket) concurrently through the locked write path. Asserts
    the final container has every expected entry, byte-exact, and no
    corruption - this is the core correctness property the whole locking
    scheme exists for."""
    cache_root = tmp_path / "cache"
    dataset = "test_dataset"
    sample_id = "concurrent_video"
    num_frames = 16

    processes = [
        multiprocessing.Process(target=_write_one_frame, args=(cache_root, dataset, sample_id, i))
        for i in range(num_frames)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=30)
        assert p.exitcode == 0

    container_path = bucket_container_path(cache_root, dataset, sample_id)
    with zipfile.ZipFile(container_path) as zf:
        assert zf.testzip() is None  # no corrupted member
        names = set(zf.namelist())
        assert names == {entry_key(sample_id, i) for i in range(num_frames)}
        for i in range(num_frames):
            expected = f"frame-{i}-payload".encode() * 1000
            assert zf.read(entry_key(sample_id, i)) == expected


def _write_one_sample(cache_root, dataset, sample_id):
    """Top-level worker for the disjoint-ownership test - each process
    targets a DIFFERENT sample_id (almost certainly a different bucket)."""
    key = entry_key(sample_id, None)
    payload = f"payload-for-{sample_id}".encode()
    write_bucket_entry(cache_root, dataset, sample_id, key, payload)


def test_concurrent_writers_to_different_buckets_do_not_interfere(tmp_path):
    """Disjoint-ownership case: correctness only, not timing (asserting
    parallelism via wall-clock would be flaky in a shared CI/runner
    environment) - just that each sample's own bucket ends up with exactly
    its own entry, nothing lost or cross-contaminated."""
    cache_root = tmp_path / "cache"
    dataset = "test_dataset"
    sample_ids = [f"sample_{i:02d}" for i in range(12)]

    processes = [
        multiprocessing.Process(target=_write_one_sample, args=(cache_root, dataset, sid))
        for sid in sample_ids
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(timeout=30)
        assert p.exitcode == 0

    for sid in sample_ids:
        container_path = bucket_container_path(cache_root, dataset, sid)
        with zipfile.ZipFile(container_path) as zf:
            assert zf.testzip() is None
            key = entry_key(sid, None)
            assert zf.read(key) == f"payload-for-{sid}".encode()


def test_corrupted_bucket_raises_for_callers_to_self_heal(tmp_path):
    cache_root = tmp_path / "cache"
    dataset = "test_dataset"
    sample_id = "s0"

    write_bucket_entry(cache_root, dataset, sample_id, entry_key(sample_id, None), b"real-bytes")
    container_path = bucket_container_path(cache_root, dataset, sample_id)
    container_path.write_bytes(b"not a zip file")

    with pytest.raises(zipfile.BadZipFile):
        read_all_bucket_entries(container_path)

    write_bucket_entry(cache_root, dataset, sample_id, entry_key(sample_id, None), b"recovered-bytes")
    with zipfile.ZipFile(container_path) as zf:
        assert zf.read(entry_key(sample_id, None)) == b"recovered-bytes"
