import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.cache_utils import cache_key_path  # noqa: E402
from dataset_processing.dataloading.face_parsing_cache import get_face_parsing  # noqa: E402


class _NoFaceDetector:
    def __call__(self, image, rgb=False):
        return None


class _UnusedXSeg:
    def parse(self, image, landmarks=None):
        raise AssertionError("XSeg should never be constructed/called on a no-face image")


class _FakeDetector:
    """Returns one detection: box(4) + score(1) + 5-point landmarks(10), matching
    RetinaFacePredictor's (N, 15) convention (preprocessing/cropping.py's
    _detect_primary_face) - enough for a real cache write, not a no-face path."""

    def __call__(self, image, rgb=False):
        return np.array(
            [[20, 20, 180, 180, 0.99, 60, 80, 140, 80, 100, 110, 70, 150, 130, 150]], dtype=np.float32,
        )


class _FakeXSeg:
    def parse(self, image, landmarks=None):
        return np.ones((224, 224), dtype=np.float32)


def test_no_face_writes_sentinel_and_returns_invalid(tmp_path):
    cache_root = tmp_path / "face_parsing_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    mask, ratio, valid = get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _NoFaceDetector(),
        get_xseg=lambda: _UnusedXSeg(),
        crop_scale=1.4, image_size=224,
    )

    assert valid is False
    assert ratio == 0.0
    assert mask.shape == (224, 224)
    assert np.all(mask == 0.0)
    key_path = cache_key_path(cache_root, "test_dataset", "s0", None)
    assert key_path.with_suffix(".noface").exists()


def test_no_face_is_a_cache_hit_on_second_access_without_reconstructing_detector(tmp_path):
    cache_root = tmp_path / "face_parsing_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    call_count = {"n": 0}

    def get_detector():
        call_count["n"] += 1
        return _NoFaceDetector()

    get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image, get_detector=get_detector,
        get_xseg=lambda: _UnusedXSeg(), crop_scale=1.4, image_size=224,
    )
    get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image, get_detector=get_detector,
        get_xseg=lambda: _UnusedXSeg(), crop_scale=1.4, image_size=224,
    )

    assert call_count["n"] == 1  # second access hits the .noface sentinel, never re-detects


def test_unreadable_source_writes_sentinel_and_returns_invalid(tmp_path):
    cache_root = tmp_path / "face_parsing_cache"

    def load_source_image():
        raise IOError("corrupt file")

    errors = []
    mask, ratio, valid = get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=load_source_image,
        get_detector=lambda: _NoFaceDetector(),
        get_xseg=lambda: _UnusedXSeg(),
        crop_scale=1.4, image_size=224,
        on_error=errors.append,
    )

    assert valid is False
    assert ratio == 0.0
    key_path = cache_key_path(cache_root, "test_dataset", "s0", None)
    assert key_path.with_suffix(".unreadable").exists()
    assert errors == ["corrupt file"]


def test_corrupted_cache_entry_is_recomputed_not_crashed(tmp_path, capsys):
    """Reproduces the real 2026-07-20 SLURM failure: a cached .npz entry that
    exists on disk but is truncated/corrupted (plausibly a Lustre cross-node
    rename-visibility race under heavy concurrent access from many DataLoader
    worker processes, or a leftover from an earlier abruptly-killed job) used
    to crash with an unhandled EOFError from np.load - which, in the real
    multi-GPU job, cascaded into an NCCL collective timeout that killed every
    other rank once the crashed rank's process never returned. It must now be
    treated as a cache miss and self-healed instead."""
    cache_root = tmp_path / "face_parsing_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    mask1, ratio1, valid1 = get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
    )
    assert valid1 is True

    key_path = cache_key_path(cache_root, "test_dataset", "s0", None)
    npz_path = key_path.with_suffix(".npz")
    assert npz_path.exists()

    # Simulate the exact failure: a torn/truncated file at the cache path.
    npz_path.write_bytes(b"\x00\x01\x02")

    mask2, ratio2, valid2 = get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
    )

    assert valid2 is True
    assert ratio2 == ratio1
    assert np.array_equal(mask2, mask1)
    assert "corrupted cache entry" in capsys.readouterr().out

    # The corrupted entry was overwritten with a valid one - a THIRD access
    # (post-recovery) must be a clean, unlogged cache hit, not another warning.
    get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
    )
    assert "corrupted cache entry" not in capsys.readouterr().out
