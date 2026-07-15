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
