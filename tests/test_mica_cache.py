import os
import sys
import zipfile

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.mica_cache import MICA_SHAPE_DIM, get_mica_shape  # noqa: E402
from utils.cache_utils import bucket_container_path, entry_key, sentinel_path  # noqa: E402


class _NoFaceDetector:
    def __call__(self, image, rgb=False):
        return None


class _FakeDetector:
    """Same RetinaFacePredictor-style (N, 15) convention as
    test_face_parsing_cache.py's fake - crop_face_arcface goes through the
    same _detect_primary_face/get_face_landmarks helpers."""

    def __call__(self, image, rgb=False):
        return np.array(
            [[20, 20, 180, 180, 0.99, 60, 80, 140, 80, 100, 110, 70, 150, 130, 150]], dtype=np.float32,
        )


class _FakeMica(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, crop_tensor):
        batch = crop_tensor.shape[0]
        return torch.full((batch, MICA_SHAPE_DIM), 0.5, dtype=torch.float32)


def test_no_face_writes_sentinel_and_returns_invalid(tmp_path):
    cache_root = tmp_path / "mica_shape_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    shape, valid = get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _NoFaceDetector(),
        get_mica=lambda: (_ for _ in ()).throw(AssertionError("MICA should never be constructed on a no-face image")),
        image_size=112,
    )

    assert valid is False
    assert shape.shape == (MICA_SHAPE_DIM,)
    assert np.all(shape == 0.0)
    assert sentinel_path(cache_root, "test_dataset", "s0", None, "noface").exists()


def test_no_face_is_a_cache_hit_on_second_access_without_reconstructing_detector(tmp_path):
    cache_root = tmp_path / "mica_shape_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    call_count = {"n": 0}

    def get_detector():
        call_count["n"] += 1
        return _NoFaceDetector()

    def unused_get_mica():
        raise AssertionError("MICA should never be constructed on a no-face image")

    get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image, get_detector=get_detector,
        get_mica=unused_get_mica, image_size=112,
    )
    get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image, get_detector=get_detector,
        get_mica=unused_get_mica, image_size=112,
    )

    assert call_count["n"] == 1  # second access hits the .noface sentinel, never re-detects


def test_unreadable_source_writes_sentinel_and_returns_invalid(tmp_path):
    cache_root = tmp_path / "mica_shape_cache"

    def load_source_image():
        raise IOError("corrupt file")

    errors = []
    shape, valid = get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=load_source_image,
        get_detector=lambda: _NoFaceDetector(),
        get_mica=lambda: _FakeMica(),
        image_size=112,
        on_error=errors.append,
    )

    assert valid is False
    assert sentinel_path(cache_root, "test_dataset", "s0", None, "unreadable").exists()
    assert errors == ["corrupt file"]


def test_success_writes_into_bucket_container(tmp_path):
    cache_root = tmp_path / "mica_shape_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    shape, valid = get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_mica=lambda: _FakeMica(),
        image_size=112,
    )

    assert valid is True
    assert shape.shape == (MICA_SHAPE_DIM,)
    assert np.allclose(shape, 0.5)

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert container_path.exists()
    with zipfile.ZipFile(container_path) as zf:
        assert entry_key("s0", None) in zf.namelist()
    assert not sentinel_path(cache_root, "test_dataset", "s0", None, "noface").exists()
    assert not sentinel_path(cache_root, "test_dataset", "s0", None, "unreadable").exists()


def test_corrupted_bucket_container_is_recomputed_not_crashed(tmp_path, capsys):
    """Same self-heal contract as face_parsing_cache.py's equivalent test -
    a corrupted whole bucket container must recompute (only the entry being
    requested), not crash."""
    cache_root = tmp_path / "mica_shape_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    shape1, valid1 = get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_mica=lambda: _FakeMica(),
        image_size=112,
    )
    assert valid1 is True

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    container_path.write_bytes(b"\x00\x01\x02")

    shape2, valid2 = get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_mica=lambda: _FakeMica(),
        image_size=112,
    )

    assert valid2 is True
    assert np.array_equal(shape2, shape1)
    assert "corrupted cache entry" in capsys.readouterr().out

    get_mica_shape(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_mica=lambda: _FakeMica(),
        image_size=112,
    )
    assert "corrupted cache entry" not in capsys.readouterr().out


def test_bucket_container_holds_multiple_frames_of_the_same_sample(tmp_path):
    cache_root = tmp_path / "mica_shape_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    for frame_index in range(5):
        get_mica_shape(
            cache_root, "test_dataset", "video0", frame_index,
            load_source_image=lambda: image,
            get_detector=lambda: _FakeDetector(),
            get_mica=lambda: _FakeMica(),
            image_size=112,
        )

    container_path = bucket_container_path(cache_root, "test_dataset", "video0")
    with zipfile.ZipFile(container_path) as zf:
        names = set(zf.namelist())
    assert names == {entry_key("video0", i) for i in range(5)}
