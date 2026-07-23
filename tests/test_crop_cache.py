import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.crop_cache import get_cropped_face  # noqa: E402
from utils.cache_utils import bucket_container_path, entry_key, sentinel_path  # noqa: E402


class _NoFaceDetector:
    def __call__(self, image, rgb=False):
        return None


class _FakeDetector:
    """Same RetinaFacePredictor-style (N, 15) convention as the other cache
    modules' tests - crop_face goes through the same _detect_primary_face."""

    def __call__(self, image, rgb=False):
        return np.array(
            [[20, 20, 180, 180, 0.99, 60, 80, 140, 80, 100, 110, 70, 150, 130, 150]], dtype=np.float32,
        )


def test_no_face_writes_sentinel_and_returns_fallback(tmp_path):
    cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    crop = get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _NoFaceDetector(),
        scale=1.4, image_size=224,
    )

    assert crop.shape == (224, 224, 3)
    assert np.all(crop == 0)
    assert sentinel_path(cache_root, "test_dataset", "s0", None, "noface").exists()


def test_no_face_is_a_cache_hit_on_second_access_without_reconstructing_detector(tmp_path):
    cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    call_count = {"n": 0}

    def get_detector():
        call_count["n"] += 1
        return _NoFaceDetector()

    get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image, get_detector=get_detector,
        scale=1.4, image_size=224,
    )
    get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image, get_detector=get_detector,
        scale=1.4, image_size=224,
    )

    assert call_count["n"] == 1  # second access hits the .noface sentinel, never re-detects


def test_unreadable_source_writes_sentinel_and_returns_fallback(tmp_path):
    cache_root = tmp_path / "face_crop_cache"

    def load_source_image():
        raise IOError("corrupt file")

    errors = []
    crop = get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=load_source_image,
        get_detector=lambda: _NoFaceDetector(),
        scale=1.4, image_size=224,
        on_error=errors.append,
    )

    assert np.all(crop == 0)
    assert sentinel_path(cache_root, "test_dataset", "s0", None, "unreadable").exists()
    assert errors == ["corrupt file"]


def test_success_writes_into_bucket_container(tmp_path):
    cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    crop = get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        scale=1.4, image_size=224,
    )

    assert crop.shape == (224, 224, 3)

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert container_path.exists()
    with zipfile.ZipFile(container_path) as zf:
        assert entry_key("s0", None) in zf.namelist()
    assert not sentinel_path(cache_root, "test_dataset", "s0", None, "noface").exists()
    assert not sentinel_path(cache_root, "test_dataset", "s0", None, "unreadable").exists()

    # A later call must hit the cache (byte-identical PNG round-trip), not re-detect.
    crop2 = get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: (_ for _ in ()).throw(AssertionError("should not reload on a cache hit")),
        get_detector=lambda: (_ for _ in ()).throw(AssertionError("should not redetect on a cache hit")),
        scale=1.4, image_size=224,
    )
    assert np.array_equal(crop2, crop)


def test_corrupted_bucket_container_is_recomputed_not_crashed(tmp_path, capsys):
    """Same self-heal contract as the other 3 caches' equivalent test - a
    corrupted whole bucket container must recompute (only the entry being
    requested), not crash. Also exercises the fix bundled into this
    rewrite: a decode failure (cv2.imdecode returning None) now goes through
    this same warn-and-recompute path instead of silently falling through
    with no warning, as the old cv2.imread-based check used to."""
    cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    crop1 = get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        scale=1.4, image_size=224,
    )

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    container_path.write_bytes(b"\x00\x01\x02")

    crop2 = get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        scale=1.4, image_size=224,
    )

    assert np.array_equal(crop2, crop1)
    assert "corrupted cache entry" in capsys.readouterr().out

    get_cropped_face(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        scale=1.4, image_size=224,
    )
    assert "corrupted cache entry" not in capsys.readouterr().out


def test_bucket_container_holds_multiple_frames_of_the_same_sample(tmp_path):
    cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    for frame_index in range(5):
        get_cropped_face(
            cache_root, "test_dataset", "video0", frame_index,
            load_source_image=lambda: image,
            get_detector=lambda: _FakeDetector(),
            scale=1.4, image_size=224,
        )

    container_path = bucket_container_path(cache_root, "test_dataset", "video0")
    with zipfile.ZipFile(container_path) as zf:
        names = set(zf.namelist())
    assert names == {entry_key("video0", i) for i in range(5)}
