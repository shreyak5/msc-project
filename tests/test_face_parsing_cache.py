import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.face_parsing_cache import get_face_parsing  # noqa: E402
from utils.cache_utils import bucket_container_path, entry_key, sentinel_path  # noqa: E402


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
    assert sentinel_path(cache_root, "test_dataset", "s0", None, "noface").exists()


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
    assert sentinel_path(cache_root, "test_dataset", "s0", None, "unreadable").exists()
    assert errors == ["corrupt file"]


def test_success_writes_into_bucket_container(tmp_path):
    """Unlike the old one-file-per-frame layout, a successful entry now lands
    inside a shared per-bucket zip container, keyed by entry_key - not its
    own standalone file."""
    cache_root = tmp_path / "face_parsing_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
    )

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert container_path.exists()
    with zipfile.ZipFile(container_path) as zf:
        assert entry_key("s0", None) in zf.namelist()
    # No standalone per-frame file - and no sentinel, since this was a hit.
    assert not sentinel_path(cache_root, "test_dataset", "s0", None, "noface").exists()
    assert not sentinel_path(cache_root, "test_dataset", "s0", None, "unreadable").exists()


def test_corrupted_bucket_container_is_recomputed_not_crashed(tmp_path, capsys):
    """Reproduces the real 2026-07-20 SLURM failure (a cached entry that
    exists on disk but is truncated/corrupted - plausibly a Lustre
    cross-node rename-visibility race under heavy concurrent access from
    many DataLoader worker processes, or a leftover from an earlier
    abruptly-killed job) at the new bucket-container granularity: a whole
    bucket zip corrupted, not just one frame's file. Used to crash with an
    unhandled error from np.load - which, in the real multi-GPU job,
    cascaded into an NCCL collective timeout that killed every other rank
    once the crashed rank's process never returned. It must still be
    treated as a cache miss and self-healed, for the entry being requested."""
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

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert container_path.exists()

    # Simulate the exact failure: a torn/truncated bucket container file.
    container_path.write_bytes(b"\x00\x01\x02")

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

    # The corrupted container was overwritten with a valid one - a THIRD
    # access (post-recovery) must be a clean, unlogged cache hit.
    get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
    )
    assert "corrupted cache entry" not in capsys.readouterr().out


def test_bucket_container_holds_multiple_frames_of_the_same_sample(tmp_path):
    """The whole point of bucketing by sample_id (not frame_index): every
    frame of one video sample shares one container file."""
    cache_root = tmp_path / "face_parsing_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    for frame_index in range(5):
        get_face_parsing(
            cache_root, "test_dataset", "video0", frame_index,
            load_source_image=lambda: image,
            get_detector=lambda: _FakeDetector(),
            get_xseg=lambda: _FakeXSeg(),
            crop_scale=1.4, image_size=224,
        )

    container_path = bucket_container_path(cache_root, "test_dataset", "video0")
    with zipfile.ZipFile(container_path) as zf:
        names = set(zf.namelist())
    assert names == {entry_key("video0", i) for i in range(5)}


def _unused_get_detector():
    raise AssertionError("get_detector should never be called when precomputed_crop is provided")


def _unused_load_source_image():
    raise AssertionError("load_source_image should never be called when precomputed_crop is provided")


def test_precomputed_crop_skips_detection_and_writes_cache(tmp_path):
    """Inference's whole point in passing precomputed_crop is skipping a
    second, redundant detector call on a frame it already detected - assert
    that literally happens (get_detector/load_source_image blow up if
    touched), and that the cache still gets written/read normally."""
    cache_root = tmp_path / "face_parsing_cache"
    cropped = np.zeros((224, 224, 3), dtype=np.uint8)
    landmarks_5pt_crop = np.array([[60, 80], [140, 80], [100, 110], [70, 150], [130, 150]], dtype=np.float32)
    box_crop = np.array([20.0, 20.0, 180.0, 180.0], dtype=np.float32)

    mask, ratio, valid = get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=_unused_load_source_image,
        get_detector=_unused_get_detector,
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
        precomputed_crop=(cropped, landmarks_5pt_crop, box_crop),
    )

    assert valid is True
    assert mask.shape == (224, 224)
    assert np.all(mask == 1.0)  # _FakeXSeg always returns all-ones
    box_area = (box_crop[2] - box_crop[0]) * (box_crop[3] - box_crop[1])
    assert ratio == pytest.approx((224 * 224) / box_area)

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    with zipfile.ZipFile(container_path) as zf:
        assert entry_key("s0", None) in zf.namelist()

    # A later call - even without precomputed_crop - must hit the cache, not re-detect.
    mask2, ratio2, valid2 = get_face_parsing(
        cache_root, "test_dataset", "s0", None,
        load_source_image=_unused_load_source_image,
        get_detector=_unused_get_detector,
        get_xseg=lambda: _UnusedXSeg(),
        crop_scale=1.4, image_size=224,
    )
    assert valid2 is True
    assert ratio2 == pytest.approx(ratio)  # round-tripped through float32 npz storage
    assert np.array_equal(mask2, mask)


def test_precomputed_crop_matches_normal_detection_path(tmp_path):
    """precomputed_crop is purely a perf shortcut (skip a redundant detector
    call the caller already ran), not a different computation - the exact
    (cropped, landmarks, box) a real crop_face_with_landmarks call produces,
    fed back in as precomputed_crop, must give byte-identical output to
    letting get_face_parsing detect it itself. Guards specifically against
    box_crop drifting from the real per-frame detected box (e.g. toward the
    fixed analytic get_cropped_face_box), which would silently corrupt
    visibility_ratio without changing the mask itself."""
    from preprocessing.cropping import crop_face_with_landmarks

    image = np.zeros((256, 256, 3), dtype=np.uint8)

    mask_direct, ratio_direct, valid_direct = get_face_parsing(
        tmp_path / "direct", "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
    )

    cropped, _tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
        image, _FakeDetector(), scale=1.4, image_size=224,
    )
    mask_precomputed, ratio_precomputed, valid_precomputed = get_face_parsing(
        tmp_path / "precomputed", "test_dataset", "s0", None,
        load_source_image=_unused_load_source_image,
        get_detector=_unused_get_detector,
        get_xseg=lambda: _FakeXSeg(),
        crop_scale=1.4, image_size=224,
        precomputed_crop=(cropped, landmarks_5pt_crop, box_crop),
    )

    assert valid_direct is True
    assert valid_precomputed is True
    assert ratio_precomputed == ratio_direct
    assert np.array_equal(mask_precomputed, mask_direct)
