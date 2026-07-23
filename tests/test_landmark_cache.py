import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset_processing.dataloading.landmark_cache as landmark_cache_module  # noqa: E402
from dataset_processing.dataloading.landmark_cache import NUM_FAN_BOUNDARY_POINTS, get_landmarks  # noqa: E402
from utils.cache_utils import bucket_container_path, entry_key  # noqa: E402


class _NoFaceDetector:
    def __call__(self, image, rgb=False):
        return None


class _FakeDetector:
    """Same RetinaFacePredictor-style (N, 15) convention as the other cache
    modules' tests - feeds into crop_cache.get_cropped_face, which this
    module calls internally to get its input crop."""

    def __call__(self, image, rgb=False):
        return np.array(
            [[20, 20, 180, 180, 0.99, 60, 80, 140, 80, 100, 110, 70, 150, 130, 150]], dtype=np.float32,
        )


class _FakeFanPredictor:
    def __call__(self, image, box, rgb=False):
        landmarks = np.zeros((1, 68, 2), dtype=np.float32)
        for i in range(68):
            landmarks[0, i] = [50.0 + i, 60.0 + i]
        scores = np.ones((1, 68), dtype=np.float32)
        return landmarks, scores


class _FakeLandmarkPoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeMediapipeResult:
    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks


class _FakeMediapipeDetector:
    def detect(self, mp_image):
        landmarks = [_FakeLandmarkPoint(0.5, 0.5, 0.0) for _ in range(478)]
        return _FakeMediapipeResult([landmarks])


def test_no_real_face_persists_all_invalid_result_not_a_sentinel(tmp_path):
    """landmark_cache has no .noface/.unreadable sentinel of its own - a
    genuine no-face crop still gets persisted as a regular bucket entry with
    both flags False, exactly like the pre-existing behavior."""
    cache_root = tmp_path / "landmark_cache"
    crop_cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result = get_landmarks(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _NoFaceDetector(),
        get_fan_predictor=lambda: (_ for _ in ()).throw(AssertionError("FAN should never run on a no-face crop")),
        get_mediapipe_detector=lambda: (_ for _ in ()).throw(AssertionError("MediaPipe should never run on a no-face crop")),
        crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
    )

    assert result["flag_landmarks_fan_valid"] is False
    assert result["flag_landmarks_mp_valid"] is False
    assert result["landmarks_fan"].shape == (NUM_FAN_BOUNDARY_POINTS, 2)
    assert np.all(result["landmarks_fan"] == 0.0)

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert container_path.exists()
    with zipfile.ZipFile(container_path) as zf:
        assert entry_key("s0", None) in zf.namelist()


def test_no_real_face_is_a_cache_hit_on_second_access_without_reconstructing_detector(tmp_path):
    cache_root = tmp_path / "landmark_cache"
    crop_cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    call_count = {"n": 0}

    def get_detector():
        call_count["n"] += 1
        return _NoFaceDetector()

    def unused(*args, **kwargs):
        raise AssertionError("should not run on a cache hit")

    for _ in range(2):
        get_landmarks(
            cache_root, "test_dataset", "s0", None,
            load_source_image=lambda: image, get_detector=get_detector,
            get_fan_predictor=unused, get_mediapipe_detector=unused,
            crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
        )

    assert call_count["n"] == 1  # second access hits landmark_cache's own bucket entry


def test_unreadable_crop_fetch_is_not_persisted(tmp_path, monkeypatch):
    """The one outcome distinct from the other 3 caches: get_cropped_face
    raising (an exceptional/defensive case, not a genuine no-face frame) must
    NOT be written to landmark_cache's own bucket - the next access should
    retry crop_cache rather than replay a cached transient failure."""
    cache_root = tmp_path / "landmark_cache"
    crop_cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    call_count = {"n": 0}

    def raising_get_cropped_face(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("simulated crop_cache internal failure")

    monkeypatch.setattr(landmark_cache_module, "get_cropped_face", raising_get_cropped_face)

    errors = []
    for _ in range(2):
        result = get_landmarks(
            cache_root, "test_dataset", "s0", None,
            load_source_image=lambda: image,
            get_detector=lambda: _NoFaceDetector(),
            get_fan_predictor=lambda: (_ for _ in ()).throw(AssertionError("unused")),
            get_mediapipe_detector=lambda: (_ for _ in ()).throw(AssertionError("unused")),
            crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
            on_error=errors.append,
        )
        assert result["flag_landmarks_fan_valid"] is False
        assert result["flag_landmarks_mp_valid"] is False

    assert errors == ["simulated crop_cache internal failure"] * 2
    assert call_count["n"] == 2  # retried both times - nothing was persisted
    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert not container_path.exists()


def test_success_writes_into_bucket_container(tmp_path):
    cache_root = tmp_path / "landmark_cache"
    crop_cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    result = get_landmarks(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_fan_predictor=lambda: _FakeFanPredictor(),
        get_mediapipe_detector=lambda: _FakeMediapipeDetector(),
        crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
    )

    assert result["flag_landmarks_fan_valid"] is True
    assert result["flag_landmarks_mp_valid"] is True
    assert result["landmarks_fan"].shape == (NUM_FAN_BOUNDARY_POINTS, 2)
    assert result["landmarks_mp"].shape == (105, 2)

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    assert container_path.exists()
    with zipfile.ZipFile(container_path) as zf:
        assert entry_key("s0", None) in zf.namelist()


def test_corrupted_bucket_container_is_recomputed_not_crashed(tmp_path, capsys):
    cache_root = tmp_path / "landmark_cache"
    crop_cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    result1 = get_landmarks(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_fan_predictor=lambda: _FakeFanPredictor(),
        get_mediapipe_detector=lambda: _FakeMediapipeDetector(),
        crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
    )

    container_path = bucket_container_path(cache_root, "test_dataset", "s0")
    container_path.write_bytes(b"\x00\x01\x02")

    result2 = get_landmarks(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_fan_predictor=lambda: _FakeFanPredictor(),
        get_mediapipe_detector=lambda: _FakeMediapipeDetector(),
        crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
    )

    assert np.array_equal(result2["landmarks_fan"], result1["landmarks_fan"])
    assert np.array_equal(result2["landmarks_mp"], result1["landmarks_mp"])
    assert "corrupted cache entry" in capsys.readouterr().out

    get_landmarks(
        cache_root, "test_dataset", "s0", None,
        load_source_image=lambda: image,
        get_detector=lambda: _FakeDetector(),
        get_fan_predictor=lambda: _FakeFanPredictor(),
        get_mediapipe_detector=lambda: _FakeMediapipeDetector(),
        crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
    )
    assert "corrupted cache entry" not in capsys.readouterr().out


def test_bucket_container_holds_multiple_frames_of_the_same_sample(tmp_path):
    cache_root = tmp_path / "landmark_cache"
    crop_cache_root = tmp_path / "face_crop_cache"
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    for frame_index in range(5):
        get_landmarks(
            cache_root, "test_dataset", "video0", frame_index,
            load_source_image=lambda: image,
            get_detector=lambda: _FakeDetector(),
            get_fan_predictor=lambda: _FakeFanPredictor(),
            get_mediapipe_detector=lambda: _FakeMediapipeDetector(),
            crop_cache_root=crop_cache_root, crop_scale=1.4, image_size=224,
        )

    container_path = bucket_container_path(cache_root, "test_dataset", "video0")
    with zipfile.ZipFile(container_path) as zf:
        names = set(zf.namelist())
    assert names == {entry_key("video0", i) for i in range(5)}
