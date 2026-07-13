import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading import datasets as datasets_module  # noqa: E402
from dataset_processing.dataloading.combined_loader import build_combined_loader  # noqa: E402
from dataset_processing.dataloading.config import CategoryConfig, DataloaderConfig, DetectorConfig  # noqa: E402
from dataset_processing.dataloading.datasets import (  # noqa: E402
    FramePoolVideoDataset,
    ImageFaceDataset,
    VideoFaceDataset,
)


class _NoFaceDetector:
    """Stands in for a real RetinaFace detector: callable like the real one, but always
    reports no detections, so tests exercise the crop cache's graceful "no face" path
    without depending on an image that actually contains a detectable face."""

    def __call__(self, image, rgb=False):
        return None


def _write_dummy_image(path: Path, color=(120, 130, 140), size=64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_dummy_flame_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"vertices": [[0.0, 0.0, 0.0]] * 5023}, f)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _base_row(**overrides) -> dict:
    row = {
        "dataset": "test_dataset",
        "sample_id": "sample_0",
        "subject_id": "subject_0",
        "dimensionality": "2d",
        "modality": "image",
        "image_paths": [],
        "flame_mesh_paths": None,
        "frame_index": None,
        "sequence_id": None,
        "camera_id": None,
        "split": "train",
        "labels": {"some_label": "should_never_appear_in_output"},
    }
    row.update(overrides)
    return row


def test_image_dataset_shape_dtype_labels_excluded_and_cache_hit(tmp_path, monkeypatch):
    image_path = tmp_path / "source" / "img0.jpg"
    _write_dummy_image(image_path)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(sample_id="s0", image_paths=[str(image_path)])])

    call_count = {"n": 0}

    def fake_get_detector(device, threshold, model_name):
        call_count["n"] += 1
        return _NoFaceDetector()

    monkeypatch.setattr(datasets_module, "get_detector", fake_get_detector)

    ds = ImageFaceDataset(
        "test_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25", with_flame=False,
    )
    assert len(ds) == 1

    item = ds[0]
    assert set(item.keys()) == {"dataset", "subject_id", "pixel_values"}  # no "labels" content leaks through
    assert item["dataset"] == "test_dataset"
    assert item["subject_id"] == "subject_0"
    assert item["pixel_values"].shape == (3, 224, 224)
    assert item["pixel_values"].dtype == torch.float32
    assert call_count["n"] == 1  # cache miss -> detector factory invoked once

    item_again = ds[0]
    assert torch.equal(item["pixel_values"], item_again["pixel_values"])
    assert call_count["n"] == 1  # second access is a cache hit -> detector never re-invoked


def test_3d_image_dataset_includes_flame_vertices(tmp_path, monkeypatch):
    image_path = tmp_path / "source" / "img0.jpg"
    _write_dummy_image(image_path)
    mesh_path = tmp_path / "source" / "mesh0.json"
    _write_dummy_flame_json(mesh_path)
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(
        sample_id="s0", dimensionality="3d",
        image_paths=[str(image_path)], flame_mesh_paths=[str(mesh_path)],
    )])

    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())

    ds = ImageFaceDataset(
        "test_3d_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25", with_flame=True,
    )
    item = ds[0]
    assert set(item.keys()) == {"dataset", "subject_id", "pixel_values", "flame_vertices"}
    assert item["flame_vertices"].shape == (5023, 3)
    assert item["flame_vertices"].dtype == torch.float32


def test_video_dataset_segments_and_padding(tmp_path, monkeypatch):
    frame_paths = []
    for i in range(40):
        p = tmp_path / "source" / f"frame_{i:03d}.jpg"
        _write_dummy_image(p)
        frame_paths.append(str(p))

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(
        sample_id="clip0", modality="video", image_paths=frame_paths,
    )])

    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())

    ds = VideoFaceDataset(
        "test_video_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25",
        with_flame=False, max_frames=16,
    )
    assert len(ds) == 3  # ceil(40/16) = 3 non-overlapping segments

    item0 = ds[0]
    assert set(item0.keys()) == {"dataset", "subject_id", "pixel_values"}
    assert item0["pixel_values"].shape == (16, 3, 224, 224)

    item2 = ds[2]  # last, short segment (frames 32-39, 8 real frames padded to 16)
    assert item2["pixel_values"].shape == (16, 3, 224, 224)


def _write_dummy_mp4(path: Path, num_frames: int, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (size, size))
    for i in range(num_frames):
        writer.write(np.full((size, size, 3), i % 256, dtype=np.uint8))
    writer.release()


def test_video_dataset_mp4_backed_tail_segment_uses_correct_frame_indices(tmp_path, monkeypatch):
    """Regression test: for a single-video-file row, len(row.image_paths) is always 1,
    never the real frame count. The tail segment's padding must be computed from the
    video's true frame count (probed via FrameSource), not from that length-1 list -
    otherwise padding silently repeats frame 0 instead of the real last frame."""
    video_path = tmp_path / "source" / "clip.mp4"
    _write_dummy_mp4(video_path, num_frames=10)

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(
        sample_id="clip0", modality="video", image_paths=[str(video_path)],
    )])

    requested_frame_indices = []

    def fake_get_cropped_face(cache_root, dataset, sample_id, frame_index, load_source_image,
                               get_detector_fn, scale, image_size, on_noface=None):
        requested_frame_indices.append(frame_index)
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    monkeypatch.setattr(datasets_module, "get_cropped_face", fake_get_cropped_face)

    ds = VideoFaceDataset(
        "test_mp4_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25",
        with_flame=False, max_frames=4,
    )
    # 10 real frames, max_frames=4 -> segments start at 0, 4, 8. The last one only has
    # frames 8 and 9 for real, and must pad by repeating frame 9 - not frame 0.
    assert len(ds) == 3

    ds[2]
    assert requested_frame_indices == [8, 9, 9, 9]


def _build_synthetic_registry(tmp_path: Path) -> Path:
    """Builds a tiny synthetic datasets.yaml + manifests covering all 4 categories, with
    2d_image deliberately much larger than the other 3 categories so the combined loader
    is forced to visibly cycle the smaller ones to keep pace."""
    def image_row(sample_id):
        p = tmp_path / "img" / f"{sample_id}.jpg"
        _write_dummy_image(p)
        return _base_row(dataset="img_a", sample_id=sample_id, image_paths=[str(p)])

    img_manifest = tmp_path / "manifests" / "img_a.jsonl"
    _write_manifest(img_manifest, [image_row(f"s{i}") for i in range(8)])

    video_frames = []
    for i in range(10):
        p = tmp_path / "vid" / f"f{i}.jpg"
        _write_dummy_image(p)
        video_frames.append(str(p))
    vid_manifest = tmp_path / "manifests" / "vid_a.jsonl"
    _write_manifest(vid_manifest, [_base_row(
        dataset="vid_a", sample_id="clip0", modality="video", image_paths=video_frames)])

    mesh_p = tmp_path / "mesh" / "mesh0.json"
    _write_dummy_flame_json(mesh_p)
    img3d_p = tmp_path / "mesh" / "img0.jpg"
    _write_dummy_image(img3d_p)
    mesh_manifest = tmp_path / "manifests" / "mesh_a.jsonl"
    _write_manifest(mesh_manifest, [_base_row(
        dataset="mesh_a", sample_id=f"s{i}", dimensionality="3d",
        image_paths=[str(img3d_p)], flame_mesh_paths=[str(mesh_p)],
    ) for i in range(2)])

    meshvid_manifest = tmp_path / "manifests" / "meshvid_a.jsonl"
    _write_manifest(meshvid_manifest, [_base_row(
        dataset="meshvid_a", sample_id="clip0", dimensionality="3d", modality="video",
        image_paths=video_frames, flame_mesh_paths=[str(mesh_p)] * len(video_frames),
    )])

    registry = {
        "datasets": {
            "img_a": {"manifest": str(img_manifest), "dimensionality": "2d", "modality": "image", "category": "2d_image", "status": "done"},
            "vid_a": {"manifest": str(vid_manifest), "dimensionality": "2d", "modality": "video", "category": "2d_video", "status": "done"},
            "mesh_a": {"manifest": str(mesh_manifest), "dimensionality": "3d", "modality": "image", "category": "3d_image", "status": "done"},
            "meshvid_a": {"manifest": str(meshvid_manifest), "dimensionality": "3d", "modality": "video", "category": "3d_video", "status": "done"},
        }
    }
    registry_path = tmp_path / "datasets.yaml"
    with open(registry_path, "w") as f:
        yaml.safe_dump(registry, f)
    return registry_path


def _synthetic_dataloader_config(tmp_path: Path) -> DataloaderConfig:
    return DataloaderConfig(
        seed=42, image_size=224, crop_scale=1.4, crop_cache_root=str(tmp_path / "cache"),
        detector=DetectorConfig(device="cpu", threshold=0.8, model_name="mobilenet0.25"),
        categories={
            "2d_image": CategoryConfig(batch_size=2, max_frames=1, num_workers=0, drop_last=True),
            "2d_video": CategoryConfig(batch_size=1, max_frames=10, num_workers=0, drop_last=True),
            "3d_image": CategoryConfig(batch_size=1, max_frames=1, num_workers=0, drop_last=True),
            "3d_video": CategoryConfig(batch_size=1, max_frames=10, num_workers=0, drop_last=True),
        },
    )


def test_combined_loader_cycles_small_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())
    registry_path = _build_synthetic_registry(tmp_path)
    cfg = _synthetic_dataloader_config(tmp_path)

    loader = build_combined_loader(cfg, split="train", rank=0, world_size=1, datasets_yaml_path=registry_path)
    # img_a: 8 rows / batch_size 2 -> 4 batches (the largest category, sets the epoch length).
    # vid_a/meshvid_a: 1 clip of 10 frames, max_frames=10 -> 1 segment -> 1 batch, must cycle 4x.
    # mesh_a: 2 rows / batch_size 1 -> 2 batches, must cycle twice to reach 4.
    assert len(loader) == 4

    loader.set_epoch(0)
    batches = list(loader)
    assert len(batches) == 4
    for batch in batches:
        assert set(batch.keys()) == {"2d_image", "2d_video", "3d_image", "3d_video"}

    # 2d_video only ever has one possible sample ("clip0") - if cycling works, every one
    # of the 4 steps must still produce it rather than raising StopIteration after step 1.
    for batch in batches:
        assert batch["2d_video"]["dataset"] == ["vid_a"]
        assert batch["3d_video"]["flame_vertices"].shape == (1, 10, 5023, 3)


def test_ddp_sampler_length_consistent_across_ranks(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())
    registry_path = _build_synthetic_registry(tmp_path)
    cfg = _synthetic_dataloader_config(tmp_path)

    loader_rank0 = build_combined_loader(cfg, split="train", rank=0, world_size=2, datasets_yaml_path=registry_path)
    loader_rank1 = build_combined_loader(cfg, split="train", rank=1, world_size=2, datasets_yaml_path=registry_path)

    # Every rank must independently compute the same epoch length, or a real 16-GPU DDP
    # job would deadlock (some ranks finishing their collective ops before others).
    assert len(loader_rank0) == len(loader_rank1)


def test_frame_pool_dataset_one_entry_per_video_not_per_frame(tmp_path, monkeypatch):
    frame_paths = []
    for i in range(20):
        p = tmp_path / "source" / f"frame_{i:03d}.jpg"
        _write_dummy_image(p)
        frame_paths.append(str(p))

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(
        sample_id="clip0", modality="video", image_paths=frame_paths,
    )])

    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())

    ds = FramePoolVideoDataset(
        "test_framepool_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25", with_flame=False,
    )
    # One video, 20 frames - a frame pool has exactly one entry (unlike VideoFaceDataset,
    # which would split this into ceil(20/max_frames) segments).
    assert len(ds) == 1

    item = ds[0]
    assert set(item.keys()) == {"dataset", "subject_id", "pixel_values"}
    assert item["pixel_values"].shape == (3, 224, 224)


def test_frame_pool_dataset_resamples_a_different_frame_across_accesses(tmp_path, monkeypatch):
    frame_paths = []
    for i in range(20):
        p = tmp_path / "source" / f"frame_{i:03d}.jpg"
        _write_dummy_image(p)
        frame_paths.append(str(p))

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(
        sample_id="clip0", modality="video", image_paths=frame_paths,
    )])

    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())

    requested_frame_indices = []

    def fake_get_cropped_face(cache_root, dataset, sample_id, frame_index, load_source_image,
                               get_detector_fn, scale, image_size, on_noface=None):
        requested_frame_indices.append(frame_index)
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

    monkeypatch.setattr(datasets_module, "get_cropped_face", fake_get_cropped_face)

    ds = FramePoolVideoDataset(
        "test_framepool_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25", with_flame=False,
    )

    for _ in range(30):
        ds[0]

    # With 20 possible frames and 30 draws, seeing only a single repeated value would be
    # astronomically unlikely if re-sampling is actually happening on every access.
    assert len(set(requested_frame_indices)) > 1
    assert all(0 <= idx < 20 for idx in requested_frame_indices)


def test_frame_pool_dataset_loads_flame_mesh_for_the_sampled_frame(tmp_path, monkeypatch):
    frame_paths = []
    mesh_paths = []
    for i in range(5):
        img_p = tmp_path / "source" / f"frame_{i:03d}.jpg"
        _write_dummy_image(img_p)
        frame_paths.append(str(img_p))
        mesh_p = tmp_path / "source" / f"mesh_{i:03d}.json"
        _write_dummy_flame_json(mesh_p)
        mesh_paths.append(str(mesh_p))

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, [_base_row(
        sample_id="clip0", dimensionality="3d", modality="video",
        image_paths=frame_paths, flame_mesh_paths=mesh_paths,
    )])

    monkeypatch.setattr(datasets_module, "get_detector", lambda *a, **k: _NoFaceDetector())

    ds = FramePoolVideoDataset(
        "test_framepool_3d_dataset", manifest_path, "train", tmp_path / "cache",
        image_size=224, crop_scale=1.4, detector_device="cpu",
        detector_threshold=0.8, detector_model_name="mobilenet0.25", with_flame=True,
    )

    item = ds[0]
    assert set(item.keys()) == {"dataset", "subject_id", "pixel_values", "flame_vertices"}
    assert item["flame_vertices"].shape == (5023, 3)
