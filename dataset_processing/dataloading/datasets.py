from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_processing.dataloading.config import DataloaderConfig
from dataset_processing.dataloading.crop_cache import get_cropped_face
from dataset_processing.dataloading.detector_pool import get_detector
from dataset_processing.dataloading.frame_count_cache import load_frame_counts
from dataset_processing.dataloading.mesh_io import load_flame_vertices
from dataset_processing.dataloading.mica_cache import get_mica_shape
from dataset_processing.dataloading.mica_pool import get_mica
from dataset_processing.dataloading.registry import DatasetEntry
from dataset_processing.dataloading.video_frames import (
    frame_indices_for_segment,
    make_frame_source,
    segment_starts,
)
from dataset_processing.manifest_schema import ManifestRow, read_manifest
from model.constants import MICA_IMAGE_SIZE

IMAGE_CATEGORIES = {"2d_image", "3d_image"}
FLAME_CATEGORIES = {"3d_image", "3d_video"}
# MICA shape distillation (implementation-plan.md Sec 6/7) applies to 2D batches
# only - 3D categories get direct mesh/Lvc supervision instead, a stronger signal
# than MICA's distilled estimate.
MICA_CATEGORIES = {"2d_image", "2d_video"}


def _crop_to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def _rows_with_frame_counts(
    dataset_name: str, manifest_path: str | Path, split: str, crop_cache_root: str | Path,
) -> list[tuple[ManifestRow, int]]:
    """Reads manifest rows for `split` and resolves each row's true frame count (from a
    prewarm-built cache, falling back to a live probe only if that's missing - see
    VideoFaceDataset for why the cache matters at scale). Rows whose video can't be
    opened at all are skipped rather than crashing the whole dataset over one bad file."""
    rows = [row for row in read_manifest(manifest_path) if row.split == split]
    cached_counts = load_frame_counts(crop_cache_root, dataset_name) or {}
    result: list[tuple[ManifestRow, int]] = []
    for row in rows:
        num_frames_total = cached_counts.get(row.sample_id)
        if num_frames_total is None:
            try:
                source = make_frame_source(row.image_paths)
                num_frames_total = source.num_frames()
                source.close()
            except Exception as exc:
                # A single unreadable/corrupted video shouldn't take down the whole
                # dataset - skip it.
                print(f"warning: skipping unreadable {dataset_name}/{row.sample_id}: {exc}")
                continue
        result.append((row, num_frames_total))
    return result


class ImageFaceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        manifest_path: str | Path,
        split: str,
        crop_cache_root: str | Path,
        image_size: int,
        crop_scale: float,
        detector_device: str,
        detector_threshold: float,
        detector_model_name: str,
        with_flame: bool,
        with_mica: bool,
        mica_cache_root: str | Path,
        mica_device: str,
    ):
        self.dataset_name = dataset_name
        self.rows = [row for row in read_manifest(manifest_path) if row.split == split]
        self.crop_cache_root = Path(crop_cache_root)
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.detector_device = detector_device
        self.detector_threshold = detector_threshold
        self.detector_model_name = detector_model_name
        self.with_flame = with_flame
        self.with_mica = with_mica
        self.mica_cache_root = Path(mica_cache_root)
        self.mica_device = mica_device

    def __len__(self) -> int:
        return len(self.rows)

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

    def _get_mica(self):
        return get_mica(self.mica_device)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = row.image_paths[0]
        crop = get_cropped_face(
            self.crop_cache_root, self.dataset_name, row.sample_id, None,
            lambda: cv2.imread(image_path),
            self._get_detector,
            self.crop_scale, self.image_size,
        )
        item: dict[str, Any] = {
            "dataset": self.dataset_name,
            "subject_id": row.subject_id,
            "pixel_values": _crop_to_tensor(crop),
        }
        if self.with_flame:
            item["flame_vertices"] = load_flame_vertices(row.flame_mesh_paths[0])
        if self.with_mica:
            mica_shape, flag_mica_valid = get_mica_shape(
                self.mica_cache_root, self.dataset_name, row.sample_id, None,
                lambda: cv2.imread(image_path),
                self._get_detector,
                self._get_mica,
                MICA_IMAGE_SIZE,
            )
            item["mica_shape"] = torch.from_numpy(mica_shape)
            item["flag_mica_valid"] = flag_mica_valid
        return item


class VideoFaceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        manifest_path: str | Path,
        split: str,
        crop_cache_root: str | Path,
        image_size: int,
        crop_scale: float,
        detector_device: str,
        detector_threshold: float,
        detector_model_name: str,
        with_flame: bool,
        max_frames: int,
        with_mica: bool,
        mica_cache_root: str | Path,
        mica_device: str,
    ):
        self.dataset_name = dataset_name
        self.crop_cache_root = Path(crop_cache_root)
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.detector_device = detector_device
        self.detector_threshold = detector_threshold
        self.detector_model_name = detector_model_name
        self.with_flame = with_flame
        self.max_frames = max_frames
        self.with_mica = with_mica
        self.mica_cache_root = Path(mica_cache_root)
        self.mica_device = mica_device

        # Each index entry carries the row's true frame count alongside it (resolved by
        # _rows_with_frame_counts, from a prewarm-built cache or a live probe - never
        # from len(row.image_paths), which is only accurate for the pre-extracted
        # frame-list representation, not for a single video file), so __getitem__ never
        # needs to re-derive it.
        self.index: list[tuple[ManifestRow, int, int]] = []
        for row, num_frames_total in _rows_with_frame_counts(dataset_name, manifest_path, split, crop_cache_root):
            for start in segment_starts(num_frames_total, max_frames):
                self.index.append((row, start, num_frames_total))

    def __len__(self) -> int:
        return len(self.index)

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

    def _get_mica(self):
        return get_mica(self.mica_device)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, start, num_frames_total = self.index[index]
        frame_indices = frame_indices_for_segment(start, self.max_frames, num_frames_total)
        source = make_frame_source(row.image_paths)

        frames = []
        meshes = []
        mica_shapes = []
        flags_mica_valid = []
        for frame_idx in frame_indices:
            crop = get_cropped_face(
                self.crop_cache_root, self.dataset_name, row.sample_id, frame_idx,
                lambda fi=frame_idx: source.read_frame(fi),
                self._get_detector,
                self.crop_scale, self.image_size,
            )
            frames.append(_crop_to_tensor(crop))
            if self.with_flame:
                meshes.append(load_flame_vertices(row.flame_mesh_paths[frame_idx]))
            if self.with_mica:
                mica_shape, flag_mica_valid = get_mica_shape(
                    self.mica_cache_root, self.dataset_name, row.sample_id, frame_idx,
                    lambda fi=frame_idx: source.read_frame(fi),
                    self._get_detector,
                    self._get_mica,
                    MICA_IMAGE_SIZE,
                )
                mica_shapes.append(torch.from_numpy(mica_shape))
                flags_mica_valid.append(flag_mica_valid)
        source.close()

        item: dict[str, Any] = {
            "dataset": self.dataset_name,
            "subject_id": row.subject_id,
            "pixel_values": torch.stack(frames, dim=0),
        }
        if self.with_flame:
            item["flame_vertices"] = torch.stack(meshes, dim=0)
        if self.with_mica:
            item["mica_shape"] = torch.stack(mica_shapes, dim=0)
            item["flag_mica_valid"] = torch.tensor(flags_mica_valid, dtype=torch.bool)
        return item


class FramePoolVideoDataset(Dataset):
    """Treats a video dataset as a pool of individual frames rather than clips: one
    dataset entry per video, and each access draws a freshly re-sampled random frame
    (implementation-plan.md Sec 5.2 - used whenever TT is frozen, so video datasets can
    mix into the same "2D-loss"/"3D-loss" image batches as real images, like SMIRK's
    own image-level passes). Re-sampled every __getitem__ call, not fixed once per
    video, so a long training run eventually covers most of a video's frames rather
    than just the one frame picked at construction time.

    Uses torch.randint for the frame choice - PyTorch's DataLoader automatically
    reseeds per-worker-process RNG state (verified empirically for this environment:
    torch, Python's stdlib random, and numpy's global RNG all diverge correctly across
    worker processes without a custom worker_init_fn)."""

    def __init__(
        self,
        dataset_name: str,
        manifest_path: str | Path,
        split: str,
        crop_cache_root: str | Path,
        image_size: int,
        crop_scale: float,
        detector_device: str,
        detector_threshold: float,
        detector_model_name: str,
        with_flame: bool,
        with_mica: bool,
        mica_cache_root: str | Path,
        mica_device: str,
    ):
        self.dataset_name = dataset_name
        self.crop_cache_root = Path(crop_cache_root)
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.detector_device = detector_device
        self.detector_threshold = detector_threshold
        self.detector_model_name = detector_model_name
        self.with_flame = with_flame
        self.with_mica = with_mica
        self.mica_cache_root = Path(mica_cache_root)
        self.mica_device = mica_device

        self.rows_with_counts = _rows_with_frame_counts(dataset_name, manifest_path, split, crop_cache_root)

    def __len__(self) -> int:
        return len(self.rows_with_counts)

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

    def _get_mica(self):
        return get_mica(self.mica_device)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, num_frames_total = self.rows_with_counts[index]
        frame_idx = int(torch.randint(0, num_frames_total, (1,)).item())
        source = make_frame_source(row.image_paths)

        crop = get_cropped_face(
            self.crop_cache_root, self.dataset_name, row.sample_id, frame_idx,
            lambda: source.read_frame(frame_idx),
            self._get_detector,
            self.crop_scale, self.image_size,
        )

        item: dict[str, Any] = {
            "dataset": self.dataset_name,
            "subject_id": row.subject_id,
            "pixel_values": _crop_to_tensor(crop),
        }
        if self.with_flame:
            item["flame_vertices"] = load_flame_vertices(row.flame_mesh_paths[frame_idx])
        if self.with_mica:
            mica_shape, flag_mica_valid = get_mica_shape(
                self.mica_cache_root, self.dataset_name, row.sample_id, frame_idx,
                lambda: source.read_frame(frame_idx),
                self._get_detector,
                self._get_mica,
                MICA_IMAGE_SIZE,
            )
            item["mica_shape"] = torch.from_numpy(mica_shape)
            item["flag_mica_valid"] = flag_mica_valid
        source.close()
        return item


def build_category_dataset(entry: DatasetEntry, split: str, cfg: DataloaderConfig) -> Dataset:
    with_flame = entry.category in FLAME_CATEGORIES
    with_mica = entry.category in MICA_CATEGORIES
    if entry.category in IMAGE_CATEGORIES:
        return ImageFaceDataset(
            entry.name, entry.manifest_path, split, cfg.crop_cache_root,
            cfg.image_size, cfg.crop_scale, cfg.detector.device,
            cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
            with_mica=with_mica, mica_cache_root=cfg.mica_cache_root, mica_device=cfg.mica_device,
        )
    category_cfg = cfg.categories[entry.category]
    return VideoFaceDataset(
        entry.name, entry.manifest_path, split, cfg.crop_cache_root,
        cfg.image_size, cfg.crop_scale, cfg.detector.device,
        cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
        max_frames=category_cfg.max_frames,
        with_mica=with_mica, mica_cache_root=cfg.mica_cache_root, mica_device=cfg.mica_device,
    )
