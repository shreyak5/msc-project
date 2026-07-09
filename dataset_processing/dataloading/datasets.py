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
from dataset_processing.dataloading.registry import DatasetEntry
from dataset_processing.dataloading.video_frames import (
    frame_indices_for_segment,
    make_frame_source,
    segment_starts,
)
from dataset_processing.manifest_schema import ManifestRow, read_manifest

IMAGE_CATEGORIES = {"2d_image", "3d_image"}
FLAME_CATEGORIES = {"3d_image", "3d_video"}


def _crop_to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


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

    def __len__(self) -> int:
        return len(self.rows)

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

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

        rows = [row for row in read_manifest(manifest_path) if row.split == split]
        # len(row.image_paths) is only the true frame count for the pre-extracted
        # frame-list representation; for a single video file it's always 1, so the actual
        # frame count must come from a prewarm-built cache, or - only if that's missing -
        # a live probe (opening tens of thousands of videos here would otherwise make
        # constructing this Dataset take minutes, every time, on every DDP rank).
        cached_counts = load_frame_counts(crop_cache_root, dataset_name) or {}
        # Each index entry carries the row's true frame count alongside it, computed once
        # here, so __getitem__ never needs to (incorrectly) re-derive it from
        # len(row.image_paths) - which is only accurate for the frame-list representation,
        # not for a single video file.
        self.index: list[tuple[ManifestRow, int, int]] = []
        for row in rows:
            num_frames_total = cached_counts.get(row.sample_id)
            if num_frames_total is None:
                try:
                    source = make_frame_source(row.image_paths)
                    num_frames_total = source.num_frames()
                    source.close()
                except Exception as exc:
                    # A single unreadable/corrupted video (e.g. a truncated .mp4 missing
                    # its trailer) shouldn't take down the whole dataset - skip it.
                    print(f"warning: skipping unreadable {dataset_name}/{row.sample_id}: {exc}")
                    continue
            for start in segment_starts(num_frames_total, max_frames):
                self.index.append((row, start, num_frames_total))

    def __len__(self) -> int:
        return len(self.index)

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, start, num_frames_total = self.index[index]
        frame_indices = frame_indices_for_segment(start, self.max_frames, num_frames_total)
        source = make_frame_source(row.image_paths)

        frames = []
        meshes = []
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
        source.close()

        item: dict[str, Any] = {
            "dataset": self.dataset_name,
            "subject_id": row.subject_id,
            "pixel_values": torch.stack(frames, dim=0),
        }
        if self.with_flame:
            item["flame_vertices"] = torch.stack(meshes, dim=0)
        return item


def build_category_dataset(entry: DatasetEntry, split: str, cfg: DataloaderConfig) -> Dataset:
    with_flame = entry.category in FLAME_CATEGORIES
    if entry.category in IMAGE_CATEGORIES:
        return ImageFaceDataset(
            entry.name, entry.manifest_path, split, cfg.crop_cache_root,
            cfg.image_size, cfg.crop_scale, cfg.detector.device,
            cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
        )
    category_cfg = cfg.categories[entry.category]
    return VideoFaceDataset(
        entry.name, entry.manifest_path, split, cfg.crop_cache_root,
        cfg.image_size, cfg.crop_scale, cfg.detector.device,
        cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
        max_frames=category_cfg.max_frames,
    )
