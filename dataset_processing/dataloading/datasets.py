from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_processing.dataloading.config import DataloaderConfig
from dataset_processing.dataloading.crop_cache import get_cropped_face
from dataset_processing.dataloading.detector_pool import get_detector
from dataset_processing.dataloading.face_parsing_cache import get_face_parsing
from dataset_processing.dataloading.face_parsing_pool import get_xseg
from dataset_processing.dataloading.frame_count_cache import load_frame_counts
from dataset_processing.dataloading.landmark_cache import get_landmarks
from dataset_processing.dataloading.landmark_pool import get_fan_predictor, get_mediapipe_detector
from dataset_processing.dataloading.mesh_io import load_flame_vertices
from dataset_processing.dataloading.mica_cache import get_mica_shape
from dataset_processing.dataloading.mica_pool import get_mica
from dataset_processing.dataloading.registry import DatasetEntry
from dataset_processing.dataloading.video_frames import (
    frame_indices_for_segment,
    make_frame_source,
    segment_starts,
    valid_mask_for_segment,
)
from dataset_processing.manifest_schema import ManifestRow, read_manifest
from model.constants import MEDIAPIPE_TASK_MODEL_PATH, MICA_IMAGE_SIZE

IMAGE_CATEGORIES = {"2d_image", "3d_image"}
CATEGORIES_3D = {"3d_image", "3d_video"}
# MICA shape distillation AND the landmark loss (implementation-plan.md Sec 6/7)
# both apply to 2D batches only - 3D categories get direct mesh/Lvc supervision
# instead, a stronger signal than either MICA's distilled estimate or a detected-
# landmark reprojection target. Shared by both since the scoping is identical.
# The face-region mask (masking -> UNet reconstruction) is 2D-only for the same
# reason. Visibility scores are different: VideoFaceDataset computes them
# unconditionally regardless of 2D/3D, since TemporalTransformer needs them for
# every video category's clips in Pass C - see build_category_dataset.
CATEGORIES_2D = {"2d_image", "2d_video"}


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
        with_landmarks: bool,
        landmark_cache_root: str | Path,
        fan_device: str,
        with_face_mask: bool,
        face_parsing_cache_root: str | Path,
        xseg_device: str,
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
        self.with_landmarks = with_landmarks
        self.landmark_cache_root = Path(landmark_cache_root)
        self.fan_device = fan_device
        self.with_face_mask = with_face_mask
        self.face_parsing_cache_root = Path(face_parsing_cache_root)
        self.xseg_device = xseg_device

    def __len__(self) -> int:
        return len(self.rows)

    def get_subject_id(self, index: int) -> str:
        """O(1), no I/O - just the already-loaded manifest row's subject_id.
        Used by identity_batch_sampler.IdentityAwareBatchSampler to group indices
        by identity at batch-construction time, without paying __getitem__'s
        full image-loading/cropping cost just to read one field."""
        return self.rows[index].subject_id

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

    def _get_mica(self):
        return get_mica(self.mica_device)

    def _get_fan_predictor(self):
        return get_fan_predictor(self.fan_device, "2dfan4")

    def _get_mediapipe_detector(self):
        return get_mediapipe_detector(MEDIAPIPE_TASK_MODEL_PATH)

    def _get_xseg(self):
        return get_xseg(self.xseg_device)

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
        if self.with_landmarks:
            landmarks = get_landmarks(
                self.landmark_cache_root, self.dataset_name, row.sample_id, None,
                lambda: cv2.imread(image_path),
                self._get_detector,
                self._get_fan_predictor,
                self._get_mediapipe_detector,
                self.crop_cache_root, self.crop_scale, self.image_size,
            )
            item["landmarks_fan"] = torch.from_numpy(landmarks["landmarks_fan"])
            item["flag_landmarks_fan_valid"] = landmarks["flag_landmarks_fan_valid"]
            item["landmarks_mp"] = torch.from_numpy(landmarks["landmarks_mp"])
            item["flag_landmarks_mp_valid"] = landmarks["flag_landmarks_mp_valid"]
        if self.with_face_mask:
            face_mask, _visibility_ratio, flag_face_mask_valid = get_face_parsing(
                self.face_parsing_cache_root, self.dataset_name, row.sample_id, None,
                lambda: cv2.imread(image_path),
                self._get_detector,
                self._get_xseg,
                self.crop_scale, self.image_size,
            )
            item["face_mask"] = torch.from_numpy(face_mask)
            item["flag_face_mask_valid"] = flag_face_mask_valid
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
        with_landmarks: bool,
        landmark_cache_root: str | Path,
        fan_device: str,
        with_face_mask: bool,
        face_parsing_cache_root: str | Path,
        xseg_device: str,
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
        self.with_landmarks = with_landmarks
        self.landmark_cache_root = Path(landmark_cache_root)
        self.fan_device = fan_device
        self.with_face_mask = with_face_mask
        self.face_parsing_cache_root = Path(face_parsing_cache_root)
        self.xseg_device = xseg_device

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

    def _get_fan_predictor(self):
        return get_fan_predictor(self.fan_device, "2dfan4")

    def _get_mediapipe_detector(self):
        return get_mediapipe_detector(MEDIAPIPE_TASK_MODEL_PATH)

    def _get_xseg(self):
        return get_xseg(self.xseg_device)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, start, num_frames_total = self.index[index]
        frame_indices = frame_indices_for_segment(start, self.max_frames, num_frames_total)
        valid_mask = valid_mask_for_segment(start, self.max_frames, num_frames_total)
        source = make_frame_source(row.image_paths)

        frames = []
        meshes = []
        mica_shapes = []
        flags_mica_valid = []
        landmarks_fan_list = []
        flags_landmarks_fan_valid = []
        landmarks_mp_list = []
        flags_landmarks_mp_valid = []
        face_masks = []
        flags_face_mask_valid = []
        visibility_ratios = []
        flags_visibility_valid = []
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
            if self.with_landmarks:
                landmarks = get_landmarks(
                    self.landmark_cache_root, self.dataset_name, row.sample_id, frame_idx,
                    lambda fi=frame_idx: source.read_frame(fi),
                    self._get_detector,
                    self._get_fan_predictor,
                    self._get_mediapipe_detector,
                    self.crop_cache_root, self.crop_scale, self.image_size,
                )
                landmarks_fan_list.append(torch.from_numpy(landmarks["landmarks_fan"]))
                flags_landmarks_fan_valid.append(landmarks["flag_landmarks_fan_valid"])
                landmarks_mp_list.append(torch.from_numpy(landmarks["landmarks_mp"]))
                flags_landmarks_mp_valid.append(landmarks["flag_landmarks_mp_valid"])
            # visibility_ratio is always computed (TemporalTransformer needs it for
            # every video category's clips in Pass C, 2D and 3D alike - unlike
            # face_mask, which only 2D categories need for masking -> UNet
            # reconstruction), so this call isn't gated behind with_face_mask.
            face_mask, visibility_ratio, flag_face_parsing_valid = get_face_parsing(
                self.face_parsing_cache_root, self.dataset_name, row.sample_id, frame_idx,
                lambda fi=frame_idx: source.read_frame(fi),
                self._get_detector,
                self._get_xseg,
                self.crop_scale, self.image_size,
            )
            if self.with_face_mask:
                face_masks.append(torch.from_numpy(face_mask))
                flags_face_mask_valid.append(flag_face_parsing_valid)
            visibility_ratios.append(visibility_ratio)
            flags_visibility_valid.append(flag_face_parsing_valid)
        source.close()

        item: dict[str, Any] = {
            "dataset": self.dataset_name,
            "subject_id": row.subject_id,
            "pixel_values": torch.stack(frames, dim=0),
            "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
            "visibility_ratio": torch.tensor(visibility_ratios, dtype=torch.float32),
            "flag_visibility_valid": torch.tensor(flags_visibility_valid, dtype=torch.bool),
        }
        if self.with_flame:
            item["flame_vertices"] = torch.stack(meshes, dim=0)
        if self.with_mica:
            item["mica_shape"] = torch.stack(mica_shapes, dim=0)
            item["flag_mica_valid"] = torch.tensor(flags_mica_valid, dtype=torch.bool)
        if self.with_face_mask:
            item["face_mask"] = torch.stack(face_masks, dim=0)
            item["flag_face_mask_valid"] = torch.tensor(flags_face_mask_valid, dtype=torch.bool)
        if self.with_landmarks:
            item["landmarks_fan"] = torch.stack(landmarks_fan_list, dim=0)
            item["flag_landmarks_fan_valid"] = torch.tensor(flags_landmarks_fan_valid, dtype=torch.bool)
            item["landmarks_mp"] = torch.stack(landmarks_mp_list, dim=0)
            item["flag_landmarks_mp_valid"] = torch.tensor(flags_landmarks_mp_valid, dtype=torch.bool)
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
        with_landmarks: bool,
        landmark_cache_root: str | Path,
        fan_device: str,
        with_face_mask: bool,
        face_parsing_cache_root: str | Path,
        xseg_device: str,
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
        self.with_landmarks = with_landmarks
        self.landmark_cache_root = Path(landmark_cache_root)
        self.fan_device = fan_device
        self.with_face_mask = with_face_mask
        self.face_parsing_cache_root = Path(face_parsing_cache_root)
        self.xseg_device = xseg_device

        self.rows_with_counts = _rows_with_frame_counts(dataset_name, manifest_path, split, crop_cache_root)

    def __len__(self) -> int:
        return len(self.rows_with_counts)

    def get_subject_id(self, index: int) -> str:
        """O(1), no I/O - see ImageFaceDataset.get_subject_id's docstring."""
        row, _num_frames_total = self.rows_with_counts[index]
        return row.subject_id

    def _get_detector(self):
        return get_detector(self.detector_device, self.detector_threshold, self.detector_model_name)

    def _get_mica(self):
        return get_mica(self.mica_device)

    def _get_fan_predictor(self):
        return get_fan_predictor(self.fan_device, "2dfan4")

    def _get_mediapipe_detector(self):
        return get_mediapipe_detector(MEDIAPIPE_TASK_MODEL_PATH)

    def _get_xseg(self):
        return get_xseg(self.xseg_device)

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
        if self.with_landmarks:
            landmarks = get_landmarks(
                self.landmark_cache_root, self.dataset_name, row.sample_id, frame_idx,
                lambda: source.read_frame(frame_idx),
                self._get_detector,
                self._get_fan_predictor,
                self._get_mediapipe_detector,
                self.crop_cache_root, self.crop_scale, self.image_size,
            )
            item["landmarks_fan"] = torch.from_numpy(landmarks["landmarks_fan"])
            item["flag_landmarks_fan_valid"] = landmarks["flag_landmarks_fan_valid"]
            item["landmarks_mp"] = torch.from_numpy(landmarks["landmarks_mp"])
            item["flag_landmarks_mp_valid"] = landmarks["flag_landmarks_mp_valid"]
        if self.with_face_mask:
            face_mask, _visibility_ratio, flag_face_mask_valid = get_face_parsing(
                self.face_parsing_cache_root, self.dataset_name, row.sample_id, frame_idx,
                lambda: source.read_frame(frame_idx),
                self._get_detector,
                self._get_xseg,
                self.crop_scale, self.image_size,
            )
            item["face_mask"] = torch.from_numpy(face_mask)
            item["flag_face_mask_valid"] = flag_face_mask_valid
        source.close()
        return item


def build_category_dataset(
    entry: DatasetEntry,
    split: str,
    cfg: DataloaderConfig,
    video_mode: Literal["frame_pool", "clip"] = "clip",
) -> Dataset:
    """video_mode only affects video categories (image categories are always
    single-frame already) - Sec 5.2: frame_pool (one re-sampled random frame per
    access, mixable into image-shaped batches) whenever TT is frozen (Stage 1;
    Stage 2 Pass A/B), clip (a full multi-frame window) for TT training (Stage 2
    Pass C). A single global switch, not per-category: video datasets all switch
    mode together based on which pass is currently running, not independently."""
    with_flame = entry.category in CATEGORIES_3D
    with_mica = entry.category in CATEGORIES_2D
    with_landmarks = entry.category in CATEGORIES_2D
    # Only 2D batches go through masking -> UNet -> photometric/VGG reconstruction
    # (Sec 7 Pass A/C) - 3D batches get direct mesh/Lvc supervision instead, no
    # rendering, so they never need the face-region mask. VideoFaceDataset itself
    # always computes visibility_ratio unconditionally (TemporalTransformer needs
    # it for every video category's clips in Pass C, 2D and 3D alike) - not
    # gated by with_face_mask, which only controls the face_mask field.
    with_face_mask = entry.category in CATEGORIES_2D
    if entry.category in IMAGE_CATEGORIES:
        return ImageFaceDataset(
            entry.name, entry.manifest_path, split, cfg.crop_cache_root,
            cfg.image_size, cfg.crop_scale, cfg.detector.device,
            cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
            with_mica=with_mica, mica_cache_root=cfg.mica_cache_root, mica_device=cfg.mica_device,
            with_landmarks=with_landmarks, landmark_cache_root=cfg.landmark_cache_root, fan_device=cfg.fan_device,
            with_face_mask=with_face_mask, face_parsing_cache_root=cfg.face_parsing_cache_root, xseg_device=cfg.xseg_device,
        )
    if video_mode == "frame_pool":
        return FramePoolVideoDataset(
            entry.name, entry.manifest_path, split, cfg.crop_cache_root,
            cfg.image_size, cfg.crop_scale, cfg.detector.device,
            cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
            with_mica=with_mica, mica_cache_root=cfg.mica_cache_root, mica_device=cfg.mica_device,
            with_landmarks=with_landmarks, landmark_cache_root=cfg.landmark_cache_root, fan_device=cfg.fan_device,
            with_face_mask=with_face_mask, face_parsing_cache_root=cfg.face_parsing_cache_root, xseg_device=cfg.xseg_device,
        )
    category_cfg = cfg.categories[entry.category]
    return VideoFaceDataset(
        entry.name, entry.manifest_path, split, cfg.crop_cache_root,
        cfg.image_size, cfg.crop_scale, cfg.detector.device,
        cfg.detector.threshold, cfg.detector.model_name, with_flame=with_flame,
        max_frames=category_cfg.max_frames,
        with_mica=with_mica, mica_cache_root=cfg.mica_cache_root, mica_device=cfg.mica_device,
        with_landmarks=with_landmarks, landmark_cache_root=cfg.landmark_cache_root, fan_device=cfg.fan_device,
        with_face_mask=with_face_mask, face_parsing_cache_root=cfg.face_parsing_cache_root, xseg_device=cfg.xseg_device,
    )
