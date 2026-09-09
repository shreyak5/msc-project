from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from methods.base import ReconstructionMethod
from model.encoding import encode_image, encode_video
from utils.inference_utils import (
    build_models,
    compute_visibility_and_mask,
    decode_and_project,
    load_available_checkpoint,
    peek_num_expression_params,
)
from utils.kernel_smoothing import smooth_encoded_params


class OursNoTemporalMethod(ReconstructionMethod):
    """SViT + ComponentHeads only (model.encoding.encode_image) - evaluates a Stage 1
    pretraining checkpoint. Frame-wise, no temporal context."""

    def setup(self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None):
        self.device = device
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.models = build_models(
            device, use_unet=False,
            num_expression_params=peek_num_expression_params(checkpoint_path),
        )
        load_available_checkpoint(self.models, checkpoint_path, device)

    @torch.no_grad()
    def predict_frames(self, cropped_bgr_frames):
        tensors = [
            torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            for f in cropped_bgr_frames
        ]
        pixel_values = torch.stack(tensors, dim=0).to(self.device)  # (N,3,H,W), one batched forward pass
        encoded = encode_image(self.models["svit"], self.models["heads"], pixel_values)
        return decode_and_project(self.models["flame"], encoded, self.crop_size)


class OursFullMethod(OursNoTemporalMethod):
    """SViT + TemporalTransformer + ComponentHeads (model.encoding.encode_video) -
    evaluates a Stage 2 checkpoint. predict_frames/predict_frame (inherited, unchanged)
    still run SViT+MLP-only, no TT, matching demo_images.py's own behavior of never
    touching TT even against a full checkpoint; predict_video is overridden to use the
    real whole-clip TT path."""

    # Caps how many frames' post-TT params get passed through FLAME (5023 vertices/
    # frame) + projection in one call inside predict_video - purely a memory-management
    # detail for that step, unrelated to TT's own windowing (SViT+TT already ran once
    # over the WHOLE clip via encode_video before this chunking ever applies). Mirrors
    # demo_videos.py's own RENDER_CHUNK_SIZE=32, used there for the same reason.
    FLAME_CHUNK_SIZE = 32

    def setup(self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None):
        super().setup(device, crop_size=crop_size, crop_scale=crop_scale, checkpoint_path=checkpoint_path)
        # Same physical cache root as eval_core.VISIBILITY_CACHE_ROOT - see that
        # constant's docstring for why this lives on /projects (Lustre) rather
        # than /home. Still buckets under dataset="inference" (compute_visibility_
        # and_mask's default) rather than a real indexed dataset name/sample_id,
        # since ours_full isn't in active use yet; revisit alongside eval_core's
        # dataset_name threading when it is, to also get the free cache hits.
        self.cache_root = "/lus/lfs1aip2/projects/u6ga/sk3925_datasets/face_parsing_cache"

    @torch.no_grad()
    def predict_video(self, cropped_bgr_frames, raw_bgr_frames, clip_id):
        num_frames = len(cropped_bgr_frames)
        pixel_values_list, visibility_list, valid_list = [], [], []
        for i, (raw_frame, cropped) in enumerate(zip(raw_bgr_frames, cropped_bgr_frames)):
            _mask, visibility_ratio, valid = compute_visibility_and_mask(
                raw_frame, self.cache_root, clip_id, i, self.device, self.device,
                self.crop_scale, self.crop_size, self.device,
            )
            frame_valid = valid and cropped is not None
            if frame_valid:
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pixel_values = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
            else:
                # RetinaFace/XSeg failed, or eval_core's own crop_face failed, for this
                # frame - fall back to a black frame, matching demo_videos.py's own
                # missing-frame handling; the frame gets discarded via encode_video's
                # own missing-frame token averaging once flag_visibility_valid=False.
                pixel_values = torch.zeros(3, self.crop_size, self.crop_size, device=self.device)
            pixel_values_list.append(pixel_values)
            visibility_list.append(visibility_ratio)
            valid_list.append(frame_valid)

        clip_pixel_values = torch.stack(pixel_values_list, dim=0).unsqueeze(0)  # (1,N,3,H,W)
        visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        frame_indices = torch.arange(num_frames, device=self.device).float().unsqueeze(0)
        flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=self.device).unsqueeze(0)

        encoded = encode_video(
            self.models["svit"], self.models["tt"], self.models["heads"],
            clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
        )
        # Squeeze the batch dim (always 1 here - a single clip) so FLAME sees N frames
        # as its own batch dimension, matching demo_videos.py's own squeeze.
        encoded = {name: value.squeeze(0) for name, value in encoded.items()}

        results = [None] * num_frames
        for start in range(0, num_frames, self.FLAME_CHUNK_SIZE):
            end = min(start + self.FLAME_CHUNK_SIZE, num_frames)
            chunk = {name: value[start:end] for name, value in encoded.items()}
            chunk_preds = decode_and_project(self.models["flame"], chunk, self.crop_size)
            for j, pred in enumerate(chunk_preds):
                idx = start + j
                if cropped_bgr_frames[idx] is not None:
                    results[idx] = pred
        return results


# Raw (no-TT) per-frame FLAME-param encoding cache for OursKernelSmoothMethod - lets a
# sweep across many (radius, sigma, temperature) settings for the SAME clip run SViT
# only once (on the first setting's cache miss) instead of once per setting. One file
# per clip (not per setting, since encoding is setting-independent) - see this
# project's temporal-smoothing-experiments implementation plan for the file-count/size
# analysis behind that choice.
_RAW_ENCODING_CACHE_ROOT = "/lus/lfs1aip2/projects/u6ga/sk3925_datasets/kernel_smooth_encoding_cache"
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
_RAW_ENCODING_PARAM_KEYS = ("shape", "expression", "eyelid", "jaw", "scale", "rotation", "translation")


def _cache_sample_id(clip_id: str) -> str:
    """Strips a video-file extension from clip_id before it's used as a cache
    sample_id, matching evaluation/eval_core.py's own _visibility_sample_id
    convention (indexed image-seq datasets' sample_id is already extension-less,
    so this is a no-op for those) - duplicated here rather than imported, since
    eval_core.py imports this module (methods.ours_method), so importing back
    from eval_core would be circular."""
    root, ext = os.path.splitext(clip_id)
    return root if ext.lower() in _VIDEO_EXTENSIONS else clip_id


def _raw_encoding_cache_path(
    checkpoint_path: str | None, dataset_name: str, crop_size: int, crop_scale: float, clip_id: str,
) -> Path:
    """One file per (checkpoint, dataset, crop config, clip) - NOT per (radius,
    sigma, temperature), since the raw encoding this caches is computed before
    smoothing ever applies and is identical across every setting swept against
    the same clip."""
    checkpoint_id = os.path.basename(checkpoint_path) if checkpoint_path else "no_checkpoint"
    bucket = f"crop{crop_size}_{crop_scale}"
    sample_id = _cache_sample_id(clip_id)
    return Path(_RAW_ENCODING_CACHE_ROOT) / checkpoint_id / dataset_name / bucket / f"{sample_id}.npz"


def _load_cached_encoding(
    path: Path, device: str,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray] | None:
    """None on a miss (including a corrupted entry, self-healing the same way
    every dataset_processing/dataloading/*_cache.py module does - warn and
    recompute rather than crash)."""
    if not path.exists():
        return None
    try:
        data = np.load(path)
        encoded = {name: torch.from_numpy(data[name]).to(device) for name in _RAW_ENCODING_PARAM_KEYS}
        return encoded, data["visibility"], data["valid"]
    except Exception as exc:
        print(f"warning: corrupted kernel-smooth encoding cache entry {path}, recomputing: {exc}")
        return None


def _save_cached_encoding(
    path: Path, encoded: dict[str, torch.Tensor], visibility: np.ndarray, valid: np.ndarray,
) -> None:
    """Atomic tmp-file + os.replace, matching utils/cache_utils.py's own
    atomic_write_bytes convention - no locking needed (unlike that module's
    bucket-container caches, which pack many clips into one shared file): this
    cache is one file per clip, and within one sweep run each clip's encoding is
    only ever written by whichever process/shard first misses on it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".tmp-{uuid.uuid4().hex}-{path.name}")
    payload = {name: encoded[name].detach().cpu().numpy() for name in _RAW_ENCODING_PARAM_KEYS}
    payload["visibility"] = np.asarray(visibility, dtype=np.float32)
    payload["valid"] = np.asarray(valid, dtype=bool)
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, path)


class OursKernelSmoothMethod(OursNoTemporalMethod):
    FLAME_CHUNK_SIZE = 32  # same rationale as OursFullMethod

    def setup(
        self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None,
        radius=4, sigma=2.0, temperature=0.1, dataset_name="inference",
    ):
        super().setup(device, crop_size=crop_size, crop_scale=crop_scale, checkpoint_path=checkpoint_path)
        # Same physical cache root as eval_core.VISIBILITY_CACHE_ROOT/OursFullMethod's
        # own identical literal - not imported from eval_core.py, since that module
        # imports this one (circular).
        self.visibility_cache_root = "/lus/lfs1aip2/projects/u6ga/sk3925_datasets/face_parsing_cache"
        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name
        self.radius = radius
        self.sigma = sigma
        self.temperature = temperature

    @torch.no_grad()
    def predict_video(self, cropped_bgr_frames, raw_bgr_frames, clip_id):
        num_frames = len(cropped_bgr_frames)
        cache_path = _raw_encoding_cache_path(
            self.checkpoint_path, self.dataset_name, self.crop_size, self.crop_scale, clip_id,
        )
        cached = _load_cached_encoding(cache_path, self.device)

        if cached is not None:
            encoded, visibility_np, _valid_np = cached
        else:
            pixel_values_list, visibility_list, valid_list = [], [], []
            for i, (raw_frame, cropped) in enumerate(zip(raw_bgr_frames, cropped_bgr_frames)):
                _mask, visibility_ratio, valid = compute_visibility_and_mask(
                    raw_frame, self.visibility_cache_root, _cache_sample_id(clip_id), i,
                    self.device, self.device, self.crop_scale, self.crop_size, self.device,
                    dataset=self.dataset_name,
                )
                frame_valid = valid and cropped is not None
                if frame_valid:
                    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    pixel_values = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255.0
                else:
                    # RetinaFace/XSeg failed, or eval_core's own crop failed, for this
                    # frame - black-frame fallback, matching OursFullMethod's own
                    # identical convention.
                    pixel_values = torch.zeros(3, self.crop_size, self.crop_size, device=self.device)
                pixel_values_list.append(pixel_values)
                visibility_list.append(visibility_ratio)
                valid_list.append(frame_valid)

            pixel_values = torch.stack(pixel_values_list, dim=0)  # (N,3,H,W)
            encoded = encode_image(self.models["svit"], self.models["heads"], pixel_values)
            visibility_np = np.asarray(visibility_list, dtype=np.float32)
            valid_np = np.asarray(valid_list, dtype=bool)
            _save_cached_encoding(cache_path, encoded, visibility_np, valid_np)

        smoothed = smooth_encoded_params(encoded, visibility_np, self.radius, self.sigma, self.temperature)

        results = [None] * num_frames
        for start in range(0, num_frames, self.FLAME_CHUNK_SIZE):
            end = min(start + self.FLAME_CHUNK_SIZE, num_frames)
            chunk = {name: value[start:end] for name, value in smoothed.items()}
            chunk_preds = decode_and_project(self.models["flame"], chunk, self.crop_size)
            for j, pred in enumerate(chunk_preds):
                idx = start + j
                if cropped_bgr_frames[idx] is not None:
                    results[idx] = pred
        return results
