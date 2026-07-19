"""SViT [+ TT] evaluation methods.

OursNoTemporalMethod evaluates a Stage-1-style checkpoint (SViT+ComponentHeads only,
no TT) - frame-wise, matching SmirkMethod's own per-frame contract exactly.

OursFullMethod evaluates a Stage-2 checkpoint (adds TT) and overrides predict_video to
run the model's real whole-clip path (model.encoding.encode_video), the way
inference/demo_videos.py/inference_videos.py already do.

Reuses utils/inference_utils.py's model-construction/checkpoint-loading/visibility-
scoring helpers directly rather than re-implementing them, and model/encoding.py's
encode_image/encode_video for the actual forward passes - see this project's
evaluation implementation plan for why this stays a live per-clip method rather than a
two-stage inference-then-evaluate pipeline against inference/*.py's saved .npz output.
"""

from __future__ import annotations

import cv2
import torch

from methods.base import ReconstructionMethod
from model.encoding import encode_image, encode_video
from utils.inference_utils import (
    build_models,
    compute_visibility_and_mask,
    decode_and_project,
    load_available_checkpoint,
)


class OursNoTemporalMethod(ReconstructionMethod):
    """SViT + ComponentHeads only (model.encoding.encode_image) - evaluates a Stage 1
    pretraining checkpoint. Frame-wise, no temporal context."""

    def setup(self, device, crop_size=224, crop_scale=1.4, checkpoint_path=None):
        self.device = device
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.models = build_models(device, use_unet=False)
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
        self.cache_root = "evaluation/output/.face_parsing_cache"

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
