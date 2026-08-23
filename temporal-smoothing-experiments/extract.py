"""Step 1: video -> raw (no-TT) FLAME parameters.

Factors out inference/demo_videos.py's frame-loading -> parallel crop+XSeg-parse
-> encode_video() block into a reusable function that always skips TT (this
experiment's whole point is a non-learned smoothing baseline to compare against
TT), and keeps everything steps 2 (smoothing.py) and 3 (metrics_utils.py,
run_*_video.py) need - including the XSeg face masks, already computed here,
rather than recomputing them later like evaluation/eval_core.py's separate
compute_visibility_and_mask call does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.encoding import encode_video  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    FrameJob,
    make_crop_parse_pool,
    run_parallel_crop_and_parse,
)


@dataclass
class ClipEncoding:
    """Everything steps 2-3 need for one clip, computed once in extract_flame_params."""

    encoded: dict[str, torch.Tensor]  # each (N, ...): shape, expression, eyelid, jaw, scale, rotation, translation
    visibility_scores: torch.Tensor  # (N,) float, XSeg visibility ratio (implementation-plan.md Sec 4.1)
    valid: torch.Tensor  # (N,) bool, face detected/parsed this frame
    face_masks: list[torch.Tensor]  # (1,H,W) each, same XSeg masks visibility_scores came from (zero-filled where invalid)
    cropped_rgb: list  # (H,W,3) uint8 each, for the "original crop" render panel (zero-filled where invalid)
    frames: list  # original (un-cropped) BGR frames, as returned by preprocessing.io.load_frames
    video_fps: int
    video_name: str


def load_clip_frames(input_path: str, fps: int = 30) -> tuple[list, int, str]:
    """input_path: a video file, or a directory of frames (image_seq mode) - detected
    via os.path.isdir, so callers never need to pass an explicit --image_seq flag
    (unambiguous: a video file is never a directory). Returns (frames, video_fps,
    video_name), matching preprocessing.io.load_frames' own return shape."""
    image_seq = os.path.isdir(input_path)
    frames, video_fps, video_name = load_frames(input_path, image_seq, fps)
    if not frames:
        raise ValueError(f"No frames found at {input_path}")
    return frames, video_fps, video_name


def extract_flame_params(
    frames: list,
    video_name: str,
    video_fps: int,
    models: dict,
    device: str,
    xseg_device: str,
    crop_scale: float,
    image_size: int,
    num_workers: int,
    svit_chunk_size: int | None = 32,
) -> ClipEncoding:
    """frames: list of BGR frames (preprocessing.io.load_frames' output). models:
    utils.inference_utils.build_models()'s dict (svit/heads/tt/flame/renderer,
    checkpoint already loaded by the caller via load_available_checkpoint - tt is
    never called here since skip_tt=True below, but is still built/loaded so a
    caller's build_models() call stays identical to demo_videos.py's).

    Runs the exact same detect+crop+XSeg-parse -> encode_video(skip_tt=True) path
    demo_videos.py's --no_tt takes, including encode_video's own unconditional
    missing-frame token-averaging fix-up (model/encoding.py's
    _fill_missing_frame_tokens, which runs before the skip_tt branch)."""
    num_frames = len(frames)
    jobs = [
        FrameJob(video_name, i, frame_bgr, crop_scale, image_size)
        for i, frame_bgr in enumerate(frames)
    ]
    executor = make_crop_parse_pool(num_workers)
    try:
        results = run_parallel_crop_and_parse(jobs, executor, xseg_device)
    finally:
        executor.shutdown()
    results_by_frame = {result.frame_index: result for result in results}

    pixel_values_list, cropped_rgb_list, visibility_list, valid_list, face_mask_list = (
        [], [], [], [], [],
    )
    for frame_index in range(num_frames):
        result = results_by_frame[frame_index]
        frame_valid = result.valid and result.cropped_rgb is not None
        if frame_valid:
            pixel_values = torch.from_numpy(result.cropped_rgb).permute(2, 0, 1).float().to(device) / 255.0
            cropped_rgb = result.cropped_rgb
        else:
            # RetinaFace/XSeg failed for this frame - black-frame fallback, matching
            # demo_videos.py's own convention; discarded via encode_video's own
            # missing-frame token averaging once flag_visibility_valid=False below.
            pixel_values = torch.zeros(3, image_size, image_size, device=device)
            cropped_rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        # result.face_mask is already a zero-filled fallback on the no-face path
        # (preprocessing/cropping+face_parsing_cache's own convention - never None),
        # so this is safe unconditionally, matching demo_videos.py's own loop.
        face_mask = torch.from_numpy(result.face_mask).unsqueeze(0).to(device)

        pixel_values_list.append(pixel_values)
        cropped_rgb_list.append(cropped_rgb)
        visibility_list.append(result.visibility_ratio)
        valid_list.append(frame_valid)
        face_mask_list.append(face_mask)

    clip_pixel_values = torch.stack(pixel_values_list, dim=0).unsqueeze(0)
    # dtype=torch.float32 explicit: torch.tensor() on a list of plain Python floats
    # defaults to float64, which nn.MultiheadAttention's fused attention kernel
    # rejects outright (attn_mask dtype must exactly match the query tensor's dtype).
    visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=device).unsqueeze(0)
    frame_indices = torch.arange(num_frames, device=device).float().unsqueeze(0)
    flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=device).unsqueeze(0)

    encoded = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
        skip_tt=True,
        svit_chunk_size=svit_chunk_size,
    )
    encoded = {name: value.squeeze(0) for name, value in encoded.items()}  # (N, ...)

    return ClipEncoding(
        encoded=encoded,
        visibility_scores=visibility_scores.squeeze(0),
        valid=flag_visibility_valid.squeeze(0),
        face_masks=face_mask_list,
        cropped_rgb=cropped_rgb_list,
        frames=frames,
        video_fps=video_fps,
        video_name=video_name,
    )

