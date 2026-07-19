"""Batch video inference: SViT -> TemporalTransformer -> ComponentHeads ->
FLAME parameters (no rendering) for any number of videos (implementation-
plan.md Sec 3, video path). See inference/demo_videos.py for the
single-video, always-renders sibling script.

Multiple videos are batched together along the batch dimension B (never
concatenated along the frame dimension N) - TemporalTransformer's local-window
attention (model/temporal.py) keeps B as a separate, un-merged axis throughout,
so frames from different videos in the same batch structurally cannot attend
to each other regardless of window size or padding. Videos of different
lengths within a batch are padded to a common frame count using this
project's own dataset_processing/dataloading/video_frames.py helpers (the
same ones training uses), each carrying its own real_frame_mask.

Usage:
    python inference/inference_videos.py --input_path <dir_of_videos> [--checkpoint <path>] [--save_vertices]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from dataset_processing.dataloading.video_frames import (  # noqa: E402
    frame_indices_for_segment,
    valid_mask_for_segment,
)
from model.encoding import encode_video  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    DETECTOR_MODEL_NAME,
    DETECTOR_THRESHOLD,
    build_models,
    compute_visibility_and_mask,
    crop_and_tensor,
    load_available_checkpoint,
    run_flame,
    shared_cache_root,
    timestamped_out_dir,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAM_KEYS = ["shape", "expression", "eyelid", "jaw", "scale", "rotation", "translation"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch video inference: SViT -> TT -> heads -> FLAME parameters.",
    )
    parser.add_argument("--input_path", type=str, required=True, help="Directory of video files.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--detector_device", type=str, default="cpu")
    parser.add_argument("--xseg_device", type=str, default="cpu")
    parser.add_argument("--out_path", type=str, default="inference/output/videos")
    parser.add_argument(
        "--save_vertices", action="store_true", help="Also decode through FLAME and save vertices/landmarks.",
    )
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of VIDEOS per encode_video call.")
    return parser.parse_args()


def gather_video_paths(input_path: str) -> list[str]:
    if not os.path.isdir(input_path):
        raise ValueError(f"input_path '{input_path}' is not a directory")
    files = sorted(f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)
    return [os.path.join(input_path, f) for f in files]


def encode_one_video(video_path: str, detector, cache_root: str, args: argparse.Namespace) -> dict:
    """Reads a video and detects+crops+scores every frame independently (no
    cross-frame tracking - matches how training treats missing/degraded
    frames). Returns per-frame lists, NOT yet padded/batched/encoded - encoding
    happens once per group of videos in main(), after padding to a common
    length."""
    frames, _fps, video_name = load_frames(video_path, image_seq=False, fps=30)
    if not frames:
        raise ValueError(f"No frames found in {video_path}")

    pixel_values_list, visibility_list, valid_list = [], [], []
    for frame_index, frame_bgr in enumerate(frames):
        pixel_values, _cropped_rgb = crop_and_tensor(
            frame_bgr, detector, args.crop_scale, args.image_size, args.device,
        )
        # visibility_ratio is needed unconditionally for TT's visibility_scores
        # input (Sec 4.1) - the accompanying face_mask isn't used here at all,
        # since inference never renders (see compute_visibility_and_mask's
        # docstring: both come from the same XSeg call regardless).
        _face_mask, visibility_ratio, valid = compute_visibility_and_mask(
            frame_bgr, cache_root, video_name, frame_index, args.detector_device, args.xseg_device,
            args.crop_scale, args.image_size, args.device,
        )
        frame_valid = valid and pixel_values is not None
        if not frame_valid:
            # Matches training's own crop-cache fallback (batched ops can't skip
            # individual samples) - discarded via encode_video's missing-frame
            # token averaging once flag_visibility_valid=False is set below.
            pixel_values = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)

        pixel_values_list.append(pixel_values.squeeze(0))
        visibility_list.append(visibility_ratio)
        valid_list.append(frame_valid)

    return {
        "video_name": video_name,
        "num_frames": len(frames),
        "pixel_values": pixel_values_list,
        "visibility": visibility_list,
        "valid": valid_list,
    }


def pad_video(video: dict, max_frames: int) -> dict:
    """Pads one video's per-frame lists to max_frames using this project's own
    frame_indices_for_segment/valid_mask_for_segment (start=0, since the whole
    video is one segment here, not a training-style sub-clip) - the same
    tail-repeat padding scheme training already relies on. `frame_indices` are
    the actual (repeated-at-the-tail) source frame numbers, matching how
    dataset_processing/dataloading/datasets.py itself feeds TT, not a plain
    arange - only distance between frame_indices matters (TT's docstring), so
    this is equivalent, but keeping it consistent with training avoids a
    silent behavioral difference."""
    num_frames = video["num_frames"]
    source_indices = frame_indices_for_segment(0, max_frames, num_frames)
    real_frame_mask = valid_mask_for_segment(0, max_frames, num_frames)

    return {
        "video_name": video["video_name"],
        "num_frames": num_frames,
        "pixel_values": torch.stack([video["pixel_values"][i] for i in source_indices], dim=0),
        # dtype=torch.float32 explicit: torch.tensor() on a list of plain Python
        # floats defaults to float64, which nn.MultiheadAttention's fused attention
        # kernel rejects outright (attn_mask dtype must match the query's dtype).
        "visibility": torch.tensor([video["visibility"][i] for i in source_indices], dtype=torch.float32),
        "flag_visibility_valid": torch.tensor([video["valid"][i] for i in source_indices], dtype=torch.bool),
        "frame_indices": torch.tensor(source_indices, dtype=torch.float32),
        "real_frame_mask": torch.tensor(real_frame_mask, dtype=torch.bool),
    }


def save_video_sample(
    out_path: str, video_name: str, real_params: dict[str, torch.Tensor], flame_out: dict | None,
) -> None:
    payload = {key: real_params[key].detach().cpu().numpy() for key in PARAM_KEYS}
    if flame_out is not None:
        payload["vertices"] = flame_out["vertices"].detach().cpu().numpy()
        payload["landmarks_fan"] = flame_out["landmarks_fan"].detach().cpu().numpy()
        payload["landmarks_mp"] = flame_out["landmarks_mp"].detach().cpu().numpy()
    np.savez(os.path.join(out_path, f"{video_name}.npz"), **payload)


def process_group(videos: list[dict], models: dict, args: argparse.Namespace) -> None:
    max_frames = max(video["num_frames"] for video in videos)
    padded = [pad_video(video, max_frames) for video in videos]

    clip_pixel_values = torch.stack([v["pixel_values"] for v in padded], dim=0).to(args.device)
    visibility_scores = torch.stack([v["visibility"] for v in padded], dim=0).to(args.device)
    frame_indices = torch.stack([v["frame_indices"] for v in padded], dim=0).to(args.device)
    flag_visibility_valid = torch.stack([v["flag_visibility_valid"] for v in padded], dim=0).to(args.device)
    real_frame_mask = torch.stack([v["real_frame_mask"] for v in padded], dim=0).to(args.device)

    encoded = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
        real_frame_mask=real_frame_mask,
    )

    for i, video in enumerate(padded):
        mask = real_frame_mask[i]
        real_params = {key: encoded[key][i][mask] for key in PARAM_KEYS}

        flame_out = None
        if args.save_vertices:
            flame_out, _camera = run_flame(models["flame"], real_params)

        save_video_sample(args.out_path, video["video_name"], real_params, flame_out)
        print(f"[ok] {video['video_name']} ({video['num_frames']} frames)")


def main() -> None:
    args = parse_args()
    cache_root = shared_cache_root()
    args.out_path = timestamped_out_dir(args.out_path)

    models = build_models(args.device, use_unet=False)
    step = load_available_checkpoint(models, args.checkpoint, args.device)

    video_paths = gather_video_paths(args.input_path)
    if not video_paths:
        raise ValueError(f"No videos found at {args.input_path}")

    detector = get_detector(args.detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)
    videos = [encode_one_video(path, detector, cache_root, args) for path in video_paths]

    for start in range(0, len(videos), args.batch_size):
        process_group(videos[start : start + args.batch_size], models, args)

    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"\nProcessed {len(videos)} videos ({status}). Saved outputs to {args.out_path}")


if __name__ == "__main__":
    main()
