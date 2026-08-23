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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.video_frames import (  # noqa: E402
    frame_indices_for_segment,
    valid_mask_for_segment,
)
from model import constants  # noqa: E402
from model.encoding import encode_video  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    FrameJob,
    build_models,
    load_available_checkpoint,
    make_crop_parse_pool,
    peek_num_expression_params,
    run_flame,
    run_parallel_crop_and_parse,
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
    parser.add_argument(
        "--tt_variant", type=str, default="original", choices=["original", "simple", "gated"],
        help="TemporalTransformer architecture --checkpoint was trained with (training/config.py's "
        "Stage2Config.tt_variant) - must match, since the variants have different parameter shapes "
        "for the 'tt' checkpoint key.",
    )
    parser.add_argument(
        "--tt_gamma", type=float, default=constants.TT_GATE_GAMMA,
        help="Visibility-gate sharpness, only used when --tt_variant gated; must match the "
        "checkpoint's training-time tt_gamma (not recoverable from the checkpoint itself).",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--xseg_device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--out_path", type=str, default="inference/output/videos")
    parser.add_argument(
        "--save_vertices", action="store_true", help="Also decode through FLAME and save vertices/landmarks.",
    )
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of VIDEOS per encode_video call.")
    parser.add_argument(
        "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
        help="Process-pool workers for the CPU-bound detect+crop step; XSeg itself runs as one "
        "batched GPU call afterward (see utils/inference_utils.py).",
    )
    return parser.parse_args()


def gather_video_paths(input_path: str) -> list[str]:
    if not os.path.isdir(input_path):
        raise ValueError(f"input_path '{input_path}' is not a directory")
    files = sorted(f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)
    return [os.path.join(input_path, f) for f in files]


def load_video_frames(video_path: str) -> tuple[list[np.ndarray], str]:
    """Sequential decode only (unchanged from before this change) - no
    detection/cropping here, that now happens via the process pool in
    process_video_group below."""
    frames, _fps, video_name = load_frames(video_path, image_seq=False, fps=30)
    if not frames:
        raise ValueError(f"No frames found in {video_path}")
    return frames, video_name


def assemble_video_from_results(
    video_name: str, num_frames: int, results: list, args: argparse.Namespace,
) -> dict:
    """Turns one video's FrameResults (from the process pool, keyed by
    frame_index) into the same per-video dict shape this script always fed
    into pad_video/process_group. visibility_ratio is kept unconditionally
    for TT's visibility_scores input (Sec 4.1) - this script never renders,
    so results' cropped_rgb is only used to build pixel_values, not kept."""
    by_frame = {result.frame_index: result for result in results}

    pixel_values_list, visibility_list, valid_list = [], [], []
    for frame_index in range(num_frames):
        result = by_frame[frame_index]
        frame_valid = result.valid and result.cropped_rgb is not None
        if frame_valid:
            pixel_values = torch.from_numpy(result.cropped_rgb).permute(2, 0, 1).float().to(args.device) / 255.0
        else:
            # Matches training's own crop-cache fallback (batched ops can't skip
            # individual samples) - discarded via encode_video's missing-frame
            # token averaging once flag_visibility_valid=False is set below.
            pixel_values = torch.zeros(3, args.image_size, args.image_size, device=args.device)

        pixel_values_list.append(pixel_values)
        visibility_list.append(result.visibility_ratio)
        valid_list.append(frame_valid)

    return {
        "video_name": video_name,
        "num_frames": num_frames,
        "pixel_values": pixel_values_list,
        "visibility": visibility_list,
        "valid": valid_list,
    }


def process_video_group(video_paths_group: list[str], executor, args: argparse.Namespace) -> list[dict]:
    """Decodes every video in this group (sequential per-video, as before),
    then flattens ALL of their frames into one job list submitted to the
    shared pool in a single call - balances load evenly across workers
    regardless of each video's individual length, rather than one worker per
    video (which would leave short videos idling while others are still
    being processed)."""
    decoded = [load_video_frames(path) for path in video_paths_group]

    jobs: list[FrameJob] = []
    for frames, video_name in decoded:
        jobs.extend(
            FrameJob(video_name, i, frame_bgr, args.crop_scale, args.image_size)
            for i, frame_bgr in enumerate(frames)
        )

    results = run_parallel_crop_and_parse(jobs, executor, args.xseg_device)

    results_by_video: dict[str, list] = {}
    for result in results:
        results_by_video.setdefault(result.video_key, []).append(result)

    return [
        assemble_video_from_results(video_name, len(frames), results_by_video[video_name], args)
        for frames, video_name in decoded
    ]


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
    args.out_path = timestamped_out_dir(args.out_path)

    models = build_models(
        args.device, use_unet=False, tt_variant=args.tt_variant, tt_gamma=args.tt_gamma,
        num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    step = load_available_checkpoint(models, args.checkpoint, args.device)

    video_paths = gather_video_paths(args.input_path)
    if not video_paths:
        raise ValueError(f"No videos found at {args.input_path}")

    # Created once and reused for every group below - pool startup (spawning
    # args.num_workers processes) is real overhead, not worth paying per group.
    executor = make_crop_parse_pool(args.num_workers)
    try:
        num_videos = 0
        for start in range(0, len(video_paths), args.batch_size):
            group_paths = video_paths[start : start + args.batch_size]
            videos = process_video_group(group_paths, executor, args)
            process_group(videos, models, args)
            num_videos += len(videos)
    finally:
        executor.shutdown()

    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"\nProcessed {num_videos} videos ({status}). Saved outputs to {args.out_path}")


if __name__ == "__main__":
    main()
