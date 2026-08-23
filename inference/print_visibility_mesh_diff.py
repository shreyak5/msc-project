"""Diagnostic: prints per-frame face-visibility score alongside a temporal
mesh-motion diff for a real video - mesh_diff[t] is the mean per-vertex L2
distance between the TT-recovered FLAME mesh at frame t-1 and frame t+1
(centered, skipping frame t itself), so a mesh wobble/jump can be eyeballed
against a nearby dip in visibility[t]. No occlusion is synthetically injected
here - encode_video runs on the real, unmodified visibility signal, same as
demo_videos.py's default (no --render_no_tt) path.

Usage:
    python inference/print_visibility_mesh_diff.py --input_path <video.mp4>
    python inference/print_visibility_mesh_diff.py --input_path <frame_dir> --image_seq
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = "/projects/u6ga/sk3925_misc/checkpoints/stage2_AC_new_occ_extreme/step_00008999.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-frame visibility score + centered mesh-motion diff (vertices[t-1] vs vertices[t+1])."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--image_seq", action="store_true", help="Treat --input_path as a directory of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Frame-loading FPS, used only with --image_seq.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
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
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
        help="Process-pool workers for the CPU-bound detect+crop step; XSeg itself runs as one "
        "batched GPU call afterward (see utils/inference_utils.py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = build_models(
        args.device, use_unet=False, tt_variant=args.tt_variant, tt_gamma=args.tt_gamma,
        num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    step = load_available_checkpoint(models, args.checkpoint, args.device)

    frames, _video_fps, video_name = load_frames(args.input_path, args.image_seq, args.fps)
    if not frames:
        raise ValueError(f"No frames found at {args.input_path}")
    num_frames = len(frames)

    jobs = [
        FrameJob(video_name, i, frame_bgr, args.crop_scale, args.image_size)
        for i, frame_bgr in enumerate(frames)
    ]
    print(f"[diag] running detect+crop+XSeg-parse for {num_frames} frames across {args.num_workers} workers...")
    executor = make_crop_parse_pool(args.num_workers)
    try:
        results = run_parallel_crop_and_parse(jobs, executor, args.xseg_device)
    finally:
        executor.shutdown()
    results_by_frame = {result.frame_index: result for result in results}

    pixel_values_list, visibility_list, valid_list = [], [], []
    for frame_index in range(num_frames):
        result = results_by_frame[frame_index]
        frame_valid = result.valid and result.cropped_rgb is not None
        if frame_valid:
            pixel_values = torch.from_numpy(result.cropped_rgb).permute(2, 0, 1).float().to(args.device) / 255.0
        else:
            # RetinaFace/XSeg failed for this frame - same black-frame fallback as demo_videos.py;
            # discarded via encode_video's own missing-frame token averaging below.
            pixel_values = torch.zeros(3, args.image_size, args.image_size, device=args.device)
        pixel_values_list.append(pixel_values)
        visibility_list.append(result.visibility_ratio)
        valid_list.append(frame_valid)

    clip_pixel_values = torch.stack(pixel_values_list, dim=0).unsqueeze(0)
    visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=args.device).unsqueeze(0)
    frame_indices = torch.arange(num_frames, device=args.device).float().unsqueeze(0)
    flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=args.device).unsqueeze(0)

    print("[diag] running SViT + TemporalTransformer encode_video...")
    encoded = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
    )
    encoded = {name: value.squeeze(0) for name, value in encoded.items()}
    flame_out, _camera = run_flame(models["flame"], encoded)
    vertices = flame_out["vertices"]  # (N, V, 3)

    print(f"{'frame':>6s}  {'visibility':>10s}  {'mesh_diff(t-1,t+1)':>20s}")
    for t in range(num_frames):
        if t == 0 or t == num_frames - 1:
            diff_str = "N/A"
        else:
            diff = (vertices[t + 1] - vertices[t - 1]).norm(dim=-1).mean().item()
            diff_str = f"{diff:.4f}"
        print(f"{t:>6d}  {visibility_list[t]:>10.4f}  {diff_str:>20s}")

    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"[diag] done ({num_frames} frames, {status})")


if __name__ == "__main__":
    main()
