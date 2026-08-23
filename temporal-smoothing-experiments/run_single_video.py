"""Step 3, single-video CLI: raw (no-TT) vs. Gaussian x visibility-softmax
smoothed FLAME params for one video - renders a 3-panel comparison video
(original crop | mesh, raw | mesh, smoothed) and saves before/after
temporal-smoothness, occlusion-temporal-smoothness, and accurate-landmark-loss
metrics. r/sigma/T are CLI-configurable so different settings can be compared;
each run's output lands under its own r{r}_sigma{sigma}_T{T}/{timestamp}/
directory so sweeps never collide.

Usage:
    python temporal-smoothing-experiments/run_single_video.py \
        [--input_path <video_or_frame_dir>] [--checkpoint <path>] \
        [--radius 4] [--sigma 2.0] [--temperature 0.1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from utils.inference_utils import (  # noqa: E402
    build_models,
    load_available_checkpoint,
    make_panel,
    peek_num_expression_params,
    run_flame,
    tensor_to_uint8_rgb,
    timestamped_out_dir,
)
from utils.kernel_smoothing import smooth_encoded_params  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))
from metrics import summarize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import extract_flame_params, load_clip_frames  # noqa: E402
from metrics_utils import build_gt_detectors, compute_clip_metrics  # noqa: E402

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = "/projects/u6ga/sk3925_misc/checkpoints/stage2_50_AAB_UNet100_v3/step_00025999.pt"
DEFAULT_INPUT_PATH = (
    "/projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/"
    "features/fullFrame-210x260px/test/01April_2011_Friday_tagesschau-3377"
)
# Purely a rendering-memory concern (bounding Renderer's rasterization memory for
# a long video), unrelated to smoothing's own window radius - matches
# inference/demo_videos.py's identical constant/rationale.
RENDER_CHUNK_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-video temporal-smoothing experiment: raw vs. kernel-smoothed FLAME params.",
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--radius", "-r", type=int, default=4, help="Smoothing window radius (Sec 4.2-style).")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian kernel std (frames).")
    parser.add_argument("--temperature", "-T", type=float, default=0.1, help="Visibility softmax temperature.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--xseg_device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=25, help="Output FPS, used only for frame-dir inputs.")
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--out_path", type=str, default="temporal-smoothing-experiments/output/single_video")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = timestamped_out_dir(
        os.path.join(args.out_path, f"r{args.radius}_sigma{args.sigma}_T{args.temperature}")
    )

    start_time = time.perf_counter()
    last_time = start_time

    def log(msg: str) -> None:
        nonlocal last_time
        now = time.perf_counter()
        print(f"[run_single_video] {msg} (+{now - last_time:.1f}s, total {now - start_time:.1f}s)")
        last_time = now

    models = build_models(
        args.device, use_unet=False, num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    step = load_available_checkpoint(models, args.checkpoint, args.device)
    log(f"models built on {args.device} (checkpoint step {step})")

    frames, video_fps, video_name = load_clip_frames(args.input_path, args.fps)
    num_frames = len(frames)
    log(f"loaded {num_frames} frames from {args.input_path}")

    clip = extract_flame_params(
        frames, video_name, video_fps, models, args.device, args.xseg_device,
        args.crop_scale, args.image_size, args.num_workers,
    )
    log("extract_flame_params (no-TT) done")

    smoothed_encoded = smooth_encoded_params(
        clip.encoded, clip.visibility_scores, args.radius, args.sigma, args.temperature,
    )
    log(f"smoothing done (r={args.radius}, sigma={args.sigma}, T={args.temperature})")

    detectors = build_gt_detectors(args.device, args.image_size, args.crop_scale)
    errors = compute_clip_metrics(clip, smoothed_encoded, models["flame"], detectors, args.image_size)
    metrics_summary = {key: summarize(values) for key, values in errors.items()}
    log("metrics computed")

    out_width, out_height = args.image_size * 3, args.image_size
    out_file = os.path.join(out_dir, f"{video_name}.mp4")
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (out_width, out_height))

    for start in range(0, num_frames, RENDER_CHUNK_SIZE):
        end = min(start + RENDER_CHUNK_SIZE, num_frames)
        chunk_orig = {name: value[start:end] for name, value in clip.encoded.items()}
        chunk_smoothed = {name: value[start:end] for name, value in smoothed_encoded.items()}

        flame_out_orig, cam_orig = run_flame(models["flame"], chunk_orig)
        flame_out_smoothed, cam_smoothed = run_flame(models["flame"], chunk_smoothed)
        mesh_orig = models["renderer"](flame_out_orig["vertices"], cam_orig)["rendered_img"]
        mesh_smoothed = models["renderer"](flame_out_smoothed["vertices"], cam_smoothed)["rendered_img"]

        for i in range(end - start):
            idx = start + i
            panels = [
                clip.cropped_rgb[idx],
                tensor_to_uint8_rgb(mesh_orig[i : i + 1]),
                tensor_to_uint8_rgb(mesh_smoothed[i : i + 1]),
            ]
            writer.write(make_panel(*panels))
        log(f"rendered {end}/{num_frames} frames")

    writer.release()

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("[run_single_video] before vs. after (mean over valid frames):")
    for key, values in sorted(metrics_summary.items()):
        mean = values["mean"]
        print(f"    {key:>40s}: {mean:.4f}" if mean is not None else f"    {key:>40s}: n/a")

    print(f"[ok] wrote {out_file} and {os.path.join(out_dir, 'metrics.json')} "
          f"({num_frames} frames, total {time.perf_counter() - start_time:.1f}s)")


if __name__ == "__main__":
    main()

