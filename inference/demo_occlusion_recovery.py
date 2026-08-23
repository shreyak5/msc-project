"""Demo: visualizes Pass C's synthetic-occlusion recovery (occlusion-
handling.md, training/stage2.py's run_pass_c) on a real video - randomly
marks some real, currently-visible frames as "occluded" (same independent
Bernoulli(--occlusion_prob) draw run_pass_c uses), feeds TT the same
corrupted visibility signal training does for those frames (routing them
through encode_video's existing _fill_missing_frame_tokens neighbor-
averaging, model/encoding.py), and renders the resulting mesh next to the
uncorrupted baseline (TT with the real, unoccluded visibility signal) so the
two can be compared frame by frame. Ground truth 3D isn't available outside
training, so the baseline mesh - not a real 3D target - is the reference
"how well did it recover" is judged against.

Usage:
    python inference/demo_occlusion_recovery.py --input_path <video.mp4>
    python inference/demo_occlusion_recovery.py --input_path <frame_dir> --image_seq
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
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
    make_panel,
    peek_num_expression_params,
    run_flame,
    run_parallel_crop_and_parse,
    tensor_to_uint8_rgb,
    timestamped_out_dir,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = "/projects/u6ga/sk3925_misc/checkpoints/stage2_AC_new_occ_extreme/step_00008999.pt"
# Purely a rendering-memory concern, same role as demo_videos.py's own constant of
# the same name - unrelated to TT's windowing (encode_video sees the whole clip at once).
RENDER_CHUNK_SIZE = 32
# RGB, not BGR: applied to cropped_rgb_list panels (RGB order, per
# utils/inference_utils.py) before make_panel's own RGB->BGR conversion.
OCCLUSION_BORDER_COLOR_RGB = (255, 0, 0)
OCCLUSION_BORDER_THICKNESS = 4
# Params compared for the end-of-run recovery-error summary - the ones TT
# actually refines per frame (shape/rotation/translation are per-clip-ish or
# camera-only and less informative about "did TT recover this frame's face").
DIFF_PARAM_NAMES = ("expression", "jaw", "eyelid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Pass C's synthetic-occlusion recovery (run_pass_c) on a real video."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--image_seq", action="store_true", help="Treat --input_path as a directory of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS, used only with --image_seq.")
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
    parser.add_argument("--out_path", type=str, default="inference/output/occlusion_recovery_demo")
    parser.add_argument(
        "--occlusion_prob", type=float, default=0.25,
        help="Per-eligible-frame probability of being synthetically marked occluded, independently "
        "per frame - same Bernoulli draw training/stage2.py's run_pass_c uses for Pass C. Training "
        "itself defaults to 0.1 (training/config.py's pass_c_synthetic_occlusion_prob); this demo "
        "defaults higher purely so enough occluded frames land on screen to eyeball recovery quality.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for which frames get marked occluded.")
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
        help="Process-pool workers for the CPU-bound detect+crop step; XSeg itself runs as one "
        "batched GPU call afterward (see utils/inference_utils.py).",
    )
    return parser.parse_args()


def _draw_occlusion_border(panel: np.ndarray) -> np.ndarray:
    """Marks a crop-space RGB panel as synthetically occluded with a red
    border + label, drawn in place on a copy so the original crop is
    unaffected (it's reused for the un-occluded comparison video, if any)."""
    marked = panel.copy()
    h, w = marked.shape[:2]
    cv2.rectangle(marked, (0, 0), (w - 1, h - 1), OCCLUSION_BORDER_COLOR_RGB, OCCLUSION_BORDER_THICKNESS)
    cv2.putText(
        marked, "OCCLUDED", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OCCLUSION_BORDER_COLOR_RGB, 2, cv2.LINE_AA,
    )
    return marked


def main() -> None:
    args = parse_args()
    args.out_path = timestamped_out_dir(args.out_path)

    start_time = time.perf_counter()
    last_time = start_time

    def log(msg: str) -> None:
        nonlocal last_time
        now = time.perf_counter()
        print(f"[demo] {msg} (+{now - last_time:.1f}s, total {now - start_time:.1f}s)")
        last_time = now

    models = build_models(
        args.device, use_unet=False, tt_variant=args.tt_variant, tt_gamma=args.tt_gamma,
        num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    log(f"models built on {args.device}")
    step = load_available_checkpoint(models, args.checkpoint, args.device)

    frames, video_fps, video_name = load_frames(args.input_path, args.image_seq, args.fps)
    if not frames:
        raise ValueError(f"No frames found at {args.input_path}")
    num_frames = len(frames)
    log(f"loaded {num_frames} frames from {args.input_path}")

    jobs = [
        FrameJob(video_name, i, frame_bgr, args.crop_scale, args.image_size)
        for i, frame_bgr in enumerate(frames)
    ]
    print(f"[demo] running detect+crop+XSeg-parse for {num_frames} frames across {args.num_workers} workers...")
    executor = make_crop_parse_pool(args.num_workers)
    try:
        results = run_parallel_crop_and_parse(jobs, executor, args.xseg_device)
    finally:
        executor.shutdown()
    log(f"crop+parse done ({num_frames} frames)")
    results_by_frame = {result.frame_index: result for result in results}

    pixel_values_list, cropped_rgb_list, visibility_list, valid_list = [], [], [], []
    for frame_index in range(num_frames):
        result = results_by_frame[frame_index]
        frame_valid = result.valid and result.cropped_rgb is not None
        if frame_valid:
            pixel_values = torch.from_numpy(result.cropped_rgb).permute(2, 0, 1).float().to(args.device) / 255.0
            cropped_rgb = result.cropped_rgb
        else:
            # RetinaFace/XSeg failed for this frame - fall back to a black frame,
            # matching training's own crop-cache fallback; discarded via encode_video's
            # own missing-frame token averaging once flag_visibility_valid=False below.
            pixel_values = torch.zeros(3, args.image_size, args.image_size, device=args.device)
            cropped_rgb = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)

        pixel_values_list.append(pixel_values)
        cropped_rgb_list.append(cropped_rgb)
        visibility_list.append(result.visibility_ratio)
        valid_list.append(frame_valid)

    invalid_count = valid_list.count(False)
    if invalid_count:
        log(f"WARNING: {invalid_count}/{num_frames} frames failed face detection/parsing - using black-frame fallback for those")

    clip_pixel_values = torch.stack(pixel_values_list, dim=0).unsqueeze(0)
    visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=args.device).unsqueeze(0)
    frame_indices = torch.arange(num_frames, device=args.device).float().unsqueeze(0)
    flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=args.device).unsqueeze(0)

    # Same eligibility + Bernoulli draw as training/stage2.py's run_pass_c: only
    # real, currently-visible frames can be picked (no valid_mask term here since
    # this demo has no tail-padding - every frame loaded is a real frame).
    torch.manual_seed(args.seed)
    eligible = flag_visibility_valid
    occlusion_mask = eligible & (torch.rand(1, num_frames, device=args.device) < args.occlusion_prob)
    num_occluded = int(occlusion_mask.sum().item())
    log(f"marked {num_occluded}/{num_frames} frames as synthetically occluded (prob={args.occlusion_prob})")

    visibility_for_tt = visibility_scores.masked_fill(occlusion_mask, 0.0)
    flag_visibility_for_tt = flag_visibility_valid & ~occlusion_mask

    print("[demo] running baseline (uncorrupted) SViT + TemporalTransformer encode_video...")
    encoded_baseline = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
    )
    log("baseline encode_video done")

    print("[demo] running recovered (synthetically occluded) encode_video...")
    encoded_recovered = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_for_tt, frame_indices, flag_visibility_for_tt,
    )
    log("recovered encode_video done")

    occlusion_mask_list = occlusion_mask.squeeze(0).cpu().tolist()

    # Recovery-error summary: mean per-parameter L2 diff between baseline and
    # recovered, split occluded vs. non-occluded, as a numeric complement to
    # the video (occluded frames diverging from baseline is expected/desired
    # here - it's TT relying on temporal context instead of hidden pixels;
    # non-occluded frames diverging would flag TT changing untouched frames).
    print("[demo] recovery error (mean L2 dist, baseline vs. recovered):")
    for name in DIFF_PARAM_NAMES:
        diff = (encoded_baseline[name] - encoded_recovered[name]).squeeze(0).norm(dim=-1)  # (N,)
        occ_diff = diff[occlusion_mask.squeeze(0)]
        clean_diff = diff[~occlusion_mask.squeeze(0)]
        occ_mean = occ_diff.mean().item() if occ_diff.numel() else float("nan")
        clean_mean = clean_diff.mean().item() if clean_diff.numel() else float("nan")
        print(f"    {name:>10s}: occluded={occ_mean:.4f}  non-occluded={clean_mean:.4f}")

    encoded_baseline = {name: value.squeeze(0) for name, value in encoded_baseline.items()}
    encoded_recovered = {name: value.squeeze(0) for name, value in encoded_recovered.items()}

    out_width, out_height = args.image_size * 3, args.image_size
    out_file = os.path.join(args.out_path, f"{video_name}.mp4")
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (out_width, out_height))

    for start in range(0, num_frames, RENDER_CHUNK_SIZE):
        end = min(start + RENDER_CHUNK_SIZE, num_frames)
        chunk_baseline = {name: value[start:end] for name, value in encoded_baseline.items()}
        chunk_recovered = {name: value[start:end] for name, value in encoded_recovered.items()}

        flame_out_baseline, cam_baseline = run_flame(models["flame"], chunk_baseline)
        mesh_baseline = models["renderer"](flame_out_baseline["vertices"], cam_baseline)["rendered_img"]
        flame_out_recovered, cam_recovered = run_flame(models["flame"], chunk_recovered)
        mesh_recovered = models["renderer"](flame_out_recovered["vertices"], cam_recovered)["rendered_img"]

        for i in range(end - start):
            idx = start + i
            crop_panel = cropped_rgb_list[idx]
            if occlusion_mask_list[idx]:
                crop_panel = _draw_occlusion_border(crop_panel)
            panels = [
                crop_panel,
                tensor_to_uint8_rgb(mesh_baseline[i : i + 1]),
                tensor_to_uint8_rgb(mesh_recovered[i : i + 1]),
            ]
            writer.write(make_panel(*panels))
        log(f"rendered {end}/{num_frames} frames")

    writer.release()
    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(
        f"[ok] wrote {out_file} ({num_frames} frames, {num_occluded} synthetically occluded, {status}) "
        f"(total {time.perf_counter() - start_time:.1f}s)"
    )
    print("[ok] panels: [crop (red border = synthetically occluded)] [baseline mesh] [recovered mesh]")


if __name__ == "__main__":
    main()
