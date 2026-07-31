"""Video demo: SViT -> TemporalTransformer -> ComponentHeads -> FLAME ->
Renderer [-> UNet] (implementation-plan.md Sec 3, video path). Mirrors SMIRK's
own baselines/smirk_experiments/demo_video_updated.py, but for this project's
own model: each frame is first passed through SViT individually, then the
WHOLE video is passed through TT in a single encode_video call - the new
O(N*w) local-attention TemporalTransformer (model/temporal.py) was built
precisely to support this, so no artificial clip-chunking is needed here. No
trained checkpoint is required to run.

Usage:
    python inference/demo_videos.py --input_path <video.mp4> [--checkpoint <path>]
    python inference/demo_videos.py --input_path <frame_dir> --image_seq
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from skimage.transform import warp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.encoding import encode_video  # noqa: E402
from model.flame.masking import load_probabilities_per_flame_triangle  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    FrameJob,
    build_models,
    load_available_checkpoint,
    make_crop_parse_pool,
    make_panel,
    render_2d_reconstruction,
    run_flame,
    run_parallel_crop_and_parse,
    tensor_to_uint8_rgb,
    timestamped_out_dir,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Purely a rendering-memory concern (bounding PyTorch3D rasterization/UNet memory
# for a long video) - unrelated to and not to be confused with TT's own windowing,
# which already sees the whole clip in one encode_video call below.
RENDER_CHUNK_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video demo (SViT -> TT -> heads -> FLAME -> Renderer [-> UNet]).")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--image_seq", action="store_true", help="Treat --input_path as a directory of frames.")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS, used only with --image_seq.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional trained checkpoint; omit to sanity-test the untrained model.",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--xseg_device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--out_path", type=str, default="inference/output/demo")
    parser.add_argument("--render_mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_2d_recon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--render_orig", action=argparse.BooleanOptionalAction, default=False,
        help="Warp renders back into the original frame's position/size instead of showing "
        "crop-space panels, matching baselines/smirk_experiments/demo_video_updated.py.",
    )
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--num_workers", type=int, default=min(32, os.cpu_count() or 1),
        help="Process-pool workers for the CPU-bound detect+crop step; XSeg itself runs as one "
        "batched GPU call afterward (see utils/inference_utils.py).",
    )
    return parser.parse_args()


def _warp_to_orig(rendered: torch.Tensor, tform, video_height: int, video_width: int) -> np.ndarray:
    """Crop-space (1,3,H,W) render -> original-frame-space (video_height,
    video_width,3) uint8 RGB. tform maps original-frame coords to crop-space
    coords (preprocessing/cropping.py's get_crop_transform, same convention
    baselines/smirk_experiments/demo_video_updated.py's own crop transform
    uses) - passing it directly as warp()'s inverse_map places the crop-space
    render back at its original position, exactly mirroring that script's own
    --render_orig path. tform=None (no face detected for this frame) -> black,
    since there's no crop position to place the render at."""
    if tform is None:
        return np.zeros((video_height, video_width, 3), dtype=np.uint8)
    rendered_np = tensor_to_uint8_rgb(rendered)
    return warp(rendered_np, tform, output_shape=(video_height, video_width), preserve_range=True).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if not args.render_mesh and not args.render_2d_recon:
        raise ValueError("At least one of --render_mesh / --render_2d_recon must be enabled")

    args.out_path = timestamped_out_dir(args.out_path)

    start_time = time.perf_counter()
    last_time = start_time

    def log(msg: str) -> None:
        nonlocal last_time
        now = time.perf_counter()
        print(f"[demo] {msg} (+{now - last_time:.1f}s, total {now - start_time:.1f}s)")
        last_time = now

    models = build_models(args.device, use_unet=args.render_2d_recon)
    log(f"models built on {args.device}")
    step = load_available_checkpoint(models, args.checkpoint, args.device)
    face_probabilities = (
        load_probabilities_per_flame_triangle().to(args.device) if args.render_2d_recon else None
    )

    frames, video_fps, video_name = load_frames(args.input_path, args.image_seq, args.fps)
    if not frames:
        raise ValueError(f"No frames found at {args.input_path}")
    num_frames = len(frames)
    video_height, video_width = frames[0].shape[:2]
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

    # visibility_ratio/face_mask both come from the same XSeg call (Sec 4.1's
    # visibility score IS the XSeg-derived ratio) - needed unconditionally for
    # TT's visibility_scores input, not just when --render_2d_recon is set.
    pixel_values_list, cropped_rgb_list, visibility_list, valid_list, face_mask_list, tform_list = (
        [], [], [], [], [], [],
    )
    for frame_index in range(num_frames):
        result = results_by_frame[frame_index]
        frame_valid = result.valid and result.cropped_rgb is not None
        if frame_valid:
            pixel_values = torch.from_numpy(result.cropped_rgb).permute(2, 0, 1).float().to(args.device) / 255.0
            cropped_rgb = result.cropped_rgb
        else:
            # RetinaFace/XSeg failed for this frame - fall back to a black frame,
            # matching training's own crop-cache fallback (batched ops can't skip
            # individual samples); the frame gets discarded via encode_video's own
            # missing-frame token averaging once flag_visibility_valid=False below.
            pixel_values = torch.zeros(3, args.image_size, args.image_size, device=args.device)
            cropped_rgb = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)

        pixel_values_list.append(pixel_values)
        cropped_rgb_list.append(cropped_rgb)
        visibility_list.append(result.visibility_ratio)
        valid_list.append(frame_valid)
        face_mask_list.append(torch.from_numpy(result.face_mask).unsqueeze(0).to(args.device))
        tform_list.append(result.tform)

    invalid_count = valid_list.count(False)
    if invalid_count:
        log(f"WARNING: {invalid_count}/{num_frames} frames failed face detection/parsing - using black-frame fallback for those")

    clip_pixel_values = torch.stack(pixel_values_list, dim=0).unsqueeze(0)
    # dtype=torch.float32 explicit: torch.tensor() on a list of plain Python floats
    # defaults to float64, which nn.MultiheadAttention's fused attention kernel
    # rejects outright (attn_mask dtype must exactly match the query tensor's dtype).
    visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=args.device).unsqueeze(0)
    frame_indices = torch.arange(num_frames, device=args.device).float().unsqueeze(0)
    flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=args.device).unsqueeze(0)

    print("[demo] running SViT + TemporalTransformer encode_video...")
    encoded = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
    )
    log("encode_video done")
    # Squeeze the batch dim (always 1 here - a single video) so FLAME/Renderer see
    # N frames as their own batch dimension, one full forward per frame's params.
    encoded = {name: value.squeeze(0) for name, value in encoded.items()}
    pixel_values_per_frame = clip_pixel_values.squeeze(0)  # (N,3,H,W)
    face_masks_per_frame = torch.cat(face_mask_list, dim=0) if args.render_2d_recon else None  # (N,H,W)

    num_panels = 1 + int(args.render_mesh) + int(args.render_2d_recon)
    if args.render_orig:
        out_width, out_height = video_width * num_panels, video_height
    else:
        out_width, out_height = args.image_size * num_panels, args.image_size
    out_file = os.path.join(args.out_path, f"{video_name}.mp4")
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (out_width, out_height))

    for start in range(0, num_frames, RENDER_CHUNK_SIZE):
        end = min(start + RENDER_CHUNK_SIZE, num_frames)
        chunk = {name: value[start:end] for name, value in encoded.items()}
        flame_out, cam_for_proj = run_flame(models["flame"], chunk)

        mesh_frames = None
        if args.render_mesh:
            render_out = models["renderer"](flame_out["vertices"], cam_for_proj)
            mesh_frames = render_out["rendered_img"]

        recon_frames = None
        if args.render_2d_recon:
            recon_frames = render_2d_reconstruction(
                models["flame"], models["renderer"], models["unet"], face_probabilities,
                chunk, pixel_values_per_frame[start:end], face_masks_per_frame[start:end],
                flag_visibility_valid.squeeze(0)[start:end],
            )

        for i in range(end - start):
            idx = start + i
            if args.render_orig:
                panels = [cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)]
                if mesh_frames is not None:
                    panels.append(_warp_to_orig(mesh_frames[i : i + 1], tform_list[idx], video_height, video_width))
                if recon_frames is not None:
                    panels.append(_warp_to_orig(recon_frames[i : i + 1], tform_list[idx], video_height, video_width))
            else:
                panels = [cropped_rgb_list[idx]]
                if mesh_frames is not None:
                    panels.append(tensor_to_uint8_rgb(mesh_frames[i : i + 1]))
                if recon_frames is not None:
                    panels.append(tensor_to_uint8_rgb(recon_frames[i : i + 1]))
            writer.write(make_panel(*panels))
        log(f"rendered {end}/{num_frames} frames")

    writer.release()
    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"[ok] wrote {out_file} ({num_frames} frames, {status}) (total {time.perf_counter() - start_time:.1f}s)")


if __name__ == "__main__":
    main()
