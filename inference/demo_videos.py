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

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from model.encoding import encode_video  # noqa: E402
from model.flame.masking import load_probabilities_per_flame_triangle  # noqa: E402
from preprocessing.io import load_frames  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    DETECTOR_MODEL_NAME,
    DETECTOR_THRESHOLD,
    build_models,
    compute_visibility_and_mask,
    crop_and_tensor,
    load_available_checkpoint,
    make_panel,
    render_2d_reconstruction,
    shared_cache_root,
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
    parser.add_argument("--detector_device", type=str, default="cpu")
    parser.add_argument("--xseg_device", type=str, default="cpu")
    parser.add_argument("--out_path", type=str, default="inference/output/demo")
    parser.add_argument("--render_mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render_2d_recon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.render_mesh and not args.render_2d_recon:
        raise ValueError("At least one of --render_mesh / --render_2d_recon must be enabled")

    cache_root = shared_cache_root()
    args.out_path = timestamped_out_dir(args.out_path)

    models = build_models(args.device, use_unet=args.render_2d_recon)
    step = load_available_checkpoint(models, args.checkpoint, args.device)
    face_probabilities = (
        load_probabilities_per_flame_triangle().to(args.device) if args.render_2d_recon else None
    )

    frames, video_fps, video_name = load_frames(args.input_path, args.image_seq, args.fps)
    if not frames:
        raise ValueError(f"No frames found at {args.input_path}")
    num_frames = len(frames)

    detector = get_detector(args.detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)

    pixel_values_list, cropped_rgb_list, visibility_list, valid_list, face_mask_list = [], [], [], [], []
    for frame_index, frame_bgr in enumerate(frames):
        pixel_values, cropped_rgb = crop_and_tensor(
            frame_bgr, detector, args.crop_scale, args.image_size, args.device,
        )
        # visibility_ratio/face_mask both come from the same XSeg call (Sec 4.1's
        # visibility score IS the XSeg-derived ratio) - needed unconditionally for
        # TT's visibility_scores input, not just when --render_2d_recon is set.
        face_mask, visibility_ratio, valid = compute_visibility_and_mask(
            frame_bgr, cache_root, video_name, frame_index, args.detector_device, args.xseg_device,
            args.crop_scale, args.image_size, args.device,
        )
        frame_valid = valid and pixel_values is not None
        if not frame_valid:
            # RetinaFace/XSeg failed for this frame - fall back to a black frame,
            # matching training's own crop-cache fallback (batched ops can't skip
            # individual samples); the frame gets discarded via encode_video's own
            # missing-frame token averaging once flag_visibility_valid=False below.
            pixel_values = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
            cropped_rgb = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)

        pixel_values_list.append(pixel_values)
        cropped_rgb_list.append(cropped_rgb)
        visibility_list.append(visibility_ratio)
        valid_list.append(frame_valid)
        face_mask_list.append(face_mask)

    clip_pixel_values = torch.stack([p.squeeze(0) for p in pixel_values_list], dim=0).unsqueeze(0)
    # dtype=torch.float32 explicit: torch.tensor() on a list of plain Python floats
    # defaults to float64, which nn.MultiheadAttention's fused attention kernel
    # rejects outright (attn_mask dtype must exactly match the query tensor's dtype).
    visibility_scores = torch.tensor(visibility_list, dtype=torch.float32, device=args.device).unsqueeze(0)
    frame_indices = torch.arange(num_frames, device=args.device).float().unsqueeze(0)
    flag_visibility_valid = torch.tensor(valid_list, dtype=torch.bool, device=args.device).unsqueeze(0)

    encoded = encode_video(
        models["svit"], models["tt"], models["heads"],
        clip_pixel_values, visibility_scores, frame_indices, flag_visibility_valid,
    )
    # Squeeze the batch dim (always 1 here - a single video) so FLAME/Renderer see
    # N frames as their own batch dimension, one full forward per frame's params.
    encoded = {name: value.squeeze(0) for name, value in encoded.items()}
    pixel_values_per_frame = clip_pixel_values.squeeze(0)  # (N,3,H,W)
    face_masks_per_frame = torch.cat(face_mask_list, dim=0) if args.render_2d_recon else None  # (N,H,W)

    out_width = args.image_size * (1 + int(args.render_mesh) + int(args.render_2d_recon))
    out_file = os.path.join(args.out_path, f"{video_name}.mp4")
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (out_width, args.image_size))

    for start in range(0, num_frames, RENDER_CHUNK_SIZE):
        end = min(start + RENDER_CHUNK_SIZE, num_frames)
        chunk = {name: value[start:end] for name, value in encoded.items()}
        flame_out = models["flame"](
            chunk["shape"], chunk["expression"], chunk["jaw"], chunk["eyelid"], chunk["rotation"],
        )
        cam_for_proj = torch.cat([chunk["scale"], chunk["translation"]], dim=-1)

        mesh_frames = None
        if args.render_mesh:
            render_out = models["renderer"](flame_out["vertices"], cam_for_proj)
            mesh_frames = render_out["rendered_img"]

        recon_frames = None
        if args.render_2d_recon:
            recon_frames = render_2d_reconstruction(
                models["flame"], models["renderer"], models["unet"], face_probabilities,
                chunk, pixel_values_per_frame[start:end], face_masks_per_frame[start:end],
            )

        for i in range(end - start):
            panels = [cropped_rgb_list[start + i]]
            if mesh_frames is not None:
                panels.append(tensor_to_uint8_rgb(mesh_frames[i : i + 1]))
            if recon_frames is not None:
                panels.append(tensor_to_uint8_rgb(recon_frames[i : i + 1]))
            writer.write(make_panel(*panels))

    writer.release()
    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"[ok] wrote {out_file} ({num_frames} frames, {status})")


if __name__ == "__main__":
    main()
