"""Single-image demo: SViT -> ComponentHeads -> FLAME -> Renderer [-> UNet]
(implementation-plan.md Sec 3: "Single image: skip step 2" - no TT). Mirrors
SMIRK's own baselines/smirk_experiments/demo_updated.py, but for this
project's own model. No trained checkpoint is required to run - omitting
--checkpoint sanity-tests the model at its FaRL-initialized-but-otherwise-
untrained state.

Usage:
    python inference/demo_images.py --input_path <image> [--checkpoint <path>]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from model.encoding import encode_image  # noqa: E402
from model.flame.masking import load_probabilities_per_flame_triangle  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image demo (SViT -> heads -> FLAME -> Renderer [-> UNet]).")
    parser.add_argument("--input_path", type=str, default="samples/no-occlusion/000006.jpg")
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
    # Only needed by mesh_based_mask_uniform_faces (inside render_2d_reconstruction) to
    # bias which mesh-surface points become the UNet's sparse real-pixel input - not
    # needed at all for the plain mesh render.
    face_probabilities = (
        load_probabilities_per_flame_triangle().to(args.device) if args.render_2d_recon else None
    )

    detector = get_detector(args.detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)
    image_bgr = cv2.imread(args.input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.input_path}")

    pixel_values, cropped_rgb = crop_and_tensor(image_bgr, detector, args.crop_scale, args.image_size, args.device)
    if pixel_values is None:
        raise RuntimeError(f"No face detected in {args.input_path}")

    face_mask = None
    if args.render_2d_recon:
        sample_id = os.path.splitext(os.path.basename(args.input_path))[0]
        face_mask, _visibility_ratio, valid = compute_visibility_and_mask(
            image_bgr, cache_root, sample_id, None, args.detector_device, args.xseg_device,
            args.crop_scale, args.image_size, args.device,
        )
        if not valid:
            raise RuntimeError(f"XSeg face-parsing failed for {args.input_path}")

    encoded = encode_image(models["svit"], models["heads"], pixel_values)
    flame_out = models["flame"](
        encoded["shape"], encoded["expression"], encoded["jaw"], encoded["eyelid"], encoded["rotation"],
    )
    cam_for_proj = torch.cat([encoded["scale"], encoded["translation"]], dim=-1)

    panels = [cropped_rgb]

    if args.render_mesh:
        render_out = models["renderer"](flame_out["vertices"], cam_for_proj)
        panels.append(tensor_to_uint8_rgb(render_out["rendered_img"]))

    if args.render_2d_recon:
        reconstructed = render_2d_reconstruction(
            models["flame"], models["renderer"], models["unet"], face_probabilities,
            encoded, pixel_values, face_mask,
        )
        panels.append(tensor_to_uint8_rgb(reconstructed))

    panel_image = make_panel(*panels)
    out_file = os.path.join(args.out_path, os.path.basename(args.input_path))
    cv2.imwrite(out_file, panel_image)

    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"[ok] wrote {out_file} ({status})")


if __name__ == "__main__":
    main()
