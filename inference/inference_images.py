"""Batch image inference: SViT -> ComponentHeads -> FLAME parameters (no
rendering) for any number of images (implementation-plan.md Sec 3, image
path, TT skipped). See inference/demo_images.py for the single-image,
always-renders sibling script.

Usage:
    python inference/inference_images.py --input_path <image_or_dir> [--checkpoint <path>] [--save_vertices]
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
from model import constants  # noqa: E402
from model.encoding import encode_image  # noqa: E402
from preprocessing.io import is_image_file  # noqa: E402
from utils.inference_utils import (  # noqa: E402
    DETECTOR_MODEL_NAME,
    DETECTOR_THRESHOLD,
    build_models,
    crop_and_tensor,
    load_available_checkpoint,
    peek_num_expression_params,
    run_flame,
    timestamped_out_dir,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARAM_KEYS = ["shape", "expression", "eyelid", "jaw", "scale", "rotation", "translation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image inference: SViT -> heads -> FLAME parameters.")
    parser.add_argument("--input_path", type=str, required=True, help="Image file or directory of images.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--tt_variant", type=str, default="original", choices=["original", "simple", "gated"],
        help="TemporalTransformer architecture --checkpoint was trained with (training/config.py's "
        "Stage2Config.tt_variant) - must match, since the variants have different parameter shapes "
        "for the 'tt' checkpoint key (unused by this image-only script, but still loaded).",
    )
    parser.add_argument(
        "--tt_gamma", type=float, default=constants.TT_GATE_GAMMA,
        help="Visibility-gate sharpness, only used when --tt_variant gated; must match the "
        "checkpoint's training-time tt_gamma (not recoverable from the checkpoint itself).",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--detector_device", type=str, default="cpu")
    parser.add_argument("--out_path", type=str, default="inference/output/images")
    parser.add_argument(
        "--save_vertices", action="store_true", help="Also decode through FLAME and save vertices/landmarks.",
    )
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def gather_image_paths(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if is_image_file(f))
        return [os.path.join(input_path, f) for f in files]
    raise ValueError(f"input_path '{input_path}' is not a file or directory")


def save_sample(
    out_path: str, basename: str, encoded: dict[str, torch.Tensor], index: int, flame_out: dict | None,
) -> None:
    payload = {key: encoded[key][index].detach().cpu().numpy() for key in PARAM_KEYS}
    if flame_out is not None:
        payload["vertices"] = flame_out["vertices"][index].detach().cpu().numpy()
        payload["landmarks_fan"] = flame_out["landmarks_fan"][index].detach().cpu().numpy()
        payload["landmarks_mp"] = flame_out["landmarks_mp"][index].detach().cpu().numpy()
    out_file = os.path.join(out_path, f"{os.path.splitext(basename)[0]}.npz")
    np.savez(out_file, **payload)


def process_batch(paths: list[str], models: dict, args: argparse.Namespace) -> int:
    """Returns the number of paths in this batch skipped (failed to load / no face)."""
    detector = get_detector(args.detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)
    batch_pixel_values, batch_basenames = [], []
    num_skipped = 0
    for path in paths:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"[skip] failed to load image: {path}")
            num_skipped += 1
            continue
        pixel_values, _cropped_rgb = crop_and_tensor(
            image_bgr, detector, args.crop_scale, args.image_size, args.device,
        )
        if pixel_values is None:
            print(f"[skip] no face detected: {path}")
            num_skipped += 1
            continue
        batch_pixel_values.append(pixel_values.squeeze(0))
        batch_basenames.append(os.path.basename(path))

    if not batch_pixel_values:
        return num_skipped

    pixel_values = torch.stack(batch_pixel_values, dim=0)
    encoded = encode_image(models["svit"], models["heads"], pixel_values)

    flame_out = None
    if args.save_vertices:
        flame_out, _camera = run_flame(models["flame"], encoded)

    for i, basename in enumerate(batch_basenames):
        save_sample(args.out_path, basename, encoded, i, flame_out)
        print(f"[ok] {basename}")

    return num_skipped


def main() -> None:
    args = parse_args()
    args.out_path = timestamped_out_dir(args.out_path)

    models = build_models(
        args.device, use_unet=False, tt_variant=args.tt_variant, tt_gamma=args.tt_gamma,
        num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    step = load_available_checkpoint(models, args.checkpoint, args.device)

    image_paths = gather_image_paths(args.input_path)
    if not image_paths:
        raise ValueError(f"No images found at {args.input_path}")

    num_skipped = 0
    for start in range(0, len(image_paths), args.batch_size):
        num_skipped += process_batch(image_paths[start : start + args.batch_size], models, args)

    num_processed = len(image_paths) - num_skipped
    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"\nProcessed {num_processed}/{len(image_paths)} images ({status}). Skipped {num_skipped}.")
    print(f"Saved outputs to {args.out_path}")


if __name__ == "__main__":
    main()
