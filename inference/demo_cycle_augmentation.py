"""Diagnostic: visualizes Pass B's expression-cycle augmentation pipeline
(training/stage2.py's compute_cycle_losses) on a single real image, for each
of SMIRK's four augmentation types (model/losses/cycle.py: perturbation,
permutation, template injection, zero-expression) - lets you inspect whether
the augmented poses/renders themselves already look implausible (column 2,
before the UNet ever touches them) versus the UNet's own reconstruction
(column 4) being where things go wrong.

All `--num_samples` draws are computed from the SAME input image repeated
into a batch (not multiple different faces) - this mirrors compute_cycle_
losses' per-type augmentation functions directly, but means `permutation`
(which borrows another group member's expression) degenerates to "the
original expression rescaled + jittered" here, since every row in the batch
starts identical; it's still informative about the scale/jitter component of
that augmentation, just not the cross-sample borrowing part.

Usage:
    python inference/demo_cycle_augmentation.py --checkpoint <path>
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.detector_pool import get_detector  # noqa: E402
from model import constants  # noqa: E402
from model.encoding import encode_image  # noqa: E402
from model.flame.masking import (  # noqa: E402
    load_probabilities_per_flame_triangle, masking, mesh_based_mask_uniform_faces, transfer_pixels,
)
from model.flame.renderer import transform_vertices  # noqa: E402
from model.losses.cycle import (  # noqa: E402
    _augment_eyelids, _augment_jaw, _augment_perturbation, _augment_permutation,
    _augment_template_injection, _augment_zero_expression, load_expression_templates,
)
from utils.inference_utils import (  # noqa: E402
    DETECTOR_MODEL_NAME, DETECTOR_THRESHOLD, build_models, crop_tensor_and_compute_xseg_mask,
    load_available_checkpoint, make_panel, peek_num_expression_params, tensor_to_uint8_rgb, timestamped_out_dir,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = "/projects/u6ga/sk3925_misc/checkpoints/stage2_AAB/step_00001999.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Pass B's expression-cycle augmentations (training/stage2.py's compute_cycle_losses) on a real image."
    )
    parser.add_argument("--input_path", type=str, default="samples/no-occlusion/000006.jpg")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tt_variant", type=str, default="original", choices=["original", "simple", "gated"],
        help="TemporalTransformer architecture --checkpoint was trained with (training/config.py's "
        "Stage2Config.tt_variant) - must match, since the variants have different parameter shapes "
        "for the 'tt' checkpoint key (unused by this single-image demo, but still loaded).",
    )
    parser.add_argument(
        "--tt_gamma", type=float, default=constants.TT_GATE_GAMMA,
        help="Visibility-gate sharpness, only used when --tt_variant gated; must match the "
        "checkpoint's training-time tt_gamma (not recoverable from the checkpoint itself).",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--detector_device", type=str, default="cpu")
    parser.add_argument("--xseg_device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--out_path", type=str, default="inference/output/cycle_augmentation_demo")
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_samples", type=int, default=4, help="Augmented draws per augmentation type.")
    return parser.parse_args()


def _run_augmented_batch(
    flame, renderer, unet, face_probabilities,
    encoded_repeated: dict[str, torch.Tensor], npoints1: torch.Tensor, coords: dict[str, torch.Tensor],
    aug_expression: torch.Tensor, aug_jaw: torch.Tensor, aug_eyelid: torch.Tensor,
    pixel_values_repeated: torch.Tensor, face_mask_repeated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exact render -> resample -> transfer-pixels -> mask -> UNet chain
    training/stage2.py's compute_cycle_losses runs per Pass B step (batched
    over every augmented draw at once), just without a backward pass - this
    script only inspects the forward outputs. Returns (augmented mesh render,
    masked UNet input, UNet reconstruction), each (N, 3, H, W)."""
    flame_out_aug = flame(
        encoded_repeated["shape"], aug_expression, aug_jaw, aug_eyelid, encoded_repeated["rotation"],
    )
    cam_for_proj = torch.cat([encoded_repeated["scale"], encoded_repeated["translation"]], dim=-1)
    render_out_aug = renderer(flame_out_aug["vertices"], cam_for_proj)

    npoints2, _coords = mesh_based_mask_uniform_faces(
        render_out_aug["transformed_vertices"], flame.faces_tensor, face_probabilities, coords=coords,
    )
    extra_points = transfer_pixels(pixel_values_repeated, npoints1, npoints2)
    background_mask = 1 - face_mask_repeated.unsqueeze(1)
    masked_img = masking(pixel_values_repeated, background_mask, extra_points)

    unet_input = torch.cat([render_out_aug["rendered_img"], masked_img], dim=1)
    reconstructed = unet(unet_input)
    return render_out_aug["rendered_img"], masked_img, reconstructed


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out_path = timestamped_out_dir(args.out_path)

    models = build_models(
        args.device, use_unet=True, tt_variant=args.tt_variant, tt_gamma=args.tt_gamma,
        num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    step = load_available_checkpoint(models, args.checkpoint, args.device)
    face_probabilities = load_probabilities_per_flame_triangle().to(args.device)
    templates = load_expression_templates()

    detector = get_detector(args.detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)
    image_bgr = cv2.imread(args.input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.input_path}")

    pixel_values, cropped_rgb, face_mask, _visibility_ratio, valid = crop_tensor_and_compute_xseg_mask(
        image_bgr, detector, args.xseg_device, args.crop_scale, args.image_size, args.device,
    )
    if pixel_values is None:
        raise RuntimeError(f"No face detected in {args.input_path}")
    if not valid:
        raise RuntimeError(f"XSeg face-parsing failed for {args.input_path}")

    encoded = encode_image(models["svit"], models["heads"], pixel_values)

    n = args.num_samples
    encoded_repeated = {k: v.repeat(n, *([1] * (v.dim() - 1))) for k, v in encoded.items()}
    pixel_values_repeated = pixel_values.repeat(n, 1, 1, 1)
    face_mask_repeated = face_mask.repeat(n, 1, 1)

    cam_for_proj = torch.cat([encoded_repeated["scale"], encoded_repeated["translation"]], dim=-1)
    flame_out = models["flame"](
        encoded_repeated["shape"], encoded_repeated["expression"], encoded_repeated["jaw"],
        encoded_repeated["eyelid"], encoded_repeated["rotation"],
    )
    transformed_vertices = transform_vertices(flame_out["vertices"], cam_for_proj)
    npoints1, coords = mesh_based_mask_uniform_faces(
        transformed_vertices, models["flame"].faces_tensor, face_probabilities,
    )

    aug_jaw_mild = _augment_jaw(encoded_repeated["jaw"])
    aug_eyelid_mild = _augment_eyelids(encoded_repeated["eyelid"])

    augmentations = {
        "perturbation": (_augment_perturbation(encoded_repeated["expression"]), aug_jaw_mild, aug_eyelid_mild),
        "permutation": (_augment_permutation(encoded_repeated["expression"]), aug_jaw_mild, aug_eyelid_mild),
        "template_injection": (
            _augment_template_injection(
                encoded_repeated["expression"], templates, constants.EXPRESSION_TEMPLATE_NUM_DIMS,
            ),
            aug_jaw_mild, aug_eyelid_mild,
        ),
        "zero_expression": (
            _augment_zero_expression(encoded_repeated["expression"]),
            torch.zeros_like(aug_jaw_mild),
            torch.rand_like(aug_eyelid_mild),
        ),
    }

    for aug_name, (aug_expression, aug_jaw, aug_eyelid) in augmentations.items():
        rendered_img, masked_img, reconstructed = _run_augmented_batch(
            models["flame"], models["renderer"], models["unet"], face_probabilities,
            encoded_repeated, npoints1, coords,
            aug_expression, aug_jaw, aug_eyelid,
            pixel_values_repeated, face_mask_repeated,
        )

        rows = [
            make_panel(
                cropped_rgb, tensor_to_uint8_rgb(rendered_img[i : i + 1]),
                tensor_to_uint8_rgb(masked_img[i : i + 1]), tensor_to_uint8_rgb(reconstructed[i : i + 1]),
            )
            for i in range(n)
        ]
        grid = cv2.vconcat(rows) if len(rows) > 1 else rows[0]
        out_file = os.path.join(args.out_path, f"{aug_name}.jpg")
        cv2.imwrite(out_file, grid)
        print(f"[ok] wrote {out_file}")

    status = f"checkpoint step {step}" if step is not None else "no checkpoint - sanity test"
    print(f"[done] {status}, input {args.input_path}, out dir {args.out_path}")


if __name__ == "__main__":
    main()