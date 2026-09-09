from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_processing.dataloading.config import load_dataloader_config  # noqa: E402
from dataset_processing.dataloading.datasets import VideoFaceDataset  # noqa: E402
from model import constants  # noqa: E402
from model.config import COMPONENT_TOKENS  # noqa: E402
from model.encoding import _decode_params, _expression_dim, _fill_missing_frame_tokens, _pool_identity  # noqa: E402
from utils.inference_utils import build_models, load_available_checkpoint, peek_num_expression_params  # noqa: E402

DATASETS_WITH_MANIFEST = ["csl_daily", "how2sign", "phoenix2014t"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="phoenix2014t", choices=DATASETS_WITH_MANIFEST)
    parser.add_argument(
        "--tt_variant", type=str, default="original", choices=["original", "simple", "gated"],
        help="Must match whatever --checkpoint was actually trained with (training/config.py's "
        "Stage2Config.tt_variant) - see utils/inference_utils.py's build_models docstring for why "
        "this isn't auto-detectable from the checkpoint file.",
    )
    parser.add_argument("--tt_gamma", type=float, default=constants.TT_GATE_GAMMA, help="Only used when --tt_variant gated.")
    parser.add_argument(
        "--pool_identity", action="store_true",
        help="Apply identity pooling (model/encoding.py's _pool_identity) to the post-TT path, matching "
        "Stage2Config.pass_c_identity_pooling=true - use this when probing a checkpoint trained with "
        "that flag on, so the probe reflects what the checkpoint actually does at train/inference time.",
    )
    parser.add_argument(
        "--sample_id", type=str, default=None,
        help="Which video to probe (dataset_processing/manifests/<dataset>.jsonl's sample_id column). "
        "Omit to auto-pick the lowest-in-clip-visibility segment out of --scan_limit randomly sampled "
        "train segments.",
    )
    parser.add_argument(
        "--scan_limit", type=int, default=200,
        help="Only used when --sample_id is omitted - how many random train segments to scan (reading "
        "each one's cached visibility_ratio is cheap, but scanning a whole large dataset isn't) before "
        "picking the one with the lowest in-clip visibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_config", type=str, default="dataset_processing/config/dataloader.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace, cfg) -> VideoFaceDataset:
    cat_cfg = cfg.categories["2d_video"]
    return VideoFaceDataset(
        dataset_name=args.dataset,
        manifest_path=f"dataset_processing/manifests/{args.dataset}.jsonl",
        split="train",
        crop_cache_root=cfg.crop_cache_root,
        image_size=cfg.image_size,
        crop_scale=cfg.crop_scale,
        detector_device=cfg.detector.device,
        detector_threshold=cfg.detector.threshold,
        detector_model_name=cfg.detector.model_name,
        with_flame=False,
        max_frames=cat_cfg.max_frames,
        with_mica=False,
        mica_cache_root=cfg.mica_cache_root,
        mica_device=cfg.mica_device,
        with_landmarks=False,
        landmark_cache_root=cfg.landmark_cache_root,
        fan_device=cfg.fan_device,
        with_face_mask=False,
        face_parsing_cache_root=cfg.face_parsing_cache_root,
        xseg_device=cfg.xseg_device,
    )


def pick_segment(ds: VideoFaceDataset, args: argparse.Namespace) -> dict:
    """Returns the __getitem__ dict for whichever segment the probe should run
    on: the requested --sample_id's segment with the lowest in-clip
    visibility (if there are several), or - when --sample_id is omitted - the
    lowest-in-clip-visibility segment out of a random --scan_limit-sized
    sample of the whole train index."""
    if args.sample_id is not None:
        candidates = [i for i, (row, _start, _total) in enumerate(ds.index) if row.sample_id == args.sample_id]
        if not candidates:
            raise ValueError(f"no segments found for sample_id={args.sample_id!r} in {args.dataset}")
    else:
        rng = random.Random(args.seed)
        candidates = rng.sample(range(len(ds.index)), min(args.scan_limit, len(ds.index)))

    best_item, best_min_vis = None, 2.0
    for i in candidates:
        item = ds[i]
        vis = item["visibility_ratio"][item["valid_mask"]]
        if vis.numel() == 0:
            continue
        min_vis = vis.min().item()
        if min_vis < best_min_vis:
            best_min_vis, best_item = min_vis, item
    if best_item is None:
        raise ValueError("no valid segment found (every candidate had zero real frames?)")
    print(f"picked sample_id={best_item['subject_id']!r}, min in-clip visibility={best_min_vis:.4f}")
    return best_item


def main() -> None:
    args = parse_args()
    device = args.device

    cfg = load_dataloader_config(args.dataloader_config)
    ds = build_dataset(args, cfg)
    item = pick_segment(ds, args)

    num_expression_params = peek_num_expression_params(args.checkpoint)
    models = build_models(
        device, use_unet=False, tt_variant=args.tt_variant, tt_gamma=args.tt_gamma,
        num_expression_params=num_expression_params,
    )
    step = load_available_checkpoint(models, args.checkpoint, device)
    print(f"checkpoint step: {step}")
    svit, heads, tt = models["svit"], models["heads"], models["tt"]

    pixel_values = item["pixel_values"].unsqueeze(0).to(device)  # (1, N, 3, H, W)
    visibility_scores = item["visibility_ratio"].unsqueeze(0).to(device)
    flag_visibility_valid = item["flag_visibility_valid"].unsqueeze(0).to(device)
    real_frame_mask = item["valid_mask"].unsqueeze(0).to(device)
    batch_size, num_frames = pixel_values.shape[:2]
    # arange, not frame_indices_for_segment: matches training/stage2.py's own
    # run_pass_c construction (the pattern every checkpoint here was actually
    # trained against), not the whole-video-relative indices inference
    # scripts use.
    frame_indices = torch.arange(num_frames, device=device).unsqueeze(0).expand(batch_size, -1)

    vis_np = item["visibility_ratio"].numpy()
    valid_np = item["valid_mask"].numpy()

    with torch.no_grad():
        flat_pixel_values = pixel_values.reshape(batch_size * num_frames, *pixel_values.shape[2:])
        component_outputs = svit(flat_pixel_values)
        component_names = [t.name for t in COMPONENT_TOKENS]
        tokens = torch.stack(
            [component_outputs[name].reshape(batch_size, num_frames, -1) for name in component_names], dim=2
        )
        tokens = _fill_missing_frame_tokens(tokens, real_frame_mask, flag_visibility_valid)

        features_pre = {name: tokens[:, :, i, :] for i, name in enumerate(component_names)}
        decoded_pre = _decode_params(heads(features_pre), _expression_dim(heads))

        refined = tt(tokens, visibility_scores, frame_indices, valid_mask=real_frame_mask)
        features_post = {name: refined[:, :, i, :] for i, name in enumerate(component_names)}
        decoded_post = _decode_params(heads(features_post), _expression_dim(heads))
        if args.pool_identity:
            decoded_post["shape"] = _pool_identity(decoded_post["shape"], real_frame_mask)

    print("\nper-frame jaw param (pre-TT vs post-TT vs |delta|):")
    jaw_pre, jaw_post = decoded_pre["jaw"][0], decoded_post["jaw"][0]
    jaw_delta = (jaw_post - jaw_pre).norm(dim=-1)
    for n in range(num_frames):
        if not valid_np[n]:
            continue
        print(
            f"  frame {n:2d} vis={vis_np[n]:.3f}  |jaw_pre|={jaw_pre[n].norm():.4f}  "
            f"|jaw_post|={jaw_post[n].norm():.4f}  |delta|={jaw_delta[n]:.5f}"
        )

    print("\nper-frame expression param (pre-TT vs post-TT vs |delta|):")
    expr_pre, expr_post = decoded_pre["expression"][0], decoded_post["expression"][0]
    expr_delta = (expr_post - expr_pre).norm(dim=-1)
    for n in range(num_frames):
        if not valid_np[n]:
            continue
        print(
            f"  frame {n:2d} vis={vis_np[n]:.3f}  |expr_pre|={expr_pre[n].norm():.4f}  "
            f"|expr_post|={expr_post[n].norm():.4f}  |delta|={expr_delta[n]:.5f}"
        )

    print("\nper-frame token delta norm by component (shape, expression, jaw, camera) - pre-TT vs post-TT tokens:")
    token_delta_norm = (refined - tokens).norm(dim=-1)[0]  # (N, 4)
    for n in range(num_frames):
        if not valid_np[n]:
            continue
        print(f"  frame {n:2d} vis={vis_np[n]:.3f}  {token_delta_norm[n].tolist()}")


if __name__ == "__main__":
    main()

