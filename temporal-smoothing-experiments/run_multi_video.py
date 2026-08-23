"""Step 3, multi-video CLI: same raw-vs-smoothed pipeline as run_single_video.py,
run over a fixed set of clips (edit VIDEO_PATHS below to change which clips are
evaluated - same edit-to-change convention evaluation/eval_core.py's own
METRICS/LANDMARK_SETS use), swept across a fixed set of (r, sigma, T) settings
(edit RADII/SIGMAS/TEMPERATURES below - Cartesian product of all three). Never
builds a Renderer or writes a video (per the project's "1 video -> save video, N
videos -> don't" convention) - only before/after metrics are computed.

Extraction (crop+XSeg+SViT) and GT (FAN/MediaPipe) landmark detection don't
depend on (r, sigma, T), and neither does the 'orig' (unsmoothed) baseline - so
each clip is only extracted/GT-detected/orig-scored ONCE, and only the smoothing
+ smoothed-metrics step (cheap) reruns per (clip, setting) pair. Results are
printed per (clip, setting), then aggregated two ways: per-setting (across all
clips - results.csv/summary.json, one folder per setting) and across all
settings (one comparison table/comparison.csv, to see which setting wins).

Usage:
    python temporal-smoothing-experiments/run_multi_video.py [--checkpoint <path>]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from utils.inference_utils import (  # noqa: E402
    build_models,
    load_available_checkpoint,
    peek_num_expression_params,
)
from utils.kernel_smoothing import smooth_encoded_params  # noqa: E402

sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))
from metrics import summarize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import extract_flame_params, load_clip_frames  # noqa: E402
from metrics_utils import (  # noqa: E402
    METRIC_NAMES,
    VARIANTS,
    build_gt_detectors,
    compute_variant_metrics,
    detect_gt_landmarks,
)

# Edit this list to change which clips are evaluated. Mixes video files
# (how2sign) and frame directories (csl-daily, phoenix2014t) - image_seq is
# auto-detected per path (load_clip_frames), so both kinds can sit in one list.
VIDEO_PATHS = [
    "/projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test/S000040_P0008_T00",
    "/projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test/S000185_P0004_T00",
    "/projects/u6ga/sk3925_datasets/sign_datasets/csl-daily/test/S000201_P0000_T00",
    "/projects/u6ga/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/_FzvMVnR_aU_2-10-rgb_front.mp4",
    "/projects/u6ga/sk3925_datasets/sign_datasets/how2sign/test_rgb_front_clips/_FzvMVnR_aU_3-10-rgb_front.mp4",
    "/projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/"
    "features/fullFrame-210x260px/test/01April_2011_Friday_tagesschau-3374",
    "/projects/u6ga/sk3925_datasets/sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/"
    "features/fullFrame-210x260px/test/01April_2011_Friday_tagesschau-3377",
]

# Edit these to change which smoothing settings are compared - Cartesian product
# of all three (e.g. 2 radii x 2 sigmas x 1 temperature = 4 settings swept).
RADII = [4]
SIGMAS = [0.5, 1, 2]
TEMPERATURES = [0.1, 0.05]

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = "/projects/u6ga/sk3925_misc/checkpoints/stage2_50_AAB_UNet100_v3/step_00025999.pt"
RESULT_KEYS = [f"{metric}_{variant}" for metric in METRIC_NAMES for variant in VARIANTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-video temporal-smoothing experiment: raw vs. kernel-smoothed FLAME params, "
        "aggregated across a fixed clip list (VIDEO_PATHS) and swept across a fixed setting list "
        "(RADII x SIGMAS x TEMPERATURES) - edit those constants above to change either.",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--xseg_device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--crop_scale", type=float, default=1.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=30, help="Output FPS, used only for frame-dir inputs.")
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--out_path", type=str, default="temporal-smoothing-experiments/output/multi_video")
    return parser.parse_args()


def _combo_label(radius: int, sigma: float, temperature: float) -> str:
    return f"r={radius} sigma={sigma} T={temperature}"


def _format_side_by_side(clip_summary: dict[str, dict]) -> list[str]:
    """clip_summary: {f'{metric}_{variant}': summarize()-dict}. One line per
    metric, orig/smoothed means shown side by side for quick visual comparison."""
    lines = []
    for metric in METRIC_NAMES:
        orig_mean = clip_summary[f"{metric}_orig"]["mean"]
        smoothed_mean = clip_summary[f"{metric}_smoothed"]["mean"]
        orig_str = f"{orig_mean:.4f}" if orig_mean is not None else "n/a"
        smoothed_str = f"{smoothed_mean:.4f}" if smoothed_mean is not None else "n/a"
        lines.append(f"        {metric:>35s}: orig={orig_str:>10s}  smoothed={smoothed_str:>10s}")
    return lines


class _ComboState:
    """Per-(radius, sigma, temperature) accumulator: its own results.csv (one
    row per clip) plus the running per-clip means summary.json/the final
    comparison table are built from."""

    def __init__(self, out_dir: str, fieldnames: list[str]):
        self.out_dir = out_dir
        self.results_path = os.path.join(out_dir, "results.csv")
        self.summary_path = os.path.join(out_dir, "summary.json")
        self.results_file = open(self.results_path, "w", newline="")
        self.writer = csv.DictWriter(self.results_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.per_clip_means = {key: [] for key in RESULT_KEYS}
        self.num_skipped = 0
        self.overall_mean: dict[str, float | None] = {}
        self.overall_std: dict[str, float | None] = {}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    combos = list(itertools.product(RADII, SIGMAS, TEMPERATURES))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M")  # shared across every combo in this sweep

    start_time = time.perf_counter()

    models = build_models(
        args.device, use_unet=False, num_expression_params=peek_num_expression_params(args.checkpoint),
    )
    step = load_available_checkpoint(models, args.checkpoint, args.device)
    print(f"[run_multi_video] models built on {args.device} (checkpoint step {step})")
    print(f"[run_multi_video] sweeping {len(combos)} setting(s): "
          + ", ".join(_combo_label(*combo) for combo in combos))

    detectors = build_gt_detectors(args.device, args.image_size, args.crop_scale)

    fieldnames = (
        ["name"]
        + [f"{key}_mean" for key in RESULT_KEYS]
        + [f"{key}_std" for key in RESULT_KEYS]
        + [f"{key}_valid_frames" for key in RESULT_KEYS]
        + [f"{key}_total_frames" for key in RESULT_KEYS]
    )
    combo_states = {}
    for combo in combos:
        radius, sigma, temperature = combo
        out_dir = os.path.join(args.out_path, f"r{radius}_sigma{sigma}_T{temperature}", run_stamp)
        os.makedirs(out_dir, exist_ok=True)
        combo_states[combo] = _ComboState(out_dir, fieldnames)

    num_clip_skipped = 0  # extraction/GT-detection failures - these skip every combo for that clip

    try:
        for i, clip_path in enumerate(VIDEO_PATHS, start=1):
            name = os.path.basename(clip_path.rstrip("/"))
            try:
                frames, video_fps, video_name = load_clip_frames(clip_path, args.fps)
                clip = extract_flame_params(
                    frames, video_name, video_fps, models, args.device, args.xseg_device,
                    args.crop_scale, args.image_size, args.num_workers,
                )
                gt_fan_list, gt_mp_list = detect_gt_landmarks(clip.cropped_rgb, detectors)
                orig_errors = compute_variant_metrics(
                    clip, clip.encoded, models["flame"], gt_fan_list, gt_mp_list, args.image_size, "orig",
                )
            except Exception as e:
                num_clip_skipped += 1
                print(f"[{i}/{len(VIDEO_PATHS)}] SKIPPED {name}: {e}")
                continue

            print(f"[{i}/{len(VIDEO_PATHS)}] {name}:")
            for combo in combos:
                radius, sigma, temperature = combo
                state = combo_states[combo]
                try:
                    smoothed_encoded = smooth_encoded_params(clip.encoded, clip.visibility_scores, radius, sigma, temperature)
                    smoothed_errors = compute_variant_metrics(
                        clip, smoothed_encoded, models["flame"], gt_fan_list, gt_mp_list, args.image_size, "smoothed",
                    )
                except Exception as e:
                    state.num_skipped += 1
                    print(f"    [{_combo_label(*combo)}] SKIPPED: {e}")
                    continue

                errors = {**orig_errors, **smoothed_errors}
                clip_summary = {key: summarize(errors[key]) for key in RESULT_KEYS}

                row = {"name": name}
                for key in RESULT_KEYS:
                    row[f"{key}_mean"] = clip_summary[key]["mean"]
                    row[f"{key}_std"] = clip_summary[key]["std"]
                    row[f"{key}_valid_frames"] = clip_summary[key]["num_valid_frames"]
                    row[f"{key}_total_frames"] = clip_summary[key]["num_frames"]
                    if clip_summary[key]["mean"] is not None:
                        state.per_clip_means[key].append(clip_summary[key]["mean"])
                state.writer.writerow(row)
                state.results_file.flush()

                print(f"    [{_combo_label(*combo)}]")
                for line in _format_side_by_side(clip_summary):
                    print(line)
    finally:
        for state in combo_states.values():
            state.results_file.close()

    num_clips_total = len(VIDEO_PATHS)
    for combo, state in combo_states.items():
        radius, sigma, temperature = combo
        state.overall_mean = {key: (float(np.mean(vals)) if vals else None) for key, vals in state.per_clip_means.items()}
        state.overall_std = {key: (float(np.std(vals)) if vals else None) for key, vals in state.per_clip_means.items()}
        num_skipped = num_clip_skipped + state.num_skipped
        summary = {
            "num_videos_total": num_clips_total,
            "num_videos_processed": num_clips_total - num_skipped,
            "num_videos_skipped": num_skipped,
            "radius": radius,
            "sigma": sigma,
            "temperature": temperature,
            "overall_mean": state.overall_mean,
            "overall_std": state.overall_std,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        with open(state.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Orig baseline doesn't depend on (r, sigma, T) - print once, from whichever
    # combo actually has data (all should agree, barring a combo-specific failure).
    baseline_state = next((s for s in combo_states.values() if s.overall_mean.get(f"{METRIC_NAMES[0]}_orig") is not None), None)
    if baseline_state is not None:
        print("\n[run_multi_video] === Orig baseline (unsmoothed, same across every setting) ===")
        for metric in METRIC_NAMES:
            mean = baseline_state.overall_mean[f"{metric}_orig"]
            print(f"    {metric:>35s}: {mean:.4f}" if mean is not None else f"    {metric:>35s}: n/a")

    print("\n[run_multi_video] === Settings comparison (smoothed mean across all clips) ===")
    comparison_fieldnames = ["radius", "sigma", "temperature"] + [f"{metric}_smoothed_mean" for metric in METRIC_NAMES] \
        + [f"{metric}_smoothed_std" for metric in METRIC_NAMES]
    comparison_rows = []
    for combo in combos:
        radius, sigma, temperature = combo
        state = combo_states[combo]
        parts = []
        row = {"radius": radius, "sigma": sigma, "temperature": temperature}
        for metric in METRIC_NAMES:
            mean = state.overall_mean.get(f"{metric}_smoothed")
            std = state.overall_std.get(f"{metric}_smoothed")
            row[f"{metric}_smoothed_mean"] = mean
            row[f"{metric}_smoothed_std"] = std
            parts.append(f"{metric}={mean:.4f}" if mean is not None else f"{metric}=n/a")
        comparison_rows.append(row)
        print(f"    [{_combo_label(*combo)}]  " + "  ".join(parts))

    comparison_path = os.path.join(args.out_path, f"comparison_{run_stamp}.csv")
    with open(comparison_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\n[ok] per-setting results/summaries under {args.out_path}/r*_sigma*_T*/{run_stamp}/, "
          f"comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()

