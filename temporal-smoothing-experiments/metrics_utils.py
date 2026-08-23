"""Step 3 (metrics half): GT (FAN/MediaPipe) landmark detection + before/after
(raw vs. smoothed) temporal-smoothness, occlusion-temporal-smoothness, and
accurate-landmark-loss metrics for one clip.

Reuses evaluation/eval_core.py's and evaluation/metrics.py's own formulas
directly (imported, not reimplemented) - this just runs each formula twice
(against extract.ClipEncoding's raw params and smoothing.smooth_encoded_params'
output) instead of once, against the same GT/visibility data computed once
here rather than the redundant XSeg pass evaluation/eval_core.py's own
compute_visibility_and_mask would otherwise trigger.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from model import constants  # noqa: E402
from model.flame.flame import FLAME  # noqa: E402
from preprocessing.cropping import get_cropped_face_box  # noqa: E402
from utils.inference_utils import decode_and_project  # noqa: E402
from utils.landmark_utils import (  # noqa: E402
    build_fan_predictor,
    build_mediapipe_detector,
    run_fan,
    run_mediapipe,
)

# evaluation/'s own modules (eval_core, metrics) resolve as bare top-level
# imports there, so they need evaluation/ on sys.path - same pattern
# inference/baselines/demo_videos_ours_no_temporal.py already uses.
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))
from eval_core import MEDIAPIPE_MODEL_PATH, OCCLUSION_VISIBILITY_DELTA_THRESHOLD, _visible_landmark_mask  # noqa: E402
from metrics import per_frame_euclidean_error_masked, per_frame_vertex_error  # noqa: E402

from extract import ClipEncoding  # noqa: E402

# Same curated 105-point FLAME/DECA/EMOCA-lineage MediaPipe embedding
# evaluation/methods/base.py's ReconstructionMethod.mediapipe_gt_indices() loads -
# a fixed data constant, not method-specific, so loaded directly here rather than
# instantiating a whole ReconstructionMethod just for this one array.
MEDIAPIPE_GT_INDICES = np.load(
    os.path.join(_REPO_ROOT, constants.FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH)
)["landmark_indices"]

METRIC_NAMES = [
    "fan_accurate_landmark_loss",
    "mediapipe_accurate_landmark_loss",
    "temporal_smoothness",
    "occlusion_temporal_smoothness",
]
VARIANTS = ["orig", "smoothed"]


@dataclass
class GTDetectors:
    fan_predictor: object
    mediapipe_detector: object
    fan_box: np.ndarray


def build_gt_detectors(device: str, crop_size: int, crop_scale: float) -> GTDetectors:
    return GTDetectors(
        fan_predictor=build_fan_predictor(device),
        mediapipe_detector=build_mediapipe_detector(MEDIAPIPE_MODEL_PATH),
        fan_box=get_cropped_face_box(image_size=crop_size, scale=crop_scale),
    )


def detect_gt_landmarks(cropped_rgb_list: list, detectors: GTDetectors) -> tuple[list, list]:
    """cropped_rgb_list: extract.ClipEncoding.cropped_rgb (RGB uint8 crops).
    run_fan/run_mediapipe both expect BGR input (they either pass rgb=False
    through, or convert BGR->RGB internally themselves) - converted back here
    since extract.py's crops are RGB (matching demo_videos.py's own convention).
    Returns (gt_fan_list, gt_mp_list), each entry (68,2)/(105,2) float or None
    (no face detected in that frame - e.g. the black-fallback frames extract.py
    substitutes for a failed crop)."""
    gt_fan_list, gt_mp_list = [], []
    for cropped_rgb in cropped_rgb_list:
        cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

        gt_fan, _scores = run_fan(detectors.fan_predictor, cropped_bgr, detectors.fan_box)
        gt_fan_list.append(gt_fan)

        gt_mp_full = run_mediapipe(detectors.mediapipe_detector, cropped_bgr)
        gt_mp = gt_mp_full[MEDIAPIPE_GT_INDICES, :2] if gt_mp_full is not None else None
        gt_mp_list.append(gt_mp)

    return gt_fan_list, gt_mp_list


def _landmark_and_temporal_errors(
    preds: list[dict[str, np.ndarray]],
    gt_fan_list: list,
    gt_mp_list: list,
    face_masks: list[torch.Tensor],
    visibility_scores: np.ndarray,
    crop_size: int,
) -> dict[str, np.ndarray]:
    """preds: utils.inference_utils.decode_and_project's per-frame output for one
    param set (raw or smoothed). Mirrors evaluation/eval_core.py's evaluate_clip
    formulas exactly (accurate-landmark-loss masking, temporal_smoothness,
    occlusion_temporal_smoothness), just factored out to run against either
    param set without duplicating the formulas themselves."""
    fan_errors, mp_errors = [], []
    for i, pred in enumerate(preds):
        face_mask = face_masks[i]
        gt_fan = gt_fan_list[i]
        fan_errors.append(per_frame_euclidean_error_masked(
            pred["fan"], gt_fan, _visible_landmark_mask(gt_fan, face_mask, crop_size)))
        gt_mp = gt_mp_list[i]
        mp_errors.append(per_frame_euclidean_error_masked(
            pred["mediapipe"], gt_mp, _visible_landmark_mask(gt_mp, face_mask, crop_size)))

    vertices = [pred["vertices"] for pred in preds]
    temporal_smoothness = [
        per_frame_vertex_error(vertices[i - 1], vertices[i]) for i in range(1, len(vertices))
    ]
    occlusion_temporal_smoothness = [
        per_frame_vertex_error(vertices[i - 1], vertices[i])
        if abs(visibility_scores[i] - visibility_scores[i - 1]) >= OCCLUSION_VISIBILITY_DELTA_THRESHOLD
        else np.nan
        for i in range(1, len(vertices))
    ]

    return {
        "fan_accurate_landmark_loss": np.array(fan_errors, dtype=np.float64),
        "mediapipe_accurate_landmark_loss": np.array(mp_errors, dtype=np.float64),
        "temporal_smoothness": np.array(temporal_smoothness, dtype=np.float64),
        "occlusion_temporal_smoothness": np.array(occlusion_temporal_smoothness, dtype=np.float64),
    }


def compute_variant_metrics(
    clip: ClipEncoding,
    encoded: dict[str, torch.Tensor],
    flame: FLAME,
    gt_fan_list: list,
    gt_mp_list: list,
    crop_size: int,
    variant_name: str,
) -> dict[str, np.ndarray]:
    """One param set (clip.encoded, or a smoothing.smooth_encoded_params() output)
    against already-detected GT landmarks (detect_gt_landmarks - shared across
    every variant of the same clip, since GT doesn't depend on smoothing) ->
    {f'{metric}_{variant_name}': errors} for metric in METRIC_NAMES. Factored out
    of compute_clip_metrics so a multi-setting sweep (run_multi_video.py) can
    compute the 'orig' variant once per clip and only rerun this for each
    smoothed (r, sigma, T) combo, instead of redoing GT detection and the
    unchanged 'orig' variant on every combo."""
    preds = decode_and_project(flame, encoded, crop_size)
    visibility_np = clip.visibility_scores.detach().cpu().numpy()
    variant_errors = _landmark_and_temporal_errors(
        preds, gt_fan_list, gt_mp_list, clip.face_masks, visibility_np, crop_size,
    )
    return {f"{metric_name}_{variant_name}": values for metric_name, values in variant_errors.items()}


def compute_clip_metrics(
    clip: ClipEncoding,
    smoothed_encoded: dict[str, torch.Tensor],
    flame: FLAME,
    detectors: GTDetectors,
    crop_size: int,
) -> dict[str, np.ndarray]:
    """Returns a flat dict keyed f'{metric}_{variant}' for metric in METRIC_NAMES,
    variant in VARIANTS ('orig'/'smoothed') - e.g. 'temporal_smoothness_orig',
    'fan_accurate_landmark_loss_smoothed'. Each value is a per-frame(-pair) error
    array (NaN where missing), ready for evaluation.metrics.summarize().

    Single-setting convenience wrapper (used by run_single_video.py, which only
    ever needs one (r, sigma, T) at a time) around compute_variant_metrics - a
    multi-setting sweep should call detect_gt_landmarks + compute_variant_metrics
    directly instead, to avoid redoing GT detection and the 'orig' variant per
    setting (see compute_variant_metrics's own docstring)."""
    gt_fan_list, gt_mp_list = detect_gt_landmarks(clip.cropped_rgb, detectors)
    errors: dict[str, np.ndarray] = {}
    errors.update(compute_variant_metrics(clip, clip.encoded, flame, gt_fan_list, gt_mp_list, crop_size, "orig"))
    errors.update(compute_variant_metrics(clip, smoothed_encoded, flame, gt_fan_list, gt_mp_list, crop_size, "smoothed"))
    return errors

