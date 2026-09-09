import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cropping import build_retinaface_detector, get_cropped_face_box  # noqa: E402
from utils.landmark_utils import build_mediapipe_detector, build_fan_predictor  # noqa: E402
from utils.inference_utils import compute_visibility_and_mask  # noqa: E402
from model.losses.landmark import landmark_visibility_mask  # noqa: E402

import gt_cache
from metrics import per_frame_euclidean_error, per_frame_euclidean_error_masked, per_frame_vertex_error
from methods.smirk_method import SmirkMethod
from methods.ours_method import OursKernelSmoothMethod, OursNoTemporalMethod, OursFullMethod
from methods.pixel3dmm_method import Pixel3dmmMethod
from methods.emica_method import EmicaMethod

METHOD_REGISTRY = {
    'smirk': SmirkMethod,
    'ours_no_temporal': OursNoTemporalMethod,
    'ours_full': OursFullMethod,
    'ours_kernel_smooth': OursKernelSmoothMethod,
    'pixel3dmm': Pixel3dmmMethod,
    'emica': EmicaMethod,
}

# Edit these lists to change which metrics / landmark sets run.
METRICS = ['landmark', 'accurate_landmark', 'temporal_smoothness', 'occlusion_temporal_smoothness']
# METRICS = ['temporal_smoothness']
LANDMARK_SETS = ['fan', 'mediapipe']  # only used if 'landmark' in METRICS

# Same physical cache root/dataset bucket OursFullMethod's predict_video already
# uses (methods/ours_method.py) for its own visibility scoring, and the same root
# dataset_processing/config/dataloader.yaml's face_parsing_cache_root points training
# at - keyed by (dataset, clip_id, frame_index), not by which method is being
# evaluated, so warming it with one method's run (e.g. ours_no_temporal) makes every
# later method's run against the same clips hit the cache for free. Living on
# /projects (Lustre), not /home, also matters: a home-directory quota is far smaller
# than this cache grows to across a full dataset's worth of frames.
VISIBILITY_CACHE_ROOT = "/lus/lfs1aip2/projects/u6ga/sk3925_datasets/face_parsing_cache"
# Same crop_cache_root dataset_processing/config/dataloader.yaml points training
# at - gt_cache.get_cropped_frame wraps dataset_processing/dataloading/crop_cache.py's
# own get_cropped_face, so this lands in the same entries training's own
# dataloading already populated (free cache hits), and is shared across every
# method/run evaluated through this framework (see gt_cache.py's own docstring).
CROP_CACHE_ROOT = "/lus/lfs1aip2/projects/u6ga/sk3925_datasets/face_crop_cache"
# New, dedicated to gt_cache.get_gt_landmarks' own entry format (68pt FAN + 105pt
# MediaPipe, raw pixel space) - NOT dataloader.yaml's landmark_cache_root, whose
# entries are a different shape/space (17pt FAN, [-1,1] normalized) meant for the
# training loss - see gt_cache.py's own module docstring for why those aren't
# interchangeable.
EVAL_GT_LANDMARK_CACHE_ROOT = "/lus/lfs1aip2/projects/u6ga/sk3925_datasets/eval_gt_landmark_cache"
OCCLUSION_VISIBILITY_DELTA_THRESHOLD = 0.1

# Stripped from clip_id before it's used as any of this module's cache sample_ids
# (visibility, crop, GT-landmark) - only matters for video-file datasets (e.g.
# how2sign): dataset_processing/indexers/index_how2sign.py's own sample_id is the
# clip's filename stem (no extension), so a video-mode clip_id (which keeps
# run_evaluation_dataset.py's ".mp4" basename) has to have that extension
# stripped to land in the same cache entries training's own data loading already
# populated for that dataset's test split. A no-op for image-seq datasets
# (csl_daily, phoenix2014t), whose clip_id is already an extension-less folder
# name matching the indexer's sample_id directly.
_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def _cache_sample_id(clip_id):
    root, ext = os.path.splitext(clip_id)
    return root if ext.lower() in _VIDEO_EXTENSIONS else clip_id

# GT-detection concern, identical regardless of which method (--method) is being
# evaluated, so this is a fixed constant rather than a threaded parameter - the main
# project's own copy of this asset (not SMIRK's baselines/smirk_experiments/ copy,
# since GT detection doesn't depend on SMIRK).
MEDIAPIPE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'face_landmarker.task')


def result_keys():
    """Names of the per-frame(-pair) error arrays evaluate_clip() will produce, given METRICS/LANDMARK_SETS."""
    keys = []
    if 'landmark' in METRICS:
        keys.extend(LANDMARK_SETS)
    if 'landmark' in METRICS and 'accurate_landmark' in METRICS:
        keys.extend(f'{name}_accurate_landmark_loss' for name in LANDMARK_SETS)
    if 'temporal_smoothness' in METRICS:
        keys.append('temporal_smoothness')
    if 'occlusion_temporal_smoothness' in METRICS:
        keys.append('occlusion_temporal_smoothness')
    return keys


@dataclass
class Evaluators:
    method: object
    face_detector: object
    fan_predictor: object
    mediapipe_detector: object
    mediapipe_gt_indices: object
    device: str
    dataset_name: str


def build_evaluators(method_name, device, crop_size, crop_scale=1.4, checkpoint_path=None, dataset_name="inference",
                      method_kwargs=None):
    method = METHOD_REGISTRY[method_name]()
    method.setup(device, crop_size=crop_size, crop_scale=crop_scale, checkpoint_path=checkpoint_path,
                 **(method_kwargs or {}))

    face_detector = build_retinaface_detector(device)
    fan_predictor = build_fan_predictor(device) if 'fan' in LANDMARK_SETS and 'landmark' in METRICS else None
    mediapipe_detector = build_mediapipe_detector(MEDIAPIPE_MODEL_PATH) \
        if 'mediapipe' in LANDMARK_SETS and 'landmark' in METRICS else None
    mediapipe_gt_indices = method.mediapipe_gt_indices() \
        if 'mediapipe' in LANDMARK_SETS and 'landmark' in METRICS else None

    return Evaluators(method, face_detector, fan_predictor, mediapipe_detector, mediapipe_gt_indices,
                       device, dataset_name)


def _visible_landmark_mask(gt_xy, face_mask, gt_crop_size):
    if gt_xy is None or face_mask is None:
        return None
    gt_norm = torch.from_numpy(gt_xy / gt_crop_size * 2 - 1).float().unsqueeze(0).to(face_mask.device)
    return landmark_visibility_mask(face_mask, gt_norm).squeeze(0).cpu().numpy()


def evaluate_clip(frames, crop_scale, crop_size, evaluators, clip_id, vis_writers=None):
    vis_writers = vis_writers or {}
    landmark_sets = LANDMARK_SETS if 'landmark' in METRICS else []
    track_mesh = 'temporal_smoothness' in METRICS or 'occlusion_temporal_smoothness' in METRICS
    track_visibility = 'occlusion_temporal_smoothness' in METRICS or 'accurate_landmark' in METRICS
    track_accurate_landmark = 'landmark' in METRICS and 'accurate_landmark' in METRICS

    gt_crop_size = evaluators.method.gt_crop_size or crop_size
    scale_to_gt = gt_crop_size / crop_size

    errors = {name: [] for name in landmark_sets}
    if track_accurate_landmark:
        errors.update({f'{name}_accurate_landmark_loss': [] for name in landmark_sets})
    per_frame_vertices = []
    # Fixed for every frame in this clip (crop_scale/gt_crop_size don't vary per-frame) -
    # see get_cropped_face_box's docstring for why this avoids a second, redundant
    # RetinaFace call on the already-cropped image just to get FAN a box.
    fan_box = get_cropped_face_box(image_size=gt_crop_size, scale=crop_scale)

    sample_id = _cache_sample_id(clip_id)

    cropped_frames = [
        gt_cache.get_cropped_frame(
            CROP_CACHE_ROOT, evaluators.dataset_name, sample_id, i, frame,
            evaluators.face_detector, crop_scale, crop_size,
        )
        for i, frame in enumerate(frames)
    ]

    visibility_scores = []
    face_masks = []
    if track_visibility:
        for i, frame in enumerate(frames):
            mask, visibility_ratio, valid = compute_visibility_and_mask(
                frame, VISIBILITY_CACHE_ROOT, sample_id, i,
                evaluators.device, evaluators.device, crop_scale, crop_size, evaluators.device,
                dataset=evaluators.dataset_name,
            )
            visibility_scores.append(visibility_ratio if valid else np.nan)
            face_masks.append(mask if valid else None)

    preds = evaluators.method.predict_video(cropped_frames, frames, clip_id)

    for i, (cropped, pred) in enumerate(zip(cropped_frames, preds)):
        if cropped is None:
            for name in landmark_sets:
                errors[name].append(np.nan)
                if track_accurate_landmark:
                    errors[f'{name}_accurate_landmark_loss'].append(np.nan)
                if name in vis_writers:
                    vis_writers[name].write(np.zeros((gt_crop_size, gt_crop_size, 3), dtype=np.uint8))
            if track_mesh:
                per_frame_vertices.append(None)
            continue

        gt_cropped = cropped if gt_crop_size == crop_size else cv2.resize(cropped, (gt_crop_size, gt_crop_size))
        frame_face_mask = face_masks[i] if track_visibility else None

        if 'fan' in landmark_sets:
            gt_fan = gt_cache.get_gt_fan(
                EVAL_GT_LANDMARK_CACHE_ROOT, evaluators.dataset_name, sample_id, i, gt_crop_size,
                gt_cropped, evaluators.fan_predictor, fan_box,
            )
            pred_fan = pred.get('fan')
            pred_fan_scaled = pred_fan * scale_to_gt if pred_fan is not None else None
            errors['fan'].append(per_frame_euclidean_error(pred_fan_scaled, gt_fan))
            if 'fan' in vis_writers:
                vis_writers['fan'].write(_draw_overlay(gt_cropped, gt_fan, pred_fan_scaled))
            if track_accurate_landmark:
                errors['fan_accurate_landmark_loss'].append(per_frame_euclidean_error_masked(
                    pred_fan_scaled, gt_fan, _visible_landmark_mask(gt_fan, frame_face_mask, gt_crop_size)))

        if 'mediapipe' in landmark_sets:
            gt_mp = gt_cache.get_gt_mediapipe(
                EVAL_GT_LANDMARK_CACHE_ROOT, evaluators.dataset_name, sample_id, i, gt_crop_size,
                gt_cropped, evaluators.mediapipe_detector, evaluators.mediapipe_gt_indices,
            )
            pred_mp = pred.get('mediapipe')
            pred_mp_scaled = pred_mp * scale_to_gt if pred_mp is not None else None
            errors['mediapipe'].append(per_frame_euclidean_error(pred_mp_scaled, gt_mp))
            if 'mediapipe' in vis_writers:
                vis_writers['mediapipe'].write(_draw_overlay(gt_cropped, gt_mp, pred_mp_scaled))
            if track_accurate_landmark:
                errors['mediapipe_accurate_landmark_loss'].append(per_frame_euclidean_error_masked(
                    pred_mp_scaled, gt_mp, _visible_landmark_mask(gt_mp, frame_face_mask, gt_crop_size)))

        if track_mesh:
            per_frame_vertices.append(pred.get('vertices'))

    if 'temporal_smoothness' in METRICS:
        errors['temporal_smoothness'] = [
            per_frame_vertex_error(per_frame_vertices[i - 1], per_frame_vertices[i])
            for i in range(1, len(per_frame_vertices))
        ]

    if 'occlusion_temporal_smoothness' in METRICS:
        errors['occlusion_temporal_smoothness'] = [
            per_frame_vertex_error(per_frame_vertices[i - 1], per_frame_vertices[i])
            if abs(visibility_scores[i] - visibility_scores[i - 1]) >= OCCLUSION_VISIBILITY_DELTA_THRESHOLD
            else np.nan
            for i in range(1, len(per_frame_vertices))
        ]

    return {name: np.array(values, dtype=np.float64) for name, values in errors.items()}


_COLOR_GT = (0, 255, 0)
_COLOR_PRED = (0, 0, 255)


def _draw_overlay(image, gt_xy, pred_xy):
    vis = image.copy()
    if gt_xy is not None and not np.isnan(gt_xy).any():
        for x, y in gt_xy:
            cv2.circle(vis, (int(x), int(y)), 1, _COLOR_GT, -1)
    if pred_xy is not None and not np.isnan(pred_xy).any():
        for x, y in pred_xy:
            cv2.circle(vis, (int(x), int(y)), 1, _COLOR_PRED, -1)
    return vis
