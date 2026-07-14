import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cropping import build_retinaface_detector, crop_face, get_cropped_face_box  # noqa: E402
from utils.landmark_utils import build_mediapipe_detector, build_fan_predictor, run_mediapipe, run_fan  # noqa: E402

from metrics import per_frame_euclidean_error, per_frame_vertex_error
from methods.smirk_method import SmirkMethod

METHOD_REGISTRY = {
    'smirk': SmirkMethod,
}

# Edit these lists to change which metrics / landmark sets run.
METRICS = ['landmark', 'temporal_smoothness']
# METRICS = ['temporal_smoothness']
LANDMARK_SETS = ['fan', 'mediapipe']  # only used if 'landmark' in METRICS


def result_keys():
    """Names of the per-frame(-pair) error arrays evaluate_clip() will produce, given METRICS/LANDMARK_SETS."""
    keys = []
    if 'landmark' in METRICS:
        keys.extend(LANDMARK_SETS)
    if 'temporal_smoothness' in METRICS:
        keys.append('temporal_smoothness')
    return keys


@dataclass
class Evaluators:
    method: object
    face_detector: object
    fan_predictor: object
    mediapipe_detector: object
    mediapipe_gt_indices: object


def build_evaluators(method_name, device, crop_size, mediapipe_model_path):
    method = METHOD_REGISTRY[method_name]()
    method.setup(device, crop_size=crop_size)

    face_detector = build_retinaface_detector(device)
    fan_predictor = build_fan_predictor(device) if 'fan' in LANDMARK_SETS and 'landmark' in METRICS else None
    mediapipe_detector = build_mediapipe_detector(mediapipe_model_path) \
        if 'mediapipe' in LANDMARK_SETS and 'landmark' in METRICS else None
    mediapipe_gt_indices = method.mediapipe_gt_indices() \
        if 'mediapipe' in LANDMARK_SETS and 'landmark' in METRICS else None

    return Evaluators(method, face_detector, fan_predictor, mediapipe_detector, mediapipe_gt_indices)


def evaluate_clip(frames, crop_scale, crop_size, evaluators, vis_writers=None):
    """Runs the crop -> GT detect -> method.predict -> per-frame error loop over one clip's frames.

    Returns a dict keyed by result_keys(): 'fan'/'mediapipe' arrays have one entry per
    frame (NaN where missing); 'temporal_smoothness' has one entry per consecutive
    frame pair (length len(frames) - 1, NaN if either frame in the pair is missing).
    """
    vis_writers = vis_writers or {}
    landmark_sets = LANDMARK_SETS if 'landmark' in METRICS else []
    track_mesh = 'temporal_smoothness' in METRICS

    errors = {name: [] for name in landmark_sets}
    per_frame_vertices = []
    # Fixed for every frame in this clip (crop_scale/crop_size don't vary per-frame) -
    # see get_cropped_face_box's docstring for why this avoids a second, redundant
    # RetinaFace call on the already-cropped image just to get FAN a box.
    fan_box = get_cropped_face_box(image_size=crop_size, scale=crop_scale)

    for frame in frames:
        cropped, _tform = crop_face(frame, evaluators.face_detector, scale=crop_scale, image_size=crop_size)

        if cropped is None:
            for name in landmark_sets:
                errors[name].append(np.nan)
                if name in vis_writers:
                    vis_writers[name].write(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
            if track_mesh:
                per_frame_vertices.append(None)
            continue

        pred = evaluators.method.predict(cropped)

        if 'fan' in landmark_sets:
            gt_fan, _scores = run_fan(evaluators.fan_predictor, cropped, fan_box)
            errors['fan'].append(per_frame_euclidean_error(pred.get('fan'), gt_fan))
            if 'fan' in vis_writers:
                vis_writers['fan'].write(_draw_overlay(cropped, gt_fan, pred.get('fan')))

        if 'mediapipe' in landmark_sets:
            gt_mp_full = run_mediapipe(evaluators.mediapipe_detector, cropped)
            gt_mp = gt_mp_full[evaluators.mediapipe_gt_indices, :2] if gt_mp_full is not None else None
            errors['mediapipe'].append(per_frame_euclidean_error(pred.get('mediapipe'), gt_mp))
            if 'mediapipe' in vis_writers:
                vis_writers['mediapipe'].write(_draw_overlay(cropped, gt_mp, pred.get('mediapipe')))

        if track_mesh:
            per_frame_vertices.append(pred.get('vertices'))

    if track_mesh:
        errors['temporal_smoothness'] = [
            per_frame_vertex_error(per_frame_vertices[i - 1], per_frame_vertices[i])
            for i in range(1, len(per_frame_vertices))
        ]

    return {name: np.array(values, dtype=np.float64) for name, values in errors.items()}


_COLOR_GT = (0, 255, 0)
_COLOR_PRED = (0, 0, 255)


def _draw_overlay(image, gt_xy, pred_xy):
    import cv2

    vis = image.copy()
    if gt_xy is not None and not np.isnan(gt_xy).any():
        for x, y in gt_xy:
            cv2.circle(vis, (int(x), int(y)), 1, _COLOR_GT, -1)
    if pred_xy is not None and not np.isnan(pred_xy).any():
        for x, y in pred_xy:
            cv2.circle(vis, (int(x), int(y)), 1, _COLOR_PRED, -1)
    return vis
