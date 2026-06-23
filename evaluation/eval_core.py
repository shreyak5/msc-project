from dataclasses import dataclass

import numpy as np

from landmark_utils import (
    build_retinaface_detector,
    build_mediapipe_detector,
    build_fan_predictor,
    crop_face,
    run_mediapipe,
    run_fan,
)
from metrics import per_frame_euclidean_error
from methods.smirk_method import SmirkMethod

METHOD_REGISTRY = {
    'smirk': SmirkMethod,
}


@dataclass
class Evaluators:
    method: object
    face_detector: object
    fan_predictor: object
    mediapipe_detector: object
    mediapipe_gt_indices: object


def build_evaluators(method_name, device, crop_size, mediapipe_model_path, enabled_sets):
    method = METHOD_REGISTRY[method_name]()
    method.setup(device, crop_size=crop_size)

    face_detector = build_retinaface_detector(device)
    fan_predictor = build_fan_predictor(device) if 'fan' in enabled_sets else None
    mediapipe_detector = build_mediapipe_detector(mediapipe_model_path) if 'mediapipe' in enabled_sets else None
    mediapipe_gt_indices = method.mediapipe_gt_indices() if 'mediapipe' in enabled_sets else None

    return Evaluators(method, face_detector, fan_predictor, mediapipe_detector, mediapipe_gt_indices)


def evaluate_clip(frames, crop_scale, crop_size, evaluators, enabled_sets, vis_writers=None):
    """Runs the crop -> GT detect -> method.predict -> per-frame error loop over one clip's frames.

    Returns {'fan': np.ndarray, 'mediapipe': np.ndarray} of per-frame errors (NaN where missing),
    restricted to the sets in enabled_sets.
    """
    vis_writers = vis_writers or {}
    errors = {name: [] for name in enabled_sets}

    for frame in frames:
        cropped, _tform = crop_face(frame, evaluators.face_detector, scale=crop_scale, image_size=crop_size)

        if cropped is None:
            for name in enabled_sets:
                errors[name].append(np.nan)
                if name in vis_writers:
                    vis_writers[name].write(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
            continue

        pred = evaluators.method.predict(cropped)

        if 'fan' in enabled_sets:
            gt_fan, _scores = run_fan(evaluators.face_detector, evaluators.fan_predictor, cropped)
            errors['fan'].append(per_frame_euclidean_error(pred.get('fan'), gt_fan))
            if 'fan' in vis_writers:
                vis_writers['fan'].write(_draw_overlay(cropped, gt_fan, pred.get('fan')))

        if 'mediapipe' in enabled_sets:
            gt_mp_full = run_mediapipe(evaluators.mediapipe_detector, cropped)
            gt_mp = gt_mp_full[evaluators.mediapipe_gt_indices, :2] if gt_mp_full is not None else None
            errors['mediapipe'].append(per_frame_euclidean_error(pred.get('mediapipe'), gt_mp))
            if 'mediapipe' in vis_writers:
                vis_writers['mediapipe'].write(_draw_overlay(cropped, gt_mp, pred.get('mediapipe')))

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
