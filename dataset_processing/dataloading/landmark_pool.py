from __future__ import annotations

_fan_predictor = None
_mediapipe_detector = None


def get_fan_predictor(device: str, model_name: str):
    """Lazy per-worker-process FAN predictor singleton, mirroring
    detector_pool.py's get_detector."""
    global _fan_predictor
    if _fan_predictor is None:
        from utils.landmark_utils import build_fan_predictor
        _fan_predictor = build_fan_predictor(device=device, model_name=model_name)
    return _fan_predictor


def get_mediapipe_detector(model_asset_path: str):
    """Lazy per-worker-process MediaPipe FaceLandmarker singleton, mirroring
    detector_pool.py's get_detector."""
    global _mediapipe_detector
    if _mediapipe_detector is None:
        from utils.landmark_utils import build_mediapipe_detector
        _mediapipe_detector = build_mediapipe_detector(model_asset_path)
    return _mediapipe_detector
