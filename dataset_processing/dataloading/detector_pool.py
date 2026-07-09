from __future__ import annotations

_detector = None


def get_detector(device: str, threshold: float, model_name: str):
    global _detector
    if _detector is None:
        from preprocessing.cropping import build_retinaface_detector
        _detector = build_retinaface_detector(device=device, threshold=threshold, model_name=model_name)
    return _detector
