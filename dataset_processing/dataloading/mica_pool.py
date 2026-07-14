from __future__ import annotations

_mica = None


def get_mica(device: str):
    """Lazy per-worker-process MICA singleton, mirroring detector_pool.py's
    get_detector - DataLoader workers are separate processes, so each gets its
    own module-level global rather than sharing one CUDA-resident model."""
    global _mica
    if _mica is None:
        from model.mica.mica import MICA
        _mica = MICA().to(device).eval()
    return _mica
