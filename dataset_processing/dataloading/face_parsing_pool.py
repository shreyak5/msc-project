from __future__ import annotations

_xseg = None


def get_xseg(device: str, align_size: int = 256, blur_sigma: float = 0.0):
    """Lazy per-worker-process XSeg singleton, mirroring mica_pool.py's
    get_mica. XSeg runs via a PyTorch conversion of its ONNX weights
    (uniface.torch_utils) rather than onnxruntime, since onnxruntime-gpu has
    no PyPI wheel for this project's CPU-only onnxruntime install on aarch64.
    `device` is passed straight through as a torch device string."""
    global _xseg
    if _xseg is None:
        from uniface.parsing import XSeg
        _xseg = XSeg(align_size=align_size, blur_sigma=blur_sigma, device=device)
    return _xseg
