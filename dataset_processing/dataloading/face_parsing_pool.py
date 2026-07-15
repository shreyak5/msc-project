from __future__ import annotations

_xseg = None


def get_xseg(device: str, align_size: int = 256, blur_sigma: float = 0.0):
    """Lazy per-worker-process XSeg singleton, mirroring mica_pool.py's
    get_mica. XSeg runs on onnxruntime, not torch, so `device` is translated
    to an onnxruntime execution-provider list rather than passed through as a
    torch device string - onnxruntime silently skips any provider that isn't
    actually available (e.g. this project's CPU-only onnxruntime install on
    aarch64, where onnxruntime-gpu has no PyPI wheel), so this is a "prefer
    CUDA, fall back to CPU" request rather than a hard requirement."""
    global _xseg
    if _xseg is None:
        from uniface.parsing import XSeg
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        _xseg = XSeg(align_size=align_size, blur_sigma=blur_sigma, providers=providers)
    return _xseg
