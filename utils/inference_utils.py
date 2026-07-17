"""Shared model-construction, checkpoint-loading, detection/cropping, and
rendering utilities for the demo/inference entry points under inference/
(demo_images.py, demo_videos.py, inference_images.py, inference_videos.py).

Mirrors training/stage2.py's own model-construction pattern (SViT -> FaRL init
-> heads/flame/renderer/tt/[unet], every module built with config defaults,
since those already match training) and reuses training/stage2.py's
_render_and_reconstruct directly for the optional UNet photoreal path, rather
than duplicating it.

Face detection/cropping and visibility-scoring reuse this project's own
dataset-pipeline building blocks (dataset_processing/dataloading/detector_pool.
py, face_parsing_pool.py, face_parsing_cache.py, preprocessing/cropping.py) -
these already have a fully live (cache-optional) code path with no dependency
on any prewarm/cache-warming script, which is exactly what a fresh demo input
needs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from dataset_processing.dataloading.detector_pool import get_detector
from dataset_processing.dataloading.face_parsing_cache import get_face_parsing
from dataset_processing.dataloading.face_parsing_pool import get_xseg
from model.encoder import SViT
from model.farl_weights import load_farl_pretrained
from model.flame.flame import FLAME
from model.flame.renderer import Renderer
from model.generator import UNetGenerator
from model.heads import ComponentHeads
from model.temporal import TemporalTransformer
from preprocessing.cropping import crop_face

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FARL_CHECKPOINT_PATH = _REPO_ROOT / "pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"

DETECTOR_THRESHOLD = 0.8
DETECTOR_MODEL_NAME = "mobilenet0.25"
DEFAULT_OUT_ROOT = "inference/output"


def shared_cache_root() -> str:
    """Face-parsing cache shared by every script that needs live visibility/mask
    scoring (demo_videos.py, inference_videos.py, demo_images.py's --render_2d_recon
    path) - always under the top-level output root regardless of each script's own
    --out_path, so repeated runs across different scripts/out_paths still hit the
    same cache instead of each maintaining its own copy."""
    return os.path.join(DEFAULT_OUT_ROOT, ".face_parsing_cache")


def timestamped_out_dir(base_out_path: str) -> str:
    """{base_out_path}/{yyyymmdd_hhmm}/ - each run gets its own subdirectory so
    a later run never silently overwrites an earlier run's results."""
    out_dir = os.path.join(base_out_path, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_models(device: str, use_unet: bool = False) -> dict[str, nn.Module]:
    """Mirrors training/stage2.py's train() construction block: every module
    built with config defaults (already tuned to match training). `flame` and
    `renderer` are included in the returned dict for convenience (callers need
    both regardless of which encode_* path they use), even though they're not
    checkpointed modules themselves."""
    svit = SViT().to(device)
    load_farl_pretrained(svit, str(_FARL_CHECKPOINT_PATH))
    heads = ComponentHeads().to(device)
    flame = FLAME().to(device)
    renderer = Renderer(flame.faces_tensor).to(device)
    tt = TemporalTransformer().to(device)

    models = {"svit": svit, "heads": heads, "flame": flame, "renderer": renderer, "tt": tt}
    if use_unet:
        models["unet"] = UNetGenerator().to(device)
    for module in models.values():
        module.eval()
    return models


def load_available_checkpoint(models: dict[str, nn.Module], checkpoint_path: str | None, device: str) -> int | None:
    """None -> no-op (pure sanity-test mode: svit keeps its FaRL-pretrained
    backbone, heads/tt/unet stay at from-scratch init). Otherwise loads
    whichever of `models`' keys are actually present in the checkpoint file -
    training.checkpoint.load_checkpoint itself requires an exact match
    (checkpoint[name] KeyErrors on a miss), so this peeks at the file's keys
    first to gracefully support a Stage-1-only checkpoint (svit, heads), a full
    Stage-2 checkpoint (+ unet, tt), or anything in between, without the caller
    needing to know in advance which kind of checkpoint it is."""
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    available = {name: module for name, module in models.items() if name in checkpoint}
    missing = set(models) - {"flame", "renderer"} - set(available)
    unused = set(checkpoint) - {"step", "optimizer"} - set(models)
    if missing:
        print(f"[checkpoint] no weights for {sorted(missing)} in {checkpoint_path} - staying at current init")
    if unused:
        print(f"[checkpoint] ignoring unrequested keys in {checkpoint_path}: {sorted(unused)}")

    for name, module in available.items():
        module.load_state_dict(checkpoint[name])
    return checkpoint["step"]


def crop_and_tensor(
    image_bgr: np.ndarray, detector, scale: float, image_size: int, device: str,
) -> tuple[torch.Tensor | None, np.ndarray | None]:
    """Returns (pixel_values: Tensor(1,3,H,W) in [0,1], cropped_rgb: uint8
    (H,W,3)) or (None, None) if no face was detected. [0,1] scaling only,
    matching training's own normalization exactly - no mean/std anywhere in
    model/, FaRL's own ln_pre LayerNorm handles that internally."""
    cropped_bgr, _tform = crop_face(image_bgr, detector, scale=scale, image_size=image_size)
    if cropped_bgr is None:
        return None, None
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    pixel_values = torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return pixel_values.to(device), cropped_rgb


def compute_visibility_and_mask(
    image_bgr: np.ndarray,
    cache_root: str,
    sample_id: str,
    frame_index: int | None,
    detector_device: str,
    xseg_device: str,
    crop_scale: float,
    image_size: int,
    device: str,
) -> tuple[torch.Tensor, float, bool]:
    """Wraps dataset_processing.dataloading.face_parsing_cache.get_face_parsing
    - the exact live detect+XSeg-parse computation training's own cache-miss
    path already runs per frame. `cache_root` just needs to be a writable
    scratch directory (speeds up repeated runs over the same input; harmless
    if left empty/fresh). Returns (face_mask: Tensor(1,H,W) on `device` - no
    channel dim, matching training/stage2.py's _render_and_reconstruct's own
    `face_mask.unsqueeze(1)` convention - visibility_ratio, valid)."""
    face_mask_np, visibility_ratio, valid = get_face_parsing(
        cache_root, "inference", sample_id, frame_index,
        lambda: image_bgr,
        lambda: get_detector(detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME),
        lambda: get_xseg(xseg_device),
        crop_scale, image_size,
    )
    face_mask = torch.from_numpy(face_mask_np).unsqueeze(0).to(device)
    return face_mask, visibility_ratio, valid


def render_2d_reconstruction(
    flame: FLAME, renderer: Renderer, unet: UNetGenerator, face_probabilities: torch.Tensor,
    encoded: dict[str, torch.Tensor], pixel_values: torch.Tensor, face_mask: torch.Tensor,
) -> torch.Tensor:
    """Thin call into training/stage2.py's _render_and_reconstruct - reused
    directly rather than duplicated, since it's already the exact mask -> sample
    1% pixels -> UNet path this needs (Sec 7 Pass A/C)."""
    from training.stage2 import _render_and_reconstruct

    reconstructed, _projected_fan, _projected_mp = _render_and_reconstruct(
        flame, renderer, unet, face_probabilities, encoded, pixel_values, face_mask,
    )
    return reconstructed


def tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float in [0,1] -> (H,W,3) uint8 RGB, for panel assembly."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


def make_panel(*rgb_uint8_panels: np.ndarray) -> np.ndarray:
    """Horizontal concat of however many panels were actually requested (crop
    always first, then mesh/2d-recon if enabled), converted to BGR for cv2
    writing - generalizes SMIRK's fixed 2-or-3-panel torch.cat(..., dim=3)
    pattern to any number of panels."""
    bgr_panels = [cv2.cvtColor(panel, cv2.COLOR_RGB2BGR) for panel in rgb_uint8_panels]
    return np.concatenate(bgr_panels, axis=1)
