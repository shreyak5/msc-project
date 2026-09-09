from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform
from torch import nn

from dataset_processing.dataloading.detector_pool import get_detector
from dataset_processing.dataloading.face_parsing_cache import (
    FaceParsingResult,
    compute_face_parsing,
    get_face_parsing,
    visibility_ratio_from_mask_and_box,
)
from dataset_processing.dataloading.face_parsing_pool import get_xseg
from model import constants
from model.config import GatedTTConfig
from model.encoder import SViT
from model.farl_weights import load_farl_pretrained
from model.flame.flame import FLAME
from model.flame.renderer import Renderer, project_landmarks
from model.generator import UNetGenerator
from model.heads import ComponentHeads
from model.temporal import GatedTemporalTransformer, SimpleTemporalTransformer, TemporalTransformer
from preprocessing.cropping import crop_face, crop_face_with_landmarks

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FARL_CHECKPOINT_PATH = _REPO_ROOT / "pretrained_weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"

DETECTOR_THRESHOLD = 0.8
DETECTOR_MODEL_NAME = "mobilenet0.25"


def timestamped_out_dir(base_out_path: str) -> str:
    """{base_out_path}/{yyyymmdd_hhmm}/ - each run gets its own subdirectory so
    a later run never silently overwrites an earlier run's results."""
    out_dir = os.path.join(base_out_path, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def peek_num_expression_params(checkpoint_path: str | None) -> int:
    if checkpoint_path is None:
        return constants.FLAME_EXPRESSION_DIM
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("num_expression_params", constants.FLAME_EXPRESSION_DIM)


def build_models(
    device: str,
    use_unet: bool = False,
    tt_variant: str = "original",
    tt_gamma: float = constants.TT_GATE_GAMMA,
    num_expression_params: int = constants.FLAME_EXPRESSION_DIM,
) -> dict[str, nn.Module]:
    svit = SViT().to(device)
    load_farl_pretrained(svit, str(_FARL_CHECKPOINT_PATH))
    heads = ComponentHeads(expression_dim=num_expression_params).to(device)
    flame = FLAME(n_exp=num_expression_params).to(device)
    renderer = Renderer(flame.faces_tensor).to(device)
    if tt_variant == "simple":
        tt = SimpleTemporalTransformer().to(device)
    elif tt_variant == "gated":
        tt = GatedTemporalTransformer(GatedTTConfig(gamma=tt_gamma)).to(device)
    else:
        tt = TemporalTransformer().to(device)

    models = {"svit": svit, "heads": heads, "flame": flame, "renderer": renderer, "tt": tt}
    if use_unet:
        models["unet"] = UNetGenerator().to(device)
    for module in models.values():
        module.eval()
    return models


def load_available_checkpoint(
    models: dict[str, nn.Module], checkpoint_path: str | None, device: str,
    mismatched_out: set[str] | None = None,
) -> int | None:
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
        result = module.load_state_dict(checkpoint[name], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"[checkpoint] {name}: architecture mismatch against {checkpoint_path} - "
                f"missing {result.missing_keys}, unexpected {result.unexpected_keys} "
                f"(unmatched params stay at current init)"
            )
            if mismatched_out is not None:
                mismatched_out.add(name)
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
    dataset: str = "inference",
) -> tuple[torch.Tensor, float, bool]:
    face_mask_np, visibility_ratio, valid = get_face_parsing(
        cache_root, dataset, sample_id, frame_index,
        lambda: image_bgr,
        lambda: get_detector(detector_device, DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME),
        lambda: get_xseg(xseg_device),
        crop_scale, image_size,
    )
    face_mask = torch.from_numpy(face_mask_np).unsqueeze(0).to(device)
    return face_mask, visibility_ratio, valid


def _resolve_face_parsing(result: FaceParsingResult, image_size: int) -> tuple[np.ndarray, float, bool]:
    """FaceParsingResult -> (face_mask, visibility_ratio, valid), same fallback
    convention face_parsing_cache.get_face_parsing uses for its cached callers -
    a zero mask + 0.0 ratio on noface/unreadable, so downstream code (which
    always needs a same-shape mask tensor) never has to special-case None."""
    if result.status != "ok":
        return np.zeros((image_size, image_size), dtype=np.float32), 0.0, False
    return result.face_mask, result.visibility_ratio, True


def _crop_and_parse_np(
    image_bgr: np.ndarray,
    detector,
    xseg_device: str,
    crop_scale: float,
    image_size: int,
) -> tuple[np.ndarray | None, np.ndarray, float, bool, SimilarityTransform | None]:
    cropped_bgr, tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
        image_bgr, detector, scale=crop_scale, image_size=image_size,
    )

    cropped_rgb, precomputed_crop = None, None
    if cropped_bgr is not None:
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        precomputed_crop = (cropped_bgr, landmarks_5pt_crop, box_crop)

    result = compute_face_parsing(
        lambda: image_bgr,
        lambda: detector,
        lambda: get_xseg(xseg_device),
        crop_scale, image_size,
        precomputed_crop=precomputed_crop,
    )
    face_mask, visibility_ratio, valid = _resolve_face_parsing(result, image_size)
    return cropped_rgb, face_mask, visibility_ratio, valid, tform


def crop_tensor_and_compute_xseg_mask(
    image_bgr: np.ndarray,
    detector,
    xseg_device: str,
    crop_scale: float,
    image_size: int,
    device: str,
) -> tuple[torch.Tensor | None, np.ndarray | None, torch.Tensor, float, bool]:
    """Non-pooled replacement for separately calling crop_and_tensor +
    compute_visibility_and_mask (which each ran their own independent
    detector pass on the same frame) - thin device-placement wrapper around
    _crop_and_parse_np. Returns (pixel_values, cropped_rgb, face_mask,
    visibility_ratio, valid); pixel_values/cropped_rgb are None if no face
    was detected, matching crop_and_tensor's own no-face contract."""
    cropped_rgb, face_mask_np, visibility_ratio, valid, _tform = _crop_and_parse_np(
        image_bgr, detector, xseg_device, crop_scale, image_size,
    )
    pixel_values = None
    if cropped_rgb is not None:
        pixel_values = torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    face_mask = torch.from_numpy(face_mask_np).unsqueeze(0).to(device)
    return pixel_values, cropped_rgb, face_mask, visibility_ratio, valid


class FrameJob(NamedTuple):
    """One frame's worth of work for the CPU-bound crop process pool (see
    run_parallel_crop_and_parse). video_key identifies which video/image a
    result belongs to once results come back, since a group of many videos'
    frames are flattened into one job list for even load-balancing across
    workers (video lengths vary, so one-worker-per-video would leave short
    videos idling while others are still decoding)."""

    video_key: str
    frame_index: int
    frame_bgr: np.ndarray
    crop_scale: float
    image_size: int


class _PendingXseg(NamedTuple):
    """A frame whose crop/landmarks/box are already known but whose XSeg mask
    hasn't been computed yet - carries everything run_parallel_crop_and_parse's
    batched XSeg.parse_batch call needs, once every pool worker has returned."""

    cropped: np.ndarray
    landmarks_5pt_crop: np.ndarray
    box_crop: np.ndarray


class FrameResult(NamedTuple):
    video_key: str
    frame_index: int
    cropped_rgb: np.ndarray | None
    face_mask: np.ndarray | None
    visibility_ratio: float
    valid: bool
    tform: SimilarityTransform | None = None
    # Set only when a face was detected and XSeg hasn't run yet.
    # run_parallel_crop_and_parse always resolves every pending result (via a
    # single batched GPU call) before returning, so callers outside this
    # module never see a non-None pending field.
    pending: _PendingXseg | None = None


def _crop_and_defer_np(
    image_bgr: np.ndarray,
    detector,
    crop_scale: float,
    image_size: int,
) -> tuple[np.ndarray | None, tuple[np.ndarray, float, bool] | _PendingXseg, SimilarityTransform | None]:
    """Pool-worker counterpart to _crop_and_parse_np: detects+crops only, never
    calls XSeg itself - a face was found -> returns a _PendingXseg for
    run_parallel_crop_and_parse to batch through XSeg.parse_batch on GPU
    afterward; no face -> resolved immediately (mask_fallback, 0.0, False),
    matching _crop_and_parse_np's/compute_face_parsing's own no-face contract."""
    cropped_bgr, tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
        image_bgr, detector, scale=crop_scale, image_size=image_size,
    )
    if cropped_bgr is None:
        mask_fallback = np.zeros((image_size, image_size), dtype=np.float32)
        return None, (mask_fallback, 0.0, False), None

    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return cropped_rgb, _PendingXseg(cropped_bgr, landmarks_5pt_crop, box_crop), tform


def _pool_worker_crop_and_parse(job: FrameJob) -> FrameResult:
    detector = get_detector("cpu", DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)
    cropped_rgb, outcome, tform = _crop_and_defer_np(job.frame_bgr, detector, job.crop_scale, job.image_size)
    if isinstance(outcome, _PendingXseg):
        return FrameResult(job.video_key, job.frame_index, cropped_rgb, None, 0.0, False, tform, pending=outcome)
    face_mask, visibility_ratio, valid = outcome
    return FrameResult(job.video_key, job.frame_index, cropped_rgb, face_mask, visibility_ratio, valid, tform)


_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _init_pool_worker() -> None:
    """ProcessPoolExecutor initializer - runs once per worker process, before
    it takes any jobs. Belt-and-suspenders on top of the env vars
    make_crop_parse_pool sets in the parent (those cover numpy/torch/cv2's
    *BLAS-backend* threading, read at library-load time inside each freshly
    spawned interpreter - this covers OpenCV's own separate thread pool,
    which isn't OMP_NUM_THREADS-governed, and pins torch explicitly in case
    anything already imported it before reading the env)."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_crop_parse_pool(num_workers: int) -> ProcessPoolExecutor:
    for var in _THREAD_LIMIT_ENV_VARS:
        os.environ[var] = "1"
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
    )


def run_parallel_crop_and_parse(
    jobs: list[FrameJob], executor: ProcessPoolExecutor, xseg_device: str,
) -> list[FrameResult]:
    results = list(executor.map(_pool_worker_crop_and_parse, jobs))

    pending_indices = [i for i, result in enumerate(results) if result.pending is not None]
    if not pending_indices:
        return results

    xseg = get_xseg(xseg_device)
    images = [results[i].pending.cropped for i in pending_indices]
    landmarks_list = [results[i].pending.landmarks_5pt_crop for i in pending_indices]
    masks = xseg.parse_batch(images, landmarks_list)

    final_results = list(results)
    for i, mask in zip(pending_indices, masks):
        pending = final_results[i].pending
        mask = mask.astype(np.float32)
        visibility_ratio = visibility_ratio_from_mask_and_box(mask, pending.box_crop)
        final_results[i] = final_results[i]._replace(
            face_mask=mask, visibility_ratio=visibility_ratio, valid=True, pending=None,
        )
    return final_results


def run_flame(flame: FLAME, encoded: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    flame_out = flame(encoded["shape"], encoded["expression"], encoded["jaw"], encoded["eyelid"], encoded["rotation"])
    camera = torch.cat([encoded["scale"], encoded["translation"]], dim=-1)
    return flame_out, camera


def decode_and_project(flame: FLAME, encoded: dict[str, torch.Tensor], crop_size: int) -> list[dict[str, np.ndarray]]:
    """Builds on run_flame: also projects landmarks to crop-pixel space (mirrors
    SmirkMethod's own projection convention) -> list of B per-sample {'fan': (68,2),
    'mediapipe': (105,2), 'vertices': (V,3)} numpy dicts. Only evaluation needs this -
    none of the four inference/*.py scripts currently project landmarks to 2D pixel
    space (they either rasterize the mesh via Renderer, or save raw 3D FLAME output)."""
    flame_out, camera = run_flame(flame, encoded)
    fan_px = ((project_landmarks(flame_out["landmarks_fan"], camera) + 1) * (crop_size / 2)).cpu().numpy()
    mp_px = ((project_landmarks(flame_out["landmarks_mp"], camera) + 1) * (crop_size / 2)).cpu().numpy()
    vertices = flame_out["vertices"].cpu().numpy()
    return [{"fan": fan_px[i], "mediapipe": mp_px[i], "vertices": vertices[i]} for i in range(vertices.shape[0])]


def render_2d_reconstruction(
    flame: FLAME, renderer: Renderer, unet: UNetGenerator, face_probabilities: torch.Tensor,
    encoded: dict[str, torch.Tensor], pixel_values: torch.Tensor, face_mask: torch.Tensor,
    valid_recon: torch.Tensor,
) -> torch.Tensor:
    """Thin call into training/stage2.py's _render_and_reconstruct - reused
    directly rather than duplicated, since it's already the exact mask -> sample
    1% pixels -> UNet path this needs (Sec 7 Pass A/C).

    valid_recon: (B,) bool - rows without a validly detected/parsed face are
    skipped by _render_and_reconstruct's masking step (see its own docstring);
    callers here always pass an already-filtered/known-valid batch."""
    from training.stage2 import _render_and_reconstruct

    reconstructed, _projected_fan, _projected_mp = _render_and_reconstruct(
        flame, renderer, unet, face_probabilities, encoded, pixel_values, face_mask, valid_recon,
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
