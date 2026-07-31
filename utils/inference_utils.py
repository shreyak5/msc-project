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
from dataset_processing.dataloading.face_parsing_cache import get_face_parsing
from dataset_processing.dataloading.face_parsing_pool import get_xseg
from model.encoder import SViT
from model.farl_weights import load_farl_pretrained
from model.flame.flame import FLAME
from model.flame.renderer import Renderer, project_landmarks
from model.generator import UNetGenerator
from model.heads import ComponentHeads
from model.temporal import TemporalTransformer
from preprocessing.cropping import crop_face, crop_face_with_landmarks

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


def _crop_and_parse_np(
    image_bgr: np.ndarray,
    detector,
    cache_root: str,
    sample_id: str,
    frame_index: int | None,
    xseg_device: str,
    crop_scale: float,
    image_size: int,
) -> tuple[np.ndarray | None, np.ndarray, float, bool, SimilarityTransform | None]:
    """Numpy-only core: one detection feeding both the model's crop and
    XSeg's mask, via crop_face_with_landmarks + get_face_parsing's
    precomputed_crop param (instead of letting get_face_parsing re-detect
    internally, which is the redundant-detector-call this was written to
    avoid). Shared by crop_tensor_and_compute_xseg_mask (non-pooled callers,
    which add device placement on top) and the process-pool worker below
    (which can't return CUDA tensors across a process boundary anyway, so it
    needs this numpy form regardless). Returns (cropped_rgb | None, face_mask,
    visibility_ratio, valid, tform); cropped_rgb/tform are None only when no
    face was detected, in which case get_face_parsing is still called (with
    precomputed_crop=None) so it runs and caches its own detection attempt -
    a second detector call, but only on that rare no-face frame. tform maps
    original-frame coordinates to crop-space coordinates (same convention as
    baselines/smirk_experiments/demo_utils.py's own crop transform) - kept
    around so a caller that wants to warp a crop-space render back into the
    original frame's position (demo_videos.py's --render_orig) can do so
    without re-detecting."""
    cropped_bgr, tform, landmarks_5pt_crop, box_crop = crop_face_with_landmarks(
        image_bgr, detector, scale=crop_scale, image_size=image_size,
    )

    cropped_rgb, precomputed_crop = None, None
    if cropped_bgr is not None:
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        precomputed_crop = (cropped_bgr, landmarks_5pt_crop, box_crop)

    face_mask, visibility_ratio, valid = get_face_parsing(
        cache_root, "inference", sample_id, frame_index,
        lambda: image_bgr,
        lambda: detector,
        lambda: get_xseg(xseg_device),
        crop_scale, image_size,
        precomputed_crop=precomputed_crop,
    )
    return cropped_rgb, face_mask, visibility_ratio, valid, tform


def crop_tensor_and_compute_xseg_mask(
    image_bgr: np.ndarray,
    detector,
    cache_root: str,
    sample_id: str,
    frame_index: int | None,
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
        image_bgr, detector, cache_root, sample_id, frame_index, xseg_device, crop_scale, image_size,
    )
    pixel_values = None
    if cropped_rgb is not None:
        pixel_values = torch.from_numpy(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    face_mask = torch.from_numpy(face_mask_np).unsqueeze(0).to(device)
    return pixel_values, cropped_rgb, face_mask, visibility_ratio, valid


class FrameJob(NamedTuple):
    """One frame's worth of work for the CPU-bound crop+XSeg process pool
    (see run_parallel_crop_and_parse). video_key identifies which video/image
    a result belongs to once results come back, since a group of many videos'
    frames are flattened into one job list for even load-balancing across
    workers (video lengths vary, so one-worker-per-video would leave short
    videos idling while others are still decoding)."""

    video_key: str
    frame_index: int
    frame_bgr: np.ndarray
    sample_id: str
    cache_root: str
    crop_scale: float
    image_size: int


class FrameResult(NamedTuple):
    video_key: str
    frame_index: int
    cropped_rgb: np.ndarray | None
    face_mask: np.ndarray
    visibility_ratio: float
    valid: bool
    tform: SimilarityTransform | None = None


def _pool_worker_crop_and_parse(job: FrameJob) -> FrameResult:
    """ProcessPoolExecutor worker: runs one frame's detect+crop+XSeg entirely
    on CPU via _crop_and_parse_np. Always CPU, regardless of what device the
    main process's own models are on - onnxruntime has no CUDA execution
    provider in this environment for XSeg regardless, and a GPU RetinaFace
    wouldn't actually parallelize across many pool *processes* sharing one
    physical GPU (separate CUDA contexts contending for one device, no MPS
    set up here), so there's nothing to gain by not also keeping the detector
    CPU-side here - RetinaFace's mobilenet0.25 backbone is a lightweight,
    real-time-on-CPU detector by design. Builds its own detector/XSeg via the
    same lazy-per-process-singleton pools (detector_pool.get_detector,
    face_parsing_pool.get_xseg) scripts/prewarm_face_parsing_cache.py's own
    --num_shards workers already rely on - each pool worker process pays
    that construction cost exactly once, on its first job."""
    detector = get_detector("cpu", DETECTOR_THRESHOLD, DETECTOR_MODEL_NAME)
    cropped_rgb, face_mask, visibility_ratio, valid, tform = _crop_and_parse_np(
        job.frame_bgr, detector, job.cache_root, job.sample_id, job.frame_index,
        "cpu", job.crop_scale, job.image_size,
    )
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
    """Creates the shared pool inference_videos.py/demo_videos.py create once
    in main() and reuse across every video/group, rather than paying process
    startup cost repeatedly. Explicitly uses the 'spawn' start method, not
    this platform's 'fork' default: by the time this is called, the calling
    script has already loaded torch models onto the GPU (build_models), and
    CUDA does not support being inherited across a fork - a forked worker
    that touches anything CUDA-adjacent (even indirectly, via importing torch
    modules that lazily touch it) can hang or crash. spawn re-imports each
    worker fresh instead, which is the only combination that's safe here.

    Also pins every worker to single-threaded BLAS/OpenMP, same reasoning as
    uniface/onnx_utils.py's own intra_op_num_threads=1 (this project's
    parallelism model throughout is "many single-threaded workers", never
    "N processes each also fanning out internally") - without this, each of
    N spawned processes independently imports numpy/torch/cv2, each of which
    sizes its own thread pool off the *node's full core count* by default;
    multiplied by num_workers processes this exhausts a shared HPC node's
    max-user-processes ulimit almost immediately (observed here: `libgomp:
    Thread creation failed` / BrokenProcessPool with a 1900 ulimit -u and 32
    workers). Setting os.environ here, in the parent, before spawning is
    required for it to take effect - each child is a fresh interpreter that
    reads these at its own numpy/torch import time, inheriting whatever the
    parent's environ held at spawn time; setting them any later (e.g. inside
    the worker itself, after those libraries already imported) would be too
    late for the BLAS backends that read the env var only once at load."""
    for var in _THREAD_LIMIT_ENV_VARS:
        os.environ[var] = "1"
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_pool_worker,
    )


def run_parallel_crop_and_parse(jobs: list[FrameJob], executor: ProcessPoolExecutor) -> list[FrameResult]:
    """Submits already-decoded frames to `executor` for the CPU-bound
    detect+crop+XSeg work and returns FrameResults in the same order as
    `jobs` (executor.map preserves submission order). Frames are decoded
    sequentially up front by the caller (cv2.VideoCapture, as before this
    change) rather than having each worker seek into the video itself -
    VideoFileFrameSource.read_frame's seek falls back to a full
    from-scratch redecode whenever it lands off-target (codec/keyframe
    dependent), which scattered per-frame random access across many workers
    could hit constantly."""
    return list(executor.map(_pool_worker_crop_and_parse, jobs))


def run_flame(flame: FLAME, encoded: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """encoded: dict of decoded FLAME/camera params, each (B, ...) - the direct output
    of model.encoding.encode_image/encode_video (post-squeeze for encode_video's clip
    dim, post-chunking for demo_videos.py's render loop). Runs FLAME once (the same
    5-arg call every consumer needs) and assembles the (B,3)=[scale,tx,ty] projection
    camera Renderer.forward/project_landmarks both expect - callers that don't project
    (inference_images.py/inference_videos.py's --save_vertices path) just ignore the
    second return value; the torch.cat to build it is negligible either way.
    Returns (flame_out, camera)."""
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
