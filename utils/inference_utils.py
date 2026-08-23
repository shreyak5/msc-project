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
these already have a fully live code path with no dependency on any
prewarm/cache-warming script, which is exactly what a fresh demo input needs.
No disk caching happens here (unlike training's dataset loading and
scripts/prewarm_face_parsing_cache.py, which do cache): every frame is
redetected/reparsed fresh on every run, matching how a demo/inference input is
actually used - normally seen once per run, not across repeated epochs."""

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
    """Reads just the num_expression_params field training/checkpoint.py's
    save_checkpoint always stores, without needing any model built yet -
    build_models needs this value up front to size ComponentHeads'/FLAME's
    expression dimension correctly, since load_available_checkpoint's per-module
    load_state_dict(strict=False) only tolerates missing/unexpected *keys*, not a
    same-named key with a mismatched shape (a checkpoint trained with a different
    expression_dim has a differently-shaped heads.expression.linear.{weight,bias}).
    None (no checkpoint - sanity-test mode) or a checkpoint that predates this
    field -> constants.FLAME_EXPRESSION_DIM, the same historical-default fallback
    training/checkpoint.py's own load_checkpoint uses."""
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
    """Mirrors training/stage2.py's train() construction block: every module
    built with config defaults (already tuned to match training). `flame` and
    `renderer` are included in the returned dict for convenience (callers need
    both regardless of which encode_* path they use), even though they're not
    checkpointed modules themselves.

    tt_variant/tt_gamma mirror training/config.py's Stage2Config fields of the
    same name (see training/stage2.py's own tt construction block) - a
    checkpoint's tt weights were saved by whichever TT class that run's
    tt_variant selected, and load_available_checkpoint's per-module
    load_state_dict(strict=False) only tolerates missing/unexpected *keys*, not
    a same-named key with a mismatched shape (e.g. GatedTTBlock/SimpleTTBlock's
    full-width q_proj/k_proj vs TemporalTransformer's TTBlock, which sizes them
    to the QK+ALiBi heads only) - so this must match whatever the checkpoint was
    actually trained with. "simple" and "gated" TT have identical parameter
    shapes to each other (only "original" differs), so which of those two a
    checkpoint needs can't be auto-detected from its state dict; tt_gamma only
    affects GatedTemporalTransformer's forward pass, not its parameter shapes,
    so it isn't recoverable from the checkpoint either.

    num_expression_params mirrors training/config.py's Stage2Config/PretrainConfig
    field of the same name - unlike tt_variant, this IS recoverable from a
    checkpoint (peek_num_expression_params above), since save_checkpoint always
    records it; callers should peek it from args.checkpoint before calling this
    rather than leaving it at the default, except in sanity-test (no-checkpoint)
    mode where the default is already correct."""
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
    """None -> no-op (pure sanity-test mode: svit keeps its FaRL-pretrained
    backbone, heads/tt/unet stay at from-scratch init). Otherwise loads
    whichever of `models`' keys are actually present in the checkpoint file -
    training.checkpoint.load_checkpoint itself requires an exact match
    (checkpoint[name] KeyErrors on a miss), so this peeks at the file's keys
    first to gracefully support a Stage-1-only checkpoint (svit, heads), a full
    Stage-2 checkpoint (+ unet, tt), or anything in between, without the caller
    needing to know in advance which kind of checkpoint it is.

    Per-module loading is strict=False, not strict=True: extends that same
    graceful handling one level down, to a module whose *architecture* has
    since changed (e.g. an older checkpoint's TT saved before the split-head
    attention rework) rather than only a module missing outright. Any
    missing/unexpected parameter keys are printed exactly like a missing
    module is above, rather than either crashing (strict=True) or silently
    leaving them unmentioned.

    mismatched_out: if given, mismatched module names are added to this set,
    so a caller can react (e.g. skip a module whose weights didn't load)."""
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
    """Wraps dataset_processing.dataloading.face_parsing_cache.get_face_parsing
    - the exact live detect+XSeg-parse computation training's own cache-miss
    path already runs per frame. `cache_root` just needs to be a writable
    scratch directory (speeds up repeated runs over the same input; harmless
    if left empty/fresh). `dataset` defaults to "inference" (this module's
    original demo/inference-script bucket) but a caller whose sample_id/frame
    indexing matches an existing indexed dataset (e.g. evaluation/eval_core.py
    evaluating a dataset's own test split) can pass that dataset's real name to
    land in - and reuse - the same cache buckets training's own data loading
    already populated, instead of a separate empty "inference" bucket. Returns
    (face_mask: Tensor(1,H,W) on `device` - no channel dim, matching
    training/stage2.py's _render_and_reconstruct's own `face_mask.unsqueeze(1)`
    convention - visibility_ratio, valid)."""
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
    """Numpy-only core: one detection feeding both the model's crop and XSeg's
    mask, via crop_face_with_landmarks + compute_face_parsing's precomputed_crop
    param (instead of letting compute_face_parsing re-detect internally, which
    is the redundant-detector-call this was written to avoid). No caching -
    every call redetects/reparses from scratch, matching a demo/inference
    script's actual usage pattern (each input is normally seen once per run).
    Shared by crop_tensor_and_compute_xseg_mask (non-pooled callers, which add
    device placement on top) and the process-pool worker below (which can't
    return CUDA tensors across a process boundary anyway, so it needs this
    numpy form regardless). Returns (cropped_rgb | None, face_mask,
    visibility_ratio, valid, tform); cropped_rgb/tform are None only when no
    face was detected, in which case compute_face_parsing is still called (with
    precomputed_crop=None) so it runs its own detection attempt - a second
    detector call, but only on that rare no-face frame. tform maps
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
    """ProcessPoolExecutor worker: runs one frame's detect+crop on CPU via
    _crop_and_defer_np, and either resolves its result immediately (no face
    detected) or hands back a _PendingXseg for the main process to finish in a
    single batched GPU XSeg call (see run_parallel_crop_and_parse) - never runs
    XSeg itself. Detection stays CPU-side regardless of what device the main
    process's own models are on: RetinaFace's mobilenet0.25 backbone is a
    lightweight, real-time-on-CPU detector by design, and a GPU RetinaFace
    wouldn't actually parallelize across many pool *processes* sharing one
    physical GPU (separate CUDA contexts contending for one device, no MPS set
    up here) - the same reason XSeg itself is deferred to a single
    main-process GPU call rather than run per-worker. Builds its own detector
    via the same lazy-per-process-singleton pool (detector_pool.get_detector)
    scripts/prewarm_face_parsing_cache.py's own --num_shards workers already
    rely on - each pool worker process pays that construction cost exactly
    once, on its first job."""
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


def run_parallel_crop_and_parse(
    jobs: list[FrameJob], executor: ProcessPoolExecutor, xseg_device: str,
) -> list[FrameResult]:
    """Submits already-decoded frames to `executor` for the CPU-bound
    detect+crop work (executor.map preserves submission order). Frames are
    decoded sequentially up front by the caller (cv2.VideoCapture, as before
    this change) rather than having each worker seek into the video itself -
    VideoFileFrameSource.read_frame's seek falls back to a full from-scratch
    redecode whenever it lands off-target (codec/keyframe dependent), which
    scattered per-frame random access across many workers could hit constantly.

    Second phase, after the pool returns: every FrameResult with a detected
    face comes back with a non-None `pending` (see _pool_worker_crop_and_parse)
    instead of a computed mask - every one of these is collected across the
    whole job list and resolved in a single batched XSeg.parse_batch GPU call
    (get_xseg(xseg_device) - already torch-backed, see uniface.torch_utils),
    rather than each of up to num_workers pool *processes* opening its own
    CUDA context to compute one frame at a time. No caching: every frame is
    redetected/reparsed fresh on every call, matching a demo/inference script's
    actual usage pattern (each input is normally seen once per run). The
    FrameResults this function returns are always fully resolved
    (pending=None), so callers never see a _PendingXseg."""
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
