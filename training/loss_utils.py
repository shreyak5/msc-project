"""Shared helper for the flag_*_valid-gated loss pattern (landmark_cache.py's
flag_landmarks_fan_valid/flag_landmarks_mp_valid, mica_cache.py's
flag_mica_valid) used throughout the Stage 1/Stage 2 training loops.

Matches SMIRK's own actual gating approach (smirk_trainer.py: predicted/target
landmarks are index-filtered by the valid-samples boolean mask BEFORE being
passed to F.mse_loss, not "compute the loss over the whole batch, then zero the
result if invalid") - a batch can have a MIX of valid and invalid samples, and
computing the loss over the whole batch regardless would let invalid/garbage
rows contaminate the loss value for the samples that were actually valid.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import torch

from dataset_processing.dataloading.combined_loader import CombinedFaceLoader
from model import constants
from model.losses.regularization import l2_regularization, log_scale_regularization


_LOSS_TERM_WEIGHTS: dict[str, float] = {
    "photometric": constants.PHOTOMETRIC_LOSS_WEIGHT,
    "vgg": constants.VGG_LOSS_WEIGHT,
    "emotion": constants.EMOTION_LOSS_WEIGHT,
    "landmark": constants.LANDMARK_LOSS_WEIGHT,
    "closure": constants.CLOSURE_LOSS_WEIGHT,
    "mica": constants.MICA_SHAPE_LOSS_WEIGHT,
    "mesh": constants.MESH_LOSS_LAMBDA,
    "lvc": constants.VERTEX_CONSISTENCY_LOSS_LAMBDA,
    "expr_cycle": constants.CYCLE_LOSS_WEIGHT,
    "id_cycle": constants.IDENTITY_CYCLE_LOSS_WEIGHT,
    "vel_expr": constants.TEMPORAL_VELOCITY_WEIGHT,
    "vel_jaw": constants.TEMPORAL_VELOCITY_WEIGHT,
    "vel_camera": constants.TEMPORAL_VELOCITY_WEIGHT,
    "vel_shape": constants.TEMPORAL_VELOCITY_WEIGHT,
}


def weighted_metrics(metrics: dict, extra_weights: dict[str, float] | None = None) -> dict:
    """Console-only companion to a raw metrics dict (compute_2d_losses/
    compute_3d_losses/run_pass_a/run_pass_b/run_pass_c's return values):
    multiplies each entry that has a known weight (the same value each term
    is scaled by before being summed into that function's returned total) by
    that weight, so a step's print line can show both the raw per-term loss
    and its actual contribution to the total. Recurses into nested dicts
    (e.g. run_pass_a's {"2d": {...}, "3d": {...}}).

    Deliberately not sent to wandb (unlike the raw metrics dict) - wandb
    already has the weight constants as run config, so a weighted curve
    there would just be a scalar multiple of the raw one, and the raw curve
    is what stays comparable across runs if a weight ever changes.

    extra_weights: for terms whose weight is a per-call config value rather
    than a fixed constants.py value - currently just "vertex_smooth"
    (Stage2Config.temporal_vertex_smoothness_weight), which
    compute_temporal_smoothness_losses scales by a caller-supplied argument,
    not a module-level constant, so it can't live in _LOSS_TERM_WEIGHTS.
    training/stage2.py's train() passes {"vertex_smooth":
    cfg.temporal_vertex_smoothness_weight} here for that reason. Overrides
    _LOSS_TERM_WEIGHTS on key collision.

    Terms with no weight found in either mapping (reg_2d/reg_3d/reg/
    num_pairs) are silently dropped: regularization_loss already bakes its
    own internal per-field weights into the value it returns (no further
    outer multiplication happens before summing), and num_pairs isn't a
    loss.
    """
    weights = {**_LOSS_TERM_WEIGHTS, **(extra_weights or {})}
    result: dict = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            nested = weighted_metrics(value, extra_weights)
            if nested:
                result[key] = nested
        elif key in weights:
            result[key] = weights[key] * value
    return result


def gated_loss(loss_fn: Callable[..., torch.Tensor], valid_mask: torch.Tensor, *tensors: torch.Tensor) -> torch.Tensor:
    """loss_fn: any of this project's (predicted, target) -> scalar loss
    functions (fan_boundary_loss, mediapipe_landmark_loss, eye_closure_loss,
    lip_closure_loss, mica_shape_loss all fit this shape). valid_mask: (B,)
    bool. *tensors: each (B, ...) - filtered by valid_mask (matching rows only)
    before being passed to loss_fn. Returns a zero scalar (not NaN) if nothing
    in the batch is valid, so callers can always add this into a running total
    without a separate presence check."""
    if valid_mask.sum() == 0:
        return torch.zeros((), device=tensors[0].device)
    filtered = [t[valid_mask] for t in tensors]
    return loss_fn(*filtered)


def regularization_loss(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    """Shape/expression/jaw (Sec 6: "L2 on expression parameters (and standard
    FLAME param regularizers - shape, jaw)"), plus camera scale if present in
    `encoded` (see below) - applied in every pass (Sec 6: "Regularization |
    ... | all passes").

    Rotation is still deliberately unregularized: zero isn't a sensible prior
    for it (would bias against genuinely non-frontal poses, which matter here
    given sign language video's real head-orientation variation), unlike
    shape/expression's zero-centered PCA coefficients where zero legitimately
    means "neutral". Scale WAS in that same "no sensible zero prior" boat
    (still true - see log_scale_regularization's own docstring for why it
    regularizes toward a reference value instead of 0) but empirically needs
    one anyway: measured drifting well below its Stage-1-converged value
    (~7.4) across multiple Stage 2 runs, with nothing else anchoring it.

    `"scale" in encoded` gate: one caller (Stage 2 Pass B's cycle consistency,
    training/stage2.py's run_pass_b) intentionally passes a filtered
    {"shape", "expression", "jaw"}-only dict here (not the full re_encoded
    output) - this degrades gracefully there (no scale term, matching the
    prior behavior) rather than KeyError."""
    total = (
        constants.REG_SHAPE_WEIGHT * l2_regularization(encoded["shape"])
        + constants.REG_EXPRESSION_WEIGHT * l2_regularization(encoded["expression"])
        + constants.REG_JAW_WEIGHT * l2_regularization(encoded["jaw"])
    )
    if "scale" in encoded:
        total = total + constants.REG_CAMERA_SCALE_WEIGHT * log_scale_regularization(
            encoded["scale"], constants.CAMERA_SCALE_REFERENCE
        )
    return total


def concat_category_fields(batch: dict, categories: list[str], keys: list[str], device: str) -> dict[str, torch.Tensor]:
    """batch: one yielded step from CombinedFaceLoader (dict keyed by category
    name). categories: which of that step's categories to combine (e.g.
    ["2d_image", "2d_video"]) - safe to concatenate along the batch dimension
    even though their configured batch_sizes differ (dataset_processing/config/
    dataloader.yaml), and safe for 3D categories specifically because
    IdentityAwareBatchSampler's identity pairs are already guaranteed within
    each category's own batch before this concatenation ever happens - combining
    afterward doesn't lose that guarantee, it just makes one bigger batch out of
    two already-valid ones. Returns a flat dict (not nested by category) - each
    key maps to one tensor spanning all combined samples, with the first
    category's rows first, second category's rows after (torch.cat preserves
    list order)."""
    return {key: torch.cat([batch[category][key] for category in categories], dim=0).to(device) for key in keys}


def next_batch(
    loader: CombinedFaceLoader, iterator: Iterator[dict[str, Any]], epoch: int,
) -> tuple[dict[str, Any], Iterator[dict[str, Any]], int]:
    """Returns (batch, iterator, epoch) - the caller holds iterator/epoch as
    loop state and passes them back in on the next call. Restarts the loader
    (new epoch -> set_epoch -> fresh iterator, so DistributedSampler produces a
    different shuffle) instead of raising StopIteration when the current
    iterator runs dry - needed once training is measured in steps rather than
    epochs (training/pretrain.py, training/stage2.py), since there's no outer
    `for epoch in range(...)` loop left to trigger a restart automatically."""
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        loader.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch
