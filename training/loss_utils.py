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
