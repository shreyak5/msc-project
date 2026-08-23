"""Checkpoint save/load, shared by every training loop (Stage 1's training/
pretrain.py, Stage 2's training/stage2.py).

Generalized to a named-modules dict rather than fixed (svit, heads) positional
args: Stage 1 checkpoints only ever contain {"svit", "heads"}, but Stage 2
needs to (a) do a one-time load of svit+heads *only* from a Stage 1 checkpoint
at startup (no optimizer - Stage 2 builds its own fresh optimizer state, and
no unet/tt - Stage 1 never saved any), then (b) save/resume its own checkpoints
containing {"svit", "heads", "unet", "tt"} plus its own optimizer. The same two
functions serve both cases: callers just pass whichever modules dict is
relevant, and load_checkpoint only looks up the keys it's asked for.

save_checkpoint always saves the *underlying* model's state dict (training.
distributed.unwrap_model), never a DistributedDataParallel wrapper's own state
dict (DDP prefixes every key with "module.") - so a checkpoint is portable
between single-GPU and multi-GPU runs either direction, no key surgery needed.
load_checkpoint has no unwrap_model call: by design (training/pretrain.py's
own load-then-wrap ordering) it's only ever called on plain, not-yet-DDP-
wrapped models - loading before wrapping lets DDP's own constructor-time
broadcast-from-rank-0 step guarantee every rank starts from identical
(checkpoint-loaded) weights, rather than trusting every rank's checkpoint read
to independently agree."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from training.distributed import unwrap_model


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    modules: dict[str, nn.Module],
    optimizer: Optimizer | None = None,
    num_expression_params: int | None = None,
) -> Path:
    """No epoch here (unlike an earlier version of this function): both
    training loops are step-based, not epoch-based (see training/config.py's
    PretrainConfig docstring for why "epoch" isn't a well-defined progress
    unit once a loop draws from a loader indefinitely via training/loss_utils.
    py's next_batch, and Stage 2 draws from two independently-cycling loaders
    at different relative rates besides). Each loader still tracks its own
    internal epoch counter (needed for DistributedSampler.set_epoch's
    reshuffling), but that's loader-local bookkeeping, not something checkpoints
    need to persist for correctness - see next_batch's own docstring for the
    step//len(loader) approximation used to avoid replaying early shuffle
    orders after a resume.

    num_expression_params: the expression parameter count (training/config.py's
    PretrainConfig/Stage2Config field of the same name) this checkpoint's heads
    module was trained with, if known - recorded so a later load_checkpoint call
    can validate it matches the resuming run's own configured dim, rather than
    silently loading mismatched expression-head weights."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"step_{step:08d}.pt"
    checkpoint = {"step": step}
    if num_expression_params is not None:
        checkpoint["num_expression_params"] = num_expression_params
    checkpoint.update({name: unwrap_model(module).state_dict() for name, module in modules.items()})
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    modules: dict[str, nn.Module],
    optimizer: Optimizer | None = None,
    device: str = "cpu",
    expected_num_expression_params: int | None = None,
) -> int:
    """Loads weights (and optimizer state, if given) in place, returns step to
    resume from. Only loads the modules/optimizer actually passed in - e.g.
    Stage 2 seeding from a Stage 1 checkpoint passes just {"svit": svit,
    "heads": heads} and no optimizer, ignoring that checkpoint's lack of
    unet/tt/optimizer keys entirely, since those were never requested.

    Per-module loading is strict=False, not strict=True (mirroring utils/
    inference_utils.py's load_available_checkpoint): a module whose
    *architecture* has since changed (e.g. an older checkpoint's TT saved
    before the split-head attention rework) has its still-matching keys
    loaded and its unmatched ones left at current init and printed, rather
    than crashing every rank.

    expected_num_expression_params: if given, checked up front against this
    checkpoint's own recorded num_expression_params (save_checkpoint's field of
    the same name; checkpoints saved before that field existed are assumed 100,
    the historical default) and raises a clear ValueError on mismatch - a
    different expression parameter count means the "heads" module's expression
    head is a different shape, which strict=False's per-key matching can't
    safely paper over (unlike a genuinely missing/renamed key, a same-named key
    with a different shape is a PyTorch RuntimeError deep inside
    load_state_dict, not a clean "missing/unexpected keys" report)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if expected_num_expression_params is not None:
        checkpoint_dim = checkpoint.get("num_expression_params", 100)
        if checkpoint_dim != expected_num_expression_params:
            raise ValueError(
                f"{path} was saved with num_expression_params={checkpoint_dim}, but this run is "
                f"configured for num_expression_params={expected_num_expression_params} - these are "
                f"incompatible (the expression head's weights are a different shape); use a checkpoint "
                f"trained with matching num_expression_params, or start a fresh run instead of resuming."
            )
    for name, module in modules.items():
        result = module.load_state_dict(checkpoint[name], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"[checkpoint] {name}: architecture mismatch against {path} - "
                f"missing {result.missing_keys}, unexpected {result.unexpected_keys} "
                f"(unmatched params stay at current init)"
            )
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]
