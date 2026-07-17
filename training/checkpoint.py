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
    orders after a resume."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"step_{step:08d}.pt"
    checkpoint = {"step": step}
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
) -> int:
    """Loads weights (and optimizer state, if given) in place, returns step to
    resume from. Only loads the modules/optimizer actually passed in - e.g.
    Stage 2 seeding from a Stage 1 checkpoint passes just {"svit": svit,
    "heads": heads} and no optimizer, ignoring that checkpoint's lack of
    unet/tt/optimizer keys entirely, since those were never requested."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]
