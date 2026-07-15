"""Checkpoint save/load for the Stage 1 pretraining loop (training/pretrain.py).

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
    epoch: int,
    step: int,
    svit: nn.Module,
    heads: nn.Module,
    optimizer: Optimizer,
) -> Path:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "svit": unwrap_model(svit).state_dict(),
            "heads": unwrap_model(heads).state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    svit: nn.Module,
    heads: nn.Module,
    optimizer: Optimizer,
    device: str,
) -> tuple[int, int]:
    """Loads weights/optimizer state in place, returns (epoch, step) to resume from."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    svit.load_state_dict(checkpoint["svit"])
    heads.load_state_dict(checkpoint["heads"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"], checkpoint["step"]
