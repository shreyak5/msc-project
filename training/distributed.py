"""Multi-GPU (DistributedDataParallel) setup shared by every training loop
(Stage 1's training/pretrain.py, and Stage 2 later) - factored out here rather
than duplicated per-loop.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_distributed(fallback_device: str) -> tuple[int, int, int, str]:
    """Reads RANK/WORLD_SIZE/LOCAL_RANK from the environment (set automatically
    by torchrun) and initializes the NCCL process group. If those aren't set
    (a plain `python training/pretrain.py`, no torchrun), falls back to
    single-process (0, 1, 0, fallback_device) - so the exact same training
    script works unchanged either way, no separate code path per mode.

    Returns (rank, world_size, local_rank, device)."""
    if not is_distributed():
        return 0, 1, 0, fallback_device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Restrict CUDA visibility to exactly this rank's own GPU, before any CUDA
    # context is created (including inside dist.init_process_group's NCCL
    # backend) - this makes a bare "cuda" (no index) unambiguous everywhere in
    # this rank's whole process tree, INCLUDING DataLoader worker subprocesses
    # forked later, which inherit this env var. Without this, a bare "cuda"
    # always resolves to physical GPU 0 regardless of which rank a worker
    # belongs to - a confirmed real bug (dataloader.yaml's mica_device: cuda,
    # consumed by mica_pool.get_mica inside DataLoader workers) that piled
    # dozens of worker processes' allocations onto GPU 0 alone until it OOM'd,
    # crashing exactly every local_rank-0 process (global ranks 0, 4, 8, 12 on
    # a 4-node/4-GPU job) - the one rank per node where GPU 0 is both its own
    # legitimate training GPU and every misconfigured worker's dumping ground.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)  # only one GPU visible now - always index 0
    return rank, world_size, local_rank, "cuda:0"


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    """Returns the underlying model if module is DDP-wrapped, else module
    itself. Checkpoints should always save the underlying model's state dict
    (unprefixed keys) - DistributedDataParallel prefixes every key with
    "module.", which would make a checkpoint saved from a multi-GPU run
    unloadable into a plain (single-GPU/inference) model without key surgery."""
    return module.module if isinstance(module, DistributedDataParallel) else module


def is_main_process(rank: int) -> bool:
    return rank == 0
