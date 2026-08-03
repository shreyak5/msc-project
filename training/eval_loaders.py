"""Per-dataset dev-split DataLoaders for stage-2's periodic in-training eval
(training/stage2.py's run_periodic_eval_local), covering how2sign/phoenix2014t/
csl_daily by default.

Deliberately NOT dataset_processing.dataloading.combined_loader.build_combined_
loader: that pools every dataset sharing a category (e.g. all 2d_video datasets
- afew_va, mead, csl_daily, phoenix2014t, how2sign) into one ConcatDataset, which
would make it impossible to report a separate landmark/temporal-smoothness score
per dataset. build_eval_loaders instead builds one DataLoader per requested
dataset directly via dataset_processing.dataloading.datasets.build_category_
dataset.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset_processing.dataloading.config import DataloaderConfig
from dataset_processing.dataloading.datasets import build_category_dataset
from dataset_processing.dataloading.registry import load_datasets_yaml


def build_eval_loaders(
    dataloader_cfg: DataloaderConfig,
    datasets_yaml_path: str | Path,
    dataset_names: list[str],
    num_clips_per_dataset: int,
    rank: int,
    world_size: int,
) -> dict[str, DataLoader]:
    """One DataLoader per name in dataset_names, each over a fixed, deterministic
    "first K clips" dev-split subset (K = num_clips_per_dataset, clamped to the
    dataset's actual size), sharded across ALL ranks via the same
    DistributedSampler pattern build_combined_loader already uses for training -
    so num_clips_per_dataset is the TOTAL clip count per dataset per eval round
    (summed across ranks), not a per-rank count. Every rank calling this with the
    same arguments builds the exact same fixed K-clip subset (no shuffling
    happens anywhere in VideoFaceDataset construction - see its own docstring),
    so DistributedSampler's rank-based split is deterministic and stable across
    eval rounds.

    Built once per rank at the top of train(), not re-built per eval round.
    num_workers=0 deliberately: eval batches are small/infrequent, so there's no
    need for combined_loader.py's spawn-multiprocessing-context machinery
    (needed there only because persistent CUDA-touching workers require it).
    """
    entries = [e for e in load_datasets_yaml(datasets_yaml_path) if e.name in dataset_names]

    loaders: dict[str, DataLoader] = {}
    for entry in entries:
        dataset = build_category_dataset(
            entry, split="dev", cfg=dataloader_cfg, video_mode="clip", with_landmarks_fan_full=True,
        )
        num_clips = min(num_clips_per_dataset, len(dataset))
        fixed_subset = Subset(dataset, range(num_clips))

        sampler = DistributedSampler(
            fixed_subset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
        )
        category_cfg = dataloader_cfg.categories[entry.category]
        loaders[entry.name] = DataLoader(
            fixed_subset, batch_size=category_cfg.batch_size, sampler=sampler,
            num_workers=0, pin_memory=True,
        )
    return loaders
