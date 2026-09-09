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
