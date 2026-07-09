from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset_processing.dataloading.config import DataloaderConfig
from dataset_processing.dataloading.datasets import build_category_dataset
from dataset_processing.dataloading.registry import (
    DEFAULT_DATASETS_YAML,
    datasets_by_category,
    load_datasets_yaml,
)

CATEGORIES = ("2d_image", "2d_video", "3d_image", "3d_video")


def zip_max_size_cycle(loaders: dict[str, DataLoader]) -> Iterator[dict[str, Any]]:
    length = max(len(loader) for loader in loaders.values())
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    for _ in range(length):
        batch = {}
        for name, loader in loaders.items():
            try:
                batch[name] = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(loader)
                batch[name] = next(iterators[name])
        yield batch


class CombinedFaceLoader:
    def __init__(
        self,
        category_loaders: dict[str, DataLoader],
        category_samplers: dict[str, DistributedSampler],
    ):
        self.category_loaders = category_loaders
        self.category_samplers = category_samplers
        self._length = max(len(loader) for loader in category_loaders.values())

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        for sampler in self.category_samplers.values():
            sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return zip_max_size_cycle(self.category_loaders)


def build_combined_loader(
    cfg: DataloaderConfig,
    split: str,
    rank: int,
    world_size: int,
    datasets_yaml_path: str | Path = DEFAULT_DATASETS_YAML,
) -> CombinedFaceLoader:
    entries = load_datasets_yaml(datasets_yaml_path)
    by_category = datasets_by_category(entries)

    category_loaders: dict[str, DataLoader] = {}
    category_samplers: dict[str, DistributedSampler] = {}
    for category in CATEGORIES:
        category_cfg = cfg.categories[category]
        category_datasets = [build_category_dataset(entry, split, cfg) for entry in by_category[category]]
        dataset = category_datasets[0] if len(category_datasets) == 1 else ConcatDataset(category_datasets)

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=(split == "train"), seed=cfg.seed, drop_last=False,
        )
        loader = DataLoader(
            dataset, batch_size=category_cfg.batch_size, sampler=sampler,
            num_workers=category_cfg.num_workers, drop_last=category_cfg.drop_last,
            pin_memory=True, persistent_workers=category_cfg.num_workers > 0,
        )
        category_loaders[category] = loader
        category_samplers[category] = sampler

    return CombinedFaceLoader(category_loaders, category_samplers)
