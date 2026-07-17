from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset_processing.dataloading.config import DataloaderConfig
from dataset_processing.dataloading.datasets import build_category_dataset
from dataset_processing.dataloading.identity_batch_sampler import IdentityAwareBatchSampler
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
        category_samplers: dict[str, DistributedSampler | IdentityAwareBatchSampler],
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
    video_mode: Literal["frame_pool", "clip"] = "clip",
) -> CombinedFaceLoader:
    """video_mode: see build_category_dataset's docstring - applies to every video
    category in this loader uniformly (Sec 5.2's single global per-pass switch),
    not chosen independently per category."""
    entries = load_datasets_yaml(datasets_yaml_path)
    by_category = datasets_by_category(entries)

    category_loaders: dict[str, DataLoader] = {}
    category_samplers: dict[str, DistributedSampler | IdentityAwareBatchSampler] = {}
    for category in CATEGORIES:
        category_cfg = cfg.categories[category]
        category_datasets = [
            build_category_dataset(entry, split, cfg, video_mode=video_mode) for entry in by_category[category]
        ]
        dataset = category_datasets[0] if len(category_datasets) == 1 else ConcatDataset(category_datasets)

        distributed_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=(split == "train"), seed=cfg.seed, drop_last=False,
        )

        # IdentityAwareBatchSampler needs one dataset item = one single sample
        # (ImageFaceDataset/FramePoolVideoDataset), not a whole multi-frame clip
        # (VideoFaceDataset) - 3d_video only gets it in frame_pool mode, never
        # clip mode (where "pairing" would mean pairing whole clips, not the
        # single-frame granularity Lvc actually needs).
        # 'spawn', not the Linux default 'fork': workers lazily construct CUDA
        # models (MICA, FAN) on cache misses, and a forked worker can't
        # re-initialize a CUDA context the parent process already touched.
        mp_context = "spawn" if category_cfg.num_workers > 0 else None

        needs_identity_pairing = category == "3d_image" or (category == "3d_video" and video_mode == "frame_pool")
        if needs_identity_pairing:
            # Lvc (implementation-plan.md Sec 6) needs same-identity pairs within a
            # batch - the default DataLoader(sampler=..., batch_size=...) path just
            # chunks the shuffled index stream positionally, with no way to express
            # that constraint, so eligible 3D categories get a custom batch_sampler
            # instead.
            batch_sampler = IdentityAwareBatchSampler(
                dataset, distributed_sampler, category_cfg.batch_size, category_cfg.drop_last,
            )
            loader = DataLoader(
                dataset, batch_sampler=batch_sampler,
                num_workers=category_cfg.num_workers,
                pin_memory=True, persistent_workers=category_cfg.num_workers > 0,
                multiprocessing_context=mp_context,
            )
            category_loaders[category] = loader
            category_samplers[category] = batch_sampler
        else:
            loader = DataLoader(
                dataset, batch_size=category_cfg.batch_size, sampler=distributed_sampler,
                num_workers=category_cfg.num_workers, drop_last=category_cfg.drop_last,
                pin_memory=True, persistent_workers=category_cfg.num_workers > 0,
                multiprocessing_context=mp_context,
            )
            category_loaders[category] = loader
            category_samplers[category] = distributed_sampler

    return CombinedFaceLoader(category_loaders, category_samplers)
