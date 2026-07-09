from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class CategoryConfig:
    batch_size: int
    max_frames: int
    num_workers: int
    drop_last: bool = True


@dataclasses.dataclass
class DetectorConfig:
    device: str = "cpu"
    threshold: float = 0.8
    model_name: str = "mobilenet0.25"


@dataclasses.dataclass
class DataloaderConfig:
    seed: int
    image_size: int
    crop_scale: float
    crop_cache_root: str
    detector: DetectorConfig
    categories: dict[str, CategoryConfig]


def load_dataloader_config(path: str | Path) -> DataloaderConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    return DataloaderConfig(
        seed=raw["seed"],
        image_size=raw["image_size"],
        crop_scale=raw["crop_scale"],
        crop_cache_root=raw["crop_cache_root"],
        detector=DetectorConfig(**raw["detector"]),
        categories={name: CategoryConfig(**fields) for name, fields in raw["categories"].items()},
    )
