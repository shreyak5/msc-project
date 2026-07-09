from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_YAML = REPO_ROOT / "dataset_processing" / "config" / "datasets.yaml"


@dataclasses.dataclass
class DatasetEntry:
    name: str
    manifest_path: Path
    dimensionality: str
    modality: str
    category: str


def load_datasets_yaml(path: str | Path = DEFAULT_DATASETS_YAML) -> list[DatasetEntry]:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    entries = []
    for name, fields in raw["datasets"].items():
        manifest_path = Path(fields["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        entries.append(DatasetEntry(
            name=name,
            manifest_path=manifest_path,
            dimensionality=fields["dimensionality"],
            modality=fields["modality"],
            category=fields["category"],
        ))
    return entries


def datasets_by_category(entries: list[DatasetEntry]) -> dict[str, list[DatasetEntry]]:
    by_category: dict[str, list[DatasetEntry]] = {}
    for entry in entries:
        by_category.setdefault(entry.category, []).append(entry)
    return by_category