from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


@dataclasses.dataclass
class ManifestRow:
    dataset: str
    sample_id: str
    subject_id: str
    dimensionality: Literal["2d", "3d"]
    modality: Literal["image", "video"]
    image_paths: list[str]
    flame_mesh_paths: list[str] | None = None
    frame_index: int | None = None
    sequence_id: str | None = None
    camera_id: str | None = None
    split: Literal["train", "dev", "test"] = "train"
    labels: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(row.to_json())
            f.write("\n")
            count += 1
    return count


def read_manifest(path: str | Path) -> Iterator[ManifestRow]:
    path = Path(path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield ManifestRow(**json.loads(line))