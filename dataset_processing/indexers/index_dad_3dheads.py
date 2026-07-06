"""Index DAD-3DHeads into the common manifest format.

Raw layout: face_datasets/DAD-3DHeadsDataset/<split>/<split>/{images/, annotations/, <split>.json}
for split in train/val/test. Each master <split>.json entry has an item_id,
bbox, and attributes; the matching image and per-image annotation (5023
FLAME-topology mesh vertices + camera matrices) are found at
images/<item_id>.png and annotations/<item_id>.json. Note: the img_path/
annotation_path fields recorded inside the master json do not match the
actual on-disk layout (missing one level of nesting), so paths are
constructed from item_id instead of trusting those fields.

Per the project plan we are not doing a train/test split yet, so all three
original splits are indexed together into one manifest, with the original
split kept as a label for provenance. No identity/person field exists
anywhere in this dataset, so subject_id is a singleton per row.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_ROOT = FACE_DATASETS_ROOT / "DAD-3DHeadsDataset"
SPLITS = ["train", "val", "test"]
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "dad_3dheads.jsonl"


def build_rows():
    for split in SPLITS:
        split_dir = RAW_ROOT / split / split
        with (split_dir / f"{split}.json").open() as f:
            entries = json.load(f)

        for entry in entries:
            item_id = entry["item_id"]
            image_path = split_dir / "images" / f"{item_id}.png"
            annotation_path = split_dir / "annotations" / f"{item_id}.json"

            yield ManifestRow(
                dataset="dad_3dheads",
                sample_id=item_id,
                subject_id=item_id,
                dimensionality="3d",
                modality="image",
                image_paths=[str(image_path)],
                flame_mesh_paths=[str(annotation_path)],
                labels={
                    "bbox": entry["bbox"],
                    "attributes": entry.get("attributes", {}),
                    "orig_split": split,
                },
            )


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"dad_3dheads: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()