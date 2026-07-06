"""Index FFHQ into the common manifest format.

Raw layout: face_datasets/ffhq/images1024x1024/<5-digit-id>.png, flat, 70000
files. No identity/attribute/landmark metadata exists in this copy (FFHQ has
no identity labels by design -- one image per person scraped from Flickr).
"""

from __future__ import annotations

from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_DIR = FACE_DATASETS_ROOT / "ffhq" / "images1024x1024"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "ffhq.jsonl"


def build_rows():
    for image_path in sorted(RAW_DIR.glob("*.png")):
        sample_id = image_path.stem
        yield ManifestRow(
            dataset="ffhq",
            sample_id=sample_id,
            subject_id=sample_id,
            dimensionality="2d",
            modality="image",
            image_paths=[str(image_path)],
        )


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"ffhq: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
