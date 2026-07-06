"""Index CelebA into the common manifest format.

Raw layout: face_datasets/CelebA/img_celeba/<6-digit-id>.jpg, flat, 202599 files.
No identity/attribute/landmark annotation files exist in this copy of the
dataset (only the README describing them), so there is no subject grouping
available -- each image is treated as its own subject.
"""

from __future__ import annotations

from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_DIR = FACE_DATASETS_ROOT / "CelebA" / "img_celeba"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "celeba.jsonl"


def build_rows():
    for image_path in sorted(RAW_DIR.glob("*.jpg")):
        sample_id = image_path.stem
        yield ManifestRow(
            dataset="celeba",
            sample_id=sample_id,
            subject_id=sample_id,
            dimensionality="2d",
            modality="image",
            image_paths=[str(image_path)],
        )


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"celeba: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()