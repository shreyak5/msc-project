"""Index BUPT-CBFace-12 into the common manifest format.

Raw layout: face_datasets/BUPT-CBFace-12/BUPT-CBFace-12/images/<freebase_mid>/<index>.jpg
(41667 identity folders, ~12 images/identity, 500004 images total), plus a
landmark.tsv with one row per image: NAME (identity/index), bbox (X1,Y1,X2,Y2),
5-point facial landmarks (PTX1,PTY1 ... PTX5,PTY5).
"""

from __future__ import annotations

import csv
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_ROOT = FACE_DATASETS_ROOT / "BUPT-CBFace-12" / "BUPT-CBFace-12"
IMAGES_DIR = RAW_ROOT / "images"
LANDMARK_TSV = RAW_ROOT / "landmark.tsv"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "bupt_cbface12.jsonl"


def load_landmarks() -> dict[str, dict]:
    landmarks = {}
    with LANDMARK_TSV.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            landmarks[row["NAME"]] = row
    return landmarks


def build_rows():
    landmarks = load_landmarks()
    missing = 0
    for identity_dir in sorted(IMAGES_DIR.iterdir()):
        if not identity_dir.is_dir():
            continue
        identity = identity_dir.name
        for image_path in sorted(identity_dir.glob("*.jpg")):
            index = image_path.stem
            row = landmarks.get(f"{identity}/{index}")
            labels = {}
            if row is not None:
                labels = {
                    "bbox": [float(row["X1"]), float(row["Y1"]), float(row["X2"]), float(row["Y2"])],
                    "landmarks_5pt": [
                        [float(row[f"PTX{i}"]), float(row[f"PTY{i}"])] for i in range(1, 6)
                    ],
                }
            else:
                missing += 1
            yield ManifestRow(
                dataset="bupt_cbface12",
                sample_id=f"{identity}_{index}",
                subject_id=identity,
                dimensionality="2d",
                modality="image",
                image_paths=[str(image_path)],
                labels=labels,
            )
    if missing:
        print(f"bupt_cbface12: warning -- {missing} images had no matching landmark.tsv row")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"bupt_cbface12: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()