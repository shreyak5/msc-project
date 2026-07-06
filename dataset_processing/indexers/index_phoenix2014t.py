"""Index PHOENIX-2014-T into the common manifest format.

Raw layout: sign_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
  features/fullFrame-210x260px/{train,dev,test}/<sequence>/images####.png
  annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv
    (pipe-delimited: name|video|start|end|speaker|orth|translation)

The CSV's own "video" column is a stale glob pattern ("<name>/1/*.png") that
does not match the actual on-disk flat layout (verified -- there is no "1/"
subfolder), so frame paths are built from "name" + globbing the sequence
folder directly, not from that column.

Each sequence is one manifest row, image_paths = ordered frame paths.
"""

from __future__ import annotations

import csv
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

SIGN_DATASETS_ROOT = FACE_DATASETS_ROOT.parent / "sign_datasets"
RAW_ROOT = SIGN_DATASETS_ROOT / "PHOENIX-2014-T-release-v3" / "PHOENIX-2014-T"
FEATURES_ROOT = RAW_ROOT / "features" / "fullFrame-210x260px"
ANNOTATIONS_ROOT = RAW_ROOT / "annotations" / "manual"
SPLITS = ["train", "dev", "test"]
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "phoenix2014t.jsonl"


def build_rows():
    missing_dir = 0
    for split in SPLITS:
        csv_path = ANNOTATIONS_ROOT / f"PHOENIX-2014-T.{split}.corpus.csv"
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                name = row["name"]
                seq_dir = FEATURES_ROOT / split / name
                frame_paths = sorted(seq_dir.glob("images*.png"))
                if not frame_paths:
                    missing_dir += 1
                    continue

                yield ManifestRow(
                    dataset="phoenix2014t",
                    sample_id=name,
                    subject_id=row["speaker"],
                    dimensionality="2d",
                    modality="video",
                    image_paths=[str(p) for p in frame_paths],
                    split=split,
                    labels={
                        "orth": row["orth"],
                        "translation": row["translation"],
                        "speaker": row["speaker"],
                    },
                )

    if missing_dir:
        print(f"phoenix2014t: warning -- {missing_dir} annotated sequences had no frames on disk, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"phoenix2014t: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()