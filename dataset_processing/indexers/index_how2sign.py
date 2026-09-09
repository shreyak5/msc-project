from __future__ import annotations

import csv
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

SIGN_DATASETS_ROOT = FACE_DATASETS_ROOT.parent / "sign_datasets"
RAW_ROOT = SIGN_DATASETS_ROOT / "how2sign"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "how2sign.jsonl"

SPLIT_DIRS = {
    "train": RAW_ROOT / "train_rgb_front_clips",
    "dev": RAW_ROOT / "val_rgb_front_clips" / "raw_videos",
    "test": RAW_ROOT / "test_rgb_front_clips",
}
SPLIT_CSVS = {
    "train": RAW_ROOT / "how2sign_train.csv",
    "dev": RAW_ROOT / "how2sign_val.csv",
    "test": RAW_ROOT / "how2sign_test.csv",
}


def load_annotation_map(split: str) -> dict[str, dict]:
    with SPLIT_CSVS[split].open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["SENTENCE_NAME"]: row for row in reader}


def build_rows():
    missing_annotation = 0
    for split, split_dir in SPLIT_DIRS.items():
        annotation_map = load_annotation_map(split)

        for clip_path in sorted(split_dir.glob("*.mp4")):
            name = clip_path.stem
            row = annotation_map.get(name)
            if row is None:
                missing_annotation += 1
                continue

            yield ManifestRow(
                dataset="how2sign",
                sample_id=name,
                subject_id=row["VIDEO_ID"],
                dimensionality="2d",
                modality="video",
                image_paths=[str(clip_path)],
                split=split,
                camera_id="front",
                labels={
                    "sentence": row["SENTENCE"],
                    "start": float(row["START"]),
                    "end": float(row["END"]),
                },
            )

    if missing_annotation:
        print(f"how2sign: warning -- {missing_annotation} clips had no matching annotation row, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"how2sign: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()