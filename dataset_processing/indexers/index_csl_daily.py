"""Index CSL-Daily into the common manifest format.

Raw layout: sign_datasets/csl-daily/{train,dev,test}/<name>/<frame>.jpg
(pre-extracted frame sequences, one folder per sentence clip) -- the split is
taken directly from which of the three folders a clip lives in, rather than
cross-referencing split_1.txt. Annotations come from
sign_datasets/csl-daily/csl2020ct_v1.pkl: info is a list of dicts keyed by
name with length, label_gloss/label_char/label_word, signer id, and time
(repeat-performance index).

Each clip is one manifest row, image_paths = ordered frame paths for that clip.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

SIGN_DATASETS_ROOT = FACE_DATASETS_ROOT.parent / "sign_datasets"
RAW_ROOT = SIGN_DATASETS_ROOT / "csl-daily"
PKL_FILE = RAW_ROOT / "csl2020ct_v1.pkl"
SPLITS = ["train", "dev", "test"]
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "csl_daily.jsonl"


def load_info_map() -> dict[str, dict]:
    with PKL_FILE.open("rb") as f:
        data = pickle.load(f)
    return {entry["name"]: entry for entry in data["info"]}


def build_rows():
    info_map = load_info_map()
    missing_info = 0

    for split in SPLITS:
        split_dir = RAW_ROOT / split
        for clip_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            name = clip_dir.name
            entry = info_map.get(name)
            if entry is None:
                missing_info += 1
                continue

            frame_paths = sorted(clip_dir.glob("*.jpg"))
            if not frame_paths:
                continue

            yield ManifestRow(
                dataset="csl_daily",
                sample_id=name,
                subject_id=f"signer{entry['signer']}",
                dimensionality="2d",
                modality="video",
                image_paths=[str(p) for p in frame_paths],
                split=split,
                labels={
                    "gloss": entry["label_gloss"],
                    "char": entry["label_char"],
                    "word": entry["label_word"],
                    "signer": entry["signer"],
                    "time": entry["time"],
                },
            )

    if missing_info:
        print(f"csl_daily: warning -- {missing_info} clip folders had no matching annotation entry, skipped")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"csl_daily: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()