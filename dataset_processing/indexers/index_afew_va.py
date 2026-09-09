from __future__ import annotations

import json
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_DIR = FACE_DATASETS_ROOT / "afew_va"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "afew_va.jsonl"


def find_video_dirs():
    for batch_dir in sorted(RAW_DIR.iterdir()):
        if not batch_dir.is_dir():
            continue
        for video_dir in sorted(batch_dir.iterdir()):
            if video_dir.is_dir():
                yield video_dir


def build_rows():
    for video_dir in find_video_dirs():
        video_id = video_dir.name
        with (video_dir / f"{video_id}.json").open() as f:
            data = json.load(f)

        frame_paths = sorted(video_dir.glob("*.png"))
        valence, arousal, landmarks = [], [], []
        for frame_path in frame_paths:
            frame_data = data["frames"][frame_path.stem]
            valence.append(frame_data["valence"])
            arousal.append(frame_data["arousal"])
            landmarks.append(frame_data["landmarks"])

        yield ManifestRow(
            dataset="afew_va",
            sample_id=video_id,
            subject_id=video_id,
            dimensionality="2d",
            modality="video",
            image_paths=[str(p) for p in frame_paths],
            labels={
                "actor": data.get("actor"),
                "valence": valence,
                "arousal": arousal,
                "landmarks_68": landmarks,
            },
        )


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"afew_va: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()