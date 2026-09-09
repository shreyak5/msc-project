from __future__ import annotations

import re
from pathlib import Path

from dataset_processing.manifest_schema import ManifestRow, write_manifest
from dataset_processing.paths import FACE_DATASETS_ROOT

RAW_DIR = FACE_DATASETS_ROOT / "MEAD"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "mead.jsonl"

SUBJECT_RE = re.compile(r"^[MW]\d{3}$")
GENDER_MAP = {"M": "male", "W": "female"}
CAMERA_ANGLES = ["down", "front", "left_30", "left_60", "right_30", "right_60", "top"]


def find_subject_dirs():
    for subject_dir in sorted(RAW_DIR.iterdir()):
        if subject_dir.is_dir() and SUBJECT_RE.match(subject_dir.name):
            yield subject_dir


def build_rows():
    subjects_with_no_angles = []

    for subject_dir in find_subject_dirs():
        subject = subject_dir.name
        video_dir = subject_dir / "video"
        available_angles = sorted(a for a in CAMERA_ANGLES if (video_dir / a).is_dir())
        if not available_angles:
            subjects_with_no_angles.append(subject)
            continue

        combo_angles: dict[tuple[str, str], set[str]] = {}
        for angle in available_angles:
            angle_dir = video_dir / angle
            for emotion_dir in sorted(angle_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                for level_dir in sorted(emotion_dir.iterdir()):
                    if not level_dir.is_dir():
                        continue
                    combo_angles.setdefault((emotion_dir.name, level_dir.name), set()).add(angle)

        for i, (emotion, level) in enumerate(sorted(combo_angles)):
            preferred = available_angles[i % len(available_angles)]
            candidates = combo_angles[(emotion, level)]
            chosen = preferred if preferred in candidates else sorted(candidates)[0]

            clip_dir = video_dir / chosen / emotion / level
            for clip_path in sorted(clip_dir.glob("*.mp4")):
                seq = clip_path.stem
                yield ManifestRow(
                    dataset="mead",
                    sample_id=f"{subject}_{emotion}_{level}_{seq}",
                    subject_id=subject,
                    dimensionality="2d",
                    modality="video",
                    image_paths=[str(clip_path)],
                    sequence_id=f"{emotion}_{level}_{seq}",
                    camera_id=chosen,
                    labels={
                        "emotion": emotion,
                        "intensity": level,
                        "gender": GENDER_MAP[subject[0]],
                    },
                )

    if len(subjects_with_no_angles):
        print(f"mead: warning -- {len(subjects_with_no_angles)} subject(s) had no camera-angle folders at all, skipped: {subjects_with_no_angles}")


def main():
    count = write_manifest(build_rows(), MANIFEST_PATH)
    print(f"mead: wrote {count} rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()